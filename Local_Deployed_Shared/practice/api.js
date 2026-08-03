/* ================================================================
   PRACTICE API — backend / supabase / local routing
   ================================================================ */

function emitPracticeStateChanged() {
  window.dispatchEvent(new CustomEvent("delta:practice-state-changed"));
}

const PracticeAPI = {
  currentQuestion: practiceQuestionPool[0],

  outputsMatch(actualOutput, expectedOutput) {
    return (actualOutput || "").trim() === (expectedOutput || "").trim();
  },

  async recordLocalEval(questionId, correct) {
    // Guest / supabase: the Pyodide engine IS the store, so a self-rated Colab
    // drill has to go in the same way a graded submission does. It used to
    // return here, which meant nothing was recorded — and `next_question`
    // picks from the state, so rating a torch drill served the SAME question
    // back, forever. Only reachable from the torch self-rate path in these
    // modes (submitAnswer calls this for backend+Pyodide fallback only), so
    // there is no double-record.
    if (practiceMode !== "backend") {
      const q = this.currentQuestion;
      const pyodide = await initPyodide();
      if (pyodide && practiceEngineLoaded && adaptiveStateJson && q) {
        const api = pyodide.globals.get("engine_api");
        adaptiveStateJson = api.submit_answer(
          adaptiveStateJson,
          q.question_id,
          q.subtopic,
          q.difficulty || 50,
          !!correct,
        );
        // ...and count it. `submit_answer` only parks the attempt in
        // `pending_attempt`; `send_feedback` is what increments `n`, steps the
        // staircase and moves recent accuracy. Every OTHER path pairs the two
        // — grade, then "how much did you learn?" — but this one is the whole
        // submit: the Colab edition has no felt-difficulty step, so without
        // this the attempt sat pending until the next problem overwrote it and
        // the learner's practice never appeared anywhere.
        //
        // "unrated" rather than a real level: the learner was not asked, so
        // there is nothing to report. The engine treats it as no alpha.
        adaptiveStateJson = api.send_feedback(adaptiveStateJson, "unrated");
        await saveAdaptiveState();
      }
      if (!practiceProgress.completedQuestionIds.includes(questionId)) {
        practiceProgress.completedQuestionIds.push(questionId);
        savePracticeProgress(practiceProgress);
      }
      emitPracticeStateChanged();
      return;
    }
    const res = await apiFetch("/api/practice/submit-local-eval", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question_id: questionId, correct }),
    });
    if (res.status === 401) {
      handleExpiredToken();
      return;
    }
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(detail || "Failed to record local evaluation.");
    }
    await res.json();
  },

  async getNextQuestion() {
    await loadQuestionsSourceIndex();
    // Single-KC ladder (concept-graph practice) serves its own faded →
    // independent sequence first. It returns null once spent, and the normal
    // adaptive queue resumes — still pinned to that subtopic.
    if (window.KcPractice && window.KcPractice.isActive()) {
      const ladderQ = await window.KcPractice.nextQuestion();
      if (ladderQ) {
        this.currentQuestion = ladderQ;
        practiceProgress.currentQuestionId = ladderQ.question_id;
        practiceProgress.currentQuestion = ladderQ;
        savePracticeProgress(practiceProgress);
        return ladderQ;
      }
    }
    if (practiceMode === "backend") {
      // Admin on localhost — use backend API
      // focus_subtopic pins the backend queue to one subtopic for single-KC
      // practice. Older backends ignore the unknown query param and serve the
      // normal queue, so this degrades instead of breaking.
      const focus = window.__kcFocusSubtopic;
      const res = await apiFetch(
        "/api/practice/next-question" +
          (focus ? "?focus_subtopic=" + encodeURIComponent(focus) : ""),
      );
      if (res.status === 401) {
        handleExpiredToken();
        // fall through to local mode below
      } else if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Failed to load next question.");
      } else {
        const data = await res.json();
        this.currentQuestion = {
          ...data,
          target_difficulty: Number.isFinite(data.target_difficulty)
            ? data.target_difficulty
            : data.difficulty,
        };
        practiceProgress.currentQuestion = this.currentQuestion;
        practiceProgress.currentQuestionId = data.question_id;
        savePracticeProgress(practiceProgress);
        return this.currentQuestion;
      }
    }

    // supabase or local mode — use Pyodide engine
    const pyodide = await initPyodide();
    const bank = await loadQuestionsBank();
    const eligibleBank = getPracticeEligibleQuestions();

    if (pyodide && practiceEngineLoaded && bank && adaptiveStateJson) {
      if (!eligibleBank?.length) {
        throw new Error("No practice questions available for the current selection. Try reloading, or pick a concept from the Knowledge Graph.");
      }
      const api = pyodide.globals.get("engine_api");
      const resultJson = api.next_question(adaptiveStateJson, JSON.stringify(eligibleBank));
      const result = JSON.parse(resultJson);
      adaptiveStateJson = result.state;
      await saveAdaptiveState();

      if (result.question) {
        this.currentQuestion = buildPracticeQuestionFromBank(result.question);
      }
    } else {
      // Fallback to hardcoded pool — a static round-robin with NO adaptivity.
      // Never serve this silently: the practice UI still renders target-
      // difficulty/calibration widgets, and without a notice they read as a
      // live adaptive session that mysteriously never updates (tester hit
      // exactly this after a silent token-expiry demotion).
      if (typeof showPracticeModeNotice === "function") {
        showPracticeModeNotice(
          "Demo questions — the adaptive engine isn't available right now, " +
          "so difficulty won't adapt. Sign in (or reload) for the real queue.",
        );
      }
      const completed = new Set(practiceProgress.completedQuestionIds);
      let attempts = 0;
      let nextIndex = practiceQuestionIndex;
      do {
        nextIndex = (nextIndex + 1) % practiceQuestionPool.length;
        attempts++;
        if (attempts >= practiceQuestionPool.length) break;
      } while (completed.has(practiceQuestionPool[nextIndex].question_id));
      practiceQuestionIndex = nextIndex;
      this.currentQuestion = practiceQuestionPool[practiceQuestionIndex];
    }

    practiceProgress.currentQuestionId = this.currentQuestion.question_id;
    practiceProgress.currentQuestion = this.currentQuestion;
    savePracticeProgress(practiceProgress);
    return this.currentQuestion;
  },

  // --- Placement diagnostic (backend mode only) -------------------------
  // "I don't know yet" and self-rated probe results go here; answered probes
  // are recorded server-side by /submit while the diagnostic is active.
  async diagnosticAnswer(questionId, result) {
    if (practiceMode !== "backend") return null;
    const res = await apiFetch("/api/practice/diagnostic/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question_id: questionId, result }),
    });
    if (res.status === 401) {
      handleExpiredToken();
      return null;
    }
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(detail || "Failed to record placement answer.");
    }
    return await res.json();
  },

  async diagnosticStart() {
    if (practiceMode !== "backend") return null;
    const res = await apiFetch("/api/practice/diagnostic/start", {
      method: "POST",
    });
    if (res.status === 401) {
      handleExpiredToken();
      return null;
    }
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(detail || "Failed to start the placement diagnostic.");
    }
    return await res.json();
  },

  async diagnosticStatus() {
    if (practiceMode !== "backend") return null;
    try {
      const res = await apiFetch("/api/practice/diagnostic/status");
      if (!res.ok) return null;
      return await res.json();
    } catch (_) {
      return null;
    }
  },

  async submitAnswer(questionId, userCode) {
    const requiresLocalPyodide = questionNeedsEinops(this.currentQuestion);

    if (practiceMode === "backend" && !requiresLocalPyodide) {
      const res = await apiFetch("/api/practice/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question_id: questionId, user_code: userCode }),
      });
      if (res.status === 401) {
        handleExpiredToken();
        // fall through to local mode below
      } else if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Failed to submit answer.");
      } else {
        return await res.json();
      }
    }

    // supabase/local or backend+einops fallback — run code with Pyodide and AI judge
    const pyodide = await initPyodide();
    let actualOutput = "";
    if (pyodide) {
      const preamble = await buildPyodidePreamble(this.currentQuestion);
      pyodide.runPython(preamble);
      try {
        pyodide.runPython(userCode);
        actualOutput = pyodide.runPython("sys.stdout.getvalue()").trim();
      } catch (e) {
        actualOutput = "[ERROR]";
      } finally {
        pyodide.runPython("sys.stdout = sys.__stdout__\nsys.stderr = sys.__stderr__");
      }
    }
    const expected = (this.currentQuestion.expected_output || "").trim();
    const failed_tests = [];
    const solCode = this.currentQuestion.solution_code || "";
    const questionText = this.currentQuestion.question_text || "";
    let correct = false;

    // Function test cases take precedence over the stdout string compare —
    // they are the audited contract. (Mirrors backend grade_submission; the
    // old order hijacked function-mode questions into comparing against
    // stored CSV-era expected strings, some captured from unseeded runs.)
    const hasFunctionTests =
      this.currentQuestion.submission_mode === "function" &&
      this.currentQuestion.test_cases?.length;
    if (
      this.currentQuestion.task_type === "stdout_prediction" &&
      expected &&
      !this.currentQuestion.supports_visual_output &&
      !hasFunctionTests
    ) {
      correct = this.outputsMatch(actualOutput, expected);

      if (practiceEngineLoaded && adaptiveStateJson) {
        const api = pyodide.globals.get("engine_api");
        adaptiveStateJson = api.submit_answer(
          adaptiveStateJson,
          this.currentQuestion.question_id,
          this.currentQuestion.subtopic,
          this.currentQuestion.difficulty || 50,
          correct
        );
        await saveAdaptiveState();
      }

      if (practiceMode === "backend" && requiresLocalPyodide) {
        await this.recordLocalEval(questionId, correct);
      }

      return { correct, actual_output: actualOutput, expected_output: expected, failed_tests };
    }

    if (this.currentQuestion.submission_mode === "function" && this.currentQuestion.test_cases?.length) {
      const preamble = await buildPyodidePreamble(this.currentQuestion);
      const testsJsonLiteral = JSON.stringify(JSON.stringify(this.currentQuestion.test_cases));
      pyodide.runPython(preamble);
      try {
        pyodide.runPython(userCode);
        const resultJson = pyodide.runPython(`
import json
import numpy as np

def _delta_to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_delta_to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_delta_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _delta_to_jsonable(v) for k, v in value.items()}
    return value

def _delta_equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_delta_equal(x, y) for x, y in zip(a, b))
    return bool(a == b)

_delta_results = []
for _delta_case in json.loads(${testsJsonLiteral}):
    try:
        if _delta_case.get("setup_code"):
            np.random.seed(0)
            exec(_delta_case["setup_code"], globals())
        _delta_actual = eval(_delta_case["call"], globals())
        _delta_expected_setup = _delta_case.get("expected_setup_code") or _delta_case.get("setup_code")
        if _delta_expected_setup:
            np.random.seed(0)
            exec(_delta_expected_setup, globals())
        _delta_expected = eval(_delta_case["expected_expr"], globals())
        _delta_results.append({
            "passed": bool(_delta_equal(_delta_actual, _delta_expected)),
            "actual": repr(_delta_to_jsonable(_delta_actual)),
            "expected": repr(_delta_to_jsonable(_delta_expected)),
            "error": "",
        })
    except Exception as _delta_exc:
        _delta_results.append({
            "passed": False,
            "actual": "",
            "expected": "",
            "error": f"{type(_delta_exc).__name__}: {_delta_exc}",
        })
json.dumps(_delta_results)
`);
        const parsed = JSON.parse(resultJson);
        failed_tests.push(...parsed.filter((test) => !test.passed));
        correct = failed_tests.length === 0;
        actualOutput = pyodide.runPython("sys.stdout.getvalue()").trim();
      } catch (e) {
        actualOutput = pyodide.runPython("sys.stderr.getvalue()").trim() || e.message;
        correct = false;
      } finally {
        pyodide.runPython("sys.stdout = sys.__stdout__\nsys.stderr = sys.__stderr__");
      }
      if (practiceMode === "backend" && requiresLocalPyodide) {
        await this.recordLocalEval(questionId, correct);
      }
      if (practiceEngineLoaded && adaptiveStateJson) {
        const api = pyodide.globals.get("engine_api");
        adaptiveStateJson = api.submit_answer(
          adaptiveStateJson,
          this.currentQuestion.question_id,
          this.currentQuestion.subtopic,
          this.currentQuestion.difficulty || 50,
          correct
        );
        await saveAdaptiveState();
      }
      return { correct, actual_output: actualOutput, expected_output: expected, failed_tests };
    }
    try {
      const verdict = await fetchAIJudge(questionText, solCode, userCode, actualOutput, expected);
      correct = verdict === "1";
    } catch (err) {
      throw new Error("AI judge unavailable. Please sign in or use backend mode.");
    }

    // Record attempt in adaptive engine
    if (practiceEngineLoaded && adaptiveStateJson) {
      const api = pyodide.globals.get("engine_api");
      adaptiveStateJson = api.submit_answer(
        adaptiveStateJson,
        this.currentQuestion.question_id,
        this.currentQuestion.subtopic,
        this.currentQuestion.difficulty || 50,
        correct
      );
      await saveAdaptiveState();
    }

    if (practiceMode === "backend" && requiresLocalPyodide) {
      await this.recordLocalEval(questionId, correct);
    }

    return { correct, actual_output: actualOutput, expected_output: expected, failed_tests };
  },

  /**
   * Count a graded-but-unrated attempt now, instead of when the next one starts.
   *
   * `record_attempt` flushes the previous pending attempt, which is enough that
   * nothing is ever lost — but the LAST attempt of a session waits for the next
   * session to land. Ending a session tells the learner "Recorded answers are
   * kept", so the exit paths call this and make that true.
   *
   * Backend mode owns its own pending attempt server-side and this does not
   * reach it; the offline engine is the one that needed saying out loud.
   */
  async flushPendingAttempt() {
    if (practiceMode === "backend") return;
    const pyodide = await initPyodide();
    if (!pyodide || !practiceEngineLoaded || !adaptiveStateJson) return;
    const api = pyodide.globals.get("engine_api");
    const next = api.flush_pending(adaptiveStateJson);
    if (next === adaptiveStateJson) return;   // nothing was pending
    adaptiveStateJson = next;
    await saveAdaptiveState();
    emitPracticeStateChanged();
  },

  async sendFeedback(questionId, feedback) {
    if (practiceMode === "backend") {
      const res = await apiFetch("/api/practice/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question_id: questionId, feedback }),
      });
      if (res.status === 401) {
        handleExpiredToken();
        return; // feedback is non-critical, just skip it
      } else if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Failed to send feedback.");
      }
      const response = await res.json();
      // /feedback is the only backend mutation that changes subtopic
      // baselines (submit/override only touch pending_attempt). Re-pull
      // /state so the concept-graph atom-readiness bridge updates live.
      await loadBackendAdaptiveState();
      emitPracticeStateChanged();
      return response;
    }

    // supabase/local — apply feedback in Pyodide engine
    const pyodide = await initPyodide();
    if (pyodide && practiceEngineLoaded && adaptiveStateJson) {
      const api = pyodide.globals.get("engine_api");
      adaptiveStateJson = api.send_feedback(adaptiveStateJson, feedback);
      await saveAdaptiveState();
    }
    emitPracticeStateChanged();
    return { success: true };
  },

  async overrideCorrect(questionId) {
    if (practiceMode === "backend") {
      const res = await apiFetch("/api/practice/override", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question_id: questionId, correct: true }),
      });
      if (res.status === 401) {
        handleExpiredToken();
        return;
      } else if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Failed to override attempt.");
      }
      const response = await res.json();
      emitPracticeStateChanged();
      return response;
    }

    const pyodide = await initPyodide();
    if (pyodide && practiceEngineLoaded && adaptiveStateJson) {
      const api = pyodide.globals.get("engine_api");
      adaptiveStateJson = api.override_attempt(adaptiveStateJson, questionId, true);
      await saveAdaptiveState();
    }
    emitPracticeStateChanged();
    return;
  },

  // Per-problem content-quality flag (broken / unclear / wrong_image / good).
  // Best-effort and non-blocking: failures never interrupt practice. In
  // backend mode posts to the sibling log endpoint; otherwise (and on any
  // backend error) falls back to a localStorage queue so nothing is lost.
  async reportProblem(questionId, tag, note, correct) {
    const entry = {
      question_id: questionId,
      tag,
      note: note || "",
      correct: typeof correct === "boolean" ? correct : null,
    };
    if (practiceMode === "backend") {
      try {
        const res = await apiFetch("/api/practice/problem-feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(entry),
        });
        if (res.status === 401) {
          handleExpiredToken();
        } else if (res.ok) {
          return { success: true };
        }
      } catch (_) {
        /* fall through to local queue */
      }
    }
    // Local fallback queue — survives offline / non-backend modes.
    try {
      const key = "problem_feedback_queue";
      const queue = JSON.parse(localStorage.getItem(key) || "[]");
      queue.push({ ...entry, timestamp: new Date().toISOString() });
      localStorage.setItem(key, JSON.stringify(queue));
    } catch (_) {
      /* ignore storage errors */
    }
    return { success: true, queuedLocally: true };
  },
};
