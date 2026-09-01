/* ================================================================
   PRACTICE API — backend / supabase / local routing
   ================================================================ */

function emitPracticeStateChanged() {
  window.dispatchEvent(new CustomEvent("delta:practice-state-changed"));
}

/* Did a worked-example popup (practice/example-gate.js) actually open in
   front of the question being submitted? The server SCHEDULED it
   (question.ladder_example) but only the client knows it was drawn — the gate
   declines on the Colab edition, in the diagnostic, or with no KP page — and
   an example nobody saw must not be stored as assistance. Sent on both
   submit routes as `example_shown`. */
function _exampleShown(questionId) {
  const q = PracticeAPI.currentQuestion;
  return !!(q && Number(q.question_id) === Number(questionId) && q.ladder_example_shown);
}

const PracticeAPI = {
  currentQuestion: practiceQuestionPool[0],

  outputsMatch(actualOutput, expectedOutput) {
    return (actualOutput || "").trim() === (expectedOutput || "").trim();
  },

  /**
   * Record a locally-evaluated attempt, and report back what it moved.
   *
   * The return value is the point: the Colab rail draws the difficulty step the
   * learner just earned, and it can only draw a step it was TOLD about. Reading
   * the globals again after the await would be reading them after whatever else
   * ran during it. Same shape in both modes so the caller has one thing to
   * handle; every number is `null` when there is no honest value for it.
   *
   * `finalize: false` is for the caller that is not finished with the attempt —
   * the einops fallback in `submitAnswer` and the Colab verdict, both of which
   * ask how hard it felt afterwards, so the attempt has to stay pending for
   * that rating to land on. `pending` in the reply says it did: during a
   * placement diagnostic no attempt is created at all, and asking for a rating
   * there would post a /feedback with nothing to apply it to.
   *
   * The default is `true` for a caller that grades and then asks nothing — it
   * closes the attempt out as unrated rather than leaving it to be silently
   * overwritten by the next submit.
   */
  async recordLocalEval(questionId, correct, options = {}) {
    const finalize = options.finalize !== false;
    // Guest / supabase: the Pyodide engine IS the store, so a self-rated Colab
    // drill has to go in the same way a graded submission does. It used to
    // return here, which meant nothing was recorded — and `next_question`
    // picks from the state, so rating a torch drill served the SAME question
    // back, forever. Only reachable from the torch self-rate path in these
    // modes (submitAnswer calls this for backend+Pyodide fallback only, and
    // always with finalize:false), so there is no double-record.
    if (practiceMode !== "backend") {
      const q = this.currentQuestion;
      // Before the engine runs, because both of these are derived from the
      // state `submit_answer`/`send_feedback` are about to rewrite in place.
      const targetBefore = q ? getTargetDifficultyFromAdaptiveState(q.subtopic) : null;
      const pBefore = q ? getEwmaFromAdaptiveState(q.subtopic) : null;
      const pyodide = await initPyodide();
      // Whether the attempt actually went in. The demo-pool fallback runs with
      // no engine at all, and reporting `finalized: true` from there would have
      // the rail explaining a step nothing took.
      const engineRan = !!(pyodide && practiceEngineLoaded && adaptiveStateJson && q);
      if (engineRan) {
        const api = pyodide.globals.get("engine_api");
        adaptiveStateJson = api.submit_answer(
          adaptiveStateJson,
          q.question_id,
          q.subtopic,
          q.difficulty || 50,
          !!correct,
        );
        // ...and count it, but only when nothing is coming back for it.
        // `submit_answer` only parks the attempt in `pending_attempt`;
        // `send_feedback` is what increments `n`, steps the staircase and moves
        // recent accuracy. A caller that still asks how hard it felt has to
        // leave the attempt pending for that rating to land on — finalizing it
        // here would consume it and the real level would have nowhere to go.
        //
        // When we DO close it out here it goes in as "unrated" rather than one
        // of the three real levels: the learner was not asked, so there is
        // nothing to report. The engine treats it as no alpha and gives the
        // staircase its plain step.
        if (finalize) {
          adaptiveStateJson = api.send_feedback(adaptiveStateJson, "unrated");
        }
        await saveAdaptiveState();
      }
      if (!practiceProgress.completedQuestionIds.includes(questionId)) {
        practiceProgress.completedQuestionIds.push(questionId);
        savePracticeProgress(practiceProgress);
      }
      emitPracticeStateChanged();
      const scored = engineRan && finalize;
      return {
        finalized: scored,
        pending: engineRan && !finalize,
        targetBefore,
        targetAfter: scored ? getTargetDifficultyFromAdaptiveState(q.subtopic) : null,
        pBefore,
        pAfter: scored ? getEwmaFromAdaptiveState(q.subtopic) : null,
      };
    }
    const res = await apiFetch("/api/practice/submit-local-eval", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question_id: questionId, correct, finalize, example_shown: _exampleShown(questionId) }),
    });
    if (res.status === 401) {
      handleExpiredToken();
      return null;
    }
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(detail || "Failed to record local evaluation.");
    }
    const data = await res.json();
    // Every field is optional on the wire. A backend that predates the reporting
    // fields answers `{success:true}`, and even a current one returns nulls when
    // the attempt did not finalize — a placement probe places the learner rather
    // than stepping the staircase, so there is no before/after to send. Nulls
    // reach the caller as nulls; inventing a number here would put a movement on
    // screen that nothing actually did.
    //
    // `finalized` therefore has THREE states, not two. `false` is a backend
    // that ran and finalized nothing — a placement probe — and the rail says so
    // in as many words. `null` is a backend that was never asked the question,
    // which is what a deploy predating these fields answers. Collapsing the
    // second into the first is a real, dated failure: during a rolling deploy
    // the old backend answers success-only for an ordinary Colab submission,
    // and every answer would read "still placing you" until the new backend
    // finished shipping. An unknown falls through to the conservative wording.
    const num = (value) => (Number.isFinite(value) ? value : null);
    const finalized = data && "finalized" in data ? data.finalized === true : null;
    // Same three states, same reason: a backend predating the field answers
    // without it, and reading that absence as "nothing is pending" would drop
    // the felt-difficulty step during a rolling deploy. Unknown falls through
    // to `null`, and the caller decides — asking for a rating that 400s is a
    // worse failure than not asking, so the Colab path treats null as no.
    const pending = data && "pending" in data ? data.pending === true : null;
    return {
      finalized,
      pending,
      targetBefore: num(data?.target_difficulty_before),
      targetAfter: num(data?.target_difficulty_after),
      pBefore: num(data?.p_before),
      pAfter: num(data?.p_after),
      ladderStage: data?.ladder_stage ?? null,
      ladderEstimate: data?.ladder_estimate ?? null,
    };
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
      } else if (res.status === 409) {
        /* The concept's current rung holds nothing this learner has not
           already answered. The server stopped rather than re-serving a solved
           problem (backend/app/prioritization.py) and sent the sentence to
           show; `detail.message` is written for the learner, so surface it
           verbatim instead of the JSON around it. Anything unexpected in the
           body falls back to the generic line — a parse failure must not turn
           a handled state into "[object Object]". */
        let message = "";
        try {
          const body = await res.json();
          message = (body && body.detail && body.detail.message) || "";
          if (message) {
            const err = new Error(message);
            err.contentExhausted = body.detail;
            throw err;
          }
        } catch (parseErr) {
          if (parseErr && parseErr.contentExhausted) throw parseErr;
        }
        throw new Error(
          "You have finished every problem available for this concept. Ask Claude to write more drills for it, or pick a different concept from the Knowledge Graph.",
        );
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
      // The on-page notice was deleted on 2026-08-23 (see practice/mode.js).
      // This demotion is therefore console-only now, which is exactly the
      // silence that comment above warns about — read mode.js before deciding
      // that is fine.
      console.warn(
        "[practice] demo pool — the adaptive engine isn't available, so difficulty won't adapt.",
      );
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

  // --- Placement test (backend mode only) --------------------------------
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
      throw new Error(detail || "Failed to start the placement test.");
    }
    return await res.json();
  },

  /* Three different answers, and they used to be one.

     `null` meant "no placement for you" AND "the server said no" AND "the
     server never answered", so the Placement page rendered its signed-out
     copy — "Sign in to take the placement test", every button hidden — at a
     signed-in learner whose backend was restarting. Nothing to click, and an
     instruction that did not apply to them.

     Now `null` keeps its one honest meaning (this build is not talking to a
     backend at all) and a failure comes back marked, with the HTTP status
     when there was one and 0 when the request never landed. Callers must
     treat `unavailable` as "no status", not as a status. */
  async diagnosticStatus() {
    if (practiceMode !== "backend") return null;
    try {
      const res = await apiFetch("/api/practice/diagnostic/status");
      if (res.ok) return await res.json();
      return { unavailable: true, httpStatus: res.status };
    } catch (_) {
      return { unavailable: true, httpStatus: 0 };
    }
  },

  async submitAnswer(questionId, userCode) {
    /* A torch submit from a session stranded in local mode (guest provision
       failed once at boot and DDGuest.ensure() memoized it) used to throw the
       TORCH_UNAVAILABLE refusal below with no way out. Retry provisioning
       HERE, at the top, not by recursing at the dead-end: submitAnswer is
       wrapped by the XP layer, so a recursive call would award twice for one
       submit. If provisioning lands, the ordinary backend branch below grades
       this very call. A real signed-in user demoted by a 401 is not
       re-provisioned (practiceRealUserDemoted, practice/mode.js). */
    if (
      practiceMode !== "backend" &&
      !practiceRealUserDemoted &&
      needsTorchRuntime(this.currentQuestion, userCode)
    ) {
      const provisioned = await window.DDGuest?.retryProvision?.();
      if (provisioned) upgradePracticeModeToBackend();
    }

    /* Einops/visual questions grade on the LOCAL Pyodide instance, because the
       preamble there is what defines `display_array_as_img` and friends.

       But Pyodide cannot import torch, and the bank's einops questions are
       torch questions now — so this routing sent the one thing Pyodide cannot
       run to Pyodide, and every Submit came back "This code uses PyTorch,
       which can't run in the browser sandbox" with no verdict. Signed in, on
       the placement, with the countdown re-arming after each failure, that was
       a question you could not answer and could not get past.

       The backend grades BOTH: `requirements.txt` pins einops for exactly this
       ("backend code grader: einops/visual questions import it in setup_code
       and user code") and torch is preloaded into the fork runner. So local
       Pyodide keeps only the einops questions it can actually execute — the
       ones with no torch anywhere in the question, the test setup, or what the
       learner wrote. */
    const requiresLocalPyodide =
      questionNeedsEinops(this.currentQuestion) &&
      !needsTorchRuntime(this.currentQuestion, userCode);

    if (practiceMode === "backend" && !requiresLocalPyodide) {
      const res = await apiFetch("/api/practice/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question_id: questionId, user_code: userCode, example_shown: _exampleShown(questionId) }),
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

    /* Everything below grades on Pyodide, which cannot import torch. Refuse
       before touching it, the same way runSnippet does for the Run button.

       This THROWS rather than returning `{correct: false}` on purpose: the
       learner's answer was never executed, so there is no verdict to record.
       Returning a result here would mark a correct answer wrong and drag the
       mastery estimate down for a runtime failure. `blocked` tells events.js
       to state the reason instead of prefixing it with "Submit failed:". */
    if (needsTorchRuntime(this.currentQuestion, userCode)) {
      const blocked = new Error(TORCH_UNAVAILABLE);
      blocked.blocked = true;
      throw blocked;
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

      // `finalize: false` on every einops-fallback call below. This is a normal
      // graded submit that merely could not run server-side, so the felt-
      // difficulty step still follows and the attempt has to stay pending for
      // the rating to attach to. Finalizing here closes it out early and
      // /feedback then has nothing to apply — which surfaces as "Feedback
      // failed" on exactly the einops questions.
      if (practiceMode === "backend" && requiresLocalPyodide) {
        await this.recordLocalEval(questionId, correct, { finalize: false });
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
        await this.recordLocalEval(questionId, correct, { finalize: false });
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
      await this.recordLocalEval(questionId, correct, { finalize: false });
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
      client_id: DDFeedbackQueue.clientId(),
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
          // improvement_queued means the backend also queued this question for
          // repair (allowlisted accounts only). The repair itself runs later,
          // on Seth's machine, through the local Claude Code runner — so this
          // is "someone will look", not "it is being fixed right now".
          const body = await res.json().catch(() => ({}));
          DDFeedbackQueue.flush("problem_feedback_queue", "/api/practice/problem-feedback")
            .catch(() => {});
          return { success: true, improvementQueued: !!body.improvement_queued };
        }
      } catch (_) {
        /* fall through to local queue */
      }
    }
    // Local fallback queue — survives offline / non-backend modes.
    const stored = DDFeedbackQueue.queue("problem_feedback_queue", entry);
    return stored ? { success: true, queuedLocally: true } : { success: false };
  },

  // Feedback on a LESSON page. Deliberately not reportProblem: that endpoint
  // takes an integer question_id and, for an actionable tag, queues an AI
  // rewrite of THAT QUESTION — so a note about a confusing worked example
  // would file itself against the drill the lesson is gating and then try to
  // repair it. Same log directory, different subject, no repair queue.
  async reportLesson({ kc, lessonTitle, questionId, tag, note }) {
    const entry = {
      kc: kc || "",
      lesson_title: lessonTitle || "",
      question_id: typeof questionId === "number" ? questionId : null,
      tag,
      note: note || "",
      client_id: DDFeedbackQueue.clientId(),
    };
    if (practiceMode === "backend") {
      try {
        const res = await apiFetch("/api/practice/lesson-feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(entry),
        });
        if (res.status === 401) {
            handleExpiredToken();
          } else if (res.ok) {
            DDFeedbackQueue.flush("lesson_feedback_queue", "/api/practice/lesson-feedback")
              .catch(() => {});
            return { success: true };
          }
      } catch (_) {
        /* fall through to local queue */
      }
    }
    // Same local fallback contract as reportProblem — a guest, an offline
    // session or a backend error must not silently drop what someone wrote.
    const stored = DDFeedbackQueue.queue("lesson_feedback_queue", entry);
    return stored ? { success: true, queuedLocally: true } : { success: false };
  },

  // Delivery lives in practice/feedback-queue.js; this is the practice-side
  // name for it, kept so callers do not have to know which module owns it.
  flushFeedbackQueues() {
    DDFeedbackQueue.flushAll();
  },
};

// Boot-time drain. Deferred so a queue of 50 never competes with the first
// paint or the first question fetch.
if (typeof window !== "undefined") {
  const _bootFlush = () => setTimeout(() => PracticeAPI.flushFeedbackQueues(), 2000);
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", _bootFlush, { once: true });
  } else {
    _bootFlush();
  }
}

/* ================================================================
   XP HOOKS — one wrap, every path that records learner data.

   The topbar seam (../xp.js) has to move whenever the learner enters
   ANYTHING: a graded submit, a placement probe, "I don't know yet", the
   felt-difficulty rating, a torch self-rating, a content flag. Those are
   six handlers spread over events.js, colab_mode.js and diagnostic-page.js,
   but every one of them ends up calling a method on this object — so the
   award belongs HERE, wrapped once, rather than as six calls that the next
   handler to be added will forget to make.

   `window.DeltaXP` is read at call time, not captured: xp.js loads before
   this file today, and a load-order change should degrade to "no XP for
   this attempt", never to a TypeError inside submit.

   The wrappers are transparent — same arguments, same return value, same
   rejection. An award never runs on the failure path, because a submit
   that threw recorded nothing.
   ================================================================ */
(function wirePracticeXp() {
  const award = (kind) => {
    try {
      window.DeltaXP?.award(kind);
    } catch (_) {
      /* gamification must never break a graded submit */
    }
  };

  const wrap = (name, kindFor) => {
    const original = PracticeAPI[name];
    if (typeof original !== "function") return;
    PracticeAPI[name] = async function (...args) {
      const result = await original.apply(this, args);
      award(kindFor(result, args));
      return result;
    };
  };

  // A miss still pays. The placement test is BUILT out of misses, and a bar
  // that only moved on a correct answer would charge the learner for using
  // the feature that finds their level.
  wrap("submitAnswer", (result) => (result && result.correct ? "answer_correct" : "answer_wrong"));
  // Torch / Colab self-rating and the local Pyodide engine path.
  wrap("recordLocalEval", (_r, args) => (args[1] ? "answer_correct" : "answer_wrong"));
  // Placement probe answered without a code attempt ("I don't know yet").
  wrap("diagnosticAnswer", (_r, args) => (args[1] === "dont_know" ? "placement_skip" : "placement_answer"));
  wrap("sendFeedback", () => "difficulty_rating");
  wrap("overrideCorrect", () => "override");
  wrap("reportProblem", () => "problem_report");
  // A lesson report is the same act on a different surface, and the seam
  // must not go quiet just because the learner is reading rather than drilling.
  wrap("reportLesson", () => "problem_report");
})();
