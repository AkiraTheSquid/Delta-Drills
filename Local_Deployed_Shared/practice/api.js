/* ================================================================
   PRACTICE API — backend / supabase / local routing
   ================================================================ */

function emitPracticeStateChanged() {
  window.dispatchEvent(new CustomEvent("delta:practice-state-changed"));
}

const PracticeAPI = {
  currentQuestion: practiceQuestionPool[0],

  async recordLocalEval(questionId, correct) {
    if (practiceMode !== "backend") return;
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
        throw new Error("No enabled practice sections. Re-enable at least one area in Statistics.");
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

  /* Record what the learner reported, and run it through the SAME mastery
     chain a graded attempt used to.

     Practice no longer executes code — the learner runs the problem in its
     Colab notebook and says whether it ran. That changes where `correct` comes
     from and nothing downstream of it:

       * backend mode posts `/submit-local-eval`, which the backend already
         used for torch drills and which runs the identical BKT / FIRe / decay
         chain as server-side grading. No new endpoint, no second scoring path.
       * guest/local mode calls `engine_api.submit_answer` in Pyodide, exactly
         as the old in-browser grader did once it had decided `correct`.

     The shape of the return value is unchanged (`{correct, failed_tests}`) so
     the review UI, the resumable-session snapshot and the difficulty-rating
     step did not have to learn a new one. `failed_tests` is always empty:
     nothing ran here, so there are no cases to report.

     `correct` is the learner's claim. It is deliberately trusted — the whole
     design assumes an adult drilling for themselves, and `Undo` on the review
     screen (`/override`) is the correction path when they misclick. */
  async recordSelfReport(questionId, correct) {
    const q = this.currentQuestion;
    correct = !!correct;

    if (practiceMode === "backend") {
      await this.recordLocalEval(questionId, correct);
      return { correct, failed_tests: [] };
    }

    // Guest/local: the adaptive engine lives in Pyodide. Missing engine or
    // state means progress is not being tracked at all, which is already true
    // before this call — advance rather than trapping the learner.
    if (practiceEngineLoaded && adaptiveStateJson) {
      const pyodide = await initPyodide();
      if (pyodide) {
        const engine = pyodide.globals.get("engine_api");
        adaptiveStateJson = engine.submit_answer(
          adaptiveStateJson,
          questionId,
          q ? q.subtopic : "",
          (q && q.difficulty) || 50,
          correct
        );
        await saveAdaptiveState();
      }
    }
    return { correct, failed_tests: [] };
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
