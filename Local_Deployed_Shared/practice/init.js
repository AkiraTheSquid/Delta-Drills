/* ================================================================
   PRACTICE INIT — bootstrap
   ================================================================ */

function syncCurrentQuestion() {
  if (practiceProgress.currentQuestion) {
    if (!isPracticeQuestionAllowed(practiceProgress.currentQuestion)) {
      practiceProgress.currentQuestion = null;
      practiceProgress.currentQuestionId = null;
      return;
    }
    PracticeAPI.currentQuestion = practiceProgress.currentQuestion;
    return;
  }
  const savedIndex = practiceQuestionPool.findIndex(
    (q) => q.question_id === practiceProgress.currentQuestionId
  );
  practiceQuestionIndex = savedIndex >= 0 ? savedIndex : 0;
  PracticeAPI.currentQuestion = practiceQuestionPool[practiceQuestionIndex];
}

function invalidateLegacyBackendQuestion() {
  if (practiceMode !== "backend" || !practiceProgress.currentQuestion) return;

  // Backend responses now always include `p_current` (nullable). Older cached
  // questions may omit it entirely, which suppresses the accuracy delta after a
  // reload because renderQuestion can't recover the pre-answer EWMA.
  if (!Object.prototype.hasOwnProperty.call(practiceProgress.currentQuestion, "p_current")) {
    practiceProgress.currentQuestion = null;
    practiceProgress.currentQuestionId = null;
    practiceProgress.pendingFeedback = null;
    return;
  }

  // Visual questions now depend on structured test-case metadata for image
  // preview rendering. Older cached backend questions may have the image flag
  // but not the canonical visual setup data, which leads to broken previews.
  if (practiceProgress.currentQuestion.supports_visual_output) {
    // A v1 resumable-session snapshot proves this question was saved by the
    // current runtime. Keep it so Pause & exit can return to visual coding
    // work too; ordinary stale cached visual questions still refresh.
    const resumable =
      typeof PracticeSession !== "undefined" &&
      PracticeSession.hasSavedQuestion(practiceProgress.currentQuestion.question_id);
    if (resumable) return;
    // Visual questions are especially sensitive to stale cached artifacts.
    // Force a fresh backend fetch rather than trusting any restored copy.
    practiceProgress.currentQuestion = null;
    practiceProgress.currentQuestionId = null;
    practiceProgress.pendingFeedback = null;
    return;
  }
}

const initPractice = async () => {
  // Before the mode is decided, not after: getPracticeMode() answers "local"
  // for anyone without a token, and local mode has no diagnostic, no lessons
  // and no student model. guest-session.js gives a signed-out visitor a
  // backend session so they get the real app. It resolves false (leaving the
  // mode "local", exactly as before) when the backend can't be reached.
  if (typeof window.DDGuest?.ensure === "function") {
    await window.DDGuest.ensure();
  }
  detectPracticeMode();
  /* 🔴 ANNOUNCE THE MODE. practice/diagnostic-page.js parses BEFORE this file
     and its first status call is worthless until the line above has run —
     `PracticeAPI.diagnosticStatus()` short-circuits to null in local mode, and
     the Learner Home paints that null as "Sign in to take the placement test."
     Fired here rather than at the end of initPractice because the mode is the
     only thing that call is waiting on; the question bank below is a second or
     more of work the placement status does not need. */
  window.DDPracticeModeReady = true;
  window.dispatchEvent(new CustomEvent("delta:practice-mode-ready"));
  await loadQuestionsBank();

  // For supabase/local modes, load engine + questions + state
  if (practiceMode !== "backend") {
    const pyodide = await initPyodide();
    if (pyodide) {
      await loadPracticeEngine(pyodide);
    }
    await loadAdaptiveState();
  } else {
    // Backend mode: skip Pyodide engine but still hydrate adaptiveStateJson
    // from /api/practice/state so concept-graph/atom_readiness.js can
    // bridge atoms onto real per-subtopic baselines.
    await loadBackendAdaptiveState();
    // Refresh explicit Diagnostic-tab entry/status. Non-blocking.
    if (typeof refreshPlacementStartBtn === "function") {
      refreshPlacementStartBtn().catch(() => {});
    }
  }

  if (practiceProgress.currentQuestion) {
    practiceProgress.currentQuestion = hydrateSavedPracticeQuestionFromBank(
      practiceProgress.currentQuestion
    );
    if (practiceProgress.currentQuestion) {
      if (practiceProgress.currentQuestion._artifactChanged) {
        practiceProgress.pendingFeedback = null;
        practiceProgress.lastResultCorrect = null;
      }
      practiceProgress.currentQuestionId = practiceProgress.currentQuestion.question_id;
    }
  }

  // Enrich stale saved question with topic/subtopic from the current questions bank.
  // Questions saved from old backend-mode sessions lack `topic` (NextQuestionResponse
  // didn't include it) and may have a combined subtopic like "Numpy: Subtopic".
  if (practiceProgress.currentQuestion && !practiceProgress.currentQuestion.topic) {
    if (questionsBank) {
      const savedId = practiceProgress.currentQuestion.question_id;
      const bankQ = questionsBank.find((q) => q.id === savedId);
      if (bankQ) {
        practiceProgress.currentQuestion.topic = bankQ.topic;
        practiceProgress.currentQuestion.subtopic = bankQ.subtopic;
      } else {
        practiceProgress.currentQuestion = null;
      }
    } else {
      // No questions bank available (backend mode) — discard stale question so
      // a fresh one with topic is fetched from the backend.
      practiceProgress.currentQuestion = null;
    }
  }

  invalidateLegacyBackendQuestion();

  syncCurrentQuestion();

  if (practiceProgress.currentQuestion) {
    savePracticeProgress(practiceProgress);
    renderQuestion(PracticeAPI.currentQuestion, practiceQuestionCount);
    return;
  }
  /* 🔴 DO NOT FETCH A QUESTION INTO A LIVE PLACEMENT. While a placement is
     ACTIVE the backend's next-question endpoint serves PROBES — there is no
     practice stream running beside a live test — so this boot fetch handed the
     Learner Home a probe with `diagnostic_active: true`, started its 2:00 on it
     and left it counting down behind the idle surface, where it expired into a
     recorded miss on a question the learner never saw. diagnostic-page.js
     documents that measurement in its "SKIP FOR NOW" note, which is why the
     mid-test exit is hidden; this is the same hazard reached through a reload
     rather than a tab switch.

     It is also what makes pausing the placement stick (diagnostic-page.js
     `pause()`): pausing clears the saved probe, so a reload lands exactly here,
     and without this guard the next boot dealt the learner straight back into a
     running probe. NO extra state is needed to know a pause happened — "a
     placement is active and no probe is saved" is the paused state.

     A learner who reloads MID-probe never reaches this line: their probe is in
     `practiceProgress.currentQuestion` and the early return above renders it,
     clock and all.

     `diagnosticStatus()` answers null outside backend mode and
     `{unavailable:true}` when the call fails, and neither is "a test is
     running" — an outage must not leave the practice page permanently empty. */
  const diag = await PracticeAPI.diagnosticStatus();
  if (diag && !diag.unavailable && diag.active) {
    savePracticeProgress(practiceProgress);
    /* Send them where the resume is. The placement page renders "Load next
       placement question" for exactly this state; the Learner Home would show
       an idle dial and no explanation of why no question came. `switchTab` is a
       top-level const in app.js — classic scripts share one global lexical
       scope, and app.js parses first — with the window read as the fallback. */
    /* 🔴 NEVER OFF A SOLO ROUTE. `/knowledge-graph`, `/courses`, `/notebooks`
       and the rest are pathname deep links that solo-route.js stamps before
       first paint (html.dd-solo), and they are pages people are SENT. A learner
       who happens to have an unfinished placement would have had the link they
       followed swapped out from under them for the placement page. The route
       the visitor asked for wins; the skipped fetch above is what this guard is
       protecting, and it has already happened by here. */
    const solo = document.documentElement.classList.contains("dd-solo");
    const go = typeof switchTab !== "undefined" ? switchTab : window.switchTab;
    if (!solo && typeof go === "function") go("placement");
    window.DiagnosticPage?.refresh?.();
    return;
  }
  const nextQ = await PracticeAPI.getNextQuestion();
  savePracticeProgress(practiceProgress);
  renderQuestion(nextQ, practiceQuestionCount);
};

async function refreshPracticeQuestionForPreferences() {
  await loadQuestionsBank();
  if (practiceProgress.currentQuestion) {
    practiceProgress.currentQuestion = hydrateSavedPracticeQuestionFromBank(
      practiceProgress.currentQuestion
    );
    if (practiceProgress.currentQuestion) {
      practiceProgress.currentQuestionId = practiceProgress.currentQuestion.question_id;
    }
  }
  if (practiceProgress.currentQuestion && isPracticeQuestionAllowed(practiceProgress.currentQuestion)) {
    savePracticeProgress(practiceProgress);
    renderQuestion(practiceProgress.currentQuestion, practiceQuestionCount);
    return;
  }
  practiceProgress.currentQuestion = null;
  practiceProgress.currentQuestionId = null;
  savePracticeProgress(practiceProgress);
  if (document.getElementById("page-practice")?.classList.contains("hidden")) return;
  try {
    const nextQ = await PracticeAPI.getNextQuestion();
    renderQuestion(nextQ, practiceQuestionCount);
  } catch (err) {
    questionText.textContent = err.message || "No enabled practice sections.";
    practiceSubmitArea.classList.add("hidden");
    practiceFeedbackArea.classList.add("hidden");
  }
}
