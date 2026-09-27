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
  // for anyone without a token, and local mode has no lessons
  // and no student model. guest-session.js gives a signed-out visitor a
  // backend session so they get the real app. It resolves false (leaving the
  // mode "local", exactly as before) when the backend can't be reached.
  if (typeof window.DDGuest?.ensure === "function") {
    await window.DDGuest.ensure();
  }
  detectPracticeMode();
  /* 🔴 ANNOUNCE THE MODE. Every surface that reads the backend on load
     (xp.js, survey.js, course-pick.js, courses.js, activity-chart.js,
     practice-target.js) parses BEFORE this file, and its first call is
     worthless until the line above has run — local mode answers null. Fired
     here rather than at the end of initPractice because the mode is the only
     thing they wait on; the question bank below is a second or more. */
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
  /* 🔴 A FAILED BOOT FETCH LANDS ON THE IDLE SURFACE, NOT IN THE CONSOLE.
     practice.js calls initPractice() bare, so a throw here was an unhandled
     rejection and nothing on screen said why no question came. The backend
     answers 409 `content_exhausted` when the course has run dry for this
     learner (app/practice/questions_router.py), and Seth met it on
     2026-09-09 the hour he finished his placement: the page booted with a
     silent error, and Start then dead-ended on the same sentence a second
     later. `deadEnd` is what the Start path already does with it — idle
     surface back, sentence on #session-summary, saved block left offerable —
     so the boot failure reads the same as the in-session one. */
  let nextQ;
  try {
    nextQ = await PracticeAPI.getNextQuestion();
  } catch (err) {
    const message = err?.message || "Could not load a question. Try again in a moment.";
    if (typeof PracticeSession !== "undefined" && typeof PracticeSession.deadEnd === "function") {
      PracticeSession.deadEnd(message);
    } else {
      console.warn("[practice] boot fetch failed:", message);
    }
    return;
  }
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
