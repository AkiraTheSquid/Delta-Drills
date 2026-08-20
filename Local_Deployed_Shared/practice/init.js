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
  detectPracticeMode();
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
