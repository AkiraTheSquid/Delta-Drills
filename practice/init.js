/* ================================================================
   PRACTICE INIT — bootstrap
   ================================================================ */

function syncCurrentQuestion() {
  if (practiceProgress.currentQuestion) {
    PracticeAPI.currentQuestion = practiceProgress.currentQuestion;
    return;
  }
  const savedIndex = practiceQuestionPool.findIndex(
    (q) => q.question_id === practiceProgress.currentQuestionId
  );
  practiceQuestionIndex = savedIndex >= 0 ? savedIndex : 0;
  PracticeAPI.currentQuestion = practiceQuestionPool[practiceQuestionIndex];
}

const initPractice = async () => {
  detectPracticeMode();

  // For supabase/local modes, load engine + questions + state
  if (practiceMode !== "backend") {
    const pyodide = await initPyodide();
    if (pyodide) {
      await loadPracticeEngine(pyodide);
    }
    await loadQuestionsBank();
    await loadAdaptiveState();
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
