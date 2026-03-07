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
