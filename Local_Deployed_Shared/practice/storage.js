/* ================================================================
   PRACTICE STORAGE — progress persistence
   ================================================================ */

const getPracticeStorageKey = () => {
  const keyEmail = typeof authEmail === "string" && authEmail.trim() ? authEmail.trim() : "guest";
  return `practice_progress_${keyEmail}`;
};

const loadPracticeProgress = () => {
  const saved = localStorage.getItem(getPracticeStorageKey());
  if (!saved) return null;
  try {
    return JSON.parse(saved);
  } catch (e) {
    return null;
  }
};

const savePracticeProgress = (progress) => {
  localStorage.setItem(getPracticeStorageKey(), JSON.stringify(progress));
};

const savedProgress = loadPracticeProgress();

if (
  (savedProgress?.currentQuestionId && curatedExcludedIds.has(savedProgress.currentQuestionId)) ||
  (savedProgress?.currentQuestion?.question_id &&
    curatedExcludedIds.has(savedProgress.currentQuestion.question_id)) ||
  staleGaussianQuestion(savedProgress?.currentQuestion)
) {
  savedProgress.currentQuestionId = null;
  savedProgress.currentQuestion = null;
  savePracticeProgress(savedProgress);
}

const practiceProgress = {
  currentQuestion: savedProgress?.currentQuestion || null,
  currentQuestionId: savedProgress?.currentQuestionId || practiceQuestionPool[0].question_id,
  questionCount: Number.isFinite(savedProgress?.questionCount) ? savedProgress.questionCount : 1,
  completedQuestionIds: Array.isArray(savedProgress?.completedQuestionIds)
    ? savedProgress.completedQuestionIds
    : [],
  // Every drill rendered, including skips. This is distinct from completion:
  // a skipped drill was still shown and must not leak from the KC ladder back
  // into a later queue slot.
  shownQuestionIds: Array.isArray(savedProgress?.shownQuestionIds)
    ? savedProgress.shownQuestionIds.filter(Number.isFinite)
    : [],
  pendingFeedback: savedProgress?.pendingFeedback || null,
  currentTargetDifficulty: Number.isFinite(savedProgress?.currentTargetDifficulty)
    ? savedProgress.currentTargetDifficulty
    : null,
  lastResultCorrect:
    typeof savedProgress?.lastResultCorrect === "boolean" ? savedProgress.lastResultCorrect : null,
};

let practiceQuestionCount = practiceProgress.questionCount;
