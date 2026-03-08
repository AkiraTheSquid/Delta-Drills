/* ================================================================
   PRACTICE UI — rendering + feedback widgets
   ================================================================ */

function renderQuestion(q, count) {
  if (curatedExcludedIds.has(q.question_id)) {
    PracticeAPI.getNextQuestion().then((nextQ) => renderQuestion(nextQ, count));
    return;
  }
  if (staleGaussianQuestion(q)) {
    PracticeAPI.getNextQuestion().then((nextQ) => renderQuestion(nextQ, count));
    return;
  }
  practiceQuestionCount = count;
  questionNumber.textContent = "Question " + practiceQuestionCount;
  questionText.textContent = q.question_text;
  subtopicLabel.textContent = q.topic ? `${q.topic}: ${q.subtopic}` : q.subtopic;
  difficultyLabel.textContent = "Difficulty: " + q.difficulty + " / 100";
  questionMetaTop.classList.add("hidden");

  // Cold-start calibration badge
  const overrideN = Number.isFinite(q.subtopic_n) ? q.subtopic_n : undefined;
  const coldStart = q.is_cold_start ?? isColdStart(q.subtopic, overrideN);
  const csIndex = Number.isFinite(q.subtopic_n) ? q.subtopic_n + 1 : coldStartIndex(q.subtopic, overrideN);
  if (coldStart && csIndex) {
    coldStartLabel.textContent = `Calibrating — ${csIndex} of 3`;
    coldStartBadge.classList.remove("hidden");
  } else {
    coldStartBadge.classList.add("hidden");
  }
  setTargetDifficultyInitial(getTargetDifficultyForQuestion(q));
  solutionCode.textContent = q.solution_code;
  overrideRow.classList.add("hidden");

  // Reset to pre-submit state
  practiceSubmitArea.classList.remove("hidden");
  practiceSubmitBtn.disabled = false;
  practiceFeedbackArea.classList.add("hidden");
  practiceFeedbackArea.classList.remove("checking");
  ewmaAccuracy.classList.add("hidden");
  ewmaAccuracyFill.style.width = "0%";
  showFeedbackButtons();
  questionMetaTop.classList.add("hidden");

  // Set up accuracy bar initial state (mirrors setTargetDifficultyInitial).
  // Backend mode: use p_current from the question response (adaptiveStateJson is null).
  // Pyodide mode: read from the adaptive state JSON.
  ewmaAccuracyPBefore = Number.isFinite(q.p_current)
    ? q.p_current
    : getEwmaFromAdaptiveState(q.subtopic);
  showEwmaAccuracyInitial(ewmaAccuracyPBefore, q.subtopic);

  // Reset AI explanation
  aiExplanationSection.classList.add("hidden");
  aiExplanationText.textContent = "";

  // Reset timer for next question if timed mode is on
  if (timedModeToggle.checked) {
    startTimer();
  }

  const pending = practiceProgress.pendingFeedback;
  if (pending) {
    if (pending.questionId === q.question_id) {
      applyPendingFeedbackState(pending);
    } else {
      practiceProgress.pendingFeedback = null;
      savePracticeProgress(practiceProgress);
    }
  }
}

function getTargetDifficultyForQuestion(q) {
  if (q && Number.isFinite(q.target_difficulty)) return q.target_difficulty;
  const fromState = getTargetDifficultyFromAdaptiveState(q?.subtopic);
  if (Number.isFinite(fromState)) return fromState;
  return Number.isFinite(q?.difficulty) ? q.difficulty : 0;
}

function showFeedbackButtons() {
  feedbackButtons.forEach((btn) => btn.classList.remove("hidden"));
  nextProblemBtn.classList.add("hidden");
}

function showNextProblemButton() {
  feedbackButtons.forEach((btn) => btn.classList.add("hidden"));
  nextProblemBtn.classList.remove("hidden");
}

function shortSubtopicName(subtopic) {
  if (!subtopic) return subtopic;
  const colon = subtopic.indexOf(": ");
  return colon >= 0 ? subtopic.slice(colon + 2) : subtopic;
}

function applyPendingFeedbackState(pending) {
  practiceSubmitArea.classList.add("hidden");
  practiceFeedbackArea.classList.remove("hidden");
  applyResult(!!pending.correct);
  questionMetaTop.classList.remove("hidden");
  overrideRow.classList.add("hidden");
  showNextProblemButton();
  setTargetDifficultyFinal(pending.oldTarget, pending.newTarget);
  if (Number.isFinite(pending.pAfter)) {
    setEwmaAccuracyFinal(pending.pBefore, pending.pAfter, pending.subtopic);
  }
}

// Apply correct/incorrect result to the feedback area UI.
function applyResult(correct) {
  resultBadge.textContent = correct ? "Correct" : "Incorrect";
  resultBadge.className = "result-badge " + (correct ? "correct" : "incorrect");
  overrideRow.classList.toggle("hidden", correct);
  practiceFeedbackArea.classList.remove("checking");
  questionMetaTop.classList.remove("hidden");
  if (correct) {
    feedbackPrompt.textContent = "Nailed it! How hard should we go next?";
    feedbackButtons.forEach((btn, i) => {
      btn.textContent = ["Inch it up", "Rev the engine", "Full throttle"][i];
    });
  } else {
    feedbackPrompt.textContent = "Tough one. How much should we dial it back?";
    feedbackButtons.forEach((btn, i) => {
      btn.textContent = ["Just a hair easier", "Take the edge off", "Back to basics"][i];
    });
  }
}
