/* ================================================================
   PRACTICE EVENTS — submit + feedback actions
   ================================================================ */

practiceSubmitBtn.addEventListener("click", async () => {
  const q = PracticeAPI.currentQuestion;
  const userCode = codeEditor.value;
  stopTimer();
  practiceSubmitBtn.disabled = true;
  let result;
  try {
    result = await PracticeAPI.submitAnswer(q.question_id, userCode);
  } catch (err) {
    outputArea.textContent = "Submit failed: " + err.message;
    practiceSubmitBtn.disabled = false;
    return;
  }

  const solCode = q.solution_code || result.solution_code || "";
  const actualOutput = result.actual_output || "";
  const expectedOutput = result.expected_output || q.expected_output || "";

  solutionCode.textContent = solCode;
  practiceSubmitArea.classList.add("hidden");
  practiceFeedbackArea.classList.remove("hidden");

  applyResult(result.correct);
  practiceProgress.lastResultCorrect = result.correct;
  practiceProgress.currentTargetDifficulty = getTargetDifficultyForQuestion(q);
  savePracticeProgress(practiceProgress);
  resetTimerToInput();
  if (practiceMode === "backend" || practiceMode === "supabase") {
    aiExplanationSection.classList.remove("hidden");
    aiExplanationText.textContent = "Loading explanation...";
    fetchAIExplanation(q.question_text, solCode, userCode, actualOutput, expectedOutput);
  }

  // NOTE: ARENA unlock interstitial does NOT fire on Submit — student
  // needs to see feedback + give a difficulty rating first. The unlock
  // pops on the Next-problem click (handler below). We do, however,
  // warm the backend subtopic-score cache here in the background so
  // the Next-problem click can evaluate gates instantly without a
  // network round-trip.
  if (window.ArenaUnlock && typeof window.ArenaUnlock.refreshScores === "function") {
    window.ArenaUnlock.refreshScores().catch(() => {});
  }
});

overrideCorrectBtn.addEventListener("click", async () => {
  const q = PracticeAPI.currentQuestion;
  await PracticeAPI.overrideCorrect(q.question_id);

  resultBadge.textContent = "Correct";
  resultBadge.className = "result-badge correct";
  feedbackPrompt.textContent = "Nailed it! How hard should we go next?";
  const labels = ["Inch it up", "Rev the engine", "Full throttle"];
  feedbackButtons.forEach((btn, i) => {
    btn.textContent = labels[i];
  });
  overrideRow.classList.add("hidden");
  practiceProgress.lastResultCorrect = true;
  savePracticeProgress(practiceProgress);
});

feedbackButtons.forEach((btn) => {
  btn.addEventListener("click", async () => {
    const feedback = btn.dataset.feedback;
    const q = PracticeAPI.currentQuestion;
    const calibrationQuestion = typeof isCalibrationQuestion === "function" && isCalibrationQuestion(q);
    const oldTarget = Number.isFinite(practiceProgress.currentTargetDifficulty)
      ? practiceProgress.currentTargetDifficulty
      : getTargetDifficultyForQuestion(q);
    const pBefore = ewmaAccuracyPBefore;
    let response;
    try {
      response = await PracticeAPI.sendFeedback(q.question_id, feedback);
    } catch (err) {
      outputArea.textContent = "Feedback failed: " + err.message;
      return;
    }
    const backendTarget = Number.isFinite(response?.target_difficulty_after)
      ? response.target_difficulty_after
      : null;
    const newTarget = Number.isFinite(backendTarget)
      ? backendTarget
      : getTargetDifficultyFromAdaptiveState(q.subtopic) ?? oldTarget;

    const pAfter = Number.isFinite(response?.p_after)
      ? response.p_after
      : getEwmaFromAdaptiveState(q.subtopic);

    if (!practiceProgress.completedQuestionIds.includes(q.question_id)) {
      practiceProgress.completedQuestionIds.push(q.question_id);
    }

    showNextProblemButton();
    animateTargetDifficulty(oldTarget, newTarget, () => {
      setTargetDifficultyFinal(oldTarget, newTarget);
    });
    if (!calibrationQuestion && Number.isFinite(pAfter)) {
      showEwmaAccuracy(pBefore, pAfter, q.subtopic);
    } else {
      showEwmaAccuracyCalibration(q.subtopic);
    }

    practiceProgress.pendingFeedback = {
      questionId: q.question_id,
      subtopic: q.subtopic,
      oldTarget,
      newTarget,
      correct: !!practiceProgress.lastResultCorrect,
      pBefore,
      pAfter,
    };
    savePracticeProgress(practiceProgress);
  });
});

const _loadNextPracticeQuestion = async () => {
  practiceProgress.currentQuestion = null;
  practiceProgress.pendingFeedback = null;
  practiceProgress.currentTargetDifficulty = null;
  practiceProgress.lastResultCorrect = null;

  // Reset to pre-submit state (ready for next question)
  practiceSubmitArea.classList.remove("hidden");
  practiceFeedbackArea.classList.add("hidden");
  ewmaAccuracy.classList.add("hidden");
  ewmaAccuracyFill.style.width = "0%";
  showFeedbackButtons();
  questionMetaTop.classList.add("hidden");

  // Reset code editor
  codeEditor.value = "import numpy as np\nnp.random.seed(0)\n\n# Write your solution here\n";
  outputArea.textContent = "";

  // Load next question
  const nextQ = await PracticeAPI.getNextQuestion();
  const nextCount = practiceQuestionCount + 1;
  practiceProgress.questionCount = nextCount;
  savePracticeProgress(practiceProgress);
  renderQuestion(nextQ, nextCount);
};

nextProblemBtn.addEventListener("click", async () => {
  // ARENA unlock interstitial — show a card for the next-just-unlocked
  // ARENA exercise before loading the next Delta Drills question. The
  // interstitial's Continue button calls _loadNextPracticeQuestion.
  // Returns false (and we fall through to the normal flow) when there
  // is no newly-unlocked exercise waiting. tryShow is async — must await.
  if (window.ArenaUnlock && typeof window.ArenaUnlock.tryShow === "function") {
    if (await window.ArenaUnlock.tryShow(_loadNextPracticeQuestion)) return;
  }
  await _loadNextPracticeQuestion();
});
