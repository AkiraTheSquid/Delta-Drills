/* ================================================================
   PRACTICE EVENTS — submit + feedback actions
   ================================================================ */

practiceSubmitBtn.addEventListener("click", async () => {
  const q = PracticeAPI.currentQuestion;
  const userCode = codeEditor.value;
  PracticeSession.pauseForGrading();
  practiceSubmitBtn.disabled = true;
  let result;
  try {
    result = await PracticeAPI.submitAnswer(q.question_id, userCode);
  } catch (err) {
    if (PracticeAPI.currentQuestion !== q) return;
    outputArea.textContent = "Submit failed: " + err.message;
    practiceSubmitBtn.disabled = false;
    PracticeSession.resumeAnswerPhase();
    return;
  }

  // The question can change while grading is in flight (Skip, End session →
  // new session). A stale grade must not repaint the current question or
  // start its review countdown.
  if (PracticeAPI.currentQuestion !== q) return;

  const solCode = q.solution_code || result.solution_code || "";
  const actualOutput = result.actual_output || "";
  const expectedOutput = result.expected_output || q.expected_output || "";

  solutionCode.textContent = solCode;
  practiceSubmitArea.classList.add("hidden");
  practiceFeedbackArea.classList.remove("hidden");

  applyResult(result.correct);
  if (typeof renderFailedTests === "function") renderFailedTests(result, q);
  practiceProgress.lastResultCorrect = result.correct;
  practiceProgress.currentTargetDifficulty = getTargetDifficultyForQuestion(q);
  savePracticeProgress(practiceProgress);
  // Preserve enough of the grade UI to reconstruct this review after a pause
  // or reload, including the learner's submitted code and failed cases.
  PracticeSession.recordReviewResult({
    correct: !!result.correct,
    userCode,
    solutionCode: solCode,
    result: {
      correct: !!result.correct,
      failed_tests: Array.isArray(result.failed_tests) ? result.failed_tests : [],
    },
  });
  // Grade landed — strict review countdown starts now.
  PracticeSession.beginReviewPhase();

  // Placement probe: the backend already recorded it at /submit — there is no
  // pending attempt and no felt-difficulty step. Go straight to Next.
  if (q.diagnostic_active && practiceMode === "backend") {
    feedbackPrompt.textContent = result.correct
      ? "Placement recorded — on to the next probe."
      : "Placement recorded — misses here just pin down your level.";
    if (!practiceProgress.completedQuestionIds.includes(q.question_id)) {
      practiceProgress.completedQuestionIds.push(q.question_id);
    }
    savePracticeProgress(practiceProgress);
    showNextProblemButton();
    _notifyIfPlacementDone();
  }
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

if (showHintBtn) {
  showHintBtn.addEventListener("click", () => {
    if (hintSection) hintSection.classList.toggle("hidden");
  });
}

overrideCorrectBtn.addEventListener("click", async () => {
  const q = PracticeAPI.currentQuestion;
  await PracticeAPI.overrideCorrect(q.question_id);

  resultBadge.textContent = "Correct";
  resultBadge.className = "result-badge correct";
  feedbackPrompt.textContent = "Nice work. How did that feel?";
  const labels = ["About right", "A little easy", "Way too easy"];
  feedbackButtons.forEach((btn, i) => {
    btn.textContent = labels[i];
  });
  overrideRow.classList.add("hidden");
  if (typeof resetMissedFactRow === "function") resetMissedFactRow();
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
    // Acknowledge the click INSTANTLY — the network round-trip below can be
    // slow (backend boot, cold path) and a silent 5-10s wait reads as "did
    // that even register?" (tester hit exactly this). Disabling also blocks
    // a double-click from posting the BKT update twice.
    feedbackButtons.forEach((b) => (b.disabled = true));
    btn.classList.add("feedback-btn--pressed");
    let response;
    try {
      response = await PracticeAPI.sendFeedback(q.question_id, feedback);
    } catch (err) {
      outputArea.textContent = "Feedback failed: " + err.message;
      feedbackButtons.forEach((b) => (b.disabled = false));
      btn.classList.remove("feedback-btn--pressed");
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

// Per-problem content-quality flags. One click logs immediately; the optional
// note (if typed) rides along with whichever tag is clicked. Non-blocking —
// never interferes with the difficulty rating or Next-problem flow.
problemFlagButtons.forEach((btn) => {
  btn.addEventListener("click", async () => {
    const q = PracticeAPI.currentQuestion;
    if (!q) return;
    const tag = btn.dataset.flag;
    const note = problemFeedbackNote ? problemFeedbackNote.value.trim() : "";
    problemFlagButtons.forEach((b) => b.classList.remove("flagged"));
    btn.classList.add("flagged");
    try {
      await PracticeAPI.reportProblem(
        q.question_id,
        tag,
        note,
        practiceProgress.lastResultCorrect,
      );
      if (problemFeedbackStatus) {
        problemFeedbackStatus.classList.remove("hidden");
      }
    } catch (_) {
      /* feedback is best-effort; ignore */
    }
  });
});

const _resetProblemFeedbackRow = () => {
  problemFlagButtons.forEach((b) => b.classList.remove("flagged"));
  if (problemFeedbackNote) problemFeedbackNote.value = "";
  if (problemFeedbackStatus) problemFeedbackStatus.classList.add("hidden");
};

const _loadNextPracticeQuestion = async () => {
  // Session quota reached — every advance path (Next, Skip, torch self-rate,
  // "I don't know yet", forced advance) funnels through here, so this is the
  // single place the session can end.
  if (PracticeSession.shouldFinishInsteadOfAdvance()) {
    PracticeSession.finish("complete");
    return;
  }
  // Kill both countdowns while the next question loads — otherwise a Skip
  // near 00:00 leaves the old answer timer live and it force-submits the
  // question being skipped.
  PracticeSession.pauseForAdvance();
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
  _resetProblemFeedbackRow();
  if (typeof hideFailedTests === "function") hideFailedTests();
  questionMetaTop.classList.add("hidden");

  // Reset code editor
  codeEditor.value = "import numpy as np\nnp.random.seed(0)\n\n# Write your solution here\n";
  outputArea.textContent = "";

  // Load next question
  const nextQ = await PracticeAPI.getNextQuestion();
  const nextCount = practiceQuestionCount + 1;
  practiceProgress.questionCount = nextCount;
  savePracticeProgress(practiceProgress);
  // First-encounter gate: when this question's target KC has never been
  // taught, LessonGate shows the introducing lesson first and renders the
  // question (starting the session answer timer) only on Continue.
  if (window.LessonGate && (await window.LessonGate.maybeShow(nextQ, () => renderQuestion(nextQ, nextCount)))) {
    return;
  }
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

// --- Placement diagnostic helpers ------------------------------------------
// After a probe is recorded (submit or "I don't know yet"), check whether the
// diagnostic just finished; if so, tell the learner + refresh the adaptive
// state so the seeded BKT mastery reaches the graph/stats views immediately.
async function _notifyIfPlacementDone() {
  try {
    const status = await PracticeAPI.diagnosticStatus();
    if (!status || status.active || !status.completed_at) return;
    if (typeof loadBackendAdaptiveState === "function") {
      await loadBackendAdaptiveState();
    }
    refreshPlacementStartBtn().catch(() => {});
    emitPracticeStateChanged();
    if (typeof showPracticeModeNotice === "function") {
      const strongest = (status.areas || []).slice().sort((a, b) => b.theta - a.theta)[0];
      showPracticeModeNotice(
        `Placement complete after ${status.probes_done} questions — practice now starts at your level` +
        (strongest ? ` (strongest area: ${strongest.topic}).` : "."),
      );
    }
  } catch (_) {
    /* best-effort — never blocks the practice flow */
  }
}

// "Take placement diagnostic" — the way IN for accounts with existing history
// (the diagnostic only auto-starts for zero-attempt users). Shown whenever the
// backend says no diagnostic is running; label flips to "Retake" once one has
// completed. Starting re-places the learner: BKT seeding at finish only raises
// practiced atoms, so earned mastery is safe.
async function refreshPlacementStartBtn() {
  if (typeof placementStartBtn === "undefined" || !placementStartBtn) return;
  const status = await PracticeAPI.diagnosticStatus();
  if (!status || status.active) {
    placementStartBtn.classList.add("hidden");
    return;
  }
  placementStartBtn.textContent = status.completed_at
    ? "Retake placement diagnostic"
    : "Take placement diagnostic";
  placementStartBtn.disabled = false;
  placementStartBtn.classList.remove("hidden");
}
window.refreshPlacementStartBtn = refreshPlacementStartBtn;

if (typeof placementStartBtn !== "undefined" && placementStartBtn) {
  placementStartBtn.addEventListener("click", async () => {
    placementStartBtn.disabled = true;
    try {
      const status = await PracticeAPI.diagnosticStart();
      if (!status) throw new Error("not signed in to the practice backend");
      placementStartBtn.classList.add("hidden");
      if (typeof showPracticeModeNotice === "function") {
        showPracticeModeNotice(
          "Placement diagnostic started — a few adaptive questions to locate your level.",
        );
      }
      // The diagnostic is its own flow with its own (backend-driven) length —
      // don't let the current session's quota gate swallow it ("Session
      // complete" while the backend diagnostic stays active). End the block
      // and let the learner start a fresh one; its questions ARE the probes.
      if (PracticeSession.isActive()) {
        PracticeSession.finish("placement");
      } else {
        await _loadNextPracticeQuestion();
      }
    } catch (err) {
      outputArea.textContent = "Could not start the placement diagnostic: " + err.message;
      placementStartBtn.disabled = false;
    }
  });
}

// "I don't know yet" — placement-only: records a diagnostic miss WITHOUT a
// code attempt (strong evidence the item sits above the learner's level),
// then advances. This is the fast path that keeps beginners from burning
// time on problems they already know they can't solve.
if (typeof practiceDontKnowBtn !== "undefined" && practiceDontKnowBtn) {
  practiceDontKnowBtn.addEventListener("click", async () => {
    const q = PracticeAPI.currentQuestion;
    if (!q || !q.diagnostic_active) return;
    practiceDontKnowBtn.disabled = true;
    try {
      await PracticeAPI.diagnosticAnswer(q.question_id, "dont_know");
      await _notifyIfPlacementDone();
      await _loadNextPracticeQuestion();
    } catch (err) {
      outputArea.textContent = "Could not record the answer: " + err.message;
      practiceDontKnowBtn.disabled = false;
    }
  });
}

// --- Skip (P1.8): advance without grading and WITHOUT claiming a look-up.
// Nothing is recorded; the served-question list keeps it from repeating next.
if (practiceSkipBtn) {
  practiceSkipBtn.addEventListener("click", async () => {
    practiceSkipBtn.disabled = true;
    try {
      await _loadNextPracticeQuestion();
    } catch (err) {
      outputArea.textContent = "Could not load the next question: " + err.message;
    } finally {
      practiceSkipBtn.disabled = false;
    }
  });
}

// --- Torch self-rating (P0.1): torch drills run in Colab, not in-app. Record a
// local-eval attempt (no doomed server grading) when in backend mode, then
// advance. In guest/local mode recordLocalEval is a no-op; we still advance.
const _rateTorchAndAdvance = async (correct) => {
  const q = PracticeAPI.currentQuestion;
  if (!q) return;
  if (torchRateSolved) torchRateSolved.disabled = true;
  if (torchRateLookedUp) torchRateLookedUp.disabled = true;
  try {
    if (typeof PracticeAPI.recordLocalEval === "function") {
      await PracticeAPI.recordLocalEval(q.question_id, correct);
    }
  } catch (_) {
    /* best-effort — still advance so the learner isn't stuck */
  } finally {
    if (torchRateSolved) torchRateSolved.disabled = false;
    if (torchRateLookedUp) torchRateLookedUp.disabled = false;
  }
  await _loadNextPracticeQuestion();
};
if (torchRateSolved) torchRateSolved.addEventListener("click", () => _rateTorchAndAdvance(true));
if (torchRateLookedUp) torchRateLookedUp.addEventListener("click", () => _rateTorchAndAdvance(false));

// --- "Missed one concrete thing" (P1.5): a separate signal so a single missed
// fact isn't read as "too hard". Rides the non-blocking problem-feedback channel.
if (missedFactBtn) {
  missedFactBtn.addEventListener("click", async () => {
    const q = PracticeAPI.currentQuestion;
    if (!q) return;
    missedFactBtn.classList.add("flagged");
    if (missedFactStatus) missedFactStatus.classList.remove("hidden");
    try {
      await PracticeAPI.reportProblem(
        q.question_id,
        "missed_fact",
        "Learner: wrong due to one concrete thing, not overall difficulty.",
        practiceProgress.lastResultCorrect,
      );
    } catch (_) {
      /* feedback is best-effort; ignore */
    }
  });
}
