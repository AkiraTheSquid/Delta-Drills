/* ================================================================
   PRACTICE EVENTS — submit + feedback actions
   ================================================================ */

/* How long the difficulty answer lets the topbar move before it advances.
   The concept pill's fill is a 0.55s transition (styles/concept-pill.css) and
   the level pill's is the same (styles/xp.css); this is that plus a beat, so
   the reading has finished arriving rather than being cut off mid-slide. Kept
   here, in the file that does the waiting, rather than read off the
   stylesheet — a computed-style read of a transition that has not started yet
   returns the duration of whatever the element is transitioning now, which is
   nothing. If the stylesheet's duration changes, change this with it. */
const TOPBAR_SETTLE_MS = 700;

practiceSubmitBtn.addEventListener("click", async () => {
  const q = PracticeAPI.currentQuestion;
  const userCode = window.DeltaNotebook?.submissionCode() || codeEditor.value;
  PracticeSession.pauseForGrading();
  // Same contract for a placement probe's fixed clock: once the grade is in
  // flight the learner is no longer answering, so the countdown stops instead
  // of expiring underneath the result.
  window.PlacementTimer?.pauseForGrading();
  practiceSubmitBtn.disabled = true;
  let result;
  try {
    result = await PracticeAPI.submitAnswer(q.question_id, userCode);
  } catch (err) {
    if (PracticeAPI.currentQuestion !== q) return;
    /* `blocked` = we declined to run it and already said why (torch on
       Pyodide); the reason IS the message, so don't bury it behind a prefix.
       The `|| err` fallback matters for the unblocked case: a Pyodide
       PythonError carries an EMPTY `.message`, which used to render as a bare
       "Submit failed:" with nothing after it. */
    outputArea.textContent = err.blocked
      ? err.message
      : "Submit failed: " + (err.message || err);
    practiceSubmitBtn.disabled = false;
    if (err.blocked) {
      // Not a transient failure: this question cannot run in this runtime at
      // all. Re-arming a countdown means expiry force-submits, the submit is
      // refused again, and the clock bounces back to 00:30 forever.
      PracticeSession.blockOnUnrunnableQuestion();
      window.PlacementTimer?.stop();
    } else {
      PracticeSession.resumeAnswerPhase();
      window.PlacementTimer?.resumeAfterFailedSubmit();
    }
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
  /* The answer, under the code that missed it. Ordered AFTER renderFailedTests
     because showSolution re-appends itself last, so the read is: your cells →
     which cases failed → what it should have been. Only on a miss: a correct
     answer already showed you a working one, yours — and a correct RESUBMIT
     has to take the old one away, or the answer to a question you have since
     solved sits under your working code until the next question loads. */
  if (result.correct) {
    window.DeltaNotebook?.clearSolution?.();
  } else if (window.DeltaNotebook?.showSolution?.(solCode)) {
    /* Appended is not seen: the answer lands under cells as tall as whatever
       the learner just wrote, i.e. below the fold of the notebook pane. Scroll
       it into view in the next frame — the cell was appended and auto-sized in
       this one, and a frame later is when the pane's own scrollHeight has
       caught up with it. */
    requestAnimationFrame(() => window.DeltaNotebook?.scrollToSolution?.());
  }
  /* What the grade said, kept for the rating step that follows it: that is
     where `pendingFeedback` is written, and a reload replays that and nothing
     else (ui.js `recordGradedDetail`). */
  recordGradedDetail(q.question_id, result.failed_tests, solCode);
  practiceProgress.lastResultCorrect = result.correct;
  practiceProgress.currentTargetDifficulty = getTargetDifficultyForQuestion(q);
  savePracticeProgress(practiceProgress);
  /* 🔴 THE POST-ATTEMPT READING IS PAINTED WHEN THE LEARNER RATES, NOT HERE.

     Seth, 2026-08-28, live: "I clicked the button to make it significantly
     harder and I didn't see the top bar show the update for the progress like
     it usually does with the animation ... obviously it should go up a certain
     amount after doing that." Measured in a real loop, sampling the topbar
     every 40ms: on a correct answer the concept pill read 33.33% before the
     click and 33.33% for the whole 2.2s after it. Zero pixels. The reason is
     right here — the reading had already been painted at SUBMIT, seconds
     earlier, while the learner was reading the verdict at the bottom of the
     screen, so by the time they answered the difficulty question there was
     nothing left for it to show.

     The rating is also where the reading becomes TRUE: the backend does not
     score the attempt at /submit at all — `feedback_router` calls
     `finalize_attempt`, which writes the history, the per-atom BKT and the new
     target difficulty. Painting at submit was painting an estimate the server
     had not committed to yet.

     Placement probes keep the submit-time paint, because they have no rating
     step to defer to: /submit records the probe outright and the branch below
     goes straight to Next. Same condition, spelled the same way, on purpose. */
  const _ratingStepFollows = !(q.diagnostic_active && practiceMode === "backend");
  if (result.ladder_estimate && window.StageLadder && !_ratingStepFollows) {
    window.StageLadder.setProgress(result.ladder_estimate);
  }
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
    // Tutor opens on the same signal as the explanation — both need a graded
    // attempt, and both are backed by the same ChatGPT key.
    if (window.PracticeTutor) {
      PracticeTutor.open({
        questionText: q.question_text,
        solutionCode: solCode,
        userCode,
        actualOutput,
        expectedOutput,
        wasCorrect: !!result.correct,
      });
    }
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
  // Through ui.js's helper, not a second copy of the wording here. The verdict
  // just flipped, so the question has to flip with it, and this path is exactly
  // the one that goes stale when the copy lives in two places.
  paintDifficultyQuestion(true);
  overrideRow.classList.add("hidden");
  if (typeof resetMissedFactRow === "function") resetMissedFactRow();
  practiceProgress.lastResultCorrect = true;
  savePracticeProgress(practiceProgress);
});

feedbackButtons.forEach((btn) => {
  btn.addEventListener("click", async () => {
    const feedback = btn.dataset.feedback;
    const q = PracticeAPI.currentQuestion;
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
    setConceptUnderstanding({
      mastery: response?.kc_mastery_after,
      coverage: response?.kc_coverage_after,
      tier: response?.kc_tier,
      title: q.ladder_kc_title,
    });
    if (response?.ladder_estimate && window.StageLadder) {
      window.StageLadder.setProgress(response.ladder_estimate);
    }

    // Emit competency bar update (single-KC maximize mode) — includes old/new mastery
    window.dispatchEvent(
      new CustomEvent("competency:feedback-update", {
        detail: { subtopic: q.subtopic, pBefore, pAfter },
      })
    );

    practiceProgress.pendingFeedback = {
      questionId: q.question_id,
      subtopic: q.subtopic,
      oldTarget,
      newTarget,
      correct: !!practiceProgress.lastResultCorrect,
      pBefore,
      pAfter,
      kcMastery: response?.kc_mastery_after,
      kcCoverage: response?.kc_coverage_after,
      kcTier: response?.kc_tier,
      kcTitle: q.ladder_kc_title,
      ladderEstimate: response?.ladder_estimate,
      /* Carried so a reload can rebuild the whole review, not just its verdict
         (ui.js `restoreGradedFeedbackInNotebook`). `gradedDetailFor` returns
         null for any other question, so a rating can never save the previous
         question's failed cases next to this one's verdict. */
      failedTests: gradedDetailFor(q.question_id)?.failedTests || [],
      solutionCode: gradedDetailFor(q.question_id)?.solutionCode || q.solution_code || "",
    };
    savePracticeProgress(practiceProgress);

    /* THE ANSWER IS THE NEXT BUTTON. Seth, 2026-08-28: "whenever you click
       it, it should automatically move on rather than making you click it and
       then press next". The three choices are now a question about the NEXT
       problem ("how much harder / easier do you want it to be?"), so pressing
       one and then being asked to press Next was asking the learner to
       confirm a decision they had already made.

       🔴 A SYNTHETIC CLICK ON THE REAL BUTTON, not a call to
       `_loadNextPracticeQuestion`. That handler is not just a loader: it gives
       `ArenaUnlock.tryShow` its chance to put the unlock interstitial up
       first, and calling the loader directly here would silently skip the
       interstitial for every learner who unlocked an ARENA exercise on the
       question they just rated. `timer.js::_forceAdvance` reaches the next
       question the same way, for the same reason.

       Fired AFTER `savePracticeProgress` so the rating is durable before
       anything can navigate: the load is async and a learner who closes the
       tab mid-flight must not lose the answer they gave. `showNextProblemButton()`
       above has already revealed the button this clicks.

       🔴 AND IT WAITS FOR THE TOPBAR. `StageLadder.setProgress` a few lines up
       is the one moment in the loop where the concept pill moves for what the
       learner just did, and the fill it drives is a 0.55s width transition
       (styles/concept-pill.css). Measured 2026-08-28 with the topbar sampled
       every 40ms: the next question rendered 43ms after this click and
       published its OWN reading at 35ms, so the earned value was on screen for
       about one frame and never animated at all — on a miss it was replaced by
       a lower reading and the pill drained 151px to 0 instead, which is the
       opposite of the feedback it was drawn to give.

       So the click that ends the question also gives the bar its half second.
       This is not a "feels nicer" pause: without it the update is not visible
       at any speed, which is exactly what Seth reported. It is one click still
       — the buttons are already disabled and the pressed one is highlighted,
       so the wait reads as the answer landing, not as a hang.

       🔴 GUARDED ON THE QUESTION, because 700ms is long enough for the learner
       to end the session, skip, or switch tabs. `timer.js::_forceAdvance` is
       untouched by this: its watchdog does not fire for a further 10s. */
    setTimeout(() => {
      if (PracticeAPI.currentQuestion !== q) return;
      nextProblemBtn.click();
    }, TOPBAR_SETTLE_MS);
  });
});

/* Per-problem content-quality feedback moved to practice/feedback-panel.js on
   2026-08-27. The chips used to POST on click, which made the note box under
   them a decoration — anything typed after the click was never sent, and in
   basic mode (the default) the whole row was `display:none`, so there was no
   way to report a problem at all. Selecting a kind, writing a note and sending
   are three separate acts now, and the panel owns all three. The clock hold
   below stays here: it belongs to the session, not to the panel. */

if (problemFeedbackNote) {
  problemFeedbackNote.addEventListener("focus", () => {
    PracticeSession.holdClock("problem-feedback-note");
  });
  problemFeedbackNote.addEventListener("blur", () => {
    PracticeSession.releaseClock("problem-feedback-note");
  });
}

/* Between questions. The panel owns its own widgets, so this asks it to clear
   rather than reaching into them — but it still falls back to clearing them
   directly, because a feedback panel that failed to load must not leave the
   previous problem's note attached to the next one. */
const _resetProblemFeedbackRow = () => {
  if (window.DDFeedbackPanel && typeof window.DDFeedbackPanel.resetProblem === "function") {
    window.DDFeedbackPanel.resetProblem();
    return;
  }
  problemFlagButtons.forEach((b) => b.classList.remove("flagged"));
  if (problemFeedbackNote) problemFeedbackNote.value = "";
  if (problemFeedbackStatus) {
    problemFeedbackStatus.textContent = "";
    problemFeedbackStatus.classList.add("hidden");
  }
};

/* WHAT TO DO WHEN THERE IS NO NEXT QUESTION. Two different failures wear the
   same exception, and treating them alike breaks one of them:

   THE CONTENT RAN OUT — the backend answering, correctly, that this concept has
   no unseen drill left at this rung (app/content_gaps.py; `err.contentExhausted`
   is set by practice/api.js when it parses that reply). There is nothing to
   retry: retrying returns the same sentence. So the block ends, the learner is
   handed back to the idle surface, and the reason is printed on
   `#session-summary` where the next screen can carry it.

   THE FETCH FAILED — a dropped connection, a redeploy, a 500. The content is
   fine and the answer is to try again in a moment.

   🔴 A TRANSIENT FAILURE MUST NOT END A LIVE BLOCK. `deadEnd` on a running
   session calls `finish()`, which clears `pausedState` AND the saved snapshot —
   so one bad request between two questions would throw away a block the learner
   could otherwise have carried on with. A live block keeps its state and gets
   Next back as the retry control instead; the clock is already stopped
   (`pauseForAdvance`) and starts again when a question renders.

   Everything else — no session, or a `?lesson=<kc>` KC drill, which is
   deliberately sessionless — goes to `deadEnd`, which restores the idle surface
   without a session to end and leaves a PAUSED block offerable. That matters
   most for the lesson drill: the loader has already cleared the editor and
   hidden Next by the time this runs, so "leave it as it is" is the frozen
   screen this whole function exists to stop. */
const _handleNoNextQuestion = (err) => {
  const message = (err && err.message)
    || "Could not load the next question. Try again in a moment.";
  outputArea.textContent = message;
  const exhausted = !!(err && err.contentExhausted);
  const liveBlock =
    typeof PracticeSession !== "undefined" && PracticeSession.isActive?.() === true;
  if (!exhausted && liveBlock && !document.body.classList.contains("lesson-mode")) {
    showNextProblemButton();
    return;
  }
  PracticeSession.deadEnd(message);
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
  // Third countdown, same reason: the placement clock must die on every
  // advance path, or "I don't know yet" at 00:01 expires onto the NEXT probe.
  window.PlacementTimer?.stop();
  practiceProgress.currentQuestion = null;
  practiceProgress.pendingFeedback = null;
  practiceProgress.currentTargetDifficulty = null;
  practiceProgress.lastResultCorrect = null;

  // Reset to pre-submit state (ready for next question)
  practiceSubmitArea.classList.remove("hidden");
  practiceFeedbackArea.classList.add("hidden");
  showFeedbackButtons();
  _resetProblemFeedbackRow();
  if (typeof hideFailedTests === "function") hideFailedTests();
  questionMetaTop.classList.add("hidden");

  // Reset code editor
  if (window.DeltaNotebook) window.DeltaNotebook.reset(DEFAULT_EDITOR_CODE);
  else codeEditor.value = DEFAULT_EDITOR_CODE;
  outputArea.textContent = "";

  /* Load next question.

     🔴 A FAILED LOAD IS HANDLED HERE, NOT THROWN. Every advance path funnels
     through this function and every one of them discards what it throws —
     `.catch(() => {})` in timer.js's three forced advances, and a bare `await`
     in the Next-button handler. So a throw from here was invisible, and by the
     time it happened this function had already reset the screen for a question
     that was never going to arrive: feedback area hidden, editor cleared, Next
     hidden. Seth, 2026-09-01: "instead of going to the next problem it just
     froze and didn't allow me to press the next button."

     The common cause is not a broken connection. It is the backend answering
     honestly that this concept has no unseen drill left at this rung — "you
     have finished every fill-in-the-blank problem for X, all 2 of them"
     (app/content_gaps.py, which also RECORDS the gap for the content
     pipeline). That is a content state, not an error state, and the learner
     needs the sentence and a live screen rather than a stuck one.

     🔴 IT IS SWALLOWED ON PURPOSE — `deadEnd` has already put the learner
     somewhere they can act. Re-throwing would hand the same error back to the
     six callers that ignore it, which is exactly how this froze. */
  let nextQ;
  try {
    nextQ = await PracticeAPI.getNextQuestion();
  } catch (err) {
    _handleNoNextQuestion(err);
    return;
  }
  const nextCount = practiceQuestionCount + 1;
  practiceProgress.questionCount = nextCount;
  savePracticeProgress(practiceProgress);
  // First-encounter gate: when this question's target KC has never been
  // taught, LessonGate shows the introducing lesson first and renders the
  // question (starting the session answer timer) only on Continue.
  if (window.LessonGate && (await window.LessonGate.maybeShow(nextQ, () => renderQuestion(nextQ, nextCount)))) {
    return;
  }
  // Ladder `worked` rung — runs only when the gate above did NOT fire. If it
  // did, the learner has just read this KP's example and LessonGate already
  // credited the rung, so re-showing it would teach the same page twice.
  if (
    window.LadderUI &&
    (await window.LadderUI.maybeShowWorked(nextQ, () => renderQuestion(nextQ, nextCount)))
  ) {
    return;
  }
  // Worked-example popup on the drill rungs, when the server scheduled one
  // (question.ladder_example — app/example_schedule.py). After the two gates
  // above so a lesson is never followed by its own example on the same card.
  if (
    window.ExampleGate &&
    (await window.ExampleGate.maybeShow(nextQ, () => renderQuestion(nextQ, nextCount)))
  ) {
    return;
  }
  renderQuestion(nextQ, nextCount);
};

// --- Torch self-rating (P0.1): torch drills run in Colab, not in-app. Record a
// local-eval attempt (no doomed server grading) when in backend mode, then
// advance. In guest/local mode it goes into the Pyodide engine instead — graded
// AND counted in one call, because nothing asks for felt difficulty afterwards
// on this route and an uncounted attempt is one the learner never sees again.
//
// On the Colab edition the two verdict buttons ARE the submit, so rating stops
// on a review step instead of jumping to the next problem: the reference answer
// opens underneath, and Next problem is a second, deliberate click. Everywhere
// else the notice is an aside on a page that has its own submit, and pausing
// there would strand the learner behind a solution they did not ask for.
const _colabReviewMode = () =>
  !!(window.DDColab && window.DDColab.active()) &&
  !document.documentElement.classList.contains("dd-no-notebook");

/**
 * The difficulty ladder, drawn on the rail after a verdict.
 *
 * This is the WITHIN-STAGE half of what the learner is climbing: the concept
 * strip's estimate bar is mastery and its mark is the promotion to the next
 * scaffold rung, while this bar is how hard the questions get inside the rung
 * they are on. Answer well and the band runs green to the right; miss and it
 * runs red back to the left. Titling it says which is which — two bars in a
 * 400px column that both fill up need to admit they are not the same quantity.
 *
 * `record` is what `recordLocalEval` reported, not a re-read of global state:
 * the state moved during the await, and re-reading it would only tell us where
 * it ended up, never where it started.
 */
const _drawColabDifficultyStep = (q, record) => {
  const oldTarget = record ? record.targetBefore : null;
  const newTarget = record ? record.targetAfter : null;

  // Nothing to animate. Do not fall back to the question's own rating and call
  // it the old target — that number is a property of the problem, so a correct
  // answer on an easy one would draw a red band for a step that went UP.
  //
  // Three ways to land here and they are not the same news, so they do not get
  // the same sentence: a placement probe is locating the learner instead of
  // stepping the staircase (nothing finalized AND no earlier target on file),
  // this is the first answer in the concept and there is no earlier target it
  // could have moved from, or the recording never came back at all. The last is
  // the default because it is the only one that admits the attempt might not be
  // in — better to under-claim than to explain a step nobody took.
  if (!Number.isFinite(oldTarget) || !Number.isFinite(newTarget)) {
    let note = "Couldn't read the ladder for this one.";
    if (record && record.finalized === false && !Number.isFinite(oldTarget)) {
      note = "Placement in progress — these answers find your level rather than stepping the ladder.";
    } else if (record && record.finalized && Number.isFinite(newTarget)) {
      note = "First answer here — the ladder starts stepping from the next one.";
    }
    setTargetDifficultyUnavailable(note, newTarget);
    return;
  }

  animateTargetDifficulty(oldTarget, newTarget, () => {
    // The tween runs for most of a second, and Next problem is already on
    // screen. Landing the final frame on a question that has moved on would
    // pin this problem's step to the next problem's card.
    if (PracticeAPI.currentQuestion !== q) return;
    setTargetDifficultyFinal(oldTarget, newTarget);
  });

  // Nothing else to move: the strip carries no difficulty of its own now, and
  // the tick stays where it is because this problem's rating is a property of
  // the problem rather than of how the answer went.
};

const _rateTorchAndAdvance = async (correct) => {
  const q = PracticeAPI.currentQuestion;
  if (!q) return;
  if (torchRateSolved) torchRateSolved.disabled = true;
  if (torchRateLookedUp) torchRateLookedUp.disabled = true;
  let record = null;
  // On the Colab edition the verdict is followed by the felt-difficulty step,
  // so the attempt has to stay PENDING for that rating to land on — finalizing
  // it here would consume it and the rating would 400 with nothing to apply it
  // to. Everywhere else this verdict is the whole submit and closes out unrated.
  const wantsRating = _colabReviewMode();
  try {
    if (typeof PracticeAPI.recordLocalEval === "function") {
      record = await PracticeAPI.recordLocalEval(q.question_id, correct, {
        finalize: !wantsRating,
      });
    }
  } catch (_) {
    /* best-effort — still advance so the learner isn't stuck */
  } finally {
    if (torchRateSolved) torchRateSolved.disabled = false;
  if (torchRateLookedUp) torchRateLookedUp.disabled = false;
  }

  if (record?.ladderEstimate && PracticeAPI.currentQuestion === q && window.StageLadder) {
    window.StageLadder.setProgress(record.ladderEstimate);
  }

  if (_colabReviewMode()) {
    // The question can change under a slow recordLocalEval (End session, a
    // stray Skip) — repainting then would attach this review to the wrong
    // problem's solution.
    if (PracticeAPI.currentQuestion !== q) return;
    // ...and in the notebook beside the rail, where the learner's own code is.
    // Hidden there until exactly this click; no-op outside the extension.
    if (window.DDColab && typeof window.DDColab.revealSolution === "function") {
      window.DDColab.revealSolution(q.question_id);
    }
    solutionCode.textContent = q.solution_code || "";
    practiceSubmitArea.classList.add("hidden");
    practiceFeedbackArea.classList.remove("hidden");
    applyResult(correct);
    // The difficulty step, back on this route. The verdict says WHETHER it
    // worked; this says how big a step to take from here, which is the one
    // thing no grade can tell us and the thing that decides where the next
    // problem lands (adaptive.nudge_difficulty_offset). applyResult has already
    // set the three labels for this outcome — "Significantly harder" after a
    // correct answer, "Significantly easier" after a miss.
    //
    // 🔴 The learner answers this one themselves now, on every surface. Basic
    // mode used to hide the buttons and click the default here; that stand-in
    // is gone (practice/basic-mode.js).
    //
    // Only when there is an attempt parked for the rating to apply to. During a
    // placement diagnostic nothing is pending, and a null `pending` is an older
    // backend that was never asked — both fall through to the plain review.
    if (record && record.pending === true) {
      feedbackPrompt.textContent = correct
        ? "Recorded as correct. How much harder do you want the next problem to be?"
        : "Recorded as a miss. How much easier do you want the next problem to be?";
      showFeedbackButtons();
    } else {
      feedbackPrompt.textContent = correct
        ? "Recorded as correct. The reference answer is below — worth a look even when you got it."
        : "Recorded as a miss. The reference answer is below; compare it with what you ran.";
      showNextProblemButton();
      // The one thing the verdict changed that the rail could not otherwise
      // show: where the next question will be pitched, and which way this
      // answer moved it. When a rating follows, the rating draws this instead —
      // the step has not happened yet.
      _drawColabDifficultyStep(q, record);
    }
    // A second verdict would log a second attempt against the same problem.
    if (torchRateSolved) torchRateSolved.disabled = true;
    if (torchRateLookedUp) torchRateLookedUp.disabled = true;
    return;
  }

  await _loadNextPracticeQuestion();
};
if (torchRateSolved) torchRateSolved.addEventListener("click", () => _rateTorchAndAdvance(true));
if (torchRateLookedUp) torchRateLookedUp.addEventListener("click", () => _rateTorchAndAdvance(false));

// The notebook grading itself. `dd_check` prints its verdict, the extension
// reads it off the Colab page and colab_mode.js re-publishes it here — so
// running the check IS the submit, and the learner does not hand-copy a result
// they just measured. The buttons stay for the questions with no checker and
// for disagreeing with one.
window.addEventListener("dd-check-result", (event) => {
  const detail = (event && event.detail) || {};
  const q = PracticeAPI.currentQuestion;
  if (!q || String(q.question_id) !== String(detail.problem)) return;
  // Already recorded — the verdict path disables both buttons for the rest of
  // the problem, so this is also the guard against a re-run logging twice.
  if (torchRateSolved && torchRateSolved.disabled) return;
  _rateTorchAndAdvance(!!detail.correct);
});

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
