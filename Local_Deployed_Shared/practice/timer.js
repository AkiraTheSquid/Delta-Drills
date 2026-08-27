/* ================================================================
   PRACTICE SESSION — a fixed clock per QUESTION, paused and resumed

   🔴 THE LEARNER SETS NOTHING. Seth, 2026-08-23: "it's the timer per
   question, not per session ... a certain amount of time to answer the
   question and a certain amount of time to review it ... it's a
   predetermined timer that they don't control". Both allowances are the
   constants below, they apply to every question alike, and the three
   inputs that used to set them (questions / answer time / review time)
   are gone from index.html along with the panel that held them.

   A block has no LENGTH either, which is why `finish("ended")` and
   #session-end-btn went with them: there is no quota to reach and
   nothing to end early. Pause and resume are the only two states. Pause
   freezes the current question — draft code, review state, clock — and
   puts the readiness screen back; Continue practicing brings it back.
   Closing or reloading the page leaves the same resumable snapshot.

   A resumed clock depends on the length of the break — see RESUME_GRACE_SECS.
   Straight back and it picks up mid-second; after a real gap the current step
   starts over, and only that step.

   Lifecycle hooks (called from ui.js / events.js):
     PracticeSession.onQuestionRendered()   — every renderQuestion()
     PracticeSession.pauseForGrading()      — submit clicked, grading in flight
     PracticeSession.recordReviewResult()   — preserve grade/review UI for resume
     PracticeSession.resumeAnswerPhase()    — submit failed, back to answering
     PracticeSession.beginReviewPhase()     — grade landed, review starts
     PracticeSession.shouldFinishInsteadOfAdvance() — quota check before
       _loadNextPracticeQuestion() fetches another question
   ================================================================ */

/* 🔴 THE TWO ALLOWANCES. Per question, not per session, and not editable
   from anywhere in the UI — changing the model means changing these.
   02:00 each: long enough to read a prompt and write a few lines, short
   enough that the pair fits in the four minutes a question is worth. */
const ANSWER_SECS = 120;
const REVIEW_SECS = 120;

/* Bumped 1 → 2. A v1 snapshot carries the learner's OWN answerSecs and
   reviewSecs — 05:00 was the old default — and resuming one would hand back a
   clock this build has no way to set. A v2 snapshot stores neither field and
   reads both from the constants above, so the version bump is what stops an
   old block resuming under the old rules. `_readSaved` drops v1 outright,
   which costs one paused question the day this ships. */
const SESSION_STATE_VERSION = 2;

/* How long a paused clock stays paused before the step starts over.

   Leaving and coming straight back is not a break — a reload, a tab closed by
   accident, a laptop lid — and handing back a fresh five minutes for it would
   make "pause" the way to opt out of the timer entirely. So inside the grace
   window the clock resumes exactly where it stopped: one minute left is one
   minute left.

   Coming back an hour later is a different thing, and resuming at 00:01 there
   punishes the break rather than timing the work. What the strict timer
   actually measures is a continuous attempt, and after a real gap the learner
   is starting the step again — re-reading the prompt, rebuilding what they had
   in their head — so the step gets its full time back. Only the CURRENT step:
   the question, the quota and the draft code are all still theirs. */
const RESUME_GRACE_SECS = 120;

/* `parseTimerInput` was DELETED here on 2026-08-23. It read "05:00" or "300"
   out of the three setup inputs; there are no inputs, and no other file called
   it. `formatDuration` ("1 h 10 min") went with it — it only ever wrote the
   setup panel's total-session estimate, and a session has no total. */

const formatTimer = (value) => {
  const clamped = Math.max(0, Math.min(3600, Math.round(value)));
  const m = Math.floor(clamped / 60);
  const s = clamped % 60;
  return String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
};

const PracticeSession = (() => {
  const pagePractice = document.getElementById("page-practice");

  let state = null; // { total, answerSecs, reviewSecs, served, phase, review }
  let pausedState = null;
  let interval = null;
  let remaining = 0;
  let advancePoll = null;
  let resumeRefresh = null;
  let resumeReady = false;
  let resumePending = false;
  const clockHolds = new Set();

  const isActive = () => !!state;

  const _storageKey = () => `${getPracticeStorageKey()}_session`;

  const _questionId = () => {
    const raw = PracticeAPI?.currentQuestion?.question_id ?? PracticeAPI?.currentQuestion?.id;
    return raw == null ? "" : String(raw);
  };

  const _stopTick = () => {
    if (interval) {
      clearInterval(interval);
      interval = null;
    }
  };

  const _stopPoll = () => {
    if (advancePoll) {
      clearInterval(advancePoll);
      advancePoll = null;
    }
  };

  const _readSaved = () => {
    try {
      const saved = JSON.parse(localStorage.getItem(_storageKey()) || "null");
      if (!saved || saved.version !== SESSION_STATE_VERSION) return null;
      if (!Number.isFinite(saved.served) || !saved.questionId) return null;
      const phase = saved.phase === "review" && saved.review ? "review" : "answer";
      /* The allowance is read from the CONSTANT, never from the snapshot. A
         saved clock that outlived a change to ANSWER_SECS/REVIEW_SECS must not
         resume under the old rule, and clamping to the constant is also what
         stops a hand-edited localStorage entry buying unlimited time. */
      const phaseLimit = phase === "review" ? REVIEW_SECS : ANSWER_SECS;
      const savedRemaining = Number.isFinite(saved.remaining) ? saved.remaining : phaseLimit;
      return {
        version: SESSION_STATE_VERSION,
        served: Math.max(1, Math.round(saved.served)),
        phase,
        remaining: Math.max(1, Math.min(phaseLimit, Math.round(savedRemaining || 30))),
        questionId: String(saved.questionId),
        draft: typeof saved.draft === "string" || (
          saved.draft?.version === 1 && Array.isArray(saved.draft.cells)
        ) ? saved.draft : "",
        review: phase === "review" ? saved.review : null,
        savedAt: saved.savedAt || null,
      };
    } catch (_) {
      return null;
    }
  };

  const _clearSaved = () => {
    try {
      localStorage.removeItem(_storageKey());
    } catch (_) {}
  };

  const _draft = () => window.DeltaNotebook?.serialize() || codeEditor.value;
  const _restoreDraft = (draft) => {
    if (!draft) return;
    if (window.DeltaNotebook) window.DeltaNotebook.restore(draft);
    else if (typeof draft === "string") codeEditor.value = draft;
  };

  const _snapshot = () => {
    if (!state) return null;
    const review = state.review ? { ...state.review } : null;
    if (review && state.phase === "review") {
      review.feedbackComplete = !nextProblemBtn.classList.contains("hidden");
    }
    return {
      version: SESSION_STATE_VERSION,
      // No `total`, no `answerSecs`, no `reviewSecs`: a block has no length and
      // the two allowances are constants. Writing them would invite a reader
      // to resume from them.
      served: state.served,
      phase: state.phase,
      remaining,
      questionId: _questionId(),
      draft: _draft(),
      review,
      savedAt: new Date().toISOString(),
    };
  };

  const _writeSaved = (snapshot) => {
    if (!snapshot || !snapshot.questionId) return;
    try {
      localStorage.setItem(_storageKey(), JSON.stringify(snapshot));
    } catch (_) {}
  };

  const _persist = () => _writeSaved(_snapshot());

  const _updateCountdown = () => {
    sessionCountdown.textContent = formatTimer(remaining);
    sessionCountdown.classList.toggle("session-countdown--low", remaining <= 30);
  };

  const _setPhase = (phase, label) => {
    state.phase = phase;
    sessionPhaseLabel.textContent = label;
    sessionStatusRow.classList.toggle("session-status--review", phase === "review");
    // "blocked" counts as stable: the question cannot be graded here, so
    // Pause & exit is the sane way out and must not be greyed with it.
    const stable = phase === "answer" || phase === "review" || phase === "blocked";
    sessionPauseBtn.disabled = !stable;
    sessionPauseBtn.title = stable
      ? "Pause and save. You come back to this question, on this clock."
      : "Pause becomes available when this short step finishes.";
  };

  const _tick = (onExpire) => {
    _stopTick();
    _updateCountdown();
    _persist();
    if (clockHolds.size) return;
    interval = setInterval(() => {
      remaining--;
      _updateCountdown();
      _persist();
      if (remaining <= 0) {
        _stopTick();
        onExpire();
      }
    }, 1000);
  };

  /* Seconds since the snapshot was written. `_persist` stamps `savedAt` every
     tick, so this is the length of the break to within a second — except when
     the field is missing or unreadable (a snapshot from an older bundle, a
     mangled localStorage entry), where the honest answer is "no idea how long"
     and the safe one is to treat it as a long break. Erring that way costs a
     restarted step; erring the other way hands out free time on every reload.
     A clock that has gone BACKWARDS reads as 0, which resumes the timer. */
  const _awaySecs = (saved) => {
    const at = Date.parse(saved?.savedAt || "");
    if (!Number.isFinite(at)) return Infinity;
    return Math.max(0, (Date.now() - at) / 1000);
  };

  const _phaseLimit = (saved) =>
    saved.phase === "review" ? REVIEW_SECS : ANSWER_SECS;

  /* What the clock should read on resume: {secs, restarted}.

     Recomputed at the moment of resuming rather than when the snapshot was
     read, because the resume panel can sit on screen for as long as the
     learner likes and the break is still running while it does. */
  const _effectiveRemaining = (saved) => {
    if (_awaySecs(saved) <= RESUME_GRACE_SECS) {
      return { secs: saved.remaining, restarted: false };
    }
    return { secs: _phaseLimit(saved), restarted: true };
  };

  const _resumeSummary = (saved) => {
    const phase = saved.phase === "review" ? "reviewing" : "answering";
    const { secs, restarted } = _effectiveRemaining(saved);
    return `Question ${saved.served} · ${phase} · ` +
      formatTimer(secs) + (restarted ? " (this step starts over)" : " left");
  };

  /* The summary is a live number, so it has to be redrawn while the panel sits
     there: a learner reading "00:47 left" who then makes a cup of tea must not
     click Resume on a promise that expired while they were gone. Cheap, and it
     stops the moment there is nothing paused. */
  const _stopResumeRefresh = () => {
    if (resumeRefresh) {
      clearInterval(resumeRefresh);
      resumeRefresh = null;
    }
  };

  const _showResumeOption = () => {
    if (!pausedState) {
      _stopResumeRefresh();
      sessionResumePanel.classList.add("hidden");
      sessionSetupPanel.classList.remove("session-setup--has-resume");
      return;
    }
    sessionResumeSummary.textContent = _resumeSummary(pausedState);
    sessionResumeBtn.disabled = !resumeReady;
    sessionResumePanel.classList.remove("hidden");
    sessionSetupPanel.classList.add("session-setup--has-resume");
    if (!resumeRefresh) {
      resumeRefresh = setInterval(() => {
        if (!pausedState) {
          _stopResumeRefresh();
          return;
        }
        sessionResumeSummary.textContent = _resumeSummary(pausedState);
      }, 5000);
    }
  };

  // Answer time is up. Grade whatever is in the editor; when nothing is
  // submittable (torch Colab routing swaps the submit area out), advance
  // without recording anything — same contract as Skip.
  const _forceSubmitOrAdvance = () => {
    if (!isActive()) return;
    if (!practiceSubmitArea.classList.contains("hidden") && !practiceSubmitBtn.disabled) {
      practiceSubmitBtn.click();
      return;
    }
    _loadNextPracticeQuestion().catch(() => {});
  };

  // Review time is up. If the felt-difficulty rating was never given, record
  // the default ("About right") so the mastery update still lands, then click
  // Next as soon as it appears (the rating POST is async).
  const _forceAdvance = () => {
    if (!isActive()) return;
    _stopPoll();
    if (!nextProblemBtn.classList.contains("hidden")) {
      nextProblemBtn.click();
      return;
    }
    const defBtn = document.querySelector(".feedback-btn--default");
    if (defBtn && !defBtn.classList.contains("hidden") && !defBtn.disabled) {
      defBtn.click();
    }
    let tries = 0;
    advancePoll = setInterval(() => {
      tries++;
      if (!nextProblemBtn.classList.contains("hidden")) {
        _stopPoll();
        nextProblemBtn.click();
      } else if (tries >= 40) {
        // 10s — the rating POST failed or never ran; advance anyway.
        _stopPoll();
        _loadNextPracticeQuestion().catch(() => {});
      }
    }, 250);
  };

  // Detailed content feedback is outside timed problem solving. Holding this
  // clock preserves remaining review time; releasing resumes same interval.
  const holdClock = (reason = "feedback") => {
    if (!isActive() || state.phase !== "review") return;
    clockHolds.add(reason);
    _stopTick();
    sessionPhaseLabel.textContent = "Reviewing · feedback paused";
    _persist();
  };

  const releaseClock = (reason = "feedback") => {
    clockHolds.delete(reason);
    if (clockHolds.size || !isActive() || state.phase !== "review") return;
    sessionPhaseLabel.textContent = "Reviewing";
    _tick(_forceAdvance);
  };

  const _restoreReview = () => {
    const review = state.review;
    if (!review) return;
    _restoreDraft(state.draft || review.userCode);
    solutionCode.textContent = review.solutionCode || PracticeAPI.currentQuestion?.solution_code || "";
    practiceSubmitArea.classList.add("hidden");
    practiceFeedbackArea.classList.remove("hidden");
    applyResult(!!review.correct);
    /* Both halves of the review — which cases failed and what the answer was —
       go back into the NOTEBOOK, under the restored draft, exactly where the
       live submit put them.

       🔴 AFTER `_restoreDraft` ABOVE, never before it: restoring the draft runs
       `DeltaNotebook.reset`, which begins by clearing the solution cell, so a
       cell added first is swept away by the code that puts the learner's own
       cells back. And `applyResult` on its own re-opens the left rail's copy
       (basic-mode.css keys it off `.result-incorrect`), which is why resuming
       used to move the answer back below the question. */
    if (typeof restoreGradedFeedbackInNotebook === "function") {
      restoreGradedFeedbackInNotebook(
        {
          correct: !!review.correct,
          failedTests: review.result?.failed_tests,
          solutionCode: review.solutionCode,
        },
        PracticeAPI.currentQuestion,
      );
    } else if (typeof renderFailedTests === "function") {
      renderFailedTests(review.result || { correct: !!review.correct }, PracticeAPI.currentQuestion);
    }
    const feedbackSaved =
      review.feedbackComplete ||
      practiceProgress.pendingFeedback?.questionId === PracticeAPI.currentQuestion?.question_id ||
      PracticeAPI.currentQuestion?.diagnostic_active;
    if (feedbackSaved) showNextProblemButton();
  };

  const start = () => {
    _clearSaved();
    pausedState = null;
    resumeReady = false;
    _showResumeOption();
    /* No `total`. `shouldFinishInsteadOfAdvance` returns false forever now, so
       `served` is a counter and not a quota — it is what the progress readout
       and the resume summary say, and nothing acts on it. */
    state = { served: 0, phase: null, review: null };
    sessionSummary.classList.add("hidden");
    sessionProgressLabel.textContent = "0";
    sessionStatusRow.classList.remove("hidden");
    pagePractice.classList.remove("session-idle");
    sessionStartBtn.disabled = true;
    // Always begin on a FRESH question — nothing about the one rendered in
    // the background at init is recorded (same contract as Skip).
    _loadNextPracticeQuestion()
      .catch((err) => {
        outputArea.textContent = "Could not start the session: " + (err?.message || err);
        finish("error");
      })
      .finally(() => {
        sessionStartBtn.disabled = false;
      });
  };

  /* A placement probe is timed by the PLACEMENT's rule, not the learner's
     session settings. Starting the placement ends the running session, but a
     learner can start a fresh session while a placement is still open, and
     then the probes were inheriting whatever answer time that session was set
     to — 5:00 for one probe and 2:00 for the next is exactly the comparison
     the fixed allowance exists to prevent. */
  const _probeOnScreen = () => {
    const api = typeof PracticeAPI !== "undefined" ? PracticeAPI : window.PracticeAPI;
    return !!api?.currentQuestion?.diagnostic_active;
  };

  const _answerSecsFor = () =>
    _probeOnScreen() && window.PlacementTimer
      ? window.PlacementTimer.secondsPerQuestion()
      : ANSWER_SECS;

  const onQuestionRendered = () => {
    if (!isActive()) {
      if (pausedState) {
        // Resume no longer depends on the saved question happening to be the
        // one on screen — `_restoreSavedQuestion` puts it back from the bank.
        // Tying the button to a coincidence of rendering is what made it dead:
        // any tab switch, reload, or background fetch swapped the question out
        // and the button greyed itself with "no longer available" while the
        // session was perfectly resumable.
        resumeReady = true;
        sessionResumeSummary.textContent = _resumeSummary(pausedState);
        sessionResumeBtn.disabled = false;
      }
      return;
    }
    _stopPoll();
    state.served += 1;
    state.review = null;
    sessionProgressLabel.textContent = String(state.served);
    _setPhase("answer", "Answering");
    remaining = _answerSecsFor();
    _tick(_forceSubmitOrAdvance);
  };

  const pauseForGrading = () => {
    if (!isActive()) return;
    _stopTick();
    _setPhase("grading", "Grading…");
    _persist();
  };

  // Every advance path funnels through _loadNextPracticeQuestion; both
  // countdowns must die there, or a Skip clicked near 00:00 leaves the old
  // answer timer running and its expiry force-submits the skipped question.
  const pauseForAdvance = () => {
    if (!isActive()) return;
    clockHolds.clear();
    _stopTick();
    _stopPoll();
    _setPhase("loading", "Loading…");
    _persist();
  };

  const recordReviewResult = (review) => {
    if (!isActive() || state.phase !== "grading") return;
    state.review = review;
    _persist();
  };

  const resumeAnswerPhase = () => {
    if (!isActive() || state.phase !== "grading") return;
    _setPhase("answer", "Answering");
    // A failed submit at 00:00 must not retry-loop forever; grant a short
    // grace window instead of skipping the learner's work.
    remaining = Math.max(remaining, 30);
    _tick(_forceSubmitOrAdvance);
  };

  /* The submit could never have run here (torch on Pyodide, and anything else
     that reports `blocked`). Re-arming the answer clock for that is a loop the
     learner cannot break: expiry force-submits, the submit is refused for the
     same reason it was refused a moment ago, and the countdown pops back to
     00:30 forever — which is exactly what Seth saw. Stop the clock and say so;
     Skip / "I don't know yet" / the Colab link are the ways out, and none of
     them are on a timer. */
  const blockOnUnrunnableQuestion = () => {
    if (!isActive()) return;
    _stopTick();
    _stopPoll();
    _setPhase("blocked", "Can't be run here");
    _persist();
  };

  const beginReviewPhase = () => {
    // Only a grade we are actually waiting on may start review — a stale
    // response landing after End session → Start session must not hijack the
    // new session's first question.
    if (!isActive() || state.phase !== "grading") return;
    _setPhase("review", "Reviewing");
    remaining = REVIEW_SECS;
    _tick(_forceAdvance);
  };

  const pause = () => {
    if (!isActive() || !["answer", "review"].includes(state.phase)) return;
    _stopTick();
    _stopPoll();
    clockHolds.clear();
    pausedState = _snapshot();
    _writeSaved(pausedState);
    state = null;
    resumeReady = true;
    sessionStatusRow.classList.add("hidden");
    pagePractice.classList.add("session-idle");
    sessionSummary.textContent = "Paused. Your question, code, clock and review state are saved.";
    sessionSummary.classList.remove("hidden");
    _showResumeOption();
  };

  /* Put the saved question back on screen. The paused snapshot stores only the
     id, but the static question bank is complete in BOTH modes (backend mode
     still ships questions.json for offline grading), so the question can always
     be rebuilt from it — no server round-trip and no dependence on whatever the
     queue happens to be holding.

     Returns false only when the id genuinely is not in the bank any more, which
     is the one case where "saved question is no longer available" is true. */
  const _restoreSavedQuestion = async () => {
    if (!pausedState) return false;
    if (_questionId() === pausedState.questionId) return true;
    try {
      if (typeof loadQuestionsBank === "function") await loadQuestionsBank();
      const bankQ =
        typeof getQuestionFromBank === "function"
          ? getQuestionFromBank(Number(pausedState.questionId))
          : null;
      if (!bankQ) return false;
      /* 🔴 THE LADDER FIELDS ARE NOT IN THE BANK, so rebuilding from it alone
         LOSES them. `buildPracticeQuestionFromBank` maps a bank record to the
         render shape and the bank has no `ladder_kc` / `ladder_stage` /
         `ladder_kc_title` — those come from the backend queue, per served
         question. Rebuilding from the bank therefore handed `renderQuestion` a
         question with no concept on it, and `LadderUI.decorate` reads exactly
         those two fields: no kc, no stage, so `StageLadder.hide()`. Resuming a
         paused session took the concept off the screen — the heading fell back
         to the subtopic, the ladder card went, and the topbar's concept pill
         went with it — and it only came back at the NEXT question, which is
         served by the queue and carries the fields again. Seth, 2026-08-27:
         "once I pressed the button to continue practice, the top bar
         completely disappears ... only appears again after going to the next
         problem."

         The saved question is the one place those fields still exist: `api.js`
         writes the whole served question into `practiceProgress.currentQuestion`
         and `storage.js` persists it, so it survives a reload the same way the
         snapshot does. `hydrateSavedPracticeQuestionFromBank` is the function
         built for this exact pair — it spreads the saved question FIRST and
         then overwrites every artifact field from the bank, so the bank stays
         authoritative for the question itself (a re-authored prompt still
         wins) and only the fields the bank has no opinion about survive.

         Falls back to the plain build when the saved question is missing or is
         a different question, which is the behaviour this had before. A resume
         with no ladder context is worse than one with it; it is not broken. */
      const saved = practiceProgress.currentQuestion;
      const canHydrate =
        typeof hydrateSavedPracticeQuestionFromBank === "function" &&
        saved &&
        String(saved.question_id ?? "") === String(pausedState.questionId);
      const restored = canHydrate
        ? hydrateSavedPracticeQuestionFromBank(saved)
        : buildPracticeQuestionFromBank(bankQ);
      PracticeAPI.currentQuestion = restored;
      practiceProgress.currentQuestion = restored;
      practiceProgress.currentQuestionId = restored.question_id;
      savePracticeProgress(practiceProgress);
      // Render before the lesson gate runs, so the gate sees the right KC.
      renderQuestion(restored, pausedState.served);
      return _questionId() === pausedState.questionId;
    } catch (err) {
      console.warn("[session] could not restore the saved question:", err);
      return false;
    }
  };

  const resume = async () => {
    if (resumePending || !pausedState) return;
    resumePending = true;
    if (!(await _restoreSavedQuestion())) {
      resumePending = false;
      resumeReady = false;
      sessionResumeSummary.textContent =
        "Saved question is no longer available. Discard this session and start a new one.";
      sessionResumeBtn.disabled = true;
      return;
    }
    // A reload during the lesson-gate overlay leaves the question resumable
    // with its KC still unexposed — re-show the lesson before the question
    // becomes visible. Already-exposed KCs (the normal case) never gate, and
    // review-phase resumes are post-answer so teaching first is moot.
    try {
      if (
        window.LessonGate &&
        pausedState.phase !== "review" &&
        (await window.LessonGate.maybeShow(PracticeAPI.currentQuestion, () => {
          resumePending = false;
          _resumeCore();
        }))
      ) {
        return;
      }
    } catch (err) {
      console.warn("[session] lesson gate failed during resume:", err);
    }
    resumePending = false;
    _resumeCore();
  };

  const _resumeCore = () => {
    if (!pausedState) return;
    resumePending = false;
    // Read the clock before `pausedState` is cleared below, and read it HERE
    // rather than at load: the break is still running while the resume panel
    // is on screen.
    const clock = _effectiveRemaining(pausedState);
    state = {
      served: pausedState.served,
      phase: pausedState.phase,
      review: pausedState.review,
      draft: pausedState.draft,
    };
    remaining = clock.secs;
    pausedState = null;
    resumeReady = false;
    _stopResumeRefresh();
    sessionResumePanel.classList.add("hidden");
    sessionSetupPanel.classList.remove("session-setup--has-resume");
    sessionSummary.classList.add("hidden");
    sessionProgressLabel.textContent = String(state.served);
    sessionStatusRow.classList.remove("hidden");
    pagePractice.classList.remove("session-idle");
    _restoreDraft(state.draft);
    if (state.phase === "review") {
      _restoreReview();
      _setPhase("review", "Reviewing");
      _tick(_forceAdvance);
    } else {
      _setPhase("answer", "Answering");
      _tick(_forceSubmitOrAdvance);
    }
  };

  const discard = () => {
    _clearSaved();
    pausedState = null;
    resumeReady = false;
    _showResumeOption();
    sessionSummary.textContent = "Saved session discarded. Set up a new block when you're ready.";
    sessionSummary.classList.remove("hidden");
  };

  /* 🔴 ALWAYS FALSE, and kept as a function on purpose. A block has no length
     any more (2026-08-23), so nothing ends a session but a pause — but this is
     the hook `_loadNextPracticeQuestion` asks before every fetch, and deleting
     it would mean editing every call site to stop asking. Restoring a quota is
     one line here; finding all the callers again is not. */
  const shouldFinishInsteadOfAdvance = () => false;

  const hasSavedQuestion = (questionId) =>
    !!pausedState && String(questionId ?? "") === pausedState.questionId;

  /* Reasons a block ends WITHOUT a pause. "ended" is gone with the button that
     sent it, and so is "complete": the quota it counted down to no longer
     exists. What is left is a failure to load and the placement taking over —
     both of which happen TO the learner, which is why each one says what
     happened rather than congratulating them. */
  const finish = (reason) => {
    if (!state) return;
    const { served } = state;
    // "Recorded answers are kept" is printed below, so make it true: an attempt
    // that was graded and never rated is still pending in the offline engine,
    // and would otherwise wait for the learner's next session to be counted.
    // Best-effort — a session ends whether or not the engine is up.
    if (typeof PracticeAPI.flushPendingAttempt === "function") {
      PracticeAPI.flushPendingAttempt().catch(() => {});
    }
    _stopTick();
    _stopPoll();
    clockHolds.clear();
    state = null;
    pausedState = null;
    resumeReady = false;
    _clearSaved();
    _showResumeOption();
    sessionStatusRow.classList.add("hidden");
    pagePractice.classList.add("session-idle");
    sessionSummary.textContent =
      reason === "error"
        ? "Could not load a question — check the connection and try again."
        : reason === "placement"
          ? "Placement test started — its questions are timed on their own clock, one at a time."
          : `Session stopped after ${served} question${served === 1 ? "" : "s"}. Recorded answers are kept.`;
    sessionSummary.classList.remove("hidden");
  };

  sessionStartBtn.addEventListener("click", start);
  sessionPauseBtn.addEventListener("click", pause);
  sessionResumeBtn.addEventListener("click", resume);
  sessionDiscardBtn.addEventListener("click", discard);
  (document.getElementById("practice-notebook") || codeEditor).addEventListener("input", () => {
    if (isActive()) _persist();
  });

  // A reload or browser close acts like a pause. Snapshots written during
  // transient grading/loading phases safely reopen in answer mode.
  window.addEventListener("pagehide", () => {
    if (!state) return;
    const snapshot = _snapshot();
    if (snapshot && !["answer", "review"].includes(snapshot.phase)) {
      snapshot.phase = "answer";
      snapshot.review = null;
      snapshot.remaining = Math.max(30, snapshot.remaining || 0);
    }
    _writeSaved(snapshot);
  });

  /* Nothing to prefill. `delta_drills_session_setup` — the localStorage key
     that carried the learner's last questions/answer-time/review-time — is not
     read or written anywhere any more; it is left on disk rather than migrated
     because deleting it buys nothing and a stale key is inert. */
  pausedState = _readSaved();
  // A restored session is resumable the moment it loads. It used to stay
  // disabled until some later render happened to put the saved question on
  // screen, which on a fresh page load never happens — the queue renders
  // whatever comes next, not what was paused.
  resumeReady = !!pausedState;
  _showResumeOption();

  return {
    isActive,
    /* What the notch shows when nothing is running: the allowance the NEXT
       question's answer phase will get. Exposed rather than duplicated so
       notch-menu.js never holds a second copy of the number — and returned
       already formatted, because that file is forbidden a clock of its own
       (practice/watch.py) and mm:ss is a clock's job. */
    idleClockText: () => formatTimer(ANSWER_SECS),
    answerSeconds: () => ANSWER_SECS,
    reviewSeconds: () => REVIEW_SECS,
    // True when a session was paused and is waiting to be resumed. switchTab
    // needs this to know the question on screen belongs to that session and
    // must not be replaced by a preference refresh.
    hasPausedSession: () => !!pausedState,
    start,
    pause,
    resume,
    discard,
    hasSavedQuestion,
    onQuestionRendered,
    blockOnUnrunnableQuestion,
    pauseForGrading,
    pauseForAdvance,
    recordReviewResult,
    resumeAnswerPhase,
    beginReviewPhase,
    holdClock,
    releaseClock,
    shouldFinishInsteadOfAdvance,
    finish,
  };
})();
