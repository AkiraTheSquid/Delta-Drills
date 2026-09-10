/* ================================================================
   PRACTICE SESSION — one clock per QUESTION, paused and resumed

   🔴 THE ALLOWANCE IS THE PROBLEM'S OWN, AS OF 2026-09-09, and it is NOT
   a constant in this file. Seth: "keyed to how much time you need for each
   of them rather than what you select at the beginning. it should be like
   the diagnostic for the timing of the questions ... using the same time."
   Every practice question arrives stamped with `secs_allowed` — its
   concept's cap in lessons/placement_time_caps.json, the same number a
   placement probe on that concept gets — and practice/session-clock.js
   reads it off the question. It is what EACH STEP gets, answering and
   reviewing alike, per question.

   That reverses the 2026-08-28 picker (one number chosen on the idle
   screen before the block, "No limit" among the options), which itself
   reversed 2026-08-23 ("a predetermined timer that they don't control").
   The picker existed because a first-encounter lesson is read on the
   answer clock; the table carries that now (a concept's minutes are what
   its drills need). What has stayed gone since 08-23: no question quota,
   no session length, no End session button.

   🔴 "No limit" IS STILL A REAL STATE, not a large number — `null` from
   an ARENA exercise session's own setup (`sessionConfig`), and then this
   file runs no interval, never expires and never force-submits.
   `remaining` is `null` for the whole step, so every piece of arithmetic
   on it below asks first. SessionClock itself never answers null: a
   question the server did not stamp gets the ceiling.

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

/* THE ALLOWANCE, per question and per step. `null` means no limit.

   🔴 READ THROUGH THE FUNCTION, NEVER CACHED IN A CONSTANT. It changes with
   every question served — it is the QUESTION's number — so a value captured
   at load is one concept's clock running over another concept's problem.
   Both steps get the same number; that is what "time per problem" means.

   The fallback is the server's ceiling (20:00), and it is only reachable if
   practice/session-clock.js failed to load: a page without the reader still
   times a question, on the longest clock the table can hand out. */
const FALLBACK_SECS = 1200;
const _clockPrefs = () => window.SessionClock || null;
/* 🔴 WAIT FOR THE TABLES BEFORE THE FIRST QUESTION IS ON SCREEN. session-clock.js
   resolves a question's allowance out of two JSON files it fetches at parse
   time; until they land it can only answer with the ceiling. Both session entry
   points below wait on this once, so answering, reviewing and the pause
   snapshot are guaranteed to be reading the same settled table rather than
   whichever one happened to have arrived. A failed fetch resolves too — the
   ceiling then stands everywhere, consistently. Codex, 2026-09-09. */
const _clockReady = () => Promise.resolve(_clockPrefs()?.ready);

/* AN EXERCISE SESSION BRINGS ITS OWN NUMBERS (2026-09-06). "Practice
   make_rays_1d" on an ARENA notebook page asks for answer time, review time
   and a question count BEFORE it starts (practice/exercise-session.js), and
   for that block those three override the idle picker. Seth: "its own time
   limit, that you can set manually for the amount of answer time and the
   amount of review time ... 3. how many problems". Absent (`null`), nothing
   changes: the two readers below fall through to the picker exactly as
   before.

   🔴 `"answer" in cfg`, NOT `cfg.answer ??`: a value of `null` is "No limit"
   for that step and has to win over the picker's number. The fields are
   named answer/review/quota — the pause snapshot may not carry a per-session
   `answerSecs:`/`reviewSecs:` pair (practice/watch_lessons.py, the v1 → v2
   rule), and this is a different thing: the learner's choice FOR THIS BLOCK,
   restored with it, not a copy of the picker. */
let sessionConfig = null; // { answer, review, quota, exercise: {…} } | null
const _configuredSecs = (field) =>
  sessionConfig && field in sessionConfig ? sessionConfig[field] : undefined;
const ANSWER_SECS = () => {
  const own = _configuredSecs("answer");
  if (own !== undefined) return own;
  const prefs = _clockPrefs();
  return prefs ? prefs.answerSecs() : FALLBACK_SECS;
};
const REVIEW_SECS = () => {
  const own = _configuredSecs("review");
  if (own !== undefined) return own;
  const prefs = _clockPrefs();
  return prefs ? prefs.reviewSecs() : FALLBACK_SECS;
};

/* THE SNAPSHOT LIVES IN practice/session-snapshot.js (split out 2026-09-09,
   Modulario LOC limit). `SESSION_STATE_VERSION`, `RESUME_GRACE_SECS`, the
   reader, the writer and the resume arithmetic all moved there verbatim; this
   file keeps only what a RUNNING question does. `_snap()` is the handle, and
   the version is aliased below because `_snapshot()` stamps it. */
/* 🔴 ONE EXPLICIT ASSERTION, NOT TWENTY OPTIONAL CHAINS. Every `_snap().x()`
   below is unconditional, so a load-order break (index.html dropping the
   script tag, a cache-buster left behind on a stale copy) has to fail HERE,
   naming the missing file, rather than as "cannot read write of undefined"
   from whichever call site happened to run first. Codex, 2026-09-09. */
const _snap = () => {
  const snap = window.SessionSnapshot;
  if (!snap) {
    throw new Error(
      "practice/timer.js: window.SessionSnapshot is missing — " +
        "practice/session-snapshot.js must load BEFORE practice/timer.js."
    );
  }
  return snap;
};
const SESSION_STATE_VERSION = window.SessionSnapshot?.VERSION ?? 2;
/* 🔴 THE ALLOWANCES GO THE OTHER WAY. session-snapshot.js is parsed FIRST and
   must not reach back into this file for them — `sessionConfig` (an exercise
   session's own numbers) and `window.SessionClock` are read here, so they are
   handed over as functions, once, before anything reads a snapshot. */
_snap().install({ answerSecs: ANSWER_SECS, reviewSecs: REVIEW_SECS });

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

/* What a clock READS, including when there is no clock. `null` is the untimed
   state and it has to draw as something — a blank where a countdown lives says
   "broken", and 00:00 says "your time is up", which is the opposite of what is
   true. The infinity sign is the one glyph that is honest at a glance and fits
   the same narrow slot in the topbar as mm:ss.

   🔴 EVERY CLOCK READOUT GOES THROUGH HERE. `formatTimer(null)` is "00:00" —
   `Math.round(null)` is 0 — so a single caller that skips this prints an
   expired countdown over a question that will never expire. */
const NO_LIMIT_TEXT = "∞";
const clockText = (secs) => (secs === null ? NO_LIMIT_TEXT : formatTimer(secs));

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

  const _draft = () => window.DeltaNotebook?.serialize() || codeEditor.value;
  const _restoreDraft = (draft) => {
    if (!draft) return;
    if (window.DeltaNotebook) window.DeltaNotebook.restore(draft);
    else if (typeof draft === "string") codeEditor.value = draft;
  };

  /* Snapshots written before `ladder` joined the paused-session record have
     no concept/rung to restore. The static bank supplies prompt/tests, but the
     server alone knows the learner's current rung and authored Faded starter.
     Re-stage this exact id read-only; fetching the queue would consume another
     question and reproduce the mismatch this recovery exists to prevent. */
  const _recoverQuestionContext = async (question) => {
    if (
      !question ||
      (question.ladder_kc && question.ladder_stage) ||
      practiceMode !== "backend" ||
      typeof apiFetch !== "function"
    ) return question;
    try {
      const qid = Number(question.question_id ?? question.id);
      if (!Number.isFinite(qid)) return question;
      const res = await apiFetch(
        `/api/practice/question-context?question_id=${encodeURIComponent(qid)}`,
      );
      if (!res.ok) return question;
      const context = await res.json();
      if (!context?.ladder_kc || !context?.ladder_stage) return question;
      question.ladder_kc = context.ladder_kc;
      question.ladder_stage = context.ladder_stage;
      question.ladder_kc_title = context.ladder_kc_title || context.ladder_kc;
      question.ladder_estimate = context.ladder_estimate || null;
      question.ladder_support = context.ladder_support !== false;
      question.ladder_integrated = !!context.ladder_integrated;
      if (context.starter_code) question.starter_code = context.starter_code;
    } catch (err) {
      console.warn("[session] could not recover saved question context:", err);
    }
    return question;
  };

  /* THE CONCEPT CONTEXT FOR THE QUESTION ON SCREEN, captured at pause.

     🔴 IT CANNOT BE READ BACK OFF `practiceProgress.currentQuestion`, and a fix
     that did exactly that shipped on 2026-08-27 and did not hold. `ladder_kc` /
     `ladder_stage` / `ladder_kc_title` exist only on a question the BACKEND
     QUEUE served — the static bank has no opinion about them — and
     `practiceProgress.currentQuestion` is NOT a record of what is on the
     screen: the queue overwrites it with whatever it serves next. Measured on
     prod, 2026-08-28: paused on question 484, the persisted currentQuestion was
     already 224, so the id check below failed, the restore fell back to
     `buildPracticeQuestionFromBank`, `LadderUI.decorate` found no kc, and
     `StageLadder.hide()` took the topbar pill out for the whole resumed
     question. Seth: "it's still not showing the top bar when I first jump in."

     The snapshot is the one record that is BY DEFINITION about the question on
     screen — `pause()` writes it from the live state — so the concept travels
     with it and no other writer can move it.

     🔑 WHY THIS ONLY EVER SHOWED UP AFTER A RELOAD. An in-memory resume never
     rebuilds the question at all: `_restoreSavedQuestion` returns early when
     the paused id is already on screen, ladder fields and all. Pause and
     Continue inside one page load therefore looked correct while the same two
     clicks either side of a reload did not — which is why the earlier fix
     verified green. Reload BETWEEN the pause and the resume or the path under
     test is not the one the learner takes. */
  const _ladderContext = () => {
    const q = typeof PracticeAPI !== "undefined" ? PracticeAPI.currentQuestion : null;
    if (!q || !q.ladder_kc || !q.ladder_stage) return null;
    return {
      kc: String(q.ladder_kc),
      stage: String(q.ladder_stage),
      title: q.ladder_kc_title ? String(q.ladder_kc_title) : null,
      estimate: q.ladder_estimate && typeof q.ladder_estimate === "object"
        ? q.ladder_estimate : null,
      support: q.ladder_support !== false,
      integrated: !!q.ladder_integrated,
    };
  };

  const _snapshot = () => {
    if (!state) return null;
    const review = state.review ? { ...state.review } : null;
    if (review && state.phase === "review") {
      review.feedbackComplete = !nextProblemBtn.classList.contains("hidden");
    }
    return {
      version: SESSION_STATE_VERSION,
      // No `total`, no `answerSecs`, no `reviewSecs`: a block has no length.
      // `secsAllowed` is not a per-session pair: it is the PAUSED QUESTION's
      // own clock (its concept's cap), copied off the question so a resume
      // rebuilt from the static bank — which carries no stamp — is timed the
      // way the question was served. Optional; see SESSION_STATE_VERSION.
      served: state.served,
      phase: state.phase,
      remaining,
      questionId: _questionId(),
      secsAllowed: window.SessionClock?.secsFor?.(PracticeAPI.currentQuestion) ?? null,
      /* 🔴 NOT COVERED BY `SESSION_STATE_VERSION`, deliberately. Bumping the
         version DISCARDS every paused session that is already on a learner's
         machine, and this field is optional in both directions: an older
         snapshot simply has no `ladder` and restores exactly as it did before.
         A silent data loss is not a fair price for a field that degrades. */
      attemptFirst: PracticeAPI.currentQuestion?.attempt_first === true,
      ladder: _ladderContext(),
      config: _configRecord(),
      draft: _draft(),
      review,
      savedAt: new Date().toISOString(),
    };
  };

  /* What a scoped block needs to come back as ITSELF: its three numbers, the
     exercise it was started from, and the ladder's unserved items (kc-
     practice.js serializes those; the queue is in-memory and a reload would
     otherwise resume onto the adaptive queue with the count still running). */
  const _configRecord = () => {
    if (!sessionConfig) return null;
    const ladder = window.KcPractice?.serialize?.() || null;
    return { ...sessionConfig, ladder };
  };

  const _persist = () => _snap().write(_snapshot());

  const _updateCountdown = () => {
    sessionCountdown.textContent = clockText(remaining);
    /* `null <= 30` is TRUE in JS. Without the explicit test an untimed
       question paints the last-30-seconds colour for its whole life — the one
       piece of urgency the learner turned off. */
    sessionCountdown.classList.toggle(
      "session-countdown--low",
      remaining !== null && remaining <= 30,
    );
  };

  const _setPhase = (phase, label) => {
    if (phase === "answer") window.AnswerHistory?.begin();
    else if (phase === "grading") window.AnswerHistory?.pause();
    else if (phase !== "review") window.AnswerHistory?.abandon();
    state.phase = phase;
    sessionPhaseLabel.textContent = label;
    /* 🔴 BETWEEN QUESTIONS THERE IS NO NUMBER TO SHOW, 2026-09-09. The clock
       is the QUESTION's now, and the next one's is its concept's — not known
       until the queue serves it. Leaving the last question's readout up meant
       a 20:00 einops problem handing over to a 05:00 Python one under a clock
       still reading 19:41, and on a fresh page the static 02:00 in the markup
       stood in for every allowance there is — which is exactly the flat clock
       this change removed, still on screen for the whole of the first
       question's lesson gate.

       Only `loading`. `grading` freezes a clock that is still this question's
       and resumes into review on the same number, so blanking it there would
       flicker `--:--` over a countdown that never stopped being true.

       The notch mirrors this element (practice/notch-menu.js observes the row
       for childList + characterData), so writing it here writes both — and
       `.session-status-row` is `display: none`, which makes the notch the only
       clock a learner actually sees. */
    if (phase === "loading") {
      sessionCountdown.textContent = "--:--";
      sessionCountdown.classList.remove("session-countdown--low");
    }
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
    /* 🔴 NO LIMIT MEANS NO INTERVAL. Not a very large `remaining`, not a
       countdown that is ignored at zero: the expiry callbacks are what
       force-submit an answer and force-advance a review, and the only way to
       be sure neither ever fires is for nothing to be counting. The phase, the
       pause button, the snapshot and the resume path are all unchanged — an
       untimed question is a normal question whose clock reads ∞. */
    if (remaining === null) return;
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

  const _resumeSummary = (saved) => {
    const phase = saved.phase === "review" ? "reviewing" : "answering";
    const { secs, restarted } = _snap().effectiveRemaining(saved);
    const head = `Question ${saved.served} · ${phase} · `;
    // "∞ left" is not a sentence. Untimed says what is true and stops there.
    if (secs === null) return head + "no time limit";
    return head + formatTimer(secs) +
      (restarted ? " (this step starts over)" : " left");
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

  // Review time is up. If the difficulty question was never answered, click
  // the first choice ("Slightly harder"/"Slightly easier") so the mastery
  // update still lands, then click Next as soon as it appears (the rating POST
  // is async).
  //
  // 🔴 That first choice stopped being a NEUTRAL answer on 2026-08-28. The
  // three buttons used to read "About right / A bit off / Way off" and the
  // default meant "stop correcting"; they now read "Slightly / Somewhat /
  // Significantly harder-or-easier" and the smallest one still asks for a
  // step (adaptive.DIFFICULTY_NUDGE). Timing out therefore nudges the aim by
  // the smallest amount the learner could have chosen, in the direction their
  // grade already implies, rather than by nothing. That is the closest thing
  // to silence the new question has — the alternative is finalizing UNRATED,
  // which is reserved for routes where nobody was asked at all, and the
  // learner WAS asked here; the buttons were on screen and docked to the
  // bottom of the viewport for the whole review clock.
  const _forceAdvance = () => {
    if (!isActive()) return;
    _stopPoll();
    if (!nextProblemBtn.classList.contains("hidden")) {
      nextProblemBtn.click();
      return;
    }
    const defBtn = document.querySelector(".feedback-btn--default");
    if (!defBtn || defBtn.classList.contains("hidden") || defBtn.disabled) {
      // Nothing to rate — no question was asked, so advance directly.
      _loadNextPracticeQuestion().catch(() => {});
      return;
    }
    defBtn.click();

    /* 🔴 THE RATING CLICK IS NOW THE NAVIGATION, SO THIS POLL MUST NOT CLICK
       NEXT (2026-08-28). It used to: it waited for `#next-problem-btn` to be
       revealed by the feedback handler and clicked it, because before the dock
       the handler only revealed the button and stopped. The handler clicks it
       itself now, in the same synchronous run as `showNextProblemButton()` —
       so a poll that also clicks it is a SECOND advance, and the window it
       fires in is wide: Next stays visible from the handler's click until the
       new question renders, which is a network fetch away, and this ticks
       every 250ms. The learner loses a question, ArenaUnlock is asked twice.

       What is left is a watchdog, and it deliberately does NOT click Next.
       Next being visible at 10s means the handler already ran and already
       navigated (it reveals and clicks with no yield in between) and the load
       is merely slow — clicking again would be the very double-advance this
       fixes. Next still HIDDEN at 10s means the rating POST failed: that path
       re-enables the buttons and leaves the button hidden, and nothing else
       will move the learner on. `onQuestionRendered` calls `_stopPoll`, so a
       normal advance disarms this before it ever fires. */
    let tries = 0;
    advancePoll = setInterval(() => {
      tries++;
      if (tries < 40) return;
      _stopPoll();
      if (nextProblemBtn.classList.contains("hidden")) {
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
    if (feedbackSaved) showNextProblemButton("You already rated this one. Press Next to carry on.");
  };

  const start = () => {
    _snap().clear();
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
    _clockReady()
      .then(_loadNextPracticeQuestion)
      .catch((err) => {
        outputArea.textContent = "Could not start the session: " + (err?.message || err);
        finish("error");
      })
      .finally(() => {
        sessionStartBtn.disabled = false;
      });
  };

  /* 🔴 A PLACEMENT PROBE IS TIMED BY THE PLACEMENT'S RULE, and this matters
     MORE now that the session clock is the learner's again (2026-08-28). The
     placement compares a learner against the bank's difficulty, so every probe
     has to get the same fixed 2:00 — the learner's own allowance, and above
     all "No limit", would make the test measure how long they chose to sit
     there. Starting the placement ends the running session, but a learner can
     start a fresh session while a placement is still open, and then the probes
     were inheriting whatever that session was set to. */
  const _probeOnScreen = () => {
    const api = typeof PracticeAPI !== "undefined" ? PracticeAPI : window.PracticeAPI;
    return !!api?.currentQuestion?.diagnostic_active;
  };

  const _answerSecsFor = () =>
    _probeOnScreen() && window.PlacementTimer
      ? window.PlacementTimer.secondsPerQuestion()
      : ANSWER_SECS();

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
    window.AnswerHistory?.settle();
    state.review = review;
    _persist();
    /* Every grade inside an exercise session goes back to the ladder
       (kc-practice.js::onResult): a planned block re-weighs its next choice,
       a plain scoped one pulls the prerequisites on a miss. Only ever about
       the ORDER of what is served; the grade above is already recorded. The
       snapshot is rewritten when it lands so a pause right after carries
       what was just decided. */
    if (sessionConfig?.exercise && review) {
      const q = PracticeAPI?.currentQuestion;
      Promise.resolve(window.KcPractice?.onResult?.(q?.ladder_kc, !!review.correct, q?.question_id))
        .then((note) => {
          _persist();
          if (note) window.ExerciseSession?.onNote?.(note);
        })
        .catch((err) => console.warn("[session] ladder result hook failed:", err));
    }
  };

  const resumeAnswerPhase = () => {
    if (!isActive() || state.phase !== "grading") return;
    _setPhase("answer", "Answering");
    // A failed submit at 00:00 must not retry-loop forever; grant a short
    // grace window instead of skipping the learner's work. Untimed stays
    // untimed: there was no 00:00 to fail at.
    if (remaining !== null) remaining = Math.max(remaining, 30);
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
    remaining = REVIEW_SECS();
    _tick(_forceAdvance);
  };

  const pause = () => {
    if (!isActive() || !["answer", "review"].includes(state.phase)) return;
    window.AnswerHistory?.abandon();
    _stopTick();
    _stopPoll();
    clockHolds.clear();
    pausedState = _snapshot();
    _snap().write(pausedState);
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
      /* Put the concept back. The hydrate above keeps these fields when the
         persisted served question really is this one; this is what makes the
         restore correct when it is NOT, which on prod is the common case — see
         `_ladderContext`. Only fills what is missing, so a genuine hydrate is
         never overwritten by an older snapshot's copy. */
      /* 🔴 FIELD BY FIELD, AND ONLY THE MISSING ONES. Gating the whole merge
         on `!restored.ladder_kc` — the first shape of this fix — leaves a
         PARTIAL hydrate broken: a saved question carrying `ladder_kc` but no
         `ladder_stage` skips the merge entirely, and `LadderUI.decorate` wants
         BOTH, so it hides the readout exactly as if there had been no concept
         at all. Codex, 2026-08-28.

         The kc match is what makes the field-by-field merge safe. Both records
         describe the same question — the hydrate is gated on the saved id
         being the paused id, and the snapshot is written from the question on
         screen — but if they disagree about the CONCEPT, the hydrated one is
         the newer statement and grafting a rung off the other would put one
         concept's name over another's progress. */
      /* The clock, the same way: the bank has no `secs_allowed`, the snapshot
         has the served question's. Only when the restore carries none — a
         genuine hydrate keeps the record the server stamped. */
      if (restored.secs_allowed == null && pausedState.secsAllowed) {
        restored.secs_allowed = pausedState.secsAllowed;
      }
      const ladder = pausedState.ladder;
      if (ladder && (!restored.ladder_kc || restored.ladder_kc === ladder.kc)) {
        if (!restored.ladder_kc) restored.ladder_kc = ladder.kc;
        if (!restored.ladder_stage) restored.ladder_stage = ladder.stage;
        if (!restored.ladder_kc_title) {
          restored.ladder_kc_title = ladder.title || ladder.kc;
        }
        if (restored.ladder_support === undefined) {
          restored.ladder_support = ladder.support;
        }
        if (restored.ladder_integrated === undefined) {
          restored.ladder_integrated = ladder.integrated;
        }
        if (!restored.ladder_estimate && ladder.estimate) {
          restored.ladder_estimate = ladder.estimate;
        }
      }
      if (pausedState.attemptFirst) restored.attempt_first = true;
      await _recoverQuestionContext(restored);
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
    await _clockReady();
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
    /* The block's own numbers and its ladder come back BEFORE the clock is
       read: `_effectiveRemaining` clamps to the allowance in force, and for a
       scoped block that is the one it was started with. */
    if (pausedState.config) {
      sessionConfig = _snap().readConfig(pausedState.config);
      if (sessionConfig?.ladder) {
        window.KcPractice?.restore?.(sessionConfig.ladder);
        delete sessionConfig.ladder;
      }
    } else {
      sessionConfig = null;
    }
    // Read the clock before `pausedState` is cleared below, and read it HERE
    // rather than at load: the break is still running while the resume panel
    // is on screen.
    const clock = _snap().effectiveRemaining(pausedState);
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
    window.ExerciseSession?.onResume?.(sessionConfig, state.served);
  };

  const discard = () => {
    const dropped = pausedState ? _snap().readConfig(pausedState.config) : null;
    _snap().clear();
    pausedState = null;
    resumeReady = false;
    sessionConfig = null;
    window.KcPractice?.stop?.();
    _showResumeOption();
    if (dropped) window.ExerciseSession?.onEnd?.("discarded", 0, dropped);
    sessionSummary.textContent = "Saved session discarded. Set up a new block when you're ready.";
    sessionSummary.classList.remove("hidden");
  };

  /* 🔴 ALWAYS FALSE, and kept as a function on purpose. A block has no length
     any more (2026-08-23), so nothing ends a session but a pause — but this is
     the hook `_loadNextPracticeQuestion` asks before every fetch, and deleting
     it would mean editing every call site to stop asking. Restoring a quota is
     one line here; finding all the callers again is not. */
  const shouldFinishInsteadOfAdvance = () =>
    !!state &&
    ((Number.isFinite(sessionConfig?.quota) && state.served >= sessionConfig.quota) ||
      /* A planned exercise block ends the moment a variant is solved — the
         count was a maximum (practice/exercise-planner.js). */
      (!!sessionConfig?.exercise && !!window.KcPractice?.solved?.()));

  const hasSavedQuestion = (questionId) =>
    !!pausedState && String(questionId ?? "") === pausedState.questionId;

  /* Reasons a block ends WITHOUT a pause. "ended" is gone with the button that
     sent it, and so is "complete": the quota it counted down to no longer
     exists. What is left is a failure to load and the placement taking over —
     both of which happen TO the learner, which is why each one says what
     happened rather than congratulating them. */
  const finish = (reason, message) => {
    if (!state) return;
    window.AnswerHistory?.abandon();
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
    const config = sessionConfig;
    sessionConfig = null;
    // Read before stop() clears the planner: did the block solve its problem?
    const outcome = config ? window.KcPractice?.outcome?.() || null : null;
    if (config) window.KcPractice?.stop?.();
    _snap().clear();
    _showResumeOption();
    sessionStatusRow.classList.add("hidden");
    pagePractice.classList.add("session-idle");
    /* 🔴 `message` OVERRIDES THE REASON, and the caller that passes one knows
       something this function cannot. "Could not load a question — check the
       connection" is right for a dead network and WRONG for the other way a
       load fails: the backend answering, correctly, that this concept has no
       unseen drill left at this rung (app/content_gaps.py writes that
       sentence, names the concept and says what to do about it). Telling a
       learner to check their connection when the connection is fine is how a
       content gap gets read as a bug in the app. */
    sessionSummary.textContent =
      message ||
      (reason === "error"
        ? "Could not load a question — check the connection and try again."
        : reason === "placement"
          ? "Placement test started — its questions are timed on their own clock, one at a time."
          : reason === "complete" && outcome?.solved
            ? `Solved on attempt ${outcome.attempts} — ${served} of ${config.quota} questions used. Recorded answers are kept.`
          : reason === "complete" && config?.quota
            ? `Done — ${served} of ${config.quota} questions. Recorded answers are kept.`
            : `Session stopped after ${served} question${served === 1 ? "" : "s"}. Recorded answers are kept.`);
    sessionSummary.classList.remove("hidden");
    if (config) window.ExerciseSession?.onEnd?.(reason, served, config, outcome);
  };

  /* Install the exercise session's numbers for the NEXT `start()`. `null`
     clears them. Refused while a block is live — a running countdown must not
     change under the learner's hands (the same rule the idle picker keeps). */
  const configure = (cfg) => {
    if (state) return false;
    sessionConfig = _snap().readConfig(cfg);
    return true;
  };

  /* NO NEXT QUESTION, AND THE SCREEN MUST NOT FREEZE.

     Seth, 2026-09-01: the review clock ran out, the app tried to advance, and
     "instead of going to the next problem it just froze and didn't allow me to
     press the next button". The backend had answered the truth — every
     fill-in-the-blank drill for that concept was already served, "all 2 of
     them" — and that answer arrived as a thrown Error into
     `_loadNextPracticeQuestion`, whose six callers ALL discard it
     (`.catch(() => {})` here in `_forceSubmitOrAdvance`, `_forceAdvance` and
     the watchdog poll; an unhandled rejection in the Next-button handler).
     By then the loader had already hidden the feedback area, cleared the
     editor and hidden Next — so nothing was left on screen to press.

     🔴 IT IS NOT ALWAYS A SESSION. The same wall is reachable from a
     `?lesson=<kc>` KC drill, which is deliberately sessionless — so `finish()`
     alone could not be the answer: it returns immediately when `state` is null
     and the learner stays frozen.

     🔴 AND IT MUST NOT EAT A PAUSED BLOCK. `finish()` clears `pausedState` and
     the saved snapshot, which is correct for the block it is ending and
     destructive for a block that is merely waiting to be resumed. With no live
     session there is nothing to end: put the idle surface back, say why, and
     leave the saved session offerable. */
  const deadEnd = (message) => {
    if (state) {
      finish("error", message);
      return;
    }
    _stopTick();
    _stopPoll();
    clockHolds.clear();
    sessionStatusRow.classList.add("hidden");
    pagePractice.classList.add("session-idle");
    _showResumeOption();
    sessionSummary.textContent = message;
    sessionSummary.classList.remove("hidden");
  };

  /* The idle button starts a PLAIN block: whatever an exercise page installed
     and never started is dropped here, not carried into the adaptive queue. */
  sessionStartBtn.addEventListener("click", () => {
    sessionConfig = null;
    start();
  });
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
      /* `|| 0` on a null `remaining` would hand an untimed question 30
         seconds on the next load — the one thing "No limit" promises it will
         not do. Untimed snapshots pass through untouched. */
      if (snapshot.remaining !== null) {
        snapshot.remaining = Math.max(30, snapshot.remaining || 0);
      }
    }
    _snap().write(snapshot);
  });

  /* Nothing to prefill. `delta_drills_session_setup` — the localStorage key
     that carried the learner's last questions/answer-time/review-time — is not
     read or written anywhere any more; it is left on disk rather than migrated
     because deleting it buys nothing and a stale key is inert. */
  pausedState = _snap().read();
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
    /* Between blocks nothing is on screen and the next question's clock is
       its concept's, unknown until the queue serves it — so the idle notch
       shows a placeholder rather than one concept's number as if it were a
       rule. A paused question is on screen in every sense that matters, and
       its own clock is what a resume gets. */
    /* 🔴 THE PRECEDENCE THE RESUME WILL ACTUALLY USE, not a second copy of it.
       This read `pausedState.secsAllowed` directly, which is the paused
       QUESTION's concept cap — and `_phaseLimit` asks the block's OWN config
       first. An ARENA exercise session set to 05:00, paused on a 20:00
       concept, therefore advertised 20:00 on the notch and resumed at 05:00.
       Asking the snapshot module keeps the two the same statement. `null` (a
       scoped block's "No limit") falls through `clockText` as ∞, which is what
       that resume really gets. Codex, 2026-09-09. */
    idleClockText: () => {
      if (pausedState) return clockText(_snap().phaseLimit(pausedState));
      return window.SessionClock?.hasOwn?.() ? clockText(ANSWER_SECS()) : "--:--";
    },
    answerSeconds: () => ANSWER_SECS(),
    reviewSeconds: () => REVIEW_SECS(),
    // True when a session was paused and is waiting to be resumed. switchTab
    // needs this to know the question on screen belongs to that session and
    // must not be replaced by a preference refresh.
    hasPausedSession: () => !!pausedState,
    pausedConfig: () => (pausedState ? _snap().readConfig(pausedState.config) : null),
    pausedServed: () => (pausedState ? pausedState.served : 0),
    config: () => (sessionConfig ? { ...sessionConfig } : null),
    configure,
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
    deadEnd,
  };
})();
