/* ================================================================
   PRACTICE SESSION — resumable blocks + strict per-question timers

   The learner commits up front: how many questions, how much time to
   answer each one, and how much time to review each grade. Timers stay
   strict while a session is open, but Pause & exit freezes the current
   question so it can be resumed later. Closing/reloading the page also
   leaves a resumable snapshot.

   Lifecycle hooks (called from ui.js / events.js):
     PracticeSession.onQuestionRendered()   — every renderQuestion()
     PracticeSession.pauseForGrading()      — submit clicked, grading in flight
     PracticeSession.recordReviewResult()   — preserve grade/review UI for resume
     PracticeSession.resumeAnswerPhase()    — submit failed, back to answering
     PracticeSession.beginReviewPhase()     — grade landed, review starts
     PracticeSession.shouldFinishInsteadOfAdvance() — quota check before
       _loadNextPracticeQuestion() fetches another question
   ================================================================ */

const SESSION_SETUP_KEY = "delta_drills_session_setup";
const SESSION_STATE_VERSION = 1;

const parseTimerInput = (value, fallback) => {
  const raw = String(value || "").trim();
  if (!raw) return fallback;
  if (raw.includes(":")) {
    const [mStr, sStr] = raw.split(":");
    const m = Number(mStr);
    const s = Number(sStr);
    if (!Number.isFinite(m) || !Number.isFinite(s)) return fallback;
    return Math.max(1, Math.min(3600, m * 60 + s));
  }
  const asNumber = Number(raw);
  if (!Number.isFinite(asNumber)) return fallback;
  return Math.max(1, Math.min(3600, Math.round(asNumber)));
};

const formatTimer = (value) => {
  const clamped = Math.max(0, Math.min(3600, Math.round(value)));
  const m = Math.floor(clamped / 60);
  const s = clamped % 60;
  return String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
};

// "1 h 10 min" / "45 min" / "90 sec" — for the setup-panel total estimate.
const formatDuration = (secs) => {
  const total = Math.max(0, Math.round(secs));
  if (total < 120) return `${total} sec`;
  const h = Math.floor(total / 3600);
  const m = Math.round((total % 3600) / 60);
  if (h === 0) return `${m} min`;
  if (m === 0) return `${h} h`;
  return `${h} h ${m} min`;
};

const PracticeSession = (() => {
  const pagePractice = document.getElementById("page-practice");

  let state = null; // { total, answerSecs, reviewSecs, served, phase, review }
  let pausedState = null;
  let interval = null;
  let remaining = 0;
  let advancePoll = null;
  let resumeReady = false;
  let resumePending = false;

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
      if (!Number.isFinite(saved.total) || !Number.isFinite(saved.served) || !saved.questionId) {
        return null;
      }
      const phase = saved.phase === "review" && saved.review ? "review" : "answer";
      const phaseLimit = phase === "review" ? saved.reviewSecs : saved.answerSecs;
      const savedRemaining = Number.isFinite(saved.remaining) ? saved.remaining : phaseLimit;
      return {
        version: SESSION_STATE_VERSION,
        total: Math.max(1, Math.min(50, Math.round(saved.total))),
        answerSecs: Math.max(1, Math.min(3600, Math.round(saved.answerSecs || 300))),
        reviewSecs: Math.max(1, Math.min(3600, Math.round(saved.reviewSecs || 120))),
        served: Math.max(1, Math.min(Math.round(saved.served), Math.round(saved.total))),
        phase,
        remaining: Math.max(1, Math.min(3600, Math.round(savedRemaining || 30))),
        questionId: String(saved.questionId),
        draft: typeof saved.draft === "string" ? saved.draft : "",
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

  const _snapshot = () => {
    if (!state) return null;
    const review = state.review ? { ...state.review } : null;
    if (review && state.phase === "review") {
      review.feedbackComplete = !nextProblemBtn.classList.contains("hidden");
    }
    return {
      version: SESSION_STATE_VERSION,
      total: state.total,
      answerSecs: state.answerSecs,
      reviewSecs: state.reviewSecs,
      served: state.served,
      phase: state.phase,
      remaining,
      questionId: _questionId(),
      draft: codeEditor.value,
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
    const stable = phase === "answer" || phase === "review";
    sessionPauseBtn.disabled = !stable;
    sessionPauseBtn.title = stable
      ? "Pause this session and return to setup. Resume later from this question."
      : "Pause becomes available when this short step finishes.";
  };

  const _tick = (onExpire) => {
    _stopTick();
    _updateCountdown();
    _persist();
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
    return `Question ${saved.served} of ${saved.total} · ${phase} · ${formatTimer(saved.remaining)} left`;
  };

  const _showResumeOption = () => {
    if (!pausedState) {
      sessionResumePanel.classList.add("hidden");
      sessionSetupPanel.classList.remove("session-setup--has-resume");
      return;
    }
    sessionResumeSummary.textContent = _resumeSummary(pausedState);
    sessionResumeBtn.disabled = !resumeReady;
    sessionResumePanel.classList.remove("hidden");
    sessionSetupPanel.classList.add("session-setup--has-resume");
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

  const _restoreReview = () => {
    const review = state.review;
    if (!review) return;
    codeEditor.value = state.draft || review.userCode || codeEditor.value;
    solutionCode.textContent = review.solutionCode || PracticeAPI.currentQuestion?.solution_code || "";
    practiceSubmitArea.classList.add("hidden");
    practiceFeedbackArea.classList.remove("hidden");
    applyResult(!!review.correct);
    if (typeof renderFailedTests === "function") {
      renderFailedTests(review.result || { correct: !!review.correct }, PracticeAPI.currentQuestion);
    }
    const feedbackSaved =
      review.feedbackComplete ||
      practiceProgress.pendingFeedback?.questionId === PracticeAPI.currentQuestion?.question_id ||
      PracticeAPI.currentQuestion?.diagnostic_active;
    if (feedbackSaved) showNextProblemButton();
  };

  // Live total-time readout: questions × (answer + review), recomputed as
  // the learner edits any of the three setup fields.
  const _updateEstimate = () => {
    const total = Math.max(1, Math.min(50, Math.round(Number(sessionQuestionCountInput.value) || 10)));
    const answerSecs = parseTimerInput(sessionAnswerTimeInput.value, 300);
    const reviewSecs = parseTimerInput(sessionReviewTimeInput.value, 120);
    const perQuestion = answerSecs + reviewSecs;
    sessionTimeEstimate.innerHTML =
      `Total session time: <strong>${formatDuration(total * perQuestion)}</strong>` +
      ` · ${total} question${total === 1 ? "" : "s"} × ${formatTimer(perQuestion)} each`;
  };

  const start = () => {
    const total = Math.max(1, Math.min(50, Math.round(Number(sessionQuestionCountInput.value) || 10)));
    const answerSecs = parseTimerInput(sessionAnswerTimeInput.value, 300);
    const reviewSecs = parseTimerInput(sessionReviewTimeInput.value, 120);
    sessionQuestionCountInput.value = String(total);
    sessionAnswerTimeInput.value = formatTimer(answerSecs);
    sessionReviewTimeInput.value = formatTimer(reviewSecs);
    _updateEstimate();
    try {
      localStorage.setItem(SESSION_SETUP_KEY, JSON.stringify({ total, answerSecs, reviewSecs }));
    } catch (_) {}

    _clearSaved();
    pausedState = null;
    resumeReady = false;
    _showResumeOption();
    state = { total, answerSecs, reviewSecs, served: 0, phase: null, review: null };
    sessionSummary.classList.add("hidden");
    sessionProgressLabel.textContent = `0 / ${total}`;
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

  const onQuestionRendered = () => {
    if (!isActive()) {
      if (pausedState) {
        resumeReady = _questionId() === pausedState.questionId;
        sessionResumeSummary.textContent = resumeReady
          ? _resumeSummary(pausedState)
          : "Saved question is no longer available. Discard this session and start a new one.";
        sessionResumeBtn.disabled = !resumeReady;
      }
      return;
    }
    _stopPoll();
    state.served += 1;
    state.review = null;
    sessionProgressLabel.textContent = `${Math.min(state.served, state.total)} / ${state.total}`;
    _setPhase("answer", "Answering");
    remaining = state.answerSecs;
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

  const beginReviewPhase = () => {
    // Only a grade we are actually waiting on may start review — a stale
    // response landing after End session → Start session must not hijack the
    // new session's first question.
    if (!isActive() || state.phase !== "grading") return;
    _setPhase("review", "Reviewing");
    remaining = state.reviewSecs;
    _tick(_forceAdvance);
  };

  const pause = () => {
    if (!isActive() || !["answer", "review"].includes(state.phase)) return;
    _stopTick();
    _stopPoll();
    pausedState = _snapshot();
    _writeSaved(pausedState);
    state = null;
    resumeReady = true;
    sessionStatusRow.classList.add("hidden");
    pagePractice.classList.add("session-idle");
    sessionSummary.textContent = "Session paused. Your question, code, timer, and review state are saved.";
    sessionSummary.classList.remove("hidden");
    _showResumeOption();
  };

  const resume = async () => {
    if (resumePending || !pausedState || !resumeReady || _questionId() !== pausedState.questionId) return;
    resumePending = true;
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
    state = {
      total: pausedState.total,
      answerSecs: pausedState.answerSecs,
      reviewSecs: pausedState.reviewSecs,
      served: pausedState.served,
      phase: pausedState.phase,
      review: pausedState.review,
      draft: pausedState.draft,
    };
    remaining = pausedState.remaining;
    pausedState = null;
    resumeReady = false;
    sessionResumePanel.classList.add("hidden");
    sessionSetupPanel.classList.remove("session-setup--has-resume");
    sessionSummary.classList.add("hidden");
    sessionProgressLabel.textContent = `${Math.min(state.served, state.total)} / ${state.total}`;
    sessionStatusRow.classList.remove("hidden");
    pagePractice.classList.remove("session-idle");
    codeEditor.value = state.draft || codeEditor.value;
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

  const shouldFinishInsteadOfAdvance = () => isActive() && state.served >= state.total;

  const hasSavedQuestion = (questionId) =>
    !!pausedState && String(questionId ?? "") === pausedState.questionId;

  const finish = (reason) => {
    if (!state) return;
    const { served, total } = state;
    _stopTick();
    _stopPoll();
    state = null;
    pausedState = null;
    resumeReady = false;
    _clearSaved();
    _showResumeOption();
    sessionStatusRow.classList.add("hidden");
    pagePractice.classList.add("session-idle");
    sessionSummary.textContent =
      reason === "ended"
        ? `Session ended early — ${served} of ${total} questions. Recorded answers are kept.`
        : reason === "error"
          ? "Could not load a question — check the connection and start again."
          : reason === "placement"
            ? "Placement diagnostic started — begin a session to answer the probes."
            : `Session complete — ${total} question${total === 1 ? "" : "s"} done. Set up the next block when you're ready.`;
    sessionSummary.classList.remove("hidden");
  };

  sessionStartBtn.addEventListener("click", start);
  [sessionQuestionCountInput, sessionAnswerTimeInput, sessionReviewTimeInput].forEach((input) => {
    input.addEventListener("input", _updateEstimate);
  });
  sessionPauseBtn.addEventListener("click", pause);
  sessionEndBtn.addEventListener("click", () => finish("ended"));
  sessionResumeBtn.addEventListener("click", resume);
  sessionDiscardBtn.addEventListener("click", discard);
  codeEditor.addEventListener("input", () => {
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

  // Prefill setup fields from last session.
  try {
    const saved = JSON.parse(localStorage.getItem(SESSION_SETUP_KEY) || "null");
    if (saved) {
      if (Number.isFinite(saved.total)) sessionQuestionCountInput.value = String(saved.total);
      if (Number.isFinite(saved.answerSecs)) sessionAnswerTimeInput.value = formatTimer(saved.answerSecs);
      if (Number.isFinite(saved.reviewSecs)) sessionReviewTimeInput.value = formatTimer(saved.reviewSecs);
    }
  } catch (_) {}
  _updateEstimate();

  pausedState = _readSaved();
  _showResumeOption();

  return {
    isActive,
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
    pauseForGrading,
    pauseForAdvance,
    recordReviewResult,
    resumeAnswerPhase,
    beginReviewPhase,
    shouldFinishInsteadOfAdvance,
    finish,
  };
})();
