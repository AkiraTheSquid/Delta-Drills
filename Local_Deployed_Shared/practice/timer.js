/* ================================================================
   PRACTICE SESSION — rigid session setup + strict per-question timers

   The learner commits up front: how many questions, how much time to
   answer each one, and how much time to review each grade. Once the
   session starts the timers are strict — answer time expiring force-
   submits (or force-advances when there is nothing to submit), review
   time expiring force-advances. No pause, no mid-question re-negotiation.

   Lifecycle hooks (called from ui.js / events.js):
     PracticeSession.onQuestionRendered()   — every renderQuestion()
     PracticeSession.pauseForGrading()      — submit clicked, grading in flight
     PracticeSession.resumeAnswerPhase()    — submit failed, back to answering
     PracticeSession.beginReviewPhase()     — grade landed, review starts
     PracticeSession.shouldFinishInsteadOfAdvance() — quota check before
       _loadNextPracticeQuestion() fetches another question
   ================================================================ */

const SESSION_SETUP_KEY = "delta_drills_session_setup";

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

const PracticeSession = (() => {
  const pagePractice = document.getElementById("page-practice");

  let state = null; // { total, answerSecs, reviewSecs, served, phase }
  let interval = null;
  let remaining = 0;
  let advancePoll = null;

  const isActive = () => !!state;

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

  const _updateCountdown = () => {
    sessionCountdown.textContent = formatTimer(remaining);
    sessionCountdown.classList.toggle("session-countdown--low", remaining <= 30);
  };

  const _setPhase = (phase, label) => {
    state.phase = phase;
    sessionPhaseLabel.textContent = label;
    sessionStatusRow.classList.toggle("session-status--review", phase === "review");
  };

  const _tick = (onExpire) => {
    _stopTick();
    _updateCountdown();
    interval = setInterval(() => {
      remaining--;
      _updateCountdown();
      if (remaining <= 0) {
        _stopTick();
        onExpire();
      }
    }, 1000);
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

  const start = () => {
    const total = Math.max(1, Math.min(50, Math.round(Number(sessionQuestionCountInput.value) || 10)));
    const answerSecs = parseTimerInput(sessionAnswerTimeInput.value, 300);
    const reviewSecs = parseTimerInput(sessionReviewTimeInput.value, 120);
    sessionQuestionCountInput.value = String(total);
    sessionAnswerTimeInput.value = formatTimer(answerSecs);
    sessionReviewTimeInput.value = formatTimer(reviewSecs);
    try {
      localStorage.setItem(SESSION_SETUP_KEY, JSON.stringify({ total, answerSecs, reviewSecs }));
    } catch (_) {}

    state = { total, answerSecs, reviewSecs, served: 0, phase: null };
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
    if (!isActive()) return;
    _stopPoll();
    state.served += 1;
    sessionProgressLabel.textContent = `${Math.min(state.served, state.total)} / ${state.total}`;
    _setPhase("answer", "Answering");
    remaining = state.answerSecs;
    _tick(_forceSubmitOrAdvance);
  };

  const pauseForGrading = () => {
    if (!isActive()) return;
    _stopTick();
    _setPhase("grading", "Grading…");
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
    if (!isActive()) return;
    _setPhase("review", "Reviewing");
    remaining = state.reviewSecs;
    _tick(_forceAdvance);
  };

  const shouldFinishInsteadOfAdvance = () => isActive() && state.served >= state.total;

  const finish = (reason) => {
    if (!state) return;
    const { served, total } = state;
    _stopTick();
    _stopPoll();
    state = null;
    sessionStatusRow.classList.add("hidden");
    pagePractice.classList.add("session-idle");
    sessionSummary.textContent =
      reason === "ended"
        ? `Session ended early — ${served} of ${total} questions.`
        : reason === "error"
          ? "Could not load a question — check the connection and start again."
          : `Session complete — ${total} questions done. Set up the next block when you're ready.`;
    sessionSummary.classList.remove("hidden");
  };

  sessionStartBtn.addEventListener("click", start);
  sessionEndBtn.addEventListener("click", () => finish("ended"));

  // Prefill the setup fields from the last session.
  try {
    const saved = JSON.parse(localStorage.getItem(SESSION_SETUP_KEY) || "null");
    if (saved) {
      if (Number.isFinite(saved.total)) sessionQuestionCountInput.value = String(saved.total);
      if (Number.isFinite(saved.answerSecs)) sessionAnswerTimeInput.value = formatTimer(saved.answerSecs);
      if (Number.isFinite(saved.reviewSecs)) sessionReviewTimeInput.value = formatTimer(saved.reviewSecs);
    }
  } catch (_) {}

  return {
    isActive,
    start,
    onQuestionRendered,
    pauseForGrading,
    resumeAnswerPhase,
    beginReviewPhase,
    shouldFinishInsteadOfAdvance,
    finish,
  };
})();
