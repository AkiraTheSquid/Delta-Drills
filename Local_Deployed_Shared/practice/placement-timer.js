/* ================================================================
   PLACEMENT TEST — one fixed clock per probe

   The placement test runs OUTSIDE a practice session: starting it calls
   PracticeSession.finish("placement") (see events.js), so none of the
   session's strict timers apply and, until now, a probe had no time limit
   at all. Seth's rule for the test is deliberately simpler than a session:
   EVERY question gets the SAME allowance, and the learner never chooses it.
   That is the whole point of a placement — comparable evidence per probe.

   So: PLACEMENT_ANSWER_SECS for each probe, no setup panel, no review
   countdown. Reviewing a graded probe is untimed; the next probe's clock
   starts when that probe renders.

   When time runs out we record what the learner actually has:
     - typed something ≠ the starter code  → click Submit and grade it
     - nothing of their own                 → click "I don't know yet",
       which records a placement miss with no attempt (events.js)
   Both are honest signals for the estimator; neither invents an answer.

   A reload is not free time. The deadline is persisted per question id, so
   coming straight back resumes the same clock; after a real break (longer
   than RETURN_GRACE_SECS) the probe gets its full time again — the same
   trade PracticeSession makes for a paused step.
   ================================================================ */
const PLACEMENT_ANSWER_SECS = 120;

const PlacementTimer = (() => {
  /* How long a break may be before the probe's clock is handed back whole.
     Inside the window the countdown resumes where it stopped, so reloading
     is not a way to opt out of the limit. */
  const RETURN_GRACE_SECS = 120;
  /* Never resume onto a dead clock: a restored probe that already expired
     while the tab was gone would auto-record itself the instant it painted,
     which reads as the app answering for the learner. */
  const RESUME_FLOOR_SECS = 15;
  const STORE_KEY = "delta_drills_placement_probe_clock";

  /* The clock is a DEADLINE, not a counter.

     Decrementing a counter once per interval callback measures CALLBACKS, not
     time: a background or throttled tab fires the callback once every few
     seconds (once a minute, in a fully backgrounded one) and each late callback
     took a single second off — so leaving the tab handed the learner minutes of
     extra thinking time on a supposedly fixed 2:00, and `_write` pushed the
     persisted deadline forward with it. Every read derives from `deadlineAt`
     instead, so a late callback is late, not cheap. */
  let deadlineAt = 0;
  let interval = null;
  let chip = null;
  let expiring = false;

  const _remaining = () => (deadlineAt ? Math.max(0, Math.ceil((deadlineAt - Date.now()) / 1000)) : 0);

  /* `PracticeAPI` is a top-level `const` in api.js, and a top-level const of a
     classic script is NOT a property of `window` — reading it off `window`
     silently yields undefined, so the clock would never see a probe and never
     start. notebook-view.js carries the same note for the same reason. Read the
     script-scope binding first; the window fallback is for a future module. */
  const _api = () => (typeof PracticeAPI !== "undefined" ? PracticeAPI : window.PracticeAPI);
  const q = () => _api()?.currentQuestion || null;
  const isProbe = () => !!q()?.diagnostic_active;

  // A session and a placement never time the same question: placement ends
  // the session before its first probe. If one is somehow live, it owns the
  // clock and this module stays out of the way rather than double-submitting.
  const sessionOwnsClock = () =>
    typeof PracticeSession !== "undefined" && PracticeSession.isActive?.();

  const _questionId = () => {
    const raw = q()?.question_id ?? q()?.id;
    return raw == null ? "" : String(raw);
  };

  const _read = () => {
    try {
      const saved = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
      if (!saved || !saved.questionId || !Number.isFinite(saved.deadline)) return null;
      return saved;
    } catch (_) {
      return null;
    }
  };

  const _write = () => {
    if (!isProbe() || !deadlineAt) return;
    try {
      localStorage.setItem(
        STORE_KEY,
        JSON.stringify({ questionId: _questionId(), deadline: deadlineAt, savedAt: Date.now() }),
      );
    } catch (_) {}
  };

  const _clearSaved = () => {
    try { localStorage.removeItem(STORE_KEY); } catch (_) {}
  };

  /* Seconds this probe should start with. A snapshot for a DIFFERENT question,
     or one written before a real break, is worth nothing — full time. */
  const _startingSecs = () => {
    const saved = _read();
    if (!saved || saved.questionId !== _questionId()) return PLACEMENT_ANSWER_SECS;
    const away = (Date.now() - (Number(saved.savedAt) || 0)) / 1000;
    if (!Number.isFinite(away) || away > RETURN_GRACE_SECS) return PLACEMENT_ANSWER_SECS;
    const left = Math.round((saved.deadline - Date.now()) / 1000);
    return Math.max(RESUME_FLOOR_SECS, Math.min(PLACEMENT_ANSWER_SECS, left));
  };

  const _format = (secs) => {
    const clamped = Math.max(0, Math.round(secs));
    return String(Math.floor(clamped / 60)).padStart(2, "0") + ":" +
      String(clamped % 60).padStart(2, "0");
  };

  /* The chip lives in .question-number-row, which is part of the practice
     workspace — the same node the Placement page re-parents into itself. Being
     a child of it means the countdown follows the question wherever the
     workspace is hosted, with no second copy to keep in sync.

     It was in #cold-start-badge until 2026-08-23, when that badge was deleted
     along with its standing explanation copy. The row is a strictly better
     home: the badge only appeared on placement and calibration questions, so
     the chip's anchor came and went with it. */
  const _chip = () => {
    if (chip && chip.isConnected) return chip;
    const row = document.querySelector(".question-number-row");
    if (!row) return null;
    chip = row.querySelector(".placement-timer") || document.getElementById("placement-timer");
    if (!chip) {
      chip = document.createElement("span");
      chip.className = "placement-timer hidden";
      chip.id = "placement-timer";
      chip.setAttribute("aria-live", "off");
      chip.dataset.ddInfo = "placement-timer";
      chip.dataset.ddInfoPlace = "after";
      row.appendChild(chip);
    }
    return chip;
  };

  const _paint = () => {
    const el = _chip();
    if (!el) return;
    const left = _remaining();
    el.textContent = _format(left);
    el.classList.remove("hidden");
    el.classList.toggle("placement-timer--low", left <= 30);
  };

  const _hideChip = () => {
    const el = chip && chip.isConnected ? chip : null;
    el?.classList.add("hidden");
  };

  const _stopTick = () => {
    if (interval) {
      clearInterval(interval);
      interval = null;
    }
  };

  /* Did the learner write anything of their own?

     Not a string compare against the starter code: `submissionCode()` returns
     the notebook's cells with `# --- cell N ---` separators inserted, so an
     untouched editor never equals its own starter text and every expiry would
     look like work worth grading. Strip the separators, drop blank lines, and
     compare what is left. */
  const _hasOwnWork = () => {
    const current = window.DeltaNotebook?.submissionCode?.() ??
      (typeof codeEditor !== "undefined" ? codeEditor.value : "");
    const starter = q()?.starter_code ||
      (typeof DEFAULT_EDITOR_CODE !== "undefined" ? DEFAULT_EDITOR_CODE : "");
    const norm = (text) =>
      String(text || "")
        .split("\n")
        .filter((line) => !/^\s*#\s*---\s*cell\s*\d+\s*---\s*$/i.test(line))
        .map((line) => line.trim())
        .filter(Boolean)
        .join("\n");
    const typed = norm(current);
    return !!typed && typed !== norm(starter);
  };

  /* Time is up. Prefer the learner's own code — a graded miss and a
     "don't know" are both evidence, but only one of them is theirs. */
  const _expire = () => {
    if (expiring) return;
    expiring = true;
    _stopTick();
    // Drop the deadline before clicking: the click path re-enters this module
    // (grading pauses it, advancing stops it) and a stale deadline left behind
    // is what a returning `visibilitychange` would try to resume.
    deadlineAt = 0;
    _clearSaved();
    _hideChip();
    const submitBtn = document.getElementById("practice-submit-btn");
    const submitArea = document.getElementById("practice-submit-area");
    const dontKnowBtn = document.getElementById("practice-dontknow-btn");
    const canSubmit = submitBtn && submitArea &&
      !submitArea.classList.contains("hidden") && !submitBtn.disabled;
    if (canSubmit && _hasOwnWork()) submitBtn.click();
    else if (dontKnowBtn && !dontKnowBtn.classList.contains("hidden") && !dontKnowBtn.disabled) {
      dontKnowBtn.click();
    } else if (canSubmit) submitBtn.click();
    expiring = false;
  };

  const _tick = () => {
    _paint();
    _write();
    if (_remaining() <= 0) {
      _stopTick();
      _expire();
    }
  };

  const _run = (secs) => {
    _stopTick();
    deadlineAt = Date.now() + Math.max(0, secs) * 1000;
    _paint();
    _write();
    interval = setInterval(_tick, 1000);
  };

  // Every renderQuestion() lands here (ui.js). Non-probe questions stop the
  // clock instead of starting one, so leaving the placement mid-run cannot
  // leave a countdown ticking over ordinary practice.
  const onQuestionRendered = () => {
    if (!isProbe() || sessionOwnsClock()) {
      stop();
      return;
    }
    _run(_startingSecs());
  };

  // Grading is in flight — the learner is no longer answering, so the clock
  // stops rather than force-submitting a second time underneath the grade.
  const pauseForGrading = () => {
    if (!interval && !deadlineAt) return;
    _stopTick();
    deadlineAt = 0;
    _clearSaved();
    _hideChip();
  };

  // Submit failed and the editor came back: a short grace window, never the
  // full allowance (that would make a failed submit the way to buy time).
  const resumeAfterFailedSubmit = () => {
    if (!isProbe() || sessionOwnsClock()) return;
    _run(Math.max(_remaining(), 30));
  };

  const stop = () => {
    _stopTick();
    deadlineAt = 0;
    _clearSaved();
    _hideChip();
  };

  // A reload is a pause: keep the deadline so returning inside the grace
  // window resumes the same clock (see _startingSecs).
  window.addEventListener("pagehide", () => {
    if (interval) _write();
    _stopTick();
  });

  /* Coming BACK needs its own handler, and for two different returns.

     A back/forward-cache restore does not re-run the scripts and does not
     re-render the question, so nothing would call onQuestionRendered again:
     without this the countdown sits frozen at whatever it read when the page
     was put away, and the probe never expires. `persisted` marks that case.
     A tab that was merely hidden kept ticking, but its interval was throttled,
     so the first thing to do on return is settle up against the deadline. */
  const _resumeFromDeadline = () => {
    if (!deadlineAt || !isProbe() || sessionOwnsClock()) return;
    if (_remaining() <= 0) {
      _stopTick();
      _expire();
      return;
    }
    _stopTick();
    _paint();
    _write();
    interval = setInterval(_tick, 1000);
  };

  window.addEventListener("pageshow", (event) => {
    if (event.persisted) _resumeFromDeadline();
  });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) _resumeFromDeadline();
  });

  return {
    onQuestionRendered,
    pauseForGrading,
    resumeAfterFailedSubmit,
    stop,
    secondsPerQuestion: () => PLACEMENT_ANSWER_SECS,
    isRunning: () => !!interval,
  };
})();
window.PlacementTimer = PlacementTimer;
