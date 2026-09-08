/* ================================================================
   PLACEMENT TEST — one fixed clock per CONCEPT, inside a total cap

   The placement test runs OUTSIDE a practice session: starting it calls
   PracticeSession.finish("placement") (see events.js), so none of the
   session's strict timers apply and, until now, a probe had no time limit
   at all. Seth's rule for the test is deliberately simpler than a session:
   the learner never chooses the allowance, and it never varies by question
   or by learner — every probe on a concept gets that concept's clock, so
   the probes on one concept are comparable evidence.

   THE SERVER OWNS THE NUMBER. lessons/placement_time_caps.json holds one
   clock per concept (Seth, 2026-09-07: einops and broadcasting get ARENA's
   own ~10 minutes, the one-call Python/NumPy drills five or six, make_rays_1d
   fifteen; a flat 20:00 had him at 5 problems an hour). app/diagnostic.py
   sends it on the question as `diagnostic_secs_allowed`, already clamped to
   what is left of the plan, and charges serve-to-answer time against that
   same cap. This file DISPLAYS it: PLACEMENT_ANSWER_SECS (20:00) is only the
   ceiling and the fallback for a payload that carries no clock. No review
   countdown: reviewing a graded problem is untimed; the next problem's clock
   starts when it renders.

   🔴 THE TOTAL PLAN IS A HARD CAP TOO. The learner picked 1h / 3h / 6h
   (placement-plan.js) and the server refuses to go past it, so the LAST
   problem of a run gets only what is left: the server says how much in
   `plan.problem_secs_allowed` on every status, and `_startingSecs` takes the
   smaller of that and the constant. The server also charges its own
   serve-to-answer time regardless of what this clock shows, so the clock
   cannot buy time — it can only tell the truth about how much there is.

   When time runs out we record what the learner actually has:
     - typed something ≠ the starter code  → click Submit and grade it
     - nothing of their own                 → click "I don't know yet",
       which records a placement miss with no attempt (events.js)
   Both are honest signals for the estimator; neither invents an answer.

   A reload is not free time. The deadline is persisted per question id, so
   coming straight back resumes the same clock. After a real break the SERVER
   decides: inside the clock it sends the same problem with what is left;
   past the clock (+ its grace) it records that probe as a timed-out miss and
   serves a NEW one — so the "full time again after RETURN_GRACE_SECS" below
   only ever applies to a problem the server still considers live.
   ================================================================ */
const PLACEMENT_ANSWER_SECS = 1200;

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

  /* What the server will let THIS problem take: the concept's clock, or the
     plan's remainder when that is shorter (the last problem of a capped run).
     Read off the QUESTION first (`diagnostic_secs_allowed`, computed for this
     probe as it was served); the status api.js kept is the fallback — it was
     fetched before this question was picked, so its `problem_secs_allowed` may
     describe the previous probe's concept. With neither, the ceiling — the
     server still caps what it charges. */
  const _plan = () => _api()?.lastDiagnosticStatus?.plan || null;
  const _allowedSecs = () => {
    const raw = q()?.diagnostic_secs_allowed;
    const own = Number(raw);
    // 🔴 ZERO IS AN ANSWER. A probe resumed after its clock ran out comes back
    // with 0 — treating that as "no value" fell through to the cached status
    // or the 20:00 ceiling, and a dead clock showed as a live one (codex,
    // 2026-09-07). Only a missing/invalid field falls back.
    if (raw != null && Number.isFinite(own) && own >= 0) {
      return Math.min(PLACEMENT_ANSWER_SECS, Math.round(own));
    }
    const allowed = Number(_plan()?.problem_secs_allowed);
    if (!Number.isFinite(allowed) || allowed <= 0) return PLACEMENT_ANSWER_SECS;
    return Math.min(PLACEMENT_ANSWER_SECS, Math.round(allowed));
  };
  /* Seconds this problem has been on the clock — advisory, sent with a
     "don't know" so the server has a number even when its own clock predates
     the run. */
  let startedAt = 0;
  const elapsedSecs = () =>
    startedAt ? Math.max(0, Math.round((Date.now() - startedAt) / 1000)) : null;

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
    const allowed = _allowedSecs();
    const saved = _read();
    if (!saved || saved.questionId !== _questionId()) return allowed;
    const away = (Date.now() - (Number(saved.savedAt) || 0)) / 1000;
    if (!Number.isFinite(away) || away > RETURN_GRACE_SECS) return allowed;
    const left = Math.round((saved.deadline - Date.now()) / 1000);
    // The floor never exceeds what the server allows this problem — on the
    // last problem of a capped plan the allowance can be under 15 s.
    return Math.max(Math.min(RESUME_FLOOR_SECS, allowed), Math.min(allowed, left));
  };

  const _format = (secs) => {
    const clamped = Math.max(0, Math.round(secs));
    return String(Math.floor(clamped / 60)).padStart(2, "0") + ":" +
      String(clamped % 60).padStart(2, "0");
  };

  /* The plan's own remainder, beside the problem clock: "12:34 · 2h 41m
     left". The server's number minus what this problem has used so far, so it
     moves with the clock rather than jumping once per answer. Empty when the
     status carries no plan (a pre-plan build). */
  const _planLeft = () => {
    const plan = _plan();
    const remaining = Number(plan?.remaining_secs);
    if (!Number.isFinite(remaining)) return "";
    const used = elapsedSecs() || 0;
    const left = Math.max(0, remaining - used);
    const h = Math.floor(left / 3600);
    const m = Math.floor((left % 3600) / 60);
    return h > 0 ? `${h}h ${String(m).padStart(2, "0")}m left` : `${m}m left`;
  };

  /* The clock lives on the NOTCH TAB (.practice-notch-tab in index.html), in
     the slot beside the session countdown it takes turns with. Seth,
     2026-08-23: "for the placement test, it has the problem that the timer is
     on the left next to the title thing instead of being on the tab."

     It was in .question-number-row until then, and in #cold-start-badge before
     that. Both were the wrong shape for the same reason: a countdown is a
     session control, and every other session control is on the tab. What has
     not changed is that it sits inside `.practice-container`, so it still
     re-parents onto the Placement page with the workspace and there is still
     no second copy to keep in sync.

     🔴 STATIC ONLY — no `createElement` fallback any more. The old one existed
     because the anchor had moved once already; what it actually bought was a
     chip minted into whatever row happened to be on screen, unstyled by the
     notch's rules and invisible to `check_infotips`. If the element is gone
     from index.html the right outcome is no countdown and a failing watch, not
     a second one built at runtime. */
  const _chip = () => {
    if (chip && chip.isConnected) return chip;
    chip = document.getElementById("placement-timer");
    return chip;
  };

  /* The tab shows ONE clock. notch-menu.js decides which by asking
     `isRunning()`, so every show/hide here has to poke it — nothing else
     observes this module, and the session clock would otherwise sit next to
     the probe's countdown, idling at 02:00, reading as a second timer. */
  const _syncNotch = () => window.PracticeNotch?.syncClock?.();

  const _paint = () => {
    const el = _chip();
    if (!el) return;
    const left = _remaining();
    const total = _planLeft();
    el.textContent = total ? `${_format(left)} · ${total}` : _format(left);
    el.classList.remove("hidden");
    /* The SESSION clock's low class, not one of this module's own. The element
       carries `.practice-notch-clock`, so the placement's last 30 seconds are
       the same red in the same slot as a practice question's — Seth wants the
       two indistinguishable, and a private `--low` was the last thing making
       them different. */
    el.classList.toggle("practice-notch-clock--low", left <= 30);
    _syncNotch();
  };

  const _hideChip = () => {
    const el = chip && chip.isConnected ? chip : null;
    el?.classList.add("hidden");
    _syncNotch();
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
    if (!startedAt) startedAt = Date.now();
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
    startedAt = 0;
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
    startedAt = 0;
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
    elapsedSecs,
    isRunning: () => !!interval,
  };
})();
window.PlacementTimer = PlacementTimer;
