/* ================================================================
   THE NOTCH — session controls, off the page until asked for

   The session row used to sit above the question: progress, phase, a
   countdown, Pause & exit and End session, all of it on screen for every
   second of every question. Four pieces of chrome for two buttons anyone
   presses twice a session, in the band of the page the learner reads the
   question in.

   So the buttons moved into a notch straddling the seam between the two
   panels — a tab with three dots, and a menu that opens under it.

   The COUNTDOWN came back with the second pass. Hiding the row took the clock
   off the screen while leaving it running: answer time still auto-submitted
   with nothing counting down to it. The tab carries it now — mirrored, never
   re-implemented (see `_syncClock`).

   🔴 This is a PROXY, not a second implementation. The real handlers live in
   practice/timer.js, bound to #session-pause-btn and #session-end-btn, and
   they carry state this file must not duplicate: whether a pause is safe
   right now (`sessionPauseBtn.disabled` flips mid-question while grading is
   in flight), what a pause writes to the resume snapshot, what "ended" does
   to the block. Clicking the hidden button is what keeps those two paths
   from drifting. The row is hidden by CSS (styles/practice/timer.css); the
   buttons in it are still live, still bound, and still the only place the
   behaviour is written down.
   ================================================================ */

(function () {
  const notch = document.getElementById("practice-notch");
  const clock = document.getElementById("practice-notch-clock");
  const btn = document.getElementById("practice-notch-btn");
  const menu = document.getElementById("practice-notch-menu");
  const pauseItem = document.getElementById("practice-notch-pause");
  /* Finish the placement early — placement only, hidden for a session. Calls
     DiagnosticPage.finish() directly rather than clicking the card's button:
     that button is `.hidden` while a probe is on screen (exactly when this
     row is needed) and a hidden button's click still fires, but the two-click
     arm/confirm then shows its label nowhere. This row carries the label. */
  const finishItem = document.getElementById("practice-notch-finish");
  /* The square, to the LEFT of the clock. Same destination as the menu's
     Pause item — one behaviour, two ways to reach it, neither of them a copy
     of what pausing does. There is no End counterpart: the session model has
     no end (practice/timer.js). */
  const stopBtn = document.getElementById("practice-notch-stop");
  const note = document.getElementById("practice-notch-note");
  if (!notch || !btn || !menu) return;

  /* The hidden originals. Looked up per click, not cached: the practice tab
     is one long-lived document, but a lookup that fails once and is cached
     forever is a menu that silently stops working. */
  /* 🔴 `PracticeSession` is a top-level `const` in practice/timer.js, and a
     classic script's top-level `const` does NOT become a property of `window`
     — `window.PracticeSession` is undefined and every optional-chained call
     through it silently answers nothing. (The same trap has bitten
     `PracticeAPI`.) Resolved through the lexical binding, with a typeof guard
     so a page that never loaded timer.js still parses. */
  const _session = () =>
    typeof PracticeSession !== "undefined" ? PracticeSession : window.PracticeSession;

  const _pauseBtn = () => document.getElementById("session-pause-btn");
  /* The placement's own pause control (index.html, owned by
     practice/diagnostic-page.js). A placement runs OUTSIDE a session, so
     `#session-pause-btn` is inert for the whole test — proxying to it was what
     left the square and the menu row dead while a probe was on the clock. */
  const _placementPauseBtn = () => document.getElementById("placement-pause-btn");

  /* WHICH pause this notch is offering right now. The placement wins whenever
     its clock is up, because that is the clock the notch is showing (see
     `_syncClock`) and pausing has to mean the thing the learner is looking at.
     Returns null when neither is available, which is what disables both
     controls. */
  function _pauseTarget() {
    if (_placementOnClock()) return _placementPauseBtn();
    return _sessionOpen() ? _pauseBtn() : null;
  }

  /* A session is running when timer.js has unhidden its row. That class is
     the single fact both this menu and the row itself read — see timer.js
     `sessionStatusRow.classList.remove("hidden")` at session start and
     `.add("hidden")` at finish. */
  function _sessionOpen() {
    const row = document.getElementById("session-status-row");
    return !!row && !row.classList.contains("hidden");
  }

  /* The clock on the tab is a MIRROR of #session-countdown, not a timer.
     timer.js owns the only clock there is — the one that auto-submits — and a
     second interval here would drift from it within a question and disagree
     with it at the moment that matters. So this copies three facts and
     computes none of them: the text, whether the session is running at all,
     and which phase's time is being counted.

     `--low` is timer.js's own last-30-seconds class, and it is applied only
     while ANSWERING: review time running out costs nothing, so painting it
     like a deadline would teach the wrong urgency. */
  /* 🔴 THE TAB SHOWS ONE CLOCK. The placement test runs OUTSIDE a session —
     starting it calls PracticeSession.finish("placement") — so `_sessionOpen()`
     is false for the whole test and the session clock would sit beside the
     probe's countdown greyed at 02:00, which reads as a second timer that has
     stopped. While a probe is being timed the placement clock is the clock.

     Read from the ELEMENT, not from `PlacementTimer.isRunning()`: the two
     disagree for the moments that matter — `pauseForGrading` clears the
     interval and hides the chip, `_expire` clears the interval before clicking
     Submit — and the element's own `.hidden` is what is actually on screen.
     placement-timer.js calls back in here (`_syncNotch`) on every show and
     hide, so this runs at each of those transitions. */
  function _placementOnClock() {
    const el = document.getElementById("placement-timer");
    return !!el && !el.classList.contains("hidden");
  }

  function _syncClock() {
    if (!clock) return;
    const open = _sessionOpen();
    const tab = document.getElementById("practice-notch-tab");
    const srPhase = document.getElementById("practice-notch-phase");
    /* 🔴 THE NOTCH IS OFF THE SCREEN BETWEEN PROBLEMS (Seth, 2026-09-11: "get
       rid of the notch at the top … showing that notch timer only after
       starting a problem that has the leetcode style interface"). It is on
       screen for a running session and for a placement — a placement stays
       up between its probes too, because the chip goes away mid-grade on a
       test that is very much running (see `_syncItems`). The idle allowance
       below is still WRITTEN, so the tab is right the instant it appears;
       it is just not shown. This reverses 2026-08-23 ("even after you exit
       the session, the notch should still be there") on Seth's word.

       A notebook exercise on its own clock (practice/exercise-timer.js) is
       NOT a problem in this sense: its clock draws in the notebook, over the
       buttons that started it, and never comes here. */
    const placementActive = _placementOnClock() || window.DiagnosticPage?.isRunning?.() === true;
    if (notch) notch.classList.toggle("is-idle", !open && !placementActive);
    if (_placementOnClock()) {
      clock.classList.add("hidden");
      /* The placement's phase is not the session's, and the session's is
         stale — nothing is answering or reviewing in a block that ended when
         the test started. Both readers get the placement's own words.

         🔴 THIS TOOLTIP IS NOW THE ONLY PLACE THE PROGRESS IS SAID. The
         placement page's status line and progress bar are hidden while a probe
         is on screen (styles/practice/diagnostic.css) — Seth wants a probe to
         look exactly like a practice question — so "3 of at most 14" lives
         here, on the surface a practice session already uses for its phase.
         `progressLabel()` returns "" when nothing is running, which is the
         same as not having asked. */
      const where = window.DiagnosticPage?.progressLabel?.() || "";
      const rule = "Each concept has its own clock, the same for every learner.";
      if (tab) tab.title = where ? `${where} · ${rule}` : `Placement test — ${rule}`;
      if (srPhase) srPhase.textContent = where || "Placement question";
      return;
    }
    clock.classList.remove("hidden");
    clock.classList.toggle("practice-notch-clock--idle", !open);
    if (!open) {
      /* 🔴 The clock STAYS between sessions (Seth, 2026-08-23: "even after you
         exit the session, the notch should still be there ... with the amount
         of time that you would want to set"). What it shows is the allowance
         the next question will get — read from timer.js, never computed here —
         greyed, so it reads as the rule rather than as a running clock.

         Nothing is being counted, so nothing may still SAY it is: the phase
         classes, the tooltip and the screen-reader phase all come off, because
         a tab that still reads "Reviewing" is describing a session that
         ended. */
      clock.classList.remove(
        "practice-notch-clock--review",
        "practice-notch-clock--low",
      );
      const idle = _session()?.idleClockText?.() || "";
      if (clock.textContent !== idle) clock.textContent = idle;
      if (tab) tab.title = "Each problem has its own clock, set by its concept.";
      if (srPhase) srPhase.textContent = "";
      return;
    }
    const src = document.getElementById("session-countdown");
    const row = document.getElementById("session-status-row");
    const review = !!row && row.classList.contains("session-status--review");
    const text = src ? src.textContent.trim() : "";
    if (clock.textContent !== text) clock.textContent = text;
    clock.classList.toggle("practice-notch-clock--review", review);
    clock.classList.toggle(
      "practice-notch-clock--low",
      !review && !!src && src.classList.contains("session-countdown--low"),
    );
    /* The phase WORD is off the screen with the rest of the row. It rides in
       two places, for two readers: the tab's tooltip, so the clock's colour is
       not the only thing carrying it for a sighted learner, and a clipped
       span, so it is carried at all for a screen reader — `--review` and
       `--low` are colour, and colour is not in the accessibility tree. */
    const phase = document.getElementById("session-phase");
    const phaseText = phase ? phase.textContent.trim() : "";
    if (tab) tab.title = phaseText;
    if (srPhase && srPhase.textContent !== phaseText) {
      srPhase.textContent = phaseText;
    }
  }

  function _syncItems() {
    const open = _sessionOpen();
    const placement = _placementOnClock();
    const target = _pauseTarget();
    /* Refused for one of two reasons, and they are not the same: nothing to
       pause at all, or a question that is mid-grade. timer.js owns the second
       and writes the reason into the session button's `title`; the placement's
       control carries no such state — a probe on the clock can always be put
       down. */
    const refused = !target || target.disabled;
    /* 🔴 THE PLACEMENT'S PAUSE IS REAL NOW (Seth, 2026-09-06). This used to
       read `!open || …`, which dimmed both controls for the whole of a test —
       the test runs outside a session by design, so `open` is false throughout
       — and the tooltip told the learner the test "can't be paused" while a
       clock counted down in front of them. It was true when it was written and
       it is not any more: practice/diagnostic-page.js `pause()` stops the probe
       clock, drops the probe and leaves the placement ACTIVE and resumable.
       The comparability rule it protected is unharmed — the probe comes back
       whole, on a fresh 2:00 — see that function's note. */
    /* 🔴 "No session running." IS STILL A LIE MID-GRADE. `pauseForGrading`
       hides the placement chip while a probe's answer is in flight, so
       `_placementOnClock()` goes false for those seconds on a test that is very
       much running — and the fallback reason would have gone back to claiming
       nothing was happening. Ask the page, not the chip, for whether a
       placement exists; the chip only says whether it is being timed. */
    const placementActive = placement || window.DiagnosticPage?.isRunning?.() === true;
    /* 🔴 AND WHICH KIND OF "not right now" IT IS. Between probes there is
       nothing to pause and the way on is a button on the page; mid-grade there
       IS a probe, it is just not interruptible. Both states hide the chip, so
       the chip cannot tell them apart — the PROBE can. Same script-scope-const
       trap as everywhere else: `PracticeAPI` is not on `window`. */
    const _api = typeof PracticeAPI !== "undefined" ? PracticeAPI : window.PracticeAPI;
    const probeHeldBack = !!_api?.currentQuestion?.diagnostic_active;
    const why = placement
      ? "Pause the placement test. It stays where it is; you come back to a fresh 2:00 on this question."
      : placementActive
        ? probeHeldBack
          ? "Pause becomes available when this placement question finishes."
          : "Nothing to pause — load the next placement question to carry on."
        : open && target
          ? target.title || ""
          : "No session running.";
    if (pauseItem) {
      pauseItem.disabled = refused;
      pauseItem.title = why;
    }
    if (finishItem) {
      // …and only once something has been answered: with zero answers the
      // server treats /finish as a decline, not a finish.
      const answered = Number(_api?.lastDiagnosticStatus?.probes_done) || 0;
      finishItem.classList.toggle("hidden", !placementActive || answered < 1);
      finishItem.title = "End the placement test now. Your concepts are estimated from what you have answered so far; the question on screen is dropped, not counted.";
    }
    if (stopBtn) {
      stopBtn.disabled = refused;
      stopBtn.title = why;
    }
    /* The note explains the idle clock, and a placement is not idle. */
    if (note) note.classList.toggle("hidden", open || placement);
  }

  function _open() {
    _syncItems();
    menu.classList.remove("hidden");
    btn.setAttribute("aria-expanded", "true");
    notch.classList.add("is-open");
  }

  function _close() {
    menu.classList.add("hidden");
    btn.setAttribute("aria-expanded", "false");
    notch.classList.remove("is-open");
  }

  function _isOpen() {
    return !menu.classList.contains("hidden");
  }

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (_isOpen()) _close();
    else _open();
  });

  /* Anywhere else closes it, including inside the panels — the menu floats
     over the question and the editor, and a click meant for either of them
     should not have to be spent dismissing this first. */
  document.addEventListener("click", (e) => {
    if (!_isOpen()) return;
    if (notch.contains(e.target)) return;
    _close();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && _isOpen()) {
      _close();
      btn.focus();
    }
  });

  function _proxy(target) {
    return (e) => {
      e.stopPropagation();
      const el = target();
      if (!el || el.disabled) return;
      _close();
      /* The real button, with the real handler. Not a copy of what it does. */
      el.click();
    };
  }

  /* `_pauseTarget`, not `_pauseBtn`: which pause is on offer is decided per
     click, because a placement can start or finish under an open menu. */
  if (pauseItem) pauseItem.addEventListener("click", _proxy(_pauseTarget));
  if (finishItem) {
    /* No timer here — the notch's watch forbids a second clock in this file.
       DiagnosticPage.finish() owns the two-click arm (and its lapse) and
       writes the armed label onto THIS row as well as the card's button, so
       the row only forwards the click and keeps the menu open for the second
       one. Closing happens when the finish actually lands (the page re-renders
       and the placement clock goes away). */
    finishItem.addEventListener("click", async (e) => {
      e.stopPropagation();
      const page = window.DiagnosticPage;
      if (!page?.finish) return;
      const done = await page.finish();
      if (done) _close();
    });
  }
  if (stopBtn) stopBtn.addEventListener("click", _proxy(_pauseTarget));

  /* Session state changes underneath an open menu — a question submits, the
     block ends on its own clock — so the items re-read it rather than
     trusting what they were told when the menu opened. */
  const _row = document.getElementById("session-status-row");
  if (_row && typeof MutationObserver === "function") {
    new MutationObserver(() => {
      _syncClock();
      /* 🔴 `_syncItems` unconditionally, not only when the menu is open. The
         square is ON the tab and always visible, so its enabled state has to
         track the session even with the menu shut — it used to be enough to
         re-read only what an open menu was showing. */
      _syncItems();
    }).observe(_row, {
      attributes: true,
      /* `class` is the session opening and closing, and the review tint;
         `disabled` and `title` are timer.js flipping the pause button
         mid-question while a grade is in flight (timer.js `_setPhase`).
         Watching only `class` left an open menu showing the state it had when
         it opened — greyed out after grading finished, or live and silently
         doing nothing before it did. */
      attributeFilter: ["class", "disabled", "title"],
      /* The countdown is a TEXT NODE that timer.js rewrites once a second.
         `characterData` alone does not see it — assigning `textContent`
         REPLACES the node rather than editing it, which is a childList
         record — so both are needed to mirror a ticking clock. */
      characterData: true,
      childList: true,
      subtree: true,
    });
  }

  /* The row can already be open when this runs — a resumed session unhides it
     from timer.js's own init, and load order between the two is not a thing
     to depend on. */
  _syncClock();
  _syncItems();

  window.PracticeNotch = {
    close: _close,
    /* Kept for callers that need to place themselves around this tab. The
       stage ladder used to be one — its percentage reading hung under the bar
       and could land here — but that reading is a tab ON the bar now, inside
       the strip, and nothing measures against the notch any more. */
    el: notch,
    /* Exposed for tests and for anything that changes the session out of
       band; the mirror is otherwise driven entirely by the observer.

       🔴 BOTH HALVES, despite the name. placement-timer.js calls this every
       time its clock appears or disappears, and the square's tooltip depends
       on which clock is up ("The placement test can't be paused" vs "No
       session running") — the observer that normally pairs them watches
       #session-status-row, and a placement changes nothing in there. Calling
       only `_syncClock` left the square explaining a state that ended two
       questions ago. */
    syncClock: () => {
      _syncClock();
      _syncItems();
    },
  };
})();
