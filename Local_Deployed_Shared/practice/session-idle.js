/* ================================================================
   THE IDLE SCREEN — a dial, a sentence, and one way back in

   WHAT REPLACED WHAT
     "Set up your session" was three inputs and a total-time estimate —
     questions, answer time, review time — and it went on 2026-08-23. What
     took its place is how far through the map the learner is ("N% ready
     for the ARENA curriculum", Seth's words) and a button that starts the
     next block.

     ONE of those three inputs came back on 2026-08-28, as a picker rather
     than a text field: the time each QUESTION gets, including "No limit"
     (practice/session-clock.js). The other two did not, and must not — a
     block still has no question quota and no length, which is why there
     is no End session button for them to imply.

   🔴 THE BUTTON IS A PROXY, not a third implementation of "start".
     #session-resume-btn and #session-start-btn are still the real
     controls and still carry timer.js's handlers; they are off the
     screen, not gone. Which one Continue forwards to is the ONLY
     decision made here, and it is made from `hasPausedSession()` —
     timer.js's own answer — rather than from anything this file tracks.
     The same pattern as the notch (practice/notch-menu.js), for the same
     reason: two copies of what resuming means is how they drift apart.

   WHEN IT REDRAWS
     `#page-practice.session-idle` is the class timer.js adds on pause and
     removes on start/resume, and it is the one fact that says this screen
     is on. A MutationObserver on it is what triggers a re-read — a
     readiness number that was computed once at load would still be the
     one from before the session that just ended.
   ================================================================ */

(function initSessionIdle() {
  const page = document.getElementById("page-practice");
  const dial = document.getElementById("readiness-dial");
  const pctEl = document.getElementById("readiness-pct");
  const captionEl = document.getElementById("readiness-caption");
  const detailEl = document.getElementById("readiness-detail");
  const continueBtn = document.getElementById("session-continue-btn");
  if (!page) return;

  /* Only one read may be in flight. The observer fires on any class change to
     #page-practice — several of them arrive in the same frame when a session
     ends — and each read walks 63 concepts. */
  let reading = false;

  /* 🔴 `PracticeSession` is a top-level `const` in practice/timer.js, and a
     classic script's top-level `const` does NOT become a property of `window`
     — `window.PracticeSession` is undefined and every optional-chained call
     through it silently answers nothing. (The same trap has bitten
     `PracticeAPI`.) Resolved through the lexical binding, with a typeof guard
     so a page that never loaded timer.js still parses. */
  const _session = () =>
    typeof PracticeSession !== "undefined" ? PracticeSession : window.PracticeSession;

  const _paint = (info) => {
    if (!dial || !pctEl) return;
    if (!info) {
      /* The registry could not be read. NOT 0% — that is a claim about the
         learner, and this is a claim about the network. */
      dial.style.setProperty("--dd-ready-pct", "0");
      dial.classList.add("readiness-dial--unknown");
      pctEl.textContent = "—";
      if (captionEl) captionEl.textContent = "readiness unavailable right now";
      if (detailEl) detailEl.textContent = "";
      return;
    }
    dial.classList.remove("readiness-dial--unknown");
    dial.style.setProperty("--dd-ready-pct", String(info.pct));
    pctEl.textContent = `${info.pct}%`;
    dial.setAttribute(
      "aria-label",
      `${info.pct} percent ready for the ARENA curriculum`,
    );
    if (captionEl) captionEl.textContent = "ready for the ARENA curriculum";
    /* 🔴 THE WORDS COME FROM readiness.js TOO, not just the number. This
       screen and the placement results card show the same figure, and two
       hand-written captions for one number is how they start disagreeing
       again in a smaller way — "8 concepts mastered" here against "12 of 63
       measured" there, about the same learner in the same minute. */
    if (detailEl) detailEl.textContent = window.PracticeReadiness.detail(info);
  };

  const refresh = () => {
    if (reading || !window.PracticeReadiness) return;
    reading = true;
    window.PracticeReadiness.read()
      .then(_paint)
      .catch(() => _paint(null))
      .finally(() => {
        reading = false;
      });
  };

  const _resumeBtn = () => document.getElementById("session-resume-btn");

  /* A paused session whose saved question can no longer be rebuilt. timer.js
     says so by DISABLING #session-resume-btn and writing the reason into
     #session-resume-summary — and both of those are off the screen now, so
     this screen has to carry the news or the learner presses Continue and
     watches nothing happen. That is exactly what it did the first time. */
  const _resumeRefused = () => {
    const btn = _resumeBtn();
    return !!_session()?.hasPausedSession?.() && !!btn && btn.disabled;
  };

  /* Resume if there is something paused, start a fresh block otherwise, and
     if the paused one has been refused, throw it away first — `discard` is
     what clears the snapshot, and starting on top of a snapshot nobody can
     resume leaves it there to refuse again next time.

     The hidden buttons are CLICKED rather than their handlers called: their
     `disabled` state is the refusal, and calling through would walk past it. */
  const _continue = () => {
    const session = _session();
    const resumeBtn = _resumeBtn();
    const startBtn = document.getElementById("session-start-btn");
    if (session?.hasPausedSession?.()) {
      if (resumeBtn && !resumeBtn.disabled) {
        resumeBtn.click();
        return;
      }
      const discardBtn = document.getElementById("session-discard-btn");
      if (discardBtn) discardBtn.click();
    }
    if (startBtn && !startBtn.disabled) startBtn.click();
  };

  /* What the button SAYS depends on which of the three things it is about to
     do, and the differences matter: "Continue practicing" over a paused
     question is a promise to put that question back, "Start practicing" over
     nothing at all is a promise to begin, and over a refused resume it is a
     promise to drop what was saved. All three are true; saying the wrong one
     is not. Never DISABLED — there is always one of the three to do, and a
     dead button on a screen with nothing else on it is a dead end. */
  const _syncLabel = () => {
    if (!continueBtn) return;
    const paused = !!_session()?.hasPausedSession?.();
    const refused = _resumeRefused();
    continueBtn.disabled = false;
    continueBtn.textContent = refused
      ? "Start a new question"
      : paused
        ? "Continue practicing"
        : "Start practicing";
    const summary = document.getElementById("session-summary");
    if (summary && refused) {
      const why = document.getElementById("session-resume-summary");
      summary.textContent =
        (why && why.textContent.trim()) ||
        "The saved question is no longer available.";
      summary.classList.remove("hidden");
    }
  };

  /* ── THE PER-QUESTION CLOCK PICKER ──────────────────────────────
     Seth, 2026-08-28: "I can change the amount of time that I have per problem
     before I start the practice ... Or I can disable the timer entirely."

     🔴 THIS FILE DECIDES NOTHING ABOUT TIME. The presets, the default and the
     stored choice all live in practice/session-clock.js, and the clock that
     enforces them lives in practice/timer.js. What happens here is drawing and
     one `set` call — the same proxy discipline as the Continue button above,
     for the same reason: a list of minutes written twice is a picker that
     offers an option the countdown does not honour.

     Buttons rather than a <select>: five options that have to be readable and
     hittable at a glance on the one screen between blocks, and a radiogroup is
     what says "these are alternatives" to a screen reader. */
  const clockPicker = document.getElementById("question-clock-picker");
  const clockNote = document.getElementById("question-clock-note");
  const _clockPrefs = () => window.SessionClock || null;

  /* 🔴 THE NOTE STATES THE CONSEQUENCE, not the setting — the chosen number is
     already on the button that is pressed, and repeating it teaches nothing.
     What a learner cannot see from the buttons is that the allowance is per
     STEP (answering and reviewing each get it), and that the placement test is
     not on this clock at all. Both are said here rather than in a tooltip: the
     second one is the difference between "the app ignored my setting" and "the
     test is timed the same for everyone". */
  const _clockNoteText = (option) => {
    if (!option) return "";
    if (option.secs === null) {
      return "No countdown. A question ends when you submit it, and nothing " +
        "is auto-submitted. The placement test keeps its own 2:00 per question.";
    }
    return `${option.label} to answer, then ${option.label} to review — for ` +
      "every question. The placement test keeps its own 2:00 per question.";
  };

  /* Draw once, then only flip the state. Re-rendering the buttons on every
     change would rip the pressed one out from under the click that changed it
     and lose keyboard focus with it. */
  const _paintClock = () => {
    const prefs = _clockPrefs();
    if (!clockPicker || !prefs) return;
    const currentId = prefs.currentId();
    if (!clockPicker.childElementCount) {
      prefs.OPTIONS.forEach((option) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "question-clock-opt";
        btn.dataset.clockId = option.id;
        btn.setAttribute("role", "radio");
        btn.textContent = option.label;
        btn.addEventListener("click", () => {
          prefs.set(option.id);
          _paintClock();
        });
        clockPicker.appendChild(btn);
      });
    }
    clockPicker.querySelectorAll("[data-clock-id]").forEach((btn) => {
      const on = btn.dataset.clockId === currentId;
      btn.classList.toggle("question-clock-opt--on", on);
      /* `aria-checked` is the state a radio actually exposes; the class is
         only paint. Both, because neither reader can see the other's. */
      btn.setAttribute("aria-checked", on ? "true" : "false");
      /* One tab stop for the group, arrow keys inside it — the standard radio
         pattern, and what stops five presets costing five tabs to walk past. */
      btn.tabIndex = on ? 0 : -1;
    });
    if (clockNote) clockNote.textContent = _clockNoteText(prefs.current());
    /* The topbar clock shows the allowance the NEXT question will get while
       nothing is running (practice/notch-menu.js `_syncClock`), and it is
       driven by an observer on #session-status-row — which a change made on
       THIS screen never touches. Without this call the picker said 10:00 while
       the clock above it still read 02:00 until the next question rendered. */
    window.PracticeNotch?.syncClock?.();
  };

  /* Arrow keys move the CHOICE, not just the focus: for a radiogroup the two
     are the same gesture, and a focus-only implementation leaves a learner
     pressing → and wondering why nothing changed. */
  if (clockPicker) {
    clockPicker.addEventListener("keydown", (event) => {
      const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
      const prefs = _clockPrefs();
      if (!step || !prefs) return;
      event.preventDefault();
      const ids = prefs.OPTIONS.map((o) => o.id);
      const at = ids.indexOf(prefs.currentId());
      const next = ids[(at + step + ids.length) % ids.length];
      prefs.set(next);
      _paintClock();
      clockPicker.querySelector(`[data-clock-id="${next}"]`)?.focus();
    });
  }

  /* Another tab of the same app is another writer of the same key. The picker
     is cheap to repaint and a stale one is a screen that lies about the clock
     the next question will get. */
  window.addEventListener("storage", _paintClock);

  if (continueBtn) continueBtn.addEventListener("click", _continue);

  if (typeof MutationObserver === "function") {
    new MutationObserver(() => {
      if (!page.classList.contains("session-idle")) return;
      refresh();
      _syncLabel();
      /* The choice can have been changed in another tab while a block ran
         here; this screen is where it is stated, so it re-reads on the way
         back in rather than showing what it drew before the block. */
      _paintClock();
    }).observe(page, { attributes: true, attributeFilter: ["class"] });

    /* The resume button's own state changes underneath this screen — timer.js
       disables it when the saved question turns out to be unavailable, which
       happens asynchronously, after the panel is already up. */
    const resumePanel = document.getElementById("session-resume-panel");
    if (resumePanel) {
      new MutationObserver(_syncLabel).observe(resumePanel, {
        attributes: true,
        attributeFilter: ["class", "disabled"],
        /* `characterData` and `childList` too: the REASON a resume was refused
           is written as text into #session-resume-summary, and an attribute
           filter alone never sees it. */
        characterData: true,
        childList: true,
        subtree: true,
      });
    }
  }

  /* The practice page starts idle, so paint once at load rather than waiting
     for the first pause. index.html loads this AFTER timer.js in the same
     ordered block of classic scripts, so `hasPausedSession` already knows
     about a snapshot restored from localStorage by the time this runs. */
  refresh();
  _syncLabel();
  _paintClock();

  /* And again once the DEFER scripts have run. concept-graph/lesson-graph.js
     is deferred, so `window.deltaKcReadinessInfo` does not exist during the
     paint above; DOMContentLoaded is the first moment it is guaranteed to.
     Nothing is lost if the first read already succeeded — `read()` is a pure
     re-read of state this file does not own. */
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", refresh, { once: true });
  }

  window.PracticeIdleScreen = {
    refresh,
    syncLabel: _syncLabel,
    paintClock: _paintClock,
  };
})();
