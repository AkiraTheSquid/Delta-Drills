/* ================================================================
   THE IDLE SCREEN'S ONE WAY BACK IN

   The Learner Home between blocks is measured learning and two graphs
   (practice/xp-panel.js) and this button. The readiness dial that stood
   here from 2026-08-23 went on 2026-09-26 (Seth: "remove ... the %
   readiness"), and with it the per-block read of every concept.

   There is nothing to set up: each problem brings its own clock
   (practice/session-clock.js), and a block has no question quota and no
   length, which is why there is no End session button either.

   🔴 THE BUTTON IS A PROXY, not a third implementation of "start".
     #session-resume-btn and #session-start-btn are still the real
     controls and still carry timer.js's handlers; they are off the
     screen, not gone. Which one Continue forwards to is the ONLY
     decision made here, and it is made from `hasPausedSession()` —
     timer.js's own answer — rather than from anything this file tracks.
     The same pattern as the notch (practice/notch-menu.js), for the same
     reason: two copies of what resuming means is how they drift apart.
   ================================================================ */

(function initSessionIdle() {
  const page = document.getElementById("page-practice");
  const continueBtn = document.getElementById("session-continue-btn");
  if (!page) return;

  /* 🔴 `PracticeSession` is a top-level `const` in practice/timer.js, and a
     classic script's top-level `const` does NOT become a property of `window`
     — `window.PracticeSession` is undefined and every optional-chained call
     through it silently answers nothing. (The same trap has bitten
     `PracticeAPI`.) Resolved through the lexical binding, with a typeof guard
     so a page that never loaded timer.js still parses. */
  const _session = () =>
    typeof PracticeSession !== "undefined" ? PracticeSession : window.PracticeSession;

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

  if (continueBtn) continueBtn.addEventListener("click", _continue);

  if (typeof MutationObserver === "function") {
    new MutationObserver(() => {
      if (!page.classList.contains("session-idle")) return;
      _syncLabel();
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

  /* The practice page starts idle, so label once at load rather than waiting
     for the first pause. index.html loads this AFTER timer.js in the same
     ordered block of classic scripts, so `hasPausedSession` already knows
     about a snapshot restored from localStorage by the time this runs. */
  _syncLabel();

  window.PracticeIdleScreen = { syncLabel: _syncLabel };
})();
