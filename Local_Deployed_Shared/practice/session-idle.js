/* ================================================================
   THE IDLE SCREEN — a dial, a sentence, and one way back in

   WHAT REPLACED WHAT
     "Set up your session" was three inputs and a total-time estimate.
     The learner does not set the clock any more (practice/timer.js holds
     both allowances as constants), so the only thing left worth putting
     on a screen the learner sees between blocks is how far through the
     map they are — "N% ready for the ARENA curriculum", Seth's words,
     2026-08-23 — and a button that starts the next one.

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
    if (detailEl) {
      /* A bare 0% with no denominator reads as a grade. The count says what
         the percentage is a percentage OF, and the not-started case says it
         in words instead — nothing has been measured, so nothing failed. */
      /* `borrowed` are concepts sitting above the mastery threshold on a
         number that is the LESSON's, not this concept's — readiness.js keeps
         them out of the percentage. Naming them is the difference between a
         number that looks stuck and one the learner can act on. */
      const borrowed = info.borrowed
        ? `, ${info.borrowed} nearly there`
        : "";
      detailEl.textContent = info.measured
        ? `${info.mastered} of ${info.total} concepts mastered${borrowed}`
        : `${info.total} concepts to go — none measured yet`;
    }
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

  if (continueBtn) continueBtn.addEventListener("click", _continue);

  if (typeof MutationObserver === "function") {
    new MutationObserver(() => {
      if (!page.classList.contains("session-idle")) return;
      refresh();
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

  /* The practice page starts idle, so paint once at load rather than waiting
     for the first pause. index.html loads this AFTER timer.js in the same
     ordered block of classic scripts, so `hasPausedSession` already knows
     about a snapshot restored from localStorage by the time this runs. */
  refresh();
  _syncLabel();

  /* And again once the DEFER scripts have run. concept-graph/lesson-graph.js
     is deferred, so `window.deltaKcReadinessInfo` does not exist during the
     paint above; DOMContentLoaded is the first moment it is guaranteed to.
     Nothing is lost if the first read already succeeded — `read()` is a pure
     re-read of state this file does not own. */
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", refresh, { once: true });
  }

  window.PracticeIdleScreen = { refresh, syncLabel: _syncLabel };
})();
