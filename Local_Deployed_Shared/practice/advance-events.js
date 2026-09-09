/* ================================================================
   QUESTION ADVANCE EVENTS — Next, placement lifecycle, Skip

   Loaded directly after events.js. That file owns the shared
   `_loadNextPracticeQuestion`; this focused classic script owns controls that
   invoke it, keeping the public/global browser contract unchanged.
   ================================================================ */

/* 🪦 THE UNLOCK INTERSTITIAL IS GONE — 2026-09-09. Seth, on a screenshot of it:
   "this should be deleted. stop routing to collab. same content goes in web
   app."

   Next used to hand the click to `ArenaUnlock.tryShow` first, which put a
   full-page card between the learner and their next question whenever a
   procedural or composite drill crossed its unlock gate. The card's whole
   offer was Open in Colab ↗ — leave the app, work the exercise in a notebook,
   come back and self-rate. Three things had gone wrong with that:

     • The app can teach these concepts itself now. The exercise on the card in
       the screenshot targets `Einops: Reduce` and `Numpy: Vectorization and
       broadcasting`; both are authored KCs with full four-rung ladders in the
       bank, reachable from the readiness screen and from "Drill this concept"
       on the ARENA notebook page. Sending someone to Colab for a concept the
       editor can drill is the app arguing with itself.
     • ARENA exercises themselves are IN the app since 2026-09-06 — the
       notebook tab renders them, and practice/exercise-timer.js times an
       attempt where they are. Colab is no longer where the work happens.
     • The rating it collected was a self-report on a notebook the app never
       saw, and it fed the mastery model directly: the same screenshot shows
       one click taking both atoms from 92.0% to 54.8%.

   So Next now just loads the next question. `ArenaUnlock.showFor` is still
   there and still reachable, but only from Targeted Practice's explicit
   "Practice this problem" button — a trip the learner asks for, not one the
   app takes on their behalf mid-block. */
nextProblemBtn.addEventListener("click", async () => {
  await _advancePlacementOrFinish();
});

// After a probe is recorded, refresh placement results when its stopping rule
// fired. Returns true only when no further probe should be loaded.
async function _notifyIfPlacementDone(knownStatus = null) {
  try {
    const status = knownStatus || await PracticeAPI.diagnosticStatus();
    if (!status || status.unavailable || status.active || !status.completed_at) return false;
    if (typeof loadBackendAdaptiveState === "function") {
      await loadBackendAdaptiveState();
    }
    refreshPlacementStartBtn().catch(() => {});
    window.DiagnosticPage?.refresh();
    emitPracticeStateChanged();
    return true;
  } catch (_) {
    /* best-effort — never blocks the practice flow */
    return false;
  }
}

/* A placement answer may be the stopping-rule answer. The backend then marks
   the diagnostic complete, so its ordinary /next-question fallback is a real
   practice drill. Active → next probe. Complete → Placement results. */
async function _advancePlacementOrFinish(knownStatus = null) {
  const q = PracticeAPI.currentQuestion;
  if (!q?.diagnostic_active) {
    await _loadNextPracticeQuestion();
    return;
  }
  const status = knownStatus || await PracticeAPI.diagnosticStatus();
  if (await _notifyIfPlacementDone(status)) return;
  await _loadNextPracticeQuestion();
}

// The placement start button — explicit entry from its own page. The test
// never auto-starts inside Practice. Label flips to "Retake" once completed.
async function refreshPlacementStartBtn() {
  if (typeof placementStartBtn === "undefined" || !placementStartBtn) return;
  const status = await PracticeAPI.diagnosticStatus();
  if (!status || status.unavailable) {
    placementStartBtn.classList.add("hidden");
    return;
  }
  // diagnostic-page.js owns label + visibility. Optional because it loads
  // after this script; click/status callbacks run once it is available.
  window.DiagnosticPage?.renderStartButton?.(status, placementStartBtn);
}
window.refreshPlacementStartBtn = refreshPlacementStartBtn;

if (typeof placementStartBtn !== "undefined" && placementStartBtn) {
  placementStartBtn.addEventListener("click", async () => {
    placementStartBtn.disabled = true;
    try {
      const status = await PracticeAPI.diagnosticStart();
      if (!status) throw new Error("not signed in to the practice backend");
      placementStartBtn.classList.add("hidden");
      // Placement owns its backend-driven length; a practice-session quota
      // must never finish it while probes remain.
      if (PracticeSession.isActive()) PracticeSession.finish("placement");
      await window.DiagnosticPage?.refresh();
      await _loadNextPracticeQuestion();
      await window.DiagnosticPage?.refresh();
    } catch (err) {
      outputArea.textContent = "Could not start the placement test: " + err.message;
      placementStartBtn.disabled = false;
    }
  });
}

window.addEventListener("delta:diagnostic-next", async () => {
  try {
    await _loadNextPracticeQuestion();
    await window.DiagnosticPage?.refresh();
  } catch (err) {
    outputArea.textContent = "Could not load the next placement question: " + err.message;
  }
});

// Timer expiry with no learner work clicks this same control. Keep returned
// status: it says whether another probe exists before generic next can fall
// through to ordinary practice.
if (typeof practiceDontKnowBtn !== "undefined" && practiceDontKnowBtn) {
  practiceDontKnowBtn.addEventListener("click", async () => {
    const q = PracticeAPI.currentQuestion;
    if (!q || !q.diagnostic_active) return;
    practiceDontKnowBtn.disabled = true;
    try {
      const status = await PracticeAPI.diagnosticAnswer(q.question_id, "dont_know");
      await _advancePlacementOrFinish(status);
    } catch (err) {
      outputArea.textContent = "Could not record the answer: " + err.message;
      practiceDontKnowBtn.disabled = false;
    }
  });
}

// Skip advances without grading and without claiming a look-up.
if (practiceSkipBtn) {
  practiceSkipBtn.addEventListener("click", async () => {
    practiceSkipBtn.disabled = true;
    try {
      await _loadNextPracticeQuestion();
    } catch (err) {
      outputArea.textContent = "Could not load the next question: " + err.message;
    } finally {
      practiceSkipBtn.disabled = false;
    }
  });
}
