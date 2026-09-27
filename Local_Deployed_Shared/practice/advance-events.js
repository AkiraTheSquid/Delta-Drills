/* ================================================================
   QUESTION ADVANCE EVENTS — Next, Skip

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
  await _loadNextPracticeQuestion();
});

/* 🪦 The placement lifecycle that lived here — the start button, "I don't
   know yet", the stopping-rule check after each probe — went with the
   placement test (Seth, 2026-09-26). The Practice tab's survey
   (practice/survey.js) sets the starting point now. */

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
