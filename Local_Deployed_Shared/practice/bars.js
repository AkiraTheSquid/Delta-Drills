/* ================================================================
   PRACTICE BARS — the difficulty numbers, on their way to the caption
   ================================================================

   What is left of a file that used to draw two full-width animated tracks.
   Both are gone:

     - The target-difficulty track was a 0-100 bar with two markers, a delta
       band, two threshold zones and a tween. It drew the AIM — where the queue
       is pointing — in the same visual language as a mastery bar, next to a
       tick for the rating of the problem on screen, which is a different
       quantity that routinely sits far away from it. Read as a level, which is
       not what it is. It is now one clause of the stage ladder's caption.
     - The accuracy track duplicated mastery in a weaker form and spent most of
       its life reading "Accuracy appears after calibration". Deleted outright.

   These four functions survive because the call sites in `ui.js` and
   `events.js` are the places that KNOW the numbers — on render, on submit, on
   a restored pending answer. They now compute and forward. The animation went
   with the bar: a line of text has nothing to tween, and `onComplete` fires on
   the spot so nothing downstream waits on a frame that will not come.

   See practice/stage-ladder.js for what replaced them and why. */

/* ── Shared helpers ──────────────────────────────────────────── */

function clampDifficulty(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value));
}

/* `formatDifficulty` lived here. The caption formats its own numbers
   (`stage-ladder.js#_fmt`) because it decides how many of them to print and in
   what order; a second formatter with no caller was one more thing to keep in
   step with a decision made elsewhere. */

/* ── Target difficulty ───────────────────────────────────────── */

/* No reading to show. Two ways to get here: the very first answer in a
   concept, where there is no previous target to have moved FROM; and a
   placement probe, where the answer locates the learner instead of stepping
   the staircase. `currentValue` is drawn when we have it, so the learner still
   sees where the queue is aiming; with nothing at all the caption drops the
   clause rather than printing a zero, which would be the one reading we know
   we do not have. */
function setTargetDifficultyUnavailable(note, currentValue) {
  const known = Number.isFinite(currentValue);
  window.StageLadder?.setDifficulty(undefined, known ? clampDifficulty(currentValue) : null);
}

function setTargetDifficultyInitial(targetDifficulty) {
  window.StageLadder?.setDifficulty(undefined, clampDifficulty(targetDifficulty));
}

function setTargetDifficultyFinal(oldTarget, newTarget) {
  window.StageLadder?.setDifficulty(undefined, clampDifficulty(newTarget));
}

/* Kept as a function rather than deleted at the call sites: `events.js` hands
   it a completion callback that advances the review flow, and that flow is not
   this file's to reorganise. The old value is no longer drawn — the caption
   states where the queue is aiming now, and a "was 25.0, now 20.0" reading of
   a number the learner never asked for was most of what made the old bar look
   like a score.

   It applies nothing itself. Both call sites already apply the final reading
   inside the callback — one of them behind a stale-question guard — so doing
   it here as well rendered the same caption twice per rating. All that is left
   to do is run the completion step, now that there is no frame to wait for. */
function animateTargetDifficulty(oldTarget, newTarget, onComplete) {
  if (typeof onComplete === "function") onComplete();
}
