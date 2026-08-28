/* ================================================================
   BASIC MODE — the one behavioural consequence of the stripped screen

   styles/practice/basic-mode.css hides most of the practice tab's rails
   for learners who have not ticked Advanced mode on the Account tab
   (`body.dd-basic-mode`, written by app.js `applyModeVisibility`). All of
   it is display-only except for one block, and that block is load-bearing
   twice over:

     the felt-difficulty rating — "About right / A bit off / Way off".

   1. `POST /api/practice/feedback` is the ONLY backend mutation that moves
      a subtopic baseline. Submit and override write `pending_attempt` and
      stop there; the attempt is committed to mastery when the rating
      lands. Hide the row and never send one and the learner answers
      questions all day while the model never moves.
   2. `#next-problem-btn` is revealed by `showNextProblemButton()`, which
      runs inside that button's own click handler. Hide the row and there
      is no way forward but waiting out the 02:00 review clock, which
      force-advances by clicking the same default button anyway
      (practice/timer.js `_forceAdvance`).

   So basic mode settles the rating itself, at the neutral default, by
   CLICKING THE REAL BUTTON. Not by calling sendFeedback: the handler in
   events.js also records the completed question, animates the target
   difficulty, writes the concept-understanding readout, pushes the ladder
   estimate, emits `competency:feedback-update` and parks
   `pendingFeedback` for resume. A second copy of that list is how the two
   drift apart, and the review-timer fallback already proves a synthetic
   click is a supported way in.

   🔴 WHAT THIS COSTS, stated plainly: in basic mode the difficulty signal
   is not the learner's, it is `not_much` every time. `not_much` is the
   engine's "stop correcting" — it commits the attempt and applies no
   nudge to `adaptive.nudge_difficulty_offset` — so difficulty still
   tracks correctness through BKT, but the learner's own "that was way too
   hard" never reaches it. Advanced mode is where that signal comes back.
   ================================================================ */

const PracticeBasicMode = (() => {
  "use strict";

  /* 🔴 The SAME predicate styles/practice/basic-mode.css states as
     `html:not(.dd-colab-edition) body.dd-basic-mode`, and it has to stay the
     same one. On the Colab edition the rating row is NOT hidden — that deploy
     restyles it into a stacked column precisely because "nothing advances
     until one is pressed" (colab-edition.css) — so settling it here would
     answer a question the learner was being asked, on a screen where the
     buttons are still sitting in front of them.

     If the CSS guard ever changes, change this with it: a JS predicate that is
     true where the CSS is false auto-rates a visible row, and one that is false
     where the CSS is true dead-ends a hidden one. */
  const active = () =>
    !!document.body &&
    document.body.classList.contains("dd-basic-mode") &&
    !document.documentElement.classList.contains("dd-colab-edition");

  /* 🪦 `settleRating()` LIVED HERE UNTIL 2026-08-28, AND IS NOT COMING BACK
     WITHOUT THE HIDING COMING BACK WITH IT.

     It clicked `.feedback-btn--default` on the learner's behalf, because
     basic-mode.css hid the three rating buttons from everyone who had not
     ticked Advanced mode, and that row is both the only backend mutation
     that moves a subtopic baseline and the only thing that revealed Next
     problem. A stand-in was the right answer to a hidden question.

     The question is not hidden any more. Seth asked for the three choices to
     BE the post-submit interface — docked to the bottom of the viewport,
     phrased as "how much harder / easier do you want the next problem to
     be?", and advancing the question on the same click (see
     practice/difficulty-dock.js and the top of styles/practice/basic-mode.css
     for the reversal). Answering it for the learner would now do two wrong
     things at once: invent an opinion they were visibly being asked for, and
     skip past the problem before they had given it.

     🔴 If a future change hides the rating again, restore BOTH halves
     together — the click-the-real-button stand-in and its call sites in
     events.js. watch_basic_mode.py still holds the whole contract; it stops
     checking it the moment basic-mode.css stops hiding `.feedback-btn`. */

  return { active };
})();

window.PracticeBasicMode = PracticeBasicMode;
