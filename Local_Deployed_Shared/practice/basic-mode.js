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

  /* Settle the felt-difficulty rating for a grade that has just landed.

     No-op in Advanced mode, and no-op whenever the rating is not the thing
     standing between the learner and Next — which covers every path that
     already called `showNextProblemButton()` itself: placement probes, a
     Colab verdict with no pending attempt, and a review restored from a
     paused session (`applyPendingFeedbackState`), where the rating was
     sent before the pause and must not be sent twice.

     Safe to call more than once per question: the handler disables all
     three buttons on its first click, and this refuses a disabled one. */
  const settleRating = () => {
    if (!active()) return false;
    const area = document.getElementById("practice-feedback-area");
    if (!area || area.classList.contains("hidden")) return false;
    const next = document.getElementById("next-problem-btn");
    if (!next || !next.classList.contains("hidden")) return false;
    const def = document.querySelector(".feedback-btn--default");
    if (!def || def.disabled || def.classList.contains("hidden")) return false;
    def.click();
    return true;
  };

  return { active, settleRating };
})();

window.PracticeBasicMode = PracticeBasicMode;
