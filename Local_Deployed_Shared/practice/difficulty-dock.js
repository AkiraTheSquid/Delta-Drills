/* ================================================================
   practice/difficulty-dock.js — the post-submit difficulty question,
   frozen to the bottom of the viewport.

   Seth, 2026-08-28, live-testing: "rather than it being on the left pane
   or whatever, I want it to be frozen to the bottom part so that it's
   extremely obvious. It should have a question: how hard do you want the
   next problem to be, or how easy do you want the next problem to be, and
   then below it it has the like three answer choices."

   🔴 THIS FILE MOVES NODES, IT DOES NOT MINT A SECOND COPY OF THEM.
   `#feedback-prompt` and `.feedback-buttons` are re-parented out of
   `#practice-feedback-area` and into the dock, so the three buttons the
   learner clicks down here are the SAME elements `events.js` bound its
   handler to and the SAME ones `ui.js::applyResult` relabels. A dock that
   rendered its own buttons and forwarded the clicks would be a second copy
   of a list that already drifted once (see practice/README.md on
   `settleRating`), and every future change would have to be made twice.

   Because those two nodes leave `#practice-feedback-area`, hiding that area
   no longer hides them — which is the whole point (they are pinned to the
   viewport now, not to the rail), and also the one thing that can go wrong.
   The dock therefore mirrors the area's own visibility: a MutationObserver
   on its `class` attribute opens the dock when the area is shown and closes
   it when it is hidden. That is deliberately the SAME signal the rest of the
   app already uses to mean "there is a graded result on screen", rather than
   a second state of our own that could disagree with it.

   🔴 NOT ON THE COLAB EDITION. `colab-edition.css` already stacks the rating
   into a column of its own because "nothing advances until one is pressed",
   and `basic-mode.js::active()` reads the same `dd-colab-edition` predicate.
   Docking it there would fight a layout that was written on purpose.
   ================================================================ */

(() => {
  "use strict";

  if (document.documentElement.classList.contains("dd-colab-edition")) return;

  let dock = null;
  let area = null;
  let page = null;

  const _open = (on) => {
    if (dock) dock.classList.toggle("is-open", !!on);
    // The dock floats over the bottom of the page, so the column underneath it
    // needs room to scroll clear of it. A class on <body> rather than a inline
    // pad so the stylesheet keeps every measurement in one place.
    document.body.classList.toggle("dd-dock-open", !!on);
  };

  /* 🔴 TWO CONDITIONS, NOT ONE. The dock is a child of <body>, not of
     `#page-practice` — that is what lets it sit over the bottom of the
     viewport, and it is also why leaving the practice tab does NOT take it
     off screen the way it takes the rest of the practice UI. `.page` elements
     are hidden with the same `hidden` class the feedback area uses, so the
     dock asks about both: a graded result is on screen AND the page it
     belongs to is the one being looked at. Miss the second and the learner
     lands on Account or Concepts with a difficulty question from another tab
     pinned to the bottom of the window. */
  const _areaIsShowing = () =>
    !!area &&
    !area.classList.contains("hidden") &&
    !!page &&
    !page.classList.contains("hidden");

  const build = () => {
    area = document.getElementById("practice-feedback-area");
    page = document.getElementById("page-practice");
    const prompt = document.getElementById("feedback-prompt");
    const buttons = area && area.querySelector(".feedback-buttons");
    // Every one of the three is required. A partial dock would strand the
    // rating somewhere the learner cannot reach it, and there is no path out
    // of a graded question that does not go through those buttons.
    if (!area || !page || !prompt || !buttons) return;

    dock = document.createElement("div");
    dock.id = "difficulty-dock";
    dock.className = "difficulty-dock";
    // Announced, not silent: the question changes under the learner between
    // questions and a screen reader should hear it when it does.
    dock.setAttribute("role", "group");
    dock.setAttribute("aria-live", "polite");

    const inner = document.createElement("div");
    inner.className = "difficulty-dock-inner";

    /* 🔴 THE QUESTION AND ITS ⓘ SHARE A ROW, and that row is a real element.
       `#feedback-prompt` is a block and the ⓘ is inline-flex, so as bare
       siblings the icon wraps onto a line of its own and renders as a stray
       "i" floating between the question and the answers — which is what the
       first build actually looked like. A flex row keeps them on one line and
       lets the pair centre as a unit. */
    const row = document.createElement("div");
    row.className = "difficulty-dock-question";
    // Order is the spec: the question, then the three answers under it.
    row.appendChild(prompt);
    /* 🔴 The ⓘ travels WITH the prompt. `infotips.js` mints its icon as the
       next sibling of the element carrying `data-dd-info` (`#feedback-prompt`
       carries `feedback-rating`), so whether it already exists here depends on
       script order — and the one outcome we cannot have is the icon left
       behind in `#practice-feedback-area` as a bare "i" under the result
       badge. It goes into the question ROW, not the dock body, for the
       reason spelled out above that row. That exact orphan was found and fixed once already, when basic
       mode hid the prompt (practice/README.md). Moving the icon when it is
       there covers infotips-ran-first; leaving `data-dd-info` on the prompt
       covers infotips-ran-later, because it inserts relative to the element,
       wherever the element now lives. */
    const infoIcon = prompt.nextElementSibling;
    if (infoIcon && infoIcon.classList.contains("dd-info")) {
      row.appendChild(infoIcon);
    }
    inner.appendChild(row);
    inner.appendChild(buttons);
    dock.appendChild(inner);
    document.body.appendChild(dock);

    /* `#feedback-help` stays where it is and stays hidden (see
       styles/practice/difficulty-dock.css). Its copy described the OLD
       question — "difficulty is set by your mastery ... About right is the
       safe default" — and there is no "about right" any more; the dock asks
       for a direction and a size, and Seth asked for the interface to stay
       simple. Left in the DOM rather than deleted because deleting markup is
       index.html's call, not ours, and `ui.js` still writes to it. */

    _open(_areaIsShowing());
    const watch = new MutationObserver(() => _open(_areaIsShowing()));
    for (const el of [area, page]) {
      watch.observe(el, { attributes: true, attributeFilter: ["class"] });
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", build, { once: true });
  } else {
    build();
  }
})();
