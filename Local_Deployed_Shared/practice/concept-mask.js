/* ================================================================
   CONCEPT MASK — the concept's name is hidden until the drill is answered
   ================================================================

   Seth, 2026-09-20: "for the vocabulary, the problem is that it often shows
   the progress bar at the top with the very vocabulary that you are supposed
   to be trying to recall, so it basically gives the answer away, at least in
   the very beginning. So I would want it to hide the term for the top bar
   until you have answered the question."

   He is right, and the registry shows how right: the concept titles are
   "Locating extremes: argmin and argmax", "Numeric ranges: arange and
   linspace", "Filling a tensor in place with out=". On a syntax concept the
   drill IS "recall the call", and the title is the call. Printed in the
   topbar over the problem, it turns a recall drill into a copy exercise.

   ── ONE STATE, ONE OWNER ─────────────────────────────────────────
   `masked` is true while a drill is on screen and NOT yet graded. It is read
   off two facts the app already keeps, never off a flag of our own:

     `#practice-feedback-area.hidden` — no graded result on screen. Removed
        by events.js when the grade lands (and by timer.js when a reload
        restores a graded review), put back by the advance path before the
        next question loads. difficulty-dock.js reads the same class for the
        same reason: it is what the app means by "there is a result on
        screen".
     `body.lesson-mode` — a lesson page is up (practice/lessons.js). A lesson
        NAMES its concept, that is what a lesson is for; nothing is masked.

   This file publishes the answer two ways, and the readers pick whichever
   fits: `body.dd-concept-masked` for stylesheets and for a synchronous read
   at render time, and a `dd-concept-mask` event for the topbar pill, which
   has to REPAINT when the state flips underneath a concept it already drew.

   ── WHO HIDES WHAT ───────────────────────────────────────────────
   Three places name the concept while a drill is up, and each hides its own:

     concept-pill.js   — the topbar chip; draws MASK_TEXT instead of the title
                         and says why in its tooltip.
     practice/ui.js    — the `#question-number` heading (the left-panel card,
                         on screen under 620px). Writes the real title into
                         `data-concept` and MASK_TEXT into the text; the
                         reveal below puts the title back.
     stage-ladder.js   — the concept button in the ladder card, hidden when
                         its label repeats the heading. Compares against
                         `data-concept` while masked, so the button does not
                         pop up naming the concept the heading just hid.

   Nothing else needs to. "Understanding of <concept>" (bars.js) lives INSIDE
   the feedback area, so it is on screen only once the grade is. The lesson
   heading is "Lesson". The notch menu's graph jump says "See in knowledge
   graph" with no name in it.

   🔴 THE PROBLEM TEXT ITSELF IS NOT TOUCHED. A prompt that names the call is
   an authoring fault, and the content rubric already forbids it; hiding
   words in the prompt at render time would be a second, weaker rule that
   also breaks the prompt.
   ================================================================ */

(function () {
  "use strict";

  const MASK_TEXT = "Concept hidden until you answer";
  const MASK_TIP =
    "The concept's name is hidden until you answer — on a syntax drill the name would give the call away.";

  let masked = null;

  const _compute = () => {
    if (document.body.classList.contains("lesson-mode")) return false;
    const area = document.getElementById("practice-feedback-area");
    if (!area) return false;
    return area.classList.contains("hidden");
  };

  /* The heading's reveal. ui.js writes the real title into `data-concept`
     and the mask into the text; when the grade lands, the title goes back.
     Only when the node still carries a concept — the lesson page writes
     "Lesson" straight into the text and carries none. */
  const _revealHeading = () => {
    const h = document.getElementById("question-number");
    if (!h) return;
    const real = h.dataset.concept;
    if (real && h.textContent === MASK_TEXT) h.textContent = real;
  };
  const _maskHeading = () => {
    const h = document.getElementById("question-number");
    if (!h) return;
    if (h.dataset.concept && h.textContent === h.dataset.concept) h.textContent = MASK_TEXT;
  };

  const _apply = () => {
    const next = _compute();
    if (next === masked) return;
    masked = next;
    document.body.classList.toggle("dd-concept-masked", masked);
    if (masked) _maskHeading();
    else _revealHeading();
    window.dispatchEvent(new CustomEvent("dd-concept-mask", { detail: { masked } }));
  };

  window.ConceptMask = {
    MASK_TEXT,
    MASK_TIP,
    /* Recomputed from the DOM on every read, not the cached flag: a caller
       in the same task that hid the feedback area (ui.js rendering the next
       drill) asks before the MutationObserver has run. */
    masked: () => _compute(),
    /* The name to draw for `title` right now. */
    label: (title) => (window.ConceptMask.masked() ? MASK_TEXT : title),
  };

  const _observe = () => {
    if (typeof MutationObserver !== "function") return;
    const area = document.getElementById("practice-feedback-area");
    if (area) {
      new MutationObserver(_apply).observe(area, { attributes: true, attributeFilter: ["class"] });
    }
    new MutationObserver(_apply).observe(document.body, { attributes: true, attributeFilter: ["class"] });
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => { _observe(); _apply(); });
  } else {
    _observe();
    _apply();
  }
})();
