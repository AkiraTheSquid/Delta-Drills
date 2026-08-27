/* AUTOGROW — a feedback note is exactly as tall as what you wrote in it.
 *
 * Seth, 2026-08-27: the feedback boxes were one row tall and never changed,
 * so writing the deep, specific note an instructor actually has to write
 * meant typing into a slot and scrolling inside it, or dragging the resize
 * corner first. Both are the box asking the reviewer to do its job. This
 * grows the box on every keystroke instead; the PAGE scrolls, never the
 * textarea. `resize: none` in the stylesheets goes with it — a drag handle
 * on a box that sizes itself only lets you make it wrong.
 *
 * WIRED BY ATTRIBUTE, NOT BY ID: any `textarea[data-autogrow]` is grown, and
 * the `input` listener is DELEGATED on document, so a note that did not exist
 * at boot works with no registration. instructor-review.js builds a fresh
 * `.ir-q-note` per question card — those are covered by the markup carrying
 * the attribute, and nothing else.
 *
 * The three notes on it today: `.ir-q-note` (per question, instructor
 * review), `#ir-form-note` (graph flags), `#problem-feedback-note` (the
 * in-practice flag). Each keeps its OWN min-height in CSS — that is the
 * empty box's size, which is a design choice per surface, not this file's
 * business. min-height also means this file never needs a floor: a height
 * below it simply loses to the stylesheet.
 *
 * A HIDDEN textarea is skipped on purpose. `scrollHeight` inside a
 * `display: none` subtree is 0, and writing height: 0 there leaves a
 * collapsed box that nothing measures again once it is shown.
 *
 * WHICH IS WHY THE MEASURING IS NOT DRIVEN BY `input` ALONE. A note can
 * hold text it never received a keystroke for — a value the browser
 * restored, a box that was `display: none` when the page loaded and is
 * shown later — and with `overflow-y: hidden` an unmeasured box of that
 * kind hides its own text with no scrollbar to say so. So a
 * ResizeObserver re-measures on any WIDTH change, which covers both of
 * those (0 → rendered, and a re-wrap), and `focusin` catches whatever is
 * left the moment someone goes to write in it. Height changes are
 * ignored on purpose: those are ours, and reacting to them is a loop.
 *
 * 🔴 THE ONE CASE THIS CANNOT SEE: assigning `note.value = …` from script
 * fires no `input` event and changes no width, so nothing here runs and
 * the box keeps its old height until the next focus. A SURFACE THAT
 * PREFILLS A NOTE MUST CALL `DDAutoGrow.grow(note)` ITSELF — which is
 * what instructor-review.js's openForm does after it clears #ir-form-note.
 * Patching the `value` setter would remove the rule, but the only sane
 * place for a patch this file could use is HTMLTextAreaElement.prototype,
 * and practice/runner.js:197 already redefines `value` on the code-cell
 * textareas themselves — one `defineProperty` shadowing another is how
 * that patch breaks, silently, on the elements it matters most for. Not
 * worth it for a case a one-line `grow()` call covers.
 *
 * 🔴 AND THE STYLESHEETS DEPEND ON THIS FILE BEING LOADED. The notes are
 * `resize: none; overflow-y: hidden` — with no script to size them, a long
 * note is clipped with no scrollbar and no drag handle, which is worse than
 * the one-line box this replaced. Ship the <script> tag and the CSS
 * together; never the CSS alone. */
(() => {
  "use strict";

  const SELECTOR = "textarea[data-autogrow]";

  /* One observer for every note on the page. Registration happens inside
     grow(), BEFORE the rendered-check below, so a note that is still hidden
     is watched too and re-measures itself the moment it gets a width. */
  let ro = null;
  const observe = (ta) => {
    if (!window.ResizeObserver || ta.__ddObserved) return;
    if (!ro) {
      ro = new ResizeObserver((entries) => {
        entries.forEach((e) => {
          const w = e.contentRect.width;
          if (w === e.target.__ddW) return;
          e.target.__ddW = w;
          grow(e.target);
        });
      });
    }
    ta.__ddObserved = true;
    ro.observe(ta);
  };

  const grow = (ta) => {
    if (!ta) return;
    observe(ta);
    /* getClientRects().length === 0 means "not rendered" — display:none, a
       hidden ancestor, or not in the document. Measuring one of those is
       what writes the height that nothing corrects. */
    if (!ta.getClientRects().length) return;

    if (ta.__ddBorders === undefined) {
      /* `scrollHeight` is content + padding and EXCLUDES the border, but the
         app is border-box (base.css), so the height we assign has to carry
         the border too — otherwise every measured-to-fit box lands 2px short
         and shows a permanent hairline scrollbar. Same fix, same reason as
         practice/notebook-editor.js's cell resize. Read ONCE per element:
         this runs on every keystroke and the border is a constant. */
      const box = getComputedStyle(ta);
      ta.__ddBorders = box.boxSizing === "border-box"
        ? (parseFloat(box.borderTopWidth) || 0) + (parseFloat(box.borderBottomWidth) || 0)
        : 0;
    }

    /* `height: auto` first is not optional: `scrollHeight` on an element with
       an explicit height IS that height, so without the reset the box can
       only ever grow — delete a paragraph and it keeps the taller size. */
    ta.style.height = "auto";
    ta.style.height = `${ta.scrollHeight + ta.__ddBorders}px`;
  };

  const growAll = () => document.querySelectorAll(SELECTOR).forEach(grow);

  /* `focusin`, not `focus`: focus does not bubble, and these listeners are
     delegated so a note built after boot needs no registration. */
  ["input", "focusin"].forEach((type) =>
    document.addEventListener(type, (e) => {
      const t = e.target;
      if (t && t.matches && t.matches(SELECTOR)) grow(t);
    })
  );

  /* The fallback for a browser with no ResizeObserver, and the only thing
     that measures a note nobody has typed in yet after the window changes
     width — a narrower box re-wraps the same text onto more lines, and
     nothing fires `input` for that. */
  window.addEventListener("resize", growAll);

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", growAll);
  } else {
    growAll();
  }

  window.DDAutoGrow = { grow, growAll };
})();
