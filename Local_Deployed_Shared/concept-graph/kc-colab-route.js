/* ================================================================
   KC-COLAB-ROUTE.JS — clicking a concept on the map opens the notebook
   section that teaches it.

   WHY IT IS ITS OWN FILE
     `lesson-graph.js` is 971 LOC and RED. It already knows how to draw the
     graph, colour it by mastery, dock a readout and render a lesson; it does
     not need to also know which deploy it is on. It calls two functions here
     and nothing else — `onSelect(kc)` when a bubble is chosen, `onDeselect()`
     when the selection clears.

   WHAT IT DOES
     On the Colab edition the lesson the learner is meant to read lives in the
     notebook, not in the rail — that is the whole premise of the fork. So the
     map should behave like every other route on this deploy: choose a concept,
     and the tab beside you is already at it. `DDColab.hrefForKc` does the join
     (kc → lesson notebook → `#scrollTo=dd-kp-<slug>`, straight out of the
     generated index) and `DDColab.openNotebook` asks the side panel to steer
     the tab, exactly as a routed question does.

   THE LINK IS NOT DECORATION
     `openNotebook` is a no-op in a plain browser tab: unframed there is no
     panel holding the `tabs` permission, and `window.open` without a gesture is
     blocked as a popup. The anchor is the route that always works, and it is
     also the way back after wandering off in Colab. It is a real `<a href>` for
     that reason — a button calling `window.open` from a click handler would
     work too, but not on middle-click, and not for "copy link address".

   ON THE NORMAL DEPLOY
     `DDColab.active()` is false, `hrefForKc` answers "" and nothing here shows.
     The graph keeps rendering the lesson in its own pane, which on that deploy
     is where the lesson belongs.
   ================================================================ */
(function () {
  "use strict";

  // The concept currently selected, so a slow index load cannot paint the link
  // for a bubble the learner has already clicked past.
  let selected = null;

  function colab() {
    const dd = window.DDColab;
    return dd && typeof dd.active === "function" && dd.active() ? dd : null;
  }

  function linkEl() {
    return document.getElementById("kg-colab-link");
  }

  function hide() {
    const a = linkEl();
    if (a) a.hidden = true;
  }

  /* All 63 registry KCs have a published section today, so this branch is for
     the gap that opens the moment one is ADDED: the graph is built from
     `kc_registry.json` and the anchors come from the last notebook publish, so
     a new concept is on the map before it is in a notebook. Say so rather than
     showing a dead link or nothing at all — "nothing at all" is also what the
     normal deploy looks like, and the two are worth telling apart on screen. */
  function paint(kc) {
    if (kc !== selected) return;
    const dd = colab();
    const a = linkEl();
    if (!dd || !a) return;

    const href = dd.hrefForKc(kc);
    a.hidden = false;
    if (!href) {
      a.removeAttribute("href");
      a.classList.add("is-missing");
      a.textContent = "Not in a notebook";
      a.title = "“" + kc + "” has no published notebook section — the graph "
        + "carries it, the notebooks do not. Read it here instead.";
      return;
    }

    a.href = href;
    a.classList.remove("is-missing");
    a.textContent = "Open in Colab ↗";
    a.title = "Open this concept's lesson notebook at the section that teaches it";
    // `force` because this is an explicit click every time. Without it a
    // learner who wandered off in the notebook and clicked the same bubble
    // again would be told, silently, that they were already there.
    dd.openNotebook(href, { force: true });
  }

  function onSelect(kc) {
    selected = kc || null;
    const dd = colab();
    if (!dd || !selected) { hide(); return; }
    // The index is fetched once, asynchronously. Selecting a concept before it
    // lands is normal on a first visit, not an error.
    dd.whenReady(() => paint(kc));
  }

  function onDeselect() {
    selected = null;
    hide();
  }

  /* ── The lesson pane does not belong on this deploy ──────────────────
     The whole premise of the fork is that the lesson is in the notebook. A
     pane rendering a second copy of it beside the map is the same two-sources-
     of-truth problem the practice rail already solved by dropping the prompt
     and the worked example — and here it also cost the map most of its width
     for something the learner is being sent to Colab to read.

     `colab-edition.css` hides the aside; these two controls lived inside its
     header and would go with it, so they move to the graph's own control strip
     rather than being duplicated in the markup — one set of buttons, wired
     once, wherever they end up. */
  function relocate() {
    const controls = document.querySelector(".kg2-controls");
    const link = linkEl();
    if (!controls || !link || link.parentElement === controls) return;
    controls.appendChild(link);
    const max = document.getElementById("kg-maximize");
    if (max) controls.appendChild(max);
  }

  if (colab()) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", relocate, { once: true });
    } else {
      relocate();
    }
  }

  window.DDGraphColab = { onSelect, onDeselect };
})();
