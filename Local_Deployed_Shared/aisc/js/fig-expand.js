/* A "Full screen" button under every figure's graphic (Seth, 2026-09-25:
   "a button below the graph … goes into the expanded mode that's full screen,
   similar to the other knowledge graph").

   Most figures go full screen here: the card gets `.is-max` (aisc.css fixes it
   over the viewport) and a resize is announced, because every figure sizes
   itself from its stage — the SVG ones redraw on window `resize`, and
   fig-graphs.js refits its Cytoscape graphs from a ResizeObserver.

   Fig. 2 is the exception. It is the app's concept map (why-graph.js), whose
   own Maximize already opens the real Knowledge Graph full screen, so its
   button presses that one; the frame then covers this button, and the frame's
   Minimize (or Escape) is the way back. */
(function () {
  "use strict";

  var root = document.getElementById("aisc-root");
  if (!root) return;
  var page = root.closest(".page");
  var open = null; // the figure that is full screen, if any

  function announce() { window.dispatchEvent(new Event("resize")); }

  function label(btn, on) {
    btn.textContent = on ? "⤡ Exit full screen" : "⤢ Full screen";
    btn.setAttribute("aria-pressed", on ? "true" : "false");
  }

  // `quiet`: closing because the page went away; its button is hidden then,
  // so focus has nowhere sensible to go.
  function set(fig, on, quiet) {
    var btn = fig.querySelector(".fig-expand");
    fig.classList.toggle("is-max", on);
    document.body.classList.toggle("aisc-fig-max-open", on);
    open = on ? fig : null;
    label(btn, on);
    announce();
    if (!quiet) btn.focus();
  }

  // What a keyboard can land on inside the open figure, in document order;
  // `offsetParent` drops anything hidden.
  var TAB_STOPS = 'a[href], button:not([disabled]), input:not([disabled]),' +
    ' select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
  function tabStops(fig) {
    return Array.prototype.filter.call(fig.querySelectorAll(TAB_STOPS), function (el) { return el.offsetParent !== null; });
  }

  root.querySelectorAll("figure.fig").forEach(function (fig) {
    var body = fig.querySelector(".fig-body");
    var stage = body && body.querySelector(".fig-stage");
    if (!stage || fig.querySelector(".fig-expand")) return;
    var row = document.createElement("div");
    row.className = "fig-expand-row";
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "fig-expand";
    label(btn, false);
    row.appendChild(btn);
    // Straight after the stage, so a single-column (narrow) card reads
    // graphic, button, controls.
    stage.insertAdjacentElement("afterend", row);

    var mapMax = stage.querySelector(".wta-graph-btn");
    btn.addEventListener("click", function () {
      if (mapMax) { mapMax.click(); return; }
      if (open && open !== fig) set(open, false);
      set(fig, !fig.classList.contains("is-max"));
    });
  });

  document.addEventListener("keydown", function (e) {
    if (!open) return;
    if (e.key === "Escape") { set(open, false); return; }
    // The full-screen card covers the page but does not take it out of the
    // tab order, so Tab wraps inside the card instead of walking into what
    // is behind it (as why-graph.js does for Fig. 2's frame).
    if (e.key !== "Tab") return;
    var stops = tabStops(open);
    if (!stops.length) return;
    var first = stops[0], last = stops[stops.length - 1], at = stops.indexOf(document.activeElement);
    if (at === -1) { e.preventDefault(); (e.shiftKey ? last : first).focus(); }
    else if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  });
  // Leaving the page while a figure is full screen would strand the lifted
  // write-up and the locked body scroll.
  if (page) new MutationObserver(function () {
    if (open && page.classList.contains("hidden")) set(open, false, true);
  }).observe(page, { attributes: true, attributeFilter: ["class"] });
})();
