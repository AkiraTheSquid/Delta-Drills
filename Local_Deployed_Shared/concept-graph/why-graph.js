/* ================================================================
   WHY-GRAPH.JS — the illustrative map on "Why this app exists"

   A DEMONSTRATION, not a readout. The Knowledge Graph tab
   (lesson-graph.js) draws ~210 real concepts coloured by this
   learner's real BKT posteriors, with a side panel, a docked
   learner-model readout, gate ticks and confidence intervals. None
   of that belongs on a page someone reads before they have done
   anything: there is no learner yet, so every number would be a
   prior dressed up as a measurement.

   So this file draws nine real concepts from the very start of the
   curriculum, wired by their real prerequisite edges, with FIXED
   illustrative mastery values that are not read from anywhere and
   are not written anywhere. It is the same picture, in miniature,
   for someone who wants to see what the thing looks like.

   THINGS THIS MAKES TRUE
     1. Nothing here reads learner state — no localStorage, no
        /api/practice/*, no adaptive_state_*. If you ever find
        yourself wanting real numbers on this page, that is the
        Knowledge Graph tab and it already exists.
     2. It borrows lesson-graph.js's LOOK, deliberately and by
        copy: the same round-rectangle nodes, the same red→blue
        mastery ramp, the same red prerequisite arrows, the same
        yellow "next up" ring. If that styling changes over there
        and this page starts looking like a different product, this
        is the file to update. It is not imported, because
        lesson-graph.js's initialiser is entangled with the lattice
        fetch, the crosswalk and the info pane this page must not
        have.
     3. There is no info pane, on purpose. Hover highlights the
        prerequisite path and that is the whole interaction.
   ================================================================ */

(function installWhyGraph(global) {
  const CONTAINER_ID = "wta-graph-cy";

  // Straight from lesson-graph.js so the two pages read as one product.
  const ACCENT = "#ffd23f";                  // path highlight + "next up"
  const UNKNOWN_COLOR = "#5b5b70";           // no estimate yet
  const EDGE_COLOR = "#e3212c";
  // red (low) → muted purple → blue (high)
  const masteryColor = (r) => {
    if (!Number.isFinite(r)) return UNKNOWN_COLOR;
    const t = Math.max(0, Math.min(1, r));
    const lo = [214, 72, 72], hi = [59, 130, 246];
    const c = lo.map((v, i) => Math.round(v + (hi[i] - v) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  };

  // Nine concepts and their prerequisite edges, taken verbatim from
  // lessons/kc_registry.json (lessons np-1 and np-2 — the first thing the
  // curriculum teaches). Labels are shortened for a small canvas; the ids
  // are the real ones so this stays checkable against the registry.
  //
  // `m` is ILLUSTRATIVE. It falls off as you move up the graph because that
  // is the shape a real learner's frontier has, and it ends in two concepts
  // with no estimate at all — which is the point the page is making: the
  // app serves the boundary, not the top.
  const CONCEPTS = [
    { id: "numpy.ndarray-model",      label: "What an ndarray is",     m: 0.93, prereqs: [] },
    { id: "numpy.constructors",       label: "Array constructors",     m: 0.81, prereqs: ["numpy.ndarray-model"] },
    { id: "numpy.slicing-views",      label: "Slicing and views",      m: 0.74, prereqs: ["numpy.ndarray-model"] },
    { id: "numpy.elementwise-ufuncs", label: "Elementwise math",       m: 0.58, prereqs: ["numpy.ndarray-model"] },
    { id: "numpy.reshape-flatten",    label: "Reshape and ravel",      m: 0.41, prereqs: ["numpy.ndarray-model"] },
    { id: "numpy.ranges",             label: "arange and linspace",    m: 0.30, prereqs: ["numpy.constructors"] },
    { id: "numpy.aggregations",       label: "Aggregations",           m: 0.16, prereqs: ["numpy.elementwise-ufuncs"], nextUp: true },
    { id: "numpy.boolean-masking",    label: "Boolean masks",          m: null, prereqs: ["numpy.slicing-views", "numpy.elementwise-ufuncs"] },
    { id: "numpy.where-select",       label: "np.where",               m: null, prereqs: ["numpy.boolean-masking"] },
  ];

  const buildElements = () => {
    const elements = CONCEPTS.map((c) => ({
      data: { id: c.id, label: c.label, color: masteryColor(c.m) },
      classes: c.nextUp ? "next-up" : "",
    }));
    let i = 0;
    CONCEPTS.forEach((c) => {
      c.prereqs.forEach((p) => {
        elements.push({ data: { id: `wg-e${i++}`, source: p, target: c.id } });
      });
    });
    return elements;
  };

  let cy = null;

  const draw = (container) => {
    cy = global.cytoscape({
      container,
      elements: buildElements(),
      wheelSensitivity: 0.2,
      minZoom: 0.4,
      maxZoom: 2.5,
      style: [
        { selector: "node", style: {
            "background-color": "data(color)",
            "shape": "round-rectangle",
            "label": "data(label)",
            "width": "label", "height": "label", "padding": "11px",
            "text-wrap": "wrap", "text-max-width": "110px",
            "text-valign": "center", "text-halign": "center",
            "font-size": 12, "font-weight": 600, "color": "#15151f",
            "border-width": 1, "border-color": "rgba(0,0,0,.28)",
            "transition-property": "opacity, border-width, border-color",
            "transition-duration": "120ms",
        }},
        { selector: "edge", style: {
            "curve-style": "bezier", "width": 1.8,
            "line-color": EDGE_COLOR, "target-arrow-shape": "triangle",
            "target-arrow-color": EDGE_COLOR, "arrow-scale": 0.8, "opacity": 0.9,
        }},
        { selector: ".faded", style: { "opacity": 0.12 } },
        { selector: "node.hl", style: {
            "opacity": 1, "border-width": 3, "border-color": ACCENT, "z-index": 50,
        }},
        { selector: "edge.hl", style: {
            "opacity": 1, "width": 3, "line-color": ACCENT,
            "target-arrow-color": ACCENT, "z-index": 60,
        }},
        // The one thing on this page that is a claim rather than a colour:
        // this is where the queue would send you next.
        { selector: "node.next-up", style: {
            "border-width": 4, "border-color": ACCENT, "border-style": "solid",
            "outline-width": 6, "outline-color": ACCENT, "outline-opacity": 0.35,
            "z-index": 80,
        }},
      ],
      layout: {
        name: global.cytoscapeDagre ? "dagre" : "cose",
        rankDir: "BT", nodeSep: 22, rankSep: 76, edgeSep: 10,
        animate: false, fit: true, padding: 26,
      },
    });

    // Hover lights the whole prerequisite path INTO a concept — the thing the
    // graph is for. `predecessors()` walks every incoming edge transitively,
    // which is exactly "everything you need before this".
    const highlight = (node) => {
      const path = node.predecessors().union(node);
      cy.elements().addClass("faded");
      path.removeClass("faded").addClass("hl");
    };
    const clear = () => cy.elements().removeClass("faded hl");

    cy.on("mouseover", "node", (evt) => highlight(evt.target));
    cy.on("mouseout", "node", clear);
    // Touch has no hover; tap is the same gesture there, and tapping the
    // background clears it.
    cy.on("tap", "node", (evt) => highlight(evt.target));
    cy.on("tap", (evt) => { if (evt.target === cy) clear(); });

    const status = document.getElementById("wta-graph-status");
    if (status) status.style.display = "none";
  };

  // Cytoscape measures the container, so it cannot be drawn while the page is
  // display:none — which it is for anyone whose landing tab is Practice (i.e.
  // every returning visitor). Draw on first reveal, and refit on later ones.
  const whenVisible = (page, container) => {
    const visible = () => !page.classList.contains("hidden") && container.offsetParent !== null;
    const run = () => {
      if (!visible()) return false;
      if (!cy) draw(container);
      else { cy.resize(); cy.fit(undefined, 26); }
      return true;
    };
    if (run()) return;
    // `.page` visibility is a class app.js toggles, so the class list is the
    // signal. No polling, and nothing for app.js to have to call.
    const observer = new MutationObserver(() => { if (run()) observer.disconnect(); });
    observer.observe(page, { attributes: true, attributeFilter: ["class"] });
  };

  // Idempotent: deltaInitWhyGraph is exported, and every call used to add
  // another resize listener and — while the page was hidden — another
  // MutationObserver that nothing ever disconnected.
  let installed = false;

  const init = () => {
    if (installed) return;
    const container = document.getElementById(CONTAINER_ID);
    const page = document.getElementById("page-why-this-app");
    if (!container || !page) return;
    if (typeof global.cytoscape !== "function") {
      const status = document.getElementById("wta-graph-status");
      if (status) status.textContent = "The map couldn't load.";
      return;
    }
    installed = true;
    whenVisible(page, container);
    global.addEventListener("resize", () => {
      if (cy && page && !page.classList.contains("hidden")) { cy.resize(); cy.fit(undefined, 26); }
    });
  };

  // cytoscape + the dagre layout are `defer`red script tags; this file is one
  // too and is placed after them, so document order has already run them by
  // the time this executes. `load` is the belt-and-braces path for a cached
  // reorder.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }

  global.deltaInitWhyGraph = init;
})(window);
