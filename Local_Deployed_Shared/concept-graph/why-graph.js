/* ================================================================
   WHY-GRAPH.JS — the map on "Why this app exists"

   A DEMONSTRATION, not a readout. It draws the WHOLE lesson graph —
   every knowledge component in lessons/kc_registry.json, wired by
   its real prerequisite edges — because the point of the page is
   how much structure is underneath, and a nine-node excerpt read as
   "that's the curriculum?". The one thing that is not real is the
   colour: mastery here is DERIVED FROM GRAPH DEPTH, not from a
   learner, because the page is read before anyone has done anything
   and every number on it would otherwise be a prior dressed up as a
   measurement.

   The Knowledge Graph tab (lesson-graph.js) draws the same concepts
   coloured by this learner's real BKT posteriors, with a side
   panel, a docked learner-model readout, gate ticks and confidence
   intervals. None of that belongs here.

   THINGS THIS MAKES TRUE
     1. Nothing here reads learner state — no localStorage, no
        /api/practice/*, no adaptive_state_*. The only fetch is the
        registry, which is static content.
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
        prerequisite path; maximise fills the window. That is the
        whole interaction.
   ================================================================ */

(function installWhyGraph(global) {
  const CONTAINER_ID = "wta-graph-cy";
  const REGISTRY_URL = "lessons/kc_registry.json";

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

  /* ---- the illustrative colouring --------------------------------
     Deterministic, so the page looks the same to everyone and to
     anyone comparing two screenshots of it. Mastery falls off with
     longest-path depth, which is the shape a real frontier has: the
     things you were taught first are the things you know, and the
     boundary is a band, not a line. Past the band there is no
     estimate at all — the grey nodes are the honest part of the
     picture, because a learner who has answered nothing HAS no
     estimate there. The jitter exists so the result reads as
     measurements rather than as a gradient.
     --------------------------------------------------------------- */
  const FRONTIER_DEPTH = 5;   // depth at which the illustrative estimate runs out
  const JITTER = 0.16;

  // FNV-1a over the concept id → [0,1). Any stable hash would do; this one
  // is short and needs no dependency.
  const hash01 = (s) => {
    let h = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return ((h >>> 0) % 100000) / 100000;
  };

  // Longest path from a root. `seen` guards a cycle: the registry is a DAG
  // and a cycle would be a data bug, but this file must not hang on one.
  const depthsOf = (byId) => {
    const cache = {};
    const walk = (id, seen) => {
      if (cache[id] !== undefined) return cache[id];
      if (seen.has(id)) return 0;
      seen.add(id);
      const ps = (byId[id].prereqs || []).filter((p) => byId[p]);
      const d = ps.length ? 1 + Math.max(...ps.map((p) => walk(p, seen))) : 0;
      seen.delete(id);
      cache[id] = d;
      return d;
    };
    Object.keys(byId).forEach((id) => walk(id, new Set()));
    return cache;
  };

  // Long titles ("Array constructors: zeros, ones, full, empty, ...") are the
  // lesson's own headline; the half before the colon is the concept's name,
  // and it is the half that stays readable on a node at fit-zoom.
  const shortLabel = (title) => {
    const cut = String(title || "").split(":")[0].trim();
    return cut || String(title || "");
  };

  const buildElements = (registry) => {
    const kcs = (registry && registry.kcs) || [];
    const byId = {};
    kcs.forEach((k) => { byId[k.id] = k; });
    const depth = depthsOf(byId);

    const mastery = {};
    kcs.forEach((k) => {
      const base = 1 - depth[k.id] / FRONTIER_DEPTH;
      const m = base + (hash01(k.id) - 0.5) * JITTER;
      mastery[k.id] = m < 0.06 ? null : Math.min(1, m);
    });

    // "Next up" is the claim the page is making, so it is picked the way the
    // app picks: among the concepts with no estimate, the one whose
    // prerequisites are furthest along. Ties break on id, so it is stable.
    let nextUp = null, bestScore = -1;
    kcs.forEach((k) => {
      if (mastery[k.id] !== null) return;
      const ps = (k.prereqs || []).filter((p) => byId[p]);
      if (!ps.length) return;
      const known = ps.map((p) => mastery[p]).filter((v) => v !== null);
      if (known.length !== ps.length) return;   // something underneath is unknown too
      const score = known.reduce((a, b) => a + b, 0) / known.length;
      if (score > bestScore || (score === bestScore && nextUp && k.id < nextUp)) {
        bestScore = score;
        nextUp = k.id;
      }
    });

    const elements = kcs.map((k) => ({
      data: {
        id: k.id,
        label: shortLabel(k.title),
        title: k.title,
        color: masteryColor(mastery[k.id]),
      },
      classes: k.id === nextUp ? "next-up" : "",
    }));
    let i = 0;
    kcs.forEach((k) => {
      (k.prereqs || []).forEach((p) => {
        if (byId[p]) elements.push({ data: { id: `wg-e${i++}`, source: p, target: k.id } });
      });
    });
    return elements;
  };

  let cy = null;

  const refit = () => {
    if (!cy) return;
    const c = cy.container();
    if (!c || !c.clientWidth || !c.clientHeight) return;
    cy.resize();
    cy.fit(undefined, 24);
  };

  // Tighter than lesson-graph.js's spacing for the same reason as the label
  // width: this view fits all 63 concepts at once and the real estate is the
  // constraint. BT so prerequisites sit BENEATH what they unlock, which is
  // what the page's copy says and what the Knowledge Graph tab does.
  const LAYOUT = {
    name: "dagre",
    rankDir: "BT", nodeSep: 12, rankSep: 58, edgeSep: 8,
    animate: false, fit: true, padding: 24,
  };
  const layoutOpts = () => (global.cytoscapeDagre ? LAYOUT : { name: "cose", animate: false, fit: true, padding: 24 });

  const draw = (container, elements) => {
    cy = global.cytoscape({
      container,
      elements,
      wheelSensitivity: 0.2,
      minZoom: 0.08,
      maxZoom: 2.5,
      style: [
        { selector: "node", style: {
            "background-color": "data(color)",
            "shape": "round-rectangle",
            "label": "data(label)",
            "width": "label", "height": "label", "padding": "9px",
            // 96px, not lesson-graph.js's 120px, and the labels are cut at the
            // colon: the whole graph is 63 concepts, and every pixel of node
            // width costs zoom in a view whose job is to fit all of them.
            "text-wrap": "wrap", "text-max-width": "96px",
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
      layout: layoutOpts(),
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

    // The one thing that keeps the drawing honest across the band, the
    // maximised state and a window drag: watch the box, not the events that
    // are supposed to change it.
    if (typeof global.ResizeObserver === "function") {
      new global.ResizeObserver(refit).observe(container);
    }

    const status = document.getElementById("wta-graph-status");
    if (status) status.style.display = "none";
  };

  /* ---- maximise ---------------------------------------------------
     Sixty-three concepts do not fit a 460px band at a readable size —
     which is the honest thing for the band to say, and the reason the
     button exists. Maximised is `position: fixed; inset: 0` OVER the
     topbar, so the graph is the only thing on screen, and the fit lands
     at roughly half again the zoom the band can give it.
     ----------------------------------------------------------------- */
  const setMaximised = (frame, btn, on) => {
    frame.classList.toggle("is-max", on);
    document.body.classList.toggle("wta-max-open", on);
    if (btn) {
      btn.setAttribute("aria-pressed", on ? "true" : "false");
      btn.setAttribute("aria-label", on ? "Minimize the map" : "Maximize the map");
      btn.title = on ? "Minimize" : "Maximize";
      btn.textContent = on ? "⤡ Minimize" : "⤢ Maximize";
    }
    // Refit straight away rather than from a requestAnimationFrame: rAF does
    // not fire in a hidden tab, so the deferred version left the graph
    // full-screen at the band's zoom. Cytoscape's resize() reads the
    // container's offset size, which forces the pending reflow for the class
    // just toggled, so measuring here is safe. The ResizeObserver installed in
    // draw() is the backstop for every other way this box can change size.
    // Only the fit's zoom differs between the two states (~0.33 in the band,
    // ~0.5 full-screen); dagre's packing does not depend on the viewport, so
    // the layout is not re-run.
    refit();
    // Focus follows the surface: maximising with the keyboard should leave the
    // focus on the control that gets you back out.
    if (on && btn) btn.focus();
  };

  const wireMaximise = (frame, page) => {
    const btn = document.getElementById("wta-graph-max");
    if (!btn) return;
    btn.addEventListener("click", () => {
      setMaximised(frame, btn, !frame.classList.contains("is-max"));
    });
    document.addEventListener("keydown", (evt) => {
      if (!frame.classList.contains("is-max")) return;
      // Escape is what every full-screen surface answers to, and the button is
      // the only other way out — there is no chrome behind it to click.
      if (evt.key === "Escape") { setMaximised(frame, btn, false); return; }
      // The maximised frame covers the page but does not remove it from the
      // tab order, so Tab walked into a topbar and an article nobody can see.
      // Containment is cheap here because the frame has exactly one focusable
      // child: the button itself. The graph is a canvas.
      if (evt.key === "Tab") { evt.preventDefault(); btn.focus(); }
    });
    // Leaving the page while maximised would strand `body.wta-max-open` — the
    // whole document unscrollable, with nothing on screen explaining why. The
    // tab strip is covered, but solo-route, the back button and any programmatic
    // switchTab can all still move off this page.
    new MutationObserver(() => {
      if (page.classList.contains("hidden") && frame.classList.contains("is-max")) {
        setMaximised(frame, btn, false);
      }
    }).observe(page, { attributes: true, attributeFilter: ["class"] });
  };

  // Cytoscape measures the container, so it cannot be drawn while the page is
  // display:none — which it is for anyone whose landing tab is Practice (i.e.
  // every returning visitor). Draw on first reveal, and refit on later ones.
  const whenVisible = (page, container, elements) => {
    const visible = () => !page.classList.contains("hidden") && container.offsetParent !== null;
    const run = () => {
      if (!visible()) return false;
      if (!cy) draw(container, elements);
      else refit();
      return true;
    };
    if (run()) return;
    // `.page` visibility is a class app.js toggles, so the class list is the
    // signal. No polling, and nothing for app.js to have to call.
    const observer = new MutationObserver(() => { if (run()) observer.disconnect(); });
    observer.observe(page, { attributes: true, attributeFilter: ["class"] });
  };

  // Idempotent: deltaInitWhyGraph is exported, and every call used to add
  // another observer — a MutationObserver while the page was hidden, and now a
  // ResizeObserver on the canvas — that nothing ever disconnected.
  let installed = false;

  const fail = (msg) => {
    const status = document.getElementById("wta-graph-status");
    if (status) status.textContent = msg;
  };

  const init = async () => {
    if (installed) return;
    const container = document.getElementById(CONTAINER_ID);
    const page = document.getElementById("page-why-this-app");
    const frame = container && container.closest(".wta-graph");
    if (!container || !page || !frame) return;
    if (typeof global.cytoscape !== "function") {
      fail("The map couldn't load.");
      return;
    }
    installed = true;

    let registry;
    try {
      const res = await fetch(REGISTRY_URL, { cache: "no-cache" });
      if (!res.ok) throw new Error(String(res.status));
      registry = await res.json();
    } catch (_) {
      // The registry is static content served beside index.html, so this is a
      // deploy fault rather than a session fault. Say so and stop; there is
      // nothing to draw and no learner state to fall back on.
      fail("The map couldn't load.");
      installed = false;
      return;
    }

    const elements = buildElements(registry);
    wireMaximise(frame, page);
    whenVisible(page, container, elements);
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
