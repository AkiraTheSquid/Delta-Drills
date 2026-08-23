/* ================================================================
   WHY-GRAPH.JS — the map on "Why this app exists"

   A PREVIEW of the Knowledge Graph, on a page anyone can read.
   It draws the whole lesson graph — every knowledge component in
   lessons/kc_registry.json and every prerequisite edge — with no
   side panel, no learner-model dock and no lesson pane. Two colour
   readings, chosen at the bottom of the map:

     COLD START  (the default) — the map as it stands before anyone
        has answered anything: every concept grey and dashed,
        because "no estimate" is what the model actually holds
        then. This is the honest picture of what the app knows
        about a visitor who has just arrived, and it is the picture
        the page is arguing about.
     YOUR MASTERY — the same reading the Knowledge Graph tab uses,
        borrowed from lesson-graph.js via window.deltaKcReadinessInfo
        so the two surfaces cannot disagree about the same learner.

   Nothing here is invented. An earlier version coloured the map
   from graph depth plus a hash, as an illustration; that is gone,
   because a made-up frontier on the page that explains the
   frontier is the one thing this page must not do.

   MAXIMIZE hands the whole window to the REAL graph: it moves
   `.kg-container.kg2` out of the Knowledge Graph tab and into this
   frame, so what fills the screen is lesson-graph.js itself —
   side panel, dock, gate ticks, Mastery/Lessons switch — and puts
   it back on the way out. Moved, never copied: the graph is live
   Cytoscape state, and a second instance would be a second
   learner-model reader to keep in step.

   It still borrows lesson-graph.js's LOOK by copy for the preview
   — same round-rectangle nodes, same red→blue ramp, same red
   prerequisite arrows. If that styling changes over there and this
   page starts looking like a different product, this is the file
   to update.
   ================================================================ */

(function installWhyGraph(global) {
  const CONTAINER_ID = "wta-graph-cy";
  const REGISTRY_URL = "lessons/kc_registry.json";
  const KG_SELECTOR = ".kg-container.kg2";

  // Straight from lesson-graph.js so the two maps read as one product.
  const ACCENT = "#ffd23f";                  // prerequisite-path highlight
  const UNKNOWN_COLOR = "#5b5b70";           // no estimate
  const EDGE_COLOR = "#e3212c";
  // red (low) → muted purple → blue (high)
  const masteryColor = (r) => {
    if (!Number.isFinite(r)) return UNKNOWN_COLOR;
    const t = Math.max(0, Math.min(1, r));
    const lo = [214, 72, 72], hi = [59, 130, 246];
    const c = lo.map((v, i) => Math.round(v + (hi[i] - v) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
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
    const elements = kcs.map((k) => ({
      data: { id: k.id, label: shortLabel(k.title), title: k.title },
    }));
    let i = 0;
    kcs.forEach((k) => {
      (k.prereqs || []).forEach((p) => {
        if (byId[p]) elements.push({ data: { id: `wg-e${i++}`, source: p, target: k.id } });
      });
    });
    return elements;
  };

  /* ---- the two readings ------------------------------------------------
     "Cold start" is not a fake dataset: it is what kcReadinessInfo answers
     for a learner with no history — r = NaN, source "none" — so it is drawn
     the way lesson-graph.js draws that case, grey at 0.42 opacity with a
     dashed border. "Your mastery" asks lesson-graph.js itself. If that file
     has not loaded, the switch degrades to cold rather than inventing a
     number.
     --------------------------------------------------------------------- */
  let mode = "cold";

  const readingFor = (id) => {
    if (mode !== "mine") return { r: NaN, measured: false };
    const info = typeof global.deltaKcReadinessInfo === "function"
      ? global.deltaKcReadinessInfo(id)
      : null;
    const measured = typeof global.deltaKcIsMeasured === "function"
      ? !!global.deltaKcIsMeasured(id)
      : false;
    return { r: info ? info.r : NaN, measured };
  };

  let cy = null;

  const applyColours = () => {
    if (!cy) return;
    cy.batch(() => cy.nodes().forEach((n) => {
      const { r, measured } = readingFor(n.id());
      n.style({
        "background-color": masteryColor(r),
        // Same test lesson-graph.js uses: less than one attempt's worth of
        // evidence bearing on this concept is an inference, not a measurement,
        // and must never look like one.
        "background-opacity": measured ? 1 : 0.42,
        "border-style": measured ? "solid" : "dashed",
      });
    }));
    refreshNote();
  };

  const anyMeasured = () => {
    if (!cy || typeof global.deltaKcIsMeasured !== "function") return false;
    return cy.nodes().some((n) => global.deltaKcIsMeasured(n.id()));
  };

  // The corner note. In cold start it says what the view is; switched to the
  // learner's own reading with nothing behind it, it says so plainly rather
  // than leaving a grey map to be read as a bug.
  const refreshNote = () => {
    const el = document.getElementById("wta-graph-nodata");
    if (!el) return;
    if (mode === "cold") {
      el.textContent = "Cold start — the map before anyone has answered anything.";
      el.hidden = false;
      return;
    }
    const has = anyMeasured();
    // Borrowed, not restated: lesson-graph.js decides what this says, because a
    // learner who has finished the placement test HAS answered problems and
    // both maps were telling them otherwise. Falls back to the old wording only
    // when lesson-graph.js has not loaded, where the old wording is correct.
    el.textContent = typeof global.deltaKcNoDataText === "function"
      ? global.deltaKcNoDataText()
      : "No problems answered yet — nothing on this map is measured.";
    el.hidden = has;
  };

  const wireModes = () => {
    const box = document.getElementById("wta-graph-modes");
    if (!box) return;
    box.querySelectorAll("button[data-wta-mode]").forEach((b) => {
      b.addEventListener("click", () => {
        if (mode === b.dataset.wtaMode) return;
        mode = b.dataset.wtaMode;
        box.querySelectorAll("button[data-wta-mode]").forEach((x) => {
          const on = x === b;
          x.classList.toggle("active", on);
          x.setAttribute("aria-pressed", on ? "true" : "false");
        });
        applyColours();
        // The learner's reading needs the server's report, and on THIS page
        // nothing has fetched it: lesson-graph.js fetches inside build(), which
        // only runs on the Knowledge Graph tab. Without this the switch showed
        // a signed-in learner the offline answer — grey where the queue holds a
        // number. Both calls are cached and idempotent; paint again when they
        // land, because the first paint above is the honest read of what is
        // known right now. It goes through lesson-graph.js rather than the
        // loader directly, so that file's own `lattice` is updated too — see
        // deltaRefreshKcLattice.
        if (mode !== "mine") return;
        if (typeof global.deltaRefreshKcLattice === "function") {
          global.deltaRefreshKcLattice().then(applyColours);
        }
      });
    });
  };

  /* ---- layout ---------------------------------------------------------- */
  // Tighter than lesson-graph.js's spacing, and the labels are cut at the
  // colon: this view fits all 63 concepts at once and every pixel of node
  // width costs zoom. BT so prerequisites sit BENEATH what they unlock, which
  // is what the page's copy says and what the Knowledge Graph tab does.
  const LAYOUT = {
    name: "dagre",
    rankDir: "BT", nodeSep: 12, rankSep: 58, edgeSep: 8,
    animate: false, fit: true, padding: 24,
  };
  const layoutOpts = () =>
    (global.cytoscapeDagre ? LAYOUT : { name: "cose", animate: false, fit: true, padding: 24 });

  const refit = () => {
    if (!cy) return;
    const c = cy.container();
    if (!c || !c.clientWidth || !c.clientHeight) return;
    cy.resize();
    cy.fit(undefined, 24);
  };

  const draw = (container, elements) => {
    cy = global.cytoscape({
      container,
      elements,
      wheelSensitivity: 0.2,
      minZoom: 0.08,
      maxZoom: 2.5,
      style: [
        { selector: "node", style: {
            "background-color": UNKNOWN_COLOR,
            "background-opacity": 0.42,
            "shape": "round-rectangle",
            "label": "data(label)",
            "width": "label", "height": "label", "padding": "9px",
            "text-wrap": "wrap", "text-max-width": "96px",
            "text-valign": "center", "text-halign": "center",
            "font-size": 12, "font-weight": 600, "color": "#15151f",
            "border-width": 1, "border-color": "rgba(0,0,0,.28)",
            "border-style": "dashed",
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

    applyColours();
    const status = document.getElementById("wta-graph-status");
    if (status) status.style.display = "none";
  };

  /* ---- maximize: hand the window to the real graph ----------------------
     `.kg-container.kg2` is MOVED here and moved back, the way nav-drawer.js
     moves the tab strip: it is live Cytoscape state with a learner-model dock
     reading it, and a copy would be a second reader to keep in step. Moving
     it also means every behaviour of the real graph — the lesson pane, the
     gate ticks, the Practice hand-off — arrives for free and cannot drift
     from the Knowledge Graph tab, because it IS the Knowledge Graph tab.

     `fitWrap()` in lesson-graph.js looks the wrap up as `.kg2 .kg2-wrap`, so
     the element that moves has to be the `.kg2` container and not the wrap
     inside it, or the graph loses its height the moment it arrives.
     ----------------------------------------------------------------------- */
  let kgHome = null;   // where to put the Knowledge Graph back

  const hostKg = (frame) => {
    const kg = document.querySelector(KG_SELECTOR);
    if (!kg || kgHome) return false;
    kgHome = { parent: kg.parentNode, next: kg.nextSibling };
    frame.appendChild(kg);
    frame.classList.add("is-hosting-kg");
    // Builds on first use; on later ones it just resizes and refits.
    if (typeof global.deltaInitConceptGraph === "function") global.deltaInitConceptGraph();
    return true;
  };

  const releaseKg = (frame) => {
    if (!kgHome) return;
    const kg = frame.querySelector(KG_SELECTOR);
    // insertBefore(node, null) appends, which is the right answer when the
    // graph was the last child of its page.
    if (kg) kgHome.parent.insertBefore(kg, kgHome.next);
    kgHome = null;
    frame.classList.remove("is-hosting-kg");
  };

  const setMaximised = (frame, btn, on) => {
    frame.classList.toggle("is-max", on);
    document.body.classList.toggle("wta-max-open", on);
    if (btn) {
      btn.setAttribute("aria-pressed", on ? "true" : "false");
      btn.setAttribute("aria-label", on ? "Minimize the map" : "Maximize the map");
      btn.title = on ? "Minimize" : "Maximize";
      btn.textContent = on ? "⤡ Minimize" : "⤢ Maximize";
    }
    // If the Knowledge Graph is not in this document for some reason, the
    // preview fills the screen instead — a smaller thing, but not nothing.
    if (on) hostKg(frame);
    else releaseKg(frame);
    // Refit straight away rather than from a requestAnimationFrame: rAF does
    // not fire in a hidden tab, so the deferred version left the graph
    // full-screen at the band's zoom. Cytoscape's resize() reads the
    // container's offset size, which forces the pending reflow, and refit()
    // no-ops while the preview canvas is the hidden one.
    refit();
    // Focus follows the surface: maximising with the keyboard should leave the
    // focus on the control that gets you back out.
    if (on && btn) btn.focus();
  };

  // Everything inside the maximised frame a keyboard can land on, in document
  // order. `offsetParent` drops what is hidden — the preview canvas and its
  // notes while the real graph is hosted, and the reverse — so the cycle only
  // ever contains what is actually on screen.
  const TAB_STOPS =
    'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]),' +
    ' textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
  const tabStops = (frame) =>
    Array.from(frame.querySelectorAll(TAB_STOPS)).filter((el) => el.offsetParent !== null);

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
      // The frame's own controls still have to be reachable — hosting the real
      // Knowledge Graph puts a dozen of them in here — so this is a wrap, not
      // a pin: Tab cycles what is inside the frame and never leaves it.
      if (evt.key !== "Tab") return;
      const stops = tabStops(frame);
      if (!stops.length) { evt.preventDefault(); btn.focus(); return; }
      const first = stops[0];
      const last = stops[stops.length - 1];
      const at = stops.indexOf(document.activeElement);
      if (at === -1) { evt.preventDefault(); (evt.shiftKey ? last : first).focus(); return; }
      if (evt.shiftKey && document.activeElement === first) { evt.preventDefault(); last.focus(); }
      else if (!evt.shiftKey && document.activeElement === last) { evt.preventDefault(); first.focus(); }
    });
    // Leaving the page while maximised would strand `body.wta-max-open` — the
    // whole document unscrollable — AND leave the Knowledge Graph parked in
    // this frame, so its own tab would open empty. Both are undone here.
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

    wireModes();
    refreshNote();
    wireMaximise(frame, page);
    whenVisible(page, container, buildElements(registry));

    // A graded attempt moves the learner's reading, so the map has to be
    // repainted with it — but only when it is the learner's reading on show.
    // TWO events, and both are needed. `delta:adaptive-state-changed` is the
    // attempt itself, which is all there is when the Knowledge Graph has never
    // been built. `delta:kc-readiness-changed` is lesson-graph.js saying its
    // lattice refresh has landed — repainting on the first event alone reads
    // the numbers it is in the middle of replacing.
    const repaint = () => { if (mode === "mine") applyColours(); };
    global.addEventListener("delta:adaptive-state-changed", repaint);
    global.addEventListener("delta:kc-readiness-changed", repaint);
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
