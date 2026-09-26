/* concept-graph/kg-look.js — how the Knowledge Graph DRAWS, kept apart from
 * what it draws (lesson-graph.js) and which part it shows (graph-views.js).
 *
 * Three things, all cosmetic, all swappable without touching either file's
 * logic (Seth, 2026-09-25: "make it modular … a setting to swap between the
 * two different versions for the nodes"):
 *
 *   1. NODE LOOK — a learner setting, remembered in localStorage:
 *        "label" — the original: a rounded box with the concept's name inside,
 *                  filled with its colour. The colour is the whole node, so
 *                  it reads at a glance.
 *        "dot"   — the AISC write-up's small circle with the name underneath.
 *      `node(scale, ink)` returns the Cytoscape style for the current look;
 *      a switch fires `delta:kg-look-changed` and the owners rebuild their
 *      stylesheets and re-run their layouts (a box and a dot are different
 *      sizes, so dagre has to place them again).
 *
 *   2. EDGES — soft slate instead of solid red, width by encompassing
 *      weight, drawn along ROUTES: after dagre, kg-layout.js (in a Worker)
 *      moves the nodes off dagre's rows and routes every edge around the
 *      nodes to cut crossings; the nodes glide there and each edge becomes an
 *      unbundled bezier through its route's control points (`routeLayout`).
 *      An edge with no route (dragged node, shortcut lit on a chain, an
 *      instructor's proposal) is an S-bend leaving and entering vertically.
 *      Control points follow node positions once per animation frame while
 *      anything moves (`curve`).
 *
 *   3. SHORTCUTS — an edge A→C is a shortcut when A already reaches C through
 *      another path (A→B→C). A quarter of the graph's links are shortcuts
 *      (66 of 268 on 2026-09-25) and they carry nothing the chain doesn't:
 *      they get the `kg-shortcut` class, are hidden unless on a highlighted
 *      chain, and are left OUT of the layout, so dagre routes around fewer
 *      crossings. A learner can show them again from the legend's Key.
 *
 * 🔴 Styles only. Nothing here adds, removes or reorders elements; the
 * instructor editor's add/remove contract (lesson-graph.js) is untouched.
 * Control points are per-edge style bypasses on `control-point-*` only. */
(function () {
  "use strict";

  const LOOK_KEY = "dd_kg_node_look";
  const SHORTCUT_KEY = "dd_kg_shortcuts";
  const LOOKS = ["label", "dot"];
  let look = "label";
  let showShortcuts = false;
  try { const v = localStorage.getItem(LOOK_KEY); if (LOOKS.includes(v)) look = v; } catch (_) {}
  try { showShortcuts = localStorage.getItem(SHORTCUT_KEY) === "1"; } catch (_) {}

  const fire = () => window.dispatchEvent(new CustomEvent("delta:kg-look-changed", { detail: { look, showShortcuts } }));

  /* ---------------- node look ------------------------------------------ */
  // `scale` 1 = the main canvas; the condensed view's opened sections draw
  // their concepts a little smaller. `ink` is a function (the pane's text
  // colour, re-read on a theme switch) and only matters for the dot, whose
  // label sits on the canvas rather than on its own fill.
  const node = (scale, ink) => {
    const k = scale || 1;
    if (look === "dot") {
      return {
        "shape": "ellipse", "label": "data(label)",
        "width": 28 * k, "height": 28 * k, "padding": "0px",
        "text-wrap": "wrap", "text-max-width": 118 * k + "px",
        "text-valign": "bottom", "text-halign": "center", "text-margin-y": 5 * k,
        "font-size": 12.5 * k, "font-weight": 600, "line-height": 1.2,
        "color": typeof ink === "function" ? ink : () => "#c8cdd8",
        "text-outline-width": 0,
        "border-width": 1, "border-color": "rgba(0,0,0,.28)",
      };
    }
    return {
      "shape": "round-rectangle", "label": "data(label)",
      "width": "label", "height": "label", "padding": 11 * k + "px",
      "text-wrap": "wrap", "text-max-width": 120 * k + "px",
      "text-valign": "center", "text-halign": "center", "text-margin-y": 0,
      "font-size": 12.5 * k, "font-weight": 600, "line-height": 1.25,
      "color": "#15151f", "text-outline-width": 0,
      "border-width": 1, "border-color": "rgba(0,0,0,.22)",
    };
  };

  /* ---------------- edges ---------------------------------------------- */
  // Theme-aware slate, read from `--kg-edge` on the graph pane
  // (styles/kg-toolbar.css) so light and dark each get a line that recedes.
  let edgeInk = "#8f9bb8";
  const readEdgeInk = () => {
    const g = document.querySelector(".kg2-graph");
    const v = g ? getComputedStyle(g).getPropertyValue("--kg-edge").trim() : "";
    if (v) edgeInk = v;
  };
  window.addEventListener("delta:theme-changed", () => { readEdgeInk(); fire(); });

  // The base edge rule, and the two kinds on top of it. Encompassing edges are
  // the load-bearing ones and scale with their weight; a plain prerequisite is
  // a hairline. Shortcuts hide, except on a highlighted chain (`.hl`, set by
  // lesson-graph.js's selectNode) or while graph-views.js's fan lights them.
  const edgeRules = () => [
    { selector: "edge", style: {
        "curve-style": "unbundled-bezier", "edge-distances": "node-position",
        "control-point-distances": 0, "control-point-weights": 0.5,
        "width": 1.4, "line-color": () => edgeInk, "target-arrow-color": () => edgeInk,
        "target-arrow-shape": "triangle", "arrow-scale": 0.7, "opacity": 0.55,
        "line-cap": "round",
    }},
    { selector: "edge[kind = 'prereq']", style: { "width": 1, "opacity": 0.38 } },
    { selector: "edge[kind = 'encompassing']", style: {
        "width": (e) => 1.3 + 3.4 * (e.data("w") || 0), "opacity": (e) => 0.5 + 0.35 * (e.data("w") || 0),
    }},
    { selector: "edge.kg-shortcut", style: { "display": () => (showShortcuts ? "element" : "none") } },
    { selector: "edge.kg-shortcut.hl", style: { "display": "element" } },
  ];

  /* ---------------- shortcuts ------------------------------------------ */
  // An edge is a shortcut when its source reaches its target through a
  // longer chain. A second direct s→t edge is not a chain: counting it would
  // hide both. 125 nodes × 268 edges: a DFS per edge is nothing.
  //
  // Only edges present at the first mark may be hidden. Edges added later are
  // instructor-graph-edit.js's proposals (drawn green), which must stay in
  // view even when a chain implies them; they still count as chains.
  const markShortcuts = (cy) => {
    if (!cy) return 0;
    if (!cy.__kgBaseEdges) {
      cy.__kgBaseEdges = new Set(cy.edges().map((e) => e.id()));
      // Re-mark when the topology changes (instructor edits add, remove and
      // restore edges), once per frame.
      let queued = false;
      cy.on("add remove", "edge", () => {
        if (queued) return;
        queued = true;
        requestAnimationFrame(() => { queued = false; if (!cy.destroyed()) markShortcuts(cy); });
      });
    }
    const base = cy.__kgBaseEdges;
    const out = {};
    cy.edges().forEach((e) => { (out[e.source().id()] = out[e.source().id()] || []).push(e); });
    let n = 0;
    cy.batch(() => cy.edges().forEach((e) => {
      const s = e.source().id(), t = e.target().id();
      if (!base.has(e.id())) { e.removeClass("kg-shortcut"); return; }
      const seen = new Set([s]);
      const stack = (out[s] || []).filter((x) => x.target().id() !== t).map((x) => x.target().id());
      let reach = false;
      while (stack.length && !reach) {
        const v = stack.pop();
        if (v === t) { reach = true; break; }
        if (seen.has(v)) continue;
        seen.add(v);
        (out[v] || []).forEach((x) => stack.push(x.target().id()));
      }
      e.toggleClass("kg-shortcut", reach);
      if (reach) n += 1;
    }));
    return n;
  };
  // What a layout should see: everything but the hidden shortcuts.
  const layoutEles = (cy) => (showShortcuts ? cy.elements() : cy.elements().not("edge.kg-shortcut"));

  /* ---------------- layout with routes -------------------------------- */
  // Two stages. dagre first, synchronously: it places the nodes in rows and
  // ROUTES every edge that spans more than one row through bend points it
  // ordered to cut crossings (cytoscape-dagre throws those away; this keeps
  // them). Then kg-layout.js, in a Worker: it moves nodes off the rows, up,
  // down and sideways, and routes every edge on a grid around the nodes,
  // lowering the drawn crossings (608 → ~120 on the full graph, 2026-09-25).
  // When it answers, the nodes glide to their new places and the edges take
  // its routes. Its answer is cached by the graph, so a return visit skips
  // straight to it. Callers still get an ordinary `preset` layout back, so
  // `.run()`, `.stop()`, `layoutstop` and the animation are theirs as before.
  //
  // Routes are control points in model coordinates, per edge id, on the
  // graph (`cy.__kgRoutes`); they hold only for the positions they were
  // computed for, so a node dragged by hand drops its own.
  const ptsToCps = (s, t, pts) => {
    const mid = pts.slice(1, -1);
    if (mid.length < 2) return null;
    const a = mid[0], z = mid[mid.length - 1];
    return [{ x: s.x, y: (s.y + a.y) / 2 }, ...mid, { x: t.x, y: (t.y + z.y) / 2 }];
  };
  // What kg-layout.js needs from a node: its box with the label, and the
  // shape alone (a dot's label hangs under it, so its edges enter the side).
  const layoutInput = (eles, pos) => {
    const nodes = eles.nodes().map((n) => {
      const bb = n.boundingBox({ includeLabels: true, includeOverlays: false });
      const p = n.position(), q = pos[n.id()] || p;
      return { id: n.id(), x: q.x, y: q.y, w: Math.max(1, bb.w), h: Math.max(1, bb.h),
        ox: (bb.x1 + bb.x2) / 2 - p.x, oy: (bb.y1 + bb.y2) / 2 - p.y,
        cw: n.outerWidth(), ch: n.outerHeight() };
    });
    const edges = eles.edges().map((e) => ({ id: e.id(), s: e.source().id(), t: e.target().id() }));
    return { nodes, edges, opt: { route: { entry: look === "dot" ? "side" : "bottom" } } };
  };
  // A pointer or wheel on the canvas after a layout started: the learner has
  // taken the view, so a late refinement must not refit it.
  const watchTouch = (cy) => {
    if (cy.__kgTouchWatch) return;
    cy.__kgTouchWatch = true;
    const mark = () => { cy.__kgTouched = Date.now(); };
    const el = cy.container();
    if (el) { el.addEventListener("wheel", mark, { passive: true }); el.addEventListener("pointerdown", mark); }
  };
  const routeLayout = (cy, o) => {
    const eles = layoutEles(cy);
    const g = new window.dagre.graphlib.Graph({ multigraph: true });
    g.setGraph({ rankdir: o.rankDir || "BT", nodesep: o.nodeSep || 26, ranksep: o.rankSep || 150,
                 edgesep: o.edgeSep || 12, ranker: o.ranker || "network-simplex" });
    g.setDefaultEdgeLabel(() => ({}));
    eles.nodes().forEach((n) => {
      const bb = n.boundingBox({ includeLabels: true, includeOverlays: false });
      g.setNode(n.id(), { width: Math.max(1, bb.w), height: Math.max(1, bb.h) });
    });
    eles.edges().forEach((e) => {
      g.setEdge(e.source().id(), e.target().id(), { weight: 1, minlen: 1 }, e.id());
    });
    window.dagre.layout(g);
    const pos = {};
    g.nodes().forEach((id) => { const d = g.node(id); if (d) pos[id] = { x: d.x, y: d.y }; });
    const dagreRoutes = {};
    g.edges().forEach((ed) => {
      const d = g.edge(ed);
      if (!d || !d.points || !pos[ed.v] || !pos[ed.w]) return;
      const cps = ptsToCps(pos[ed.v], pos[ed.w], d.points);
      if (cps) dagreRoutes[ed.name] = { cps };
    });

    const gen = (cy.__kgGen = (cy.__kgGen || 0) + 1);
    // A glide from the layout before this one would keep writing positions.
    if (cy.__kgGlide) { try { cy.__kgGlide.stop(); } catch (_) {} cy.__kgGlide = null; }
    const started = Date.now();
    watchTouch(cy);
    const L = window.DeltaKgLayout;
    const input = L ? layoutInput(eles, pos) : null;
    const hit = L ? L.cached(input.nodes, input.edges, input.opt) : null;
    const target = hit ? hit.pos : pos;
    const anim = {
      fit: false, padding: o.padding || 40, animate: !!o.animate,
      animationDuration: o.animationDuration || 320, animationEasing: o.animationEasing || "ease-out",
    };
    // While nodes move, edges draw as plain S-bends; routes land at the end.
    cy.__kgRoutes = {};
    const lay = eles.nodes().layout(Object.assign({ name: "preset", positions: (n) => target[n.id()] || n.position() }, anim));
    lay.one("layoutstop", () => {
      if (cy.destroyed() || cy.__kgGen !== gen) return;
      // A copy: dragging a node deletes entries, and the cache's own map
      // must survive for the next time this layout is asked for.
      cy.__kgRoutes = Object.assign({}, hit ? hit.routes : dagreRoutes);
      if (hit) cy.__kgLayoutStats = Object.assign({ cached: true }, hit.stats);
      kickCurve(cy);
      if (hit || !L) return;
      // A tick later: a caller that stops this layout to start the next one
      // fires this `layoutstop` first, and the next layout's generation
      // then says this one is stale before any worker is asked.
      setTimeout(() => { if (!cy.destroyed() && cy.__kgGen === gen) refine(); }, 0);
    });
    const refine = () => {
      const t0 = Date.now();
      L.run(input.nodes, input.edges, input.opt).then((res) => {
        if (cy.destroyed() || cy.__kgGen !== gen) return;
        cy.__kgLayoutStats = Object.assign({ ms: Date.now() - t0, cached: !!res.cached }, res.stats);
        cy.__kgRoutes = {};
        const glide = eles.nodes().filter((n) => !n.removed()).layout({
          name: "preset", positions: (n) => res.pos[n.id()] || n.position(), fit: false,
          animate: true, animationDuration: 650, animationEasing: "ease-in-out-cubic",
        });
        glide.one("layoutstop", () => {
          if (cy.destroyed() || cy.__kgGen !== gen) return;
          cy.__kgRoutes = Object.assign({}, res.routes);
          kickCurve(cy);
          if (typeof o.refit === "function" && !(cy.__kgTouched > started)) o.refit();
        });
        cy.__kgGlide = glide;
        glide.run();
      }).catch(() => { /* superseded, or no Worker: dagre's layout stands */ });
    };
    return lay;
  };
  // A node dragged by hand invalidates the routes through it (on the first
  // move — `grab` also fires on a plain tap, which moves nothing), and any
  // other edge's route it was dropped onto (free); those edges fall back to
  // the S-bend until the next layout.
  const dropRoutes = (node) => {
    const r = node.cy().__kgRoutes;
    if (r) node.connectedEdges().forEach((e) => { delete r[e.id()]; });
  };
  const dropRoutesUnder = (node) => {
    const cy = node.cy(), r = cy.__kgRoutes;
    if (!r) return;
    const bb = node.boundingBox({ includeLabels: true });
    const inside = (p) => p.x > bb.x1 && p.x < bb.x2 && p.y > bb.y1 && p.y < bb.y2;
    Object.keys(r).forEach((id) => {
      const e = cy.getElementById(id);
      if (!e.length || e.source().same(node) || e.target().same(node)) return;
      if (curvePts(e.source().position(), e.target().position(), r[id].cps).some(inside)) delete r[id];
    });
  };

  /* ---------------- curves --------------------------------------------- */
  // Points along the curve Cytoscape draws for an unbundled bezier: from the
  // source through quadratics whose ends are the midpoints of consecutive
  // control points, `k` samples each, at most 4 px apart.
  const curvePts = (s, t, cps) => {
    const out = [];
    let cur = s;
    for (let i = 0; i <= cps.length; i++) {
      const last = i >= cps.length - 1;
      const end = last ? t : { x: (cps[i].x + cps[i + 1].x) / 2, y: (cps[i].y + cps[i + 1].y) / 2 };
      const ctl = i < cps.length ? cps[i] : { x: (cur.x + end.x) / 2, y: (cur.y + end.y) / 2 };
      const k = Math.max(4, Math.ceil((Math.hypot(ctl.x - cur.x, ctl.y - cur.y) + Math.hypot(end.x - ctl.x, end.y - ctl.y)) / 4));
      for (let f = 0; f <= k; f++) {
        const a = f / k, b = 1 - a;
        out.push({ x: b * b * cur.x + 2 * b * a * ctl.x + a * a * end.x, y: b * b * cur.y + 2 * b * a * ctl.y + a * a * end.y });
      }
      cur = end;
      if (last) break;
    }
    return out;
  };
  // Control points as Cytoscape's unbundled-bezier wants them: a weight along
  // the source→target line and a signed distance off it.
  const toCtrl = (s, t, pts) => {
    const dx = t.x - s.x, dy = t.y - s.y, L2 = dx * dx + dy * dy;
    if (L2 < 1) return null;
    const L = Math.sqrt(L2), d = [], w = [];
    pts.forEach((p) => {
      const rx = p.x - s.x, ry = p.y - s.y;
      w.push((rx * dx + ry * dy) / L2);
      d.push((ry * dx - rx * dy) / L);
    });
    return { d: d.map((v) => v.toFixed(1)).join(" "), w: w.map((v) => v.toFixed(3)).join(" ") };
  };
  // An edge between neighbouring ranks: an S-bend, both control points at
  // mid-height, one over the source and one under the target, so it leaves
  // and arrives vertically.
  const bend = (s, t) => {
    if (Math.abs(t.x - s.x) < 1.5) return null;
    const my = (s.y + t.y) / 2;
    return toCtrl(s, t, [{ x: s.x, y: my }, { x: t.x, y: my }]);
  };
  const curveNow = (cy) => {
    cy.batch(() => cy.edges().forEach((e) => {
      if (e.removed()) return;
      const s = e.source().position(), t = e.target().position();
      const r = cy.__kgRoutes && cy.__kgRoutes[e.id()];
      const b = r ? (r.cps.length ? toCtrl(s, t, r.cps) : null) : bend(s, t);
      if (b) e.style({ "control-point-distances": b.d, "control-point-weights": b.w });
      else e.style({ "control-point-distances": "0", "control-point-weights": "0.5" });
    }));
  };
  // Once per frame at most, however many nodes moved in it.
  const kickCurve = (cy) => { if (cy.__kgKick) cy.__kgKick(); };
  const curve = (cy) => {
    if (!cy || cy.__kgCurved) return;
    cy.__kgCurved = true;
    let queued = false;
    const kick = () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; if (!cy.destroyed()) curveNow(cy); });
    };
    cy.__kgKick = kick;
    cy.on("position", "node", kick);
    cy.on("add", "edge", kick);
    cy.on("layoutstop", kick);
    // The first move of a drag: drop this node's routes, and cancel a
    // refinement still on its way (it would glide the node back).
    let dragging = null;
    cy.on("drag", "node", (evt) => {
      if (dragging === evt.target) return;
      dragging = evt.target;
      dropRoutes(evt.target);
      if (cy.__kgGlide) { try { cy.__kgGlide.stop(); } catch (_) {} cy.__kgGlide = null; }
      cy.__kgGen = (cy.__kgGen || 0) + 1;
    });
    cy.on("free", "node", (evt) => {
      if (dragging !== evt.target) return;
      dragging = null;
      dropRoutesUnder(evt.target);
      kick();
    });
    kick();
  };

  /* ---------------- API ------------------------------------------------- */
  readEdgeInk();
  window.DeltaKgLook = {
    look: () => look,
    looks: () => LOOKS.slice(),
    setLook: (v) => {
      if (!LOOKS.includes(v) || v === look) return;
      look = v;
      try { localStorage.setItem(LOOK_KEY, v); } catch (_) {}
      fire();
    },
    shortcutsShown: () => showShortcuts,
    setShortcuts: (on) => {
      showShortcuts = !!on;
      try { localStorage.setItem(SHORTCUT_KEY, showShortcuts ? "1" : "0"); } catch (_) {}
      fire();
    },
    node,
    edgeRules,
    edgeInk: () => { readEdgeInk(); return edgeInk; },
    markShortcuts,
    layoutEles,
    // Null when dagre isn't loaded; callers fall back to their own layout.
    routeLayout: (cy, o) => (window.dagre && window.dagre.layout ? routeLayout(cy, o || {}) : null),
    curve,
  };
})();
