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
 *   2. EDGES — soft slate instead of solid red, curved S-bends that leave and
 *      enter each node vertically, width by encompassing weight. The curve is
 *      an unbundled bezier whose two control points sit at the source's and
 *      the target's x on the edge's mid-height; they are recomputed from node
 *      positions once per animation frame while anything moves (`curve`).
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
  // dagre does two jobs: it places the nodes, and it ROUTES every edge that
  // spans more than one rank through a chain of bend points placed between
  // that rank's nodes, ordering them to cut crossings. cytoscape-dagre keeps
  // the first and throws the second away, so a long edge was drawn straight
  // through everything dagre had steered it around. This runs dagre itself,
  // keeps the routes (`routes`, model coordinates, keyed by edge id), and
  // hands back an ordinary `preset` layout for the nodes — so callers keep
  // `.run()`, `.stop()`, `layoutstop` and the animation.
  let routes = {};
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
    routes = {};
    g.edges().forEach((ed) => { const d = g.edge(ed); if (d && d.points) routes[ed.name] = d.points; });
    return eles.nodes().layout({
      name: "preset", positions: (n) => pos[n.id()] || n.position(), fit: false, padding: o.padding || 40,
      animate: !!o.animate, animationDuration: o.animationDuration || 320, animationEasing: o.animationEasing || "ease-out",
    });
  };
  // A node dragged by hand invalidates the routes through it; its edges fall
  // back to the S-bend until the next layout.
  const dropRoutes = (node) => node.connectedEdges().forEach((e) => { delete routes[e.id()]; });

  /* ---------------- curves --------------------------------------------- */
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
  // A routed edge: through dagre's bend points, entering and leaving
  // vertically like the S-bend (a point straight off each end, half-way to the
  // first/last bend). Only the INTERIOR points are dagre's; its first and last
  // sit on the node borders.
  const routed = (s, t, pts) => {
    const mid = pts.slice(1, -1);
    if (mid.length < 2) return null;
    const a = mid[0], z = mid[mid.length - 1];
    return toCtrl(s, t, [{ x: s.x, y: (s.y + a.y) / 2 }, ...mid, { x: t.x, y: (t.y + z.y) / 2 }]);
  };
  const curveNow = (cy) => {
    cy.batch(() => cy.edges().forEach((e) => {
      if (e.removed()) return;
      const s = e.source().position(), t = e.target().position();
      const r = routes[e.id()];
      const b = (r && routed(s, t, r)) || bend(s, t);
      if (b) e.style({ "control-point-distances": b.d, "control-point-weights": b.w });
      else e.style({ "control-point-distances": "0", "control-point-weights": "0.5" });
    }));
  };
  // Once per frame at most, however many nodes moved in it.
  const curve = (cy) => {
    if (!cy || cy.__kgCurved) return;
    cy.__kgCurved = true;
    let queued = false;
    const kick = () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; if (!cy.destroyed()) curveNow(cy); });
    };
    cy.on("position", "node", kick);
    cy.on("add", "edge", kick);
    cy.on("layoutstop", kick);
    cy.on("grab", "node", (evt) => dropRoutes(evt.target));
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
