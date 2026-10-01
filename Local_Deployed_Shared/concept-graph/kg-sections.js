/* concept-graph/kg-sections.js — AREA NAMES on the Knowledge Graph: each
 * area's title, large, in the area's colour, sitting among its concepts.
 *
 * Seth, 2026-10-01: "close to the center of the nodes it essentially has the
 * title of the section … it doesn't have the rectangle … the title of the
 * section is the same color as the nodes … much larger text that stays in
 * that position central to all the nodes … doesn't overlap with any other
 * nodes and also still allows you to click on it". Before that (2026-09-30)
 * each area had a rectangle round it with the title on its top edge.
 * Toolbar "Area names" (kg-toolbar.js) turns them off; on by default, and
 * the choice stays in this browser (dd_kg_area_names).
 *
 *   Where   — the free spot nearest the CENTRE of the area's drawn concepts
 *             (their mean position): a title never covers a bubble or its
 *             label, nor another title. Worked out in graph units and kept
 *             while you pan; redone when the zoom, a node, or what is shown
 *             changes. The layout itself is never touched.
 *   Size    — scales with the zoom (between 15 and 34 px on screen), so it
 *             reads as part of the map, not as a floating badge.
 *   Colour  — Sections mode: the area's own colour (the one its bubbles
 *             wear). Mastery mode: the mastery ramp at the area's AVERAGE
 *             reading, so a title says how strong you are there.
 *   Click   — calls `window.deltaSelectKgSection(sid)` (lesson-graph.js):
 *             every concept in the area lights up and the side panel shows
 *             the area's diagnostics (kg-panel.js). A wheel over a title
 *             still zooms the graph.
 *
 * One layer of buttons over #kg-cy. Hidden in the condensed view (it has
 * its own canvas).
 *
 * 🔴 Reads the graph, never writes it: no elements, no styles, no classes and
 * no positions on lesson-graph.js's instance. */
(function () {
  "use strict";

  const SVGNS = "http://www.w3.org/2000/svg";
  // Short headers. The section objects' own labels are sentences written for
  // the old legend ("Section 0.1 — ARENA's ray tracing"); on the map a header
  // must read at a glance.
  const SHORT = {
    sm10: "Prep −1.0 · Python",
    sm11: "Prep −1.1 · Arrays & tensors",
    sm12: "Prep −1.2 · Maths",
    s00: "Ch 0.0 · Prerequisites",
    s01: "Ch 0.1 · Ray tracing",
    s02: "Ch 0.2 · CNNs & ResNets",
    sNN: "Later ARENA chapters",
    clc: "LeetCode Patterns",
    cdd: "Delta Drills",
  };
  const label = (s) => (s && SHORT[s.id]) || (s && s.label) || "";
  const UNLOCK_T = 0.85, MASTERY_T = 0.95;

  let cy = null, heads = null;
  let active = null;             // sid whose title is selected
  let queued = false;

  const SHOW_KEY = "dd_kg_area_names";
  let shown = true;
  try { shown = localStorage.getItem(SHOW_KEY) !== "0"; } catch (_) {}
  const setShown = (v) => {
    v = !!v;
    if (v === shown) return;
    shown = v;
    try { localStorage.setItem(SHOW_KEY, v ? "1" : "0"); } catch (_) {}
    window.dispatchEvent(new CustomEvent("delta:kg-area-names-changed", { detail: { shown } }));
    if (init()) kick();
  };

  const view = () => window.deltaKgView || null;
  const colorMode = () => (typeof window.deltaKgColorMode === "function" ? window.deltaKgColorMode() : "mastery");
  const sections = () => (view() && view().sections ? view().sections() : []);

  /* ---------------- per-area numbers (also the panel's) ---------------- */
  // Mean of the readings the bubbles show (the same `deltaKcReadinessInfo`),
  // over the concepts the learner has NOT switched off. Counts per band use the
  // panel's words. `measured` = concepts with a graded attempt's worth of
  // direct evidence (lesson-graph.js `_isMeasured`).
  const stats = (sid) => {
    const sec = sections().find((s) => s.id === sid);
    if (!sec) return null;
    const L = typeof window.getKcLattice === "function" ? window.getKcLattice() : null;
    const info = window.deltaKcReadinessInfo;
    const measuredFn = window.deltaKcIsMeasured;
    let sum = 0, n = 0, measured = 0, off = 0;
    const bands = { strong: 0, learning: 0, starting: 0, none: 0 };
    const rows = sec.kcs.map((kc) => {
      const row = L && L.kcs ? L.kcs[kc] : null;
      const isOff = !!(row && row.state === "disabled");
      const r = typeof info === "function" ? (info(kc) || {}).r : NaN;
      const m = typeof measuredFn === "function" ? !!measuredFn(kc) : false;
      if (isOff) { off += 1; return { kc, r, measured: m, off: true, state: row && row.state }; }
      if (m) measured += 1;
      if (Number.isFinite(r)) {
        sum += r; n += 1;
        if (r >= UNLOCK_T) bands.strong += 1;
        else if (r >= 0.30) bands.learning += 1;
        else bands.starting += 1;
      } else bands.none += 1;
      return { kc, r, measured: m, off: false, state: row && row.state };
    });
    return {
      id: sec.id, label: label(sec), fullLabel: sec.label, color: sec.color,
      mean: n ? sum / n : NaN, total: sec.kcs.length, active: sec.kcs.length - off,
      measured, off, bands, rows,
      mastered: rows.filter((x) => !x.off && Number.isFinite(x.r) && x.r >= MASTERY_T).length,
    };
  };
  // Cached: a pan redraws every frame, and the numbers only move on the
  // events that clear this (see wiring).
  let colors = {};
  const headColor = (sec) => {
    if (colorMode() !== "mastery") return sec.color;
    if (colors[sec.id]) return colors[sec.id];
    const st = stats(sec.id);
    const mc = window.deltaKcMasteryColor;
    return (colors[sec.id] = typeof mc === "function" ? mc(st ? st.mean : NaN) : "#9aa3b2");
  };

  /* ---------------- layer -------------------------------------------- */
  const ensureLayer = () => {
    const main = document.getElementById("kg-cy");
    if (!main || !main.parentNode) return false;
    if (!heads) {
      heads = document.createElement("div");
      heads.className = "kgs-heads";
      heads.setAttribute("role", "group");
      heads.setAttribute("aria-label", "Areas of the graph");
      main.insertAdjacentElement("afterend", heads);
      heads.addEventListener("click", (e) => {
        const b = e.target.closest("[data-sid]");
        if (!b) return;
        e.stopPropagation();
        if (typeof window.deltaSelectKgSection === "function") window.deltaSelectKgSection(b.dataset.sid);
      });
      // A title sits over the canvas; a wheel over one zooms the graph as a
      // wheel anywhere else on it would.
      heads.addEventListener("wheel", (e) => {
        const target = cy && !cy.destroyed() ? cy.container() : null;
        if (!target) return;
        e.preventDefault();
        target.dispatchEvent(new WheelEvent("wheel", e));
      }, { passive: false });
    }
    return true;
  };
  // The layer sits exactly over #kg-cy (its inset changes with the toolbar).
  const matchBox = (el, main) => {
    el.style.left = main.offsetLeft + "px";
    el.style.top = main.offsetTop + "px";
    el.style.width = main.offsetWidth + "px";
    el.style.height = main.offsetHeight + "px";
  };

  /* ---------------- placement ------------------------------------------ */
  // Spots to try round an area's centre, nearest first: a grid in screen px
  // (STEP apart, out to REACH), scaled into graph units per placement.
  const STEP = 10, REACH = 320;
  const RING = (() => {
    const n = Math.ceil(REACH / STEP), out = [];
    for (let i = -n; i <= n; i++) {
      for (let j = -n; j <= n; j++) {
        const d = Math.hypot(i, j * 1.6);     // a step down costs more than a step aside:
        if (d * STEP <= REACH) out.push([i * STEP, j * STEP, d]); // a title is wide and short
      }
    }
    return out.sort((a, b) => a[2] - b[2]);
  })();
  const overlaps = (a, b) => a.x1 < b.x2 && a.x2 > b.x1 && a.y1 < b.y2 && a.y2 > b.y1;

  // Graph-unit centre of each title, kept until `geom` changes (a node
  // moved, what is shown) or the zoom settles. A pan only re-projects; while
  // a zoom is still moving the titles ride along where they were and are
  // placed again 150 ms after it stops, so a wheel spin costs no searching.
  let geom = 0;
  let placed = { shape: "", zoom: 0, at: {} };
  let settle = null;
  const bump = () => { geom += 1; };

  /** Free spot nearest each area's centre. `items`: [{sid, cx, cy, w, h}] in
      graph units, biggest area first so it gets its centre. */
  const placeAll = (items, zoom) => {
    const GAP = 6 / zoom;
    const blocks = cy.nodes().filter((n) => n.visible()).map((n) => {
      const bb = n.boundingBox({ includeLabels: true, includeOverlays: false });
      return { x1: bb.x1 - GAP, y1: bb.y1 - GAP, x2: bb.x2 + GAP, y2: bb.y2 + GAP };
    });
    // Inside the span of the drawn graph where it fits: Fit frames that span,
    // so a name out past the outermost concept would be cut off by the edge.
    const all = cy.nodes().filter((n) => n.visible()).boundingBox({ includeLabels: true, includeOverlays: false });
    const at = {};
    items.forEach((it) => {
      const reach = REACH / zoom + it.w;
      const near = blocks.filter((b) => b.x2 > it.cx - reach && b.x1 < it.cx + reach && b.y2 > it.cy - reach && b.y1 < it.cy + reach);
      const fitsX = it.w <= all.w, fitsY = it.h <= all.h;
      const inside = (r) => (!fitsX || (r.x1 >= all.x1 && r.x2 <= all.x2)) && (!fitsY || (r.y1 >= all.y1 && r.y2 <= all.y2));
      let best = null;
      for (const strict of [true, false]) {
        for (const [dx, dy] of RING) {
          const x = it.cx + dx / zoom, y = it.cy + dy / zoom;
          const r = { x1: x - it.w / 2, y1: y - it.h / 2, x2: x + it.w / 2, y2: y + it.h / 2 };
          if (strict && !inside(r)) continue;
          if (!near.some((b) => overlaps(r, b))) { best = { x, y, r }; break; }
        }
        if (best) break;
      }
      // Nowhere free within reach (a packed map): no name, rather than a
      // button over concepts that would swallow their clicks. Zooming in
      // opens space and it comes back.
      if (!best) { at[it.sid] = null; return; }
      const pad = { x1: best.r.x1 - GAP, y1: best.r.y1 - GAP, x2: best.r.x2 + GAP, y2: best.r.y2 + GAP };
      blocks.push(pad);
      at[it.sid] = { x: best.x, y: best.y };
    });
    return at;
  };

  /* ---------------- draw ----------------------------------------------- */
  const hidden = () => (view() && view().mode && view().mode() === "condensed");
  const draw = () => {
    queued = false;
    if (!cy || cy.destroyed() || !ensureLayer()) return;
    const main = document.getElementById("kg-cy");
    const off = !shown || hidden() || !main.offsetWidth;
    heads.style.display = off ? "none" : "";
    if (off) return;
    matchBox(heads, main);
    const zoom = cy.zoom(), pan = cy.pan();
    // Title size on screen follows the zoom, like the bubbles' labels do.
    const fs = Math.round(Math.max(15, Math.min(34, 30 * zoom)));
    heads.style.setProperty("--kgs-fs", fs + "px");

    // The areas with concepts on the canvas, and the centre of those.
    const secs = sections().map((s) => {
      const ids = new Set(s.kcs);
      const nodes = cy.nodes().filter((n) => ids.has(n.id()) && n.visible());
      if (!nodes.length) return null;
      let sx = 0, sy = 0;
      nodes.forEach((n) => { const p = n.position(); sx += p.x; sy += p.y; });
      return { s, n: nodes.length, cx: sx / nodes.length, cy: sy / nodes.length };
    }).filter(Boolean);

    const seen = new Set();
    secs.forEach(({ s }) => {
      seen.add(s.id);
      let b = heads.querySelector(`[data-sid="${CSS.escape(s.id)}"]`);
      if (!b) {
        b = document.createElement("button");
        b.type = "button";
        b.className = "kgs-head";
        b.dataset.sid = s.id;
        heads.appendChild(b);
      }
      const text = label(s);
      if (b.textContent !== text) b.textContent = text;
      b.title = `${s.label} — ${s.kcs.length} concept${s.kcs.length === 1 ? "" : "s"}. Click for how you're doing here.`;
      b.style.setProperty("--kgs-c", headColor(s));
      b.classList.toggle("is-active", active === s.id);
      b.classList.toggle("is-dim", !!active && active !== s.id);
    });
    heads.querySelectorAll("[data-sid]").forEach((b) => { if (!seen.has(b.dataset.sid)) b.remove(); });

    const shape = [geom, secs.map((it) => `${it.s.id}:${it.n}`).join(",")].join("|");
    const zoomOnly = placed.shape === shape && placed.zoom !== zoom;
    if (zoomOnly) {
      clearTimeout(settle);
      settle = setTimeout(() => { placed.shape = ""; kick(); }, 150);
    } else if (placed.shape !== shape) {
      clearTimeout(settle);
      const items = secs.slice().sort((a, b) => b.n - a.n).map((it) => {
        const b = heads.querySelector(`[data-sid="${CSS.escape(it.s.id)}"]`);
        b.hidden = false;        // measured at its real size, even if hidden last time
        return { sid: it.s.id, cx: it.cx, cy: it.cy, w: b.offsetWidth / zoom, h: b.offsetHeight / zoom };
      });
      placed = { shape, zoom, at: placeAll(items, zoom) };
    }
    secs.forEach(({ s }) => {
      const b = heads.querySelector(`[data-sid="${CSS.escape(s.id)}"]`);
      const p = placed.at[s.id];
      if (!b) return;
      b.hidden = !p;
      if (!p) return;
      const x = p.x * zoom + pan.x - b.offsetWidth / 2;
      const y = p.y * zoom + pan.y - b.offsetHeight / 2;
      b.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
    });
  };
  const kick = () => { if (!queued) { queued = true; requestAnimationFrame(draw); } };
  const kickMoved = () => { bump(); kick(); };

  /* ---------------- wiring --------------------------------------------- */
  function init() {
    const c = typeof window.deltaConceptGraphCy === "function" ? window.deltaConceptGraphCy() : null;
    if (!c) return false;
    if (cy === c) return true;
    cy = c;
    cy.on("viewport", kick);
    cy.on("resize position add remove style layoutstop", kickMoved);
    kick();
    return true;
  }
  [
    "delta:kg-view-changed", "delta:kg-course-changed", "delta:kg-colormode-changed",
    "delta:kc-readiness-changed", "delta:kc-prefs-changed", "delta:kg-layout-done", "resize",
  ].forEach((ev) => window.addEventListener(ev, () => { colors = {}; bump(); if (init()) kick(); }));
  // lesson-graph.js says what is selected: an area lights its title; a
  // concept or nothing clears it.
  window.addEventListener("delta:kg-selection-changed", (e) => {
    const d = (e && e.detail) || {};
    active = d.kind === "section" ? d.id : null;
    if (init()) kick();
  });
  const boot = () => {
    if (init()) return;
    window.addEventListener("delta:practice-target-graph-ready", () => { setTimeout(() => { if (init()) kick(); }, 0); });
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  window.DeltaKgSections = { label, stats, redraw: kick, shown: () => shown, setShown };
  // kg-toolbar.js loads first and may have painted its button before this
  // existed: tell it the remembered state.
  window.dispatchEvent(new CustomEvent("delta:kg-area-names-changed", { detail: { shown } }));
})();
