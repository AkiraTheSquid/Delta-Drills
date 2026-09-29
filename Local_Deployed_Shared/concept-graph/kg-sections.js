/* concept-graph/kg-sections.js — a HEADER over each area of the Knowledge
 * Graph, joined to its concepts by light dotted lines (Seth, 2026-09-29:
 * "certain areas that have a larger header … straight dotted lines that are
 * kind of light to the nodes themselves … ch 0.1 raytracing"). It replaced
 * the legend: the map now labels itself.
 *
 *   Colour  — Sections mode: the area's own colour (the one its bubbles
 *             wear). Mastery mode: the mastery ramp at the area's AVERAGE
 *             reading, so a header says how strong you are there.
 *   Click   — `window.deltaSelectKgSection(sid)` (lesson-graph.js): every
 *             concept in the area lights up and the side panel shows the
 *             area's diagnostics (kg-panel.js).
 *
 * Two layers over the main canvas, both in the same box as #kg-cy:
 *   an SVG of leader lines BEHIND the canvas (its background is the pane's,
 *   so the lines show through between bubbles and never cross over one), and
 *   the headers themselves ABOVE it, as buttons.
 * Positions are recomputed from the nodes once per frame while the viewport
 * or any node moves. Hidden in the condensed view (it has its own canvas).
 *
 * 🔴 Reads the graph, never writes it: no elements, no styles, no classes on
 * lesson-graph.js's instance. The layout keeps each area together only because
 * graph-views.js lays it out grouped (kg-look.js `groupBy`). */
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

  let cy = null, host = null, lines = null, heads = null;
  let active = null;             // sid whose header is selected
  let queued = false;

  const view = () => window.deltaKgView || null;
  const colorMode = () => (typeof window.deltaKgColorMode === "function" ? window.deltaKgColorMode() : "mastery");
  const sections = () => (view() && view().sections ? view().sections() : []);
  const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

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

  /* ---------------- layers --------------------------------------------- */
  const ensureLayers = () => {
    const main = document.getElementById("kg-cy");
    if (!main || !main.parentNode) return false;
    host = main.parentNode;
    if (!lines) {
      lines = document.createElementNS(SVGNS, "svg");
      lines.setAttribute("class", "kgs-lines");
      lines.setAttribute("aria-hidden", "true");
      host.insertBefore(lines, main);
    }
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
    }
    return true;
  };
  // Both layers sit exactly over #kg-cy (its inset changes with the toolbar).
  const matchBox = (el, main) => {
    el.style.left = main.offsetLeft + "px";
    el.style.top = main.offsetTop + "px";
    el.style.width = main.offsetWidth + "px";
    el.style.height = main.offsetHeight + "px";
  };

  /* ---------------- draw ----------------------------------------------- */
  const hidden = () => (view() && view().mode && view().mode() === "condensed");
  const draw = () => {
    queued = false;
    if (!cy || cy.destroyed() || !ensureLayers()) return;
    const main = document.getElementById("kg-cy");
    const off = hidden() || !main.offsetWidth;
    lines.style.display = heads.style.display = off ? "none" : "";
    if (off) return;
    matchBox(lines, main);
    matchBox(heads, main);
    const W = main.offsetWidth;
    const zoom = cy.zoom();

    // One header per area on the canvas: centred over the area's box, a
    // little above it. Measured after insertion (width depends on the text).
    const secs = sections().map((s) => {
      const ids = new Set(s.kcs);
      const nodes = cy.nodes().filter((n) => ids.has(n.id()));
      if (!nodes.length) return null;
      return { s, nodes, bb: nodes.renderedBoundingBox({ includeLabels: false }) };
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
    });
    heads.querySelectorAll("[data-sid]").forEach((b) => { if (!seen.has(b.dataset.sid)) b.remove(); });

    // Place, then push apart: a header that overlaps one already placed moves
    // up above it. Bounded passes; the graph clusters rarely stack deeper.
    const placed = [];
    const GAP = 6, LIFT = 26 + 10 * Math.min(1, zoom);
    secs.sort((a, b) => a.bb.y1 - b.bb.y1).forEach((it) => {
      const b = heads.querySelector(`[data-sid="${CSS.escape(it.s.id)}"]`);
      const w = b.offsetWidth, h = b.offsetHeight;
      let x = (it.bb.x1 + it.bb.x2) / 2 - w / 2;
      x = Math.max(4, Math.min(W - w - 4, x));
      // Never above the canvas: the topmost area's header would sit under the
      // toolbar. Pinned to the top edge it may overlap its own first row,
      // which beats not being there; a collision there goes below instead.
      const TOP = 4;
      let y = Math.max(TOP, it.bb.y1 - LIFT - h);
      for (let pass = 0; pass < 12; pass++) {
        const hit = placed.find((p) => x < p.x + p.w + GAP && x + w + GAP > p.x && y < p.y + p.h + GAP && y + h + GAP > p.y);
        if (!hit) break;
        y = hit.y - h - GAP >= TOP ? hit.y - h - GAP : hit.y + hit.h + GAP;
      }
      placed.push({ x, y, w, h });
      b.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
      it.anchor = { x: x + w / 2, y: y + h };
    });

    // Leader lines: header's foot → each concept's top edge. Behind the
    // canvas, so they read as a light fan between bubbles.
    let svg = "";
    secs.forEach((it) => {
      if (!it.anchor) return;
      const c = headColor(it.s);
      const dim = active && active !== it.s.id;
      let d = "";
      it.nodes.forEach((n) => {
        const p = n.renderedPosition();
        const top = p.y - n.renderedOuterHeight() / 2;
        d += `M${it.anchor.x.toFixed(1)} ${it.anchor.y.toFixed(1)}L${p.x.toFixed(1)} ${top.toFixed(1)}`;
      });
      svg += `<path d="${d}" stroke="${esc(c)}" class="kgs-line${active === it.s.id ? " is-active" : ""}${dim ? " is-dim" : ""}"/>`;
    });
    lines.innerHTML = svg;
  };
  const kick = () => { if (!queued) { queued = true; requestAnimationFrame(draw); } };

  /* ---------------- wiring --------------------------------------------- */
  const init = () => {
    const c = typeof window.deltaConceptGraphCy === "function" ? window.deltaConceptGraphCy() : null;
    if (!c) return false;
    if (cy === c) return true;
    cy = c;
    cy.on("viewport resize position add remove layoutstop", kick);
    kick();
    return true;
  };
  [
    "delta:kg-view-changed", "delta:kg-course-changed", "delta:kg-colormode-changed",
    "delta:kc-readiness-changed", "delta:kc-prefs-changed", "delta:kg-layout-done", "resize",
  ].forEach((ev) => window.addEventListener(ev, () => { colors = {}; if (init()) kick(); }));
  // lesson-graph.js says what is selected: an area lights its header; a
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

  window.DeltaKgSections = { label, stats, redraw: kick };
})();
