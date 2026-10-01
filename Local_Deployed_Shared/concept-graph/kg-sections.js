/* concept-graph/kg-sections.js — optional AREA BOXES on the Knowledge Graph:
 * a rectangle around each area's concepts with the area's title on its top
 * edge (Seth, 2026-09-30: "an optional thing that you can enable and disable
 * for the sections where it shows a rectangle with a title around the
 * different nodes … but it shouldn't reposition the nodes of the graph").
 * Off by default; the toolbar's "Area boxes" button (kg-toolbar.js) flips it
 * and the choice stays in this browser (dd_kg_area_boxes).
 *
 * It replaced the 2026-09-29 headers with dotted leader lines, which needed
 * the layout grouped by area — that moved every concept, and was reverted.
 * The layout is the graph's own again, so areas interleave and their boxes
 * may overlap; that is the honest picture, not a bug.
 *
 *   Colour  — Sections mode: the area's own colour (the one its bubbles
 *             wear). Mastery mode: the mastery ramp at the area's AVERAGE
 *             reading, so a box says how strong you are there.
 *   Click   — a title calls `window.deltaSelectKgSection(sid)`
 *             (lesson-graph.js): every concept in the area lights up and the
 *             side panel shows the area's diagnostics (kg-panel.js).
 *
 * Two layers in the same box as #kg-cy: an SVG of rectangles BEHIND the
 * canvas (its background is the pane's, so a box shows between bubbles and
 * never covers one), and the titles ABOVE it, as buttons. Positions are
 * recomputed from the nodes once per frame while the viewport or any node
 * moves. Hidden in the condensed view (it has its own canvas).
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

  let cy = null, host = null, boxes = null, heads = null;
  let active = null;             // sid whose title is selected
  let queued = false;

  const BOX_KEY = "dd_kg_area_boxes";
  let shown = false;
  try { shown = localStorage.getItem(BOX_KEY) === "1"; } catch (_) {}
  const setShown = (v) => {
    v = !!v;
    if (v === shown) return;
    shown = v;
    try { localStorage.setItem(BOX_KEY, v ? "1" : "0"); } catch (_) {}
    window.dispatchEvent(new CustomEvent("delta:kg-area-boxes-changed", { detail: { shown } }));
    if (init()) kick();
  };

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
    if (!boxes) {
      boxes = document.createElementNS(SVGNS, "svg");
      boxes.setAttribute("class", "kgs-boxes");
      boxes.setAttribute("aria-hidden", "true");
      host.insertBefore(boxes, main);
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
    const off = !shown || hidden() || !main.offsetWidth;
    boxes.style.display = heads.style.display = off ? "none" : "";
    if (off) return;
    matchBox(boxes, main);
    matchBox(heads, main);
    const W = main.offsetWidth;
    // Padding round the bubbles, in screen px: roomy when zoomed in, never
    // so wide that a zoomed-out map is all frame.
    const PAD = Math.max(6, Math.min(18, 16 * cy.zoom()));

    // One box per area on the canvas, round the concepts actually drawn.
    const secs = sections().map((s) => {
      const ids = new Set(s.kcs);
      const nodes = cy.nodes().filter((n) => ids.has(n.id()) && n.visible());
      if (!nodes.length) return null;
      const bb = nodes.renderedBoundingBox({ includeLabels: true, includeOverlays: false });
      return { s, x: bb.x1 - PAD, y: bb.y1 - PAD, w: bb.w + 2 * PAD, h: bb.h + 2 * PAD };
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

    // Titles sit on the box's top edge, at its left (a fieldset's legend).
    // Areas overlap, so a title that would cover one already placed steps
    // down inside its own box, then sideways. Bounded passes.
    const placed = [];
    const GAP = 4;
    secs.slice().sort((a, b) => a.y - b.y || a.x - b.x).forEach((it) => {
      const b = heads.querySelector(`[data-sid="${CSS.escape(it.s.id)}"]`);
      const w = b.offsetWidth, h = b.offsetHeight;
      let x = Math.max(4, Math.min(W - w - 4, it.x + 10));
      let y = Math.max(4, it.y - h / 2);
      for (let pass = 0; pass < 12; pass++) {
        const hit = placed.find((p) => x < p.x + p.w + GAP && x + w + GAP > p.x && y < p.y + p.h + GAP && y + h + GAP > p.y);
        if (!hit) break;
        if (pass % 2 === 0) y = hit.y + hit.h + GAP;
        else x = Math.max(4, Math.min(W - w - 4, hit.x + hit.w + GAP));
      }
      placed.push({ x, y, w, h });
      b.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
    });

    // Big boxes first, so a small area inside a big one is drawn on top.
    let svg = "";
    secs.slice().sort((a, b) => b.w * b.h - a.w * a.h).forEach((it) => {
      const c = headColor(it.s);
      const cls = "kgs-box" + (active === it.s.id ? " is-active" : active ? " is-dim" : "");
      svg += `<rect x="${it.x.toFixed(1)}" y="${it.y.toFixed(1)}" width="${it.w.toFixed(1)}" height="${it.h.toFixed(1)}"` +
        ` rx="12" stroke="${esc(c)}" fill="${esc(c)}" class="${cls}"/>`;
    });
    boxes.innerHTML = svg;
  };
  const kick = () => { if (!queued) { queued = true; requestAnimationFrame(draw); } };

  /* ---------------- wiring --------------------------------------------- */
  function init() {
    const c = typeof window.deltaConceptGraphCy === "function" ? window.deltaConceptGraphCy() : null;
    if (!c) return false;
    if (cy === c) return true;
    cy = c;
    cy.on("viewport resize position add remove style layoutstop", kick);
    kick();
    return true;
  }
  [
    "delta:kg-view-changed", "delta:kg-course-changed", "delta:kg-colormode-changed",
    "delta:kc-readiness-changed", "delta:kc-prefs-changed", "delta:kg-layout-done", "resize",
  ].forEach((ev) => window.addEventListener(ev, () => { colors = {}; if (init()) kick(); }));
  // lesson-graph.js says what is selected: an area lights its box; a
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
  window.dispatchEvent(new CustomEvent("delta:kg-area-boxes-changed", { detail: { shown } }));
})();
