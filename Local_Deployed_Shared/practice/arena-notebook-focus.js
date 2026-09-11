/* ================================================================
   ARENA NOTEBOOK FOCUS — which exercise the reader is in, and how ready
   they are for its concept.
   ================================================================
   Seth, 2026-09-11: "for whichever area you are in for the notebook
   scrolling down, that is the area that is displayed at the top for the
   progress bar ... whenever you are close enough to it with it in the
   center of the screen or whatever, or close enough such that you're
   basically displaying most of the problem and less focused on the problem
   above it, it essentially gets highlighted for the color of skill that
   you are, for whether it's red or blue or whatever for how ready for it
   you are. And the top bar at the top ... the text for it gets replaced
   and then it ... displays how ready you are for the specific concept
   related to that problem".

   WHAT THIS FILE OWNS
     • the SECTIONS: one per exercise block practice/exercise-session.js
       puts under a cell (`.dd-ex-block`, `_exercise`, `_sourceCell`). A
       section starts at the heading above the exercise and runs to the
       next section's start.
     • the READING POINT: 40% of the way down the viewport. The section
       whose start is the last one above that line is the one the reader
       is in. Not the top edge (the rail's 20% mark is for "which heading
       just scrolled past", a different question) and not dead centre: at
       40% the next problem takes over once it holds the lower three-fifths
       of the screen, which is "displaying most of the problem".
     • `dd-ex-focus` on that section's heading, with `--dd-ready-color`
       (styles/practice/arena-notebook-focus.css draws it), and the
       `dd-notebook-concept` event the topbar pill draws (concept-pill.js).

   WHAT IT DOES NOT OWN
     The number. `window.deltaKcReadinessInfo` (concept-graph/lesson-graph.js)
     is the one ladder every readiness surface in the app reads — in-memory
     posterior → persisted mastery → the server lattice → the lesson's
     average → an extrapolation — and it reports the SOURCE with the number.
     This file passes both on. A "not yet estimated" concept is announced
     with `pct: null`, which the pill draws as an empty, dashed track and
     says so; it is never drawn as 0%.

   🔴 THE COLOUR RAMP IS THE GRAPH'S. Red #d64848 at 0 → blue #3b82f6 at 1,
   grey #5b5b70 for no reading — `masteryColor` in lesson-graph.js, which
   is not exported. `window.deltaKcMasteryColor` is read first so the day
   it is, this copy stops being consulted; until then the two endpoints are
   restated here rather than a third palette invented. A heading tinted
   one red and a graph node tinted another for the same learner and
   concept would be the bug.
   ================================================================ */
(() => {
  "use strict";

  const PAGE = "page-arena-notebook";
  const FOCAL = 0.4;
  const EVENT = "dd-notebook-concept";
  const UNKNOWN = "#5b5b70";
  const LO = [214, 72, 72];
  const HI = [59, 130, 246];

  let sections = [];
  let active = null;
  let announced = "";
  let raf = 0;

  const _page = () => document.getElementById(PAGE);
  const _onScreen = () => {
    const page = _page();
    return !!page && !page.classList.contains("hidden");
  };

  const _color = (r) => {
    if (typeof window.deltaKcMasteryColor === "function") return window.deltaKcMasteryColor(r);
    if (!Number.isFinite(r)) return UNKNOWN;
    const t = Math.max(0, Math.min(1, r));
    const c = LO.map((v, i) => Math.round(v + (HI[i] - v) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  };

  const _reading = (kc) => {
    const read = window.deltaKcReadinessInfo;
    const info = kc && typeof read === "function" ? read(kc) : null;
    const r = info && Number.isFinite(info.r) ? Math.max(0, Math.min(1, info.r)) : null;
    return { r, source: (info && info.source) || "none" };
  };

  /* The heading a cell sits under: the nearest `.nbv-md` at or above it
     that opens with one. A code-cell exercise (0.0's `def rearrange_1(`)
     has no heading of its own, so its section is named by the prose above. */
  const _headingFor = (cell) => {
    for (let node = cell; node; node = node.previousElementSibling) {
      if (!node.classList || !node.classList.contains("nbv-md")) continue;
      const h = node.querySelector("h1, h2, h3, h4");
      if (h) return h;
    }
    return null;
  };

  /* Sections in DOM order. The first exercise under a heading starts at the
     heading — the problem begins where its name does — and a later exercise
     under the SAME heading starts at its own cell, so two problems that share
     a title still hand over as you scroll from one to the next. Several
     blocks on one cell (0.0's five einsum defs) are one section; the first
     def's concept names it. */
  const _collect = (host) => {
    const out = [];
    let lastHeading = null;
    host.querySelectorAll(".dd-ex-block").forEach((block) => {
      const ex = block._exercise;
      const cell = block._sourceCell;
      if (!ex || !cell) return;
      const heading = _headingFor(cell);
      const start = heading && heading !== lastHeading ? heading.closest(".nbv-cell") || cell : cell;
      lastHeading = heading;
      if (out.length && out[out.length - 1].start === start) return;
      out.push({ ex, cell, heading, start });
    });
    sections = out;
  };

  /* 🔴 A CELL WITH NO BOX IS NOT "ABOVE THE LINE". While the clock runs,
     practice/exercise-timer.js's focus mode puts `display: none` on every
     cell but the live exercise's, and a closed disclosure hides the cells
     inside it; both measure a rect of all zeros, and 0 is above any reading
     point — so the LAST hidden section on the page won, and the pill named
     max pooling while ReLU was the one on the clock (caught 2026-09-11 in
     the browser). Same test the contents rail uses: no client rects, no
     section. */
  const _pick = () => {
    const focal = window.innerHeight * FOCAL;
    let found = null;
    for (const section of sections) {
      if (!section.start.isConnected || !section.start.getClientRects().length) continue;
      if (section.start.getBoundingClientRect().top <= focal) found = section;
      else break;
    }
    return found;
  };

  const _announce = (detail) => {
    const key = detail ? JSON.stringify(detail) : "";
    if (key === announced) return;
    announced = key;
    window.dispatchEvent(new CustomEvent(EVENT, { detail }));
  };

  const _paint = () => {
    raf = 0;
    const next = _onScreen() ? _pick() : null;
    if (next !== active) {
      if (active && active.heading) active.heading.classList.remove("dd-ex-focus");
      active = next;
    }
    if (!active) {
      _announce(null);
      return;
    }
    const { ex, heading } = active;
    const { r, source } = _reading(ex.kc);
    const color = _color(r);
    if (heading) {
      heading.classList.add("dd-ex-focus");
      heading.style.setProperty("--dd-ready-color", color);
    }
    _announce({
      kc: ex.kc,
      title: ex.kcTitle || ex.title,
      exercise: ex.title,
      pct: r === null ? null : r * 100,
      color,
      source,
    });
  };

  const _schedule = () => {
    if (raf) return;
    raf = requestAnimationFrame(_paint);
  };

  /* The lattice — the server's per-concept report — is only fetched by the
     Knowledge Graph tab's own build; on this page it would be null and every
     reading would fall through to the offline ladder. One refresh per
     decoration, through the sanctioned door (see practice/readiness.js on
     why not `loadKcLattice` directly), then a repaint with the real numbers. */
  const _refresh = () => {
    const refresh = window.deltaRefreshKcLattice;
    if (typeof refresh !== "function") return;
    Promise.resolve()
      .then(() => refresh())
      .then(_schedule, () => {});
  };

  document.addEventListener("dd-exercise-blocks:decorated", (e) => {
    const host = e.detail && e.detail.host;
    if (!host) return;
    _collect(host);
    active = null;
    _schedule();
    _refresh();
  });
  // A re-render replaces the cells; the old headings and blocks are gone.
  document.addEventListener("arena-notebook:rendered", () => {
    sections = [];
    active = null;
    _schedule();
  });
  window.addEventListener("scroll", _schedule, { passive: true });
  window.addEventListener("resize", _schedule, { passive: true });
  // A graded answer, a placement, a synced state: the reading may have moved.
  window.addEventListener("delta:practice-state-changed", _schedule);
  if (typeof MutationObserver === "function") {
    const page = _page();
    if (page) {
      new MutationObserver(_schedule).observe(page, { attributes: true, attributeFilter: ["class"] });
    }
  }
})();
