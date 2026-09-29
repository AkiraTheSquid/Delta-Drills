/* ================================================================
   XP PANEL — the Learner Home's left column: the course, the week's XP,
   the daily bars.

   Seth, 2026-09-26: XP is "measured by your actual learning progress",
   the same yardstick for every learner. Seth, 2026-09-29: "in the top left
   it says which course you're in, and to the right of the left column it
   displays your percentage completion in the course. below that it doesn't
   have all the fancy drop downs: it just has the bar graph with your
   learning every day and above it it says how much exp you gained" — and
   "remove the leveling system". So, all from GET /api/practice/xp
   (app/learning_xp.py):

     ARENA                                       6%
     through 0.2 · 125 concepts            complete
     157 XP gained this week · 12 today
     ▁ ▃ ▂ ▅ ▁ ▆ ▒                          the week, one bar a day

   🪦 Gone with that: the Time / Graph / Measure dropdowns, the course line
   and its drag-to-set target (xp-target-drag.js), the problems-solved
   measure (xp-solved-chart.js), and the Today / Level / Course / Finish /
   Target card. A target already saved stays in the learner's state and in
   the summary; nothing here draws it.

   THE BARS (#learner-xp-graphs) are drawn by ./xp-group-view.js: yours
   alone, or in a study group one row per member. This file hands it `solo`.

   It draws from `delta:xp-summary`, which ../xp.js broadcasts after every
   read. `.is-empty` hides both halves when there is no summary (no
   session, or the read failed); the idle card then narrows to its button.
   ================================================================ */
(function () {
  "use strict";

  const WEEK = 7;

  let summary = null;
  let revealed = false;
  let drawnWidth = 0;

  const host = () => document.getElementById("learner-xp");
  const graphs = () => document.getElementById("learner-xp-graphs");
  const { el, fmt, addDays } = window.DDXpCharts.util;
  const coursePct = (s) => {
    const pct = s.course.total_xp ? (s.knowledge / s.course.total_xp) * 100 : 0;
    return `${pct < 10 ? pct.toFixed(1) : Math.round(pct)}%`;
  };

  /* The chart is drawn at the column's real width, so the SVG is never
     scaled and its 10px labels stay 10px on a phone. A hidden column (0
     wide) draws at a default; the resize observer below redraws it. */
  const chartWidth = () => {
    const g = graphs();
    if (!g) return 560;
    const cs = getComputedStyle(g);
    const w = g.clientWidth - parseFloat(cs.paddingLeft || 0) - parseFloat(cs.paddingRight || 0);
    return w > 0 ? Math.max(280, Math.min(900, Math.round(w))) : 560;
  };

  // ── the course, and how far into it ────────────────────────────
  function course(s) {
    const head = el("header", "xp-course");
    const id = el("div", "xp-course-id");
    id.append(
      el("h2", "xp-course-name", s.course.name || "Your course"),
      el("p", "xp-course-sub", `through ${s.course.through} · ${fmt(s.course.concepts)} concepts`),
    );
    const done = el("p", "xp-course-pct");
    done.append(el("span", "xp-course-pct-num", coursePct(s)), el("span", "xp-course-pct-label", "complete"));
    done.title = `${fmt(s.knowledge)} of ${fmt(s.course.total_xp)} XP`;
    head.append(id, done);
    return head;
  }

  // ── the week's XP, over its bars ───────────────────────────────
  function gained(s) {
    const from = addDays(s.today.date, -(WEEK - 1));
    const week = s.days.reduce((a, d) => a + (d.date >= from ? d.xp || 0 : 0), 0);
    const p = el("p", "xp-gained");
    p.append(el("span", "xp-gained-num", fmt(week)), el("span", "xp-gained-unit", " XP gained this week"));
    p.appendChild(el("span", "xp-gained-today", ` · ${fmt(s.today.xp)} today`));
    return p;
  }

  /** The bars column alone. */
  function paintGraphs() {
    const right = graphs();
    if (!right || !summary) return;
    // Measured after the column is shown: a hidden column is 0 wide.
    drawnWidth = chartWidth();
    const solo = () => [window.DDXpCharts.bars(summary, { width: drawnWidth, count: WEEK, bare: true, target: false })];
    const ctx = { width: drawnWidth, days: WEEK, solo };
    if (window.DDXpGroupView) window.DDXpGroupView.paint(right, summary, ctx);
    else right.replaceChildren(...solo()); // the group view's file did not load
  }

  // ── paint ──────────────────────────────────────────────────────
  function paint() {
    const section = host();
    const right = graphs();
    if (!section) return;
    const root = section.querySelector(".xp-panel-body") || section;
    if (!summary || !Array.isArray(summary.days) || !summary.days.length || !summary.course) {
      section.classList.add("is-empty");
      right?.classList.add("is-empty");
      return;
    }
    section.classList.remove("is-empty");
    right?.classList.remove("is-empty");
    root.replaceChildren(course(summary), gained(summary));
    paintGraphs();
    if (!revealed) {
      revealed = true;
      const setup = section.closest(".session-setup");
      setup?.classList.add("is-revealing");
      setTimeout(() => setup?.classList.remove("is-revealing"), 1400);
    }
  }

  window.addEventListener("delta:xp-summary", (e) => {
    summary = e.detail || null;
    paint();
  });

  // Redraw at the new width when the column changes size (rotation, the tab
  // becoming visible). Small jitters are ignored.
  const watchWidth = () => {
    const right = graphs();
    if (!right || typeof ResizeObserver !== "function") return;
    let timer = null;
    new ResizeObserver(() => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (summary && Math.abs(chartWidth() - drawnWidth) > 24) paintGraphs();
      }, 150);
    }).observe(right);
  };

  const boot = () => {
    summary = window.DeltaXP?.summary?.() || null;
    paint();
    watchWidth();
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  window.DDXpPanel = { paint, paintGraphs };
})();
