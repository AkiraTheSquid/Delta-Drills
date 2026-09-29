/* ================================================================
   XP GROUP VIEW — the Learner Home's graph column, in a group or not.

   Seth, 2026-09-27: "if you join a group, it modifies the practice page
   to be such that instead of displaying both graphs, it displays one
   type, and it has above the graphs two dropdowns: one for the time
   horizon and one for the graph type" — and a further type, "the combined
   view where it only shows you and both graphs rather than your graph
   alongside other people's graphs".

   So in a group the graph column is:
     * two selects: Time (Week / Month / 3 months / All — one stored
       choice, ./xp-panel.js's) and Graph;
     * Graph = Daily XP or Toward the course → one row per member, you
       first, the chart titled with their name (Seth, 2026-09-28: "rows
       instead of the grid with one row for each user"). Every row shares one
       scale (bar height, the course graph's y window, All's x window) so
       the cards can be compared by eye; your own course graph still sets
       your target (./xp-target-drag.js);
     * Graph = Just me → both of your own graphs, no one else's;
     * a third select, Measure (Seth, 2026-09-28): XP, or problems solved —
       the bars count right answers and the course graph becomes their
       running total (./xp-solved-chart.js), in every Graph choice.

   OUT OF A GROUP (Seth, 2026-09-28: "on the left should be a singular
   graph that you can switch in between for displaying either the bar graph
   or the cumulative progress along with the second drop down for the time
   horizon") the same three selects, no group name, and ONE chart: Graph =
   the daily bars or the course line (no "Just me" — that is all there is).
   Before the roster has come back the column is drawn solo too.

   The roster (GET /api/practice/groups/xp, app/group_xp.py) is read on
   boot and again each time the Learner Home is shown: joining and leaving
   happen on the Groups tab, so arriving back here is when it changes.
   Your own card draws from the live summary, never the roster's copy, so a
   target you set shows at once.
   ================================================================ */
(function () {
  "use strict";

  const { el, shortDate } = window.DDXpCharts.util;
  const VIEW_KEY = "dd_xp_group_graph";
  const MEASURE_KEY = "dd_xp_measure";
  const MEASURES = [
    { id: "xp", label: "XP" },
    { id: "solved", label: "Problems solved" },
  ];
  // The Graph options are named in the chosen measure.
  const views = (m) => [
    { id: "bars", label: m === "solved" ? "Solved per day" : "Daily XP" },
    { id: "course", label: m === "solved" ? "Solved in all" : "Toward the course" },
    { id: "me", label: "Just me — both graphs" },
  ];
  const PAD = 14; // a row's inner padding (styles/xp-group.css)

  let roster; // undefined = not read yet, null = in no group, else {group, members}
  // Per-viewer conveniences only: a blocked store just means the defaults.
  const recall = (key, options, fallback) => {
    try {
      const saved = localStorage.getItem(key);
      return options.some((o) => o.id === saved) ? saved : fallback;
    } catch (_) {
      return fallback;
    }
  };
  const remember = (key, id) => {
    try {
      localStorage.setItem(key, id);
    } catch (_) {
      /* convenience only */
    }
  };
  let view = recall(VIEW_KEY, views("xp"), "bars");
  let measure = recall(MEASURE_KEY, MEASURES, "xp");

  const fetcher = () => (typeof apiFetch === "function" ? apiFetch : window.apiFetch);
  const signedIn = () => window.DDIdentity?.isSignedIn?.() === true;
  const repaint = () => window.DDXpPanel?.paintGraphs?.();

  let seq = 0;
  async function load() {
    const mine = ++seq;
    const _fetch = fetcher();
    if (!signedIn() || typeof _fetch !== "function") {
      const had = !!roster;
      roster = null;
      if (had) repaint();
      return;
    }
    try {
      const res = await _fetch(`/api/practice/groups/xp?${window.DeltaXP.tzQuery()}`);
      if (!res?.ok) throw new Error(`groups/xp ${res?.status}`);
      const data = await res.json();
      if (mine !== seq) return;
      roster = data?.group ? data : null;
    } catch (err) {
      // A failed read keeps whatever was drawn; the next arrival tries again.
      console.warn("[xp-group-view]", err);
      return;
    }
    repaint();
  }

  // ── the controls ───────────────────────────────────────────────
  function select(label, options, value, onChange) {
    const wrap = el("label", "xp-select");
    wrap.appendChild(el("span", "xp-select-label", label));
    const sel = el("select", "xp-select-input");
    options.forEach((o) => {
      const opt = el("option", "", o.label);
      opt.value = o.id;
      opt.selected = o.id === value;
      sel.appendChild(opt);
    });
    sel.addEventListener("change", () => onChange(sel.value));
    wrap.appendChild(sel);
    return wrap;
  }

  // "Just me" means something only beside other people.
  const graphOptions = () => (roster ? views(measure) : views(measure).filter((v) => v.id !== "me"));
  const shown = () => (!roster && view === "me" ? "bars" : view);

  function controls(ctx) {
    const bar = el("div", "xp-group-controls");
    bar.append(
      select("Time", ctx.ranges, ctx.range, (id) => {
        ctx.setRange(id);
        focusSelect(0);
      }),
      select("Graph", graphOptions(), shown(), (id) => {
        view = id;
        remember(VIEW_KEY, id);
        repaint();
        focusSelect(1);
      }),
      select("Measure", MEASURES, measure, (id) => {
        measure = id;
        remember(MEASURE_KEY, id);
        repaint();
        focusSelect(2);
      }),
    );
    if (roster) {
      const n = roster.members.length;
      bar.appendChild(el("p", "xp-group-name", `${roster.group.name} · ${n} member${n === 1 ? "" : "s"}`));
    }
    return bar;
  }
  // The column is rebuilt on every change; keep the keyboard where it was.
  const focusSelect = (i) =>
    document.querySelectorAll("#learner-xp-graphs .xp-select-input")[i]?.focus();

  // ── the rows ───────────────────────────────────────────────────
  const rangeOf = (ctx) => ctx.ranges.find((r) => r.id === ctx.range)?.days ?? 7;

  /** One full-width row per member, one scale. `you` is the live summary. */
  function cards(ctx, you) {
    const days = rangeOf(ctx);
    const people = roster.members.map((m) => ({ ...m, xp: m.is_you ? you : m.xp }));
    const drawable = people.filter((m) => m.xp && Array.isArray(m.xp.days));
    const width = ctx.width - 2 * PAD;
    const barCount = (s) => (Number.isFinite(days) ? days : s.days.length);
    const solved = measure === "solved";
    const course = (s, opts) =>
      solved ? window.DDXpSolvedChart.cumulative(s, opts) : window.DDXpCharts.trajectory(s, opts);

    let draw;
    if (view === "bars") {
      // Tallest bar or target anywhere in the range sets everybody's scale.
      const peak = Math.max(solved ? 4 : 10, ...drawable.map((m) => {
        const recent = m.xp.days.slice(-barCount(m.xp)).map((d) => (solved ? d.solved : d.xp) || 0);
        return Math.max(solved ? 0 : m.xp.today.target || 0, ...recent);
      }));
      draw = (m) => window.DDXpCharts.bars(m.xp, { width, count: barCount(m.xp), peak, measure });
    } else {
      // First pass: each row's own windows; the union is everybody's.
      const base = { width, count: days, self: false };
      const plots = drawable.map((m) => course(m.xp, base).xpPlot).filter(Boolean);
      const pinned = plots.length
        ? {
          window: {
            start: plots.map((p) => p.start).reduce((a, b) => (b < a ? b : a)),
            end: plots.map((p) => p.end).reduce((a, b) => (b > a ? b : a)),
          },
          yWindow: { lo: Math.min(...plots.map((p) => p.lo)), hi: Math.max(...plots.map((p) => p.hi)) },
        }
        : {};
      draw = (m) => {
        const opts = { ...base, ...pinned, self: m.is_you };
        const box = course(m.xp, opts);
        // The target is in XP: it is only set on the XP graph.
        return m.is_you && !solved ? window.DDXpTargetDrag.attach(box, m.xp, opts) : box;
      };
    }

    const rows = el("div", "xp-group-rows");
    people.forEach((m) => {
      const card = el("div", `xp-group-card${m.is_you ? " is-you" : ""}`);
      if (!m.xp || !Array.isArray(m.xp.days)) {
        card.append(el("h3", "xp-chart-title", m.display_name), el("p", "xp-chart-note", "Could not be read just now."));
      } else {
        const box = draw(m);
        const title = box.querySelector(".xp-chart-title");
        if (title) {
          title.textContent = m.display_name;
          if (m.is_you) title.appendChild(el("span", "xp-you-tag", "you"));
        }
        const finish = m.xp.projected_finish;
        if (view === "course" && !solved && finish) box.querySelector(".xp-chart-note")?.append(` · done ${shortDate(finish)}`);
        card.appendChild(box);
      }
      rows.appendChild(card);
    });
    return rows;
  }

  /** Draw the column into `right`: the dropdowns, then the group's rows,
      or out of a group the one chart (`ctx.solo(measure, which)`). */
  function paint(right, summary, ctx) {
    const refocus = !!document.activeElement?.matches?.(".xp-trajectory");
    const top = controls(ctx);
    if (!roster) {
      right.replaceChildren(top, ...ctx.solo(measure, shown()));
    } else if (view === "me") {
      right.replaceChildren(top, ...ctx.solo(measure));
    } else {
      right.replaceChildren(top, cards(ctx, summary));
    }
    if (refocus) right.querySelector(".xp-group-card.is-you .xp-trajectory, .xp-trajectory")?.focus();
    return true;
  }

  // Read the roster again whenever the Learner Home comes into view.
  const watchArrival = () => {
    const page = document.getElementById("page-practice");
    if (!page || typeof MutationObserver !== "function") return;
    let shown = !page.classList.contains("hidden");
    new MutationObserver(() => {
      const now = !page.classList.contains("hidden");
      if (now && !shown) load();
      shown = now;
    }).observe(page, { attributes: true, attributeFilter: ["class"] });
  };
  // A different account (or none) must not see the last one's group, even
  // for the length of a request (codex, 2026-09-27): drop it and repaint first.
  window.addEventListener("delta:auth-state-changed", () => {
    const had = !!roster;
    roster = undefined;
    if (had) repaint();
    load();
  });

  const boot = () => {
    watchArrival();
    load();
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  window.DDXpGroupView = { paint, reload: load };
})();
