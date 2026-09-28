/* ================================================================
   XP GROUP VIEW — the Learner Home's graphs once you are in a group.

   Seth, 2026-09-27: "if you join a group, it modifies the practice page
   to be such that instead of displaying both graphs, it displays one
   type, and it has above the graphs two dropdowns: one for the time
   horizon and one for the graph type" — and a further type, "the combined
   view where it only shows you and both graphs rather than your graph
   alongside other people's graphs".

   So in a group the right column is:
     * two selects: Time (Week / Month / 3 months / All — the same range
       the solo tabs keep, one stored choice) and Graph;
     * Graph = Daily XP or Toward the course → one card per member, you
       first, the chart titled with their name. Every card shares one
       scale (bar height, the course graph's y window, All's x window) so
       the cards can be compared by eye; your own course graph still sets
       your target (./xp-target-drag.js);
     * Graph = Just me → the solo column, both graphs, without the tabs.

   Out of a group, or before the roster has come back, `paint` answers
   false and ./xp-panel.js draws the solo column as it always has.

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
  const VIEWS = [
    { id: "bars", label: "Daily XP" },
    { id: "course", label: "Toward the course" },
    { id: "me", label: "Just me — both graphs" },
  ];
  const CARD_MIN = 280; // px: below two of these side by side, one column
  const GAP = 16;
  const PAD = 14; // a card's inner padding (styles/xp-group.css)

  let roster; // undefined = not read yet, null = in no group, else {group, members}
  let view = "bars";
  try {
    const saved = localStorage.getItem(VIEW_KEY);
    if (VIEWS.some((v) => v.id === saved)) view = saved;
  } catch (_) {
    /* per-viewer convenience only */
  }

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

  function controls(ctx) {
    const bar = el("div", "xp-group-controls");
    bar.append(
      select("Time", ctx.ranges, ctx.range, (id) => {
        ctx.setRange(id);
        focusSelect(0);
      }),
      select("Graph", VIEWS, view, (id) => {
        view = id;
        try {
          localStorage.setItem(VIEW_KEY, id);
        } catch (_) {
          /* convenience only */
        }
        repaint();
        focusSelect(1);
      }),
    );
    const n = roster.members.length;
    bar.appendChild(el("p", "xp-group-name", `${roster.group.name} · ${n} member${n === 1 ? "" : "s"}`));
    return bar;
  }
  // The column is rebuilt on every change; keep the keyboard where it was.
  const focusSelect = (i) =>
    document.querySelectorAll("#learner-xp-graphs .xp-select-input")[i]?.focus();

  // ── the cards ──────────────────────────────────────────────────
  /** Card width for `count` cards in a column `width` wide. */
  const cardWidth = (width, count) => {
    const cols = Math.max(1, Math.min(count, Math.floor((width + GAP) / (CARD_MIN + GAP))));
    return Math.floor((width - GAP * (cols - 1)) / cols);
  };

  const rangeOf = (ctx) => ctx.ranges.find((r) => r.id === ctx.range)?.days ?? 7;

  /** One chart per member on one scale. `you` is the live summary. */
  function cards(ctx, you) {
    const days = rangeOf(ctx);
    const people = roster.members.map((m) => ({ ...m, xp: m.is_you ? you : m.xp }));
    const drawable = people.filter((m) => m.xp && Array.isArray(m.xp.days));
    const width = cardWidth(ctx.width, people.length) - 2 * PAD;
    const barCount = (s) => (Number.isFinite(days) ? days : s.days.length);

    let draw;
    if (view === "bars") {
      // Tallest bar or target anywhere in the range sets everybody's scale.
      const peak = Math.max(10, ...drawable.map((m) => {
        const recent = m.xp.days.slice(-barCount(m.xp)).map((d) => d.xp);
        return Math.max(m.xp.today.target || 0, ...recent);
      }));
      draw = (m) => window.DDXpCharts.bars(m.xp, { width, count: barCount(m.xp), peak });
    } else {
      // First pass: each card's own windows; the union is everybody's.
      const base = { width, count: days, self: false };
      const plots = drawable.map((m) => window.DDXpCharts.trajectory(m.xp, base).xpPlot).filter(Boolean);
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
        const box = window.DDXpCharts.trajectory(m.xp, opts);
        return m.is_you ? window.DDXpTargetDrag.attach(box, m.xp, opts) : box;
      };
    }

    const grid = el("div", "xp-group-grid");
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
        if (view === "course" && finish) box.querySelector(".xp-chart-note")?.append(` · done ${shortDate(finish)}`);
        card.appendChild(box);
      }
      grid.appendChild(card);
    });
    return grid;
  }

  /** Draw the group column into `right`. False when not in a group. */
  function paint(right, summary, ctx) {
    if (!roster) return false;
    const refocus = !!document.activeElement?.matches?.(".xp-trajectory");
    const top = controls(ctx);
    if (view === "me") {
      right.replaceChildren(top, ...ctx.solo());
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
