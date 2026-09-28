/* ================================================================
   XP PANEL — the Learner Home: measured learning and its two graphs.

   Seth, 2026-09-26: XP is "measured by your actual learning progress",
   the same yardstick for every learner, with a daily target that keeps
   the learner on pace to finish the course — and a graph of it. Later
   the same day: one screen, measured learning on the left, the graphs on
   the right, daily XP a week by default.

   LEFT (#learner-xp, all from GET /api/practice/xp, app/learning_xp.py)
     * today's XP against today's target — the one number to look at;
     * level (80 XP = one concept's worth), course knowledge out of the
       fixed total, and the projected finish at the learner's pace;
     * the target, one line, with the setter behind "Change": finish by a
       date (the server spreads what is left over the days that remain,
       re-read every day) or a fixed XP per day.

   RIGHT (#learner-xp-graphs)
     * one row of range tabs across the column — Week (the default),
       Month, 3 months, All — that controls BOTH graphs (Seth, 2026-09-27);
     * DAILY XP bars and TOWARD THE COURSE under it, drawn by
       ./xp-charts.js for that range.

   It draws from `delta:xp-summary`, which ../xp.js broadcasts after every
   read it makes for the level pill — one request feeds both. Nothing here
   fetches the summary itself except through `DeltaXP.refresh`.

   `.is-empty` hides both halves when there is no summary (no session, or
   the read failed); the idle card then narrows to its one button.
   ================================================================ */
(function () {
  "use strict";

  const RANGES = [
    { id: "7", label: "Week", days: 7 },
    { id: "30", label: "Month", days: 30 },
    { id: "90", label: "3 months", days: 90 },
    { id: "all", label: "All", days: Infinity },
  ];
  // A new key: the old one ("dd_xp_panel_range") remembered 30 days, and the
  // default is a week now.
  const RANGE_KEY = "dd_xp_range";

  let summary = null;
  let range = "7";
  try {
    const saved = localStorage.getItem(RANGE_KEY);
    if (RANGES.some((r) => r.id === saved)) range = saved;
  } catch (_) {
    /* per-viewer convenience only */
  }
  let mode = null; // the target form's selected mode, kept across repaints
  let editing = false; // the target setter is open
  let focusToggle = false; // after a save, focus returns to Change / Set
  let revealed = false;

  const host = () => document.getElementById("learner-xp");
  const graphs = () => document.getElementById("learner-xp-graphs");
  const { el, fmt, shortDate, longDate, daysBetween, addDays } = window.DDXpCharts.util;
  const compact = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });
  const coursePct = (s) => {
    const pct = s.course.total_xp ? (s.knowledge / s.course.total_xp) * 100 : 0;
    return `${pct < 10 ? pct.toFixed(1) : Math.round(pct)}%`;
  };

  /* The charts are drawn at the column's real width, so the SVG is never
     scaled and its 10px labels stay 10px on a phone. A hidden column (0
     wide) draws at a default; the resize observer below redraws it. */
  const chartWidth = () => {
    const g = graphs();
    if (!g) return 560;
    const cs = getComputedStyle(g);
    const w = g.clientWidth - parseFloat(cs.paddingLeft || 0) - parseFloat(cs.paddingRight || 0);
    return w > 0 ? Math.max(280, Math.min(900, Math.round(w))) : 560;
  };

  // ── today, level, course, finish ──────────────────────────────
  function hero(s) {
    const wrap = el("div", "xp-hero");

    const today = el("div", "xp-today");
    today.appendChild(el("p", "xp-eyebrow", "Today"));
    const figure = el("p", "xp-today-figure");
    figure.appendChild(el("span", "xp-today-num", fmt(s.today.xp)));
    const target = s.today.target;
    figure.appendChild(el("span", "xp-today-of", target ? `/ ${fmt(target)} XP` : "XP"));
    today.appendChild(figure);

    const meter = el("div", "xp-meter");
    const fill = el("span", "xp-meter-fill");
    if (target) {
      const pct = Math.min(100, (s.today.xp / target) * 100);
      // A meter only with a range to be in; past the target it reads full.
      meter.setAttribute("role", "meter");
      meter.setAttribute("aria-valuemin", "0");
      meter.setAttribute("aria-valuemax", String(target));
      meter.setAttribute("aria-valuenow", String(Math.min(target, Math.round(s.today.xp))));
      meter.setAttribute("aria-label", `Today: ${fmt(s.today.xp)} of ${fmt(target)} XP`);
      fill.style.setProperty("--xp-meter-pct", `${pct}%`);
      if (s.today.xp >= target) meter.classList.add("is-met");
    } else {
      meter.classList.add("is-open");
      meter.setAttribute("aria-hidden", "true");
    }
    meter.appendChild(fill);
    today.appendChild(meter);

    let line;
    if (!target) line = "Set a target and this becomes today's number to hit.";
    else if (s.today.xp >= target) line = "Done for today. Anything more is ahead of pace.";
    else line = `${fmt(target - s.today.xp)} XP to go today.`;
    today.appendChild(el("p", "xp-today-line", line));
    wrap.appendChild(today);

    const stats = el("dl", "xp-stats");
    const stat = (label, value, sub, cls) => {
      const row = el("div", `xp-stat${cls ? " " + cls : ""}`);
      row.append(el("dt", "xp-eyebrow", label), el("dd", "xp-stat-value", value));
      if (sub) row.appendChild(el("dd", "xp-stat-sub", sub));
      stats.appendChild(row);
    };
    stat("Level", fmt(s.level), `${fmt(s.into)} / ${s.need} to ${fmt(s.level + 1)}`);
    stat("Course", coursePct(s), `${compact.format(s.knowledge)} / ${compact.format(s.course.total_xp)} XP`);
    const pace = pacing(s);
    stat("Finish", pace.value, pace.sub, pace.cls);
    wrap.appendChild(stats);
    return wrap;
  }

  /** The finish reading: the projected date at the learner's pace, coloured
      — with a date target — by whether that pace is enough. The target line
      below it says what the date asks for. */
  function pacing(s) {
    const finish = s.projected_finish;
    if (!(s.remaining > 0)) return { value: "Done", sub: "every concept ready", cls: "is-ahead" };
    let cls = "";
    if (s.target?.mode === "date" && finish) {
      cls = s.target.date < s.today.date || finish > s.target.date ? "is-behind" : "is-ahead";
    }
    return {
      value: finish ? shortDate(finish) : "—",
      sub: s.pace > 0 ? `at ${fmt(s.pace)} XP/day` : "no net gain yet",
      cls,
    };
  }

  // ── the target: one line, the setter behind it ─────────────────
  function targetRow(s) {
    const box = el("div", "xp-target");
    const now = el("div", "xp-target-now");
    let words = "none set";
    if (s.target?.mode === "date") {
      words = s.target.date < s.today.date
        ? `by ${shortDate(s.target.date)} (passed)`
        : `by ${shortDate(s.target.date)} · ${fmt(s.today.target)} XP/day`;
    }
    else if (s.target?.mode === "daily") words = `${fmt(s.target.daily)} XP/day`;
    now.append(el("span", "xp-eyebrow", "Target"), el("span", "xp-target-value", words));
    const toggle = el("button", "xp-link", editing ? "Close" : s.target ? "Change" : "Set");
    toggle.type = "button";
    toggle.setAttribute("aria-expanded", String(editing));
    // Only this row is rebuilt: the graphs have not changed. Focus follows
    // the learner — into the setter on open, back to the toggle on close.
    toggle.addEventListener("click", () => {
      editing = !editing;
      const next = targetRow(s);
      box.replaceWith(next);
      next.querySelector(editing ? ".xp-input" : ".xp-link")?.focus();
    });
    now.appendChild(toggle);
    box.appendChild(now);
    if (editing) box.appendChild(targetForm(s));
    return box;
  }

  function targetForm(s) {
    const form = el("form", "xp-target-form");
    if (!mode) mode = s.target?.mode || "date";

    const seg = el("div", "xp-seg");
    seg.setAttribute("role", "group");
    seg.setAttribute("aria-label", "Target kind");
    const fields = el("div", "xp-target-fields");
    const status = el("p", "xp-target-status");
    status.setAttribute("role", "status");

    const dateIn = el("input", "xp-input");
    dateIn.type = "date";
    dateIn.min = s.today.date;
    dateIn.value = s.target?.mode === "date" ? s.target.date : addDays(s.today.date, 84);
    dateIn.setAttribute("aria-label", "Finish the course by");

    const dailyIn = el("input", "xp-input xp-input-num");
    dailyIn.type = "number";
    dailyIn.min = "1";
    dailyIn.max = "2000";
    // step 1, not 5: with min 1 a step of 5 makes only 1, 6, 11 … valid, so the
    // browser silently refused to submit 80 — or the prefilled value.
    dailyIn.step = "1";
    dailyIn.value = String(s.target?.mode === "daily" ? s.target.daily : Math.max(20, Math.round(s.pace / 5) * 5 || 60));
    dailyIn.setAttribute("aria-label", "XP per day");

    const preview = el("p", "xp-target-preview");
    const updatePreview = () => {
      if (mode === "date" && dateIn.value) {
        const left = Math.max(1, daysBetween(s.today.date, dateIn.value) + 1);
        const need = Math.ceil((s.remaining_open ?? s.remaining) / left);
        preview.textContent = `≈ ${fmt(need)} XP a day for ${fmt(left)} days, re-read each morning.`;
      } else if (mode === "daily" && Number(dailyIn.value) > 0) {
        const days = Math.ceil(s.remaining / Number(dailyIn.value));
        preview.textContent = `Done around ${longDate(addDays(s.today.date, days))}.`;
      } else preview.textContent = "";
    };

    const option = (value, label) => {
      const b = el("button", "xp-seg-btn", label);
      b.type = "button";
      b.setAttribute("aria-pressed", String(mode === value));
      b.addEventListener("click", () => {
        mode = value;
        seg.querySelectorAll(".xp-seg-btn").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
        draw();
      });
      seg.appendChild(b);
    };
    option("date", "By a date");
    option("daily", "XP per day");

    const save = el("button", "xp-save", "Save");
    save.type = "submit";
    const clear = el("button", "xp-clear", "Clear");
    clear.type = "button";
    clear.hidden = !s.target;

    function draw() {
      fields.replaceChildren(mode === "date" ? dateIn : dailyIn, save, clear);
      updatePreview();
    }
    dateIn.addEventListener("input", updatePreview);
    dailyIn.addEventListener("input", updatePreview);

    const send = async (body) => {
      save.disabled = clear.disabled = true;
      status.classList.remove("is-error");
      status.textContent = "Saving…";
      try {
        await window.DeltaXP.setTarget(body);
        // Saved with Enter, focus is still in the input and the summary event
        // skipped its repaint; close the setter and draw the saved target now.
        editing = false;
        focusToggle = true;
        document.activeElement?.blur?.();
        paint();
      } catch (err) {
        status.textContent = String(err?.message || err);
        status.classList.add("is-error");
        save.disabled = clear.disabled = false;
      }
    };
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      if (mode === "date") send({ mode: "date", date: dateIn.value });
      else send({ mode: "daily", daily: Math.round(Number(dailyIn.value)) });
    });
    clear.addEventListener("click", () => {
      mode = null;
      send({ mode: "none" });
    });

    draw();
    form.append(seg, fields, preview, status);
    return form;
  }

  function explainer(s) {
    const d = el("details", "xp-explain");
    d.appendChild(el("summary", "", "How XP is measured"));
    const ul = el("ul");
    [
      `1 XP is one point of knowledge gained on one concept, counted up to the ready line (${s.course.ready_at}%). The course is ${s.course.concepts} concepts × ${s.course.ready_at} = ${fmt(s.course.total_xp)} XP, the same for everyone.`,
      "Knowledge is the model's belief you have learned a concept, times how well you still recall it. Finding out what you already knew counts as starting knowledge, not XP.",
      "Today's number settles as you answer more: later answers can move credit to the day the learning actually happened.",
      "Forgetting lowers your knowledge but never takes XP away. Relearning earns XP again.",
      `Finish divides what is left by your net learning over the last ${s.pace_days || 14} days and today so far, forgetting included, so it moves as you answer.`,
    ].forEach((t) => ul.appendChild(el("li", "", t)));
    d.appendChild(ul);
    return d;
  }

  // ── the range: one row of tabs over both graphs ────────────────
  function rangeTabs() {
    const tabs = el("div", "xp-range");
    // Plain toggle buttons, not a tablist: there are no tab panels, and
    // Tab + Enter is the whole keyboard contract a pressed button needs.
    tabs.setAttribute("role", "group");
    tabs.setAttribute("aria-label", "Time shown in both graphs");
    RANGES.forEach((r) => {
      const b = el("button", "xp-range-btn", r.label);
      b.type = "button";
      b.setAttribute("aria-pressed", String(r.id === range));
      b.addEventListener("click", () => {
        range = r.id;
        try {
          localStorage.setItem(RANGE_KEY, range);
        } catch (_) {
          /* convenience only */
        }
        paintGraphs();
        graphs()?.querySelector('.xp-range-btn[aria-pressed="true"]')?.focus();
      });
      tabs.appendChild(b);
    });
    return tabs;
  }

  /** The right column alone: the tabs and the two graphs for the range. */
  function paintGraphs() {
    const right = graphs();
    if (!right || !summary) return;
    const days = RANGES.find((r) => r.id === range)?.days ?? 7;
    // Measured after the column is shown: a hidden column is 0 wide.
    drawnWidth = chartWidth();
    const opts = { width: drawnWidth };
    const course = { ...opts, count: days };
    // A target set from the keyboard repaints this column; keep focus on the graph.
    const refocus = !!document.activeElement?.matches?.(".xp-trajectory");
    right.replaceChildren(
      rangeTabs(),
      window.DDXpCharts.bars(summary, { ...opts, count: Number.isFinite(days) ? days : summary.days.length }),
      window.DDXpTargetDrag.attach(window.DDXpCharts.trajectory(summary, course), summary, course),
    );
    if (refocus) right.querySelector(".xp-trajectory")?.focus();
  }

  // ── paint ──────────────────────────────────────────────────────
  function paint() {
    const section = host();
    const right = graphs();
    if (!section) return;
    const root = section.querySelector(".xp-panel-body") || section;
    if (!summary || !Array.isArray(summary.days) || !summary.days.length) {
      section.classList.add("is-empty");
      right?.classList.add("is-empty");
      return;
    }
    const s = summary;
    pending = false;
    section.classList.remove("is-empty");
    right?.classList.remove("is-empty");
    root.replaceChildren(hero(s), targetRow(s), explainer(s));
    paintGraphs();
    if (focusToggle) {
      focusToggle = false;
      section.querySelector(".xp-link")?.focus();
    }
    if (!revealed) {
      revealed = true;
      const setup = section.closest(".session-setup");
      setup?.classList.add("is-revealing");
      setTimeout(() => setup?.classList.remove("is-revealing"), 1400);
    }
  }

  // A repaint under a focused input would eat the learner's typing.
  const typing = () => {
    const active = document.activeElement;
    return !!(active && host()?.contains(active) && active.matches("input"));
  };

  let pending = false; // a summary arrived while the learner was typing
  window.addEventListener("delta:xp-summary", (e) => {
    summary = e.detail || null;
    if (summary && typing()) {
      pending = true;
      return;
    }
    paint();
  });

  // Redraw at the new width when the column changes size (rotation, the tab
  // becoming visible). Small jitters are ignored.
  let drawnWidth = 0;
  const watchWidth = () => {
    // The repaint the learner's typing held back lands when focus leaves.
    host()?.addEventListener("focusout", (e) => {
      if (pending && !host()?.contains(e.relatedTarget)) paint();
    });
    const right = graphs();
    if (!right || typeof ResizeObserver !== "function") return;
    let timer = null;
    new ResizeObserver(() => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        const w = chartWidth();
        if (summary && Math.abs(w - drawnWidth) > 24 && !typing()) paint();
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

  window.DDXpPanel = { paint };
})();
