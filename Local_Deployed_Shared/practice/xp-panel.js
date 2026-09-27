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
     * DAILY XP bars with today's target line — a week, a month, three
       months or everything. Today's bar is hatched: it is provisional,
       because later answers can still revise when the model believes the
       learning happened;
     * TOWARD THE COURSE: knowledge over time on the full 0 → total scale,
       the pace projection, and the target date if there is one.

   It draws from `delta:xp-summary`, which ../xp.js broadcasts after every
   read it makes for the level pill — one request feeds both. Nothing here
   fetches the summary itself except through `DeltaXP.refresh`.

   `.is-empty` hides both halves when there is no summary (no session, or
   the read failed); the idle card then narrows to its one button.
   ================================================================ */
(function () {
  "use strict";

  const SVG = "http://www.w3.org/2000/svg";
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
  const nf = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });
  const fmt = (n) => nf.format(Math.round(Number(n) || 0));
  const compact = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });

  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const svgEl = (tag, attrs = {}, text) => {
    const node = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text !== undefined) node.textContent = text;
    return node;
  };

  const dateOf = (iso) => {
    const [y, m, d] = String(iso).split("-").map(Number);
    return new Date(y, m - 1, d);
  };
  const isoOf = (d) =>
    `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  const shortDate = (iso) => dateOf(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
  const weekday = (iso) => dateOf(iso).toLocaleDateString(undefined, { weekday: "short" });
  const longDate = (iso) =>
    dateOf(iso).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
  const daysBetween = (a, b) => Math.round((dateOf(b) - dateOf(a)) / 86400000);
  // Calendar days, not 24-hour blocks: across a DST change N×86400000 ms from
  // local midnight lands at 23:00 the day before.
  const addDays = (iso, n) => {
    const d = dateOf(iso);
    d.setDate(d.getDate() + n);
    return isoOf(d);
  };
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
      sub: s.pace > 0 ? `at ${fmt(s.pace)} XP/day` : "no pace yet",
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
    ].forEach((t) => ul.appendChild(el("li", "", t)));
    d.appendChild(ul);
    return d;
  }

  // ── daily XP bars ──────────────────────────────────────────────
  function barsChart(s) {
    const box = el("section", "xp-chart");
    const head = el("div", "xp-chart-head");
    head.appendChild(el("h3", "xp-chart-title", "Daily XP"));
    const tabs = el("div", "xp-range");
    // Plain toggle buttons, not a tablist: there are no tab panels, and
    // Tab + Enter is the whole keyboard contract a pressed button needs.
    tabs.setAttribute("role", "group");
    tabs.setAttribute("aria-label", "Days shown");
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
        paint();
      });
      tabs.appendChild(b);
    });
    head.appendChild(tabs);
    box.appendChild(head);

    const wanted = RANGES.find((r) => r.id === range)?.days ?? 7;
    // Pad a short history with empty days so a week always reads as seven.
    const byDate = new Map(s.days.map((d) => [d.date, d]));
    const last = s.today.date;
    const count = Number.isFinite(wanted) ? wanted : s.days.length;
    const days = [];
    for (let i = count - 1; i >= 0; i -= 1) {
      const iso = addDays(last, -i);
      days.push(byDate.get(iso) || { date: iso, xp: 0, answers: 0, knowledge: null });
    }

    const W = chartWidth(), H = 196, L = 34, R = W - 8, T = 16, B = 168;
    const target = s.today.target || 0;
    const top = Math.max(10, target, ...days.map((d) => d.xp)) * 1.12;
    const y = (v) => B - (v / top) * (B - T);
    const slot = (R - L) / days.length;
    const bw = Math.max(2, Math.min(40, slot * 0.62));

    const svg = svgEl("svg", {
      viewBox: `0 0 ${W} ${H}`, class: "xp-bars", role: "img",
      "aria-label": `Daily XP, ${shortDate(days[0].date)} to ${shortDate(last)}${target ? `; today's target ${fmt(target)}` : ""}`,
    });
    const defs = svgEl("defs");
    const grad = svgEl("linearGradient", { id: "xp-bar-grad", x1: "0", y1: "1", x2: "0", y2: "0" });
    grad.append(svgEl("stop", { offset: "0", class: "xp-stop-from" }), svgEl("stop", { offset: "1", class: "xp-stop-to" }));
    const hatch = svgEl("pattern", { id: "xp-hatch", width: "6", height: "6", patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
    hatch.appendChild(svgEl("rect", { width: "3", height: "6", class: "xp-hatch-ink" }));
    defs.append(grad, hatch);
    svg.appendChild(defs);

    // Four quiet gridlines and their values.
    for (let i = 0; i <= 3; i += 1) {
      const v = (top / 1.12) * (i / 3);
      svg.appendChild(svgEl("line", { x1: L, x2: R, y1: y(v), y2: y(v), class: "xp-grid" }));
      svg.appendChild(svgEl("text", { x: L - 6, y: y(v) + 3.5, "text-anchor": "end", class: "xp-axis" }, fmt(v)));
    }

    const readout = el("p", "xp-readout");
    const describe = (d) =>
      `${longDate(d.date)} · ${fmt(d.xp)} XP · ${d.answers || 0} answer${d.answers === 1 ? "" : "s"}` +
      (d.date === last ? " · today, still settling" : "");
    readout.textContent = describe(days[days.length - 1]);

    const every = Math.ceil(days.length / 7);
    days.forEach((d, i) => {
      const cx = L + slot * (i + 0.5);
      const h = Math.max(d.xp > 0 ? 2 : 0, B - y(d.xp));
      const isToday = d.date === last;
      const g = svgEl("g", { class: `xp-bar${isToday ? " is-today" : ""}`, style: `--i:${i}` });
      g.appendChild(svgEl("rect", { x: cx - slot / 2, y: T, width: slot, height: B - T, class: "xp-bar-hit" }));
      g.appendChild(svgEl("rect", {
        x: cx - bw / 2, y: B - h, width: bw, height: h, rx: Math.min(3, bw / 3),
        class: "xp-bar-fill", fill: isToday ? "url(#xp-hatch)" : "url(#xp-bar-grad)",
      }));
      g.appendChild(svgEl("title", {}, describe(d)));
      g.addEventListener("pointerenter", () => { readout.textContent = describe(d); });
      svg.appendChild(g);
      if ((days.length - 1 - i) % every === 0) {
        svg.appendChild(svgEl("text", { x: cx, y: B + 18, "text-anchor": "middle", class: "xp-axis" },
          isToday ? "Today" : days.length <= 7 ? weekday(d.date) : shortDate(d.date)));
      }
    });
    svg.appendChild(svgEl("line", { x1: L, x2: R, y1: B, y2: B, class: "xp-floor" }));
    if (target) {
      svg.appendChild(svgEl("line", { x1: L, x2: R, y1: y(target), y2: y(target), class: "xp-target-line" }));
      svg.appendChild(svgEl("text", { x: R, y: y(target) - 6, "text-anchor": "end", class: "xp-target-label" },
        `target ${fmt(target)}`));
    }
    svg.addEventListener("pointerleave", () => { readout.textContent = describe(days[days.length - 1]); });

    const frame = el("div", "xp-chart-frame");
    frame.appendChild(svg);
    box.append(frame, readout);
    return box;
  }

  // ── trajectory toward the course ───────────────────────────────
  function trajectory(s) {
    const box = el("section", "xp-chart");
    const head = el("div", "xp-chart-head");
    head.appendChild(el("h3", "xp-chart-title", "Toward the course"));
    box.appendChild(head);

    const history = s.days.filter((d) => Number.isFinite(d.knowledge));
    if (!history.length) return box;
    const start = history[0].date;
    const today = s.today.date;
    const ends = [today];
    if (s.projected_finish) ends.push(s.projected_finish);
    if (s.target?.mode === "date") ends.push(s.target.date);
    // A projection years away would flatten the history into a sliver;
    // the axis stops a year out and the line says where it was heading.
    const horizon = addDays(today, 365);
    let end = ends.reduce((a, b) => (b > a ? b : a));
    if (end > horizon) end = horizon;
    const span = Math.max(1, daysBetween(start, end));

    const W = chartWidth(), H = 186, L = 34, R = W - 12, T = 18, B = 158;
    const total = s.course.total_xp || 1;
    const x = (iso) => L + (Math.min(span, Math.max(0, daysBetween(start, iso))) / span) * (R - L);
    const y = (v) => B - (Math.min(total, Math.max(0, v)) / total) * (B - T);

    const svg = svgEl("svg", {
      viewBox: `0 0 ${W} ${H}`, class: "xp-trajectory", role: "img",
      "aria-label": `Course knowledge ${fmt(s.knowledge)} of ${fmt(total)} XP` +
        (s.projected_finish ? `; at this pace the course is finished ${longDate(s.projected_finish)}` : ""),
    });
    const defs = svgEl("defs");
    const area = svgEl("linearGradient", { id: "xp-area-grad", x1: "0", y1: "0", x2: "0", y2: "1" });
    area.append(svgEl("stop", { offset: "0", class: "xp-area-top" }), svgEl("stop", { offset: "1", class: "xp-area-bottom" }));
    const stroke = svgEl("linearGradient", { id: "xp-line-grad", x1: "0", y1: "0", x2: "1", y2: "0" });
    stroke.append(svgEl("stop", { offset: "0", class: "xp-stop-from" }), svgEl("stop", { offset: "1", class: "xp-stop-to" }));
    defs.append(area, stroke);
    svg.appendChild(defs);

    [0, 0.25, 0.5, 0.75, 1].forEach((f) => {
      svg.appendChild(svgEl("line", { x1: L, x2: R, y1: y(total * f), y2: y(total * f), class: "xp-grid" }));
      svg.appendChild(svgEl("text", { x: L - 6, y: y(total * f) + 3.5, "text-anchor": "end", class: "xp-axis" },
        `${Math.round(f * 100)}%`));
    });

    const pts = history.map((d) => `${x(d.date).toFixed(1)},${y(d.knowledge).toFixed(1)}`);
    svg.appendChild(svgEl("path", {
      d: `M${x(start)},${B} L${pts.join(" L")} L${x(today)},${B} Z`, fill: "url(#xp-area-grad)", class: "xp-area",
    }));
    svg.appendChild(svgEl("path", { d: `M${pts.join(" L")}`, class: "xp-line", stroke: "url(#xp-line-grad)" }));

    const nowX = x(today), nowY = y(s.knowledge);
    if (s.projected_finish) {
      svg.appendChild(svgEl("line", {
        x1: nowX, y1: nowY, x2: x(s.projected_finish), y2: y(s.projected_finish > horizon
          ? s.knowledge + s.pace * daysBetween(today, horizon) : total), class: "xp-projection",
      }));
    }
    if (s.target?.mode === "date") {
      const tx = x(s.target.date);
      svg.appendChild(svgEl("line", { x1: nowX, y1: nowY, x2: tx, y2: y(total), class: "xp-needed" }));
      svg.appendChild(svgEl("line", { x1: tx, x2: tx, y1: T, y2: B, class: "xp-target-rule" }));
      svg.appendChild(svgEl("circle", { cx: tx, cy: y(total), r: 4.5, class: "xp-target-dot" }));
      svg.appendChild(svgEl("text", { x: Math.min(tx, R - 2), y: T - 5, "text-anchor": tx > R - 60 ? "end" : "middle", class: "xp-target-label" },
        `target ${shortDate(s.target.date)}`));
    }
    svg.appendChild(svgEl("circle", { cx: nowX, cy: nowY, r: 5, class: "xp-now-dot" }));
    svg.appendChild(svgEl("line", { x1: L, x2: R, y1: B, y2: B, class: "xp-floor" }));
    svg.appendChild(svgEl("text", { x: L, y: B + 18, class: "xp-axis" }, shortDate(start)));
    svg.appendChild(svgEl("text", { x: R, y: B + 18, "text-anchor": "end", class: "xp-axis" },
      end === horizon && ends.some((e) => e > horizon) ? `${shortDate(end)} →` : shortDate(end)));

    const frame = el("div", "xp-chart-frame");
    frame.appendChild(svg);
    box.appendChild(frame);

    const legend = el("ul", "xp-legend");
    const key = (cls, text) => {
      const li = el("li");
      li.append(el("i", `xp-key ${cls}`), document.createTextNode(text));
      legend.appendChild(li);
    };
    key("xp-key-line", "your knowledge");
    if (s.projected_finish) key("xp-key-projection", "at your pace");
    if (s.target?.mode === "date") key("xp-key-needed", "needed for your date");
    box.appendChild(legend);
    return box;
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
    // Measured after the column is shown: a hidden column is 0 wide.
    drawnWidth = chartWidth();
    right?.replaceChildren(barsChart(s), trajectory(s));
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
