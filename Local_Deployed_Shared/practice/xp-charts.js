/* ================================================================
   XP CHARTS — the Learner Home's two graphs, drawn for one time range.

   Seth, 2026-09-27: the Week / Month / 3 months / All tabs control BOTH
   graphs, and sit above them across the whole column. practice/xp-panel.js
   owns the tabs and the range; this file only draws.

   DAILY XP — one bar per day of the range, today's hatched (provisional:
     later answers can still move credit to the day it happened), today's
     target as a line.

   TOWARD THE COURSE — course knowledge, on the percent-of-course scale.
     * All: the whole journey, first day → finish (or the target date),
       on the full 0 → 100% axis.
     * A range of N days: the last N days and the next N, the axis zoomed to
       what is drawn. Forward, two slopes from today: the learner's pace and,
       with a target, what the target asks for — the gap between them is
       the reading. A full-scale axis would draw a week as a flat line.

   Both are drawn at the column's real width (never scaled), so the 10px
   labels stay 10px on a phone.
   ================================================================ */
(function () {
  "use strict";

  const SVG = "http://www.w3.org/2000/svg";
  const nf = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });
  const fmt = (n) => nf.format(Math.round(Number(n) || 0));

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
  const pctText = (v) => (v < 10 && v !== Math.round(v) ? v.toFixed(1) : String(Math.round(v)));

  const head = (title, note) => {
    const h = el("div", "xp-chart-head");
    h.appendChild(el("h3", "xp-chart-title", title));
    if (note) h.appendChild(el("p", "xp-chart-note", note));
    return h;
  };
  const frame = (svg) => {
    const f = el("div", "xp-chart-frame");
    f.appendChild(svg);
    return f;
  };

  // ── daily XP bars ──────────────────────────────────────────────
  /** `count` days ending today; a short history is padded with empty days
      so a week always reads as seven. */
  function bars(s, { count, width, peak: floor = 0 }) {
    const byDate = new Map(s.days.map((d) => [d.date, d]));
    const last = s.today.date;
    const days = [];
    for (let i = count - 1; i >= 0; i -= 1) {
      const iso = addDays(last, -i);
      days.push(byDate.get(iso) || { date: iso, xp: 0, answers: 0, knowledge: null });
    }
    const sum = days.reduce((a, d) => a + (d.xp || 0), 0);

    const box = el("section", "xp-chart");
    box.appendChild(head("Daily XP", `${fmt(sum)} XP · ${fmt(sum / days.length)} a day`));

    const W = width, H = 172, L = 34, R = W - 6, T = 14, B = 146;
    const target = s.today.target || 0;
    // `floor`: a shared top, so a group's cards compare by eye.
    const peak = Math.max(10, floor, target, ...days.map((d) => d.xp));
    // Round gridlines: a 1-2-5 step that fits the peak in at most four.
    const mag = 10 ** Math.floor(Math.log10(peak / 4));
    const tickStep = [1, 2, 5, 10].map((m) => m * mag).find((st) => peak / st <= 4);
    const ticks = Math.ceil(peak / tickStep);
    const top = ticks * tickStep * 1.06;
    const y = (v) => B - (v / top) * (B - T);
    const slot = (R - L) / days.length;
    const bw = Math.max(2, Math.min(36, slot * 0.58));

    const svg = svgEl("svg", {
      viewBox: `0 0 ${W} ${H}`, class: "xp-bars", role: "img",
      "aria-label": `Daily XP, ${shortDate(days[0].date)} to ${shortDate(last)}: ${fmt(sum)} XP` +
        (target ? `; today's target ${fmt(target)}` : ""),
    });
    const defs = svgEl("defs");
    const hatch = svgEl("pattern", { id: "xp-hatch", width: "5", height: "5", patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
    hatch.appendChild(svgEl("rect", { width: "2.5", height: "5", class: "xp-hatch-ink" }));
    defs.appendChild(hatch);
    svg.appendChild(defs);

    for (let i = 1; i <= ticks; i += 1) {
      const v = tickStep * i;
      svg.appendChild(svgEl("line", { x1: L, x2: R, y1: y(v), y2: y(v), class: "xp-grid" }));
      svg.appendChild(svgEl("text", { x: L - 8, y: y(v) + 3.5, "text-anchor": "end", class: "xp-axis" }, fmt(v)));
    }
    svg.appendChild(svgEl("text", { x: L - 8, y: B + 3.5, "text-anchor": "end", class: "xp-axis" }, "0"));

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
        x: cx - bw / 2, y: B - h, width: bw, height: h, rx: Math.min(2, bw / 3),
        class: "xp-bar-fill",
      }));
      g.appendChild(svgEl("title", {}, describe(d)));
      g.addEventListener("pointerenter", () => { readout.textContent = describe(d); });
      svg.appendChild(g);
      if ((days.length - 1 - i) % every === 0) {
        svg.appendChild(svgEl("text", { x: cx, y: B + 18, "text-anchor": "middle", class: `xp-axis${isToday ? " is-now" : ""}` },
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

    box.append(frame(svg), readout);
    return box;
  }

  // ── knowledge toward the course ────────────────────────────────
  /** A tick step in percent that puts three to six gridlines on `span`. */
  const niceStep = (span) => {
    const steps = [0.1, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25];
    return steps.find((st) => span / st <= 5) || 25;
  };

  const isMonday = (iso) => dateOf(iso).getDay() === 1;
  const monthName = (iso) =>
    dateOf(iso).toLocaleDateString(undefined, iso.slice(5, 7) === "01" ? { month: "short", year: "numeric" } : { month: "short" });

  /** Day rules and date labels under [start, end]. A week reads day by day
      (weekday over date, Mondays ruled heavier); a month by its Mondays;
      longer spans by month. Labels thin out rather than collide, and keep
      clear of the start / end labels when `ends` draws those. */
  function timeAxis(svg, { start, end, span, x, T, B, today, ends }) {
    const pxDay = (x(end) - x(start)) / span;
    const dates = Array.from({ length: span + 1 }, (_, i) => addDays(start, i));
    const rule = (iso, cls) => svg.appendChild(svgEl("line", { x1: x(iso), x2: x(iso), y1: T, y2: B, class: cls }));
    const label = (iso, text, row, anchor = "middle", extra = "") => svg.appendChild(svgEl("text", {
      x: x(iso), y: B + 18 + row * 13, "text-anchor": anchor,
      class: `xp-axis${iso === today ? " is-now" : ""}${extra}`,
    }, text));

    if (pxDay >= 14) {
      const weekday = pxDay >= 30 ? "short" : "narrow";
      dates.forEach((iso) => {
        rule(iso, isMonday(iso) ? "xp-day-rule is-week" : "xp-day-rule");
        label(iso, dateOf(iso).toLocaleDateString(undefined, { weekday }), 0);
        label(iso, String(dateOf(iso).getDate()), 1, "middle", " is-date");
      });
      return;
    }
    const byWeek = pxDay * 7 >= 44;
    if (pxDay * 7 >= 12) dates.filter(isMonday).forEach((iso) => rule(iso, "xp-day-rule is-week"));
    const marks = dates.filter(byWeek ? isMonday : (iso) => iso.endsWith("-01"));
    if (!byWeek && pxDay * 7 < 12) marks.forEach((iso) => rule(iso, "xp-day-rule is-week"));
    const gap = marks.length > 1 ? x(marks[1]) - x(marks[0]) : Infinity;
    const every = Math.max(1, Math.ceil(48 / gap));
    const clear = ends ? 46 : 16; // an edge label, or the frame's edge
    marks.forEach((iso, i) => {
      if (i % every || x(iso) < x(start) + clear || x(iso) > x(end) - clear) return;
      label(iso, byWeek ? shortDate(iso) : monthName(iso), 0);
    });
    if (ends) {
      label(start, shortDate(start), 0, "start");
      label(end, typeof ends === "string" ? ends : shortDate(end), 0, "end");
    }
  }

  /** `window` pins the x range while the target is dragged, so the axis
      cannot slide under the pointer. */
  function trajectory(s, { count, width, window: pinned, yWindow, self = true }) {
    const box = el("section", "xp-chart");
    const total = s.course.total_xp || 1;
    const pct = (v) => (Math.min(total, Math.max(0, v)) / total) * 100;
    const today = s.today.date;
    const all = !Number.isFinite(count);
    const history = s.days.filter((d) => Number.isFinite(d.knowledge));

    // Net change over the drawn past, forgetting included — the quantity the
    // finish date divides.
    const from = all ? null : addDays(today, -count);
    const base = all
      ? (history.length ? (s.starting_credit ?? history[0].knowledge) : s.knowledge)
      // `days` is contiguous, so a miss means the range starts before the
      // first day: its open is the starting credit.
      : (history.find((d) => d.date === from)?.knowledge ?? s.starting_credit ?? history[0]?.knowledge ?? s.knowledge);
    const net = s.knowledge - base;
    box.appendChild(head("Toward the course",
      `${pctText(pct(s.knowledge))}% · ${net >= 0 ? "+" : "−"}${fmt(Math.abs(net))} XP net`));
    if (!history.length) return box;

    const target = s.target?.mode === "date" && s.target.date >= today ? s.target.date : null;
    const need = s.today.target || 0; // XP a day the target asks for, either mode

    // The x window and the y window.
    let start, end, lo, hi;
    const horizon = addDays(today, 365);
    if (all) {
      start = history[0].date;
      // A month ahead at least, so there is always a future to click.
      const ends = [addDays(today, 30)];
      if (s.projected_finish) ends.push(s.projected_finish);
      if (target) ends.push(target);
      // A projection years away would flatten the history into a sliver;
      // the axis stops a year out and the label says it goes on.
      end = ends.reduce((a, b) => (b > a ? b : a));
      if (end > horizon) end = horizon;
      lo = 0;
      hi = 100;
    } else {
      start = addDays(today, -(count - 1));
      end = addDays(today, count);
    }
    if (pinned) ({ start, end } = pinned);
    const span = Math.max(1, daysBetween(start, end));
    const shown = history.filter((d) => d.date >= start);
    const ahead = Math.max(0, daysBetween(today, end));
    const paceEnd = s.knowledge + (s.pace || 0) * ahead;
    const needEnd = need ? s.knowledge + need * Math.max(0, daysBetween(today, target && target <= end ? target : end)) : null;
    let step = 25;
    if (!all) {
      const vals = [...shown.map((d) => pct(d.knowledge)), pct(s.knowledge), pct(paceEnd)];
      if (needEnd !== null) vals.push(pct(needEnd));
      const vmin = Math.min(...vals), vmax = Math.max(...vals);
      step = niceStep(Math.max(0.3, vmax - vmin));
      lo = Math.max(0, Math.floor(vmin / step) * step);
      hi = Math.min(100, Math.max(lo + step, Math.ceil(vmax / step) * step));
      lo = Math.min(lo, hi - step); // at 100% both clamp to the top
      // A group's cards share one y window (practice/xp-group-view.js).
      if (yWindow) {
        step = niceStep(Math.max(0.3, yWindow.hi - yWindow.lo));
        lo = Math.max(0, Math.floor(yWindow.lo / step) * step);
        hi = Math.min(100, Math.max(lo + step, Math.ceil(yWindow.hi / step) * step));
      }
    }

    const W = width, H = 200, L = 40, R = W - 10, T = 18, B = 160;
    const x = (iso) => L + (Math.min(span, Math.max(0, daysBetween(start, iso))) / span) * (R - L);
    const yp = (p) => B - ((Math.min(hi, Math.max(lo, p)) - lo) / (hi - lo)) * (B - T);
    const y = (v) => yp(pct(v));

    const svg = svgEl("svg", {
      viewBox: `0 0 ${W} ${H}`, class: "xp-trajectory", role: "img",
      "aria-label": `Course knowledge ${pctText(pct(s.knowledge))}%, ${fmt(s.knowledge)} of ${fmt(total)} XP` +
        (s.projected_finish ? `; at this pace the course is finished ${longDate(s.projected_finish)}` : ""),
    });
    const defs = svgEl("defs");
    const area = svgEl("linearGradient", { id: "xp-area-grad", x1: "0", y1: "0", x2: "0", y2: "1" });
    area.append(svgEl("stop", { offset: "0", class: "xp-area-top" }), svgEl("stop", { offset: "1", class: "xp-area-bottom" }));
    defs.appendChild(area);
    svg.appendChild(defs);

    for (let p = lo; p <= hi + 1e-9; p += step) {
      const v = Math.round(p * 100) / 100;
      svg.appendChild(svgEl("line", { x1: L, x2: R, y1: yp(v), y2: yp(v), class: v === lo ? "xp-floor" : "xp-grid" }));
      svg.appendChild(svgEl("text", { x: L - 8, y: yp(v) + 3.5, "text-anchor": "end", class: "xp-axis" }, `${pctText(v)}%`));
    }

    const beyond = all && end === horizon && [s.projected_finish, target].some((d) => d && d > horizon);
    timeAxis(svg, { start, end, span, x, T, B, today, ends: all && (beyond ? `${shortDate(end)} →` : true) });

    // Today's rule splits what happened from what is projected.
    const nowX = x(today), nowY = y(s.knowledge);
    if (!all) {
      svg.appendChild(svgEl("line", { x1: nowX, x2: nowX, y1: T, y2: B, class: "xp-now-rule" }));
      // A week names today in its own axis; longer ranges label the rule.
      if (count > 7) svg.appendChild(svgEl("text", { x: nowX, y: T - 6, "text-anchor": "middle", class: "xp-axis is-now" }, "Today"));
    }

    if (shown.length) {
      const pts = shown.map((d) => `${x(d.date).toFixed(1)},${y(d.knowledge).toFixed(1)}`);
      svg.appendChild(svgEl("path", {
        d: `M${x(shown[0].date)},${B} L${pts.join(" L")} L${nowX},${B} Z`, fill: "url(#xp-area-grad)", class: "xp-area",
      }));
      svg.appendChild(svgEl("path", { d: `M${pts.join(" L")}`, class: "xp-line" }));
    }

    if (all) {
      if (s.projected_finish) {
        svg.appendChild(svgEl("line", {
          x1: nowX, y1: nowY, x2: x(s.projected_finish),
          y2: y(s.projected_finish > horizon ? paceEnd : total), class: "xp-projection",
        }));
      }
      if (target) {
        const tx = x(target);
        svg.appendChild(svgEl("line", { x1: nowX, y1: nowY, x2: tx, y2: y(total), class: "xp-needed" }));
        svg.appendChild(svgEl("circle", { cx: tx, cy: y(total), r: 4, class: "xp-target-dot" }));
        svg.appendChild(svgEl("text", { x: Math.min(tx, R - 2), y: T - 6, "text-anchor": tx > R - 60 ? "end" : "middle", class: "xp-target-label" },
          `target ${shortDate(target)}`));
      }
    } else {
      if (s.pace > 0) svg.appendChild(svgEl("line", { x1: nowX, y1: nowY, x2: x(end), y2: y(paceEnd), class: "xp-projection" }));
      if (needEnd !== null) {
        const inside = target && target <= end;
        const nx = inside ? x(target) : x(end), ny = y(needEnd);
        svg.appendChild(svgEl("line", { x1: nowX, y1: nowY, x2: nx, y2: ny, class: "xp-needed" }));
        // The dot is the handle: past the window it waits, hollow, at the edge.
        if (target) {
          svg.appendChild(svgEl("circle", { cx: nx, cy: ny, r: 4, class: `xp-target-dot${inside ? "" : " is-beyond"}` }));
          const right = nx < R - 56;
          svg.appendChild(svgEl("text", {
            x: right ? nx + 8 : nx - 8, y: ny + (ny < T + 10 ? 12 : -8), "text-anchor": right ? "start" : "end", class: "xp-target-label",
          }, inside ? shortDate(target) : `${shortDate(target)} →`));
        }
      }
    }
    svg.appendChild(svgEl("circle", { cx: nowX, cy: nowY, r: 4.5, class: "xp-now-dot" }));

    box.appendChild(frame(svg));
    box.appendChild(el("p", "xp-readout"));
    // For practice/xp-target-drag.js: where the plot is, in SVG units.
    box.xpPlot = { svg, W, L, R, start, end, span, lo, hi };

    const legend = el("ul", "xp-legend");
    const key = (cls, text) => {
      const li = el("li");
      li.append(el("i", `xp-key ${cls}`), document.createTextNode(text));
      legend.appendChild(li);
    };
    const your = self ? "your " : "";
    key("xp-key-line", `${your}knowledge`);
    if (s.pace > 0) key("xp-key-projection", `${your}pace, ${fmt(s.pace)}/day`);
    if (all ? target : needEnd !== null) key("xp-key-needed", `${your}target, ${fmt(need)}/day`);
    box.appendChild(legend);
    return box;
  }

  window.DDXpCharts = {
    bars,
    trajectory,
    util: { el, fmt, shortDate, longDate, daysBetween, addDays },
  };
})();
