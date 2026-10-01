/* ================================================================
   XP CHARTS — the Learner Home's daily XP bars (and the date helpers the
   home shares).

   DAILY XP — one bar per day of the range, today's hatched (provisional:
     later answers can still move credit to the day it happened). `start`
     fixes the range's first day instead of ending it today: the home draws
     Monday through Sunday (Seth, 2026-10-01), and days still to come get a
     label and no bar. Today's
     target as a line unless `target: false` (a number draws the line
     there instead). `drag: true` leaves headroom above the line, draws its
     knob at the right end and hands ./xp-target.js the pieces it moves
     (`box.xpTarget`). `bare: true` drops the chart's
     own title and total, for a home that states them above it (Seth,
     2026-09-29). `measure: "solved"` counts problems solved instead
     (Seth, 2026-09-28), with no target: the target is in XP.

   🪦 TOWARD THE COURSE (the knowledge line, its pace and target slopes, and
   the drag-to-set target of xp-target-drag.js) was removed with the home's
   dropdowns (Seth, 2026-09-29: "it just has the bar graph"); git has it.

   Drawn at the column's real width (never scaled), so the 10px labels
   stay 10px on a phone.
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
  /** The smallest whole 1-2-5 step (1, 2, 5, 10, 20 …) that puts at most
      `lines` gridlines on `span`. */
  const step125 = (span, lines) => {
    for (let mag = 1; ; mag *= 10) {
      const st = [1, 2, 5].map((m) => m * mag).find((v) => span / v <= lines);
      if (st) return st;
    }
  };

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
  /** `count` days ending today, or from `start`; a short history is padded
      with empty days so a week always reads as seven. */
  function bars(s, { count, start, width, peak: floor = 0, measure = "xp", bare = false, target: withTarget = true, drag = false }) {
    const solved = measure === "solved";
    const val = (d) => (solved ? d.solved : d.xp) || 0;
    const unit = solved ? "solved" : "XP";
    const byDate = new Map(s.days.map((d) => [d.date, d]));
    const last = s.today.date;
    const first = start || addDays(last, -(count - 1));
    const days = [];
    for (let i = 0; i < count; i += 1) {
      const iso = addDays(first, i);
      days.push(byDate.get(iso) || { date: iso, xp: 0, answers: 0, solved: 0, knowledge: null });
    }
    const todayAt = Math.max(0, days.findIndex((d) => d.date === last));
    const ahead = (d) => d.date > last;
    const sum = days.reduce((a, d) => a + val(d), 0);

    const box = el("section", "xp-chart");
    if (!bare) {
      box.appendChild(head(solved ? "Solved per day" : "Daily XP",
        `${fmt(sum)} ${unit} · ${solved ? (sum / days.length).toFixed(1) : fmt(sum / days.length)} a day`));
    }

    const W = width, H = 172, L = 34, R = W - 6, T = 14, B = 146;
    const target = solved || withTarget === false ? 0
      : typeof withTarget === "number" ? withTarget : s.today.target || 0;
    // `floor`: a shared top, so a group's cards compare by eye. Dragged, the
    // line gets room above it; letting go redraws with more (./xp-target.js).
    const peak = Math.max(solved ? 4 : 10, floor, target * (drag ? 1.4 : 1), ...days.map(val));
    // Round gridlines: a 1-2-5 step that fits the peak in at most four
    // (the peak is at least 4, so the step is whole — no 0.5 solved).
    const tickStep = step125(peak, 4);
    const ticks = Math.ceil(peak / tickStep);
    const top = ticks * tickStep * 1.06;
    const y = (v) => B - (v / top) * (B - T);
    const slot = (R - L) / days.length;
    const bw = Math.max(2, Math.min(36, slot * 0.58));

    const svg = svgEl("svg", {
      viewBox: `0 0 ${W} ${H}`, class: "xp-bars", role: "img",
      "aria-label": `${solved ? "Problems solved" : "Daily XP"}, ${shortDate(days[0].date)} to ${shortDate(days[days.length - 1].date)}: ${fmt(sum)} ${unit}` +
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
      `${longDate(d.date)} · ${solved ? `${d.solved || 0} solved of` : `${fmt(d.xp)} XP ·`} ${d.answers || 0} answer${d.answers === 1 ? "" : "s"}` +
      (d.date === last ? " · today, still settling" : "");
    const describeAny = (d) => (ahead(d) ? `${longDate(d.date)} · still to come` : describe(d));
    readout.textContent = describe(days[todayAt]);

    const every = Math.ceil(days.length / 7);
    days.forEach((d, i) => {
      const cx = L + slot * (i + 0.5);
      const h = Math.max(val(d) > 0 ? 2 : 0, B - y(val(d)));
      const isToday = d.date === last;
      const g = svgEl("g", { class: `xp-bar${isToday ? " is-today" : ""}${ahead(d) ? " is-ahead" : ""}`, style: `--i:${i}` });
      g.appendChild(svgEl("rect", { x: cx - slot / 2, y: T, width: slot, height: B - T, class: "xp-bar-hit" }));
      g.appendChild(svgEl("rect", {
        x: cx - bw / 2, y: B - h, width: bw, height: h, rx: Math.min(2, bw / 3),
        class: "xp-bar-fill",
      }));
      g.appendChild(svgEl("title", {}, describeAny(d)));
      g.addEventListener("pointerenter", () => { readout.textContent = describeAny(d); });
      svg.appendChild(g);
      if ((days.length - 1 - i) % every === 0) {
        svg.appendChild(svgEl("text", { x: cx, y: B + 18, "text-anchor": "middle", class: `xp-axis${isToday ? " is-now" : ""}${ahead(d) ? " is-ahead" : ""}` },
          isToday ? "Today" : days.length <= 7 ? weekday(d.date) : shortDate(d.date)));
      }
    });
    svg.appendChild(svgEl("line", { x1: L, x2: R, y1: B, y2: B, class: "xp-floor" }));
    if (target) {
      const line = svgEl("line", { x1: L, x2: R, y1: y(target), y2: y(target), class: "xp-target-line" });
      const label = svgEl("text", { x: drag ? R - 14 : R, y: y(target) - 6, "text-anchor": "end", class: "xp-target-label" },
        `target ${fmt(target)}`);
      svg.append(line, label);
      if (drag) {
        // The knob: a grip at the line's right end. Its hit circle is larger
        // than what is drawn, for a finger.
        const knob = svgEl("g", { class: "xp-target-knob", transform: `translate(${R - 4} ${y(target)})` });
        knob.append(
          svgEl("circle", { r: 14, class: "xp-target-knob-hit" }),
          svgEl("circle", { r: 6.5, class: "xp-target-knob-dot" }),
          svgEl("path", { d: "M-2.5 -1.5 L0 -4 L2.5 -1.5 M-2.5 1.5 L0 4 L2.5 1.5", class: "xp-target-knob-grip" }),
        );
        svg.appendChild(knob);
        // XP at an SVG y, and the y of an XP value, on this chart's scale.
        box.xpTarget = { svg, line, label, knob, W, R, T, B, top, y, valueAt: (sy) => ((B - sy) / (B - T)) * top };
      }
    }
    svg.addEventListener("pointerleave", () => { readout.textContent = describe(days[todayAt]); });

    box.append(frame(svg), readout);
    return box;
  }

  window.DDXpCharts = {
    bars,
    util: { el, svgEl, head, frame, fmt, dateOf, shortDate, longDate, daysBetween, addDays, step125 },
  };
})();
