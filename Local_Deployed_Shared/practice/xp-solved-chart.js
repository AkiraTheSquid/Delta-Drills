/* ================================================================
   XP SOLVED CHART — the course graph's counterpart in problems solved.

   Seth, 2026-09-28: "make it such that you can also display the number of
   problems solved instead of the exp points. for the bar graph and
   everything else … that should be a third dropdown". The dropdown is
   ./xp-group-view.js's Measure; the bars take `measure: "solved"`
   (./xp-charts.js). This file draws what "Toward the course" becomes: the
   running total of problems solved (a right answer — app/learning_xp.py
   `days[].solved`), with the pace line ahead of today.

   A course has no size in problems, so there is no finish, no percent and
   no target here; the target is set in XP.

   Same windows as the course graph so the two read alike:
     * All: the first day → a month ahead, from 0;
     * a range of N days: the last N and the next N, the axis zoomed.
   `window` / `yWindow` pin both, so a group's rows share one scale.
   ================================================================ */
(function () {
  "use strict";

  const { el, svgEl, head, frame, fmt, daysBetween, addDays, timeAxis, step125 } =
    window.DDXpCharts.util;

  /** Solved per day over the last `pace_days` days, today included. */
  const paceOf = (s) => {
    const n = s.pace_days || 14;
    const recent = s.days.slice(-n);
    return recent.reduce((a, d) => a + (d.solved || 0), 0) / n;
  };

  function cumulative(s, { count, width, window: pinned, yWindow, self = true }) {
    const box = el("section", "xp-chart");
    const today = s.today.date;
    const all = !Number.isFinite(count);
    let run = 0;
    const history = s.days.map((d) => ({ date: d.date, total: (run += d.solved || 0) }));
    const now = run;
    const pace = paceOf(s);

    const from = all ? null : addDays(today, -count);
    // `days` is contiguous: no row for `from` means the range opens before
    // the first day, when nothing was solved yet.
    const base = all ? 0 : history.find((d) => d.date === from)?.total ?? 0;
    box.appendChild(head("Solved in all", all ? `${fmt(now)} in all` : `${fmt(now)} in all · +${fmt(now - base)}`));
    if (!history.length) return box;

    let start, end;
    if (all) {
      start = history[0].date;
      end = addDays(today, 30);
    } else {
      start = addDays(today, -(count - 1));
      end = addDays(today, count);
    }
    if (pinned) ({ start, end } = pinned);
    const span = Math.max(1, daysBetween(start, end));
    const shown = history.filter((d) => d.date >= start);
    const paceEnd = now + pace * Math.max(0, daysBetween(today, end));

    const vals = [...shown.map((d) => d.total), now, paceEnd];
    let lo = all ? 0 : Math.min(...vals);
    let hi = Math.max(...vals);
    if (yWindow) ({ lo, hi } = yWindow);
    const step = step125(Math.max(1, hi - lo), 5);
    lo = Math.max(0, Math.floor(lo / step) * step);
    hi = Math.max(lo + step, Math.ceil(hi / step) * step);

    const W = width, H = 200, L = 40, R = W - 10, T = 18, B = 160;
    const x = (iso) => L + (Math.min(span, Math.max(0, daysBetween(start, iso))) / span) * (R - L);
    const y = (v) => B - ((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * (B - T);

    const svg = svgEl("svg", {
      viewBox: `0 0 ${W} ${H}`, class: "xp-solved-total", role: "img",
      "aria-label": `${fmt(now)} problems solved in all` +
        (pace > 0 ? `; ${pace.toFixed(1)} a day over the last ${s.pace_days || 14} days` : ""),
    });
    const defs = svgEl("defs");
    const area = svgEl("linearGradient", { id: "xp-area-grad", x1: "0", y1: "0", x2: "0", y2: "1" });
    area.append(svgEl("stop", { offset: "0", class: "xp-area-top" }), svgEl("stop", { offset: "1", class: "xp-area-bottom" }));
    defs.appendChild(area);
    svg.appendChild(defs);

    for (let v = lo; v <= hi; v += step) {
      svg.appendChild(svgEl("line", { x1: L, x2: R, y1: y(v), y2: y(v), class: v === lo ? "xp-floor" : "xp-grid" }));
      svg.appendChild(svgEl("text", { x: L - 8, y: y(v) + 3.5, "text-anchor": "end", class: "xp-axis" }, fmt(v)));
    }
    timeAxis(svg, { start, end, span, x, T, B, today, ends: all });

    const nowX = x(today), nowY = y(now);
    if (!all) {
      svg.appendChild(svgEl("line", { x1: nowX, x2: nowX, y1: T, y2: B, class: "xp-now-rule" }));
      if (count > 7) svg.appendChild(svgEl("text", { x: nowX, y: T - 6, "text-anchor": "middle", class: "xp-axis is-now" }, "Today"));
    }
    if (shown.length) {
      const pts = shown.map((d) => `${x(d.date).toFixed(1)},${y(d.total).toFixed(1)}`);
      svg.appendChild(svgEl("path", {
        d: `M${x(shown[0].date)},${B} L${pts.join(" L")} L${nowX},${B} Z`, fill: "url(#xp-area-grad)", class: "xp-area",
      }));
      svg.appendChild(svgEl("path", { d: `M${pts.join(" L")}`, class: "xp-line" }));
    }
    if (pace > 0) svg.appendChild(svgEl("line", { x1: nowX, y1: nowY, x2: x(end), y2: y(paceEnd), class: "xp-projection" }));
    svg.appendChild(svgEl("circle", { cx: nowX, cy: nowY, r: 4.5, class: "xp-now-dot" }));

    box.appendChild(frame(svg));
    box.xpPlot = { svg, W, L, R, start, end, span, lo, hi };

    const legend = el("ul", "xp-legend");
    const key = (cls, text) => {
      const li = el("li");
      li.append(el("i", `xp-key ${cls}`), document.createTextNode(text));
      legend.appendChild(li);
    };
    const your = self ? "your " : "";
    key("xp-key-line", `${your}problems solved`);
    if (pace > 0) key("xp-key-projection", `${your}pace, ${pace.toFixed(1)}/day`);
    box.appendChild(legend);
    return box;
  }

  window.DDXpSolvedChart = { cumulative };
})();
