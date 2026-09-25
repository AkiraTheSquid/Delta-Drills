/* Fig. 4 — the forgetting curve. Plain SVG, redrawn on resize/theme and on
   every drag frame (the whole timeline re-simulates; it is ~300 points).

   Model (illustrative, stated on the page):
     recall(t) = r0 * 2^(-(t - tLast) / h)
     h0 = 1.5 d × speed
     full review at recall r:  recall → 1,  h ← h × (1 + speed·(1 + 3(1 − r)))
     implicit rep of weight w: recall → r + w(1 − r),  h grows by w of that factor
   "Production today" mode = what the backend does now: fixed 14-day half-life,
   a review restores recall but never lengthens the next interval.

   Six review handles sit on a track under the plot. A handle dragged past the
   plot's right edge parks in the "unused" lane and does not count. Handles
   are focusable: ←/→ move a day (Shift: five), End parks, Home returns.
   Until 2026-09-24 this figure ran on CindyJS (~550 KB); the model below is
   a line-for-line port of that CindyScript. */
(function () {
  "use strict";
  var svg = document.getElementById("aisc-forget-svg");
  if (!svg) return;
  var NS = "http://www.w3.org/2000/svg";
  var DAYS = 60, N_REV = 6, IMPLICIT_DAYS = [3, 8, 14, 21, 29, 38, 48], HANDLE = "#e6663f";
  var S = { speed: 1, target: 0.7, implicit: false, fixed: false };
  var rev = [];                               // day of review i, or null = parked
  for (var i = 0; i < N_REV; i++) rev.push(null);
  var P = null;                               // pixel geometry of the last draw
  var drag = null;                            // index of the handle being dragged
  var dragX = 0;                              // its centre x while dragging
  var grab = 0, moved = false;                // pointer-to-centre offset; did it move?

  // ---- the model --------------------------------------------------------------
  function h0() { return S.fixed ? 14 : 1.5 * S.speed; }
  function gain(r) { return S.fixed ? 1 : 1 + S.speed * (1 + 3 * (1 - r)); }
  function events() {
    var ev = [];
    rev.forEach(function (t) { if (t != null) ev.push([t, 1]); });
    if (S.implicit) IMPLICIT_DAYS.forEach(function (d) { ev.push([d, 0.3]); });
    return ev.sort(function (a, b) { return a[0] - b[0]; });
  }
  // Walk the timeline in 0.2-day steps, stopping exactly on each event.
  function simulate() {
    var ev = events(), h = h0(), r0 = 1, tl = 0, curve = [[0, 1]], jumps = [];
    var reps = 1, nrev = 0, minr = 1, k = 0, t = 0;
    while (t < DAYS) {
      var tn = Math.min(t + 0.2, DAYS);
      if (k < ev.length && ev[k][0] <= tn) tn = ev[k][0];
      var rc = r0 * Math.pow(2, -(tn - tl) / h);
      curve.push([tn, rc]);
      minr = Math.min(minr, rc);
      while (k < ev.length && ev[k][0] <= tn + 1e-4) {
        var w = ev[k][1];
        h = h * (1 + w * (gain(rc) - 1));
        var rn = rc + w * (1 - rc);
        jumps.push({ t: tn, from: rc, to: rn, w: w });
        r0 = rn; tl = tn; rc = rn; reps += w;
        if (w === 1) nrev++;
        curve.push([tn, rn]);
        k++;
      }
      t = tn;
    }
    return { curve: curve, jumps: jumps, reps: reps, nrev: nrev, minr: minr, h: h };
  }
  // Place reviews where recall first falls to the target; the rest park.
  function autoschedule() {
    for (var i = 0; i < N_REV; i++) rev[i] = null;
    var imp = S.implicit ? IMPLICIT_DAYS : [], h = h0(), r0 = 1, tl = 0, t = 0, placed = 0, j = 0;
    while (t < DAYS && placed < N_REV) {
      t += 0.05;
      var rc = r0 * Math.pow(2, -(t - tl) / h);
      if (j < imp.length && imp[j] <= t) {
        h = h * (1 + 0.3 * (gain(rc) - 1));
        r0 = rc + 0.3 * (1 - rc); tl = t; j++; rc = r0;
      }
      if (rc <= S.target && t < DAYS) {
        rev[placed++] = t;
        h = h * gain(rc); r0 = 1; tl = t;
      }
    }
    render();
  }

  // ---- helpers ----------------------------------------------------------------
  function el(tag, attrs, text, parent) {
    var e = document.createElementNS(NS, tag);
    for (var a in attrs) e.setAttribute(a, attrs[a]);
    if (text != null) e.textContent = text;
    (parent || svg).appendChild(e);
    return e;
  }
  function fmt1(v) { return (Math.round(v * 10) / 10).toFixed(1).replace(/\.0$/, ""); }
  function X(t) { return P.L + t / DAYS * (P.R - P.L); }
  function Y(v) { return P.B - v * (P.B - P.T); }
  function T(x) { return (x - P.L) / (P.R - P.L) * DAYS; }
  function slotX(i) { return P.lane0 + P.slot * i; }
  // Where handle i is drawn: under its day, in its lane slot, or at the pointer.
  function handleX(i) {
    if (i === drag) return Math.min(Math.max(dragX, P.L + 3), P.lane1);
    return rev[i] == null ? slotX(i) : X(rev[i]);
  }

  // ---- drawing ----------------------------------------------------------------
  function geometry() {
    var W = svg.parentNode.getBoundingClientRect().width || 800, narrow = W < 560;
    var H = narrow ? Math.max(320, Math.round(W * 0.95)) : Math.round(W / 1.6);
    // Full screen (fig-expand.js): the stage is wide and short, so keep the
    // plot inside its height rather than scroll.
    if (svg.closest(".is-max")) H = Math.max(320, Math.min(H, Math.floor(svg.parentNode.clientHeight) - 4));
    var L = narrow ? 40 : Math.round(W * 0.09), R = Math.round(W * (narrow ? 0.8 : 0.84));
    var lane0 = R + (narrow ? 22 : 32), lane1 = W - (narrow ? 12 : 16);
    return {
      W: W, H: H, narrow: narrow, L: L, R: R, T: narrow ? 50 : 34, B: H - 66,
      track: H - 18, lane0: lane0, lane1: lane1, slot: Math.min(15, (lane1 - lane0) / (N_REV - 1)),
    };
  }

  function render() {
    if (!P) return;
    // Read focus BEFORE clearing: removing the focused handle moves focus to <body>.
    var active = document.activeElement, focused = active && svg.contains(active) ? active.getAttribute("data-rev") : null;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    svg.setAttribute("viewBox", "0 0 " + P.W + " " + P.H);
    svg.setAttribute("width", P.W); svg.setAttribute("height", P.H);
    var c = { ink: AISC.css("--ink"), mut: AISC.css("--ink-3"), rule: AISC.css("--rule"), panel: AISC.css("--paper-2"),
      red: AISC.css("--red"), amber: AISC.css("--amber"), curve: AISC.css("--green") };
    var mono = "IBM Plex Mono, monospace";
    function label(x, y, text, o) {
      o = o || {};
      return el("text", { x: x, y: y, "font-family": mono, "font-size": o.size || 11, fill: o.fill || c.mut,
        "text-anchor": o.anchor || "start" }, text);
    }
    var sim = simulate();

    // frame: panel (page paper, so the plot reads as part of the page), the below-threshold zone, threshold, axes
    el("rect", { x: P.L, y: P.T, width: P.R - P.L, height: P.B - P.T, fill: c.panel });
    el("rect", { x: P.L, y: Y(S.target), width: P.R - P.L, height: P.B - Y(S.target), fill: c.red, "fill-opacity": 0.07 });
    el("line", { x1: P.L, x2: P.R, y1: Y(S.target), y2: Y(S.target), stroke: c.red, "stroke-width": 1, "stroke-dasharray": "5 4" });
    el("path", { d: "M" + P.L + " " + P.T + "V" + P.B + "H" + P.R, stroke: c.ink, "stroke-width": 1.2, fill: "none" });
    label(P.L - 7, P.T + 4, "100%", { anchor: "end" });
    label(P.L - 7, P.B + 3, "0%", { anchor: "end" });
    if (P.narrow) {
      el("text", { x: 12, y: (P.T + P.B) / 2, transform: "rotate(-90 12 " + (P.T + P.B) / 2 + ")", "font-family": mono,
        "font-size": 11, fill: c.mut, "text-anchor": "middle" }, "recall");
    } else {
      label(P.L - 8, (P.T + P.B) / 2 + 4, "recall", { size: 12, anchor: "end" });
    }
    var step = P.narrow ? 20 : 10;
    for (var d = 0; d <= DAYS; d += step) {
      el("line", { x1: X(d), x2: X(d), y1: P.B, y2: P.B + 6, stroke: c.ink, "stroke-width": 1 });
      label(X(d), P.B + 19, d + "d", { anchor: "middle" });
    }

    // header: where the curve starts, and which model is on
    var modeText = S.fixed ? "PRODUCTION TODAY · fixed 14-day half-life" : "PLANNED · expanding intervals";
    label(P.narrow ? P.W - 12 : P.R, P.narrow ? 16 : P.T - 12, modeText, { fill: S.fixed ? c.mut : c.curve, anchor: "end" });
    label(P.L + 6, P.T - 12, "initial lesson");

    // reviews: amber bars, an arrow on full reviews (the Math Academy glyph).
    // Half-width 7px on a desktop plot, down to 4px on a phone's.
    var bw = Math.max(4, Math.min(7, (P.R - P.L) / 90));
    sim.jumps.forEach(function (j) {
      var full = j.w === 1, wd = full ? bw : bw / 2, x = X(j.t), top = Y(j.to);
      el("rect", { x: x - wd, y: top, width: 2 * wd, height: Y(j.from) - top, fill: c.amber, "fill-opacity": full ? 0.9 : 0.55 });
      if (full) el("path", { d: "M" + (x - bw) + " " + (top + 2 * bw - 1) + "H" + (x + bw) + "L" + x + " " + (top + 2) + "Z", fill: c.ink, "fill-opacity": 0.8 });
    });
    el("polyline", { points: sim.curve.map(function (p) { return X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1); }).join(" "),
      fill: "none", stroke: c.curve, "stroke-width": 2.6, "stroke-linejoin": "round", "stroke-linecap": "round" });
    // Threshold label: under the line, where bars never reach, and drawn over
    // the curve with a panel-coloured halo so a decay through it stays legible.
    label(P.R - 4, Y(S.target) + 15, "review threshold " + Math.round(S.target * 100) + "%", { fill: c.red, anchor: "end" })
      .setAttribute("style", "paint-order:stroke;stroke:" + c.panel + ";stroke-width:4px;stroke-linejoin:round");

    // handle track + the unused lane
    el("line", { x1: P.L, x2: P.R, y1: P.track, y2: P.track, stroke: c.rule, "stroke-width": 5, "stroke-linecap": "round" });
    el("rect", { x: P.lane0 - 10, y: P.track - 12, width: P.lane1 - P.lane0 + 20, height: 24, rx: 3, fill: c.rule, "fill-opacity": 0.6 });
    label((P.lane0 + P.lane1) / 2, P.track - 19, "unused", { size: 10, anchor: "middle" });
    label(P.L, P.track - 16, "drag reviews ↓", { size: 10 });

    for (var i = 0; i < N_REV; i++) {
      var h = el("circle", { cx: handleX(i), cy: P.track, r: 7.5, fill: HANDLE, stroke: "#161a23", "stroke-width": 1.5,
        tabindex: 0, role: "slider", "aria-valuemin": 0, "aria-valuemax": DAYS,
        "aria-valuenow": rev[i] == null ? DAYS : Math.round(rev[i]),
        "aria-valuetext": rev[i] == null ? "unused" : "day " + fmt1(rev[i]),
        "aria-label": "Review " + (i + 1), class: "aisc-forget-handle", "data-rev": i });
      if (focused === String(i)) h.focus({ preventScroll: true });
    }
    readout(sim);
  }

  function readout(sim) {
    document.getElementById("aisc-forget-readout").innerHTML =
      "reviews in 60 days <b>" + sim.nrev + "</b> · repetitions <b>" + fmt1(sim.reps) + "</b> · " +
      "lowest recall <b>" + Math.round(sim.minr * 100) + "%</b> · half-life at day 60 <b>" + fmt1(sim.h) + " d</b>";
  }

  // ---- dragging + keys --------------------------------------------------------
  function pointerX(evt) { return evt.clientX - svg.getBoundingClientRect().left; }
  // Drop a handle where the pointer is: on the plot it becomes a day, past the
  // plot's right edge it parks.
  function place(i, x) {
    if (x > P.R + 6) rev[i] = null;
    else rev[i] = Math.min(DAYS, Math.max(T(P.L + 3), T(x)));
  }
  svg.addEventListener("pointerdown", function (evt) {
    var t = evt.target;
    if (!t.dataset || t.dataset.rev == null) return;
    evt.preventDefault();
    var i = +t.dataset.rev;
    dragX = handleX(i); drag = i;           // handleX first: it reads dragX while dragging
    // Keep the grab offset, so a click that never moves changes nothing.
    grab = pointerX(evt) - dragX; moved = false;
    svg.setPointerCapture(evt.pointerId);   // the svg survives every redraw; handles don't
    t.focus({ preventScroll: true });
    render();
  });
  svg.addEventListener("pointermove", function (evt) {
    if (drag == null) return;
    dragX = pointerX(evt) - grab; moved = true;
    place(drag, dragX);
    render();
  });
  function endDrag() {
    if (drag == null) return;
    if (moved) place(drag, dragX);
    drag = null; render();
  }
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);
  svg.addEventListener("keydown", function (evt) {
    var t = evt.target;
    if (!t.dataset || t.dataset.rev == null) return;
    var i = +t.dataset.rev, by = evt.shiftKey ? 5 : 1, cur = rev[i];
    if (evt.key === "ArrowRight" || evt.key === "ArrowUp") rev[i] = cur == null ? null : cur + by > DAYS ? null : cur + by;
    else if (evt.key === "ArrowLeft" || evt.key === "ArrowDown") rev[i] = Math.max(T(P.L + 3), (cur == null ? DAYS + by : cur) - by);
    else if (evt.key === "End") rev[i] = null;
    else if (evt.key === "Home") rev[i] = T(P.L + 3);
    else return;
    evt.preventDefault();
    render();
  });

  // ---- controls ---------------------------------------------------------------
  function seg(id, fn) {
    document.querySelectorAll("#" + id + " button").forEach(function (b) {
      b.addEventListener("click", function () {
        document.querySelectorAll("#" + id + " button").forEach(function (x) {
          x.classList.toggle("on", x === b); x.setAttribute("aria-pressed", x === b ? "true" : "false");
        });
        fn(b);
      });
    });
  }
  // Speed, retention target and implicit reviews stay at their defaults
  // (S): the figure keeps the built-vs-planned switch and the scheduler
  // (Seth, 2026-09-24: "significantly simplified").
  seg("aisc-model-seg", function (b) { S.fixed = b.dataset.m === "today"; autoschedule(); });
  document.getElementById("aisc-forget-auto").addEventListener("click", autoschedule);

  function layout() { P = geometry(); render(); }
  AISC.whenVisible(svg, function () { P = geometry(); autoschedule(); });
  var rt; window.addEventListener("resize", function () { if (!P) return; clearTimeout(rt); rt = setTimeout(layout, 120); });
  AISC.onTheme(function () { if (P) render(); });
})();
