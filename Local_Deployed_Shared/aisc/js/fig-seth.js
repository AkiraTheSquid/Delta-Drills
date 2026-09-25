/* Fig. 7 — one learner's September, from data/seth_progress.json
   (built by scripts/build_seth_progress.py from the production log).
   Three panels on one day axis; plain SVG, redrawn on resize/theme. */
(function () {
  "use strict";
  var svg = document.getElementById("aisc-seth-svg");
  if (!svg) return;
  var NS = "http://www.w3.org/2000/svg";
  var data = null, section = "0.0", hover = null;

  function el(tag, attrs, text) {
    var e = document.createElementNS(NS, tag);
    Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text != null) e.textContent = text;
    svg.appendChild(e);
    return e;
  }
  function path(pts) {
    var d = "", pen = false;
    pts.forEach(function (p) {
      if (p == null) { pen = false; return; }
      d += (pen ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1);
      pen = true;
    });
    return d;
  }
  function fmtDay(iso) { var d = new Date(iso + "T12:00:00"); return d.toLocaleDateString("en-US", { month: "short", day: "numeric" }); }

  function draw() {
    if (!data) return;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var W = svg.parentNode.getBoundingClientRect().width || 800;
    var L = 44, R = 14, gap = 34;
    var H1 = 190, H2 = 110, H3 = 110, T = 22;
    var y1 = T, y2 = y1 + H1 + gap, y3 = y2 + H2 + gap, H = y3 + H3 + 30;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("width", W); svg.setAttribute("height", H);
    var S = data.series, n = S.length, iw = W - L - R;
    var X = function (i) { return L + (i + 0.5) * iw / n; };
    var c = { ink: AISC.css("--ink"), mut: AISC.css("--ink-3"), rule: AISC.css("--rule"), green: AISC.css("--green"),
      soft: AISC.css("--green-soft"), coral: AISC.css("--coral"), slate: AISC.css("--slate"), amber: AISC.css("--amber"), olive: AISC.css("--olive") };
    var mono = "IBM Plex Mono";

    function frame(y0, h, title, ticks, fmt) {
      el("text", { x: L, y: y0 - 8, "font-family": "Instrument Serif", "font-size": 17, fill: c.ink }, title);
      ticks.forEach(function (t) {
        var y = y0 + h - t.v * h;
        el("line", { x1: L, x2: W - R, y1: y, y2: y, stroke: c.rule, "stroke-width": 1 });
        el("text", { x: L - 6, y: y + 4, "text-anchor": "end", "font-family": mono, "font-size": 10, fill: c.mut }, fmt(t.label));
      });
    }

    // ---- panel 1: competency --------------------------------------------------
    frame(y1, H1, "Competency · " + data.sections[section].label + " (" + data.sections[section].n + " concepts)",
      [0, 25, 50, 75, 100].map(function (v) { return { v: v / 100, label: v }; }), String);
    var P1 = function (v) { return y1 + H1 - v / 100 * H1; };
    var tried = S.map(function (d, i) { return [X(i), P1(d.tried[section])]; });
    var demo = S.map(function (d, i) { return [X(i), P1(d.demo[section])]; });
    el("path", { d: path(tried) + "L" + X(n - 1) + " " + P1(0) + "L" + X(0) + " " + P1(0) + "Z", fill: c.soft, opacity: 0.55 });
    el("path", { d: path(demo) + "L" + X(n - 1) + " " + P1(0) + "L" + X(0) + " " + P1(0) + "Z", fill: c.green, opacity: 0.35 });
    el("path", { d: path(demo), fill: "none", stroke: c.green, "stroke-width": 2.2 });
    var app = S.map(function (d, i) { return d.app[section] == null ? null : [X(i), P1(d.app[section])]; });
    el("path", { d: path(app), fill: "none", stroke: c.mut, "stroke-width": 2, "stroke-dasharray": "5 4" });
    var last = S[n - 1];
    if (last.app[section] != null) el("text", { x: X(n - 1) - 4, y: P1(last.app[section]) + (last.app[section] > last.tried[section] - 8 && last.app[section] <= last.tried[section] + 8 ? 15 : -7), "text-anchor": "end", "font-family": mono, "font-size": 10, fill: c.mut }, "app " + Math.round(last.app[section]));
    el("text", { x: X(n - 1) - 4, y: P1(last.tried[section]) - 6, "text-anchor": "end", "font-family": mono, "font-size": 10, fill: c.green }, "tried " + Math.round(last.tried[section]) + "%");

    // ---- panel 2: predicted vs actual -----------------------------------------
    frame(y2, H2, "Right answers, trailing 7 days: actual vs. model's prediction",
      [0.5, 0.75, 1].map(function (v) { return { v: (v - 0.5) / 0.5, label: v }; }), function (v) { return Math.round(v * 100) + "%"; });
    var P2 = function (v) { return y2 + H2 - (v - 0.5) / 0.5 * H2; };
    var act = S.map(function (d, i) { return d.cal ? [X(i), P2(d.cal.actual)] : null; });
    var pre = S.map(function (d, i) { return d.cal ? [X(i), P2(d.cal.predicted)] : null; });
    var band = [], bandB = [];
    S.forEach(function (d, i) { if (d.cal) { band.push([X(i), P2(d.cal.actual)]); bandB.unshift([X(i), P2(d.cal.predicted)]); } });
    if (band.length) el("path", { d: path(band.concat(bandB)) + "Z", fill: c.coral, opacity: 0.12 });
    el("path", { d: path(pre), fill: "none", stroke: c.slate, "stroke-width": 2 });
    el("path", { d: path(act), fill: "none", stroke: c.coral, "stroke-width": 2.4 });

    // ---- panel 3: hours/day ---------------------------------------------------
    var maxMin = Math.max.apply(null, S.map(function (d) { return (d.min.drill || 0) + (d.min.lesson || 0) + (d.min.placement || 0); }));
    var topH = Math.max(1, Math.ceil(maxMin / 60));
    frame(y3, H3, "Study time per day (estimated)",
      [0, topH / 2, topH].map(function (v) { return { v: v / topH, label: v }; }), function (v) { return v + " h"; });
    var bw = Math.max(3, iw / n * 0.62);
    S.forEach(function (d, i) {
      var y = y3 + H3;
      [["drill", c.coral], ["lesson", c.amber], ["placement", c.slate]].forEach(function (k) {
        var m = d.min[k[0]] || 0; if (!m) return;
        var h = m / 60 / topH * H3;
        el("rect", { x: X(i) - bw / 2, y: y - h, width: bw, height: h, fill: k[1], opacity: 0.85 });
        y -= h;
      });
    });
    [["drills", c.coral], ["lessons", c.amber], ["placement", c.slate]].forEach(function (k, j) {
      el("rect", { x: W - R - 230 + j * 78, y: y3 - 19, width: 9, height: 9, fill: k[1] });
      el("text", { x: W - R - 217 + j * 78, y: y3 - 11, "font-family": mono, "font-size": 10, fill: c.mut }, k[0]);
    });

    // ---- day axis + hover -----------------------------------------------------
    var every = W < 560 ? 7 : 3;
    S.forEach(function (d, i) {
      if (i % every) return;
      el("text", { x: X(i), y: H - 10, "text-anchor": "middle", "font-family": mono, "font-size": 10, fill: c.mut }, fmtDay(d.date));
    });
    if (hover != null) {
      el("line", { x1: X(hover), x2: X(hover), y1: y1, y2: y3 + H3, stroke: c.ink, "stroke-width": 1, "stroke-dasharray": "2 3" });
    }
    var hit = el("rect", { x: L, y: 0, width: iw, height: H, fill: "transparent" });
    function pick(evt) {
      var r = svg.getBoundingClientRect(), x = (evt.touches ? evt.touches[0].clientX : evt.clientX) - r.left;
      var i = Math.max(0, Math.min(n - 1, Math.floor((x - L) / iw * n)));
      if (i !== hover) { hover = i; draw(); readout(); }
    }
    hit.addEventListener("mousemove", pick);
    hit.addEventListener("touchstart", pick, { passive: true });
  }

  function readout() {
    var d = data.series[hover == null ? data.series.length - 1 : hover];
    var m = d.min, tot = (m.drill || 0) + (m.lesson || 0) + (m.placement || 0);
    document.getElementById("aisc-seth-readout").innerHTML = "<b>" + fmtDay(d.date) + "</b> · " +
      "app score <b>" + (d.app[section] == null ? "—" : Math.round(d.app[section])) + "</b> · " +
      "demonstrated <b>" + Math.round(d.demo[section]) + "%</b> · tried <b>" + Math.round(d.tried[section]) + "%</b><br>" +
      (d.cal ? "7-day right <b>" + Math.round(d.cal.actual * 100) + "%</b> vs predicted <b>" + Math.round(d.cal.predicted * 100) + "%</b> (n=" + d.cal.n + ") · " : "") +
      "study ≈ <b>" + (tot / 60).toFixed(1) + " h</b>";
  }

  var ORDER = ["0.0", "prep-arrays", "0.1", "aggregate"];
  fetch("aisc/data/seth_progress.json").then(function (r) { return r.json(); }).then(function (j) {
    data = j;
    var sm = j.summary;
    document.getElementById("aisc-seth-gap").textContent = "across all " + sm.attempts + " September drills, " +
      Math.round(sm.actual * 100) + "% right against " + Math.round(sm.predicted * 100) + "% predicted; " +
      sm.hours + " h over " + sm.study_days + " study days";
    var at = sm.all_time, tot = document.getElementById("aisc-seth-total");
    if (at && tot) tot.innerHTML = "<b>" + at.hours + " h</b> studied in the app since " + fmtDay(at.since) +
      " (" + at.study_days + " study days): " + at.by_kind_hours.drill + " h drills, " + at.by_kind_hours.lesson + " h lessons, " +
      at.by_kind_hours.placement + " h timed placement; " + at.drills_answered + " drills answered, " + at.placement_problems + " placement problems.";
    var box = document.getElementById("aisc-seth-sections");
    ORDER.forEach(function (k) {
      if (!j.sections[k]) return;
      var b = document.createElement("button");
      b.className = "btn ghost" + (k === section ? " on" : "");
      b.textContent = j.sections[k].label;
      b.addEventListener("click", function () {
        section = k;
        box.querySelectorAll("button").forEach(function (x) { x.classList.toggle("on", x === b); });
        draw(); readout();
      });
      box.appendChild(b);
    });
    AISC.whenVisible(svg, function () { draw(); readout(); });
    var t; window.addEventListener("resize", function () { clearTimeout(t); t = setTimeout(draw, 120); });
    AISC.onTheme(draw);
  });
})();
