/* Fig. 1 — one chart: AI research effort vs. shared human understanding,
   and what Delta Drills moves. Plain SVG, redrawn on resize/theme; the green
   line tweens when a slider moves.

   Four sliders: AGI year, deploy date, learner effect, expert effect.
   Units: human-researcher equivalents.
     AI capabilities research   cap(t) = base(AGI) · 25^(t − AGI)
                                (AGI year := AI research overtakes human understanding)
     AI safety research         saf(t) = 0.1 · cap(t) · U(t) / base(t)
                                (humans directing AI agents: scales with understanding)
     Shared human understanding U(t) = people(t) · expert(t)
       people: monthly stock-and-flow of safety researchers
       expert: from the deploy date, an AI-generated tutor lets the SAME people
               keep up: × expert effect, phased in over a year.
   Learners (from the deploy date): ramp ÷ learner effect; freed program
   budget reopens seats; remote learners join. */
(function () {
  "use strict";
  var svg = document.getElementById("aisc-effort-svg");
  if (!svg) return;
  var NS = "http://www.w3.org/2000/svg";
  var X0 = 2024, X1 = 2036, BURN = 2012, DT = 1 / 12, NOW = 2026 + 8.7 / 12;
  var BASE = { E0: 1000, g: 0.15, T0: 12, fin: 0.4, exit: 0.15, P0: 250 };
  var A = { learn: 1.9, expert: 1.3, reuse: 0.5, remote: 0.2 };   // sliders + Fermi box
  var scale = "lin", deploy = 2028, agi = 2030, cur = null, base = null, ymaxFixed = 1, tween = null;

  // ---- the model ------------------------------------------------------------
  function ramp1() { return BASE.T0 / A.learn; }
  function seatMult() { return 1 + A.reuse * (A.learn - 1); }
  function understanding(D) {              // D = null → no tool, ever
    var n = Math.round((X1 - BURN) / DT) + 1, pipe = new Float64Array(n + 40), out = [], P = BASE.P0;
    var T1 = ramp1(), k = seatMult() * (1 + A.remote);
    function put(m, months, v) {           // fractional delay, split across two months
      var f = Math.floor(months), w = months - f;
      pipe[m + f] += v * (1 - w); pipe[m + f + 1] += v * w;
    }
    for (var m = 0; m < n; m++) {
      var t = BURN + m * DT, e = BASE.E0 * Math.pow(1 + BASE.g, t - 2024) * DT, on = D != null && t >= D - 1e-9;
      if (on) put(m, T1, e * k * BASE.fin); else put(m, BASE.T0, e * BASE.fin);
      P += pipe[m] - P * BASE.exit * DT;
      var ex = on ? 1 + (A.expert - 1) * Math.min(1, t - D) : 1;
      if (t >= X0 - 1e-9) out.push(P * ex);
    }
    return out;                             // monthly from X0
  }
  function at(arr, t) {
    var i = (t - X0) / DT, a = Math.max(0, Math.min(arr.length - 1, Math.floor(i))), b = Math.min(arr.length - 1, a + 1);
    return arr[a] + (arr[b] - arr[a]) * Math.max(0, Math.min(1, i - a));
  }
  function gainBefore(arr, until) {        // researcher-years above today's line, now → until
    var s = 0; arr.forEach(function (v, i) { var t = X0 + i * DT; if (t >= NOW - 1e-9 && t < until - 1e-9) s += (v - base[i]) * DT; }); return s;
  }
  var K25 = Math.log(25);
  function cap(t) { return at(base, agi) * Math.exp(K25 * (t - agi)); }
  function saf(t, arr) { return 0.1 * cap(t) * at(arr, t) / at(base, t); }

  // ---- helpers ----------------------------------------------------------------
  function el(tag, attrs, text, parent) {
    var e = document.createElementNS(NS, tag);
    for (var a in attrs) e.setAttribute(a, attrs[a]);
    if (text != null) e.textContent = text;
    (parent || svg).appendChild(e);
    return e;
  }
  var MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  function fmtAgi(t) { var d = fmtDate(t); return d.slice(0, 3) === "Jan" ? d.slice(4) : d; }
  function fmtDate(t) { var y = Math.floor(t + 1e-6), m = Math.round((t - y) * 12); if (m === 12) { y++; m = 0; } return MON[m] + " " + y; }
  function kfmt(v) { return v >= 10000 ? Math.round(v / 1000) + "k" : v >= 1000 ? (v / 1000).toFixed(1).replace(/\.0$/, "") + "k" : String(Math.round(v)); }
  function num(v) { return Math.round(v).toLocaleString("en-US"); }

  // ---- drawing ----------------------------------------------------------------
  var G = {};                               // geometry of the last draw (for dragging)
  function draw() {
    if (!cur) return;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var W = svg.parentNode.getBoundingClientRect().width || 800, narrow = W < 560;
    var H = narrow ? 380 : 460, L = narrow ? 40 : 58, R = narrow ? 10 : 18, T = 18, B = 40;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.setAttribute("width", W); svg.setAttribute("height", H);
    var c = { ink2: AISC.css("--ink-2"), mut: AISC.css("--ink-3"), rule: AISC.css("--rule"), card: AISC.css("--card"),
      coral: AISC.css("--coral"), slate: AISC.css("--slate"), olive: AISC.css("--olive"), green: AISC.css("--green") };
    var serif = "Instrument Serif", mono = "IBM Plex Mono", fs = narrow ? 12.5 : 17, fm = narrow ? 9.5 : 11.5;
    var X = function (t) { return L + (t - X0) / (X1 - X0) * (W - L - R); };
    var top = T, bot = H - B, lin = scale === "lin";
    var ymax = ymaxFixed, LMIN = 1, LMAX = 7;
    var Y = lin ? function (v) { return bot - v / ymax * (bot - top); }
      : function (v) { return bot - (Math.log10(Math.max(v, 1e-9)) - LMIN) / (LMAX - LMIN) * (bot - top); };
    G = { L: L, R: R, W: W };

    // axes + grid
    var ticks = [];
    if (lin) { for (var v = 0; v <= ymax + 1e-9; v += ymax / 4) ticks.push([v, kfmt(v)]); }
    else for (var e = LMIN; e <= LMAX; e++) ticks.push([Math.pow(10, e), e < 4 ? String(Math.pow(10, e)) : "10" + "⁰¹²³⁴⁵⁶⁷⁸⁹"[e]]);
    ticks.forEach(function (tk) {
      var y = Y(tk[0]);
      el("line", { x1: L, x2: W - R, y1: y, y2: y, stroke: c.rule, "stroke-width": 1 });
      el("text", { x: L - 7, y: y + 4, "text-anchor": "end", "font-family": mono, "font-size": 10, fill: c.mut }, tk[1]);
    });
    el("line", { x1: L, x2: W - R, y1: bot, y2: bot, stroke: c.ink2, "stroke-width": 1.2 });
    el("line", { x1: L, x2: L, y1: top - 6, y2: bot, stroke: c.ink2, "stroke-width": 1.2 });
    for (var yr = X0; yr <= X1; yr += narrow ? 4 : 2) el("text", { x: X(yr), y: bot + 18, "text-anchor": "middle", "font-family": mono, "font-size": 10, fill: c.mut }, String(yr));
    el("text", { x: L + 6, y: top + 2, "font-family": mono, "font-size": 10, fill: c.mut }, "human-researcher equivalents" + (lin ? "" : " · log"));
    el("line", { x1: X(NOW), x2: X(NOW), y1: bot, y2: bot - 8, stroke: c.ink2, "stroke-width": 1.2 });
    el("text", { x: X(NOW), y: bot + 32, "text-anchor": "middle", "font-family": mono, "font-size": 9.5, fill: c.mut }, "now");
    // AGI marker: where AI research overtakes human understanding
    el("line", { x1: X(agi), x2: X(agi), y1: top + 4, y2: bot + 22, stroke: c.coral, "stroke-width": 1, "stroke-dasharray": "2 4" });
    el("text", { x: X(agi), y: bot + 32, "text-anchor": "middle", "font-family": mono, "font-size": 9.5, "font-weight": 600, fill: c.coral }, "AGI " + fmtAgi(agi));

    var clip = el("clipPath", { id: "effort-clip" }, null, el("defs", {}));
    el("rect", { x: L, y: top - 4, width: W - L - R, height: bot - top + 4 }, null, clip);
    var g = el("g", { "clip-path": "url(#effort-clip)" });

    function pathOf(fn, t0, t1, dt) {
      var d = "", pen = false;
      for (var t = t0; t <= t1 + 1e-9; t += dt) {
        var y = Y(fn(t));
        if (y > bot + 40) { pen = false; continue; }
        if (y < top - 40) { if (pen) d += "L" + X(t).toFixed(1) + " " + (top - 40); break; }
        d += (pen ? "L" : "M") + X(t).toFixed(1) + " " + y.toFixed(1); pen = true;
      }
      return d;
    }
    function crossY(fn, y) { for (var t = X0; t <= X1; t += 1 / 96) if (Y(fn(t)) <= y) return t; return X1; }
    var safCur = function (t) { return saf(t, cur); };

    // gain area between the two understanding lines
    var dFrom = Math.max(X0, deploy), area = "", t;
    for (t = dFrom; t <= X1 + 1e-9; t += DT) area += (area ? "L" : "M") + X(t).toFixed(1) + " " + Y(at(cur, t)).toFixed(1);
    for (t = X1; t >= dFrom - 1e-9; t -= DT) area += "L" + X(t).toFixed(1) + " " + Y(at(base, t)).toFixed(1);
    el("path", { d: area + "Z", fill: c.green, opacity: 0.14 }, null, g);

    el("path", { d: pathOf(function (t) { return at(base, t); }, X0, X1, DT), fill: "none", stroke: c.olive, "stroke-width": 2.4 }, null, g);
    el("path", { d: pathOf(function (t) { return at(cur, t); }, deploy, X1, DT), fill: "none", stroke: c.green, "stroke-width": 3 }, null, g);
    el("path", { d: pathOf(safCur, X0, X1, 1 / 96), fill: "none", stroke: c.slate, "stroke-width": 2.2, "stroke-dasharray": "7 5" }, null, g);
    el("path", { d: pathOf(cap, X0, X1, 1 / 96), fill: "none", stroke: c.coral, "stroke-width": 2.6 }, null, g);

    // direct labels, reference-chart style: name in serif, rate in bold mono, optional leader
    function label(x, y, anchor, color, name, rate, lead) {
      if (lead) el("path", { d: "M" + lead[0] + " " + lead[1] + " Q " + lead[2] + " " + lead[3] + " " + lead[4] + " " + lead[5], fill: "none", stroke: color, "stroke-width": 1, opacity: 0.7 });
      var grp = el("g", {});
      name.forEach(function (s, i) { el("text", { x: x, y: y + i * (fs + 1), "text-anchor": anchor, "font-family": serif, "font-size": fs, fill: color }, s, grp); });
      if (rate) el("text", { x: x, y: y + name.length * (fs + 1) + 2, "text-anchor": anchor, "font-family": mono, "font-size": fm, "font-weight": 600, fill: color }, rate, grp);
      var bb = grp.getBBox(), over = bb.x + bb.width - (W - R - 2);   // keep inside the plot
      if (over > 0) grp.setAttribute("transform", "translate(" + (-over) + ",0)");
    }
    // AI labels sit beside the point where each curve crosses a fixed height.
    var yC = top + (lin ? 26 : 40), tc = crossY(cap, yC + fs + 6);   // clear the curve at the block's lowest line
    label(X(tc) - 10, yC, "end", c.coral, [narrow ? "AI capabilities" : "AI capabilities research"], narrow ? "~25×/yr" : "~25× / year");
    var yS = top + (narrow ? 84 : lin ? 96 : 80), ts = crossY(safCur, yS - fs), flip = lin && X(ts) > W - (narrow ? 150 : 300);
    if (flip) ts = crossY(safCur, yS + 2 * fs + 8);          // clear the curve at the block's lowest line
    var xs = X(ts) + (flip ? -1 : 1) * (narrow ? 12 : 22);
    label(xs, yS, flip ? "end" : "start", c.slate, narrow ? ["AI safety research", "(humans + AI agents)"] : ["AI safety research:", "humans directing AI agents"],
      narrow ? "~25×/yr" : lin ? "~25× / year, grows with understanding" : "~25× / year", flip ? null : [xs - 4, yS - 5, xs - 12, yS - 5, X(crossY(safCur, yS + 12)) + 3, yS + 12]);
    var xe = X(X1) - 4, vc = at(cur, X1), vb = at(base, X1), yCur = Y(Math.min(vc, ymax));
    var gBot = Y(at(base, X1)) - 14, gTop = gBot - 3 * (fs + 1);
    var inside = gTop > Y(Math.min(at(cur, X1 - (narrow ? 4 : 3.2)), ymax)) + 6 && lin;
    label(xe, inside ? gTop + fs : Math.max(top + 30, yCur - (narrow ? 40 : 50)), "end", c.green, narrow ? ["Understanding,", "with Delta Drills"] : ["Shared human understanding,", "with Delta Drills"], "+" + Math.round((vc / vb - 1) * 100) + "% by " + X1);
    var yUnder = Y(at(base, X1 - (narrow ? 5 : 3.4))) + fs + 8;   // below the line across the label's width
    label(xe, Math.min(bot - (narrow ? 26 : 34), Math.max(yUnder, yCur + 40)), "end", c.olive, [narrow ? "Understanding, today" : "Shared human understanding, today"], narrow ? null : "who understand the systems");

    // deploy marker (draggable anywhere on the plot)
    var xd = X(deploy), right = xd > W * 0.62;
    el("line", { x1: xd, x2: xd, y1: top + 4, y2: bot, stroke: c.green, "stroke-width": 1.2, "stroke-dasharray": "3 4" });
    el("circle", { cx: xd, cy: bot, r: 7, fill: c.green, stroke: c.card, "stroke-width": 2 });
    el("text", { x: xd + (right ? -10 : 10), y: bot - 12, "text-anchor": right ? "end" : "start", "font-family": mono, "font-size": 10.5, "font-weight": 600, fill: c.green },
      narrow ? fmtDate(deploy) : (right ? "" : "◂ ") + "Delta Drills everywhere · " + fmtDate(deploy) + (right ? " ▸" : ""));
    var hit = el("rect", { x: L, y: top, width: W - L - R, height: bot - top + 20, fill: "transparent", style: "cursor:ew-resize;touch-action:none" });
    hit.addEventListener("pointerdown", startDrag);
  }

  // ---- interaction -------------------------------------------------------------
  function $(id) { return document.getElementById(id); }
  var sDeploy = $("aisc-s-deploy"), sAgi = $("aisc-s-agi"), sLearn = $("aisc-s-learn"), sExpert = $("aisc-s-expert");
  function update(animate) {               // recompute the green line; tween it if asked
    var next = understanding(deploy);
    readout(next);
    cancelAnimationFrame(tween);
    if (!animate || !cur) { cur = next; draw(); return; }
    var from = cur.slice(), t0 = performance.now();
    (function step(now) {
      var u = Math.min(1, (now - t0) / 380), e = 1 - Math.pow(1 - u, 3);
      cur = from.map(function (v, i) { return v + (next[i] - v) * e; });
      draw();
      if (u < 1) tween = requestAnimationFrame(step); else cur = next;
    })(t0);
  }
  function setDeploy(t, animate) {
    deploy = Math.max(+sDeploy.min, Math.min(+sDeploy.max, Math.round(t * 12) / 12));
    sDeploy.value = deploy; $("aisc-o-deploy").textContent = fmtDate(deploy);
    update(animate);
  }
  function tFromX(clientX) { var r = svg.getBoundingClientRect(); return X0 + (clientX - r.left - G.L) / (G.W - G.L - G.R) * (X1 - X0); }
  function startDrag(ev) {
    stopPlay(); ev.preventDefault();
    var move = function (e) { setDeploy(tFromX(e.clientX), false); };
    move(ev);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", function () { window.removeEventListener("pointermove", move); }, { once: true });
  }
  sDeploy.addEventListener("input", function () { stopPlay(); setDeploy(+sDeploy.value, true); });
  sAgi.addEventListener("input", function () { agi = +sAgi.value; $("aisc-o-agi").textContent = fmtAgi(agi); readout(); draw(); });
  sLearn.addEventListener("input", function () { A.learn = +sLearn.value; $("aisc-o-learn").textContent = A.learn.toFixed(2) + "×"; update(true); });
  sExpert.addEventListener("input", function () { A.expert = +sExpert.value; $("aisc-o-expert").textContent = A.expert.toFixed(2) + "×"; update(true); });

  // ▶ sweeps the deploy date from latest to earliest, so the green line grows.
  var playBtn = $("aisc-effort-play"), playing = null;
  function stopPlay() { if (playing) { clearInterval(playing); playing = null; playBtn.textContent = "▶ Play"; } }
  playBtn.addEventListener("click", function () {
    if (playing) return stopPlay();
    var t = +sDeploy.max; setDeploy(t, false); playBtn.textContent = "■ Stop";
    playing = setInterval(function () {
      t -= 1 / 12;
      if (t < +sDeploy.min - 1e-9) return stopPlay();
      setDeploy(t, false);
    }, 45);
  });

  function readout(arr) {
    arr = arr || cur;
    var y = fmtAgi(agi), now = at(arr, agi), was = at(base, agi);
    var gained = gainBefore(arr, agi), delay = gained - gainBefore(understanding(deploy + 1), agi);
    $("aisc-effort-mech").innerHTML =
      "<li><b>Learners, ×" + A.learn.toFixed(2) + ".</b> " + BASE.T0 + " → <b>" + ramp1().toFixed(1) + " mo</b> to useful; freed program budget reopens seats <b>×" + seatMult().toFixed(2) +
        "</b>; remote learners <b>+" + Math.round(A.remote * 100) + "%</b>. The pool grows.</li>" +
      "<li><b>Experts, ×" + A.expert.toFixed(2) + ".</b> An AI-generated tutor lets the same researchers keep up with superhuman systems, conceptually and technically. Phased in over a year.</li>";
    $("aisc-effort-readout").innerHTML = deploy >= agi
      ? "Delta Drills arrives after AGI (" + y + "):<br><b>no gain before it</b>. Move the date earlier."
      : "Understanding at AGI (" + y + ")<br>today <b>" + num(was) + "</b> · with Delta Drills <b>" + num(now) + "</b> (+" + Math.round((now / was - 1) * 100) + "%)<br>" +
        "Researcher-years gained before AGI: <b>+" + num(gained) + "</b><br>" +
        "Each year of delay costs <b>" + num(delay) + "</b> of them";
  }

  document.querySelectorAll("#aisc-fig-effort [data-scale]").forEach(function (b) {
    b.addEventListener("click", function () {
      scale = b.dataset.scale;
      document.querySelectorAll("#aisc-fig-effort [data-scale]").forEach(function (x) { x.classList.toggle("on", x === b); });
      draw();
    });
  });
  function rebuild() {
    base = understanding(null);
    update(false);
  }
  function fixAxis() {                     // once, at the defaults and the earliest date: sliders never rescale it
    var raw = Math.max.apply(null, understanding(+sDeploy.min)) * 1.08 / 4, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    ymaxFixed = 4 * mag * [1, 2, 2.5, 5, 10].filter(function (m) { return m * mag >= raw; })[0];
  }
  document.querySelectorAll("#aisc-effort-assume input").forEach(function (inp) {
    inp.addEventListener("input", function () {
      var v = parseFloat(inp.value); if (!isFinite(v) || v < 0) return;
      A[inp.dataset.k] = v / 100;
      rebuild();
    });
  });

  AISC.whenVisible(svg, function () { fixAxis(); base = understanding(null); setDeploy(deploy, false); });
  var rt; window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(draw, 120); });
  AISC.onTheme(draw);
})();
