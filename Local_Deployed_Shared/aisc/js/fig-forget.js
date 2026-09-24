/* Fig. 4 — the forgetting curve, in CindyJS.

   The model lives in CindyScript so dragging a review marker re-simulates on
   every frame. JS only wires the buttons and the colour tokens.

   Model (illustrative, stated on the page):
     recall(t) = r0 * 2^(-(t - tLast) / h)
     h0 = 1.5 d × speed
     full review at recall r:  recall → 1,  h ← h × (1 + speed·(1 + 3(1 − r)))
     implicit rep of weight w: recall → r + w(1 − r),  h grows by w of that factor
   "Production today" mode = what the backend does now: fixed 14-day half-life,
   a review restores recall but never lengthens the next interval.
   Reviews dragged past the plot's right edge sit in the "unused" lane. */
(function () {
  "use strict";
  var host = document.getElementById("aisc-forgetCanvas");
  if (!host) return;
  var W = 100, H = 62.5;
  host.style.height = "auto";
  host.style.aspectRatio = W + " / " + H;

  var CS = [
    "PL = 9; PR = 84; PB = 11; PT = 58; TY = 4.2; DAYS = 60;",
    "LANE0 = 88; LANE1 = 98.5;",
    "X(t) := PL + t / DAYS * (PR - PL);",
    "Y(v) := PB + v * (PT - PB);",
    "T(x) := (x - PL) / (PR - PL) * DAYS;",
    "clampx(x) := min(max(x, PL + 0.4), LANE1);",
    "implicitDays = [3, 8, 14, 21, 29, 38, 48];",
    "h0() := if(fixed, 14, 1.5 * speed);",
    "gain(r) := if(fixed, 1, 1 + speed * (1 + 3 * (1 - r)));",
    // All events, sorted: [t, weight]
    "events() := (",
    "  ev = [];",
    "  forall([R1, R2, R3, R4, R5, R6], p, if(p.x <= PR, ev = append(ev, [T(p.x), 1])));",
    "  if(implicit, forall(implicitDays, d, ev = append(ev, [d, 0.3])));",
    "  sort(ev, #_1);",
    ");",
    // Walk the timeline; returns [curve, jumps, reps, nReviews, minRecall, finalH]
    "simulate() := (",
    "  ev = events(); h = h0(); r0 = 1; tl = 0; curve = [[X(0), Y(1)]]; jumps = [];",
    "  reps = 1; nrev = 0; minr = 1; k = 1; t = 0; dt = 0.2;",
    "  while(t < DAYS,",
    "    tn = min(t + dt, DAYS);",
    "    if(k <= length(ev), if((ev_k)_1 <= tn, tn = (ev_k)_1));",
    "    rc = r0 * 2^(-(tn - tl) / h);",
    "    curve = append(curve, [X(tn), Y(rc)]);",
    "    minr = min(minr, rc);",
    "    while(if(k <= length(ev), (ev_k)_1 <= tn + 0.0001, false),",
    "      w = (ev_k)_2;",
    "      h = h * (1 + w * (gain(rc) - 1));",
    "      rn = rc + w * (1 - rc);",
    "      jumps = append(jumps, [X(tn), Y(rc), Y(rn), w]);",
    "      r0 = rn; tl = tn; rc = rn; reps = reps + w;",
    "      if(w == 1, nrev = nrev + 1);",
    "      curve = append(curve, [X(tn), Y(rn)]);",
    "      k = k + 1;",
    "    );",
    "    t = tn;",
    "  );",
    "  [curve, jumps, reps, nrev, minr, h];",
    ");",
    // Place review markers where recall first falls to the target.
    "setR(i, x) := (",
    "  if(i == 1, R1.xy = [x, TY]); if(i == 2, R2.xy = [x, TY]); if(i == 3, R3.xy = [x, TY]);",
    "  if(i == 4, R4.xy = [x, TY]); if(i == 5, R5.xy = [x, TY]); if(i == 6, R6.xy = [x, TY]);",
    ");",
    "autoschedule() := (",
    "  forall(1..6, i, setR(i, LANE0 + 1.6 * (i - 1)));",
    "  imp = if(implicit, implicitDays, []);",
    "  h = h0(); r0 = 1; tl = 0; t = 0; placed = 0; j = 1;",
    "  while(t < DAYS & placed < 6,",
    "    t = t + 0.05;",
    "    rc = r0 * 2^(-(t - tl) / h);",
    "    if(j <= length(imp), if(imp_j <= t,",
    "      h = h * (1 + 0.3 * (gain(rc) - 1));",
    "      r0 = rc + 0.3 * (1 - rc); tl = t; j = j + 1; rc = r0;",
    "    ));",
    "    if(rc <= target & t < DAYS,",
    "      placed = placed + 1; setR(placed, X(t));",
    "      h = h * gain(rc); r0 = 1; tl = t;",
    "    );",
    "  );",
    ");",
    "lastReport = \"\";",
  ].join("\n");

  var DRAW = [
    "forall([R1, R2, R3, R4, R5, R6], p, p.xy = [clampx(p.x), TY]);",
    "sim = simulate(); curve = sim_1; jumps = sim_2;",
    // frame
    "fillpoly([[PL, PB], [PR, PB], [PR, PT], [PL, PT]], color -> cPanel, alpha -> 1);",
    "fillpoly([[PL, PB], [PR, PB], [PR, Y(target)], [PL, Y(target)]], color -> cRed, alpha -> 0.07);",
    "draw([PL, Y(target)], [PR, Y(target)], color -> cRed, size -> 1, dashpattern -> [5, 4]);",
    "drawtext([PR - 0.5, Y(target) + 1], \"review threshold \" + round(target * 100) + \"%\", size -> 11, color -> cRed, align -> \"right\", family -> \"IBM Plex Mono\");",
    "draw([PL, PB], [PR, PB], color -> cInk, size -> 1.2);",
    "draw([PL, PB], [PL, PT], color -> cInk, size -> 1.2);",
    "drawtext([PL - 1, PT - 1], \"100%\", size -> 11, color -> cMuted, align -> \"right\", family -> \"IBM Plex Mono\");",
    "drawtext([PL - 1, PB - 0.3], \"0%\", size -> 11, color -> cMuted, align -> \"right\", family -> \"IBM Plex Mono\");",
    "drawtext([PL - 5.5, (PB + PT) / 2], \"recall\", size -> 12, color -> cMuted, family -> \"IBM Plex Mono\");",
    "forall([0, 10, 20, 30, 40, 50, 60], d, draw([X(d), PB], [X(d), PB - 0.9], color -> cInk, size -> 1); drawtext([X(d), PB - 3.4], d + \"d\", size -> 11, color -> cMuted, align -> \"mid\", family -> \"IBM Plex Mono\"));",
    // jumps: amber bars with an arrow, the Math Academy review glyph
    "forall(jumps, j,",
    "  wd = if(j_4 == 1, 0.9, 0.45);",
    "  fillpoly([[j_1 - wd, j_2], [j_1 + wd, j_2], [j_1 + wd, j_3], [j_1 - wd, j_3]], color -> cAmber, alpha -> if(j_4 == 1, 0.9, 0.55));",
    "  if(j_4 == 1, fillpoly([[j_1 - 0.9, j_3 - 1.6], [j_1 + 0.9, j_3 - 1.6], [j_1, j_3 - 0.2]], color -> cInk, alpha -> 0.8));",
    ");",
    "connect(curve, color -> cCurve, size -> 2.6);",
    // review handle track
    "draw([PL, TY], [PR, TY], color -> cRule, size -> 5);",
    "fillpoly([[LANE0 - 0.8, TY - 1.6], [LANE1 + 0.8, TY - 1.6], [LANE1 + 0.8, TY + 1.6], [LANE0 - 0.8, TY + 1.6]], color -> cRule, alpha -> 0.6);",
    "drawtext([(LANE0 + LANE1) / 2, TY + 2.6], \"unused\", size -> 10, color -> cMuted, align -> \"mid\", family -> \"IBM Plex Mono\");",
    "drawtext([PL, TY + 2.4], \"drag reviews ↓\", size -> 10, color -> cMuted, family -> \"IBM Plex Mono\");",
    "drawtext([PL + 1, PT + 1.5], \"initial lesson\", size -> 11, color -> cMuted, family -> \"IBM Plex Mono\");",
    "drawtext([PR, PT + 1.5], if(fixed, \"PRODUCTION TODAY · fixed 14-day half-life\", \"PLANNED · expanding intervals\"), size -> 11, color -> if(fixed, cMuted, cCurve), align -> \"right\", family -> \"IBM Plex Mono\");",
    // report to the page when something changed
    "rep = format(sim_3, 1) + \"|\" + sim_4 + \"|\" + round(sim_5 * 100) + \"|\" + format(sim_6, 1);",
    "if(rep != lastReport, lastReport = rep; javascript(\"window.AISCForgetReport('\" + rep + \"')\"));",
  ].join("\n");

  function colours() {
    var c = function (n) { return "[" + AISC.rgb01(AISC.css(n)).map(function (v) { return v.toFixed(3); }).join(",") + "]"; };
    return "cInk=" + c("--ink") + ";cMuted=" + c("--ink-3") + ";cRule=" + c("--rule") + ";cPanel=" + c("--paper-2") +
      ";cRed=" + c("--red") + ";cAmber=" + c("--amber") + ";cCurve=" + c("--green") + ";";
  }

  window.AISCForgetReport = function (s) {
    var p = s.split("|");
    document.getElementById("aisc-forget-readout").innerHTML =
      "Full reviews in 60 days: <b>" + p[1] + "</b><br>Repetitions accrued: <b>" + p[0] + "</b><br>" +
      "Lowest recall: <b>" + p[2] + "%</b><br>Half-life at day 60: <b>" + p[3] + " d</b>";
  };

  AISC.whenVisible(host, function () {
    var pts = [];
    for (var i = 1; i <= 6; i++) {
      pts.push({ name: "R" + i, type: "Free", pos: [88 + 1.6 * (i - 1), 4.2], color: [0.9, 0.4, 0.25], size: 7, printname: "", labeled: false });
    }
    var cdy = CindyJS({
      ports: [{ id: "aisc-forgetCanvas", fill: "parent", transform: [{ visibleRect: [0, H, W, 0] }] }],
      scripts: { init: colours() + "speed = 1; target = 0.7; implicit = false; fixed = false;\n" + CS, draw: DRAW },
      geometry: pts,
      autoplay: false,
      language: "en",
    });
    window.AISCForget = cdy;
    function run(code) { cdy.evokeCS(code); }
    // First schedule once the instance has initialised.
    setTimeout(function () { run("autoschedule();"); }, 60);

    document.querySelectorAll("#aisc-speed-seg button").forEach(function (b) {
      b.addEventListener("click", function () {
        document.querySelectorAll("#aisc-speed-seg button").forEach(function (x) { x.classList.toggle("on", x === b); });
        run("speed = " + b.dataset.s + "; autoschedule();");
      });
    });
    var st = document.getElementById("aisc-s-target");
    st.addEventListener("input", function () {
      document.getElementById("aisc-o-target").textContent = st.value + "%";
      run("target = " + st.value / 100 + "; autoschedule();");
    });
    document.querySelectorAll("#aisc-model-seg button").forEach(function (b) {
      b.addEventListener("click", function () {
        document.querySelectorAll("#aisc-model-seg button").forEach(function (x) { x.classList.toggle("on", x === b); });
        var f = b.dataset.m === "today";
        document.getElementById("aisc-speed-seg").classList.toggle("off", f);
        run("fixed = " + f + "; autoschedule();");
      });
    });
    document.getElementById("aisc-forget-auto").addEventListener("click", function () { run("autoschedule();"); });
    var imp = document.getElementById("aisc-forget-implicit"), on = false;
    imp.addEventListener("click", function () {
      on = !on;
      imp.textContent = "Implicit reviews: " + (on ? "on" : "off");
      imp.classList.toggle("on", on);
      run("implicit = " + on + "; autoschedule();");
    });
    AISC.onTheme(function () { run(colours()); });
  });
})();
