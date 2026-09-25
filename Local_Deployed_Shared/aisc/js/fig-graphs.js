/* Figs. 3, 5, 6 — the knowledge-graph figures, on Cytoscape.

   All three read the real concept registry (data/kc_graph.json, exported from
   Local_Deployed_Shared/lessons/kc_registry.json) and run DeltaEngine, the
   browser port of the production update rules.

   Fig. 2 is NOT here: it is the app's own concept map, drawn by
   concept-graph/why-graph.js into #wta-graph-cy. These three borrow that
   map's look (Seth, 2026-09-24: "use the old embedded graph"): labelled
   round-rectangle nodes, red prerequisite arrows, yellow highlight, the
   dagre bottom-to-top layout. Each figure's own state still sets the fill. */
(function () {
  "use strict";
  var E = window.DeltaEngine;

  function shortName(id) { return id.split(".").slice(1).join(".").replace(/-/g, " "); }
  function mix(a, b, t) {
    var A = AISC.rgb01(a), B = AISC.rgb01(b);
    return "rgb(" + A.map(function (v, i) { return Math.round(255 * (v + (B[i] - v) * t)); }).join(",") + ")";
  }
  // why-graph.js's palette, so the four maps read as one product
  var EDGE = "#e3212c", ACCENT = "#ffd23f", LABEL = "#15151f";
  function baseStyle() {
    return [
      { selector: "node", style: {
        "background-color": "#9a9ab0", shape: "round-rectangle", label: "data(label)",
        width: "label", height: "label", padding: "7px", "text-wrap": "wrap", "text-max-width": 90,
        "text-valign": "center", "text-halign": "center", "font-size": 11, "font-weight": 600, color: LABEL,
        "border-width": 1, "border-color": "rgba(0,0,0,.28)",
      } },
      { selector: "edge", style: {
        "curve-style": "bezier", width: 1.6, "line-color": EDGE, "target-arrow-color": EDGE,
        "target-arrow-shape": "triangle", "arrow-scale": 0.8, opacity: 0.9,
      } },
      { selector: ".dim", style: { opacity: 0.12 } },
      { selector: "node.hl", style: { "border-width": 3, "border-color": ACCENT, "z-index": 5 } },
      { selector: "edge.hl", style: { "line-color": ACCENT, "target-arrow-color": ACCENT, width: 3, "z-index": 6 } },
    ];
  }
  // `extra` is a FUNCTION so a theme switch can rebuild styles from the new tokens.
  function makeCy(el, elements, extra, layout) {
    var cy = cytoscape({
      container: el, elements: elements, style: baseStyle().concat(extra()),
      layout: layout, minZoom: 0.2, maxZoom: 3, wheelSensitivity: 0.2, boxSelectionEnabled: false,
      userPanningEnabled: true, userZoomingEnabled: false, autoungrabify: true,
    });
    cy.restyle = function () { cy.style(baseStyle().concat(extra())); };
    // Full screen (fig-expand.js) and narrow layouts change the stage's box;
    // refit to the new one rather than keep the old framing.
    if (typeof ResizeObserver === "function") {
      var last = el.clientWidth + "x" + el.clientHeight;
      new ResizeObserver(function () {
        var now = el.clientWidth + "x" + el.clientHeight;
        if (now === last || !el.clientWidth) return;
        last = now; cy.resize(); cy.fit(undefined, 16);
      }).observe(el);
    }
    return cy;
  }
  function dagreLayout(opts) {
    return Object.assign({ name: "dagre", rankDir: "BT", nodeSep: 12, rankSep: 58, edgeSep: 8, padding: 16, fit: true }, opts || {});
  }
  function graphElements(kcs, keep) {
    var ids = {}; kcs.forEach(function (k) { if (!keep || keep[k.id]) ids[k.id] = true; });
    var els = [];
    kcs.forEach(function (k) {
      if (!ids[k.id]) return;
      els.push({ data: { id: k.id, label: shortName(k.id), lesson: k.lesson, title: k.title } });
    });
    kcs.forEach(function (k) {
      if (!ids[k.id]) return;
      k.prereqs.forEach(function (p) {
        if (!ids[p]) return;
        var w = k.enc[p];
        els.push({ data: { id: p + ">" + k.id, source: p, target: k.id, w: w || 0, wl: w ? w.toFixed(1) : "" }, classes: w ? "enc" : "pre" });
      });
    });
    return els;
  }
  // Wheel-zoom only with a modifier, so the page scroll isn't hijacked.
  function ctrlZoom(cy, el) {
    el.addEventListener("wheel", function (e) {
      if (!(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      var r = el.getBoundingClientRect();
      cy.zoom({ level: cy.zoom() * (e.deltaY < 0 ? 1.12 : 0.89), renderedPosition: { x: e.clientX - r.left, y: e.clientY - r.top } });
    }, { passive: false });
  }

  AISC.graph.then(function (G) {
    var byId = {}; G.kcs.forEach(function (k) { byId[k.id] = k; });
    var nEdges = G.kcs.reduce(function (s, k) { return s + k.prereqs.length; }, 0);
    document.getElementById("aisc-stat-kcs").textContent = G.kcs.length;
    document.getElementById("aisc-g-kcs").textContent = G.kcs.length;
    document.getElementById("aisc-g-edges").textContent = nEdges;
    document.getElementById("aisc-stat-q").textContent = G.n_questions.toLocaleString();

    // ================================================================ Fig 3
    AISC.whenVisible(document.getElementById("aisc-cy-place"), function () {
      var el = document.getElementById("aisc-cy-place");
      var graph = new E.Graph(G.kcs), value = E.coreness(graph);
      var B, probed, current, timer = null, truth = null;
      var cy = makeCy(el, graphElements(G.kcs), function () { return [
        // not yet asked = dashed, as why-graph.js draws an inferred reading
        { selector: "node", style: { "border-style": "dashed" } },
        { selector: "node.probed", style: { "border-style": "solid" } },
        { selector: "node.frontier", style: { "border-width": 3, "border-style": "solid", "border-color": ACCENT } },
        { selector: "node.probe", style: { "border-width": 4, "border-style": "solid", "border-color": AISC.css("--coral"), "z-index": 9 } },
      ]; }, dagreLayout());
      ctrlZoom(cy, el);

      function paint() {
        var P = B.probs(), counts = { known: 0, uncertain: 0, unknown: 0 }, frontier = 0;
        cy.batch(function () {
          cy.nodes().forEach(function (n) {
            var p = P[n.id()], c = E.classify(p);
            counts[c]++;
            var col = p >= 0.5 ? mix(AISC.css("--amber"), AISC.css("--green"), (p - 0.5) / 0.5) : mix(AISC.css("--red"), AISC.css("--amber"), p / 0.5);
            n.style("background-color", col);
            var isF = c !== "known" && graph.parents[n.id()].every(function (q) { return P[q] >= E.PL.IN_STATE; });
            n.toggleClass("frontier", isF);
            if (isF) frontier++;
            n.toggleClass("probe", n.id() === current);
            n.toggleClass("probed", !!probed[n.id()]);
          });
        });
        document.getElementById("aisc-place-readout").innerHTML =
          "asked <b>" + Object.keys(probed).length + "</b> · known <b>" + counts.known + "</b> · uncertain <b>" + counts.uncertain +
          "</b> · unknown <b>" + counts.unknown + "</b> · frontier <b>" + frontier + "</b>";
      }
      function next() {
        var cands = graph.kcs.filter(function (k) { return !probed[k]; });
        var ranked = E.rankProbes(B, graph, value, cands);
        current = ranked.length ? ranked[0][1] : null;
        document.getElementById("aisc-probe-title").textContent = current ? byId[current].title : "Every concept probed.";
        paint();
      }
      function answer(r) {
        if (!current) return;
        E.applyProbe(B, graph, current, r);
        probed[current] = r;
        next();
      }
      function reset() {
        if (timer) { clearInterval(timer); timer = null; }
        B = new E.Beliefs(graph.kcs); probed = {}; truth = null;
        document.getElementById("aisc-place-auto").textContent = "Simulate a learner";
        next();
      }
      // A simulated learner: knows every concept at or below a random cut in
      // the topological order, plus a few gaps — the shape a real one has.
      function makeTruth() {
        var depth = {};
        function d(k) { if (depth[k] != null) return depth[k]; var ps = graph.parents[k]; return (depth[k] = ps.length ? 1 + Math.max.apply(null, ps.map(d)) : 0); }
        graph.kcs.forEach(d);
        var cut = 2 + Math.floor(Math.random() * 4), t = {};
        graph.kcs.forEach(function (k) { t[k] = depth[k] <= cut && Math.random() > 0.12; });
        // Knowing a concept requires knowing its prerequisites.
        var changed = true;
        while (changed) { changed = false; graph.kcs.forEach(function (k) { if (t[k] && graph.parents[k].some(function (p) { return !t[p]; })) { t[k] = false; changed = true; } }); }
        return t;
      }
      document.querySelectorAll("#aisc-fig-placement [data-r]").forEach(function (b) {
        b.addEventListener("click", function () { answer(b.dataset.r); });
      });
      document.getElementById("aisc-place-reset").addEventListener("click", reset);
      document.getElementById("aisc-place-auto").addEventListener("click", function () {
        var btn = this;
        if (timer) { clearInterval(timer); timer = null; btn.textContent = "Simulate a learner"; return; }
        reset(); truth = makeTruth(); btn.textContent = "Stop";
        var n = 0;
        timer = setInterval(function () {
          if (!current || n++ >= 24) { clearInterval(timer); timer = null; btn.textContent = "Simulate a learner"; return; }
          var k = truth[current];
          // Slips and lucky guesses at the model's own rates.
          var ok = k ? Math.random() > E.PL.P_SLIP : Math.random() < E.PL.P_GUESS;
          answer(ok ? "correct" : k ? "incorrect" : (Math.random() < 0.5 ? "dont_know" : "incorrect"));
        }, 550);
      });
      reset();
      AISC.onTheme(function () { cy.restyle(); paint(); });
    });

    // ================================================================ Fig 5
    AISC.whenVisible(document.getElementById("aisc-cy-fire"), function () {
      var el = document.getElementById("aisc-cy-fire");
      var keep = {};
      G.kcs.forEach(function (k) { if (k.id.indexOf("einops.") === 0) keep[k.id] = true; });
      ["torch.axis-reductions", "torch.aggregations", "torch.broadcasting-rules"].forEach(function (k) { keep[k] = true; });
      var encIndex = {};
      G.kcs.forEach(function (k) {
        if (!keep[k.id]) return;
        encIndex[k.id] = Object.keys(k.enc).filter(function (t) { return keep[t]; }).map(function (t) { return { kc: t, w: k.enc[t] }; });
      });
      var cy = makeCy(el, graphElements(G.kcs, keep), function () { return [
        { selector: "edge.enc", style: { width: 2.2, "line-color": AISC.css("--green"), "target-arrow-color": AISC.css("--green"), label: "data(wl)", "font-size": 9, "font-family": "IBM Plex Mono", color: AISC.css("--green"), "text-background-color": AISC.css("--paper"), "text-background-opacity": 1, "text-background-padding": 2 } },
        { selector: "edge.pre", style: { "line-style": "dashed", "line-dash-pattern": [3, 3] } },
        { selector: "node.flash", style: { "border-width": 4, "border-color": ACCENT } },
        { selector: "node.target", style: { "border-width": 4, "border-color": AISC.css("--coral") } },
      ]; }, dagreLayout({ rankSep: 50, nodeSep: 14, nodeDimensionsIncludeLabels: true }));
      ctrlZoom(cy, el);
      var mastery, day, touched, target = "einops.pooling";
      function effective(k) { var L = mastery[k] == null ? E.BKT.P_INIT : mastery[k]; return touched[k] == null ? L : E.decay(L, day - touched[k]); }
      function paint() {
        cy.nodes().forEach(function (n) {
          var p = effective(n.id());
          n.style("background-color", mix("#9a9ab0", AISC.css("--green"), Math.min(1, (p - E.BKT.P_INIT) / (1 - E.BKT.P_INIT))));
          n.toggleClass("target", n.id() === target);
        });
        document.getElementById("aisc-fire-target").innerHTML = "<span class='id'>Practising</span> <span class='t'>" + byId[target].title + "</span> <span class='id'>P(known) " + effective(target).toFixed(2) + "</span>";
      }
      function reset() {
        mastery = {}; touched = {}; day = 0;
        Object.keys(keep).forEach(function (k) { mastery[k] = 0.55; touched[k] = 0; });
        document.getElementById("aisc-s-days").value = 0; document.getElementById("aisc-o-days").textContent = "0 d";
        document.getElementById("aisc-fire-readout").textContent = "Every concept starts at 0.55. Click one, answer, and watch where the credit lands.";
        paint();
      }
      function practise(ok) {
        // Bake the decay in at the current day, then update.
        Object.keys(keep).forEach(function (k) { mastery[k] = effective(k); touched[k] = day; });
        var before = Object.assign({}, mastery);
        var ch = E.applyAttempt(mastery, encIndex, target, ok);
        var lines = Object.keys(ch).map(function (k) {
          return (k === target ? "direct " : "implicit ") + shortName(k) + ": " + before[k].toFixed(2) + " → <b>" + ch[k].toFixed(2) + "</b>";
        });
        if (ok && lines.length === 1) lines.push("<span class='muted'>it encompasses nothing, so no implicit credit</span>");
        if (!ok) lines.push("<span class='muted'>a wrong answer credits nothing downstream</span>");
        document.getElementById("aisc-fire-readout").innerHTML = lines.join(" · ");
        Object.keys(ch).forEach(function (k) { if (k !== target) { var n = cy.getElementById(k); n.addClass("flash"); setTimeout(function () { n.removeClass("flash"); }, 900); } });
        paint();
      }
      cy.on("tap", "node", function (e) { target = e.target.id(); paint(); });
      document.getElementById("aisc-fire-ok").addEventListener("click", function () { practise(true); });
      document.getElementById("aisc-fire-no").addEventListener("click", function () { practise(false); });
      document.getElementById("aisc-fire-reset").addEventListener("click", reset);
      document.getElementById("aisc-s-days").addEventListener("input", function () {
        day = +this.value; document.getElementById("aisc-o-days").textContent = day + " d"; paint();
      });
      reset();
      AISC.onTheme(function () { cy.restyle(); paint(); });
    });

    // ================================================================ Fig 6
    AISC.whenVisible(document.getElementById("aisc-cy-remed"), function () {
      var el = document.getElementById("aisc-cy-remed");
      var TARGET = "einops.attention-einsum";
      var keep = {}; keep[TARGET] = true;
      // Two layers of prerequisites below the target.
      byId[TARGET].prereqs.forEach(function (p) { keep[p] = true; byId[p].prereqs.forEach(function (q) { keep[q] = true; }); });
      var START = { "einops.attention-einsum": 0.30, "einops.einsum": 0.88, "einops.dl-flatten-heads": 0.52, "einops.merge-axes": 0.64, "einops.split-axes": 0.41, "einops.pattern-language": 0.83 };
      var cy = makeCy(el, graphElements(G.kcs, keep), function () { return [
        { selector: "node.goal", style: { "border-width": 5, "border-style": "double", "border-color": LABEL } },
        { selector: "node.serving", style: { "border-width": 4, "border-style": "solid", "border-color": ACCENT } },
      ]; }, dagreLayout({ rankSep: 50, nodeSep: 18, nodeDimensionsIncludeLabels: true }));
      var mastery, attempts, seq, log;
      function trailingMisses(kc) {
        var out = [], rows = attempts[kc] || [];
        for (var i = rows.length - 1; i >= 0; i--) { if (rows[i].ok) break; out.push(rows[i]); }
        return out;
      }
      function dosed(p, since) { return (attempts[p] || []).filter(function (a) { return a.seq > since; }).length >= 3; }
      // remediation.redirect, line for line.
      function redirect(kc) {
        var misses = trailingMisses(kc);
        if (misses.length < 2) return kc;
        var since = misses[0].seq, seen = {}; seen[kc] = true;
        var layer = byId[kc].prereqs.filter(function (p) { return keep[p]; });
        while (layer.length) {
          var eligible = layer.filter(function (p) { return !seen[p] && !dosed(p, since); });
          if (eligible.length) {
            return eligible.sort(function (a, b) { return mastery[a] - mastery[b] || (a < b ? -1 : 1); })[0];
          }
          layer.forEach(function (p) { seen[p] = true; });
          var nextLayer = [];
          layer.forEach(function (p) { byId[p].prereqs.forEach(function (g) { if (keep[g] && !seen[g] && nextLayer.indexOf(g) < 0) nextLayer.push(g); }); });
          layer = nextLayer;
        }
        return kc;
      }
      var serving;
      function paint() {
        cy.nodes().forEach(function (n) {
          var p = mastery[n.id()];
          n.style("background-color", p >= 0.5 ? mix(AISC.css("--amber"), AISC.css("--green"), (p - 0.5) / 0.5) : mix(AISC.css("--red"), AISC.css("--amber"), p / 0.5));
          n.data("label", shortName(n.id()) + "\n" + p.toFixed(2));
          n.toggleClass("serving", n.id() === serving);
          n.toggleClass("goal", n.id() === TARGET);
        });
        document.getElementById("aisc-remed-id").textContent = serving === TARGET ? "frontier concept" : "prerequisite, redirected";
        document.getElementById("aisc-remed-title").textContent = byId[serving].title;
        document.getElementById("aisc-remed-log").innerHTML = log.slice(-4).join("<br>");
      }
      function reset() {
        mastery = Object.assign({}, START); attempts = {}; seq = 0; log = ["Serving the frontier concept. Try missing it twice."];
        serving = TARGET; paint();
      }
      document.querySelectorAll("#aisc-fig-remed [data-a]").forEach(function (b) {
        b.addEventListener("click", function () {
          var ok = b.dataset.a === "1";
          seq++;
          (attempts[serving] = attempts[serving] || []).push({ ok: ok, seq: seq });
          mastery[serving] = E.observe(mastery[serving], ok);
          var prev = serving;
          serving = redirect(TARGET);
          var msg = (ok ? "✓ " : "✗ ") + shortName(prev) + " → P " + mastery[prev].toFixed(2);
          if (serving !== prev) msg += serving === TARGET ? " · <b>back to the concept</b>" : " · <b>redirect → " + shortName(serving) + "</b>";
          log.push(msg);
          paint();
        });
      });
      document.getElementById("aisc-remed-reset").addEventListener("click", reset);
      reset();
      AISC.onTheme(function () { cy.restyle(); paint(); });
    });
  });
})();
