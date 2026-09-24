/* Shared page plumbing for the AISC write-up on the app's "Learn about the
   app" page (#page-learn-about-app): theme, reveal-on-scroll, section-nav
   highlight, colour tokens for canvas code, the KC graph fetch, and the hero
   ornament.

   Ported from the standalone delta-drills-aisc site on 2026-09-24. Differences
   from the original:
   - Every token lives on `.aisc` (#aisc-root), not on :root, so `css()` reads
     that element. The site's own theme toggle is gone; the app's theme.js owns
     `html[data-theme]` (light | dark | blue) and aisc.css maps it, so a theme
     switch arrives as `delta:theme-changed`.
   - Section nav links scroll inside the app instead of setting location.hash.
   - The global is `AISC`, not `DD`, and every id carries an `aisc-` prefix. */
(function (root) {
  "use strict";

  var host = document.getElementById("aisc-root") || document.documentElement;
  var themeListeners = [];

  function css(name) {
    return getComputedStyle(host).getPropertyValue(name).trim();
  }
  // "#rrggbb" → [r,g,b] in 0..1 (CindyScript colour form).
  function rgb01(hex) {
    var h = hex.replace("#", "");
    if (h.length === 3) h = h.split("").map(function (c) { return c + c; }).join("");
    var n = parseInt(h, 16);
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
  }
  function onTheme(fn) { themeListeners.push(fn); }
  function fireTheme() { themeListeners.forEach(function (fn) { try { fn(); } catch (e) { console.error(e); } }); }
  // theme.js stamps the attribute before dispatching, so the new tokens are
  // already resolved when the listeners read them.
  window.addEventListener("delta:theme-changed", fireTheme);

  // Reveal on scroll.
  var io = new IntersectionObserver(function (ents) {
    ents.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
  }, { rootMargin: "0px 0px -8% 0px" });
  host.querySelectorAll(".rv").forEach(function (el) { io.observe(el); });

  // Section nav: highlight the section in view, and scroll on click without
  // touching location.hash (the app routes on its own state, not the hash).
  var links = {};
  host.querySelectorAll(".rail nav a").forEach(function (a) {
    var id = a.getAttribute("href").slice(1);
    links[id] = a;
  });
  host.addEventListener("click", function (e) {
    var a = e.target.closest && e.target.closest('a[href^="#aisc-"]');
    if (!a) return;
    var target = document.getElementById(a.getAttribute("href").slice(1));
    if (!target) return;
    e.preventDefault();
    target.scrollIntoView({ block: "start", behavior: "smooth" });
  });
  var navIo = new IntersectionObserver(function (ents) {
    ents.forEach(function (e) {
      if (!e.isIntersecting || !links[e.target.id]) return;
      Object.keys(links).forEach(function (k) { links[k].classList.toggle("on", k === e.target.id); });
    });
  }, { rootMargin: "-45% 0px -50% 0px" });
  host.querySelectorAll(".aisc-main > section[id]").forEach(function (s) { navIo.observe(s); });

  // Run `fn` once the element is near the viewport (figures are heavy).
  function whenVisible(el, fn) {
    var o = new IntersectionObserver(function (ents) {
      if (ents.some(function (e) { return e.isIntersecting; })) { o.disconnect(); fn(); }
    }, { rootMargin: "400px 0px" });
    o.observe(el);
  }

  var graphPromise = fetch("aisc/data/kc_graph.json").then(function (r) { return r.json(); });

  // Lesson palette — muted, distinguishable in both themes.
  var LESSON_HUES = {
    "py-0": 28, "ma-00": 205, "ma-01": 220, "np-1": 150, "np-2": 165, "np-3": 180,
    "eo-1": 45, "eo-2": 55, "eo-3": 65, "es-1": 330, "tr-1": 265,
    "ar-00": 10, "ar-01": 350, "ar-02": 0,
  };
  function lessonColor(lesson) {
    var p = rgb01(css("--paper")), dark = p[0] + p[1] + p[2] < 1.5;
    var h = LESSON_HUES[lesson] != null ? LESSON_HUES[lesson] : 0;
    return "hsl(" + h + (dark ? ",35%,42%)" : ",45%,72%)");
  }

  // Hero ornament: a small DAG with a frontier, drawn once.
  (function hero() {
    var svg = document.getElementById("aisc-hero-svg");
    if (!svg) return;
    var N = [
      [200, 40, "f"], [110, 120, "f"], [290, 120, ""], [60, 210, "k"], [160, 210, "k"], [250, 210, "f"],
      [340, 210, ""], [40, 300, "k"], [120, 300, "k"], [200, 300, "k"], [290, 300, "k"], [360, 300, "k"],
      [100, 370, "k"], [220, 370, "k"], [320, 370, "k"],
    ];
    var E = [[1, 0], [2, 0], [3, 1], [4, 1], [5, 2], [6, 2], [7, 3], [8, 3], [8, 4], [9, 4], [9, 5], [10, 5], [10, 6], [11, 6], [12, 7], [12, 8], [13, 9], [13, 10], [14, 10], [14, 11]];
    var s = "";
    E.forEach(function (e) {
      var a = N[e[0]], b = N[e[1]];
      s += '<path class="h-edge" d="M' + a[0] + " " + a[1] + " C " + a[0] + " " + (a[1] - 40) + ", " + b[0] + " " + (b[1] + 40) + ", " + b[0] + " " + b[1] + '"/>';
    });
    N.forEach(function (n, i) {
      s += '<circle class="h-node ' + n[2] + '" cx="' + n[0] + '" cy="' + n[1] + '" r="' + (n[2] === "f" ? 13 : 10) + '"/>';
    });
    s += '<text x="226" y="44">frontier</text>';
    svg.innerHTML = s;
  })();

  root.AISC = { css: css, rgb01: rgb01, onTheme: onTheme, whenVisible: whenVisible, graph: graphPromise, lessonColor: lessonColor };
})(window);
