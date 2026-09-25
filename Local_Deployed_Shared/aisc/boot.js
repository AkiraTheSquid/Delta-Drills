/* Loads the AISC write-up's scripts the first time its page is on screen.

   The write-up (#aisc-root, on #page-learn-about-app) carries seven figures.
   Most visits to the app never open this page, so nothing here is fetched
   until #aisc-root first intersects the viewport, which a `display: none`
   page never does.

   🔴 ORDER IS LOAD-BEARING. about-page-editor.js may swap the page's HTML for
   the saved server copy; every figure script grabs its elements by id at run
   time, so they must start only after `window.DDAboutContentReady` settles,
   exactly like concept-graph/why-graph.js. Scripts then run one at a time:
   common.js defines `AISC`, which field.js and every fig-*.js read at load. Cytoscape and the dagre layout come from the
   app's own vendor/graph/ tags (deferred, so they have run by DOMContentLoaded). */
(function () {
  "use strict";

  var SCRIPTS = [
    "aisc/js/engine.js?v=1",
    "aisc/js/common.js?v=3",
    "aisc/js/field.js?v=1",
    "aisc/js/fig-effort.js?v=3",
    "aisc/js/fig-forget.js?v=10",
    "aisc/js/fig-graphs.js?v=4",
    "aisc/js/fig-seth.js?v=2",
  ];
  var started = false;

  function loadInOrder(list) {
    return list.reduce(function (prev, src) {
      return prev.then(function () {
        return new Promise(function (resolve) {
          var s = document.createElement("script");
          s.src = src;
          s.onload = resolve;
          // One figure failing must not take the rest of the write-up with it.
          s.onerror = function () { console.warn("[aisc] failed to load", src); resolve(); };
          document.body.appendChild(s);
        });
      });
    }, Promise.resolve());
  }

  function start() {
    if (started) return;
    var root = document.getElementById("aisc-root");
    if (!root) return;
    started = true;
    var io = new IntersectionObserver(function (ents) {
      if (!ents.some(function (e) { return e.isIntersecting; })) return;
      io.disconnect();
      loadInOrder(SCRIPTS);
    });
    io.observe(root);
  }

  function boot() {
    var ready = window.DDAboutContentReady;
    if (ready && typeof ready.finally === "function") ready.finally(start);
    else start();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
