/* The About page's articles (Seth, 2026-09-25: "break up the page for the
   different sections into different articles").

   #aisc-root holds five <article class="aisc-article">s. One is open at a
   time; this file shows it, hides the rest with the `hidden` attribute, and
   marks its link in the topbar (#aisc-toc) with `.on` + aria-current. (An
   in-page index of article cards went the same day; the topbar is the only
   switcher.)

   Any `#aisc-…` link on the page or in the topbar lands here: an article id
   opens that article; any other id opens the article holding it, then scrolls
   to it. account-menu.js calls `AISCArticles.openFor(el)` for the same reason.

   Loaded eagerly (not by boot.js): the account menu and the topbar need it
   before the figures are ever fetched.

   🔴 about-page-editor.js may swap the page's HTML for the saved copy, which
   carries whatever `hidden` attributes were live when it was saved; the open
   article is re-applied once `DDAboutContentReady` settles. Look elements up
   every time, never cache them.

   🔴 Figures sized while their article was hidden read a width of 0 and fall
   back to 800px; a window `resize` after each switch makes the SVG figures
   redraw at their real width (fig-graphs.js refits on its own ResizeObserver,
   why-graph.js draws on the `hidden` attribute changing). */
(function () {
  "use strict";

  var current = null;

  function root() { return document.getElementById("aisc-root"); }
  function articles() {
    var r = root();
    return r ? Array.prototype.slice.call(r.querySelectorAll("article.aisc-article[id]")) : [];
  }

  function show(id) {
    var list = articles();
    if (!list.length) return null;
    var want = list.filter(function (a) { return a.id === id; })[0] || list[0];
    var changed = want.hidden || want.id !== current;
    list.forEach(function (a) { a.hidden = a !== want; });
    current = want.id;
    document.querySelectorAll("#aisc-toc a").forEach(function (a) {
      var on = a.getAttribute("href") === "#" + want.id;
      a.classList.toggle("on", on);
      if (on) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current");
    });
    if (changed) requestAnimationFrame(function () { window.dispatchEvent(new Event("resize")); });
    return want;
  }

  // Open the article that holds `el` (or `el` itself, if it is one).
  function openFor(el) {
    if (!el || !el.closest) return null;
    var art = el.closest("article.aisc-article[id]");
    return art ? show(art.id) : null;
  }

  function topbar() {
    return parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--dd-topbar-h")) || 44;
  }

  document.addEventListener("click", function (e) {
    var a = e.target.closest && e.target.closest('a[href^="#aisc-"]');
    var r = root(), toc = document.getElementById("aisc-toc");
    if (!a || !r || !(r.contains(a) || (toc && toc.contains(a)))) return;
    var target = document.getElementById(a.getAttribute("href").slice(1));
    if (!target) return;
    // The app routes on its own state, never the hash.
    e.preventDefault();
    var art = openFor(target);
    requestAnimationFrame(function () {
      if (art === target) {
        // An article: scroll only if its top is off screen or far down it.
        var top = target.getBoundingClientRect().top;
        if (top >= topbar() && top <= window.innerHeight * 0.5) return;
      }
      target.scrollIntoView({ block: "start", behavior: "smooth" });
    });
  });

  function init() { show(current); }
  function boot() {
    init();
    var ready = window.DDAboutContentReady;
    if (ready && typeof ready.finally === "function") ready.finally(init);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot, { once: true });
  else boot();

  window.AISCArticles = { show: show, openFor: openFor, current: function () { return current; } };
})();
