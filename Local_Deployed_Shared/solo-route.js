/* ================================================================
   SOLO-ROUTE.JS — pathname deep links: one page, no app chrome
   ================================================================ */

(function installSoloRoutes(global) {
  const SOLO_CLASS = "dd-solo";
  const READY_CLASS = "dd-solo-ready";

  /* 🔴 SOLO IS OPT-IN SINCE 2026-09-29. A bare pathname (/knowledge-graph)
     opens the FULL app on that page now (deep-link.js; Seth chose it over the
     chromeless view). This one-page, no-navigation view is for embeds and
     asks for itself: /knowledge-graph?solo=1. The route table is
     deep-link.js's, so the two can never disagree about what a path names. */
  const isSolo = () => {
    try {
      return new URLSearchParams(global.location?.search || "").get("solo") === "1";
    } catch (_) {
      return false;
    }
  };

  const read = () => (isSolo() ? global.DDDeepLink?.read?.() || "" : "");

  const mountFullAppLink = () => {
    if (document.querySelector(".dd-solo-exit")) return;
    const link = document.createElement("a");
    link.className = "dd-solo-exit";
    // The same page, with the navigation back.
    link.href = global.location?.pathname || "/";
    link.textContent = "Open full app";
    link.title = "Open Delta Drills with full navigation";
    if (global.top !== global.self) {
      link.target = "_blank";
      link.rel = "noopener";
    }
    document.body.appendChild(link);
  };

  const apply = () => {
    const page = read();
    const root = document.documentElement;
    root.classList.toggle(SOLO_CLASS, page !== "");
    root.classList.toggle(READY_CLASS, page !== "");
    if (!page) {
      root.removeAttribute("data-solo-page");
      return "";
    }
    root.dataset.soloPage = page;
    mountFullAppLink();
    return page;
  };

  global.DDSoloRoute = Object.freeze({ read, apply });
})(window);
