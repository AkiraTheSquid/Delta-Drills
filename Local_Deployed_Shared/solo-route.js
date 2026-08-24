/* ================================================================
   SOLO-ROUTE.JS — pathname deep links: one page, no app chrome
   ================================================================ */

(function installSoloRoutes(global) {
  const SOLO_CLASS = "dd-solo";
  const READY_CLASS = "dd-solo-ready";

  // Public URL slug -> existing internal tab/page id. Slugs match without case
  // or an optional .html suffix; values stay identical to data-tab/page-*.
  const ROUTES = Object.freeze({
    "learn-about-app": "learn-about-app",
    // Legacy slugs. "Why This App Exists" and "How to use it" were two tabs
    // until 2026-08-23 and both were linkable pathnames; they are one page
    // now, and a link that used to open one of them still opens it.
    "why-this-app": "learn-about-app",
    "how-to-use": "learn-about-app",
    "knowledge-graph": "knowledge-graph",
    diagnostic: "diagnostic",
    "split-tool": "split-tool",
    account: "account",
    courses: "courses",
    practice: "practice",
    notebooks: "notebooks",
    "targeted-practice": "targeted-practice",
  });

  const pathSlug = () => String(global.location?.pathname || "")
    .replace(/\.html?$/i, "")
    .replace(/^\/+|\/+$/g, "")
    .toLowerCase();

  const read = () => ROUTES[pathSlug()] || "";

  const mountFullAppLink = () => {
    if (document.querySelector(".dd-solo-exit")) return;
    const link = document.createElement("a");
    link.className = "dd-solo-exit";
    link.href = "/";
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
