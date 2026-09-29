/* ================================================================
   DEEP-LINK.JS — one URL per page, and the address bar follows you

   WHAT IT DOES (Seth, 2026-09-29: "create deeplinks for all the different
   pages ... like the why this app exists page and the how it works pages")
     Every page of the app has a pathname: /practice, /knowledge-graph,
     /groups, /courses, /why-this-app, /how-it-works, /about-placement, ...
     Opening one boots the FULL app (topbar, account menu, article switcher)
     on that page. Moving around inside the app writes the matching pathname
     with history.pushState, so the address bar is always a link to where you
     are, and Back / Forward walk the pages you visited.

   WHAT IT REPLACED
     solo-route.js (2026-08-20) gave a handful of pages a pathname that opened
     them WITHOUT the app chrome. That view is still there for embeds, behind
     `?solo=1` (solo-route.js asks this file which page a path names). Seth
     chose the full app as the default for a link on 2026-09-29: a chromeless
     /why-this-app hid the topbar, and the About page's article switcher
     (#aisc-toc) lives IN the topbar.

   🔴 ONE PATH SEGMENT, NEVER TWO
     Every script, stylesheet and fetch in this app uses a RELATIVE url
     (`src="app.js"`, `fetch("lessons/…")`). They resolve against the
     document's url, and pushState changes that url. At /about/placement the
     next `fetch("lessons/x.json")` would ask for /about/lessons/x.json, which
     vercel.json rewrites to index.html. So sections of the About page are
     `/about-placement`, not `/about/placement`.

   🔴 `switchTab` IS NOT ON `window` (app.js top-level const; see
     account-menu.js). This is a classic script loaded BEFORE app.js, so it
     can only reach `switchTab` at call time, by name, never at load.

   ROUTE SHAPES
     /<tab>                 a page, by its data-tab name (plus a few aliases)
     /why-this-app …        a named place on the About page (ABOUT below)
     /about-<x>             any other About article or section: #aisc-a-<x>
                            (an article) or #aisc-<x> (a section / figure)
     /arena-<n>-<m>         an ARENA notebook section, e.g. /arena-0-1
     /                      the normal landing page (fork or Learner Home)
   ================================================================ */

(function installDeepLinks(global) {
  const ABOUT_TAB = "learn-about-app";
  const ARENA_TAB = "arena-notebook";

  /* Pathname -> page. Values are exact data-tab / page-<name> names, so
     switchTab stays the one owner of which page is showing. */
  const TABS = Object.freeze({
    practice: "practice",
    "learner-home": "practice",
    welcome: "welcome",
    "course-pick": "course-pick",
    "learn-about-app": ABOUT_TAB,
    about: ABOUT_TAB,
    "knowledge-graph": "knowledge-graph",
    "instructor-review": "instructor-review",
    "split-tool": "split-tool",
    account: "account",
    settings: "account",
    courses: "courses",
    "course-builder": "course-builder",
    notebooks: "notebooks",
    // A notebook page with no section named has nothing to show; the list of
    // sections is the Courses page. /arena-0-1 names one (see parse).
    [ARENA_TAB]: "courses",
    groups: "groups",
    "concept-chat": "concept-chat",
    "targeted-practice": "targeted-practice",
    // The placement test's old pathname. Retired 2026-09-26; its survey is on
    // the Learner Home now.
    diagnostic: "practice",
    placement: "practice",
  });

  /* Named places on the About page -> the element id to open. The FIRST slug
     listed for an id is the one the address bar shows for it. The two account
     menu rows ("Why this app exists" / "How this app works") and the five
     articles in the topbar switcher each get a plain-English name. */
  const ABOUT = Object.freeze([
    ["why-this-app", "aisc-a-why"],
    ["how-this-app-works", "aisc-why-use"],
    ["how-it-works", "aisc-a-how"],
    ["the-case", "aisc-a-case"],
    ["evidence", "aisc-a-evidence"],
    ["aisc-proposal", "aisc-a-aisc"],
    // Linkable pathname from before the 2026-08-23 merge ("How to use it").
    ["how-to-use", "aisc-why-use"],
  ]);
  const ABOUT_BY_SLUG = new Map(ABOUT);

  /* Query parameters that describe the whole visit rather than the page it
     started on, so they ride along on every URL this file writes. Everything
     else (?invite=, ?arena=, ?notebook=, ?lesson=) is a one-shot landing
     instruction and is dropped when you move on. */
  const KEEP_PARAMS = ["mode"];

  const slugOf = (pathname) => String(pathname || "")
    .replace(/\.html?$/i, "")
    .replace(/^\/+|\/+$/g, "")
    .toLowerCase();

  /* A route is { tab, anchor?, arena? }, or null for "/" and for any path
     that names nothing (vercel.json rewrites every path to the app). */
  const parse = (pathname) => {
    const slug = slugOf(pathname);
    if (!slug || slug.includes("/")) return null;
    if (ABOUT_BY_SLUG.has(slug)) return { tab: ABOUT_TAB, anchor: ABOUT_BY_SLUG.get(slug) };
    // Own keys only: `/constructor` must not resolve to Object's.
    if (Object.prototype.hasOwnProperty.call(TABS, slug)) return { tab: TABS[slug] };
    const arena = /^arena-(\d+-\d+)$/.exec(slug);
    if (arena) return { tab: ARENA_TAB, arena: arena[1] };
    const about = /^about-([a-z0-9-]+)$/.exec(slug);
    if (about) return { tab: ABOUT_TAB, anchor: `about:${about[1]}` };
    return null;
  };

  /* `about:<x>` is resolved against the live page, because the element may
     not exist until about-page-editor.js has settled. An article wins over a
     section of the same name: #aisc-a-how opens on #aisc-how anyway. */
  const resolveAnchor = (anchor) => {
    if (!anchor) return null;
    if (!anchor.startsWith("about:")) return document.getElementById(anchor);
    const x = anchor.slice("about:".length);
    return document.getElementById(`aisc-a-${x}`) || document.getElementById(`aisc-${x}`);
  };

  const aboutSlug = (id) => {
    const named = ABOUT.find(([, target]) => target === id);
    if (named) return named[0];
    if (/^aisc-/.test(id || "")) return `about-${id.replace(/^aisc-(a-)?/, "")}`;
    return "about";
  };

  // The pathname for a place in the app. "" means "/".
  const pathFor = (route) => {
    if (!route) return "";
    if (route.tab === ABOUT_TAB) {
      const id = route.anchor && !route.anchor.startsWith("about:")
        ? route.anchor
        : resolveAnchor(route.anchor)?.id || global.AISCArticles?.current?.();
      return id ? aboutSlug(id) : "about";
    }
    if (route.tab === ARENA_TAB && route.arena) return `arena-${route.arena}`;
    return route.tab;
  };

  // ---- state ------------------------------------------------------------
  let landingTab = "";   // what "/" showed on this load
  let applying = false;  // true while we are the ones moving the app
  let pendingArena = ""; // ArenaNotebook.open() names its section here first
  let replaceNext = false; // the next page switch is a redirect, not a visit

  /* ?solo=1 (a chromeless embed) and ?embed=1 (the Knowledge Graph's practice
     iframe) are a page inside something else. Their address bar is not the
     learner's, and must never be written. */
  const params = () => new URLSearchParams(global.location?.search || "");
  const isEmbedded = () => params().get("solo") === "1" || params().get("embed") === "1";

  const urlFor = (path) => {
    const keep = new URLSearchParams();
    const now = params();
    KEEP_PARAMS.forEach((k) => { if (now.has(k)) keep.set(k, now.get(k)); });
    const q = keep.toString();
    return `/${path}${q ? `?${q}` : ""}`;
  };

  // Is the address bar already a link to this place?
  const isCurrent = (route) => {
    const here = parse(global.location.pathname);
    if (!here) return route.tab === landingTab && !route.anchor && !route.arena;
    return pathFor(here) === pathFor(route);
  };

  const write = (route, { replace = false } = {}) => {
    if (isEmbedded() || !route) return;
    if (!replace && isCurrent(route)) return;
    const url = urlFor(pathFor(route));
    const state = { dd: true, tab: route.tab, anchor: route.anchor || "", arena: route.arena || "" };
    try {
      if (replace) global.history.replaceState(state, "", url);
      else global.history.pushState(state, "", url);
    } catch (_) { /* opaque origin / sandboxed frame: the app still works */ }
  };

  // ---- opening a place --------------------------------------------------
  const topbarHeight = () =>
    parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--dd-topbar-h")) || 44;

  /* Open the About article holding the target, then bring the target on
     screen. Same two steps as account-menu.js's openDisclosure and
     aisc/articles.js's link handler; the scroll waits a frame because a
     just-unhidden element has no box yet. An article opens at the top of the
     page, a section scrolls to itself. */
  const openAnchor = (anchor, { smooth = true } = {}) => {
    const el = resolveAnchor(anchor);
    if (!el) return null;
    if (el.tagName === "DETAILS") el.open = true;
    const article = global.AISCArticles?.openFor?.(el) || null;
    requestAnimationFrame(() => {
      const behavior = smooth ? "smooth" : "auto";
      if (article === el) global.scrollTo({ top: 0, behavior });
      else el.scrollIntoView({ block: "start", behavior });
    });
    return el;
  };

  // The About page can be swapped for the owner's saved copy, and its article
  // switcher is a deferred script, so anything aimed inside it waits for both.
  const whenAboutReady = (fn) => {
    const run = () => {
      const ready = global.DDAboutContentReady;
      if (ready && typeof ready.finally === "function") ready.finally(fn);
      else fn();
    };
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", run, { once: true });
    else run();
  };

  const whenLoaded = (fn) => {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn, { once: true });
    else fn();
  };

  /* A section this build cannot open (a stale or mistyped /arena-9-9) lands
     on the section list, and REPLACES the entry: pushing /courses on top of it
     would make Back return to the bad link, which pushes again, forever. */
  const openArena = (slug) => {
    whenLoaded(() => {
      if (global.ArenaNotebook?.canOpen?.(slug)) {
        global.ArenaNotebook.open(slug);
        return;
      }
      if (typeof switchTab !== "function") return;
      applying = true;
      try { switchTab("courses"); } finally { applying = false; }
      write({ tab: "courses" }, { replace: true });
    });
  };

  /* Put the app at `route` without writing history (the caller decides). */
  const show = (route, { smooth = true } = {}) => {
    if (typeof switchTab !== "function") return;
    applying = true;
    try {
      switchTab(route.tab);
    } finally {
      applying = false;
    }
    if (route.anchor) whenAboutReady(() => openAnchor(route.anchor, { smooth }));
    if (route.arena) openArena(route.arena);
  };

  // ---- the hooks other files call ---------------------------------------

  /* app.js, at the end of every switchTab. Writes the page's pathname unless
     this file is the one switching (it writes once, itself, afterwards). */
  const onSwitch = (tab) => {
    if (applying) return;
    const replace = replaceNext;
    replaceNext = false;
    if (tab === ARENA_TAB) {
      const arena = pendingArena || global.ArenaNotebook?.currentId?.() || "";
      pendingArena = "";
      write(arena ? { tab, arena } : { tab }, { replace });
      return;
    }
    write({ tab }, { replace });
  };

  // practice/arena-notebook-resume.js, just before Courses jumps to the
  // notebook you were last reading.
  const redirecting = () => { replaceNext = true; };

  /* aisc/articles.js (a #aisc-… link or the article switcher) and
     account-menu.js (the two About rows). `replace` when the same click has
     already written the page's own path via switchTab. */
  const onAbout = (id, opts) => {
    if (!id) return;
    write({ tab: ABOUT_TAB, anchor: id }, opts);
  };

  // practice/arena-notebook.js, before it switches to the notebook page.
  const onArena = (slug) => { pendingArena = slug || ""; };

  // ---- links, Back and Forward -----------------------------------------

  /* A root-relative link to a known route (`<a href="/knowledge-graph">`)
     moves the app in place instead of reloading it. `#aisc-…` links are
     aisc/articles.js's; anything with a modifier key, a target or a
     download is the browser's. */
  document.addEventListener("click", (e) => {
    if (e.defaultPrevented || e.button !== 0) return;
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const a = e.target.closest && e.target.closest("a[href]");
    if (!a || a.target || a.hasAttribute("download")) return;
    // The owner's About editor: clicking a link there places the caret.
    if (a.closest('[contenteditable="true"]')) return;
    const href = a.getAttribute("href") || "";
    if (!href.startsWith("/") || href.startsWith("//")) return;
    // A query string is an instruction for the page that loads it (?arena=,
    // ?lesson=, ?invite=), read once at boot — so that link really loads.
    if (/[?#]/.test(href)) return;
    // A chromeless embed hands a link to the full app, as a real navigation.
    if (isEmbedded()) return;
    const route = parse(href);
    if (!route || typeof switchTab !== "function") return;
    e.preventDefault();
    show(route);
    write(route);
  });

  global.addEventListener("popstate", () => {
    if (isEmbedded()) return;
    const route = parse(global.location.pathname) || (landingTab ? { tab: landingTab } : null);
    if (!route) return;
    /* Back onto /courses is "show me the list" — what "← The course" says —
       so the auto-resume must not bounce you back into the notebook. */
    if (route.tab === "courses" && typeof ArenaNotebookResume !== "undefined") {
      ArenaNotebookResume.suppress?.();
    }
    show(route);
  });

  // ---- boot (app.js) -----------------------------------------------------

  /* The page the address bar asks for, or "" for "/". Read by app.js before
     its first switchTab, and by switchTab's advanced-mode guard: a link is an
     explicit request for that page. */
  const read = () => parse(global.location?.pathname)?.tab || "";

  /* app.js calls this with the page it actually landed on. The switch has
     already happened; this only opens a place INSIDE the page and stamps the
     entry so Back can come back to it. */
  const boot = (landedTab) => {
    const route = parse(global.location?.pathname);
    if (!route || route.tab !== landedTab) {
      /* "/" (or a path that names nothing) showed the landing page. A path
         that names a DIFFERENT page lost to something above it in app.js's
         boot order (an invite, a test-user swap, the advanced-mode guard), so
         the address bar is corrected to the page actually on screen. */
      const named = !!route;
      landingTab = named ? "" : landedTab;
      if (!isEmbedded()) {
        try {
          // The query string stays: an ?invite= is read again when the join
          // actually happens (groups/groups_store.js).
          const url = named
            ? `/${pathFor({ tab: landedTab })}${global.location.search}${global.location.hash}`
            : global.location.href;
          global.history.replaceState({ dd: true, tab: landedTab }, "", url);
        } catch (_) {}
      }
      return;
    }
    if (route.anchor) whenAboutReady(() => openAnchor(route.anchor, { smooth: false }));
    if (route.arena) openArena(route.arena);
    if (!isEmbedded()) {
      try {
        global.history.replaceState(
          { dd: true, tab: route.tab, anchor: route.anchor || "", arena: route.arena || "" },
          "",
          global.location.href,
        );
      } catch (_) {}
    }
  };

  /* Suppression for app.js's boot switchTab: the address bar already says
     where we are, and boot() stamps it. */
  const booting = (fn) => {
    applying = true;
    try { return fn(); } finally { applying = false; }
  };

  global.DDDeepLink = Object.freeze({
    read, parse, pathFor, boot, booting, onSwitch, onAbout, onArena, redirecting,
  });
})(window);
