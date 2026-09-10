/* ================================================================
   LESSON-LINKS.JS — one shareable URL per lesson in the knowledge graph

   Every bubble on the Knowledge Graph tab is a lesson: a concept with its
   own teaching text, worked segments and "watch out" notes. Until now none
   of them had an address. You could not send someone a lesson, bookmark the
   one you are stuck on, or link to it from a notebook — the only way in was
   "open the graph, hunt for the bubble".

   So each concept gets a pathname built from its OWN id, with the dot that
   already separates family from concept becoming the slash:

       einops.repeat-model   ->  /einops/repeat-model
       numpy.broadcasting-rules -> /numpy/broadcasting-rules
       python.indexing       ->  /python/indexing

   Derived, not tabulated. A hand-written slug table is a second source of
   truth for a name the registry already owns, and it goes stale the first
   time a concept is added — which is exactly when a link is minted for it.
   The cost of deriving is that the URL inherits the id's wording, so
   `/einops/repeat-model` and not `/einops/repeat`. `kcFromPath` therefore
   also accepts the SHORTER form when it names exactly one concept in that
   family (see `resolve`), so a link typed by hand lands and is then
   rewritten to the canonical one.

   🔴 THE FAMILY LIST IS AN ALLOWLIST, and it is what keeps this from
   swallowing the whole URL space. Anything else with two path segments is
   not a lesson link and is left alone. concept-graph/watch.py asserts the
   list covers every family in lessons/kc_registry.json and collides with
   none of solo-route.js's page slugs — the two route readers share one
   pathname and must not both claim it.

   Where the pieces live:
     - app.js      asks `read()` at boot so a lesson URL lands on the graph
                   tab instead of the auth-aware default page, and tells us
                   when the learner leaves that tab.
     - lesson-graph.js calls `onSelect` / `onDeselect` and takes `pathFor`
                   for its prerequisite/unlocks chips. It is RED in Modulario,
                   so the URL contract lives HERE and it keeps the hooks.
     - This file mounts its own permalink control into `.kg2-info-head`
                   rather than adding markup to index.html.
   ================================================================ */

(function (global) {
  "use strict";

  const doc = global.document;

  /* Every family in lessons/kc_registry.json — the part of a KC id before the
     dot. Guarded by concept-graph/watch.py against the registry, so adding a
     concept in a NEW family fails the folder's watch instead of silently
     minting links that route nowhere. */
  const FAMILIES = Object.freeze(["einops", "numpy", "python", "raytracing", "torch"]);

  /* An embedded instance owns no address bar. The graph's own "Practice ⤢"
     overlay hosts `index.html?lesson=<kc>&embed=1` in an iframe, and the
     arena-book pages embed the app too; a pushState from in there would
     rewrite the HOST page's URL out from under a learner who never navigated.
     Links are still FORMATTED in an embed (the permalink is the one control
     that is more useful there, not less) — only the writing is off. */
  const isEmbedded = (() => {
    try {
      if (global.top !== global.self) return true;
    } catch (_) {
      return true;              // cross-origin parent: embedded by definition
    }
    return /(^|[?&])embed=1(&|$)/.test(String(global.location.search || ""));
  })();

  /* ---------------- the pathname <-> concept id mapping ------------------ */

  const familyOf = (kc) => String(kc || "").split(".")[0];

  /** "/einops/repeat-model" for a concept this file can route back, "" for
      anything else. Returning "" rather than a best guess is what lets every
      caller ask one question — "is there a link for this?" — instead of
      minting an href and discovering on the click that it goes nowhere. */
  const pathFor = (kc) => {
    const id = String(kc || "");
    const dot = id.indexOf(".");
    if (dot <= 0 || dot === id.length - 1) return "";
    if (FAMILIES.indexOf(id.slice(0, dot)) === -1) return "";
    if (id.indexOf("/") !== -1) return "";
    return "/" + id.slice(0, dot) + "/" + id.slice(dot + 1);
  };

  const urlFor = (kc) => {
    const path = pathFor(kc);
    return path ? global.location.origin + path : "";
  };

  /** The concept id a pathname NAMES — syntax only, no registry, so app.js can
      ask at boot without waiting on a fetch. The answer may be an alias
      (`einops.repeat`); `resolve` turns it into a real id. */
  const kcFromPath = (pathname) => {
    let raw = String(pathname == null ? global.location.pathname : pathname);
    raw = raw.replace(/\.html?$/i, "");
    let parts;
    try {
      parts = raw.split("/").filter(Boolean).map((s) => decodeURIComponent(s).toLowerCase());
    } catch (_) {
      return "";
    }
    // A single segment carrying the dot — /einops.repeat-model — is the id
    // itself pasted into the address bar. Cheap to accept and it is the form
    // every error message and log line in this app prints.
    if (parts.length === 1 && parts[0].indexOf(".") > 0) parts = parts[0].split(".");
    if (parts.length !== 2) return "";
    if (FAMILIES.indexOf(parts[0]) === -1) return "";
    if (!/^[a-z0-9][a-z0-9-]*$/.test(parts[1])) return "";
    return parts[0] + "." + parts[1];
  };

  const read = () => kcFromPath(global.location.pathname);

  /* ---------------- registry: alias -> canonical ------------------------- */

  let idsPromise = null;
  /* 🔴 A FAILED FETCH IS NOT AN EMPTY REGISTRY, and collapsing the two would
     make every link on a flaky connection report "no concept is called that" —
     an accusation against a URL that is in fact fine. It resolves to null,
     which `resolve` passes through as "cannot say", and the failure is NOT
     cached: the next call gets a real attempt rather than a memoised lie. */
  const loadIds = () => {
    if (!idsPromise) {
      idsPromise = fetch("lessons/kc_registry.json")
        .then((r) => r.json())
        .then((reg) => (reg && Array.isArray(reg.kcs) ? reg.kcs.map((k) => k.id) : null))
        .catch(() => null)
        .then((ids) => {
          if (!ids || !ids.length) idsPromise = null;
          return ids;
        });
    }
    return idsPromise;
  };

  /** Alias -> real concept id, or "" when the URL names nothing.

      Exact first. Then the SHORTER form: `/einops/repeat` for
      `einops.repeat-model`, accepted only when the prefix stops at a hyphen
      AND names exactly one concept in that family. The boundary and the
      uniqueness test are both load-bearing — `/numpy/random` prefixes three
      concepts (samplers, seeding, threading), and silently picking the first
      would hand a learner a lesson they did not ask for under a URL that
      looks deliberate. Ambiguous resolves to nothing and says so. */
  const resolve = (candidate) =>
    loadIds().then((ids) => {
      const id = String(candidate || "");
      if (!id) return "";
      if (!ids) return null;          // registry unavailable — see loadIds
      if (ids.indexOf(id) !== -1) return id;
      const dot = id.indexOf(".");
      if (dot <= 0) return "";
      const family = id.slice(0, dot);
      const rest = id.slice(dot + 1);
      const hits = ids.filter((known) => {
        if (known.slice(0, family.length + 1) !== family + ".") return false;
        const tail = known.slice(family.length + 1);
        return tail === rest || tail.indexOf(rest + "-") === 0;
      });
      return hits.length === 1 ? hits[0] : "";
    });

  /* ---------------- writing the address bar ------------------------------ */

  // The query string and fragment are carried through every rewrite. They are
  // not ours: `?invite=` is a group invitation groups_store.js clears only
  // once the join lands, and dropping it here would break the link a friend
  // sent by the act of clicking a bubble.
  const withCurrentQuery = (path) => path + (global.location.search || "") + (global.location.hash || "");

  /* 🔴 THE GRAPH IS NOT ALWAYS ON ITS OWN TAB. `.kg-container.kg2` is a single
     live element that two other surfaces BORROW by moving it into themselves —
     concept-graph/why-graph.js for the landing page's maximise, and
     instructor-review.js for the graph review door. Selecting a bubble in
     either of those is not a learner opening a lesson, and writing
     /einops/repeat-model from there would mean a reload dropped an instructor
     mid-review onto the Knowledge Graph tab. The permalink still appears in
     both (a reviewer wanting to link a concept is exactly who needs it); only
     the address bar is withheld. */
  const graphIsOnItsOwnTab = () => {
    const el = doc.querySelector(".kg-container.kg2");
    return !!(el && el.closest("#page-knowledge-graph"));
  };

  const currentPath = () => {
    const p = String(global.location.pathname || "").replace(/\.html?$/i, "");
    return p.length > 1 ? p.replace(/\/+$/, "") : p;
  };

  const setPath = (path, mode) => {
    if (isEmbedded || !global.history || !path) return;
    if (currentPath() === path) return;
    try {
      global.history[mode === "replace" ? "replaceState" : "pushState"](
        { ddLesson: kcFromPath(path) || null },
        "",
        withCurrentQuery(path),
      );
    } catch (_) { /* a sandboxed frame refuses history writes; the app is fine */ }
  };

  /* ---------------- the permalink control -------------------------------- */

  // Mounted from here, not from index.html, so the whole feature is one file
  // plus its hooks. `.kg2-info-head` is the graph's lesson-pane header; it is
  // re-parented wholesale into instructor review, and the anchor travels with
  // it because it is a child rather than a sibling looked up by id.
  let link = null;
  let copiedTimer = 0;

  const LABEL = "Link";
  const COPIED = "Copied ✓";

  const copy = (text) => {
    if (global.navigator && global.navigator.clipboard && global.navigator.clipboard.writeText) {
      return global.navigator.clipboard.writeText(text);
    }
    // http:// on the LAN has no clipboard API — Seth opens this app off a
    // local server as well as off Vercel, and the control has to work in both.
    return new Promise((resolve_, reject) => {
      try {
        const ta = doc.createElement("textarea");
        ta.value = text;
        ta.setAttribute("readonly", "");
        ta.style.cssText = "position:fixed;top:-1000px;opacity:0";
        doc.body.appendChild(ta);
        ta.select();
        const ok = doc.execCommand("copy");
        doc.body.removeChild(ta);
        ok ? resolve_() : reject(new Error("copy refused"));
      } catch (e) { reject(e); }
    });
  };

  const flash = (text) => {
    if (!link) return;
    link.textContent = text;
    link.classList.add("is-copied");
    global.clearTimeout(copiedTimer);
    copiedTimer = global.setTimeout(() => {
      if (!link) return;
      link.textContent = LABEL;
      link.classList.remove("is-copied");
    }, 1600);
  };

  const onLinkClick = (event) => {
    // A modifier or the middle button is the browser's own "open this
    // elsewhere", and the href is real precisely so those keep working. Only
    // the plain click is ours, and what a plain click on a permalink should do
    // is hand you the link — not navigate you to the page you are already on.
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    const href = link && link.href;
    if (!href) return;
    copy(href).then(() => flash(COPIED), () => flash("Press ⌘C"));
  };

  const mount = () => {
    if (link && link.isConnected) return link;
    const head = doc.querySelector(".kg2-info-head");
    if (!head) return null;
    // Re-attach the element we already built rather than minting a second one:
    // a detached-then-rebuilt control leaves its click listener on an orphan
    // and doubles up every time the head is re-parented.
    if (link) { head.insertBefore(link, head.querySelector("#kg-maximize") || null); return link; }
    link = doc.createElement("a");
    link.className = "kg2-permalink";
    link.id = "kg-permalink";
    link.hidden = true;
    link.textContent = LABEL;
    link.addEventListener("click", onLinkClick);
    const before = head.querySelector("#kg-maximize");
    head.insertBefore(link, before || null);
    return link;
  };

  const showPermalink = (kc) => {
    const el = mount();
    if (!el) return;
    const url = urlFor(kc);
    if (!url) { el.hidden = true; return; }
    el.href = url;
    el.hidden = false;
    el.title = "Copy a link to this lesson — " + url;
    el.textContent = LABEL;
    el.classList.remove("is-copied");
  };

  const hidePermalink = () => { if (link) link.hidden = true; };

  /* ---------------- hooks the graph calls -------------------------------- */

  // Called from lesson-graph.js's renderContent, i.e. every time a concept
  // becomes the selected one — a bubble tap, a chip, a jump from a practice
  // question, the next-concept offer after mastery. pushState rather than
  // replaceState so Back walks the concepts the learner walked; setPath is a
  // no-op when the path already matches, which is what keeps the boot link and
  // a popstate from each leaving a duplicate entry behind them.
  const onSelect = (kc) => {
    showPermalink(kc);
    if (!graphIsOnItsOwnTab()) return;
    setPath(pathFor(kc), "push");
  };

  // Background tap on the canvas: the learner deliberately dropped the
  // selection, so the address goes back to the map itself.
  const onDeselect = () => {
    hidePermalink();
    if (!graphIsOnItsOwnTab()) return;
    if (read()) setPath("/", "push");
  };

  // app.js, on every switchTab. A lesson path is an address for one bubble on
  // ONE tab; carrying it onto Practice would mean a reload landed the learner
  // back on the graph they had just left.
  const onTab = (tabName) => {
    if (tabName === "knowledge-graph") return;
    if (read()) setPath("/", "replace");
  };

  /* ---------------- arriving on a lesson URL ----------------------------- */

  /* 🔴 NO "we are routing, don't write" flag, and deliberately none. Every
     arrival here — a deep link, Back, Forward — happens with the address bar
     ALREADY on the concept's path, so when the selection lands and calls
     onSelect, `setPath` finds the path unchanged and does nothing. A flag
     would have to be released on a timer (the graph retries for seconds
     before it exists) and every bubble the learner tapped inside that window
     would silently stop updating the URL. Matching on the path is exact and
     has no window. */
  const focus = (kc) => {
    if (typeof global.deltaFocusConceptGraphKc === "function") global.deltaFocusConceptGraphKc(kc);
  };

  /* An address that names no concept is a typo, a rename, or a link written
     before the concept was retired. Say which — the alternative is a graph
     that opens on the whole map and looks like the link "worked", so nobody
     ever finds out the link is dead. Written straight into the lesson pane
     once the graph has finished building (its own setPlaceholder runs at the
     end of build and would otherwise wipe this). */
  const reportUnknown = (candidate) => {
    let tries = 0;
    const tick = () => {
      const body = doc.getElementById("kg-info-body");
      // The PLACEHOLDER, not merely a live Cytoscape: build() assigns `cy`
      // several steps before it calls setPlaceholder(), and a note written
      // into the gap is wiped by the placeholder that follows it.
      const settled = body && body.querySelector(".kg2-placeholder");
      if (!settled) {
        if (tries++ < 80) global.setTimeout(tick, 150);
        return;
      }
      const note = doc.createElement("p");
      note.className = "kg2-deadlink";
      note.textContent =
        'No concept is called "' + candidate + '", so this link opens the whole map. ' +
        "Pick a bubble to get a link that works.";
      body.insertBefore(note, body.firstChild);
    };
    global.setTimeout(tick, 150);
  };

  const openFromLocation = () => {
    const candidate = read();
    if (!candidate) return;
    resolve(candidate).then((kc) => {
      // null is "the registry did not load", "" is "no such concept". Only the
      // second one is the link's fault, and only the second one is reported.
      if (kc === null) return;
      if (!kc) { reportUnknown(candidate); return; }
      // Canonicalise BEFORE focusing: the selection that follows writes the
      // same path, sees no change and pushes nothing, so an alias link costs
      // no history entry.
      setPath(pathFor(kc), "replace");
      focus(kc);
    });
  };

  /* Back and forward. The graph is one page, so history motion is a selection
     change and never a page load — which is the whole reason onSelect pushes. */
  const onPopState = () => {
    const candidate = read();
    if (!candidate) {
      // onDeselect fires from the graph's own reset and re-reads the location,
      // which is already "/" by the time popstate runs — so it pushes nothing.
      if (typeof global.deltaClearConceptGraphSelection === "function") {
        global.deltaClearConceptGraphSelection();
      }
      hidePermalink();
      return;
    }
    resolve(candidate).then((kc) => { if (kc) focus(kc); });
  };

  if (!isEmbedded) global.addEventListener("popstate", onPopState);

  if (doc.readyState === "loading") {
    doc.addEventListener("DOMContentLoaded", openFromLocation);
  } else {
    openFromLocation();
  }

  global.DDLessonLinks = Object.freeze({
    FAMILIES,
    read,
    kcFromPath,
    resolve,
    pathFor,
    urlFor,
    onSelect,
    onDeselect,
    onTab,
  });
})(window);
