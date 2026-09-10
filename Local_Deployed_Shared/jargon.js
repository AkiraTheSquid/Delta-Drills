/* ================================================================
   JARGON LINKS — hover a course term in a lesson, get its definition,
   click through to the concept that teaches it.

   WHAT THIS IS
     The lessons lean on ~73 words that mean nothing until you have
     met them: view, broadcasting, keepdims, contracted axis. A
     learner meeting one on page 3 had no way back to the page that
     defined it. Every one of them now underlines in lesson prose;
     hovering opens a small panel with the definition and a button
     that opens the concept, maximized, in a new browser tab.

   THE ROUTE, END TO END
     hover a term
       -> panel: definition + "Take me to the lesson"
       -> window.open("index.html?kc=<id>&maximize=1")
       -> that tab lands on the Knowledge Graph, focuses the node,
          and clicks #kg-maximize for you: the lesson full-bleed
       -> the overlay's own "⤡ Minimize" drops back to the graph
          with the lesson in the right-hand pane.
     The last two steps are lesson-graph.js's, untouched. This file
     only presses the buttons a learner would otherwise press.

   🔴 WHY IT OWNS NOTHING
     Three sessions were mid-flight in index.html, practice/lessons.js
     and concept-graph/lesson-graph.js when this was written. It is
     deliberately built so it needs NO edit to any of them:

       - it decorates RENDERED DOM off a debounced MutationObserver,
         the same trick infotips.js uses, instead of hooking either
         markdown renderer;
       - it routes through exports that already exist and are already
         used by other callers — window.deltaFocusConceptGraphKc
         (practice/graph-jump.js, practice/stage-ladder.js),
         switchTab(), and the #kg-maximize button;
       - the one thing it cannot read from the DOM — which concept
         the learner is currently ON, so a term is not linked to the
         page it is already on — it gets by WRAPPING
         LessonGate.showLesson / maybeShow rather than editing them.

     index.html carries three additive lines (a stylesheet and two
     scripts) and nothing else.

   WHAT IT WILL NOT TOUCH
     Code. Never a <pre>, <code>, <kbd>, an <a>, a heading, a button,
     or a runnable notebook cell. A term inside a code fence is a
     Python identifier, and underlining it there would suggest you
     could click your own program. Decoration is text-node only:
     nothing is re-parsed, so notebook.js still finds the same
     `pre > code` blocks it turns into cells.
   ================================================================ */

(function initJargon() {
  const GLOSSARY = window.DD_GLOSSARY || null;
  if (!GLOSSARY || !Array.isArray(GLOSSARY.terms) || !GLOSSARY.terms.length) return;

  /* The prose regions of a lesson, in both places a lesson is drawn: the
     practice screen (practice/lessons.js) and the concept graph's right-hand
     pane (concept-graph/lesson-graph.js). Deliberately NOT the whole page —
     question text, the ladder and the editor are not lessons, and a definition
     popup over a question you are being graded on is a hint. */
  const SCOPES = [
    ".lesson-body",
    ".lesson-worked",
    ".lesson-watch-out",
    ".kg2-concept",
    ".kg2-worked",
    ".kg2-watch",
  ].join(",");

  /* Where a term must stay a plain word. `.dd-jargon` is here so a rescan
     cannot decorate its own output — that, plus the `mutating` latch below, is
     what keeps the observer from feeding itself. */
  const SKIP = [
    "code", "pre", "kbd", "samp", "script", "style", "textarea",
    "a", "button", "h1", "h2", "h3", "h4", "h5", "h6",
    ".dd-jargon", ".dd-jargon-pop", ".nb-cell", ".katex",
  ].join(",");

  const MAX_LINKS_PER_SCOPE = 12;   // a page, not a dictionary
  const HOVER_IN_MS = 110;
  const HOVER_OUT_MS = 180;         // long enough to walk the pointer into the panel

  /* ---- the match index -------------------------------------------- */

  // form (lower case) -> term record. One entry per surface form.
  const byForm = new Map();
  GLOSSARY.terms.forEach((rec) => {
    if (!rec || !rec.term || !rec.kc) return;
    [rec.term].concat(rec.aliases || []).forEach((form) => {
      const key = String(form || "").trim().toLowerCase();
      if (key) byForm.set(key, rec);
    });
  });
  if (!byForm.size) return;

  const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  /* Longest form first so "boolean mask" wins over "mask" and "one-hot
     encoding" over "one-hot" — alternation is first-match-wins, so sorting IS
     the disambiguation rule.

     The boundaries are \w-and-hyphen classes rather than \b: \b treats the
     hyphen in "top-k" and the dot in "einops.reduce" as boundaries, so \btop\b
     would light up the "top" inside "top-k" and leave a stray "-k" behind. */
  const forms = Array.from(byForm.keys()).sort((a, b) => b.length - a.length);
  const RE = new RegExp(
    "(?<![\\w-])(" + forms.map(escapeRe).join("|") + ")(?![\\w-])",
    "gi",
  );

  /* ---- which concept the learner is already on --------------------- */

  /* A term is not linked inside the lesson that teaches it: clicking through
     to the page you are reading is a dead end, and the underline reads as an
     offer of something new. Two surfaces, two ways to know:

       the graph pane — #kg-maximize's dataset.kc, which lesson-graph.js sets
         on every renderContent, so it is always the node on screen;
       the practice screen — no marker in the DOM at all, so LessonGate's two
         entry points are wrapped below to record what they were asked for. */
  let currentKc = null;
  try {
    const q = new URLSearchParams(location.search).get("lesson");
    if (q) currentKc = q;
  } catch (_) { /* no search params: leave it null */ }

  const _wrapLessonGate = () => {
    const gate = window.LessonGate;
    if (!gate || gate.__ddJargonWrapped) return;
    const wrap = (name, kcArgIndex) => {
      const original = gate[name];
      if (typeof original !== "function") return;
      gate[name] = function (...args) {
        const arg = args[kcArgIndex];
        // showLesson takes one kc; maybeShow takes the list.
        const kc = Array.isArray(arg) ? arg[0] : arg;
        if (typeof kc === "string" && kc) { currentKc = kc; scheduleScan(); }
        return original.apply(this, args);
      };
    };
    wrap("showLesson", 0);
    wrap("maybeShow", 2);
    gate.__ddJargonWrapped = true;
  };

  const _selfKc = (root) => {
    // Inside the graph's right-hand pane the selected node is authoritative;
    // everywhere else the lesson the gate was asked to show is.
    if (root.closest && root.closest("#kg-info-body")) {
      const btn = document.getElementById("kg-maximize");
      const kc = btn && btn.dataset ? btn.dataset.kc : "";
      if (kc) return kc;
    }
    return currentKc;
  };

  /* ---- decoration -------------------------------------------------- */

  let mutating = false;
  let scanQueued = false;

  const _skip = (node) => {
    const el = node.parentElement;
    return !el || el.closest(SKIP);
  };

  /* Rewrites ONE text node into [text, <span>, text, …]. Returns how many
     links it made so the caller can stop at the per-scope cap. */
  const _decorateTextNode = (node, used, selfKc, budget) => {
    const text = node.nodeValue;
    if (!text || text.length < 3) return 0;
    RE.lastIndex = 0;
    if (!RE.test(text)) return 0;
    RE.lastIndex = 0;

    const frag = document.createDocumentFragment();
    let cursor = 0;
    let made = 0;
    let m;
    while ((m = RE.exec(text)) !== null) {
      if (made >= budget) break;
      const rec = byForm.get(m[0].toLowerCase());
      // Only the FIRST mention of a concept in a section is linked. Ten
      // underlines on ten "tensor"s is wallpaper; one is a signpost.
      if (!rec || used.has(rec.kc) || rec.kc === selfKc) continue;
      used.add(rec.kc);
      if (m.index > cursor) frag.appendChild(document.createTextNode(text.slice(cursor, m.index)));
      const span = document.createElement("span");
      span.className = "dd-jargon";
      span.dataset.jargonKc = rec.kc;
      span.setAttribute("role", "link");
      span.setAttribute("tabindex", "0");
      span.setAttribute("aria-label", rec.term + " — what this means");
      span.setAttribute("aria-expanded", "false");
      span.setAttribute("aria-controls", "dd-jargon-pop");
      span.textContent = m[0];   // the learner's own spelling, not the canonical one
      frag.appendChild(span);
      cursor = m.index + m[0].length;
      made += 1;
    }
    if (!made) return 0;
    if (cursor < text.length) frag.appendChild(document.createTextNode(text.slice(cursor)));
    node.parentNode.replaceChild(frag, node);
    return made;
  };

  /* Unwrap every link in a region and glue the text back together. Re-running
     decoration over prose that is ALREADY decorated is not idempotent — the
     existing spans are in SKIP, so `used` starts empty and the SECOND mention
     of a concept gets linked where the first one already is. Measured: a
     forced rescan walked one lesson 18 -> 20 -> 21 -> 22 links, with 7 repeats
     of the same concept. Undecorating first is what makes the one-link-per-
     concept rule hold no matter how decoration is re-entered. */
  const _undecorate = (root) => {
    root.querySelectorAll(".dd-jargon").forEach((el) => {
      el.parentNode.replaceChild(document.createTextNode(el.textContent), el);
    });
    // Without this the unwrapped halves stay separate text nodes, and a term
    // that spans the seam would never match again.
    root.normalize();
  };

  const _decorateScope = (root) => {
    if (!root || root.dataset.ddJargon === "done") return;
    const selfKc = _selfKc(root);
    // Belt and braces: if anything ever hands this function a region that is
    // already partly decorated, those concepts are spent before it starts.
    const used = new Set(
      [...root.querySelectorAll(".dd-jargon")].map((el) => el.dataset.jargonKc),
    );
    let budget = MAX_LINKS_PER_SCOPE;
    // Collected up front: replaceChild mutates the tree the walker is walking.
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) =>
        (n.nodeValue && n.nodeValue.trim() && !_skip(n))
          ? NodeFilter.FILTER_ACCEPT
          : NodeFilter.FILTER_REJECT,
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach((n) => {
      if (budget <= 0) return;
      if (!n.parentNode) return;   // an earlier replacement detached it
      budget -= _decorateTextNode(n, used, selfKc, budget);
    });
    root.dataset.ddJargon = "done";
  };

  const scan = () => {
    scanQueued = false;
    mutating = true;
    try {
      document.querySelectorAll(SCOPES).forEach(_decorateScope);
    } finally {
      /* 🔴 Drop the records OUR OWN decoration just generated, before the
         observer's callback can see them. `mutating` cannot do this job on its
         own: observer callbacks are delivered as microtasks, i.e. always after
         this synchronous block has already set the latch back to false. Once
         the observer started invalidating scopes (below), that gap became a
         live loop — decorate, get called back, invalidate, decorate again. */
      if (observer) observer.takeRecords();
      mutating = false;
    }
  };

  /* Half this app's lesson DOM is written at runtime and replaced mid-session
     (the gate re-renders per page; the graph pane re-renders per node), so
     scopes are re-derived from the DOM rather than decorated once at load.
     `data-dd-jargon="done"` on the scope is what makes a rescan cheap: a
     re-rendered section arrives without it and is picked up; an untouched one
     is skipped whole. */
  const scheduleScan = () => {
    if (scanQueued) return;
    scanQueued = true;
    requestAnimationFrame(() => setTimeout(scan, 30));
  };

  let observer = null;

  const observe = () => {
    observer = new MutationObserver((records) => {
      if (mutating) return;
      let touched = false;
      for (const r of records) {
        if (r.type !== "childList") continue;
        if (!r.addedNodes.length && !r.removedNodes.length) continue;
        touched = true;
        /* A scope that is re-RENDERED arrives as a fresh element with no
           done-marker, so it is picked up anyway. A scope mutated IN PLACE
           does not: practice/notebook.js rewrites blocks inside .lesson-body
           long after it was decorated, and the marker made every one of those
           regions permanently invisible to decoration. Clear the marker on the
           scope the mutation landed in, and let the (now idempotent) rescan
           redo it. */
        const scope = r.target && r.target.closest ? r.target.closest(SCOPES) : null;
        if (scope) delete scope.dataset.ddJargon;
      }
      if (touched) scheduleScan();
    });
    observer.observe(document.body, { childList: true, subtree: true });
  };

  /* ---- the panel --------------------------------------------------- */

  let pop = null;
  let popTitle = null;
  let popBody = null;
  let popWhere = null;
  let popGo = null;
  let openFor = null;      // the .dd-jargon element the panel belongs to
  let inTimer = null;
  let outTimer = null;
  let restoreFocusTo = null;   // set only when the keyboard opened the panel

  const esc = (s) => {
    const d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  };

  const buildPop = () => {
    if (pop) return pop;
    pop = document.createElement("div");
    pop.className = "dd-jargon-pop";
    pop.id = "dd-jargon-pop";
    pop.setAttribute("role", "dialog");
    pop.hidden = true;
    pop.innerHTML =
      '<div class="dd-jargon-pop-term"></div>' +
      '<div class="dd-jargon-pop-def"></div>' +
      '<div class="dd-jargon-pop-where"></div>' +
      '<button type="button" class="dd-jargon-pop-go">Take me to the lesson ↗</button>';
    popTitle = pop.querySelector(".dd-jargon-pop-term");
    popBody = pop.querySelector(".dd-jargon-pop-def");
    popWhere = pop.querySelector(".dd-jargon-pop-where");
    popGo = pop.querySelector(".dd-jargon-pop-go");
    popGo.addEventListener("click", () => {
      if (openFor) openLesson(openFor.dataset.jargonKc);
    });
    // The pointer has to be able to LEAVE the word and land on the button
    // without the panel closing under it — that is the whole reason both the
    // word and the panel share one grace timer.
    pop.addEventListener("mouseenter", () => clearTimeout(outTimer));
    pop.addEventListener("mouseleave", () => scheduleClose());
    mutating = true;
    document.body.appendChild(pop);
    mutating = false;
    return pop;
  };

  const place = (anchor) => {
    const r = anchor.getBoundingClientRect();
    const pr = pop.getBoundingClientRect();
    const margin = 8;
    let left = r.left + r.width / 2 - pr.width / 2;
    left = Math.max(margin, Math.min(left, window.innerWidth - pr.width - margin));
    // Below the word by default; above it when there is no room below, so a
    // term near the fold does not open a panel the learner has to scroll to.
    let top = r.bottom + 10;
    if (top + pr.height > window.innerHeight - margin) {
      const above = r.top - pr.height - 10;
      if (above > margin) top = above;
      else top = Math.max(margin, window.innerHeight - pr.height - margin);
    }
    // Viewport coordinates, no scroll offset: the panel is position: fixed
    // (styles/jargon.css) so it can escape the overflow:hidden ancestors
    // between it and the word. Adding scrollY here would push it off screen
    // by exactly the scroll distance on any scrolled lesson.
    pop.style.left = Math.round(left) + "px";
    pop.style.top = Math.round(top) + "px";
  };

  const openPop = (el, { fromKeyboard = false } = {}) => {
    const kc = el.dataset.jargonKc;
    const rec = GLOSSARY.terms.find((t) => t.kc === kc);
    if (!rec) return;
    buildPop();
    // Moving the pointer straight from one term to the next never closes the
    // first: without this the word left behind keeps the open styling while
    // the panel belongs to a different word entirely.
    if (openFor && openFor !== el) {
      openFor.classList.remove("dd-jargon-open");
      openFor.setAttribute("aria-expanded", "false");
    }
    openFor = el;
    popTitle.textContent = rec.term;
    popBody.textContent = rec.def || "";
    const where = (GLOSSARY.kcLesson || {})[kc];
    popWhere.innerHTML = where
      ? "Taught in <strong>" + esc(where[0]) + "</strong> — " + esc(where[1])
      : "";
    popWhere.hidden = !where;
    pop.hidden = false;
    place(el);
    el.classList.add("dd-jargon-open");
    el.setAttribute("aria-expanded", "true");
    /* Opened from the keyboard, the panel is appended to <body>, so the next
       Tab from the word walks the rest of the DOCUMENT before reaching the
       button — the one control the panel exists to offer. Move focus onto it,
       and closePop puts focus back on the word. */
    if (fromKeyboard) {
      restoreFocusTo = el;
      popGo.focus();
    } else {
      restoreFocusTo = null;
    }
  };

  const closePop = () => {
    clearTimeout(inTimer);
    clearTimeout(outTimer);
    if (openFor) {
      openFor.classList.remove("dd-jargon-open");
      openFor.setAttribute("aria-expanded", "false");
    }
    openFor = null;
    if (pop) pop.hidden = true;
    // Only when WE took the focus. Stealing it back after a plain hover-close
    // would yank the caret out of whatever the learner was actually doing.
    if (restoreFocusTo) {
      const back = restoreFocusTo;
      restoreFocusTo = null;
      try { back.focus(); } catch (_) { /* detached by a re-render */ }
    }
  };

  const scheduleClose = () => {
    clearTimeout(outTimer);
    outTimer = setTimeout(closePop, HOVER_OUT_MS);
  };

  /* ---- opening the lesson ------------------------------------------ */

  /* A NEW BROWSER TAB, on purpose. The learner is mid-lesson; sending THIS tab
     to the graph would lose their place and their draft. The new tab does the
     landing itself (see the ?kc= route below) rather than being handed a
     pre-built view, so the URL is shareable and survives a reload. */
  const openLesson = (kc) => {
    if (!kc) return;
    closePop();
    const url = "index.html?kc=" + encodeURIComponent(kc) + "&maximize=1";
    /* 🔴 NO "noopener" IN THE FEATURE STRING. A browser that honours it returns
       null from a window.open that SUCCEEDED, which is indistinguishable from
       the null a blocked popup returns — so the fallback below fired on the
       happy path and sent THIS tab to the lesson as well. The learner lost the
       page and the draft the new tab was supposed to protect, which is the
       exact failure this whole route exists to avoid. Opening without the
       feature and severing `opener` afterwards gives the same isolation and a
       return value that actually means something. */
    const win = window.open(url, "_blank");
    if (win) {
      try { win.opener = null; } catch (_) { /* cross-origin: already severed */ }
      return;
    }
    // Genuinely blocked: a button that does nothing at all is worse than a
    // navigation the learner did ask for.
    location.href = url;
  };

  /* ---- events ------------------------------------------------------- */

  const bind = () => {
    document.addEventListener("mouseover", (e) => {
      const el = e.target.closest ? e.target.closest(".dd-jargon") : null;
      if (!el) return;
      clearTimeout(outTimer);
      if (openFor === el) return;
      clearTimeout(inTimer);
      inTimer = setTimeout(() => openPop(el), HOVER_IN_MS);
    });
    document.addEventListener("mouseout", (e) => {
      const el = e.target.closest ? e.target.closest(".dd-jargon") : null;
      if (!el) return;
      clearTimeout(inTimer);
      scheduleClose();
    });
    // Touch and keyboard have no hover. A tap opens the panel (and a second tap
    // on the same word closes it); Enter/Space does the same from the keyboard.
    document.addEventListener("click", (e) => {
      const el = e.target.closest ? e.target.closest(".dd-jargon") : null;
      if (el) {
        e.preventDefault();
        if (openFor === el) closePop();
        else openPop(el);
        return;
      }
      if (pop && !pop.hidden && !(e.target.closest && e.target.closest(".dd-jargon-pop"))) closePop();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && pop && !pop.hidden) { closePop(); return; }
      const el = document.activeElement;
      if ((e.key === "Enter" || e.key === " ") && el && el.classList && el.classList.contains("dd-jargon")) {
        e.preventDefault();
        openPop(el, { fromKeyboard: true });
      }
    });
    // A panel pinned to a word that has scrolled away is worse than no panel.
    window.addEventListener("scroll", () => { if (pop && !pop.hidden) closePop(); }, true);
    window.addEventListener("resize", () => { if (openFor) place(openFor); });
  };

  /* ---- the ?kc= landing --------------------------------------------
     The other half of "Take me to the lesson": this runs in the NEW tab.
     Show the graph, focus the node, and — with maximize=1 — press the pane's
     own Maximize for the learner, so the tab opens ON the lesson. Minimizing
     is then lesson-graph.js's ⤡ button, which leaves them exactly where Seth
     asked: the graph, with the lesson in the right-hand pane.

     It polls rather than firing once because the graph is built lazily: the
     page has to be visible before Cytoscape can size itself, and
     deltaFocusConceptGraphKc kicks off a build that may still be running when
     it returns. #kg-maximize carrying THIS kc is the ready signal —
     lesson-graph.js sets it in renderContent, i.e. only once the node is
     selected and its lesson is on screen. */
  const route = () => {
    let kc = null;
    let wantMax = false;
    try {
      const p = new URLSearchParams(location.search);
      kc = p.get("kc");
      wantMax = p.get("maximize") === "1";
    } catch (_) { return; }
    if (!kc) return;

    const focus = () => {
      if (typeof switchTab === "function") switchTab("knowledge-graph");
      if (typeof window.deltaFocusConceptGraphKc === "function") {
        window.deltaFocusConceptGraphKc(kc);
      }
    };

    let tries = 0;
    const settle = () => {
      focus();
      if (!wantMax) return;
      const btn = document.getElementById("kg-maximize");
      if (btn && !btn.hidden && btn.dataset.kc === kc) { btn.click(); return; }
      // ~12s of 150ms ticks: a cold first visit builds the graph, reads the
      // lattice and waits on the network before the button appears.
      if (tries++ < 80) setTimeout(settle, 150);
    };
    setTimeout(settle, 350);
  };

  /* ---- boot ---------------------------------------------------------- */

  const start = () => {
    _wrapLessonGate();
    bind();
    observe();
    scheduleScan();
    route();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }

  // For watch_jargon.py and for a console poke while authoring.
  window.DDJargon = {
    rescan: () => {
      mutating = true;
      document.querySelectorAll(SCOPES).forEach((r) => {
        _undecorate(r);
        delete r.dataset.ddJargon;
      });
      mutating = false;
      scan();
    },
    open: openLesson,
    termCount: byForm.size,
  };
})();
