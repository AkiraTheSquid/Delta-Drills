/* ================================================================
   ARENA NOTEBOOK NAV — the left-edge contents rail
   ================================================================

   WHAT THIS REPLACES

   The ARENA notebook toolbar used to carry a `<select class="nbv-toc">`:
   "Jump to…", one <option> per heading. A dropdown is a list you cannot see —
   it costs a click to learn what is in the notebook, it says nothing about
   where you are in it, and at 161 headings (arena-1-4-2) it is a scrolling
   native menu.

   Seth, 2026-09-02, pointing at LessWrong: the rail on the left edge that is
   TICKS when your mouse is elsewhere and TITLES when it is over the gutter.

   🔴 NO CODE OF THEIRS IS HERE, BUT THE STRUCTURE IS MEASURED, NOT GUESSED.
   LessWrong's implementation (ForumMagnum/packages/lesswrong/components/posts/
   TableOfContents/) is React 19 + Next 16 + JSS + Apollo, and the repo is
   GPL-3.0. Neither the code nor the licence can come into this app — it is 103
   classic <script> tags and no build step. And there is nothing to copy even if
   there were: their styling is a TypeScript object literal compiled to class
   names at runtime, so that repository contains no HTML and no CSS at all.

   What their BROWSER produces does. `tools/visual-diff/dom_clone.py lw` walks
   the rendered rail and writes every node's computed style out as an ordinary
   .html and .css file; `./dom_clone.py --diff` then holds this rail against it
   property by property. The element-for-element correspondence is drawn at the
   top of arena-notebook-nav.css. Four behaviours come out of it:

     1. A dot per heading, and each row's share of the rail is that section's
        share of the document's height. That is what makes the collapsed rail a
        MAP rather than a list: a long section is a long gap.
     2. The current section is the last heading above the 1/5-viewport mark —
        not the topmost visible one, which flickers between two headings every
        time a heading sits near the fold.
     3. A 1px progress line down the left, split into a read part and an unread
        one, with a window marker at the join.
     4. Hovering the gutter fades the titles in; leaving it fades them out.

   🔴 A ROW NEVER SHRINKS, AND THAT IS DELIBERATE. Their rows are the same
   height open and closed — only the labels' opacity changes — so the collapsed
   rail is a PROPORTIONAL column of dots. When the headings outrun the window
   the rows box simply scrolls, which is what a long post of theirs does. The
   earlier version here squashed rows to a 9px floor to make them fit, and the
   result was an evenly-spaced list carrying no information about the document.

   MOUNTING. The rail is `position: fixed` but lives INSIDE
   #page-arena-notebook, which is how it disappears when you switch tabs:
   app.js hides a page with `.hidden { display: none }` (components.css) and a
   fixed child of a display:none ancestor is not rendered. A rail parked on
   <body> would have hung over the Practice page.
   ================================================================ */

const ArenaNotebookNav = (() => {
  /* Where "the current section" is measured. A FIFTH of the way down the
     window — read off getCurrentSectionMark() in their scrollUtils, whose own
     neighbouring comment says a third and is wrong about its own code. The
     reason for the rule either way: a heading you have just scrolled past
     should stay current until its successor is properly on
     screen. */
  const MARK = () => window.innerHeight / 5;
  /* What a jump target has to clear: the app topbar (44px, --dd-topbar-h) plus
     the notebook's own sticky toolbar. Scrolling a heading to y=0 would park it
     under both of them. */
  const JUMP_CLEARANCE = 96;

  /* Live state for the ONE rail on screen. Torn down and rebuilt per notebook
     rather than diffed: a notebook switch changes every heading anyway. */
  let rail = null;
  let list = null;
  let fill = null;
  let rest = null;
  let entries = []; // { el (heading), row, top }
  let contents = null; // the .nbv-cells element the rail describes
  let currentRow = null;
  /* The empty row standing for the run-up before the first heading. It carries
     no dot and no label: its only job is to take its share of the flex so the
     first dot lands level with the first heading. */
  let spacer = null;
  let ticking = false;

  const esc = (value) =>
    String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

  /* ---------- measuring ------------------------------------------------- */

  const _docTop = (el) => el.getBoundingClientRect().top + window.scrollY;

  /* Heading offsets, in document space. Recomputed rather than cached across
     events because the things that move a heading — a window resize, an ARENA
     <details> opening, an image finally loading — do not announce themselves
     with the new number. */
  const _measure = () => {
    entries.forEach((entry) => {
      entry.top = _docTop(entry.el);
    });
  };

  /* Each row's flex-grow: the fraction of the notebook this section spans.
     The last one runs to the bottom of the cells, not to the bottom of the
     document, so the app's page padding does not read as a final section. */
  const _rescale = () => {
    if (!rail || !contents || !entries.length) return;
    const top = _docTop(contents);
    const bottom = top + contents.offsetHeight;
    const span = Math.max(1, bottom - top);
    entries.forEach((entry, i) => {
      const next = i + 1 < entries.length ? entries[i + 1].top : bottom;
      /* `flex`, not `flex-grow`: the shorthand also sets basis 0, so the share
         is the WHOLE of the row's height rather than an addition to its
         content. Their rowWrapper computes to exactly this — `flex-grow: <the
         section's percentage>; flex-basis: 0%` — and its own min-height stays
         `auto`, which is the floor that keeps a one-line label legible when its
         section is a short one. */
      entry.row.style.flex = String(Math.max(0, (next - entry.top) / span) * 100);
    });
    /* The gap above the first heading is a row of its own, so the first tick
       sits level with the first heading instead of at the top of the rail.
       Theirs appears at the same threshold: a gap of more than 50px. */
    if (spacer) {
      const gap = entries[0].top - top;
      spacer.style.flex = String(gap > 50 ? (gap / span) * 100 : 0);
    }
  };

  /* The rail has no second mode, and theirs has none either. Rows keep their
     content height as a floor; when the sum of those floors passes the height
     of the rail there is no free space left to distribute, the shares stop
     having any visible effect, and the rows box scrolls instead. That handover
     is the whole of the long-notebook story — there is no dense mode to enter
     and no type to shrink. */

  /* ---------- keeping the current row reachable -------------------------- */

  /* The rows box scrolls whenever its contents outrun it — which is most long
     notebooks with the rail open, and only enormous ones with it closed. When
     it does, the highlighted row has to be brought back into it.

     🔴 BUT NOT WHILE THE READER IS SCROLLING IT BY HAND. Turning the wheel over
     the contents is a deliberate act — looking ahead at what is coming — and a
     follow that yanks the list back to the current section makes that
     impossible. Any scroll of the list that this function did not cause hands
     control over for HAND_BACK ms.

     Moved by assignment rather than scrollIntoView: that method walks every
     scrollable ancestor, and the window is one of them — it would fight the
     page scroll that triggered the highlight in the first place. */
  const HAND_BACK = 2500;
  let handOffUntil = 0;
  let selfScrollAt = 0;

  const _listTouched = () => {
    // A scroll this module just performed is not the reader taking over.
    if (Date.now() - selfScrollAt < 120) return;
    handOffUntil = Date.now() + HAND_BACK;
  };

  const _follow = (row) => {
    if (!list || !row) return;
    if (list.scrollHeight <= list.clientHeight + 1) return;
    if (Date.now() < handOffUntil) return;
    const target = row.offsetTop - list.clientHeight / 2;
    if (Math.abs(list.scrollTop - target) <= 8) return;
    selfScrollAt = Date.now();
    list.scrollTop = Math.max(0, target);
  };

  /* ---------- the current section --------------------------------------- */

  const _highlight = () => {
    const mark = window.scrollY + MARK();
    /* The LAST heading above the mark, with no early exit. The list is in
       document order and its offsets are normally monotonic, but "normally" is
       not a guarantee this loop needs to depend on — one mismeasured row
       (a heading that was hidden when it was measured) would otherwise stop
       the scan and freeze the highlight on it. */
    let found = null;
    for (const entry of entries) {
      if (entry.top <= mark) found = entry;
    }
    const row = found ? found.row : null;
    if (row === currentRow) return;
    if (currentRow) currentRow.classList.remove("is-current");
    currentRow = row;
    if (!row) return;
    row.classList.add("is-current");
    _follow(row);
  };

  /* How far through the NOTEBOOK you are, and how much of it your window is
     showing. Both measured the way theirs are (usePostReadProgress):

       progress   the bottom of the viewport, as a percentage of the content
                  box — so a window that already covers half a short notebook
                  starts at 50%, not at 0%
       window     the visible fraction of the content, drawn to scale on the
                  rail and never smaller than 10px

     The window marker is the darker segment sitting at the end of the filled
     part of the line. It is the piece that makes the rail a scrollbar for the
     document rather than a decoration. */
  const _progress = () => {
    if (!fill || !contents) return;
    const top = _docTop(contents);
    const height = Math.max(1, contents.offsetHeight);
    const pct = Math.min(100, Math.max(0, ((window.scrollY + window.innerHeight - top) / height) * 100));
    /* 🔴 TWO SIBLINGS SHARING A BASIS, NOT ONE ABSOLUTE BOX. Theirs splits the
       rule into a read part and an unread one and gives each its percentage as
       `flex-basis`; the fill is then an ordinary flex item, which is what lets
       its `::after` bottom-align inside it and become the window marker. An
       absolutely-positioned fill has no flex line to align anything to. */
    fill.style.flexBasis = `${pct}%`;
    if (rest) rest.style.flexBasis = `${100 - pct}%`;
    const track = fill.parentElement;
    if (track) {
      const railHeight = track.getBoundingClientRect().height;
      const rect = contents.getBoundingClientRect();
      const visible = Math.max(
        0,
        Math.min(window.innerHeight, rect.bottom) - Math.max(0, rect.top),
      );
      const marker =
        rect.height <= window.innerHeight
          ? railHeight
          : Math.max(10, (visible / Math.max(1, rect.height)) * railHeight);
      rail.style.setProperty("--anb-window", `${Math.round(marker)}px`);
    }
  };

  /* One rAF-coalesced pass per frame. A scroll event fires far more often than
     the screen repaints, and this reads layout. */
  const _onScroll = () => {
    if (ticking || !_laidOut()) return;
    ticking = true;
    requestAnimationFrame(() => {
      ticking = false;
      _highlight();
      _progress();
    });
  };

  /* 🔴 NOTHING MEASURES WHILE THE TAB IS AWAY. The rail is inside a page
     app.js hides with `display: none`, where clientHeight is 0 and every
     heading offset is 0 — a resize handled in that state would decide the
     notebook is dense (0 room) and hold that verdict until the next mount. The
     reopen path calls refresh() precisely so the measuring happens once the
     page is back on screen. */
  /* 🔴 `getClientRects().length`, NOT `offsetParent`. The rail is
     `position: fixed`, and a fixed element's offsetParent is null even when it
     is perfectly visible — the usual display:none test would have switched the
     whole rail off. An element in a display:none subtree generates no boxes,
     so it has no client rects; a visible one has exactly one. */
  const _laidOut = () => !!rail && rail.getClientRects().length > 0;

  const _relayout = () => {
    if (!_laidOut()) return;
    _measure();
    _rescale();
    _highlight();
    _progress();
  };

  let observer = null;
  let resizeTimer = 0;
  const _onResize = () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(_relayout, 120);
  };

  /* An ARENA notebook is full of <details>, and opening one moves every
     heading below it. `toggle` does not bubble, so this listens in the capture
     phase on the cells container. */
  const _onToggle = () => _relayout();

  /* ---------- building -------------------------------------------------- */

  const _jump = (entry) => {
    window.scrollTo({
      top: Math.max(0, _docTop(entry.el) - JUMP_CLEARANCE),
      behavior: "smooth",
    });
  };

  /* The rail's markup, one element per element of theirs — see the diagram at
     the top of arena-notebook-nav.css. The two that have no counterpart are
     ours: the gutter strip that takes the mouse (they can use `:hover`, we
     cannot; see the note further down) and the narrow-screen toggle. */
  const _shell = () =>
    '<div class="anb-toc-hit" aria-hidden="true"></div>' +
    '<button type="button" class="anb-toc-toggle" aria-label="Notebook contents" ' +
    'aria-expanded="false">☰</button>' +
    '<div class="anb-toc-wrap">' +
    '<div class="anb-toc-progress" aria-hidden="true">' +
    '<div class="anb-toc-progress-fill"></div>' +
    '<div class="anb-toc-progress-rest"></div>' +
    "</div>" +
    '<ol class="anb-toc-rows"></ol>' +
    "</div>";

  /* 🔴 THE ROW AND THE LINE ARE TWO ELEMENTS. The <li> carries the section's
     share of the notebook and nothing else; the line inside it keeps its own
     one-line height and sits at the top of that share. One element doing both
     jobs is a rail whose rows are all the same height, which is a list — the
     thing the rail exists not to be. */
  const _row = (label, level, onClick, extra) => {
    const li = document.createElement("li");
    li.className = extra ? `anb-toc-row ${extra}` : "anb-toc-row";
    li.dataset.level = String(level);
    li.innerHTML =
      '<div class="anb-toc-line">' +
      (extra === "is-title" ? "" : '<span class="anb-toc-dot" aria-hidden="true">•</span>') +
      '<span class="anb-toc-fade">' +
      '<span class="anb-toc-level">' +
      '<button type="button" class="anb-toc-label">' +
      (extra === "is-title" ? `<span class="anb-toc-title">${esc(label)}</span>` : esc(label)) +
      "</button>" +
      "</span></span>" +
      "</div>";
    li.querySelector(".anb-toc-label").addEventListener("click", onClick);
    return li;
  };

  const destroy = () => {
    window.removeEventListener("scroll", _onScroll);
    window.removeEventListener("resize", _onResize);
    if (contents) contents.removeEventListener("toggle", _onToggle, true);
    /* The height watcher outlives its element otherwise: mount() is called once
       per notebook, and an observer still holding the last notebook's cells box
       relayouts the rail against a document nobody is reading. */
    if (observer) observer.disconnect();
    observer = null;
    if (rail) rail.remove();
    rail = list = fill = rest = contents = currentRow = spacer = null;
    entries = [];
  };

  /* Build the rail for the notebook now on screen. `page` is the element the
     rail is parked in (so it hides with the tab), `host` is where the notebook
     was rendered. Returns false when there is nothing worth drawing — a
     notebook with one heading is a rail of one tick, which is noise. */
  const mount = (page, host, title) => {
    destroy();
    if (!page || !host) return false;
    const cells = host.querySelector(".nbv-cells");
    if (!cells) return false;

    /* 🔴 NOT THE HEADINGS INSIDE A <details>. ARENA writes its hints and its
       solutions as disclosures, and those carry headings of their own
       ("Stage 1: De-embeddings & logit lens", "Solution") that are not
       sections of the notebook. Worse, a closed disclosure has no layout, so
       measuring one answers 0 — which put a row at the top of a rail whose
       order is otherwise the document's, and broke the "last heading above the
       mark" rule for every row after it. Measured on arena-1-4-2: 5 such
       headings, 1 non-monotonic offset. */
    const headings = Array.from(cells.querySelectorAll("h1, h2, h3, h4")).filter(
      (heading) => !heading.closest("details"),
    );
    if (headings.length < 2) return false;

    rail = document.createElement("nav");
    rail.className = "anb-toc";
    rail.setAttribute("aria-label", "Notebook contents");
    rail.innerHTML = _shell();
    list = rail.querySelector(".anb-toc-rows");
    fill = rail.querySelector(".anb-toc-progress-fill");
    rest = rail.querySelector(".anb-toc-progress-rest");

    /* The notebook's name is the FIRST ROW, the way their post title is —
       small caps, no dot, and it scrolls away with the rest of the list. It
       used to be a separate absolutely-positioned header, which put it on top
       of row one and gave the rail two titles. */
    if (title) {
      list.appendChild(
        _row(title, 1, () => window.scrollTo({ top: 0, behavior: "smooth" }), "is-title"),
      );
    }

    spacer = document.createElement("li");
    spacer.className = "anb-toc-row is-spacer";
    spacer.setAttribute("aria-hidden", "true");
    list.appendChild(spacer);

    entries = headings.map((heading, i) => {
      /* An id is what a jump needs and what a deep link could use later.
         Upstream headings arrive without one — the markdown renderer emits a
         bare <h2> — so one is minted, namespaced so it cannot collide with the
         `arena-<cellId>` ids the cells already carry. */
      if (!heading.id) heading.id = `arena-h-${i}`;
      const level = Math.min(4, Number(heading.tagName.slice(1)) || 1);
      const row = _row(heading.textContent.trim(), level, () => _jump(entries[i]));
      list.appendChild(row);
      return { el: heading, row, top: 0 };
    });

    contents = cells;
    page.insertBefore(rail, page.firstChild);

    const toggle = rail.querySelector(".anb-toc-toggle");
    toggle.addEventListener("click", () => {
      // On a narrow screen this button IS the rail, so its state is the only
      // thing a screen reader can be told about whether the panel is open.
      toggle.setAttribute("aria-expanded", String(rail.classList.toggle("is-open")));
    });

    /* 🔴 THE OPEN STATE IS A CLASS, NOT `:hover`, and it has to be. The rail is
       300px wide at all times so its labels lay out at their real width, but it
       is `pointer-events: none` so those invisible labels do not swallow clicks
       meant for the notebook underneath. That makes a pure-CSS `:hover` rule
       flicker: once open, the gaps BETWEEN labels are still transparent to the
       mouse, so crossing one would read as leaving the rail. Entering the
       gutter strip opens it; leaving the whole rail closes it. */
    const hit = rail.querySelector(".anb-toc-hit");
    hit.addEventListener("mouseenter", () => {
      rail.classList.add("is-hover");
      /* Opening changes every row's height, so where the current one sits is
         only known after the browser has laid the labels out. */
      requestAnimationFrame(() => _follow(currentRow));
    });
    if (list) list.addEventListener("scroll", _listTouched, { passive: true });
    rail.addEventListener("mouseleave", () => rail.classList.remove("is-hover"));
    // Keyboard: tabbing into a row has to reveal what the row says.
    rail.addEventListener("focusin", () => rail.classList.add("is-hover"));
    rail.addEventListener("focusout", (event) => {
      if (!rail.contains(event.relatedTarget)) rail.classList.remove("is-hover");
    });

    _relayout();
    window.addEventListener("scroll", _onScroll, { passive: true });
    window.addEventListener("resize", _onResize);
    contents.addEventListener("toggle", _onToggle, true);
    /* Images decide the height of an ARENA notebook and they are still
       arriving when this runs. One late remeasure costs nothing and stops the
       rail from being a map of a shorter document than the one on screen. */
    window.addEventListener("load", _relayout, { once: true });
    setTimeout(_relayout, 800);

    /* 🔴 AND A TIMER IS NOT ENOUGH. The notebook is mounted long after the app
       itself loaded, so that `load` listener usually never fires again; an
       image that arrives at 900ms, or a code cell that prints forty lines of
       output, moves every heading below it and the rail goes on describing the
       document as it was. `resize` does not fire for either. So watch the
       cells box itself — its height IS the thing every row's share is measured
       against. Found by codex, 2026-09-02. */
    if (typeof ResizeObserver === "function") {
      let settle = 0;
      observer = new ResizeObserver(() => {
        clearTimeout(settle);
        settle = setTimeout(_relayout, 120);
      });
      observer.observe(contents);
    }
    return true;
  };

  /* Called when the tab comes back: the rail survived in the DOM but every
     offset it holds was measured while the page was display:none, where
     getBoundingClientRect answers 0. */
  const refresh = () => {
    if (rail) _relayout();
  };

  return { mount, refresh, destroy };
})();

window.ArenaNotebookNav = ArenaNotebookNav;
