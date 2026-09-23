/* ================================================================
   ARENA NOTEBOOK NAV — a small Colab-like contents tree
   ================================================================

   A DOCKED LEFT PANE, the way Colab lays a notebook out (Seth, 2026-09-22:
   "go with the collab format where the code and text take up all the space
   on the right ... you should be able to drag the divider between the table
   of contents on the left and the code on the right"). It no longer hides
   until hovered. Rows follow rendered h1-h4 headings, highlight the section
   at the viewport mark, and turn green once every runnable cell in that
   section has completed successfully.

   The pane's head is frozen above the rows: the notebook's own title, a ⇕
   that swaps the rows for every ARENA chapter and its sections, and a book
   icon that folds the pane away (and, folded, is the one thing left on
   screen to bring it back).

   🔴 THE WIDTH IS ONE TOKEN ON THE PAGE, `--anb-pane-w`. The pane reads it
   for its width and `.arena-notebooks-container` reads it for its left
   margin (styles/practice/arena-notebook.css), so the two can never overlap
   or leave a gap between them. Folding writes 0px; destroy() removes it.

   The nav lives inside #page-arena-notebook. Hiding that page therefore hides
   the fixed nav too, without global route cleanup.
   ================================================================ */

const ArenaNotebookNav = (() => {
  const MARK = () => window.innerHeight / 5;
  const JUMP_CLEARANCE = 96;

  let rail = null;
  let list = null;
  let contents = null;
  let entries = []; // { el, row, top, level, codeCells }
  let currentRow = null;
  let observer = null;
  let ticking = false;
  let resizeTimer = 0;
  let titleText = "";
  let mountedPage = null;
  let mountedHost = null;
  let signature = "";
  let chaptersOpen = false;
  let wasNarrow = false;

  const WIDTH_KEY = "dd_arena_toc_w";
  const FOLD_KEY = "dd_arena_toc_folded";
  const DEFAULT_W = 320;
  const MIN_W = 180;
  // Below this the pane overlays the notebook instead of docking beside it.
  const NARROW = 900;

  /* Per-viewer conveniences only, so storage that throws (private window,
     blocked site data) just means the default. */
  const _load = (key) => {
    try {
      return localStorage.getItem(key);
    } catch (_) {
      return null;
    }
  };
  const _save = (key, value) => {
    try {
      localStorage.setItem(key, value);
    } catch (_) {
      /* default next time */
    }
  };

  const ICON_BOOK =
    '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="none" ' +
    'stroke="currentColor" stroke-width="1.8" stroke-linejoin="round">' +
    '<rect x="3" y="4.5" width="18" height="15" rx="1.5"/><path d="M12 4.5v15"/>' +
    '<path d="M5.5 8.5h4M5.5 11.5h4M5.5 14.5h4" stroke-linecap="round"/></svg>';
  const ICON_UNFOLD =
    '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="none" ' +
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<path d="M7 9.5l5-5 5 5M7 14.5l5 5 5-5"/></svg>';

  const esc = (value) =>
    String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

  const _docTop = (el) => el.getBoundingClientRect().top + window.scrollY;
  const _laidOut = () => !!rail && rail.getClientRects().length > 0;

  const _measure = () => {
    entries.forEach((entry) => {
      entry.top = _docTop(entry.el);
    });
  };

  const _narrow = () => window.innerWidth < NARROW;
  const _maxW = () => Math.max(MIN_W, Math.round(window.innerWidth * 0.6));
  const _clampW = (w) => Math.min(_maxW(), Math.max(MIN_W, Math.round(w)));

  /* Writes the token both sides of the divider read. Folded, or narrow (the
     pane then overlays), the notebook keeps the whole width. */
  const _applyWidth = (w) => {
    if (!rail || !mountedPage) return;
    const width = _clampW(w);
    rail.style.setProperty("--anb-pane-w", `${width}px`);
    const docked = !rail.classList.contains("is-folded") && !_narrow();
    mountedPage.style.setProperty("--anb-pane-w", docked ? `${width}px` : "0px");
    const split = rail.querySelector(".anb-toc-split");
    if (split) {
      split.setAttribute("aria-valuenow", String(width));
      split.setAttribute("aria-valuemax", String(_maxW()));
    }
    return width;
  };

  const _currentW = () => _clampW(Number(_load(WIDTH_KEY)) || DEFAULT_W);

  const _setFolded = (folded) => {
    if (!rail) return;
    rail.classList.toggle("is-folded", folded);
    rail.querySelectorAll(".anb-toc-fold").forEach((button) => {
      button.setAttribute("aria-expanded", String(!folded));
    });
    // A narrow window's choice is not remembered: it would fold the docked
    // pane the next time the same learner is on a wide one.
    if (!_narrow()) _save(FOLD_KEY, folded ? "1" : "0");
    _applyWidth(_currentW());
    _relayout();
  };

  /* 🔴 POINTER CAPTURE, NOT DOCUMENT LISTENERS. A drag that ends over the
     notebook — or outside the window — still delivers its pointerup to the
     divider, so the pane can never be left mid-drag with the cursor stuck
     as col-resize. */
  const _wireSplit = (split) => {
    let dragging = false;
    // The width last RENDERED is the one saved — a pointerup/pointercancel
    // carries its own clientX, which a coalesced or cancelled drag can make
    // differ from what is on screen.
    let applied = 0;
    split.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      dragging = true;
      applied = _currentW();
      split.setPointerCapture(event.pointerId);
      rail.classList.add("is-dragging");
      event.preventDefault();
    });
    split.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      applied = _applyWidth(event.clientX) || applied;
    });
    const end = (event) => {
      if (!dragging) return;
      dragging = false;
      rail.classList.remove("is-dragging");
      if (split.hasPointerCapture(event.pointerId)) split.releasePointerCapture(event.pointerId);
      _save(WIDTH_KEY, String(applied));
      _relayout();
    };
    split.addEventListener("pointerup", end);
    split.addEventListener("pointercancel", end);
    split.addEventListener("dblclick", () => {
      _save(WIDTH_KEY, String(DEFAULT_W));
      _applyWidth(DEFAULT_W);
      _relayout();
    });
    split.addEventListener("keydown", (event) => {
      const step = event.key === "ArrowLeft" ? -16 : event.key === "ArrowRight" ? 16 : 0;
      if (!step) return;
      event.preventDefault();
      const width = _clampW(_currentW() + step);
      _save(WIDTH_KEY, String(width));
      _applyWidth(width);
      _relayout();
    });
  };

  /* ⇕ — every ARENA chapter and its sections, in place of the rows. The
     index is the compiled one practice/arena-notebook.js already fetches for
     the Courses tab; a section the app cannot open yet is listed, dimmed, so
     the learner sees the whole course and not a gap in it. */
  const _showChapters = async (show) => {
    if (!rail) return;
    chaptersOpen = show;
    rail.classList.toggle("is-chapters", show);
    const button = rail.querySelector(".anb-toc-switch");
    button.setAttribute("aria-expanded", String(show));
    if (!show) return;
    const host = rail.querySelector(".anb-toc-chapters");
    const view = window.ArenaNotebook;
    let sections = [];
    try {
      sections = view ? (await view.sections()) || [] : [];
    } catch (err) {
      console.warn("[arena-notebook-nav] chapter index unavailable:", err);
    }
    if (!rail || !chaptersOpen) return;
    // No index (offline, or a build without it): say so, rather than show an
    // empty pane with the notebook's own rows hidden behind it.
    if (!sections.length) {
      host.innerHTML =
        '<li class="anb-toc-chapter"><div class="anb-toc-chapter-name">' +
        "The chapter list could not be loaded.</div></li>";
      return;
    }
    const here = view && view.currentId ? view.currentId() : null;
    const groups = new Map();
    sections.forEach((section) => {
      const name = section.chapter || "ARENA";
      if (!groups.has(name)) groups.set(name, []);
      groups.get(name).push(section);
    });
    host.innerHTML = "";
    groups.forEach((list, name) => {
      const group = document.createElement("li");
      group.className = "anb-toc-chapter";
      group.innerHTML = `<div class="anb-toc-chapter-name">${esc(name)}</div><ol></ol>`;
      const ol = group.querySelector("ol");
      list.forEach((section) => {
        const li = document.createElement("li");
        li.className = "anb-toc-row";
        li.dataset.level = "2";
        const openable = !view.canOpen || view.canOpen(section.id);
        li.innerHTML =
          `<button type="button" class="anb-toc-label"${openable ? "" : " disabled"}` +
          `${openable ? "" : ' title="Not in the app yet"'}>` +
          `<span class="anb-toc-text">${esc(section.number ? `${section.number} ${section.title}` : section.title)}</span>` +
          "</button>";
        if (section.id === here) li.classList.add("is-current");
        li.querySelector("button").addEventListener("click", () => {
          if (!openable) return;
          if (section.id === here) {
            _showChapters(false);
            return;
          }
          view.open(section.id);
        });
        ol.appendChild(li);
      });
      host.appendChild(group);
    });
  };

  const _highlight = () => {
    const mark = window.scrollY + MARK();
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
    if (list) {
      // Measure against the list itself: offsetTop is relative to the row's
      // offsetParent, which overstates the top and let upward scrolls lag.
      const top =
        row.getBoundingClientRect().top - list.getBoundingClientRect().top -
        list.clientTop + list.scrollTop;
      const bottom = top + row.offsetHeight;
      if (top < list.scrollTop) list.scrollTop = top;
      else if (bottom > list.scrollTop + list.clientHeight) {
        list.scrollTop = bottom - list.clientHeight;
      }
    }
  };

  const _sectionCells = (heading, index) => {
    const level = Number(heading.tagName.slice(1)) || 1;
    const nextBoundary = entries
      .slice(index + 1)
      .find((entry) => entry.level <= level)?.el;
    return Array.from(contents.querySelectorAll('.nbv-code[data-role="code"]')).filter(
      (cell) => {
        const afterHeading = !!(
          heading.compareDocumentPosition(cell) & Node.DOCUMENT_POSITION_FOLLOWING
        );
        const beforeBoundary =
          !nextBoundary ||
          !!(cell.compareDocumentPosition(nextBoundary) & Node.DOCUMENT_POSITION_FOLLOWING);
        return afterHeading && beforeBoundary;
      },
    );
  };

  const syncCompletion = () => {
    entries.forEach((entry, index) => {
      entry.codeCells = _sectionCells(entry.el, index);
      const complete =
        entry.codeCells.length > 0 &&
        entry.codeCells.every(
          (cell) =>
            cell.classList.contains("has-run") &&
            !cell.classList.contains("has-failed") &&
            !cell.classList.contains("is-stale"),
        );
      entry.row.classList.toggle("is-complete", complete);
    });
  };

  const _relayout = () => {
    if (!_laidOut()) return;
    _measure();
    _highlight();
    syncCompletion();
  };

  const _onScroll = () => {
    if (ticking || !_laidOut()) return;
    ticking = true;
    requestAnimationFrame(() => {
      ticking = false;
      _highlight();
    });
  };

  const _onResize = () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      /* Crossing into narrow folds the pane — docked, it would now lie over
         the notebook. Crossing back restores what the learner chose there. */
      const narrow = _narrow();
      if (rail && narrow !== wasNarrow) {
        wasNarrow = narrow;
        _setFolded(narrow ? true : _load(FOLD_KEY) === "1");
        return;
      }
      _applyWidth(_currentW());
      _relayout();
    }, 120);
  };

  const _jump = (entry) => {
    // Overlaying a narrow window, the pane is in the way of where it jumped.
    if (_narrow()) _setFolded(true);
    window.scrollTo({
      top: Math.max(0, _docTop(entry.el) - JUMP_CLEARANCE),
      behavior: "smooth",
    });
  };

  /* 🔴 WHAT THE TREE IS BUILT FROM, so `refresh` can tell whether it still
     matches the page. Rebuilding is not free: `mount` calls `destroy`, and
     `destroy` REMOVES the rail — including the row the pointer is pressed on.
     A markdown cell saves on blur, and clicking a row IS a blur, so an
     unconditional rebuild there deleted the button between mousedown and
     mouseup and the first click after an edit did nothing. Found by codex,
     2026-09-03. Text edits almost never touch a heading, so comparing first
     means the rebuild happens only when the tree is actually wrong. */
  const _headingsOf = (host) => {
    const cells = host && host.querySelector(".nbv-cells");
    if (!cells) return [];
    return Array.from(cells.querySelectorAll("h1, h2, h3, h4")).filter(
      (heading) => !heading.closest("details"),
    );
  };

  const _signatureOf = (headings) =>
    headings.map((h) => `${h.tagName}:${h.textContent.trim()}`).join("\u0000");

  const _row = (label, level, onClick) => {
    const li = document.createElement("li");
    li.className = "anb-toc-row";
    li.dataset.level = String(level);
    li.innerHTML =
      '<button type="button" class="anb-toc-label">' +
      `<span class="anb-toc-text">${esc(label)}</span>` +
      '<span class="anb-toc-check" aria-hidden="true">✓</span>' +
      "</button>";
    li.querySelector(".anb-toc-label").addEventListener("click", onClick);
    return li;
  };

  const destroy = () => {
    window.removeEventListener("scroll", _onScroll);
    window.removeEventListener("resize", _onResize);
    if (contents) contents.removeEventListener("toggle", _relayout, true);
    if (observer) observer.disconnect();
    observer = null;
    if (rail) rail.remove();
    if (mountedPage) mountedPage.style.removeProperty("--anb-pane-w");
    chaptersOpen = false;
    rail = list = contents = currentRow = mountedPage = mountedHost = null;
    entries = [];
    titleText = "";
    signature = "";
  };

  const mount = (page, host, title) => {
    destroy();
    if (!page || !host) return false;
    const cells = host.querySelector(".nbv-cells");
    if (!cells) return false;
    const headings = _headingsOf(host);

    /* 🔴 THE MOUNT CONTEXT IS RECORDED BEFORE THE TREE IS EARNED. A notebook
       with fewer than two headings gets no tree — but `refresh({rebuild: true})`
       needs `mountedPage`/`mountedHost` to know WHAT to rebuild, and returning
       above these lines left them null (destroy() clears them on the way in).
       So a learner who deleted their way down to one heading and then wrote a
       new `## ` one could never get the tree back without leaving the notebook
       and coming in again. Recording the signature here too is what keeps that
       from thrashing: a refresh that changed nothing still compares equal.
       Found by codex, 2026-09-03. */
    mountedPage = page;
    mountedHost = host;
    signature = _signatureOf(headings);
    titleText = title || "Contents";
    if (headings.length < 2) return false;

    contents = cells;
    rail = document.createElement("nav");
    rail.className = "anb-toc";
    /* 🪦 THE PANEL'S OWN TITLE ROW IS DELETED (Seth, 2026-09-22: "not
       have that thing at the top left"). It restated the notebook's own h1 —
       which is on screen, two inches to the right — and because the panel is
       revealed by a pointer passing through the margin, the first thing the
       tree showed was a line you could not act on. The notebook's name is not
       lost: it is the rail's accessible name now, so a screen reader still
       hears WHICH notebook's contents these are, and nothing paints for it. */
    rail.setAttribute(
      "aria-label",
      titleText ? `${titleText} — contents` : "Notebook contents",
    );
    /* 🔴 THE HEAD IS FROZEN BY BEING OUTSIDE THE LIST THAT SCROLLS, not by
       `position: sticky` inside it — the rows are an `overflow-y: auto` box
       of their own, and the head is its flex sibling. The name is the
       notebook's first heading ("[0.0] - Prerequisites (exercises)"), which
       is what Seth asked to see there. */
    const firstHeading = headings[0] ? headings[0].textContent.trim() : "";
    const name = firstHeading || titleText;
    rail.innerHTML =
      '<button type="button" class="anb-toc-fold anb-toc-reopen" aria-label="Show contents">' +
      ICON_BOOK +
      "</button>" +
      '<div class="anb-toc-panel">' +
      '<div class="anb-toc-head">' +
      `<span class="anb-toc-name" title="${esc(name)}">${esc(name)}</span>` +
      '<button type="button" class="anb-toc-icon anb-toc-switch" aria-label="All chapters" ' +
      `aria-expanded="false" title="All chapters">${ICON_UNFOLD}</button>` +
      '<button type="button" class="anb-toc-icon anb-toc-fold" aria-label="Hide contents" ' +
      `title="Hide contents">${ICON_BOOK}</button>` +
      "</div>" +
      '<ol class="anb-toc-rows"></ol>' +
      '<ol class="anb-toc-chapters" aria-label="ARENA chapters"></ol>' +
      "</div>" +
      '<div class="anb-toc-split" role="separator" aria-orientation="vertical" ' +
      `aria-label="Resize contents" tabindex="0" aria-valuemin="${MIN_W}"></div>`;
    list = rail.querySelector(".anb-toc-rows");

    entries = headings.map((heading, index) => {
      if (!heading.id) heading.id = `arena-h-${index}`;
      const level = Math.min(4, Number(heading.tagName.slice(1)) || 1);
      const entry = { el: heading, row: null, top: 0, level, codeCells: [] };
      entry.row = _row(heading.textContent.trim(), level, () => _jump(entry));
      list.appendChild(entry.row);
      return entry;
    });

    page.insertBefore(rail, page.firstChild);
    rail.querySelectorAll(".anb-toc-fold").forEach((button) => {
      button.addEventListener("click", () => _setFolded(!rail.classList.contains("is-folded")));
    });
    rail.querySelector(".anb-toc-switch").addEventListener("click", () => {
      _showChapters(!chaptersOpen);
    });
    _wireSplit(rail.querySelector(".anb-toc-split"));
    wasNarrow = _narrow();
    _setFolded(wasNarrow ? true : _load(FOLD_KEY) === "1");

    _relayout();
    window.addEventListener("scroll", _onScroll, { passive: true });
    window.addEventListener("resize", _onResize);
    contents.addEventListener("toggle", _relayout, true);
    setTimeout(_relayout, 800);
    if (typeof ResizeObserver === "function") {
      observer = new ResizeObserver(_onResize);
      observer.observe(contents);
    }
    return true;
  };

  const refresh = ({ rebuild = false } = {}) => {
    if (rebuild && mountedPage && mountedHost) {
      // Only when the headings really moved — see `_signatureOf`.
      if (_signatureOf(_headingsOf(mountedHost)) !== signature) {
        return mount(mountedPage, mountedHost, titleText);
      }
    }
    _relayout();
    return !!rail;
  };

  return { mount, refresh, syncCompletion, destroy };
})();

window.ArenaNotebookNav = ArenaNotebookNav;
