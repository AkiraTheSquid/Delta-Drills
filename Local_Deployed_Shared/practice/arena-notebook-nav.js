/* ================================================================
   ARENA NOTEBOOK NAV — a small Colab-like contents tree
   ================================================================

   The tree stays hidden while the learner reads. Moving anywhere into the
   gutter between the viewport edge and the prose column reveals it; moving
   into the prose closes it. Rows follow rendered h1-h4 headings, highlight
   the section at the viewport mark, and turn green once every runnable cell
   in that section has completed successfully.

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

  const esc = (value) =>
    String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

  const _docTop = (el) => el.getBoundingClientRect().top + window.scrollY;
  const _laidOut = () => !!rail && rail.getClientRects().length > 0;

  /* 🔴 THE REVEAL ZONE STOPS AT THE LEFTMOST CELL, NOT AT `.nbv-cells`.
     The prose column is 682px, but a code cell BREAKS OUT to 880 (see
     styles/practice/arena-notebook.css) — so it starts ~99px to the left of
     the column that contains it. Measuring the column put the hit strip on
     top of every Run button in the notebook: `elementFromPoint` over a Run
     button answered `.anb-toc-hit`, and not one cell on the page could be
     run. Measured 2026-09-03 at 1600px, where the strip was 454px wide and
     the buttons sat at x=368.

     So: the narrowest gap any cell leaves, and no floor above it. A floor
     (this was `Math.max(220, …)`) is the same bug at a narrower window — it
     claims gutter the content is already using. Below 1180px there is no
     safe gap left at all, and the stylesheet drops the strip for the toggle
     button instead. */
  const _measure = () => {
    entries.forEach((entry) => {
      entry.top = _docTop(entry.el);
    });
    if (!contents || !rail) return;
    let left = contents.getBoundingClientRect().left;
    contents.querySelectorAll(":scope > .nbv-cell").forEach((cell) => {
      // A cell with no box (inside a closed disclosure) measures 0 and would
      // collapse the gutter to nothing.
      if (!cell.getClientRects().length) return;
      left = Math.min(left, cell.getBoundingClientRect().left);
    });
    rail.style.setProperty("--anb-gutter", `${Math.max(0, Math.round(left))}px`);
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
      const top = row.offsetTop;
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
    resizeTimer = setTimeout(_relayout, 120);
  };

  const _jump = (entry) => {
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
    rail.setAttribute("aria-label", "Notebook contents");
    rail.innerHTML =
      '<div class="anb-toc-hit" aria-hidden="true"></div>' +
      '<button type="button" class="anb-toc-toggle" aria-label="Notebook contents" aria-expanded="false">☰</button>' +
      '<div class="anb-toc-panel">' +
      `<div class="anb-toc-title">${esc(titleText)}</div>` +
      '<ol class="anb-toc-rows"></ol>' +
      "</div>";
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
    const toggle = rail.querySelector(".anb-toc-toggle");
    toggle.addEventListener("click", () => {
      const open = rail.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", String(open));
    });

    rail.querySelector(".anb-toc-hit").addEventListener("mouseenter", () => {
      rail.classList.add("is-hover");
    });
    rail.addEventListener("mouseleave", () => rail.classList.remove("is-hover"));
    rail.addEventListener("focusin", () => rail.classList.add("is-hover"));
    rail.addEventListener("focusout", (event) => {
      if (!rail.contains(event.relatedTarget)) rail.classList.remove("is-hover");
    });

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
