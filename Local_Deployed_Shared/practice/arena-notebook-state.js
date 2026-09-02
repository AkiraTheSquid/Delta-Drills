/* ================================================================
   ARENA NOTEBOOK STATE — where you were in the notebook you left
   ================================================================

   Seth, 2026-09-02: "whenever you click on one of the options and it goes to
   that notebook ... if you go to a different tab, then it saves the state such
   that if you go back to that tab, it will stay at that location that you were
   at before, including ... scrolled all the way down."

   TWO HALVES, and only one of them is this file.

   The first half is in arena-notebook.js: reopening the notebook already on
   screen no longer re-fetches and re-renders it. That is what actually
   preserves the SESSION — the cells you edited, the outputs you ran, the
   <details> you opened. A rebuild would have thrown all of it away and left
   this file restoring a scroll position into a notebook that had forgotten
   everything else.

   The second half is here: the scroll position, kept per notebook slug in
   localStorage so it also survives a reload and a return tomorrow.

   🔴 THE POSITION IS CAPTURED WHILE THE PAGE IS VISIBLE, NEVER ON THE WAY OUT.
   app.js switches tabs by toggling `.hidden`, and `.hidden { display: none }` —
   a hidden element has no layout, so every getBoundingClientRect() answers 0
   and window.scrollY is about to be clamped to whatever the SHORTER page needs.
   Reading the position inside the hide handler is reading it a moment too late.
   So a scroll handler keeps the last good reading in memory and the hide
   handler only persists it.

   🔴 AND IT IS AN ANCHOR, NOT A PIXEL COUNT. An ARENA notebook's height is not
   stable: images arrive late, a disclosure is open or closed, the window is a
   different width than it was. A remembered `scrollY` lands somewhere else
   entirely under any of those. What is remembered is which cell was at the top
   of the viewport and by how much it was clipped, with the pixel count kept
   only as the fallback for a cell that is no longer there.
   ================================================================ */

const ArenaNotebookState = (() => {
  const KEY = (slug) => `dd_arena_pos:${slug}`;
  const PAGE = () => document.getElementById("page-arena-notebook");
  /* How far a delayed re-apply may find the page from where it put it before
     it decides the learner has taken over and stops correcting. */
  const HANDOFF = 6;

  let slug = null;
  let last = null; // the most recent reading taken while the page was visible
  let applied = null; // where restore() last put the page
  let ticking = false;
  let saveTimer = 0;
  let wired = false;

  const _visible = () => {
    const page = PAGE();
    return !!page && !page.classList.contains("hidden");
  };

  const _read = (key) => {
    try {
      return JSON.parse(localStorage.getItem(KEY(key)) || "null");
    } catch (err) {
      return null;
    }
  };

  const _write = (key, value) => {
    try {
      localStorage.setItem(KEY(key), JSON.stringify(value));
    } catch (err) {
      /* A full or blocked localStorage is not worth a broken notebook. The
         cost of losing this is one scroll position. */
    }
  };

  /* The topmost cell still on screen, and how far above the fold it starts.
     `bottom > 0` rather than `top >= 0`: at any scroll position the cell you
     are reading is usually one that STARTED above the viewport, and picking
     the first one fully below it would jump you forward by a cell on every
     restore. */
  const _capture = () => {
    const cells = document.querySelectorAll("#arena-notebook-host .nbv-cell");
    /* 🔴 NO CELLS MEANS NO READING, not a reading of zero. The host holds
       "Loading the notebook…" for the whole of a notebook switch, and replacing
       a 26,000px page with one paragraph makes the browser clamp the scroll and
       fire a scroll event — which arrives here while the OUTGOING slug is still
       bound. Answering it with `{y: 40, id: null}` overwrites the position that
       notebook was left at, and it reads back as "you were at the top". */
    if (!cells.length) return null;
    for (const cell of cells) {
      const rect = cell.getBoundingClientRect();
      if (rect.bottom > 0) return { y: window.scrollY, id: cell.id, delta: rect.top };
    }
    return { y: window.scrollY, id: null, delta: 0 };
  };

  const _persist = () => {
    if (slug && last) _write(slug, last);
  };

  const _onScroll = () => {
    if (!slug || ticking || !_visible()) return;
    ticking = true;
    requestAnimationFrame(() => {
      ticking = false;
      if (!_visible()) return;
      const reading = _capture();
      if (!reading) return;
      last = reading;
      clearTimeout(saveTimer);
      saveTimer = setTimeout(_persist, 400);
    });
  };

  /* Put the page back. Called more than once on purpose: the first call runs
     against a notebook whose images have not arrived, and the correction is
     abandoned the moment the reading no longer matches where this function
     last left the page — that difference is the learner scrolling, and a
     scroll restore that fights the learner is worse than none. */
  const _apply = () => {
    const saved = slug && _read(slug);
    if (!saved) return;
    const anchor = saved.id && document.getElementById(saved.id);
    const top = anchor
      ? Math.max(0, window.scrollY + anchor.getBoundingClientRect().top - (saved.delta || 0))
      : Math.max(0, saved.y || 0);
    window.scrollTo(0, top);
    applied = window.scrollY;
  };

  const restore = () => {
    if (!slug || !_read(slug)) return;
    _apply();
    requestAnimationFrame(() => {
      if (applied === null || Math.abs(window.scrollY - applied) <= HANDOFF) _apply();
    });
    /* One late pass, for the images. An ARENA notebook's diagrams are the
       whole reason its height is not final on the frame it renders. */
    setTimeout(() => {
      if (applied !== null && Math.abs(window.scrollY - applied) <= HANDOFF) _apply();
    }, 700);
  };

  const _wire = () => {
    if (wired) return;
    wired = true;
    window.addEventListener("scroll", _onScroll, { passive: true });
    // A close or a reload is the one exit that gets no mutation to react to.
    window.addEventListener("pagehide", _persist);

    /* The tab switch itself. app.js fires no event when it routes, and this
       page is reached from several places (a section row, the `?arena=` deep
       link, the browser Back into an app that never unloaded), so the class on
       the page element is the one signal all of them share. */
    const page = PAGE();
    if (!page) return;
    let wasVisible = _visible();
    new MutationObserver(() => {
      const now = _visible();
      if (now === wasVisible) return;
      wasVisible = now;
      if (now) restore();
      else _persist();
    }).observe(page, { attributes: true, attributeFilter: ["class"] });
  };

  /* The notebook now on screen owns the position. Called by _render, and by
     the reopen path that skips it. */
  const bind = (nextSlug) => {
    if (slug && slug !== nextSlug) _persist();
    slug = nextSlug || null;
    last = null;
    applied = null;
    _wire();
  };

  /* Take a reading now — used by the reopen path, which switches tabs and must
     not lose the position it is about to restore into. */
  const save = () => {
    if (!slug || !_visible()) return;
    const reading = _capture();
    if (!reading) return;
    last = reading;
    _persist();
  };

  /* Let go of the notebook being replaced. 🔴 Between "open another section"
     and the new notebook rendering, the host is a loading paragraph and the
     page is still on screen — every scroll event in that window would be
     attributed to the notebook we just left. Saving is not enough on its own:
     the binding has to end too, or the debounced write and the next bind()
     both flush a reading taken against a page that is no longer that notebook.
     _render's bind() re-arms it. */
  const suspend = () => {
    save();
    slug = null;
    last = null;
    applied = null;
  };

  return { bind, save, suspend, restore };
})();

window.ArenaNotebookState = ArenaNotebookState;
