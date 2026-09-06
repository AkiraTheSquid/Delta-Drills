/* ================================================================
   ARENA NOTEBOOK RESUME — the Courses tab opens where you left off
   ================================================================

   Seth, 2026-09-06: "it doesn't always take you back to the main page for
   deciding which content you want to go to. Whenever you click on the courses
   or whatever, like if you had already clicked on chapter 0.0, then it would
   remember that and immediately go there."

   The Courses tab is a course page, then a chapter modal, then a section row —
   three clicks to get back to the notebook you were reading ten minutes ago,
   every time. This file collapses them: the tab remembers the last section you
   opened and goes straight there.

   🔴 AND THE LIST HAS TO STAY REACHABLE, or the tab is a trap. "← The course"
   is the learner saying "show me the list", so it SUPPRESSES the resume rather
   than merely navigating — otherwise the next visit to Courses drags them back
   into the notebook they just left and the section list is unreachable while
   the notebook exists. Opening any section clears the suppression again, so the
   two states are: "you were reading 0.1, go there" and "you asked for the list,
   here is the list", and each one is one click from the other.

   🔴 THE TRIGGER IS THE PAGE BECOMING VISIBLE, NOT A ROUTER EVENT. app.js's
   `switchTab` fires nothing and is reached from a tab button, an account-menu
   row, a `[data-goto-tab]`, a solo pathname and the boot call at the bottom of
   app.js. The class on `#page-courses` is the one signal all of them share —
   the same reason `arena-notebook-state.js` watches its own page this way. The
   boot call runs BEFORE this script (app.js is loaded ~150 lines earlier in
   index.html), so the page can already be visible when the observer is
   installed; the install therefore checks once itself rather than waiting for a
   change that has already happened.
   ================================================================ */

const ArenaNotebookResume = (() => {
  const KEY = "dd_arena_last";
  const PAGE = () => document.getElementById("page-courses");

  let resuming = false;
  let wired = false;
  /* 🔴 SUPPRESSION IS IN MEMORY FIRST, STORAGE SECOND. `suppress()` used to be
     nothing but a localStorage write, so a browser that refused the write — a
     full origin, private mode, site data blocked — left `skip:false` on disk;
     the observer then re-read it the instant #page-courses appeared and
     reopened the notebook the learner had just pressed Back out of, with the
     section list unreachable for as long as that notebook existed. The write
     is what carries the choice to tomorrow; this flag is what makes Back work
     TODAY, whatever storage says. Found by codex, 2026-09-06. */
  let skipThisLoad = false;

  const _read = () => {
    try {
      const parsed = JSON.parse(localStorage.getItem(KEY) || "null");
      if (!parsed || typeof parsed.slug !== "string" || !parsed.slug) return null;
      return parsed;
    } catch (_) {
      return null;
    }
  };

  const _write = (value) => {
    try {
      if (value) localStorage.setItem(KEY, JSON.stringify(value));
      else localStorage.removeItem(KEY);
    } catch (_) {
      /* A full or blocked localStorage costs the shortcut, not the tab. */
    }
  };

  const _visible = () => {
    const page = PAGE();
    return !!page && !page.classList.contains("hidden");
  };

  /* The section now on screen. Called by the view for both paths into a
     notebook — the fresh render and the reopen that skips it — because both
     mean "this is what the learner is reading". */
  const remember = (slug, title) => {
    if (!slug) return;
    skipThisLoad = false;
    _write({ slug, title: title || "", skip: false, at: Date.now() });
  };

  /* "Show me the list." Keeps the slug (so a later section click is not the
     only way back) but stops the tab from jumping. */
  const suppress = () => {
    skipThisLoad = true;
    const last = _read();
    if (!last || last.skip) return;
    last.skip = true;
    _write(last);
  };

  const forget = () => {
    skipThisLoad = true;
    _write(null);
  };

  const last = () => _read();

  const _courses = () => {
    if (typeof switchTab === "function") switchTab("courses");
    else if (typeof window.switchTab === "function") window.switchTab("courses");
  };

  /* 🔴 A NOTEBOOK THAT IS NOT IN THIS BUILD MUST NOT STRAND ANYONE, AND THAT
     IS TRUE OF EVERY WAY THE OPEN CAN FAIL — not only the `false` it returns
     for a missing compile. `open()` routes to the notebook page BEFORE it
     fetches, so a rejected promise (or a synchronous throw, which used to skip
     `.finally` and leave `resuming` stuck true for the rest of the page load)
     left an error page under a learner who asked for the course LIST and never
     chose this section. One recovery: forget the target, go back, unlock.
     Found by codex, 2026-09-06. */
  const _giveUp = () => {
    forget();
    _courses();
  };

  const _resume = () => {
    if (resuming || skipThisLoad || !_visible()) return;
    const saved = _read();
    if (!saved || saved.skip) return;
    const view = window.ArenaNotebook;
    if (!view || typeof view.open !== "function") return;
    resuming = true;
    let pending;
    try {
      pending = Promise.resolve(view.open(saved.slug));
    } catch (_) {
      resuming = false;
      _giveUp();
      return;
    }
    pending
      .then((opened) => {
        if (opened === false) _giveUp();
      })
      .catch(() => _giveUp())
      .finally(() => {
        resuming = false;
      });
  };

  const _wire = () => {
    if (wired) return;
    const page = PAGE();
    if (!page) return;
    wired = true;
    let wasVisible = _visible();
    new MutationObserver(() => {
      const now = _visible();
      if (now === wasVisible) return;
      wasVisible = now;
      if (now) _resume();
    }).observe(page, { attributes: true, attributeFilter: ["class"] });
    /* Already on Courses when this script ran — see the header. Deferred by a
       frame so the rest of index.html's scripts (arena-notebook.js among them)
       have been evaluated before anything tries to open a notebook. */
    if (wasVisible) requestAnimationFrame(_resume);
  };

  if (document.readyState === "loading") window.addEventListener("DOMContentLoaded", _wire);
  else _wire();

  return { remember, suppress, forget, last };
})();

window.ArenaNotebookResume = ArenaNotebookResume;
