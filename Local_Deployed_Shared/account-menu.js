/* ================================================================
   THE ACCOUNT MENU — the top-right control, and the only way around the app

   WHAT IT REPLACED
     A bare cog that jumped straight to the Account tab. Two things were wrong
     with it, and Seth named both on 2026-08-24: it sat in the MIDDLE of the
     topbar rather than the right edge (a `display: none` tab strip is removed
     from the grid, so the right cell slid into the strip's column — fixed in
     styles/layout.css, not here), and a wordless glyph does not read as
     clickable. It is a labelled button with a caret now, and it opens a menu.

   WHY A MENU AT ALL
     Basic mode has no tab strip. Before this file, the cog was the only control
     that left the Learner Home, and it went to exactly one place — so "Why this
     app exists" was reachable from the welcome fork ONCE, on a first visit, and
     never again. The menu is the route back to every part of the app that is
     not the thing you do every day.

   🔴 `switchTab` IS NOT ON `window`
     It is a top-level `const` in app.js, and a classic script's top-level const
     never becomes a window property. This file is a classic script loaded after
     app.js in the same document, so it shares that script scope and can call it
     by name; `window.switchTab` would be undefined and every route here would
     be a silent no-op. The same trap is documented for `PracticeAPI` and
     `PracticeSession` elsewhere in this tree. The typeof guard is for a page
     that loads this without app.js.

   🔴 THE ROUTING IS app.js's, NOT A SECOND COPY
     Every item carries `data-goto-tab`, which app.js already binds at load. This
     file does not re-implement the jump — it only closes the menu, and handles
     the one thing app.js cannot know about: `data-lab-open`, which names a
     <details> on #page-learn-about-app to open after the switch. That is how one
     page serves two menu rows ("Why this app exists" / "How this app works")
     without becoming two pages again.
   ================================================================ */

(function initAccountMenu() {
  const btn = document.getElementById("topbar-account");
  const menu = document.getElementById("account-menu");
  if (!btn || !menu) return;

  const isOpen = () => !menu.classList.contains("hidden");

  const setOpen = (open) => {
    menu.classList.toggle("hidden", !open);
    btn.setAttribute("aria-expanded", open ? "true" : "false");
    btn.classList.toggle("is-open", open);
  };

  const close = () => setOpen(false);

  /* 🔴 role="menu" IS A PROMISE. It tells a screen reader this is a menu, and a
     menu is arrow-key navigable — Tab-only is what role="group" would have
     meant. Codex raised this on 2026-08-24 and it was right: the semantics were
     announced and the behaviour was not there. Focus lands on the first row on
     open, Up/Down wrap, Home/End jump. */
  const items = () => Array.from(menu.querySelectorAll(".account-menu-item"));

  const focusItem = (index) => {
    const list = items();
    if (!list.length) return;
    const n = list.length;
    list[((index % n) + n) % n].focus();
  };

  const moveFocus = (delta) => {
    const list = items();
    const at = list.indexOf(document.activeElement);
    // Not on a row yet (focus is still on the trigger): Down opens at the top,
    // Up at the bottom, which is what a menu does everywhere else.
    focusItem(at === -1 ? (delta > 0 ? 0 : list.length - 1) : at + delta);
  };

  /* THE TWO EXPLAINER ROWS. Both route to #page-learn-about-app; what tells
     them apart is which disclosure is open when the learner arrives.

     Deliberately EXCLUSIVE: opening one closes the other. The page is long
     enough that two open disclosures push the second summary a screen and a
     half below the fold, and a learner who asked for "how this app works" and
     landed on the middle of "the three markers" reads it as the wrong page.

     🔴 `scrollIntoView` runs on the NEXT frame, not now. `switchTab` un-hides
     the page in this same task; the element has no layout box until the style
     recalculation that follows, and scrolling to a box that does not exist yet
     scrolls to the top of the document instead. */
  const openDisclosure = (id) => {
    const wanted = document.getElementById(id);
    if (!wanted) return;
    document.querySelectorAll("#page-learn-about-app details.lab-disclosure").forEach((d) => {
      d.open = d === wanted;
    });
    requestAnimationFrame(() => {
      wanted.scrollIntoView({ block: "start", behavior: "smooth" });
    });
  };

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const opening = !isOpen();
    setOpen(opening);
    // Next frame: the rows have no layout box until the menu is un-hidden, and
    // `.focus()` on a `display: none` element does nothing at all.
    if (opening) requestAnimationFrame(() => focusItem(0));
  });

  menu.addEventListener("click", (e) => {
    const item = e.target.closest(".account-menu-item");
    if (!item) return;
    close();
    /* app.js's own [data-goto-tab] handler runs on this same click and does the
       switching. This only adds the disclosure, and only after the page it
       belongs to is on screen — hence after, in the same task. */
    const lab = item.dataset.labOpen;
    if (lab) openDisclosure(lab);
  });

  /* Click-away and Escape. Both are what a menu is expected to do, and without
     them the only way to dismiss this is to pick something — which turns a
     misclick into a page change. */
  document.addEventListener("click", (e) => {
    if (!isOpen()) return;
    if (menu.contains(e.target) || btn.contains(e.target)) return;
    close();
  });

  document.addEventListener("keydown", (e) => {
    if (!isOpen()) return;
    if (e.key === "Escape") {
      close();
      btn.focus();
      return;
    }
    // Only while the menu owns the interaction — otherwise this would steal the
    // arrow keys from the code editor underneath.
    if (!menu.contains(document.activeElement) && document.activeElement !== btn) return;
    if (e.key === "ArrowDown") { e.preventDefault(); moveFocus(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); moveFocus(-1); }
    else if (e.key === "Home") { e.preventDefault(); focusItem(0); }
    else if (e.key === "End") { e.preventDefault(); focusItem(items().length - 1); }
  });

  /* A tab switch from anywhere else — the welcome fork's arrows, the Account
     page's escape row, a deep link — must not leave this hanging open over the
     page it just left. app.js fires no event for a switch, so the class it
     writes on the pages is the signal. */
  if (typeof MutationObserver === "function") {
    const home = document.getElementById("page-practice");
    if (home) {
      new MutationObserver(() => {
        if (isOpen()) close();
      }).observe(home, { attributes: true, attributeFilter: ["class"] });
    }
  }

  window.DDAccountMenu = { close, openDisclosure };
})();
