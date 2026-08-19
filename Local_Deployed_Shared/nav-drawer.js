/* ================================================================
   NAV-DRAWER.JS — the tab strip becomes a hamburger menu when the viewport is
   too narrow to hold it.

   WHY THIS EXISTS. `.tabs` is seven tabs plus an ⓘ each, and `styles/layout.css`
   deliberately refuses to wrap them (`white-space: nowrap`) so the strip scrolls
   sideways instead of stacking four rows deep. Below ~900px that scroll is the
   whole navigation: two tabs visible, five off the edge. The Chrome extension
   frames this site in a side panel (`extension/panel/app.html`) at ~300-400px,
   which is always in that state — the app looked like it had no nav at all.

   🔴 THE STRIP IS MOVED, NOT CLONED, AND THAT IS NOT NEGOTIABLE.
   `app.js` runs `document.querySelectorAll(".tab" / ".auth-only" / ".guest-only")`
   at eval time into STATIC NodeLists, and everything about the tabs — the click
   handler, the `.active` highlight, and which tabs a guest may see — is driven
   off those three lists. A copy of the strip is in none of them: its tabs would
   not switch pages, would never highlight, and would show Account and Split Tool
   to a signed-out visitor. `appendChild` MOVES a node — same element, same
   listeners, same NodeList membership — so the strip works identically in either
   home. If you ever find yourself writing `cloneNode` here, stop.

   The DOM position it came from is held by a comment node, not by "put it back
   in the header": `.topbar` is `space-between` over logo / tabs / auth and the
   strip has to land BETWEEN the last two again, not at the end.

   WHAT THIS FILE DOES NOT OWN. Which tab is selected, what a tab does, and who
   may see it — all `app.js`. This file moves one element and opens a panel.
   ================================================================ */

(() => {
  /* Same number as the `@media (max-width: 900px)` in styles/practice/layout.css
     — one definition of "narrow" for the whole app. The breakpoint lives here
     rather than in a media block because CSS cannot move a node between two
     parents, and a media query that only RESTYLED the strip in place would still
     leave it inside the 56px topbar. `styles/nav-drawer.css` documents the other
     half of this pair; change one, change both. */
  const NAV_DRAWER_QUERY = "(max-width: 900px)";

  const body = document.body;
  const nav = document.querySelector(".topbar .tabs");
  const toggle = document.getElementById("nav-toggle");
  const drawer = document.getElementById("nav-drawer");
  const scrim = document.getElementById("nav-scrim");
  const closeBtn = document.getElementById("nav-drawer-close");

  // Every one of these is in index.html. If any is missing the markup has been
  // edited out from under this file, and doing nothing leaves a working (if
  // cramped) topbar strip — which is a far better failure than a half-moved nav.
  if (!nav || !toggle || !drawer || !scrim) return;

  // Where the strip lives when the page is wide. A comment node survives every
  // re-render around it and costs nothing.
  const home = document.createComment(" .tabs sits here when the topbar is wide ");
  nav.parentNode.insertBefore(home, nav);

  const mq = window.matchMedia(NAV_DRAWER_QUERY);

  const isOpen = () => body.classList.contains("nav-open");

  /**
   * Open or close the drawer.
   *
   * `inert` is the load-bearing part. A closed drawer in drawer mode is still
   * `display: flex` — it is pushed off-canvas by a transform, because that is
   * what makes it slide — so without this every tab in it stays in the tab
   * order and a keyboard user tabs from the logo into seven invisible buttons.
   * `display: none` instead would kill the transition, which is the whole
   * reason the drawer reads as a drawer and not as a flash of new UI.
   */
  const setOpen = (open) => {
    /* 🔴 Order matters on the way out. `inert` (and `aria-hidden`) applied to an
       element that CONTAINS the focused node does not move focus somewhere
       sensible — it drops it on <body>, so the next Tab starts the whole page
       over from the top, and `aria-hidden` over a focused node is a violation
       in its own right. Every close path except Escape ends with focus inside
       the drawer (the × button, or the tab that was just picked), so handing it
       back to the toggle has to happen BEFORE the drawer is made inert. */
    if (!open && drawer.contains(document.activeElement)) {
      toggle.focus({ preventScroll: true });
    }
    body.classList.toggle("nav-open", open);
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    drawer.setAttribute("aria-hidden", open ? "false" : "true");
    drawer.inert = !open;
    if (open) {
      // First tab that is actually on screen — `.hidden` is how app.js hides
      // the ones this visitor may not use, so `offsetParent` is the honest test.
      const first = [...drawer.querySelectorAll(".tab")].find((t) => t.offsetParent !== null);
      (first || closeBtn || drawer).focus({ preventScroll: true });
    }
  };

  /**
   * Put the strip where this viewport wants it.
   *
   * Idempotent on purpose: it runs on load, on every breakpoint change, and the
   * parent check means re-running it is free. Closing on the way back to a wide
   * page matters — otherwise `body.nav-open` survives the resize and the topbar
   * comes back with a scrim over it.
   */
  const applyMode = () => {
    const narrow = mq.matches;
    /* Read BEFORE the move, and this is the whole reason it is a variable:
       `appendChild` on an element containing the focused node BLURS it — the
       caret is on <body> by the time the move returns, so asking afterwards
       always answers "not ours" and no rescue ever runs. On the load-time call
       this is false (focus is on <body>), so nothing here steals focus from a
       fresh page. */
    const heldFocus =
      document.activeElement === toggle ||
      drawer.contains(document.activeElement) ||
      nav.contains(document.activeElement);
    if (narrow) {
      if (nav.parentNode !== drawer) drawer.appendChild(nav);
    } else if (nav.parentNode === drawer) {
      home.parentNode.insertBefore(nav, home);
    }
    body.classList.toggle("nav-drawer-mode", narrow);
    if (!narrow) {
      setOpen(false);
      /* The toggle is display:none above the breakpoint, so setOpen's hand-back
         puts the caret on a hidden control and the browser drops it on <body> —
         the next Tab then restarts the page from the top. Land it on the strip
         that just came back instead. */
      if (heldFocus && !nav.contains(document.activeElement)) {
        const landing = nav.querySelector(".tab.active") || nav.querySelector(".tab");
        if (landing) landing.focus({ preventScroll: true });
      }
    } else {
      /* Going narrow with a tab focused walks the caret into a drawer that is
         closed and about to be inert — and the move already dropped it on
         <body>. The toggle is the only visible thing that stands for the strip
         down here, so it gets the caret back. */
      if (heldFocus && !isOpen()) toggle.focus({ preventScroll: true });
      drawer.inert = !isOpen();
    }
  };

  toggle.addEventListener("click", () => setOpen(!isOpen()));
  scrim.addEventListener("click", () => setOpen(false));
  if (closeBtn) closeBtn.addEventListener("click", () => setOpen(false));

  /* Picking a destination closes the menu — leaving it open would hide the page
     it just navigated to behind the scrim. The ⓘ dots are excluded and that is
     deliberate: `.tab-info` is a different class on a sibling button, it opens
     an explanation of the tab you are standing on, and closing the drawer under
     it would dismiss the thing the reader just asked for. `closest(".tab")`
     already makes that distinction — the dots carry `.dd-info.tab-info`, never
     `.tab`. */
  nav.addEventListener("click", (event) => {
    if (!body.classList.contains("nav-drawer-mode")) return;
    if (event.target.closest(".tab")) setOpen(false);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && isOpen()) {
      setOpen(false);
      toggle.focus({ preventScroll: true });
    }
  });

  /**
   * Keep Tab inside the open menu.
   *
   * The drawer is a modal surface — there is a scrim over the page and clicking
   * it closes the menu — but the page behind it is ordinary DOM, so without
   * this, Tab walks straight out of the drawer and onto controls the reader
   * cannot see: the Continue-with-Google button, the session inputs, whatever
   * the scrim is covering. `inert` handles the opposite direction (the CLOSED
   * drawer is unreachable); this handles the open one.
   *
   * The toggle is deliberately part of the ring even though it lives outside
   * the drawer: while the menu is open it is drawn as an ×, so it is the close
   * control, and a close control you cannot Tab to is a trap rather than a
   * cycle. `offsetParent` is the honest visibility test here — app.js hides the
   * tabs this visitor may not use with a `.hidden` class, not an attribute.
   */
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Tab" || !isOpen()) return;
    const ring = [toggle, ...drawer.querySelectorAll("button")].filter(
      (el) => el.offsetParent !== null
    );
    if (!ring.length) return;
    /* 🔴 EVERY Tab is placed from here, not just the two edges. The toggle is
       part of the ring but lives OUTSIDE the drawer and BEFORE it in the DOM,
       so an edge-only trap leaked: a forward Tab off the toggle was not on the
       forward edge, so the browser handled it and walked into whatever the
       markup puts between the two — for a signed-in visitor that is
       `.topbar-auth`, sitting behind the scrim. Reproduced at 380px. Cycling by
       index cannot leak, because the browser never gets to choose. */
    const index = ring.indexOf(document.activeElement);
    const step = event.shiftKey ? -1 : 1;
    // Focus outside the ring entirely (the browser dropped it on <body>, or a
    // click landed on the page before the scrim caught it) re-enters at the
    // near edge rather than being left to wander.
    const next =
      index < 0
        ? (event.shiftKey ? ring.length - 1 : 0)
        : (index + step + ring.length) % ring.length;
    event.preventDefault();
    ring[next].focus({ preventScroll: true });
  });

  // Safari <14 has no addEventListener on a MediaQueryList; addListener is the
  // deprecated spelling that works everywhere this app runs.
  if (mq.addEventListener) mq.addEventListener("change", applyMode);
  else mq.addListener(applyMode);

  applyMode();
})();
