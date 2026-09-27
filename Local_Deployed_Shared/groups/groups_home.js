/* ================================================================
   GROUPS_HOME.JS — the study-group board's life on the Learner Home.

   The board had a tab of its own until 2026-09-26, when Seth folded it
   into the Learner Home under the XP card ("Full board, tab removed"):
   the group is what keeps a day's XP honest, so it sits where the day's
   XP is read. groups_view.js still builds all of it into #groups-root;
   this file only decides WHEN it is live.

   Live = the Practice page is showing, idle (no question on the clock)
   and not asking the "what have you done before?" survey — the three
   states in which #learner-group is on screen (styles/practice/home.css).
   Those are classes on #page-practice written by three different files
   (app.js `hidden`, timer.js `session-idle`, survey.js `is-surveying`),
   so one MutationObserver on that class list is the single place that
   sees every transition, instead of a hook in each writer.

   🔴 LEAVING IS A TEARDOWN, same as leaving the tab was: `suspend()`
   destroys the checklist editors, and the teardown is what FLUSHES
   their half-second save debounce. Starting a session with a line
   half-typed must not lose it, so going non-idle suspends too.

   🔴 A GUEST SEES NO BOARD unless the address bar carries an invite.
   The tab was `auth-only`; a guest's progress lives in the browser, so
   the only thing the section could say to one is "sign in" — which is
   worth saying exactly when a friend's link brought them here. Sign in
   and sign out both reload the page, so this is decided once.
   ================================================================ */
(() => {
  const page = document.getElementById("page-practice");
  const section = document.getElementById("learner-group");
  if (!page || !section) return;

  const invited = !!window.DDGroupStore?.inviteFromLocation?.();
  const wanted = invited || window.DDGroupStore?.hasAccount?.() === true;
  section.hidden = !wanted;
  if (!wanted) return;

  /* An invite link is a request for THIS section, so it goes to the top of
     the Learner Home for that visit (`order: -1`, styles/practice/home.css).
     Not a scrollIntoView: the XP card and the activity bars above it fill in
     asynchronously and push a scrolled-to section back off the screen. The
     token leaves the address bar once the join happens, so the next load has
     the normal order. */
  section.classList.toggle("is-invited", invited);

  let live = false;

  const onScreen = () => !page.classList.contains("hidden")
    && page.classList.contains("session-idle")
    && !page.classList.contains("is-surveying");

  const sync = () => {
    const now = onScreen();
    if (now === live) return;
    live = now;
    if (!live) {
      window.DDGroups?.suspend?.();
      return;
    }
    /* Read on ARRIVAL and at no other time — a roster changes when
       somebody joins, not per interval. */
    window.DDGroups?.refresh();
  };

  new MutationObserver(sync).observe(page, { attributes: true, attributeFilter: ["class"] });
  sync();
})();
