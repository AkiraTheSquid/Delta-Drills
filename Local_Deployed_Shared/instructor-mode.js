/* INSTRUCTOR FEEDBACK MODE — one flag, three doors, and nothing else yet.
 *
 * The mode exists so an expert reviewing the content is never mistaken for a
 * learner practicing it (Seth, 2026-08-24): the two workflows will diverge,
 * and the flag is what every future review surface hangs off. TODAY the mode
 * only (a) stamps `body.dd-instructor-mode`, which shows the topbar badge
 * (styles/account-menu.css) and is the hook review affordances will key on,
 * and (b) keeps its three controls agreeing with each other. Anyone can turn
 * it on — it is a workflow choice, not a privilege; writes stay authed on the
 * backend regardless.
 *
 * THE THREE DOORS, all defined in index.html:
 *   #welcome-arm-instructor      the quiet third arm on the welcome fork. It
 *                                also carries [data-goto-tab="instructor-review"]
 *                                — app.js's own hook navigates to the review
 *                                surface on the same click; this file only
 *                                flips the flag.
 *   #account-menu-instructor     the account dropdown row. Its LABEL is the
 *                                state display: "Enter …" ↔ "Exit …". No
 *                                data-goto-tab, so clicking it only closes
 *                                the menu (account-menu.js) and toggles here.
 *   #account-instructor-mode     the checkbox on the Account page, same shape
 *                                as the advanced-mode toggle above it —
 *                                applies on change, no Save step.
 *
 * Future surfaces listen for `dd-instructor-mode-changed` on document (detail:
 * {on}) rather than polling localStorage; `apply()` alone never dispatches it,
 * so a plain page load is not an "it changed" signal.
 *
 * localStorage reads/writes are try/caught for the same reason theme.js's are:
 * a blocked-storage browser should get a working app in learner mode, not a
 * boot error. */
(() => {
  const KEY = "dd_instructor_mode";

  /* In-memory truth, seeded from storage once. isOn() never re-reads
     localStorage: with storage blocked the toggle still works for this page
     view (it just won't survive a reload), instead of set() writing nowhere
     and apply() reading the old nothing back. */
  let state = (() => {
    try {
      return localStorage.getItem(KEY) === "1";
    } catch {
      return false;
    }
  })();

  const isOn = () => state;

  /* One writer for everything the flag shows: body class, menu label, page
     checkbox. Every door calls set() → apply(), so the three controls cannot
     disagree no matter which one was used. */
  const apply = () => {
    const on = isOn();
    document.body.classList.toggle("dd-instructor-mode", on);
    const label = document.getElementById("account-menu-instructor-label");
    if (label) {
      label.textContent = on
        ? "Exit instructor feedback mode"
        : "Enter instructor feedback mode";
    }
    const box = document.getElementById("account-instructor-mode");
    if (box) box.checked = on;
  };

  const set = (on) => {
    /* Normalize ONCE and use the boolean everywhere: state, storage and the
       event must not be able to disagree when a caller passes a truthy
       non-boolean through window.DDInstructorMode.set(). */
    on = !!on;
    state = on;
    try {
      localStorage.setItem(KEY, on ? "1" : "0");
    } catch {
      /* Storage blocked: `state` still carries this page view. */
    }
    apply();
    document.dispatchEvent(
      new CustomEvent("dd-instructor-mode-changed", { detail: { on } })
    );
  };

  const menuItem = document.getElementById("account-menu-instructor");
  if (menuItem) {
    menuItem.addEventListener("click", () => {
      const turningOn = !isOn();
      set(turningOn);
      /* Entering the mode LANDS somewhere: the review surface. #ir-goto is a
         hidden [data-goto-tab="instructor-review"] proxy (app.js wires those
         at boot; switchTab itself is not on window). Exiting stays put —
         instructor-review.js leaves its own page when the flag drops. */
      if (turningOn) document.getElementById("ir-goto")?.click();
    });
  }

  const box = document.getElementById("account-instructor-mode");
  if (box) box.addEventListener("change", () => set(box.checked));

  const arm = document.getElementById("welcome-arm-instructor");
  if (arm) arm.addEventListener("click", () => set(true));

  apply();

  window.DDInstructorMode = { isOn, set };
})();
