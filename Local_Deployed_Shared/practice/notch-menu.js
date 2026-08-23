/* ================================================================
   THE NOTCH — session controls, off the page until asked for

   The session row used to sit above the question: progress, phase, a
   countdown, Pause & exit and End session, all of it on screen for every
   second of every question. Four pieces of chrome for two buttons anyone
   presses twice a session, in the band of the page the learner reads the
   question in.

   So the buttons moved into a notch straddling the seam between the two
   panels — a tab with three dots, and a menu that opens under it.

   🔴 This is a PROXY, not a second implementation. The real handlers live in
   practice/timer.js, bound to #session-pause-btn and #session-end-btn, and
   they carry state this file must not duplicate: whether a pause is safe
   right now (`sessionPauseBtn.disabled` flips mid-question while grading is
   in flight), what a pause writes to the resume snapshot, what "ended" does
   to the block. Clicking the hidden button is what keeps those two paths
   from drifting. The row is hidden by CSS (styles/practice/timer.css); the
   buttons in it are still live, still bound, and still the only place the
   behaviour is written down.
   ================================================================ */

(function () {
  const notch = document.getElementById("practice-notch");
  const btn = document.getElementById("practice-notch-btn");
  const menu = document.getElementById("practice-notch-menu");
  const pauseItem = document.getElementById("practice-notch-pause");
  const endItem = document.getElementById("practice-notch-end");
  const note = document.getElementById("practice-notch-note");
  if (!notch || !btn || !menu) return;

  /* The hidden originals. Looked up per click, not cached: the practice tab
     is one long-lived document, but a lookup that fails once and is cached
     forever is a menu that silently stops working. */
  const _pauseBtn = () => document.getElementById("session-pause-btn");
  const _endBtn = () => document.getElementById("session-end-btn");

  /* A session is running when timer.js has unhidden its row. That class is
     the single fact both this menu and the row itself read — see timer.js
     `sessionStatusRow.classList.remove("hidden")` at session start and
     `.add("hidden")` at finish. */
  function _sessionOpen() {
    const row = document.getElementById("session-status-row");
    return !!row && !row.classList.contains("hidden");
  }

  function _syncItems() {
    const open = _sessionOpen();
    const pauseBtn = _pauseBtn();
    if (pauseItem) {
      /* Two reasons a pause can be refused, and they are not the same: no
         session at all, or a session whose current question is mid-grade.
         timer.js owns the second one and writes the reason into `title`. */
      pauseItem.disabled = !open || !pauseBtn || pauseBtn.disabled;
      pauseItem.title = open && pauseBtn ? pauseBtn.title || "" : "";
    }
    if (endItem) endItem.disabled = !open || !_endBtn();
    if (note) note.classList.toggle("hidden", open);
  }

  function _open() {
    _syncItems();
    menu.classList.remove("hidden");
    btn.setAttribute("aria-expanded", "true");
  }

  function _close() {
    menu.classList.add("hidden");
    btn.setAttribute("aria-expanded", "false");
  }

  function _isOpen() {
    return !menu.classList.contains("hidden");
  }

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (_isOpen()) _close();
    else _open();
  });

  /* Anywhere else closes it, including inside the panels — the menu floats
     over the question and the editor, and a click meant for either of them
     should not have to be spent dismissing this first. */
  document.addEventListener("click", (e) => {
    if (!_isOpen()) return;
    if (notch.contains(e.target)) return;
    _close();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && _isOpen()) {
      _close();
      btn.focus();
    }
  });

  function _proxy(target) {
    return (e) => {
      e.stopPropagation();
      const el = target();
      if (!el || el.disabled) return;
      _close();
      /* The real button, with the real handler. Not a copy of what it does. */
      el.click();
    };
  }

  if (pauseItem) pauseItem.addEventListener("click", _proxy(_pauseBtn));
  if (endItem) endItem.addEventListener("click", _proxy(_endBtn));

  /* Session state changes underneath an open menu — a question submits, the
     block ends on its own clock — so the items re-read it rather than
     trusting what they were told when the menu opened. */
  const _row = document.getElementById("session-status-row");
  if (_row && typeof MutationObserver === "function") {
    new MutationObserver(() => {
      if (_isOpen()) _syncItems();
    }).observe(_row, {
      attributes: true,
      /* `class` is the session opening and closing; `disabled` and `title` are
         timer.js flipping the pause button mid-question while a grade is in
         flight (timer.js `_setPhase`). Watching only `class` left an
         open menu showing the state it had when it opened — greyed out after
         grading finished, or live and silently doing nothing before it did. */
      attributeFilter: ["class", "disabled", "title"],
      subtree: true,
    });
  }

  window.PracticeNotch = {
    close: _close,
    /* The stage ladder's percentage callout can grow far enough along the bar
       to sit under this notch; practice/stage-ladder.js measures both and
       offsets the notch so neither is covered. It needs the element. */
    el: notch,
  };
})();
