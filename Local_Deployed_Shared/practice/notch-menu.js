/* ================================================================
   THE NOTCH — session controls, off the page until asked for

   The session row used to sit above the question: progress, phase, a
   countdown, Pause & exit and End session, all of it on screen for every
   second of every question. Four pieces of chrome for two buttons anyone
   presses twice a session, in the band of the page the learner reads the
   question in.

   So the buttons moved into a notch straddling the seam between the two
   panels — a tab with three dots, and a menu that opens under it.

   The COUNTDOWN came back with the second pass. Hiding the row took the clock
   off the screen while leaving it running: answer time still auto-submitted
   with nothing counting down to it. The tab carries it now — mirrored, never
   re-implemented (see `_syncClock`).

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
  const clock = document.getElementById("practice-notch-clock");
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

  /* The clock on the tab is a MIRROR of #session-countdown, not a timer.
     timer.js owns the only clock there is — the one that auto-submits — and a
     second interval here would drift from it within a question and disagree
     with it at the moment that matters. So this copies three facts and
     computes none of them: the text, whether the session is running at all,
     and which phase's time is being counted.

     `--low` is timer.js's own last-30-seconds class, and it is applied only
     while ANSWERING: review time running out costs nothing, so painting it
     like a deadline would teach the wrong urgency. */
  function _syncClock() {
    if (!clock) return;
    const open = _sessionOpen();
    const tab = document.getElementById("practice-notch-tab");
    const srPhase = document.getElementById("practice-notch-phase");
    clock.classList.toggle("hidden", !open);
    if (!open) {
      /* Nothing is being counted, so nothing may still SAY it is. The clock
         itself is hidden, but the tooltip and the screen-reader phase sit on
         the tab, which stays — and a dots-only tab that still reads
         "Reviewing" is describing a session that ended. */
      clock.classList.remove(
        "practice-notch-clock--review",
        "practice-notch-clock--low",
      );
      if (tab) tab.removeAttribute("title");
      if (srPhase) srPhase.textContent = "";
      return;
    }
    const src = document.getElementById("session-countdown");
    const row = document.getElementById("session-status-row");
    const review = !!row && row.classList.contains("session-status--review");
    const text = src ? src.textContent.trim() : "";
    if (clock.textContent !== text) clock.textContent = text;
    clock.classList.toggle("practice-notch-clock--review", review);
    clock.classList.toggle(
      "practice-notch-clock--low",
      !review && !!src && src.classList.contains("session-countdown--low"),
    );
    /* The phase WORD is off the screen with the rest of the row. It rides in
       two places, for two readers: the tab's tooltip, so the clock's colour is
       not the only thing carrying it for a sighted learner, and a clipped
       span, so it is carried at all for a screen reader — `--review` and
       `--low` are colour, and colour is not in the accessibility tree. */
    const phase = document.getElementById("session-phase");
    const phaseText = phase ? phase.textContent.trim() : "";
    if (tab) tab.title = phaseText;
    if (srPhase && srPhase.textContent !== phaseText) {
      srPhase.textContent = phaseText;
    }
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
    notch.classList.add("is-open");
  }

  function _close() {
    menu.classList.add("hidden");
    btn.setAttribute("aria-expanded", "false");
    notch.classList.remove("is-open");
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
      _syncClock();
      if (_isOpen()) _syncItems();
    }).observe(_row, {
      attributes: true,
      /* `class` is the session opening and closing, and the review tint;
         `disabled` and `title` are timer.js flipping the pause button
         mid-question while a grade is in flight (timer.js `_setPhase`).
         Watching only `class` left an open menu showing the state it had when
         it opened — greyed out after grading finished, or live and silently
         doing nothing before it did. */
      attributeFilter: ["class", "disabled", "title"],
      /* The countdown is a TEXT NODE that timer.js rewrites once a second.
         `characterData` alone does not see it — assigning `textContent`
         REPLACES the node rather than editing it, which is a childList
         record — so both are needed to mirror a ticking clock. */
      characterData: true,
      childList: true,
      subtree: true,
    });
  }

  /* The row can already be open when this runs — a resumed session unhides it
     from timer.js's own init, and load order between the two is not a thing
     to depend on. */
  _syncClock();

  window.PracticeNotch = {
    close: _close,
    /* Kept for callers that need to place themselves around this tab. The
       stage ladder used to be one — its percentage reading hung under the bar
       and could land here — but that reading is a tab ON the bar now, inside
       the strip, and nothing measures against the notch any more. */
    el: notch,
    /* Exposed for tests and for anything that changes the session out of
       band; the mirror is otherwise driven entirely by the observer. */
    syncClock: _syncClock,
  };
})();
