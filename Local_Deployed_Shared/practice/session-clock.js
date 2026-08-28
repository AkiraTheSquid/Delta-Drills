/* ================================================================
   THE PER-QUESTION ALLOWANCE — one number, chosen BEFORE the block

   🔴 THIS REVERSES 2026-08-23. Both allowances used to be constants in
   practice/timer.js and the panel that set them was deleted, on Seth's
   instruction: "it's a predetermined timer that they don't control".
   Seth, 2026-08-28: "I can change the amount of time that I have per
   problem before I start the practice so that I actually have more time
   to read the problems and the lessons ... Or I can disable the timer
   entirely." The reason the old rule broke is worth writing down: the
   lesson gate does NOT hold the clock (nothing in lessons.js calls
   PracticeSession.holdClock), so a first-encounter lesson is read on the
   answer clock, and 02:00 has to cover reading a concept AND writing the
   answer to a question about it.

   WHAT THE LEARNER PICKS IS ONE NUMBER, and it is what EACH STEP gets —
   answering and reviewing alike, per QUESTION, exactly as the two 02:00
   constants were. A block still has no length, and there is still no End
   session; pause and resume remain the only two states.

   🔴 "No limit" IS AN OPTION, NOT A BIG NUMBER. `secs: null` is the whole
   representation, and timer.js branches on it: no interval, no expiry, no
   auto-submit. A 99:00 preset would look the same for an hour and then
   submit someone's unfinished work in the middle of a sentence.

   🔴 THE PLACEMENT IS NOT ON THIS CLOCK. Every probe gets the placement's
   own fixed 2:00 (practice/placement-timer.js, and timer.js's
   `_answerSecsFor`) — a test whose questions are timed differently per
   learner does not compare them, which is the whole point of it.

   WHERE IT IS CHOSEN: the idle surface, next to Continue practicing
   (practice/session-idle.js renders the picker; #question-clock-picker in
   index.html is its mount). The picker is only reachable between blocks,
   which is what "before I start the practice" asks for and also what
   keeps a running countdown from changing under the learner's hands.
   ================================================================ */

(function initSessionClock() {
  /* The presets, in the order they are drawn. 02:00 stays the default: it
     is what every learner has been on since 2026-08-23, so the shipped
     behaviour of an account that never opens the picker does not move. */
  const OPTIONS = [
    { id: "1m", secs: 60, label: "1:00" },
    { id: "2m", secs: 120, label: "2:00" },
    { id: "5m", secs: 300, label: "5:00" },
    { id: "10m", secs: 600, label: "10:00" },
    { id: "20m", secs: 1200, label: "20:00" },
    { id: "off", secs: null, label: "No limit" },
  ];
  const DEFAULT_ID = "2m";

  const _byId = (id) => OPTIONS.find((o) => o.id === id) || null;

  /* Per ACCOUNT, like every other thing this page remembers — the session
     snapshot (`..._session`) and the progress record both hang off this same
     key, so a shared browser does not hand one learner another's rules.
     Called lazily: `getPracticeStorageKey` reads `authEmail`, which is not
     resolved at the moment this file parses. */
  const _key = () => {
    try {
      return `${getPracticeStorageKey()}_clock`;
    } catch (_) {
      return "practice_progress_guest_clock";
    }
  };

  const _stored = () => {
    try {
      return localStorage.getItem(_key());
    } catch (_) {
      return null;
    }
  };

  /* 🔴 A CHOICE STORAGE WOULD NOT KEEP IS STILL THE CHOICE, for this page load.
     `set` writes and then READS BACK; when the value did not land — private
     mode, quota, site data blocked — the id is held here instead, along with
     what storage said at the time. Without this the picker snapped back to the
     old preset the instant it was clicked and the clock went on enforcing it,
     because every read below goes to storage. Codex, 2026-08-28.

     `over` is what makes it safe to hold: the moment storage stops saying what
     it said when the write failed, something else really did write — another
     tab, or a store that started working — and that is a newer statement than
     ours. Cleared, storage wins. */
  let volatileChoice = null;

  /* Read fresh every time rather than caching at load. Sign-in changes the
     key underneath us, and a cached value would then be the previous
     account's choice presented as this one's. Cheap: one localStorage hit
     per question, against a clock that ticks every second anyway. */
  const currentId = () => {
    const raw = _stored();
    if (volatileChoice) {
      if (raw === volatileChoice.over) return volatileChoice.id;
      volatileChoice = null;
    }
    return _byId(raw) ? raw : DEFAULT_ID;
  };

  const current = () => _byId(currentId()) || _byId(DEFAULT_ID);

  const listeners = new Set();

  /* Unknown id is a no-op, not a fallback to the default: this is called
     from a click handler on a rendered option, so an id nobody defined
     means the caller is out of date, and silently rewriting the learner's
     choice to 02:00 is a worse answer than leaving it alone. */
  const set = (id) => {
    if (!_byId(id) || id === currentId()) return false;
    const before = _stored();
    let persisted = false;
    try {
      localStorage.setItem(_key(), id);
      /* Read back rather than trusting the write. A store can accept a
         `setItem` and keep nothing (some private modes do exactly that), and
         a write that silently did not stick is indistinguishable from one
         that threw as far as the next read is concerned. */
      persisted = _stored() === id;
    } catch (_) {
      persisted = false;
    }
    /* Private mode, quota, a browser with storage off. The choice cannot be
       remembered past this page load, but it IS in force for it — refusing
       outright would leave the picker showing an option the clock ignores. */
    volatileChoice = persisted ? null : { id, over: before };
    listeners.forEach((fn) => {
      try {
        fn(current());
      } catch (_) {}
    });
    return true;
  };

  const subscribe = (fn) => {
    if (typeof fn !== "function") return () => {};
    listeners.add(fn);
    return () => listeners.delete(fn);
  };

  /* 🔴 `null` MEANS NO LIMIT AND MUST SURVIVE THE WHOLE PATH. Every caller
     that does arithmetic on these has to ask first — `?? 120` anywhere
     downstream is how "No limit" quietly becomes two minutes. */
  const answerSecs = () => current().secs;
  const reviewSecs = () => current().secs;
  const isUnlimited = () => current().secs === null;

  window.SessionClock = {
    OPTIONS,
    DEFAULT_ID,
    currentId,
    current,
    set,
    subscribe,
    answerSecs,
    reviewSecs,
    isUnlimited,
  };
})();
