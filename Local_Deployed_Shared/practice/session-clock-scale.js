/* ================================================================
   THE CLOCK MULTIPLIER — one factor, chosen BEFORE the block, applied to
   every problem's OWN clock

   Seth, 2026-09-14: "a setting before practice that you can modify that
   lets you create a multiplier for less time. so you would give .5 for
   half as much time as you would usually get."

   🔴 THIS DOES NOT REVERSE 2026-09-09. The number a problem gets is still
   the PROBLEM's — its concept's cap in lessons/placement_time_caps.json,
   read off the question by practice/session-clock.js. What this file
   holds is a SCALE on that number, never a number of its own: a 5:00
   drill at 0.5 is 2:30, a 20:00 concept at 0.5 is 10:00, and the
   ordering between concepts — the thing the table encodes — is exactly
   preserved. The 08-28 picker replaced the table with one flat choice;
   this multiplies it.

   RANGE [0.1, 4]. It started as "for less time" (≤ 1). Seth, 2026-09-25:
   "allow adjusting the time multiplier upwards, not just downwards ...
   increasing it to 2". So it loosens too, up to 4× — a 40:00 LeetCode
   hard at 2 is 80:00. The placement stays unscaled (below), so what it
   charges is still comparable. 1 is the default and means "the table as
   written"; an account
   that never touches the input is on exactly the 09-09 behaviour. The
   floor stops a slip of the keyboard (0.01, 0) from producing a question
   that expires as it renders; the rounding downstream keeps every scaled
   clock at ≥ 1 s regardless.

   🔴 THE PLACEMENT IS NOT SCALED. A probe is timed by PlacementTimer
   (timer.js's `_answerSecsFor` asks it first) and the server CHARGES the
   same number; a learner-tightened placement would not be comparable
   evidence. An ARENA exercise session is not scaled either — it brings
   its own answer/review numbers (`sessionConfig`), which timer.js reads
   before it asks SessionClock at all.

   WHERE IT IS SET: the idle surface, under the clock sentence
   (#session-clock-scale in index.html). Only reachable between blocks,
   which is what "before practice" asks for and what keeps a running
   countdown from changing under the learner's hands — a paused
   question's snapshot carries the scaled allowance it was served with,
   and the resume clamps to it (session-snapshot.js).

   Per ACCOUNT, like the pause snapshot and the progress record: the key
   hangs off getPracticeStorageKey(), read lazily because `authEmail` is
   not resolved when this file parses. index.html loads this AFTER
   storage.js and BEFORE session-clock.js, which reads `factor()`.
   ================================================================ */

(function initSessionClockScale() {
  const DEFAULT = 1;
  const MIN = 0.1;
  const MAX = 4;

  const _key = () => {
    try {
      return `${getPracticeStorageKey()}_clock_scale`;
    } catch (_) {
      return "practice_progress_guest_clock_scale";
    }
  };

  const _stored = () => {
    try {
      return localStorage.getItem(_key());
    } catch (_) {
      return null;
    }
  };

  /* A factor is a finite number inside the range, or nothing. "0.5", 0.5
     and " .5 " all clean to 0.5; "", "abc", 0, -1, 5, Infinity and true all
     clean to null — the caller decides whether null means "default" (a read)
     or "refuse" (a write). Rounded to 2 places so 0.333333 does not print
     as a different number than it stores. */
  const clean = (raw) => {
    const n = typeof raw === "number" ? raw : parseFloat(String(raw ?? "").trim());
    if (!Number.isFinite(n) || n < MIN || n > MAX) return null;
    return Math.round(n * 100) / 100;
  };

  /* 🔴 A CHOICE STORAGE WOULD NOT KEEP IS STILL THE CHOICE, for this page
     load — the same trap codex found in the 08-28 picker. `set` writes and
     reads back; when the value did not land (private mode, quota, site data
     blocked) it is held here with what storage said at the time. The moment
     storage stops saying that, something else really wrote, and storage
     wins.

     🔴 HELD AGAINST THE KEY IT WAS WRITTEN UNDER, not just the raw value.
     Two accounts' keys both read `null` when neither has a factor, so a
     volatile 0.5 refused under one account would otherwise be in force for
     the next one signed in on the same page. Codex, 2026-09-14. */
  let volatile = null;

  /* Read fresh every time, never cached at load: sign-in changes the key
     underneath us. One localStorage hit per question, against a clock that
     ticks every second anyway. */
  const factor = () => {
    const key = _key();
    const raw = _stored();
    if (volatile && key === volatile.key) {
      if (raw === volatile.over) return volatile.value;
      volatile = null;
    }
    return clean(raw) ?? DEFAULT;
  };

  const listeners = new Set();

  /* Returns the factor now in force. An unusable value is REFUSED (the
     current factor stands and is returned) rather than snapped to the
     default: the input is the caller, and rewriting the learner's 0.5 to 1
     because they typed "abc" on the way to "0.75" is the worse answer. */
  const set = (raw) => {
    const value = clean(raw);
    if (value === null) return factor();
    /* Same value, already persisted: nothing to do. Same value held only in
       `volatile` is NOT the same thing — the write is retried, so a store
       that has since recovered gets the choice on the next commit rather
       than only after the learner picks something else first. Codex,
       2026-09-14. */
    if (value === factor() && !volatile) return value;
    const key = _key();
    const before = _stored();
    let persisted = false;
    try {
      localStorage.setItem(key, String(value));
      persisted = _stored() === String(value);
    } catch (_) {
      persisted = false;
    }
    volatile = persisted ? null : { value, over: before, key };
    listeners.forEach((fn) => {
      try {
        fn(value);
      } catch (_) {}
    });
    return value;
  };

  const subscribe = (fn) => {
    if (typeof fn !== "function") return () => {};
    listeners.add(fn);
    return () => listeners.delete(fn);
  };

  /* Apply the factor to a problem's clock. Whole seconds, never below 1 —
     0.1 × 5 s would otherwise be a 0 that `_tick` reads as expired on
     render. `null` (no limit) passes through untouched: half of ∞ is ∞. */
  const apply = (secs) => {
    if (typeof secs !== "number" || !Number.isFinite(secs)) return secs;
    return Math.max(1, Math.round(secs * factor()));
  };

  /* THE INPUT on the idle screen. Bound here rather than in session-idle.js
     so the number and its one control live in the same file; the markup
     holds no default of its own. Committed on `change` (blur / Enter), not
     on every keystroke — "0." on the way to "0.5" is not a choice. A refused
     value snaps the box back to the factor in force so it never shows a
     number the clock is not using. */
  const _bind = () => {
    const input = document.getElementById("session-clock-scale");
    if (!input) return;
    /* The range the browser enforces is the range this file enforces — one
       source, so the spinner's bounds and `clean` cannot drift apart. */
    input.min = String(MIN);
    input.max = String(MAX);
    const paint = (value) => {
      input.value = String(value);
    };
    paint(factor());
    input.addEventListener("change", () => paint(set(input.value)));
    /* The key can change without a class flip (an account swap while the
       Practice tab is already up); showing the factor in force the moment
       the learner reaches for the box is the cheap guarantee. */
    input.addEventListener("focus", () => paint(factor()));
    /* Sign-in swaps the account (and so the key) after this ran; the
       storage key is re-read on every `factor()` call, so a repaint on
       auth change is enough. app.js dispatches nothing for it today, so
       the idle screen's own redraw is the hook: it toggles `session-idle`
       on #page-practice on every pause/start. */
    const page = document.getElementById("page-practice");
    if (page && typeof MutationObserver === "function") {
      new MutationObserver(() => {
        if (page.classList.contains("session-idle")) paint(factor());
      }).observe(page, { attributes: true, attributeFilter: ["class"] });
    }
    subscribe(paint);
  };

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", _bind, { once: true });
    } else {
      _bind();
    }
  }

  window.SessionClockScale = { DEFAULT, MIN, MAX, clean, factor, set, subscribe, apply };
})();
