/* ================================================================
   LESSON TIMER — fifteen minutes on a lesson page, and a way to put it down

   Seth, 2026-09-22: "make it such that you have a limit of 15 minutes to look
   at each lesson … so that you don't get stuck on a lesson for too long."

   A lesson screen was the one surface in the app with NO clock. The drill
   behind it is timed by its concept's cap (practice/timer.js), an ARENA
   exercise by the notebook's own "spend up to" line (practice/exercise-
   timer.js), a placement probe by a flat 2:00 — and the lesson the gate shows
   before all of them ran until the learner pressed Continue, which on a hard
   concept is a screen you can lose an afternoon to. The session clock is not
   even counting while it is up: practice/timer.js is in its `loading` phase,
   `#session-countdown` reads `--:--`, and that is what the notch was
   mirroring.

   FIFTEEN MINUTES PER CONCEPT PAGE, not per lesson. A lesson is often several
   pages ("Concept 2 of 4") and each one is its own thing to be stuck on; a
   budget spread across the whole sequence would give a four-page lesson under
   four minutes a page, which is a deadline rather than a backstop. The same
   reasoning ExerciseTimer's FALLBACK_SECS is written on, and the same number.

   AT 0:00 IT PRESSES CONTINUE. Not a notice, not a frozen clock — the point
   is to unstick, and everything else leaves the learner exactly where they
   were. It clicks the page's OWN `#lesson-continue-btn`, so timing out and
   pressing the button are one path: the same exposure write, the same `worked`
   rung credit, the same XP. The readiness gate can still bring the page back
   as a refresher later (practice/lessons.js `_pendingSteps`), which is what
   makes auto-continue safe: a page read badly is not a page lost.

   WHAT IT OWNS
     • the clock and the Pause button in the lesson's own actions row — it
       renders both into the `#lesson-clock-row` host that practice/lessons.js
       leaves for it, the way ExerciseTimer renders `.dd-ex-live` into a block
     • the per-page remaining-time record in localStorage, so a pause (or a
       reload) comes back to the time that was left rather than a fresh 15:00
     • `dd-lesson-timer:tick` / `:change`, which is all practice/notch-menu.js
       needs to MIRROR this clock the way it mirrors the other three

   WHAT IT DOES NOT OWN
     Advancing. It clicks a button practice/lessons.js wrote and wired; it does
     not mark exposure, credit a rung, or know what a lesson is made of.
     Pausing a SESSION. Inside a practice session, Pause means what it means
     everywhere else in this app — save and drop back to the idle screen, the
     behaviour in practice/timer.js — so the button forwards to
     `PracticeSession.pauseFromLesson()` and this file only stops counting.

   🔴 A HIDDEN TAB DOES NOT SPEND THE BUDGET. Fifteen minutes to LOOK at a
   lesson, and a tab switched away from is not being looked at — so the clock
   goes down on `visibilitychange` and comes back up on return. Without it, a
   learner who stepped away came back to a lesson marked read (expiry presses
   Continue, which writes exposure and credits the rung) at whatever moment a
   throttled background interval happened to fire. An explicit Pause is a
   different thing and survives the round trip; only a clock this handler put
   down is picked back up.

   🔴 REMAINING SECONDS ARE PERSISTED, NOT A DEADLINE. ExerciseTimer saves a
   `deadline` because an ARENA exercise is an attempt and a clock left running
   overnight has genuinely run out. This one is a backstop, not an attempt: a
   tab left open until morning must not auto-continue a lesson at 03:00 and
   tell the learner they read it. Saving what was LEFT means a night away
   costs nothing, which is the only behaviour worth defending here.
   ================================================================ */

const LessonTimer = (() => {
  "use strict";

  /* The same fifteen minutes ExerciseTimer falls back to when a notebook makes
     no recommendation, and for the same reason: long enough that a learner
     working through a hard concept never feels chased, short enough that an
     unattended page is not still on the clock an hour later. */
  const LESSON_SECS = 15 * 60;
  const LOW_SECS = 60;

  const SAVE_KEY = () =>
    `${typeof getPracticeStorageKey === "function" ? getPracticeStorageKey() : "practice_progress_guest"}_lesson_timer`;

  /* { key, kc, title, deadline, remaining, paused, interval, onExpire } */
  let live = null;

  const _mmss = (secs) => {
    const s = Math.max(0, Math.round(secs));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };

  /* 🔴 `PracticeSession` is a top-level `const` in practice/timer.js and a
     classic script's top-level `const` is NOT a property of `window` — the
     trap practice/notch-menu.js, practice/kc-practice.js and this file all
     have to document because the optional chain through `window.` answers
     `undefined` in silence. Both scripts share the global lexical scope, so
     the bare name resolves; the `typeof` guard is for a page that never
     loaded timer.js at all. */
  const _session = () =>
    typeof PracticeSession !== "undefined" ? PracticeSession : window.PracticeSession;

  /* ── the record that survives a pause or a reload ──────────────── */

  const _readAll = () => {
    try {
      const raw = JSON.parse(localStorage.getItem(SAVE_KEY()) || "null");
      return raw && typeof raw === "object" ? raw : {};
    } catch (_) {
      return {};
    }
  };

  const _writeAll = (all) => {
    try {
      if (!Object.keys(all).length) localStorage.removeItem(SAVE_KEY());
      else localStorage.setItem(SAVE_KEY(), JSON.stringify(all));
    } catch (_) {
      /* private mode — the clock simply does not survive a reload */
    }
  };

  const _save = () => {
    if (!live) return;
    const all = _readAll();
    all[live.key] = { remaining: _remaining(), paused: !!live.paused, at: Date.now() };
    _writeAll(all);
  };

  /* A tick does not need the disk. Writing every second is a synchronous
     read-parse-serialize-write about nine hundred times per lesson page
     (codex, 2026-09-22) to protect against losing at most a few seconds, and
     the moments that actually matter — a pause, a stop, an expiry, the tab
     going away — all call `_save` outright. Checkpoint coarsely in between,
     and finely over the last half-minute where a lost second is visible. */
  const CHECKPOINT_EVERY = 15;
  const _saveCheckpoint = (secs) => {
    if (secs <= 30 || secs % CHECKPOINT_EVERY === 0) _save();
  };

  const _clearSaved = (key) => {
    const all = _readAll();
    if (!(key in all)) return;
    delete all[key];
    _writeAll(all);
  };

  /* A record this old is not a lesson the learner set down; it is one they
     walked away from. Coming back a week later to four minutes on a page they
     no longer remember reading would be the clock working against them, which
     is the opposite of what it is for. */
  const STALE_MS = 12 * 60 * 60 * 1000;

  /* What this page has left. A key with no record — or one that has gone stale
     — is a page being read fresh and gets the whole budget. A record at or
     below zero is a page whose clock already ran out; it comes back at zero
     and expires on the first tick, which is what would have happened had the
     tab simply stayed open.

     🔴 A record only ever comes from a PAUSE or a tick of this same page, so
     "resume where I was" is the only thing it can mean. Nothing else writes
     it, and `_clearSaved` runs the moment a page is finished with. */
  const _startingSecs = (key) => {
    const saved = _readAll()[key];
    if (!saved || !Number.isFinite(saved.remaining)) return LESSON_SECS;
    if (Number.isFinite(saved.at) && Date.now() - saved.at > STALE_MS) return LESSON_SECS;
    return Math.max(0, Math.min(LESSON_SECS, saved.remaining));
  };

  /* ── the clock ─────────────────────────────────────────────────── */

  const _remaining = () => {
    if (!live) return 0;
    if (live.paused) return Math.max(0, Math.round(live.remaining));
    return Math.max(0, Math.round((live.deadline - Date.now()) / 1000));
  };

  const _stopTick = () => {
    if (live && live.interval) {
      clearInterval(live.interval);
      live.interval = null;
    }
  };

  const _announce = (name, detail) =>
    document.dispatchEvent(new CustomEvent(`dd-lesson-timer:${name}`, { detail }));

  const _paint = () => {
    if (!live) return;
    const secs = _remaining();
    const text = _mmss(secs);
    const low = !live.paused && secs <= LOW_SECS;
    /* Kept as PAINTED, not recomputed on read: the notch copies these two and
       recomputing would let the mirror land a second off across a rounding
       boundary. Same contract as ExerciseTimer's `live.text` / `live.low`. */
    live.text = text;
    live.low = low;
    const clock = document.getElementById("lesson-clock");
    if (clock) {
      if (clock.textContent !== text) clock.textContent = text;
      clock.classList.toggle("lesson-clock--low", low);
      clock.classList.toggle("lesson-clock--paused", !!live.paused);
      clock.title = live.paused
        ? "Paused. The rest of this page's fifteen minutes is kept."
        : "Time left on this page. At 0:00 it moves you on, the same as pressing Continue.";
    }
    _announce("tick", { kc: live.kc, secs, paused: !!live.paused });
  };

  const _tick = () => {
    _stopTick();
    _paint();
    _save();
    if (!live || live.paused) return;
    live.interval = setInterval(() => {
      if (!live) return;
      _paint();
      _saveCheckpoint(_remaining());
      if (_remaining() <= 0) _expire();
    }, 1000);
  };

  /* Time is up. The page's own Continue is what moves the learner on, so this
     clicks it rather than repeating what it does — the click writes exposure,
     credits the `worked` rung and either draws the next concept page or hands
     the drill back (practice/lessons.js `showPage` / `finishAll`).

     The clock is torn down BEFORE the click, and its record cleared with it:
     `showPage` starts the next page's clock on the way through, and a `live`
     left standing here would be stopped by that start and take the new one's
     interval with it. */
  const _expire = () => {
    if (!live) return;
    const { key, kc, onExpire } = live;
    _stopTick();
    live = null;
    _clearSaved(key);
    _announce("change", { kc, reason: "expired" });
    try {
      if (typeof onExpire === "function") onExpire();
    } catch (err) {
      console.warn("[lesson-timer] could not move on from an expired lesson:", err);
    }
  };

  /* ── the control in the lesson's actions row ───────────────────── */

  /* Rendered here rather than in the page HTML because this file owns what the
     button SAYS, and what it says depends on what pausing means right now: in
     a session it is "Pause & save" and it drops the learner back to the idle
     screen; on the sessionless `?lesson=<kc>` route it is "Exit lesson" and
     it leaves the page (`_exitLesson`). One button either way, which is what lets practice/notch-menu.js
     proxy to it the way it proxies to everything else. */
  const _renderControl = () => {
    const host = document.getElementById("lesson-clock-row");
    if (!host) return;
    host.innerHTML =
      '<span class="lesson-clock" id="lesson-clock" role="timer" aria-live="off">--:--</span>' +
      '<button type="button" class="lesson-pause-btn" id="lesson-pause-btn"></button>';
    host.querySelector("#lesson-pause-btn").addEventListener("click", () => _pauseClicked());
    _paintControl();
  };

  const _inSession = () => {
    const row = document.getElementById("session-status-row");
    return !!row && !row.classList.contains("hidden");
  };

  const _paintControl = () => {
    const btn = document.getElementById("lesson-pause-btn");
    if (!btn) return;
    if (!_inSession()) {
      btn.textContent = "Exit lesson";
      btn.title = "Leave this lesson. Its page keeps the time it has left for when you come back.";
      return;
    }
    if (live && live.paused) {
      btn.textContent = "Resume reading";
      btn.title = "Start this page's clock again where you left it.";
      return;
    }
    btn.textContent = "Pause & save";
    btn.title =
      "Pause and save. You come back to this lesson page, on the time it has left.";
  };

  /* THE ONE PLACE PAUSING A LESSON IS DECIDED.

     In a session, Pause means what it means on every other practice screen —
     practice/timer.js writes the resume snapshot and puts the idle screen back
     — and a second meaning on this one screen would be a second thing to
     learn. `pauseFromLesson()` is that call; the clock is stopped and saved
     first, so the time left travels with the paused session and the lesson
     comes back on it (practice/timer.js `resume()` re-shows the gate).

     Off a session the button LEAVES the lesson (`_exitLesson`). It used to
     freeze the clock and leave the page up, which is not what the square next
     to the notch clock means anywhere else. Seth, 2026-09-22: "when I press
     the button to exit … it doesn't actually exit the lesson like it should." */
  const _pauseClicked = () => {
    if (!live) return;
    /* Before the paused check: a sessionless page can be paused (a record
       saved paused, or one frozen by the old off-session Pause), and Resume
       there would be one more click that does not get the learner out. */
    if (!_inSession()) {
      _exitLesson();
      return;
    }
    if (live.paused) {
      resume();
      return;
    }
    pause();
    const ok = _session()?.pauseFromLesson?.();
    // The session refused (it was not running after all). The frozen clock and
    // the Resume button are still the honest state, so leave them.
    if (!ok) {
      _paintControl();
      return;
    }
    /* 🔴 THE PAUSED SESSION TAKES THE LESSON OFF THE SCREEN — `.practice-split`
       is `display: none` while `#page-practice` is `session-idle` — so there
       must be no live clock left behind it. Held as merely PAUSED, the notch
       went on offering it (a paused lesson keeps its place on the tab), and
       the square would have started a clock on a page nobody could see, whose
       expiry then pressed a Continue button under the idle dial.

       The record is kept (`clearSaved: false`), which is the whole point: the
       re-shown lesson starts on the time it had left. */
    stop({ clearSaved: false });
  };

  /* LEAVE A SESSIONLESS LESSON. Off a session a lesson is only ever the
     `?lesson=<kc>` route: either the Knowledge Graph's ⤢ overlay (an
     `&embed=1` iframe over the graph) or that URL opened on its own. The
     record is kept (`clearSaved: false`) so reopening the page resumes on the
     time it had left, exactly as a paused session's lesson does.

     In the overlay the way out is the graph's own Minimize, clicked rather
     than copied: it tears the iframe down and recolours the node from what
     the frame wrote. Same origin, so the parent's DOM is reachable; a frame
     that is not the graph's (or a cross-origin embedder) falls through. On
     its own the page goes to the app's front door with the `lesson` and
     `embed` params dropped, so a reload does not reopen the lesson. */
  const _exitLesson = () => {
    stop({ clearSaved: false });
    try {
      if (window.parent && window.parent !== window) {
        const min = window.parent.document.getElementById("kg-maxi-min");
        if (min) {
          min.click();
          return;
        }
      }
    } catch (_) {
      /* cross-origin embedder: leave by navigating this frame instead */
    }
    const url = new URL(window.location.href);
    url.searchParams.delete("lesson");
    url.searchParams.delete("embed");
    window.location.assign(url.pathname + url.search + url.hash);
  };

  /* ── the API ───────────────────────────────────────────────────── */

  /* Called once per lesson PAGE, from practice/lessons.js `showPage`. `key` is
     the page's identity — the same `kc#segIndex` the lesson notebook uses for
     its kernel session — and it is what the saved remaining time is filed
     under, so re-entering the same page resumes its clock and moving to the
     next one starts a fresh fifteen minutes. */
  const start = ({ key, kc, title, onExpire }) => {
    if (!key) return false;
    /* A page that is already on the clock keeps the clock it has rather than
       restarting it — a re-render of the same page (the Colab index landing
       late), or the same page being drawn again after a paused session was
       resumed. In that second case it comes back RUNNING: the learner just
       pressed Continue practicing, which is them saying they are reading
       again, and making them press Resume as well would be asking twice. */
    if (live && live.key === key) {
      /* 🔴 TAKE THE NEW CALLBACK. A re-render hands over a NEW
         `#lesson-continue-btn`, and the closure the old call left behind
         holds the detached one — clicking that at 0:00 does nothing at all,
         so the page the clock exists to move on from would sit there
         forever. The Colab edition re-renders exactly this way when its
         notebook index lands late (practice/lessons.js
         `_redrawWhenColabIndexLands`). */
      if (typeof onExpire === "function") live.onExpire = onExpire;
      _renderControl();
      if (live.paused) resume();
      else {
        _paintControl();
        _paint();
      }
      return true;
    }
    stop();
    const secs = _startingSecs(key);
    live = {
      key,
      kc: kc || null,
      title: title || null,
      deadline: Date.now() + secs * 1000,
      remaining: secs,
      paused: false,
      interval: null,
      onExpire: typeof onExpire === "function" ? onExpire : null,
    };
    _renderControl();
    _announce("change", { kc: live.kc, reason: "start" });
    _tick();
    return true;
  };

  const pause = () => {
    if (!live || live.paused) return false;
    live.remaining = _remaining();
    live.paused = true;
    _stopTick();
    _save();
    _paint();
    _paintControl();
    _announce("change", { kc: live.kc, reason: "pause" });
    return true;
  };

  const resume = () => {
    if (!live || !live.paused) return false;
    live.paused = false;
    live.deadline = Date.now() + live.remaining * 1000;
    _paintControl();
    _announce("change", { kc: live.kc, reason: "resume" });
    _tick();
    return true;
  };

  /* Stop counting. `clearSaved` is the difference between a page that is DONE
     — Continue pressed, or the gate handing the column back — and one that is
     merely off screen for now. Clearing is the default because every caller in
     practice/lessons.js is the first kind; a pause keeps its record through
     `pause()` above, which never comes through here. */
  const stop = ({ clearSaved = true } = {}) => {
    if (!live) return false;
    const { key, kc } = live;
    if (!clearSaved) _save();
    _stopTick();
    live = null;
    if (clearSaved) _clearSaved(key);
    _announce("change", { kc, reason: "stop" });
    return true;
  };

  /* DROP EVERY SAVED PAGE, live or not.

     🔴 `stop()` cannot do this job, and that is exactly how the record leaked
     (codex, 2026-09-22): pausing a SESSION from a lesson ends with
     `stop({clearSaved: false})`, which sets `live` to null and keeps the
     record on purpose — so a later `PracticeSession.discard()` calling
     `stop()` found nothing live and returned without touching a thing. The
     discarded session's lesson then came back inside the twelve-hour window
     on its shortened clock, for a block nobody is waiting on any more.

     Whole-store, not per key: the caller discarding a session does not know
     which page was on screen when it was paused, and a lesson record is only
     ever worth keeping for a session that still exists. */
  const forget = () => {
    if (live) stop();
    _writeAll({});
  };

  /* For the mirror in practice/notch-menu.js. `isLive` is "a lesson clock
     exists", paused or not — the notch keeps showing a paused lesson's time
     rather than falling back to the session's `--:--`, because the paused
     clock is the true thing on screen. `isRunning` is "and it is counting". */
  const isLive = () => !!live;
  const isRunning = () => !!live && !live.paused;
  const isPaused = () => !!live && !!live.paused;
  const clockText = () => (live ? live.text || _mmss(_remaining()) : "");
  const isLow = () => !!live && !!live.low;
  const remainingSecs = () => (live ? _remaining() : null);
  const activeLesson = () => (live ? { kc: live.kc, title: live.title } : null);

  /* 🔴 A HIDDEN TAB IS NOT A LEARNER READING, so it does not spend the budget.
     Seth's limit is fifteen minutes to LOOK at a lesson; the clock kept
     counting behind a switched-away tab, and because expiry presses Continue
     it would mark the page read — exposure written, `worked` rung credited —
     for a learner who was somewhere else entirely (codex, 2026-09-22). Worse,
     a backgrounded interval is throttled to a callback a minute, so it landed
     at an arbitrary moment rather than at 0:00.

     `hiddenPause` is what separates this from a pause the learner ASKED for:
     coming back resumes only a clock that this handler put down. An explicit
     Pause survives the tab going away and coming back, which is the whole
     point of having pressed it. */
  document.addEventListener("visibilitychange", () => {
    if (!live) return;
    if (document.hidden) {
      if (live.paused) return;
      live.hiddenPause = true;
      pause();
    } else if (live.hiddenPause) {
      live.hiddenPause = false;
      resume();
    }
  });

  /* The record is checkpointed coarsely while a page ticks, so the last few
     seconds are written here rather than lost to a close or a reload. */
  window.addEventListener("pagehide", () => {
    if (live) _save();
  });

  /* The session row opening or closing changes what the button SAYS (Pause &
     save vs Pause), and it can happen under a live lesson — `?lesson=<kc>`
     starts sessionless and `KcPractice.start()` opens a session behind the
     page. Cheap, and only ever while a lesson is on screen. */
  if (typeof MutationObserver === "function") {
    document.addEventListener("DOMContentLoaded", () => {
      const row = document.getElementById("session-status-row");
      if (!row) return;
      new MutationObserver(() => {
        if (live) _paintControl();
      }).observe(row, { attributes: true, attributeFilter: ["class"] });
    });
  }

  return {
    start,
    pause,
    resume,
    stop,
    forget,
    isLive,
    isRunning,
    isPaused,
    clockText,
    isLow,
    remainingSecs,
    activeLesson,
    LESSON_SECS,
  };
})();

window.LessonTimer = LessonTimer;
