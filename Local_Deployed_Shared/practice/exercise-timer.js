/* ================================================================
   EXERCISE TIMER — answer the notebook's own problem, on the clock, IN the
   notebook

   Seth, 2026-09-09: "for the notebook, it shouldn't like hide the problems
   until you press the button to practice it. It should display the problems
   themselves. And then next to the problem, it should say start timer to start
   answering the problem. Whenever you start the timer … it hides everything
   unrelated to that specific problem that you're doing, so if it has problems
   that are prior to it it shouldn't hide them, and it also shouldn't go into
   the alternative format where like it has the question on the left and the
   code that you add on the right. It should just be the current notebook, just
   that everything else is hidden, including the glossary on the left … Then,
   if the timer runs out, it's counted as wrong … But it doesn't automatically
   go to start doing the drills for you. It just says that you got it wrong and
   it displays on the problem whether it recommends that you start doing
   drills."

   Seth, 2026-09-11: the clock does NOT get its own chrome. It used to be a
   pill fixed to the bottom of the viewport (`.dd-nbt-bar`) — "kind of
   idiotic". The clock "should display the timer over the original button
   panel that you clicked on": the exercise's own block swaps its two buttons
   for three controls —

       [ Mark as "I don't know this yet" ]   [ 9:30 ]   [ Pause timer to
         (if no progress is made)                          come back later ]

   — and swaps them back when the clock stops. There is no "I'm done": running
   the exercise's tests is how a solution gets checked, and the clock is only
   the authority on running out.

   AND IT IS ON THE TOPBAR NOTCH TOO — "we should remain consistent. but we can
   do both" (Seth, later the same day). practice/notch-menu.js MIRRORS this
   clock the way it mirrors the session's: it never counts, it copies. This
   file announces every repaint as `dd-exercise-timer:tick` and exposes
   `clockText()` / `isLow()` so the notch has something to copy; the notch's
   pause square proxies a click to this block's own Pause button.

   SO THIS IS NOT A PRACTICE SESSION. Nothing here switches tabs, nothing here
   builds a ladder, nothing here touches `PracticeSession` — the two buttons on
   the block do one thing each, and this is the one that keeps the learner
   where they are. The drill block is still practice/exercise-session.js, still
   on the Practice tab, and it is reached deliberately or not at all.

   WHAT IT OWNS
     • the live panel (`.dd-ex-live`, written INTO the exercise's own
       `.dd-ex-block`; `is-live` on the block is what swaps the buttons out)
     • focus mode: `dd-nb-focus` on the page, `dd-nb-hidden` on the nodes it
       hides, and NOTHING else. Restoring is removing those two classes, so a
       teardown cannot half-restore a notebook.
     • the verdict line it writes into the exercise's own block when the clock
       runs out or the learner marks the exercise as not known yet
     • the pause snapshot — "come back later" means the remaining time is kept
       per notebook + exercise and offered back as "Resume timer" on the same
       block (practice/exercise-session.js reads `pausedFor`)

   WHAT IT DOES NOT OWN
     Grading. A timeout is recorded through `PracticeAPI.recordLocalEval`, the
     same call practice/notebook-view.js makes when a cell prints ✅/❌ — one
     attempt, marked wrong, into the same engine as everything else. There is
     no second scoring system here and there must never be one.

   🔴 PAUSING RECORDS NOTHING AND STOPPING RECORDS NOTHING BUT A MISS. Running
   the exercise's test cell prints a verdict and practice/notebook-view.js
   beacons it; if this file recorded a "done" too, every finished exercise
   would go in twice. The clock is the only thing this file is the authority
   on, so running out — or the learner saying "I don't know this yet" — is the
   only thing it reports.
   ================================================================ */

const ExerciseTimer = (() => {
  "use strict";

  /* ARENA's own words, in the shapes the notebooks actually use:
       "You should spend up to 10-15 minutes on this exercise."
       "You should spend up to ~20 minutes on this exercise."
     The UPPER bound is the budget — Seth: "however much time there is
     according to what they recommend from the notebook" — because the range is
     written as "up to", and taking the lower end would ring the bell on a
     learner who is inside the recommendation. */
  const SPEND_RE = /spend\s+up\s+to\s+~?\s*(\d{1,3})\s*(?:[-–—]\s*~?\s*(\d{1,3}))?\s*min/i;
  /* No recommendation in the markdown. Long enough to be a real attempt at an
     ARENA exercise, short enough that an unattended tab does not sit on a live
     clock all afternoon. */
  const FALLBACK_SECS = 15 * 60;
  /* How far past the exercise's own cell to keep reading for the "spend up to"
     line: ARENA writes it in the heading cell, or in the short prose cell
     directly under it, never further. */
  const LOOKAHEAD = 3;
  const SAVE_KEY = () => `${typeof getPracticeStorageKey === "function" ? getPracticeStorageKey() : "practice_progress_guest"}_nb_timer`;

  let live = null; // { ex, block, page, hidden: [], deadline, tick }

  const _esc = (v) =>
    String(v == null ? "" : v)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");

  const _mmss = (secs) => {
    const s = Math.max(0, Math.round(secs));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };

  const _api = () =>
    (typeof PracticeAPI !== "undefined" ? PracticeAPI : window.PracticeAPI) || null;
  /* 🔴 `apiFetch` is a script-global const, not a window property — the same
     trap practice/kc-practice.js and practice/diagnostic-page.js document. */
  const _fetch = () => (typeof apiFetch !== "undefined" ? apiFetch : window.apiFetch);

  const _sourceCell = (block) => {
    if (block?._sourceCell) return block._sourceCell;
    let node = block?.previousElementSibling;
    while (node?.classList.contains("dd-ex-block")) node = node.previousElementSibling;
    return node;
  };

  // A button follows the heading, not the answer cells. Keep the whole
  // exercise up to the next exercise/section, including its runnable tests.
  const _endCell = (block) => {
    const source = _sourceCell(block);
    const heading = source?.querySelector("h1,h2,h3,h4,h5,h6");
    const level = heading ? Number(heading.tagName.slice(1)) : 6;
    const anchors = new Set(Array.from(block.parentElement.querySelectorAll(".dd-ex-block"), _sourceCell));
    for (let node = block.nextElementSibling; node; node = node.nextElementSibling) {
      if (node.classList.contains("dd-ex-block")) continue;
      const h = node.querySelector("h1,h2,h3,h4,h5,h6");
      if (anchors.has(node) || (h && (Number(h.tagName.slice(1)) <= level || /exercise\s*[-–—:]|^\(\w{1,3}\)\s/i.test(h.textContent)))) return node;
    }
    return null;
  };

  /* ── how long the notebook says this exercise should take ─────── */

  /** Seconds recommended for the exercise whose block this is, read off the
      notebook's own prose. `null` when the notebook does not say. */
  const recommendedSecs = (block) => {
    const end = block && _endCell(block);
    let node = _sourceCell(block);
    for (let i = 0; node && i < LOOKAHEAD; i += 1) {
      const m = SPEND_RE.exec(node.textContent || "");
      if (m) {
        const upper = Number(m[2] || m[1]);
        if (Number.isFinite(upper) && upper > 0) return upper * 60;
      }
      node = node.nextElementSibling;
      while (node?.classList.contains("dd-ex-block")) node = node.nextElementSibling;
      if (node === end) break;
    }
    return null;
  };

  const budgetSecs = (block) => recommendedSecs(block) || FALLBACK_SECS;

  /* ── focus mode ────────────────────────────────────────────────── */

  /* Hide what is not this problem, and NOTHING that is.

     "Prior" is the whole point: the cells above an ARENA exercise are the
     imports it needs, the setup it runs against and the explanation it is
     testing, and a learner who cannot scroll up to them is not doing the
     exercise, they are doing a memory test. So the cut is one-directional —
     later exercises and solution disclosures go; the current exercise's
     prompt, code and tests stay, along with all prior setup cells.

     The other exercises' blocks go in both directions: a "Start timer" button
     on another problem is an invitation to start a second clock, and this file
     runs one. */
  const _enterFocus = (block) => {
    const page = document.getElementById("page-arena-notebook");
    const cells = block.parentElement;
    if (!page || !cells) return null;
    const hidden = [];
    const hide = (node) => {
      if (!node || node.classList.contains("dd-nb-hidden")) return;
      node.classList.add("dd-nb-hidden");
      hidden.push(node);
    };
    const end = _endCell(block);
    let past = false;
    for (const node of Array.from(cells.children)) {
      if (node === block) continue;
      if (node === end) past = true;
      if (past) hide(node);
      else if (node.classList.contains("dd-ex-block")) hide(node);
      else if (node.dataset.role === "details") hide(node);
    }
    // The contents rail — "the glossary on the left" — and anything else the
    // page hangs outside the cell column.
    page.querySelectorAll(".anb-toc").forEach(hide);
    page.classList.add("dd-nb-focus");
    return { page, hidden };
  };

  const _exitFocus = () => {
    if (!live) return;
    live.hidden.forEach((node) => node.classList.remove("dd-nb-hidden"));
    live.page.classList.remove("dd-nb-focus");
  };

  /* ── the live panel, in the block ──────────────────────────────── */

  /* Three controls where the two buttons were. practice/exercise-session.js
     owns the block and its `.dd-ex-row`; this module only ever adds this one
     row and the verdict under it, and `is-live` on the block is the single
     switch that decides which row is on screen (styles/practice/
     exercise-timer.css). */
  const _ensureLive = (block) => {
    let panel = block.querySelector(".dd-ex-live");
    if (panel) return panel;
    panel = document.createElement("div");
    panel.className = "dd-ex-live";
    panel.innerHTML =
      '<button type="button" class="dd-ex-live-give">' +
      'Mark as "I don\'t know this yet"' +
      '<small>(if no progress is made)</small></button>' +
      '<span class="dd-ex-live-clock" role="timer"></span>' +
      '<button type="button" class="dd-ex-live-pause">Pause timer' +
      '<small>to come back later</small></button>';
    block.querySelector(".dd-ex-row")?.insertAdjacentElement("afterend", panel) ||
      block.prepend(panel);
    panel.querySelector(".dd-ex-live-give").onclick = () => stop("gave-up");
    panel.querySelector(".dd-ex-live-pause").onclick = () => pause();
    return panel;
  };

  const _paint = () => {
    if (!live) return;
    const secs = _remaining();
    const clock = live.block.querySelector(".dd-ex-live-clock");
    if (clock) {
      const text = _mmss(secs);
      if (clock.textContent !== text) clock.textContent = text;
    }
    live.block.classList.toggle("is-low", secs <= 60);
    // What was painted, kept for the mirror — not recomputed on read, so the
    // notch cannot land one second off across a rounding boundary.
    live.text = _mmss(secs);
    live.low = secs <= 60;
    // The notch copies this repaint (practice/notch-menu.js). Fired after the
    // block is painted so a listener reading `clockText()` sees the same text.
    document.dispatchEvent(new CustomEvent("dd-exercise-timer:tick", { detail: { ex: live.ex, secs } }));
  };

  /** What the block's clock says right now, for mirrors. "" when idle. */
  const clockText = () => (live ? live.text || _mmss(_remaining()) : "");
  const isLow = () => !!live && !!live.low;

  const _remaining = () => {
    if (!live) return 0;
    return Math.max(0, Math.round((live.deadline - Date.now()) / 1000));
  };

  /* ── surviving a reload, and "come back later" ─────────────────── */

  /* A live clock that a refresh silently swallows is worse than no clock: the
     learner comes back to a notebook that looks idle and has no idea whether
     the attempt counted. Snapshots are keyed per notebook + exercise so
     restoring can only ever re-arm the same problem.

     🔴 ONE RECORD PER EXERCISE, NOT ONE RECORD. Pausing frees the other
     blocks' buttons, so a learner can set exercise A down and start B; a
     single slot (what this was until codex caught it, 2026-09-11) let B's
     `_save()` erase A's remaining time. The store is a map, and only the
     running entry is ever more than one.

     The same record is the pause. `paused: true` + `remaining` is a clock
     the learner set down on purpose; it is NOT re-armed on the next visit —
     the block offers "Resume timer · 9:30 left" and waits. `deadline` alone
     is a clock that was running when the tab went away, and that one comes
     back running (or expired, and says so). */
  const _key = (ex) => `${ex.nb}|${ex.fn}`;

  const _readAll = () => {
    try {
      const raw = JSON.parse(localStorage.getItem(SAVE_KEY()) || "null");
      if (!raw || typeof raw !== "object") return {};
      // The pre-map shape: one record at the top level. Read it as one entry.
      if (raw.fn) return { [_key(raw)]: raw };
      return raw;
    } catch (_) {
      return {};
    }
  };

  const _writeAll = (all) => {
    try {
      if (!Object.keys(all).length) localStorage.removeItem(SAVE_KEY());
      else localStorage.setItem(SAVE_KEY(), JSON.stringify(all));
    } catch (_) { /* private mode — the clock just does not survive a reload */ }
  };

  const _write = (ex, snap) => {
    const all = _readAll();
    if (snap) all[_key(ex)] = snap;
    else delete all[_key(ex)];
    _writeAll(all);
  };

  const _save = () => live && _write(live.ex, {
    nb: live.ex.nb, fn: live.ex.fn, kc: live.ex.kc,
    paused: false, remaining: null, deadline: live.deadline,
  });

  const _savedFor = (ex) => (ex && _readAll()[_key(ex)]) || null;

  /** Seconds left on a clock the learner paused on this exercise, or `null`
      when there is no paused clock for it. What "Resume timer" reads. */
  const pausedFor = (ex) => {
    const saved = _savedFor(ex);
    if (!saved || !saved.paused) return null;
    const secs = Number(saved.remaining);
    return Number.isFinite(secs) && secs > 0 ? secs : null;
  };

  /* ── the clock ─────────────────────────────────────────────────── */

  const _stopTick = () => {
    if (live && live.tick) {
      clearInterval(live.tick);
      live.tick = null;
    }
  };

  const _startTick = () => {
    _stopTick();
    live.tick = setInterval(() => {
      _paint();
      if (_remaining() <= 0) stop("expired");
    }, 500);
  };

  const isRunning = (ex) => {
    if (!live) return false;
    if (!ex) return true;
    return live.ex.fn === ex.fn && live.ex.nb === ex.nb;
  };

  const activeExercise = () => (live ? live.ex : null);

  /* Focus mode off, live panel gone, the block's buttons back. Every way out
     of a running clock goes through here so none of them can half-restore. */
  const _teardown = () => {
    _stopTick();
    _exitFocus();
    live.block.classList.remove("is-live", "is-low");
    live = null;
  };

  /** Start the clock on one notebook exercise. `secs` overrides the notebook's
      own recommendation (the resume and restore paths). */
  const start = (ex, block, secs = null) => {
    // Another clock is up (only reachable through the API — the buttons are
    // disabled while one runs). Set it down with its time rather than losing it.
    if (live) pause("switched");
    const focus = _enterFocus(block);
    if (!focus) return false;
    const budget = Number.isFinite(secs) && secs > 0 ? secs : budgetSecs(block);
    live = {
      ex, block, page: focus.page, hidden: focus.hidden,
      deadline: Date.now() + budget * 1000, tick: null, text: "", low: false,
    };
    _ensureLive(block);
    block.classList.add("is-live");
    _verdict(block, "");
    _paint();
    _startTick();
    _save();
    block.scrollIntoView({ block: "center", behavior: "smooth" });
    document.dispatchEvent(new CustomEvent("dd-exercise-timer:change", { detail: { running: true, ex } }));
    return true;
  };

  /** "Pause timer to come back later." The remaining time is kept, the
      notebook comes back, nothing is recorded. `reason` is "paused" for the
      button and "left" when the page or section went away underneath. */
  const pause = (reason = "paused") => {
    if (!live) return;
    const { ex } = live;
    const remaining = _remaining();
    // Nothing left to come back to. The budget was spent; say so and record
    // it, rather than saving a zero that reads back as "no paused clock" and
    // hands the learner a fresh budget for free.
    if (remaining <= 0) {
      stop("expired");
      return;
    }
    _write(ex, { nb: ex.nb, fn: ex.fn, kc: ex.kc, paused: true, remaining, deadline: null });
    _teardown();
    document.dispatchEvent(new CustomEvent("dd-exercise-timer:change", { detail: { running: false, ex, reason } }));
  };

  /** Pick a paused clock back up on its own block. */
  const resume = (ex, block) => {
    const secs = pausedFor(ex);
    if (!secs) return false;
    return start(ex, block, secs);
  };

  /* ── ending it ─────────────────────────────────────────────────── */

  const _verdict = (block, html) => {
    let slot = block.querySelector(".dd-ex-verdict");
    if (!slot) {
      slot = document.createElement("div");
      slot.className = "dd-ex-verdict";
      block.appendChild(slot);
    }
    slot.innerHTML = html;
    slot.classList.toggle("hidden", !html);
  };

  /* Does this concept still need drilling? The planner's own reading of the
     server's per-concept interval, and its own READY band — so the sentence on
     the problem and the block the Drill button would build are describing the
     learner with the same number. Any failure answers the prior, which is
     below READY, so an unreachable backend recommends drills rather than
     silently telling a learner they are fine. */
  const _recommendDrills = async (kc) => {
    const P = window.ExercisePlanner;
    if (!P || !kc) return null;
    let m = null;
    try {
      const f = _fetch();
      if (typeof practiceMode !== "undefined" && practiceMode === "backend" && typeof f === "function") {
        const res = await f("/api/practice/kc-estimate?kc=" + encodeURIComponent(kc));
        if (res.ok) {
          const body = await res.json();
          m = P.masteryFromEstimate(body && body.ladder_estimate);
        }
      }
    } catch (_) { /* the prior below */ }
    if (!Number.isFinite(m)) m = P.masteryFromEstimate(null);
    return { m, drill: m < P.READY };
  };

  /* One attempt, marked wrong, through the ordinary seam. `original` is the
     exercise's own bank question (lessons/arena_exercise_kcs.json); an
     exercise with none is a real case — the map carries the KC for drilling
     before anyone has authored the problem itself — and there is nothing
     honest to record against, so nothing is. */
  const _recordMiss = async (ex) => {
    const qid = ex?.original;
    const api = _api();
    if (!Number.isInteger(qid) || qid <= 0 || !api || typeof api.recordLocalEval !== "function") return false;
    // The guest implementation grades currentQuestion, not the supplied ID.
    // Never attribute this notebook timeout to an unrelated practice drill.
    if (typeof practiceMode !== "undefined" && practiceMode !== "backend") return false;
    try {
      const result = await api.recordLocalEval(qid, false);
      return !!result && result.finalized !== false;
    } catch (err) {
      console.warn("[exercise-timer] could not record the timeout:", err);
      return false;
    }
  };

  /* reason: "expired" | "gave-up" — the two misses. Anything else stops the
     clock and records nothing (kept for callers; the buttons never send it). */
  const stop = (reason = "expired") => {
    if (!live) return;
    const { ex, block } = live;
    _teardown();
    _write(ex, null);
    document.dispatchEvent(new CustomEvent("dd-exercise-timer:change", { detail: { running: false, ex, reason } }));
    if (reason !== "expired" && reason !== "gave-up") return;

    const why = reason === "expired"
      ? `Time is up — ${_esc(ex.title || ex.fn)}.`
      : `Marked "I don't know this yet" — ${_esc(ex.title || ex.fn)}.`;
    _verdict(block, `<b>${why}</b> <span class="dd-ex-verdict-note">Checking what to do next…</span>`);
    const resultSlot = block.querySelector(".dd-ex-verdict");
    const version = resultSlot._version = (resultSlot._version || 0) + 1;
    Promise.all([_recordMiss(ex), _recommendDrills(ex.kc)]).then(([recorded, rec]) => {
      if (resultSlot._version !== version || isRunning(ex)) return;
      const kept = recorded
        ? " Recorded as a miss."
        : " No result recorded: no linked drill or the practice service was unavailable.";
      let advice;
      if (!rec) {
        advice = "Drills on this concept are one button away whenever you want them.";
      } else if (rec.drill) {
        advice =
          `Recommended: drill <b>${_esc(ex.kcTitle || ex.kc)}</b> before trying again ` +
          `— about ${Math.round(rec.m * 100)}% chance of solving one cold right now.`;
      } else {
        advice =
          `You are already around ${Math.round(rec.m * 100)}% on <b>${_esc(ex.kcTitle || ex.kc)}</b>, ` +
          "so this looks more like a slow day than a gap. Read it again before drilling.";
      }
      _verdict(block, `<b>${why}</b>${kept} <span class="dd-ex-verdict-note">${advice}</span>`);
    });
  };

  /* ── restoring after a reload ──────────────────────────────────── */

  /* Called by practice/exercise-session.js once it has decorated a notebook,
     because only it knows which block belongs to which exercise. A snapshot
     for a different notebook is left alone, not cleared: the learner may be
     one click away from coming back to it. A PAUSED snapshot is left alone
     too — it is the learner's to resume, from the button on this block. */
  const restore = (ex, block) => {
    if (live) return false;
    const saved = _savedFor(ex);
    if (!saved || saved.paused) return false;
    const secs = Math.round((Number(saved.deadline) - Date.now()) / 1000);
    if (!Number.isFinite(secs) || secs <= 0) {
      // The clock ran out while the tab was closed. It still counts — the
      // budget was spent — but say so plainly rather than re-arming a dead
      // countdown.
      start(ex, block, 1);
      stop("expired");
      return false;
    }
    return start(ex, block, secs);
  };

  // Section changes replace the cells without necessarily hiding the page.
  document.addEventListener("arena-notebook:rendered", () => {
    if (live && !live.block.isConnected) pause("left");
  });

  /* Leaving the notebook page is not an answer and not a give-up. It is the
     pause: the clock is set down with its remaining time, the cells come back,
     nothing is recorded, and the block offers "Resume timer" next time.

     🔴 WATCHED, NOT LISTENED FOR. `switchTab` (app.js) announces nothing; it
     toggles `.hidden` on each `.page` and that is the whole signal. Adding an
     event there would be the honest fix and this file would rather not be the
     reason app.js changes — an observer on the one class app.js is already
     writing costs nothing and cannot fall out of sync with a second
     announcement nobody remembers to make. */
  const _watchPage = () => {
    const page = document.getElementById("page-arena-notebook");
    if (!page || !window.MutationObserver) return;
    new MutationObserver(() => {
      if (!page.classList.contains("hidden")) {
        if (!live) page.querySelectorAll(".dd-ex-block").forEach((block) => {
          if (block._exercise) restore(block._exercise, block);
        });
        return;
      }
      pause("left");
    }).observe(page, { attributes: true, attributeFilter: ["class"] });
  };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", _watchPage, { once: true });
  } else {
    _watchPage();
  }

  return {
    start, stop, pause, resume, restore, isRunning, activeExercise, pausedFor,
    clockText, isLow, recommendedSecs, budgetSecs, FALLBACK_SECS,
  };
})();

window.ExerciseTimer = ExerciseTimer;
