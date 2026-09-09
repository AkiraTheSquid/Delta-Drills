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

   SO THIS IS NOT A PRACTICE SESSION. Nothing here switches tabs, nothing here
   builds a ladder, nothing here touches `PracticeSession` — the two things the
   old single "Practice <fn>" button did in one press are now two buttons that
   do one thing each, and this is the one that keeps the learner where they
   are. The drill block is still practice/exercise-session.js, still on the
   Practice tab, and it is reached deliberately or not at all.

   WHAT IT OWNS
     • the countdown bar (`.dd-nbt-bar`, appended to #page-arena-notebook)
     • focus mode: `dd-nb-focus` on the page, `dd-nb-hidden` on the nodes it
       hides, and NOTHING else. Restoring is removing those two classes, so a
       teardown cannot half-restore a notebook.
     • the verdict line it writes into the exercise's own block when the clock
       runs out

   WHAT IT DOES NOT OWN
     Grading. A timeout is recorded through `PracticeAPI.recordLocalEval`, the
     same call practice/notebook-view.js makes when a cell prints ✅/❌ — one
     attempt, marked wrong, into the same engine as everything else. There is
     no second scoring system here and there must never be one.

   🔴 "I'M DONE" RECORDS NOTHING, ON PURPOSE. Running the exercise's test cell
   prints a verdict and practice/notebook-view.js beacons it; if this file
   recorded a result too, every finished exercise would go in twice, once from
   what the code actually did and once from a button. The clock is the only
   thing this file is the authority on, so running out is the only thing it
   reports.
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

  let live = null; // { ex, block, page, hidden: [], deadline, remaining, paused, tick }
  let bar = null;

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

  /* ── the bar ───────────────────────────────────────────────────── */

  const _ensureBar = () => {
    if (bar) return bar;
    bar = document.createElement("div");
    bar.className = "dd-nbt-bar hidden";
    bar.setAttribute("role", "status");
    bar.innerHTML =
      '<span class="dd-nbt-what"></span>' +
      '<span class="dd-nbt-clock" aria-live="off"></span>' +
      '<button type="button" class="ghost dd-nbt-pause"></button>' +
      '<button type="button" class="primary dd-nbt-done">I\'m done</button>' +
      '<button type="button" class="ghost dd-nbt-give">Give up</button>';
    document.body.appendChild(bar);
    bar.querySelector(".dd-nbt-pause").onclick = () => (live && live.paused ? resume() : pause());
    bar.querySelector(".dd-nbt-done").onclick = () => stop("done");
    bar.querySelector(".dd-nbt-give").onclick = () => stop("gave-up");
    return bar;
  };

  const _paint = () => {
    if (!live || !bar) return;
    const secs = _remaining();
    bar.querySelector(".dd-nbt-clock").textContent = _mmss(secs);
    bar.classList.toggle("is-low", secs <= 60);
    bar.classList.toggle("is-paused", !!live.paused);
    bar.querySelector(".dd-nbt-pause").textContent = live.paused ? "Resume" : "Pause";
  };

  const _remaining = () => {
    if (!live) return 0;
    if (live.paused) return live.remaining;
    return Math.max(0, Math.round((live.deadline - Date.now()) / 1000));
  };

  /* ── surviving a reload ────────────────────────────────────────── */

  /* A live clock that a refresh silently swallows is worse than no clock: the
     learner comes back to a notebook that looks idle and has no idea whether
     the attempt counted. The snapshot is per notebook + exercise so restoring
     can only ever re-arm the same problem. */
  const _save = () => {
    try {
      if (!live) localStorage.removeItem(SAVE_KEY());
      else localStorage.setItem(SAVE_KEY(), JSON.stringify({
        nb: live.ex.nb, fn: live.ex.fn, kc: live.ex.kc,
        paused: !!live.paused,
        remaining: live.paused ? live.remaining : null,
        deadline: live.paused ? null : live.deadline,
      }));
    } catch (_) { /* private mode — the clock just does not survive a reload */ }
  };

  const _readSaved = () => {
    try {
      const raw = JSON.parse(localStorage.getItem(SAVE_KEY()) || "null");
      return raw && raw.fn ? raw : null;
    } catch (_) {
      return null;
    }
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

  /** Start the clock on one notebook exercise. `secs` overrides the notebook's
      own recommendation (used by the restore path). */
  const start = (ex, block, secs = null) => {
    if (live) stop("switched");
    const focus = _enterFocus(block);
    if (!focus) return false;
    const budget = Number.isFinite(secs) && secs > 0 ? secs : budgetSecs(block);
    live = {
      ex, block, page: focus.page, hidden: focus.hidden,
      deadline: Date.now() + budget * 1000,
      remaining: budget, paused: false, tick: null,
    };
    const b = _ensureBar();
    b.querySelector(".dd-nbt-what").textContent = ex.title || ex.fn;
    b.classList.remove("hidden");
    _verdict(block, "");
    _paint();
    _startTick();
    _save();
    block.scrollIntoView({ block: "center", behavior: "smooth" });
    document.dispatchEvent(new CustomEvent("dd-exercise-timer:change", { detail: { running: true, ex } }));
    return true;
  };

  const pause = () => {
    if (!live || live.paused) return;
    live.remaining = _remaining();
    live.paused = true;
    _stopTick();
    _paint();
    _save();
  };

  const resume = () => {
    if (!live || !live.paused) return;
    live.deadline = Date.now() + live.remaining * 1000;
    live.paused = false;
    _startTick();
    _paint();
    _save();
  };

  /* ── ending it ─────────────────────────────────────────────────── */

  const _verdict = (block, html) => {
    let slot = block.querySelector(".dd-ex-verdict");
    if (!slot) {
      slot = document.createElement("div");
      slot.className = "dd-ex-verdict";
      block.querySelector(".dd-ex-row")?.insertAdjacentElement("afterend", slot);
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

  /* reason: "expired" | "gave-up" | "done" | "switched" | "left" */
  const stop = (reason = "done") => {
    if (!live) return;
    const { ex, block } = live;
    _stopTick();
    _exitFocus();
    if (bar) bar.classList.add("hidden");
    live = null;
    _save();
    document.dispatchEvent(new CustomEvent("dd-exercise-timer:change", { detail: { running: false, ex, reason } }));
    if (reason === "switched" || reason === "left") return;

    if (reason === "done") {
      _verdict(block, "Clock stopped. Run the exercise's tests to check your solution. No result was recorded by stopping the clock.");
      return;
    }

    const why = reason === "expired"
      ? `Time is up — ${_esc(ex.title || ex.fn)}.`
      : `Gave up on ${_esc(ex.title || ex.fn)}.`;
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
     one click away from coming back to it. */
  const restore = (ex, block) => {
    if (live) return false;
    const saved = _readSaved();
    if (!saved || saved.fn !== ex.fn || saved.nb !== ex.nb) return false;
    const secs = saved.paused
      ? Number(saved.remaining)
      : Math.round((Number(saved.deadline) - Date.now()) / 1000);
    if (!Number.isFinite(secs) || secs <= 0) {
      // The clock ran out while the tab was closed. It still counts — the
      // budget was spent — but say so plainly rather than re-arming a dead
      // countdown.
      start(ex, block, 1);
      stop("expired");
      return false;
    }
    start(ex, block, secs);
    if (saved.paused) pause();
    return true;
  };

  const _suspend = () => {
    if (!live) return;
    const ex = live.ex;
    pause();
    _exitFocus();
    _stopTick();
    if (bar) bar.classList.add("hidden");
    live = null;
    document.dispatchEvent(new CustomEvent("dd-exercise-timer:change", { detail: { running: false, ex, reason: "left" } }));
  };

  // Section changes replace the cells without necessarily hiding the page.
  document.addEventListener("arena-notebook:rendered", () => {
    if (live && !live.block.isConnected) _suspend();
  });

  /* Leaving the notebook page is not an answer and not a give-up. Stop the
     clock, restore the cells, record nothing — the snapshot survives, so the
     same exercise re-arms where it left off.

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
      _suspend();
    }).observe(page, { attributes: true, attributeFilter: ["class"] });
  };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", _watchPage, { once: true });
  } else {
    _watchPage();
  }

  return {
    start, stop, pause, resume, restore, isRunning, activeExercise,
    recommendedSecs, budgetSecs, FALLBACK_SECS,
  };
})();

window.ExerciseTimer = ExerciseTimer;
