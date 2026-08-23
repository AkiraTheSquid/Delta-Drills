/* ================================================================
   XP.JS — levels, and the progress bar that IS the level pill.

   The bar is not a widget parked somewhere in the chrome: it is the
   `Level N` chip itself, colouring in from the left as the learner works.
   One object carries both facts — which level, and how far into it — and
   it is on screen on every tab. It was a 2px line on the topbar's bottom
   border until 2026-08-22; a hairline is only legible if you already know
   to look for it. `styles/xp.css` owns the drawing; this file owns the
   number, and hands the drawing ONE value: `--dd-xp-pct` on `.dd-level`.

   NO NUMBER IS RENDERED FOR THE PROGRESS. Not a count, not a percent, not
   a "340 / 500". The only numeral is the level itself. A bar that states
   its own fraction turns practice into a spreadsheet; a bar that just
   moves is felt.

   ── WHAT EARNS XP ────────────────────────────────────────────────
   Anything the learner ENTERS. Every path that records learner data in
   this app funnels through a `PracticeAPI` method — graded submits, the
   felt-difficulty rating, placement probes, "I don't know yet", torch
   self-rating, content flags — so `practice/api.js` wraps those methods
   once at the bottom of the file rather than sprinkling award() calls
   through a dozen handlers. Anything outside that chokepoint (targeted
   practice, a lesson read) dispatches `delta:xp` instead; see `award`.

   A generic typing tick is included on purpose: Seth's rule is "any data
   entered of ANY kind moves the bar". Typing into the editor before a
   submit lands is data entry, so it earns 1 XP, throttled to once every
   ENTRY_THROTTLE_MS so holding a key down is not a level-up machine.

   ── WHERE IT IS STORED ───────────────────────────────────────────
   localStorage, keyed by `auth_email` exactly like practice/storage.js.
   A guest keeps their own levels and a signed-in account keeps theirs;
   there is no merge, because there is no merge anywhere else in this app
   (see the guest banner copy in index.html) and inventing one here would
   be the only place progress silently changed owner.
   ================================================================ */
(function () {
  "use strict";

  const STORE_PREFIX = "dd_xp_v1_";
  const STORE_VERSION = 1;

  /* The curve. Level 1 → 2 costs 100; every level after costs 60 more.
     Linear-with-a-step rather than exponential: the app's sessions are
     ~10 questions, and a doubling curve makes level 6 unreachable inside
     the number of drills that actually exist. */
  const LEVEL_BASE = 100;
  const LEVEL_STEP = 60;

  /* What each kind of entered data is worth. Correct answers pay most,
     but a MISS still pays — the placement test is built out of misses,
     and a bar that only moves on success would punish the learner for
     using the feature that finds their level. */
  const AWARDS = {
    answer_correct: 25,
    answer_wrong: 10,
    placement_answer: 15,
    placement_skip: 6,
    difficulty_rating: 5,
    override: 5,
    problem_report: 4,
    lesson_read: 8,
    targeted_session: 10,
    entry: 1,
  };

  const ENTRY_THROTTLE_MS = 15000;
  /* Long enough for the width transition in xp.css to finish before the
     fill is snapped back to the remainder. Kept here (not read off the
     stylesheet) because a computed-style read of a transition mid-flight
     is the kind of thing that works until someone adds a second one. */
  const LEVELUP_HOLD_MS = 560;
  const LEVELUP_FLASH_MS = 1100;

  const needFor = (level) => LEVEL_BASE + (Math.max(1, level) - 1) * LEVEL_STEP;

  const storeKey = () => {
    let email = "";
    try {
      email = (localStorage.getItem("auth_email") || "").trim();
    } catch (_) {
      /* private mode / storage disabled — fall through to the guest key */
    }
    return STORE_PREFIX + (email || "guest");
  };

  const blank = () => ({ v: STORE_VERSION, level: 1, into: 0 });

  const load = () => {
    try {
      const raw = localStorage.getItem(storeKey());
      if (!raw) return blank();
      const parsed = JSON.parse(raw);
      const level = Number.isFinite(parsed?.level) ? Math.max(1, Math.floor(parsed.level)) : 1;
      const into = Number.isFinite(parsed?.into) ? Math.max(0, parsed.into) : 0;
      /* A stored `into` past the cost of its level would draw the bar
         beyond full — clamp rather than trust, because the curve above is
         allowed to change and old saves outlive it. */
      return { v: STORE_VERSION, level, into: Math.min(into, needFor(level)) };
    } catch (_) {
      return blank();
    }
  };

  const save = (st) => {
    try {
      localStorage.setItem(storeKey(), JSON.stringify(st));
    } catch (_) {
      /* best-effort — never let a full quota break a submit */
    }
  };

  let state = load();

  // ── the DOM ────────────────────────────────────────────────────
  // Looked up lazily: this script is loaded with the other app scripts at
  // the bottom of index.html, but the Colab edition and the ?embed=1
  // knowledge-graph frame both strip app chrome, so the elements are
  // allowed to be absent and every write below has to survive that.
  let elChip = null;
  let elNum = null;
  // The second copy of the numeral, the one drawn on the fill. See the
  // two-layer note in styles/xp.css: both layers hold the same text and
  // the top one is clipped to the filled width, so this has to be written
  // in lockstep with elNum or the number changes as the fill passes it.
  let elNumOn = null;
  let flashTimer = null;
  // Non-null only while a level-up's fill-to-full is being held; render()
  // defers to it so one award cannot paint the bar twice.
  let snapTimer = null;

  const grabDom = () => {
    elChip = document.getElementById("dd-level");
    elNum = document.getElementById("dd-level-num");
    elNumOn = document.getElementById("dd-level-num-on");
  };

  /* ONE write drives the whole pill: the fill's width and the clip on the
     on-accent text layer both read `--dd-xp-pct`, so they can never drift
     out of step the way two separate JS writes eventually would. */
  const paint = (pct) => {
    if (elChip) elChip.style.setProperty("--dd-xp-pct", pct + "%");
  };

  /* Always read off `state` at the moment of painting, never off a value
     captured when the award was made. A second award landing DURING a
     level-up's hold would otherwise have its progress overwritten by the
     first award's stale remainder — which was 0%, so the bar emptied. */
  const currentPct = () =>
    Math.max(0, Math.min(100, (state.into / needFor(state.level)) * 100));

  const render = (levelsGained) => {
    const pct = currentPct();

    const levelText = String(state.level);
    if (elNum) elNum.textContent = levelText;
    if (elNumOn) elNumOn.textContent = levelText;
    if (elChip) {
      elChip.setAttribute(
        "aria-label",
        `Level ${state.level}, ${Math.round(state.into)} of ${needFor(state.level)} XP`
      );
      // Hover-only. The BAR still states nothing; this is for the person
      // who deliberately goes looking for the number.
      elChip.title = `Level ${state.level} · ${Math.round(state.into)}/${needFor(state.level)} XP`;
    }

    if (!elChip) return;

    if (!levelsGained) {
      // A level-up's snap is still queued — it re-reads `state`, so it will
      // land on this award's value too. Painting now as well would make the
      // pill grow, empty, and grow again for one award.
      if (snapTimer === null) paint(pct);
      return;
    }

    /* Level-up: run the fill to the end, then snap it back to the
       remainder WITHOUT a transition (otherwise it animates backwards
       across the whole pill, which reads as losing progress) and let it
       grow again from zero. */
    paint(100);
    if (elChip) {
      elChip.classList.add("dd-level--up");
      clearTimeout(flashTimer);
      flashTimer = setTimeout(() => elChip.classList.remove("dd-level--up"), LEVELUP_FLASH_MS);
    }
    clearTimeout(snapTimer);
    snapTimer = setTimeout(() => {
      if (!elChip) {
        snapTimer = null;
        return;
      }
      elChip.classList.add("dd-level--snap");
      paint(0);
      // Force the zero width to land before the transition comes back;
      // without the reflow the browser coalesces both writes and the
      // fill simply slides from 100% to the remainder.
      void elChip.offsetWidth;
      elChip.classList.remove("dd-level--snap");
      paint(currentPct());
      snapTimer = null;
    }, LEVELUP_HOLD_MS);
  };

  // ── the public API ─────────────────────────────────────────────
  /**
   * Add XP for a piece of entered data.
   * @param {string} kind  a key of AWARDS; unknown kinds are ignored
   *                       rather than silently worth zero-and-rendered,
   *                       so a typo in a call site cannot fake motion.
   * @param {number} [amount] explicit override, for callers with their
   *                       own weighting.
   */
  const award = (kind, amount) => {
    const gain = Number.isFinite(amount) ? amount : AWARDS[kind];
    if (!Number.isFinite(gain) || gain <= 0) return state;

    state.into += gain;
    let levelsGained = 0;
    while (state.into >= needFor(state.level)) {
      state.into -= needFor(state.level);
      state.level += 1;
      levelsGained += 1;
    }
    save(state);
    render(levelsGained);
    window.dispatchEvent(
      new CustomEvent("delta:xp-changed", {
        detail: { kind, gain, level: state.level, into: state.into, need: needFor(state.level), levelsGained },
      })
    );
    return state;
  };

  window.DeltaXP = {
    award,
    state: () => ({ level: state.level, into: state.into, need: needFor(state.level) }),
    /** Re-read the store — used when the signed-in identity changes. */
    reload: () => {
      state = load();
      render(0);
    },
    reset: () => {
      state = blank();
      save(state);
      render(0);
    },
  };

  // ── wiring ─────────────────────────────────────────────────────
  // Anything outside practice/api.js awards by dispatching this, so a
  // feature does not have to care whether xp.js loaded before it.
  window.addEventListener("delta:xp", (e) => award(e?.detail?.kind, e?.detail?.amount));

  // Sign in, sign out, guest provisioning — all three change the store
  // key, so the bar has to be re-read rather than carried across.
  window.addEventListener("delta:auth-state-changed", () => window.DeltaXP.reload());

  /* THE GENERIC ENTRY TICK. Capture phase and on `document` so it sees
     the code editor, the notebook cells, the feedback note and anything
     added later, without any of them knowing this file exists. */
  /* -Infinity, not 0: `performance.now()` is milliseconds since THIS page
     loaded, so a 0 sentinel makes the whole first 15 seconds of the session
     look like "you already earned an entry tick" — and the first thing a
     learner types after a fresh load is exactly when the bar should move. */
  let lastEntry = -Infinity;
  document.addEventListener(
    "input",
    () => {
      const now = performance.now();
      if (now - lastEntry < ENTRY_THROTTLE_MS) return;
      lastEntry = now;
      award("entry");
    },
    true
  );

  const start = () => {
    grabDom();
    render(0);
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
