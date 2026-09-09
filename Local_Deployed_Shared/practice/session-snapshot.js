/* ================================================================
   THE PAUSE SNAPSHOT — reading it, writing it, and the resume arithmetic

   Split out of practice/timer.js on 2026-09-09, when that file crossed
   Modulario's hard LOC limit. Nothing here changed in the move: these are
   the same functions, in the same order, with the same comments. What the
   split buys is that the snapshot's SHAPE and the clock's BEHAVIOUR are now
   two files — timer.js owns what a running question does, this owns what a
   paused one is worth when it comes back.

   🔴 IT DECIDES NOTHING ABOUT TIME OF ITS OWN. The per-step allowance comes
   from timer.js through `install({answerSecs, reviewSecs})`, because that is
   where an exercise session's own numbers (`sessionConfig`) live and where
   practice/session-clock.js is read. Injected rather than imported: this file
   is parsed BEFORE timer.js (index.html), so a direct read of either would be
   a load-order bug waiting for the next reorder. Uninstalled, both answer
   `null` — no limit — which is the only safe default: a clock that cannot be
   read must not force-submit anybody's work.

   🔴 EVERYTHING OUT OF localStorage IS UNTRUSTED INPUT. `_readSaved` is the
   only door, and it shape-checks every field: the ladder record reaches
   `StageLadder.show`, which writes a title into the DOM, and `remaining` is
   clamped to the allowance so a hand-edited entry cannot buy itself time the
   table never offered.

   🔴 THE RESUME ARITHMETIC IS LIFTED VERBATIM BY practice/watch_lessons.py.
   `check_a_resumed_clock_matches_the_break` slices the source between the two
   markers below and runs it under node against a stubbed clock. Keep the
   helpers pure functions of a snapshot, keep the markers, and keep them
   contiguous — the probe is the only place the shipped arithmetic is actually
   executed against the cases that matter.
   ================================================================ */

(function initSessionSnapshot() {
  /* 🔴 NOT BUMPED FOR THE PROBLEM'S-OWN CLOCK (2026-09-09), nor for the
     learner-set one before it (2026-08-28), deliberately. A bump DISCARDS every
     paused question already on a learner's machine. The snapshot now carries
     `secsAllowed` — the paused question's own number — as an OPTIONAL field:
     a v2 snapshot without it resumes on whatever the restored question says,
     and a bank-rebuilt question with no stamp gets the ceiling.

     The 1 → 2 bump it still carries is a different case. A v1 snapshot stored
     the learner's OWN answerSecs/reviewSecs from the FIRST setup panel, and
     resuming one meant honouring a per-session pair of allowances this model
     does not have. `_readSaved` drops v1 outright. */
  const SESSION_STATE_VERSION = 2;

  /* How long a paused clock stays paused before the step starts over.

     Leaving and coming straight back is not a break — a reload, a tab closed by
     accident, a laptop lid — and handing back a fresh five minutes for it would
     make "pause" the way to opt out of the timer entirely. So inside the grace
     window the clock resumes exactly where it stopped: one minute left is one
     minute left.

     Coming back an hour later is a different thing, and resuming at 00:01 there
     punishes the break rather than timing the work. What the strict timer
     actually measures is a continuous attempt, and after a real gap the learner
     is starting the step again — re-reading the prompt, rebuilding what they had
     in their head — so the step gets its full time back. Only the CURRENT step:
     the question, the quota and the draft code are all still theirs. */
  const RESUME_GRACE_SECS = 120;

  /* The two allowances, injected by timer.js. See the header: `null` until it
     installs, and `null` is "no limit", never "no clock at all". */
  let _answerSecs = () => null;
  let _reviewSecs = () => null;
  const ANSWER_SECS = () => _answerSecs();
  const REVIEW_SECS = () => _reviewSecs();

  /* Called lazily, like everything else keyed on the account:
     `getPracticeStorageKey` reads `authEmail`, which is not resolved at the
     moment this file parses. */
  const _storageKey = () => `${getPracticeStorageKey()}_session`;

  /* Everything out of localStorage is untrusted input — this one is read back
     into `StageLadder.show`, which writes the title into the DOM. Shapes only:
     an unknown rung draws no sections rather than guessing, so a junk `stage`
     is safe, but a junk `estimate` would reach `_boundOf`/`_streakOf`. */
  const _str = (value) =>
    typeof value === "string" && value.trim() ? value : null;

  const _readLadder = (raw) => {
    if (!raw || typeof raw !== "object") return null;
    /* 🔴 STRINGS, NOT TRUTHY VALUES. `String([])` is `""` and
       `String({})` is `"[object Object]"`, so a truthiness check hands the
       ladder an empty kc or a literal "[object Object]" as the concept's name
       instead of refusing the record. Codex, 2026-08-28. */
    const kc = _str(raw.kc);
    const stage = _str(raw.stage);
    if (!kc || !stage) return null;
    return {
      kc,
      stage,
      title: _str(raw.title),
      estimate: raw.estimate && typeof raw.estimate === "object" ? raw.estimate : null,
      support: raw.support !== false,
      integrated: !!raw.integrated,
    };
  };

  /* The exercise session's own settings, carried in the snapshot as an
     OPTIONAL field — a snapshot without one is a plain block and restores as
     it always did. Shape-checked: a hand-edited entry cannot hand the clock a
     string, and a quota that is not a finite positive number is no quota. */
  const _readConfig = (raw) => {
    if (!raw || typeof raw !== "object") return null;
    const secs = (v) => v === null || (Number.isFinite(v) && v > 0);
    const out = {};
    if ("answer" in raw && secs(raw.answer)) out.answer = raw.answer;
    if ("review" in raw && secs(raw.review)) out.review = raw.review;
    if (Number.isFinite(raw.quota) && raw.quota > 0) out.quota = Math.round(raw.quota);
    if (raw.exercise && typeof raw.exercise === "object") out.exercise = { ...raw.exercise };
    if (raw.ladder && typeof raw.ladder === "object") out.ladder = raw.ladder;
    return Object.keys(out).length ? out : null;
  };

  const _readSaved = () => {
    try {
      const saved = JSON.parse(localStorage.getItem(_storageKey()) || "null");
      if (!saved || saved.version !== SESSION_STATE_VERSION) return null;
      if (!Number.isFinite(saved.served) || !saved.questionId) return null;
      const phase = saved.phase === "review" && saved.review ? "review" : "answer";
      /* The allowance is the PAUSED QUESTION's own — the snapshot's
         `secsAllowed`, written from the question on screen at pause, else
         what the restored question record says (`_phaseLimit`) — and the
         snapshot's `remaining` is clamped to it. The clamp is what stops a
         hand-edited localStorage entry buying itself time the table never
         offered.

         🔴 UNLIMITED CLAMPS TO UNLIMITED. `phaseLimit === null` means there is
         no number to clamp to and no number to count down, so `remaining`
         stays null all the way through resume; `Math.min(null, x)` is 0, which
         would resume the question already expired. */
      const config = _readConfig(saved.config);
      const secsAllowed = Number.isFinite(saved.secsAllowed) && saved.secsAllowed > 0
        ? Math.round(saved.secsAllowed) : null;
      /* 🔴 `_savedLimit`, NOT `_phaseLimit`. This runs at PARSE time, before
         anything is restored, so the live readers here answer for the question
         the page rendered in the background — a different one. A legacy v2
         snapshot carries no `secsAllowed` (it was written before the clock
         became the problem's own), so it took whatever cap happened to be on
         screen: 18:00 left on a 20:00 einops question, reloaded over a 5:00
         Python drill, came back clamped to 5:00 and the other 13 minutes were
         gone for good. When the snapshot names no allowance, nothing is
         clamped here — `_effectiveRemaining` clamps at resume, by which time
         the saved question IS the one on screen. Codex, 2026-09-09. */
      const phaseLimit = _savedLimit({ phase, config, secsAllowed });
      const savedRemaining = Number.isFinite(saved.remaining) ? saved.remaining : phaseLimit;
      return {
        version: SESSION_STATE_VERSION,
        served: Math.max(1, Math.round(saved.served)),
        phase,
        config,
        remaining: phaseLimit === undefined
          ? (Number.isFinite(saved.remaining) ? Math.max(1, Math.round(saved.remaining)) : null)
          : phaseLimit === null
            ? null
            : Math.max(1, Math.min(phaseLimit, Math.round(savedRemaining || 30))),
        questionId: String(saved.questionId),
        attemptFirst: saved.attemptFirst === true,
        secsAllowed,
        ladder: _readLadder(saved.ladder),
        draft: typeof saved.draft === "string" || (
          saved.draft?.version === 1 && Array.isArray(saved.draft.cells)
        ) ? saved.draft : "",
        review: phase === "review" ? saved.review : null,
        savedAt: saved.savedAt || null,
      };
    } catch (_) {
      return null;
    }
  };

  const _writeSaved = (snapshot) => {
    if (!snapshot || !snapshot.questionId) return;
    try {
      localStorage.setItem(_storageKey(), JSON.stringify(snapshot));
    } catch (_) {}
  };

  const _clearSaved = () => {
    try {
      localStorage.removeItem(_storageKey());
    } catch (_) {}
  };

  /* ── RESUME ARITHMETIC (lifted by practice/watch_lessons.py) ────────── */

  /* Seconds since the snapshot was written. `_persist` stamps `savedAt` every
     tick, so this is the length of the break to within a second — except when
     the field is missing or unreadable (a snapshot from an older bundle, a
     mangled localStorage entry), where the honest answer is "no idea how long"
     and the safe one is to treat it as a long break. Erring that way costs a
     restarted step; erring the other way hands out free time on every reload.
     A clock that has gone BACKWARDS reads as 0, which resumes the timer. */
  const _awaySecs = (saved) => {
    const at = Date.parse(saved?.savedAt || "");
    if (!Number.isFinite(at)) return Infinity;
    return Math.max(0, (Date.now() - at) / 1000);
  };

  /* The allowance the saved step gets: the block's OWN number when the
     snapshot carries a config (a scoped exercise session, which is not yet
     the live config while the snapshot is being read); else the paused
     question's own clock as the snapshot recorded it (`secsAllowed`); else
     what the question on screen says through SessionClock. Self-contained on
     purpose — practice/watch_lessons.py lifts this helper into a node probe
     by itself. */
  /* 🔴 WHAT THE SNAPSHOT ITSELF KNOWS, and `undefined` when it knows nothing.
     `undefined` is NOT `null`: `null` is a scoped block's "No limit", an
     answer; `undefined` means this snapshot names no allowance and the only
     way to get one is to ask the question on screen — which is a different
     question until the saved one has been restored. Codex, 2026-09-09. */
  const _savedLimit = (saved) => {
    const field = saved.phase === "review" ? "review" : "answer";
    const config = saved.config;
    if (config && typeof config === "object" && field in config) return config[field];
    if (Number.isFinite(saved.secsAllowed) && saved.secsAllowed > 0) return saved.secsAllowed;
    return undefined;
  };

  /* The allowance in force for a saved step. Safe ONLY once the saved question
     is the one on screen (`_restoreSavedQuestion` has run) — before that the
     live readers answer for whatever was rendered in the background. */
  const _phaseLimit = (saved) => {
    const own = _savedLimit(saved);
    if (own !== undefined) return own;
    return saved.phase === "review" ? REVIEW_SECS() : ANSWER_SECS();
  };

  /* What the clock should read on resume: {secs, restarted}.

     Recomputed at the moment of resuming rather than when the snapshot was
     read, because the resume panel can sit on screen for as long as the
     learner likes and the break is still running while it does. */
  const _effectiveRemaining = (saved) => {
    /* 🔴 THE LIMIT DECIDES FIRST. Under "No limit" (a scoped block's own
       setting) there is no clock to hand back and no step to restart, however
       long the break was — asking about the break at all would resume an
       untimed question with `restarted: true` and tell the learner a step
       started over that was never running. */
    const limit = _phaseLimit(saved);
    if (limit === null) return { secs: null, restarted: false };
    /* A snapshot written while untimed carries `remaining: null`. If the step
       now has a real allowance there is nothing to pick up mid-step, so that
       step starts at the limit. */
    if (saved.remaining === null) return { secs: limit, restarted: true };
    if (_awaySecs(saved) <= RESUME_GRACE_SECS) {
      /* 🔴 CLAMPED HERE TOO, not only in `_readSaved`. That clamp runs when the
         snapshot is PARSED, and the picker could move after it: a question
         paused with 8:00 left under 10:00, then switched to 1:00 on the idle
         screen, resumed inside the grace window and came back with the whole
         8:00 — the allowance the learner had just replaced. Codex, 2026-08-28.

         The clamp is against the SNAPSHOT's own allowance where it has one. A
         legacy v2 snapshot has none, and `limit` there came from the live
         readers, which answer for whatever question is on screen — so it is a
         clamp against an unrelated problem's cap. Hand the break back
         unclamped in that case; the pause already enforced the clock that was
         actually running. Codex, 2026-09-09. */
      if (_savedLimit(saved) === undefined) return { secs: saved.remaining, restarted: false };
      return { secs: Math.min(saved.remaining, limit), restarted: false };
    }
    return { secs: limit, restarted: true };
  };

  /* ── END RESUME ARITHMETIC ──────────────────────────────────────────── */

  window.SessionSnapshot = {
    VERSION: SESSION_STATE_VERSION,
    RESUME_GRACE_SECS,
    /* timer.js hands over its two allowance readers at the top of its own
       IIFE, BEFORE it calls `read()` for the paused question. */
    install: ({ answerSecs, reviewSecs }) => {
      if (typeof answerSecs === "function") _answerSecs = answerSecs;
      if (typeof reviewSecs === "function") _reviewSecs = reviewSecs;
    },
    key: _storageKey,
    read: _readSaved,
    write: _writeSaved,
    clear: _clearSaved,
    readConfig: _readConfig,
    savedLimit: _savedLimit,
    phaseLimit: _phaseLimit,
    effectiveRemaining: _effectiveRemaining,
  };
})();
