/* ================================================================
   DIFFICULTY BAR — readout, thresholds, and this problem's tick
   ================================================================

   The half of #target-difficulty that is not the moving bar. `bars.js` owns
   the fill, the green/red band and the two white markers, and it is the only
   thing that animates them; this file owns the numbers beside the track, the
   two threshold zones drawn on it, the accent tick for the problem on screen,
   and the sentence underneath. Split that way because the two halves change on
   different events: the bar moves when an answer is graded, the thresholds
   move when the RUNG changes, and the tick moves when a new question renders.

   Nothing here fetches or computes a difficulty. Every number arrives from
   bars.js (which gets it from the backend) or from ConceptTopbar (the rung).

   ---------------------------------------------------------------
   WHERE THE TWO THRESHOLDS COME FROM — they are not decoration.

   The aim the queue serves is, in `app/prioritization.py`:

       target = _DIFF_FLOOR + _DIFF_SPAN * m       (20 + 80 * m, clamped)

   and when a concept is in play `m` is that concept's Wilson LOWER bound
   (`_aim_mastery`) — the same quantity the ladder promotes on. The rung is
   left when that bound clears `PROMOTE_LO` (`app/kc_graph.py`), which
   `concept-topbar.js` mirrors as `PROMOTE_AT`. So a promotion threshold in
   mastery converts to a point on THIS axis exactly:

       0.34  ->  20 + 80 * 0.34  =  47.2      (leaving Faded)
       0.51  ->  20 + 80 * 0.51  =  60.8      (leaving Worked example)

   and the floor is the aim with nothing demonstrated at all: 20.

   🔴 WHY THE LABELS SAY "PACE" AND NOT "UNLOCKED".

   `target_difficulty` adds the learner's own felt-difficulty correction AFTER
   the formula above — `difficulty_offset`, clamped to ±`DIFFICULTY_OFFSET_LIMIT`
   = ±20 points, a quarter of the whole span. So the aim this bar draws is not a
   pure function of the Wilson bound, and crossing the green edge does NOT
   entail that the bound cleared `PROMOTE_LO`: a learner who has been answering
   "way too easy" can be pushed 20 points past the mark on a record that has not
   moved, and one answering "way too hard" can clear the bound while the bar
   still sits short of it. An earlier draft of this file labelled the region
   "<rung> unlocked" and the sentence "the next question comes with less
   support" — both are claims about the ladder, and the bar cannot make them.

   What it CAN say is where the line is: 47.2 is the difficulty the queue serves
   a learner whose record has just cleared that rung, with no correction
   applied. The labels state that and stop.

   The red region is not the demotion rule either — demotion runs off the Wilson
   UPPER bound (`DEMOTE_HI`, four straight wrong), a different bound of a
   different interval. Hence "support returns" rather than "you get downgraded
   here".
   ================================================================ */

const DifficultyBar = (() => {
  /* Mirrors of `_DIFF_FLOOR` / `_DIFF_SPAN` in app/prioritization.py.
     `practice/watch.py` fails if they drift, the same way it already guards
     PROMOTE_AT — a threshold drawn in the wrong place is worse than no
     threshold, because the learner clears the mark and nothing happens. */
  const AIM_FLOOR = 20;
  const AIM_SPAN = 80;

  const _el = (id) => document.getElementById(id);
  const _clamp = (v) => Math.max(0, Math.min(100, v));
  const _fmt = (v) => (Number.isFinite(v) ? v.toFixed(1) : "--");

  /* Where the next rung starts, on this axis, cached because the sentence under
     the bar is redrawn on every frame of the tween and re-deriving the rung
     there would tie an animation to the strip's state. Null on the rungs that
     are not left by clearing a number. */
  let nextMark = null;
  /* The value currently drawn, so the sentence under the bar can be recomputed
     when the RUNG changes without an answer — a mid-screen promotion moves the
     threshold, and a distance measured against the previous rung's mark would
     survive it and read as this rung's. */
  let shownValue = null;
  /* The "no reading" wording, held for the same reason: with no value there is
     no distance to print, and the sentence slot is where that note lives. Held
     rather than left in the DOM because `_distance` rewrites that slot. */
  let noReadingNote = null;

  const reveal = () => {
    const host = _el("target-difficulty");
    if (host) host.classList.remove("hidden");
  };

  /* The accent tick. A question with no rating hides it rather than drawing a
     zero — an unrated problem is not a difficulty-zero problem. */
  const setProblem = (rating) => {
    const tick = _el("difficulty-problem-tick");
    const label = _el("difficulty-problem-label");
    if (!tick) return;
    const value = Number.isFinite(rating) ? _clamp(rating) : null;
    if (value === null) {
      tick.hidden = true;
      return;
    }
    tick.hidden = false;
    tick.style.left = `${value.toFixed(1)}%`;
    if (label) label.textContent = `this problem · ${Math.round(value)}`;
  };

  /* Redraw the green region from the rung the strip is showing. Called on
     render and whenever the rung changes mid-screen, because a promotion that
     left last rung's threshold on the track would be pointing at a line the
     learner has already crossed. */
  const refreshThresholds = () => {
    const zone = _el("difficulty-zone-next");
    const label = _el("difficulty-zone-next-label");
    const floor = _el("difficulty-zone-floor");
    if (floor) floor.style.width = `${AIM_FLOOR}%`;
    if (!zone) return;

    const mark = window.ConceptTopbar?.promotionMark?.() || null;
    nextMark = mark ? _clamp(AIM_FLOOR + AIM_SPAN * mark.bound) : null;
    // Lesson, Solo and Integrated have no entry: the first is left by reading
    // the page, the second by covering the concept's questions, and the third
    // is not cleared on one concept at all. None of the three is a number, so
    // none of them gets a line.
    zone.hidden = nextMark === null;
    // The sentence is measured against the mark, so it has to be redrawn with
    // it. Both directions matter: a rung gained moves the line up, and a strip
    // that lost its concept drops the line entirely.
    _distance(shownValue);
    if (nextMark === null) return;
    zone.style.width = `${(100 - nextMark).toFixed(1)}%`;
    if (label) label.textContent = `${mark.next} pace · ${_fmt(nextMark)} →`;
    zone.title =
      `The queue serves difficulty ${_fmt(nextMark)} to a learner whose record on this ` +
      `concept has just cleared the ${mark.next} threshold. Your own "too hard" / ` +
      `"too easy" ratings shift the aim by up to 20 points either way, so reaching ` +
      `this line is not itself the promotion — the rung moves on your answers.`;
  };

  /* How far there is to go, in the bar's own points. Not a count of questions:
     the ladder moves on a bound over a window of attempts, so "three more"
     would be a promise the engine has not made. */
  const _distance = (value) => {
    const foot = _el("difficulty-bar-foot");
    if (!foot) return;
    shownValue = Number.isFinite(value) ? _clamp(value) : null;
    if (shownValue === null) {
      foot.textContent = noReadingNote || "";
      return;
    }
    if (nextMark === null) {
      // A rung with no threshold — Lesson, Solo, Integrated — or no concept at
      // all. There is a reading, but nothing to measure it against.
      foot.textContent = "";
      return;
    }
    const gap = nextMark - shownValue;
    foot.innerHTML =
      gap > 0
        ? `<b>${_fmt(gap)}</b> points below the next rung's pace`
        : `<span class="is-past">At the next rung's pace, <b>${_fmt(-gap)}</b> past the line</span>`;
  };

  const _setNew = (value, direction) => {
    const el = _el("difficulty-bar-new");
    if (!el) return;
    el.className = `difficulty-bar-new${direction ? ` is-${direction}` : ""}`;
    el.innerHTML = `${_fmt(value)}<span class="difficulty-bar-max">/100</span>`;
  };

  /* One number and no comparison: the first question of a concept, or any
     render before an answer has moved anything. */
  const aim = (value) => {
    const v = _clamp(value);
    reveal();
    noReadingNote = null;
    ["difficulty-bar-old", "difficulty-bar-arrow", "difficulty-bar-chip"].forEach((id) => {
      const el = _el(id);
      if (el) el.hidden = true;
    });
    _setNew(v, null);
    _distance(v);
  };

  /* The move an answer made. Called twice per answer on purpose — once as the
     tween starts, so the old number and the chip are on screen for the whole
     900ms, and once when bars.js settles the bar. It is idempotent. */
  const move = (oldValue, newValue) => {
    const a = _clamp(oldValue);
    const b = _clamp(newValue);
    const moved = Math.abs(b - a) >= 0.05;
    reveal();
    noReadingNote = null;

    const oldEl = _el("difficulty-bar-old");
    const arrow = _el("difficulty-bar-arrow");
    const chip = _el("difficulty-bar-chip");
    if (oldEl) {
      oldEl.hidden = false;
      oldEl.textContent = _fmt(a);
    }
    if (arrow) arrow.hidden = false;
    if (chip) {
      chip.hidden = !moved;
      chip.className = `difficulty-bar-chip is-${b > a ? "up" : "down"}`;
      chip.textContent = `${b > a ? "+" : "−"}${_fmt(Math.abs(b - a))}`;
    }
    _setNew(b, moved ? (b > a ? "up" : "down") : null);
    _distance(b);
  };

  /* The value mid-tween. Only the big number and the sentence follow the
     animation — the old value and the chip are the endpoints of the move and
     would flicker if they were re-derived every frame. */
  const live = (value) => {
    if (!Number.isFinite(value)) return;
    const el = _el("difficulty-bar-new");
    const dir = el && el.classList.contains("is-down") ? "down"
      : el && el.classList.contains("is-up") ? "up" : null;
    _setNew(_clamp(value), dir);
    _distance(_clamp(value));
  };

  /* No number to show — a lesson screen, or an answer that located the learner
     instead of stepping the ladder. Says so rather than drawing a zero. */
  const unavailable = (note) => {
    reveal();
    // Clear the remembered value first: without this a later rung change would
    // call `_distance` with the previous question's number and replace the
    // "no reading" note with a distance measured from a reading we do not have.
    shownValue = null;
    noReadingNote = note || "no reading yet";
    ["difficulty-bar-old", "difficulty-bar-arrow", "difficulty-bar-chip"].forEach((id) => {
      const el = _el(id);
      if (el) el.hidden = true;
    });
    const el = _el("difficulty-bar-new");
    if (el) {
      el.className = "difficulty-bar-new";
      el.textContent = "—";
    }
    const foot = _el("difficulty-bar-foot");
    if (foot) foot.textContent = noReadingNote;
  };

  return {
    // Exported so `practice/watch.py` can assert the mirror of the backend's
    // difficulty range from outside the module.
    AIM_FLOOR,
    AIM_SPAN,
    aim,
    live,
    move,
    refreshThresholds,
    setProblem,
    unavailable,
  };
})();

window.DifficultyBar = DifficultyBar;
