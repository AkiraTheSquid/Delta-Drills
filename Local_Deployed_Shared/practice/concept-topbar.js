/* ================================================================
   CONCEPT TOPBAR — one persistent header for the whole practice page

   What it replaces, and why. The lesson screen used to build its own
   `.lesson-topbar` inside the question HTML. Two problems followed from that:
   it lived inside `.practice-left`, so it stopped at the panel divider instead
   of spanning the page; and it was part of an innerHTML that gets thrown away
   on every render, so nothing above the fold could survive between pages.

   This module owns that strip instead. It sits above `.practice-split` in the
   markup, so it spans both panels, and it is updated rather than rebuilt. It is
   the one place on the page that answers three questions the learner should
   never have to hunt for:

     * WHICH concept is this?          — named, and clickable into the graph
     * WHERE am I in the sequence?     — five dots, all five always visible

   It used to answer two more, with a bar each: a mastery interval and a
   difficulty track. Both are gone from the strip. Difficulty now has ONE
   readout on the page — the full-width bar directly under this one
   (index.html `#target-difficulty`) — and the mastery interval went with the
   second copy, because a 110px interval bar beside a difficulty bar was two
   measures that fill up, in the same visual language, meaning different
   things. What this module still owns of that is `promotionMark()`: the rung
   is what it knows, and the bar below needs it to place the green threshold.

   THE RUNGS

   `lesson -> faded -> worked example -> solo -> integrated`: read the teaching
   page, fill in the missing step, read a solved one and do the same move, do
   it with nothing, then meet it mixed in with the other concepts. The backend
   still speaks the older vocabulary (`worked, faded, partial, solo`) where its
   `worked` means the lesson page and its `partial` means the read-an-example
   rung; `STAGE_ALIASES` maps one onto the other so the display can be correct
   before the state migration lands. That mapping is the only place the two
   vocabularies meet — do not scatter it.

   Every dot is always drawn, including the ones already passed and the ones
   not yet reached. A single label ("Faded") tells a learner what they are
   doing; the row tells them how much support they have already given up and
   how much is left, which is the thing that makes the ladder legible as a
   ladder rather than as an arbitrary change of question format.
   ================================================================ */

const ConceptTopbar = (() => {
  "use strict";

  /* Display order. Left to right is decreasing support.

     🔴 THESE TWO USED TO BE THE WRONG WAY ROUND, and the error was not
     cosmetic. The backend's `faded` rung serves a fill-in-the-blank drill and
     was displayed as "Worked — solve beside a worked example you can still
     see"; its `partial` rung serves a drill with a solved example above it and
     was displayed as "Faded — most of the solution is written". Each dot named
     the other one's rung, so the strip promised support that was not on the
     page and withheld credit for support that was. The report was "it says
     worked when in reality it is actually a solo problem, because it doesn't
     have any example or anything or fading" — which is exactly what a
     fill-in-the-blank rung looks like when the blanks are missing AND the
     label is asking you to look for an example.

     The order is the one the course actually teaches: read it, fill in the
     step, read a solved one and do the same move, do it with nothing, then do
     it mixed in with everything else. */
  const STAGES = [
    { id: "lesson", label: "Lesson", blurb: "Read the explanation and run the examples." },
    { id: "faded", label: "Faded", blurb: "Most of the solution is written — supply the rest." },
    { id: "worked", label: "Worked example", blurb: "Read the solved example above it, then write this one yourself." },
    { id: "solo", label: "Solo", blurb: "No scaffold. You have earned it." },
    { id: "integrated", label: "Integrated", blurb: "Several concepts at once, unaided — the point of learning them." },
  ];

  /* Backend stage -> displayed rung.

     The server's `worked` is the lesson screen (it is the rung at which
     `LessonGate` takes over and no drill is served); its `faded` is the
     blank-filling rung and its `partial` is the read-an-example rung. Renaming
     those in the backend rewrites every learner's stored
     `kc_ladder[kc].attempts[].stage`, so the display is corrected first and the
     state migration follows separately. */
  const STAGE_ALIASES = {
    worked: "lesson",
    faded: "faded",
    partial: "worked",
    solo: "solo",
    independent: "solo",
    // Already-new vocabulary passes through untouched, so this file needs no
    // edit on the day the backend switches over.
    lesson: "lesson",
    integrated: "integrated",
  };

  const _index = (stage) => STAGES.findIndex((s) => s.id === stage);

  /* WHERE THE NEXT RUNG STARTS.

     The ladder promotes on the Wilson LOWER bound of the concept's own attempt
     record — `app/kc_graph.py` `_stage_from`, against `PROMOTE_LO`. The bar
     beside these numbers already draws that interval, so the threshold is a
     mark on the same track: when the LEFT END of the bar crosses it, the next
     question comes with less support. That is the whole promotion rule, drawn
     rather than described, and it is the answer to "there should be a clear
     threshold beyond which it moves you from one stage to a different stage."

     Keyed by DISPLAYED rung, which is why the names look shifted against
     `PROMOTE_LO`: the backend's `partial` is this file's `worked` (see
     STAGE_ALIASES). Three rungs have no entry, for three reasons — `lesson` is
     left by reading the page rather than by scoring, `solo` is the top of the
     per-concept ladder, and `integrated` is not reached by clearing a
     threshold on ONE concept at all.

     ⚠️ These are a copy of a backend constant. `practice/watch.py` reads both
     and fails if they drift, because a threshold drawn in the wrong place is
     worse than none: the learner clears the mark and nothing happens. */
  const PROMOTE_AT = {
    faded: 0.34,  // → worked example  (backend PROMOTE_LO.faded)
    worked: 0.51, // → solo            (backend PROMOTE_LO.partial)
  };

  const esc = (value) =>
    String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  const _el = (id) => document.getElementById(id);

  /* The concept currently on screen. `stage` is cached rather than passed
     around because a promotion can land mid-screen with nothing else in hand,
     and the difficulty bar's threshold is read back off it (`promotionMark`)
     long after `show` returned. */
  let current = { kc: null, stage: null, support: undefined };

  const normalizeStage = (stage) => STAGE_ALIASES[stage] || null;

  const _openGraph = (kc) => {
    if (!kc) return;
    if (typeof switchTab === "function") switchTab("knowledge-graph");
    requestAnimationFrame(() => {
      if (typeof window.deltaFocusConceptGraphKc === "function") {
        window.deltaFocusConceptGraphKc(kc);
      }
    });
  };

  /* WHERE THE NEXT RUNG STARTS, on the difficulty axis.

     The strip used to draw two bars of its own: an interval for the concept's
     record, with this threshold marked on it, and a second small bar for
     difficulty. Both are gone — the difficulty bar under the strip
     (index.html #target-difficulty, drawn by bars.js + difficulty-bar.js) is
     the one place either quantity is shown now.

     What survives here is the threshold itself, because the rung is what this
     module knows and nothing else does. `bound` is the Wilson lower bound the
     ladder promotes on; `difficulty-bar.js` converts it to a point on the
     0-100 difficulty scale, which is exact — the aim the queue serves IS
     `_DIFF_FLOOR + _DIFF_SPAN * bound` for the concept on screen. Null on the
     three rungs that are not left by clearing a number. */
  const promotionMark = () => {
    const bound = PROMOTE_AT[current.stage];
    const next = STAGES[_index(current.stage) + 1];
    if (bound === undefined || !next) return null;
    return { bound, stage: current.stage, next: next.label };
  };


  /* The title carries the rung's POSITION as well as its name.
     A tooltip reading "Faded — most of the solution is written" tells a learner
     what this rung asks of them but not where it sits, and the dots only convey
     position to someone who already knows the sequence runs left to right. Four
     of four is also the fact that makes the last one feel earned. */
  /* When the drill on screen carries neither blanks nor an example, the rung's
     own description is a promise the page does not keep — that is the
     "it says worked when it's actually a solo problem" report. The dot stays
     where the mastery record puts it (the rung is what the next promotion is
     measured against) and says what is actually there instead.

     Named per rung, because the two supported rungs promise different things
     and a message covering both would be wrong on one of them: a faded drill
     with no blanks still has no example either, but a worked-example rung with
     no example may well have blanks, and telling that learner "no blanks" is
     the same kind of lie in the other direction. */
  const NO_SUPPORT_BLURB = {
    faded: "No blanks were written for this one — write it unaided.",
    worked: "No example was written for this one — write it unaided.",
  };
  const _missingBlurb = (id) =>
    NO_SUPPORT_BLURB[id] || "The scaffold for this rung is not on the page — write it unaided.";

  const _stagesHtml = (stage, support) => {
    const active = _index(stage);
    return STAGES.map((s, i) => {
      const state = i < active ? "is-done" : i === active ? "is-active" : "is-todo";
      const blurb = i === active && support === false ? _missingBlurb(s.id) : s.blurb;
      const title = `Step ${i + 1} of ${STAGES.length} — ${s.label}. ${blurb}`;
      return (
        `<li class="stage-dot ${state}" data-stage="${esc(s.id)}" title="${esc(title)}"` +
        (i === active ? ' aria-current="step"' : "") +
        ">" +
        '<span class="stage-dot-mark" aria-hidden="true"></span>' +
        `<span class="stage-dot-label">${esc(s.label)}</span>` +
        "</li>"
      );
    }).join("");
  };

  /* Show the bar for one concept.

     `stage` accepts either vocabulary. An unrecognised stage hides the dots
     rather than guessing a position — showing the learner the wrong rung is
     worse than showing them none. */
  const show = ({ kc, title, eyebrow, stage, difficulty, support } = {}) => {
    const host = _el("concept-topbar");
    if (!host) return;
    const normalized = normalizeStage(stage);
    current = { kc: kc || null, stage: normalized, support };

    const eyebrowEl = _el("concept-topbar-eyebrow");
    if (eyebrowEl) {
      // With no concept there is nothing to label, and "Concept" above an empty
      // name reads as a failed load. The difficulty-only strip says what it is.
      eyebrowEl.textContent = kc ? eyebrow || "Concept" : "This problem";
    }

    const kcBtn = _el("concept-topbar-kc");
    if (kcBtn) {
      kcBtn.textContent = title || kc || "";
      kcBtn.dataset.kc = kc || "";
      kcBtn.hidden = !kc;
      kcBtn.title = kc ? `Open “${kc}” in the knowledge graph` : "";
      kcBtn.onclick = () => _openGraph(kcBtn.dataset.kc);
    }

    const stagesEl = _el("concept-topbar-stages");
    if (stagesEl) {
      stagesEl.innerHTML = normalized ? _stagesHtml(normalized, support) : "";
      stagesEl.hidden = !normalized;
    }

    /* The difficulty bar under the strip carries both numbers this function
       used to draw itself: `difficulty` is the rating of the problem on screen
       (its accent tick) and the rung decides where the promotion threshold
       sits. Lesson screens pass no rating — a page of prose has no difficulty,
       and inventing one would be the strip's only dishonest field. */
    window.DifficultyBar?.setProblem(difficulty);
    window.DifficultyBar?.refreshThresholds();
    host.classList.remove("hidden");
  };

  /* Move the dots without touching anything else — used when a rung is earned
     mid-screen (reading the worked example promotes immediately). */
  const setStage = (stage) => {
    const normalized = normalizeStage(stage);
    if (!normalized) return;
    current.stage = normalized;
    const stagesEl = _el("concept-topbar-stages");
    if (stagesEl) {
      stagesEl.innerHTML = _stagesHtml(normalized, current.support);
      stagesEl.hidden = false;
    }
    // The promotion threshold belongs to the rung, so a mid-screen promotion
    // has to move it — otherwise the bar keeps the previous rung's mark, which
    // the learner has by definition just cleared.
    window.DifficultyBar?.refreshThresholds();
  };

  const hide = () => {
    const host = _el("concept-topbar");
    if (host) host.classList.add("hidden");
    current = { kc: null, stage: null, support: undefined };
    /* The difficulty bar below does NOT go with it. This is also the path a
       KC-less question takes (a diagnostic probe, or the guest queue, which has
       no ladder at all), and the aim is still a true reading for those — see
       ladder.js `_syncTopbar`. What must go is the part of that bar this strip
       was the source of: the rung's threshold, and the tick for a problem whose
       rating we no longer have. */
    window.DifficultyBar?.setProblem(null);
    window.DifficultyBar?.refreshThresholds();
  };

  const activeKc = () => current.kc;

  return {
    show,
    hide,
    promotionMark,
    setStage,
    activeKc,
    normalizeStage,
    STAGES,
  };
})();

window.ConceptTopbar = ConceptTopbar;
