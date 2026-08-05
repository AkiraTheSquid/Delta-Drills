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
     * WHERE am I in the sequence?     — four dots, all four always visible
     * HOW WELL am I doing on it?      — the interval, not the point estimate
     * HOW HARD is this one?           — the problem's rating out of 100, on a
                                         track whose fill is the difficulty the
                                         queue is currently aiming at

   THE FOUR RUNGS

   The scaffold sequence is `lesson -> worked -> faded -> solo`: read the
   teaching page, then solve beside a visible example, then finish a partly
   written solution, then work unaided. The backend still speaks the older
   vocabulary (`worked, faded, partial, solo`) where its `worked` means the
   lesson page; `STAGE_ALIASES` maps one onto the other so the display can be
   correct before the state migration lands. That mapping is the only place the
   two vocabularies meet — do not scatter it.

   All four dots are always drawn, including the ones already passed and the
   ones not yet reached. A single label ("Faded") tells a learner what they are
   doing; four dots tell them how much support they have already given up and
   how much is left, which is the thing that makes the ladder legible as a
   ladder rather than as an arbitrary change of question format.
   ================================================================ */

const ConceptTopbar = (() => {
  "use strict";

  // Display order. Left to right is decreasing support.
  const STAGES = [
    { id: "lesson", label: "Lesson", blurb: "Read the explanation and run the examples." },
    { id: "worked", label: "Worked", blurb: "Solve beside a worked example you can still see." },
    { id: "faded", label: "Faded", blurb: "Most of the solution is written — supply the rest." },
    { id: "solo", label: "Solo", blurb: "No scaffold. You have earned it." },
  ];

  /* Backend stage -> displayed rung.

     The server's `worked` is the lesson screen (it is the rung at which
     `LessonGate` takes over and no drill is served), and its `faded` and
     `partial` are the two supported drill rungs. Renaming those in the backend
     rewrites every learner's stored `kc_ladder[kc].attempts[].stage`, so the
     display is corrected first and the state migration follows separately. */
  const STAGE_ALIASES = {
    worked: "lesson",
    faded: "worked",
    partial: "faded",
    solo: "solo",
    independent: "solo",
    // Already-new vocabulary passes through untouched, so this file needs no
    // edit on the day the backend switches over.
    lesson: "lesson",
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

     Keyed by DISPLAYED rung, which is why the numbers look shifted against
     `PROMOTE_LO`: the backend's `faded` is this file's `worked` and its
     `partial` is this file's `faded` (see STAGE_ALIASES). Two rungs have no
     entry, for two different reasons — `lesson` is left by reading the page,
     not by scoring, and `solo` is the top of the ladder.

     ⚠️ These are a copy of a backend constant. `practice/watch.py` reads both
     and fails if they drift, because a threshold drawn in the wrong place is
     worse than none: the learner clears the mark and nothing happens. */
  const PROMOTE_AT = {
    worked: 0.34, // → faded   (backend PROMOTE_LO.faded)
    faded: 0.51,  // → solo    (backend PROMOTE_LO.partial)
  };

  const esc = (value) =>
    String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  const _el = (id) => document.getElementById(id);

  /* The concept currently on screen, so `update()` can refresh the estimate
     after a submit without the caller having to re-supply everything. */
  /* `estimate` is kept because the threshold mark depends on the RUNG as well
     as on the numbers, and the two arrive from different places at different
     times: `setStage` promotes mid-screen with no estimate in hand, `setEstimate`
     lands a new interval with no rung in hand. Without the cache, whichever
     arrived second would draw the other one's half from stale state. */
  let current = { kc: null, stage: null, estimate: null };

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

  /* The estimate, rendered as an interval.

     Deliberately not the point estimate: 2-for-2 reads as 100% and tells the
     learner nothing about how little that rests on. The bar is drawn from the
     interval too — a wide bar IS the message that the number is not yet worth
     much, and it is the same quantity the ladder promotes on, so what the
     learner sees is what the system is actually deciding with. */
  const _estHtml = (est, stage) => {
    const gate = PROMOTE_AT[stage];
    const next = STAGES[_index(stage) + 1];
    /* The threshold is drawn even with no attempts behind it. A learner on an
       untouched concept is exactly the one who benefits from seeing where the
       bar has to get to, and hiding it until the first answer would make the
       mark look like something the first answer caused. */
    const gateHtml =
      gate === undefined || !next
        ? ""
        /* Focusable, and labelled, not just titled. The rule this mark carries
           exists nowhere else on the page — there is no room for "promote at
           34%" on a 110px track — so a `title` on an empty non-focusable span
           would put the whole promotion rule behind a mouse hover. `tabindex`
           and `role="img"` with the same sentence as the accessible name make
           it reachable by keyboard and readable by a screen reader. */
        : '<span class="concept-topbar-est-gate" role="img" tabindex="0" ' +
          `style="left:${(gate * 100).toFixed(1)}%" ` +
          `aria-label="${esc(
            `Promotion threshold: ${Math.round(gate * 100)} percent. ` +
            `Reaching it moves you to ${next.label}.`,
          )}" ` +
          `title="${esc(
            `Push the left end of this bar past ${Math.round(gate * 100)}% and the next ` +
            `question moves you to ${next.label}. It is the left end and not the middle ` +
            "because a short streak is not evidence yet — more answers narrow the bar, " +
            "which is what carries it across.",
          )}"></span>`;
    if (!est || !est.n) {
      return (
        '<span class="concept-topbar-est-count">0/0</span>' +
        '<span class="concept-topbar-est-bar">' +
        gateHtml +
        "</span>" +
        '<span class="concept-topbar-est-new" ' +
        'title="No graded attempts at this concept yet — nothing to estimate from.">' +
        "no attempts yet</span>"
      );
    }
    const lo = Math.max(0, Math.min(1, est.ci?.[0] ?? 0));
    const hi = Math.max(0, Math.min(1, est.ci?.[1] ?? 1));
    const loPct = Math.round(lo * 100);
    const hiPct = Math.round(hi * 100);
    const left = (lo * 100).toFixed(1);
    // A hairline interval still has to be visible, hence the 1% floor — but the
    // track no longer clips (the promotion mark has to overhang it), so the
    // floor is also capped at what is left of the track. Otherwise a learner at
    // 99-100% draws a sliver past the end of the bar it is supposed to be in.
    const width = Math.min(100 - lo * 100, Math.max(1, (hi - lo) * 100)).toFixed(1);
    return (
      `<span class="concept-topbar-est-count">${est.correct}/${est.n}</span>` +
      '<span class="concept-topbar-est-bar">' +
      `<span class="concept-topbar-est-fill" style="left:${left}%;width:${width}%"></span>` +
      gateHtml +
      "</span>" +
      `<span class="concept-topbar-est-range">${loPct}–${hiPct}%</span>`
    );
  };

  /* The difficulty of the problem on screen, and the difficulty being aimed at.

     Two different numbers, deliberately shown together. `problem` is what this
     question is rated — a fixed property of the item, out of 100. `target` is
     where the adaptive queue currently thinks the learner is, which is the
     number that moves: answer correctly and the next question is pulled from
     higher up the scale. Showing only the first would make the ladder look
     static; showing only the second would not tell the learner anything about
     the problem actually in front of them.

     The fill is the target and the tick is the problem, so the gap between
     them is readable at a glance: tick ahead of the fill means this one is a
     stretch, tick behind it means it is consolidation.

     A question with no rating hides the whole segment rather than drawing an
     empty track — an unrated problem is not a zero-difficulty problem. */
  /* The target the bar was last drawn at, so the next draw can show the MOVE
     and not just the new position. Module-level because `setDifficulty` rebuilds
     the segment's markup wholesale — anything remembered in the DOM is thrown
     away with it. Declared above `_diffHtml`, which reads it. */
  let _lastTarget = null;

  const _diffHtml = (problem, target) => {
    const p = Number.isFinite(problem) ? Math.max(0, Math.min(100, problem)) : null;
    if (p === null) return "";
    const t = Number.isFinite(target) ? Math.max(0, Math.min(100, target)) : null;
    const title =
      `This problem is rated ${Math.round(p)} out of 100.` +
      (t === null
        ? ""
        : ` The bar is the difficulty being served to you right now (${Math.round(t)}), ` +
          "which moves as you answer.");
    // Where the bar stood before this answer moved it. `null` on the first
    // question of a session — there is no move to show, so nothing is drawn.
    const from = Number.isFinite(_lastTarget) ? _lastTarget : null;
    const moved = t !== null && from !== null && Math.abs(t - from) >= 0.05;
    const lo = moved ? Math.min(from, t) : t;
    const delta = moved ? Math.abs(t - from) : 0;
    return (
      `<span class="concept-topbar-diff-label" title="${esc(title)}">Difficulty</span>` +
      `<span class="concept-topbar-diff-bar" title="${esc(title)}" aria-hidden="true">` +
      (t === null
        ? ""
        // The fill stops at the LOWER of the two, so the moving part is drawn
        // once — as the delta span — instead of being half-hidden under a fill
        // that already covers it.
        : `<span class="concept-topbar-diff-fill" style="width:${lo.toFixed(1)}%"></span>`) +
      (moved
        ? `<span class="concept-topbar-diff-delta ${t > from ? "is-gain" : "is-loss"}" ` +
          `style="left:${lo.toFixed(1)}%;width:${t > from ? 0 : delta.toFixed(1)}%"` +
          `data-delta="${delta.toFixed(1)}"></span>` +
          // Where this answer started from, so the move is legible as a move
          // rather than as a bar that is simply a different length than last
          // time. Drawn at `from`, which is the edge both animations run from.
          `<span class="concept-topbar-diff-from" style="left:${from.toFixed(1)}%"></span>`
        : "") +
      `<span class="concept-topbar-diff-tick" style="left:${p.toFixed(1)}%"></span>` +
      "</span>" +
      `<span class="concept-topbar-diff-value">${Math.round(p)}<span ` +
      'class="concept-topbar-diff-max">/100</span></span>'
    );
  };

  /* Redraw the difficulty segment on its own. Separate from `show` for the
     same reason `setEstimate` is: a submit moves the target, and re-rendering
     the concept name and dots to move one bar would flash the part of the
     strip that is supposed to hold still. */
  const setDifficulty = (problem, target) => {
    const host = _el("concept-topbar-diff");
    if (!host) return;
    const html = _diffHtml(problem, target);
    host.innerHTML = html;
    host.hidden = !html;

    /* Run the move.

       Both directions animate the same span from one width to another, which is
       what makes them read as one measure rather than two effects: a GAIN grows
       green from where you were out to where you now are, and a LOSS starts at
       the length you had and collapses red back to what is left. The loss is
       anchored on its right edge (see the CSS) so it recedes towards the new
       value instead of sliding away from it.

       Two frames, not one. The element has to be laid out at its starting width
       before the end width is set, or the browser coalesces both into a single
       style computation and there is no transition to watch — the bar simply
       appears at its final length. */
    const span = host.querySelector(".concept-topbar-diff-delta");
    if (span) {
      const delta = Number(span.dataset.delta);
      const gain = span.classList.contains("is-gain");
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          if (!span.isConnected) return;
          span.style.width = gain ? `${delta}%` : "0%";
        });
      });
    }

    // Recorded after the draw, so this render still had the previous value to
    // compare against. Only a real number counts: a lesson screen passes no
    // target, and treating that as "moved to nothing" would animate a collapse
    // to zero on the way into every lesson.
    if (Number.isFinite(target)) _lastTarget = Math.max(0, Math.min(100, target));
  };

  /* The title carries the rung's POSITION as well as its name.
     A tooltip reading "Faded — most of the solution is written" tells a learner
     what this rung asks of them but not where it sits, and the dots only convey
     position to someone who already knows the sequence runs left to right. Four
     of four is also the fact that makes the last one feel earned. */
  const _stagesHtml = (stage) => {
    const active = _index(stage);
    return STAGES.map((s, i) => {
      const state = i < active ? "is-done" : i === active ? "is-active" : "is-todo";
      const title = `Step ${i + 1} of ${STAGES.length} — ${s.label}. ${s.blurb}`;
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
  const show = ({ kc, title, eyebrow, stage, estimate, difficulty, target } = {}) => {
    const host = _el("concept-topbar");
    if (!host) return;
    const normalized = normalizeStage(stage);
    current = { kc: kc || null, stage: normalized, estimate: null };

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
      stagesEl.innerHTML = normalized ? _stagesHtml(normalized) : "";
      stagesEl.hidden = !normalized;
    }

    // The estimate is a per-concept record, so without a concept there is
    // nothing it could be an estimate OF — "no attempts yet" beside a nameless
    // problem invites the reading that the learner has done nothing at all.
    const estEl = _el("concept-topbar-est");
    if (estEl) estEl.hidden = !kc;
    if (kc) setEstimate(estimate);
    // Lesson screens pass neither number — a page of prose has no difficulty
    // rating, and inventing one would be the strip's only dishonest field.
    setDifficulty(difficulty, target);
    host.classList.remove("hidden");
  };

  /* Refresh only the estimate — what a submit changes. Kept separate so a
     grade does not have to re-render the concept name and dots, which would
     flash the one part of the page that is supposed to be stable. */
  const setEstimate = (estimate) => {
    current.estimate = estimate || null;
    const estEl = _el("concept-topbar-est");
    // The threshold belongs to the rung, and the rung is `current.stage` rather
    // than an argument: a submit refreshes the estimate without re-rendering the
    // strip, and a caller that had to remember to re-supply the rung would
    // eventually forget and leave last rung's mark under this rung's bar.
    if (estEl) estEl.innerHTML = _estHtml(current.estimate, current.stage);
  };

  /* Move the dots without touching anything else — used when a rung is earned
     mid-screen (reading the worked example promotes immediately). */
  const setStage = (stage) => {
    const normalized = normalizeStage(stage);
    if (!normalized) return;
    current.stage = normalized;
    const stagesEl = _el("concept-topbar-stages");
    if (stagesEl) {
      stagesEl.innerHTML = _stagesHtml(normalized);
      stagesEl.hidden = false;
    }
    // The promotion mark moves with the rung — 34% to leave Worked, 51% to leave
    // Faded. Redrawn from the cached interval so a mid-screen promotion does not
    // leave the previous rung's threshold sitting under the new rung's dots.
    setEstimate(current.estimate);
  };

  const hide = () => {
    const host = _el("concept-topbar");
    if (host) host.classList.add("hidden");
    current = { kc: null, stage: null, estimate: null };
  };

  const activeKc = () => current.kc;

  return {
    show,
    hide,
    setEstimate,
    setDifficulty,
    setStage,
    activeKc,
    normalizeStage,
    STAGES,
  };
})();

window.ConceptTopbar = ConceptTopbar;
