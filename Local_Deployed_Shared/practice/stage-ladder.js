/* ================================================================
   STAGE LADDER — the one progress readout on the practice screen
   ================================================================

   Replaces four things that used to share this screen and disagree with each
   other: the concept strip's stage dots, the full-width target-difficulty bar
   (`difficulty-bar.js` + the `#target-difficulty` half of `bars.js`), the EWMA
   accuracy bar, and — in the knowledge-graph focus flow only — the competency
   bar's 0-95% mastery track.

   🔴 WHY THAT WAS WORTH DELETING, not just tidying.

   Those bars measured three different quantities and two of them used the word
   "Faded" for two unrelated numbers:

     - target difficulty   the AIM: 20 + 80 * (Wilson lower bound) + the
                           learner's own felt-difficulty offset. Not a level,
                           and not the rating of the problem on screen — which
                           is why a bar reading 33.8 could sit above a tick
                           reading "this problem · 14" and look broken.
     - competency          the BKT posterior for the SUBTOPIC, with gates at
                           0.85 and 0.95 and its own phase bands at 0.40/0.75,
                           printing "Faded drills" off a number that has
                           nothing to do with the ladder rung named Faded.
     - accuracy            EWMA accuracy over the subtopic, blank until
                           calibration finishes, which is most of the time a
                           learner is looking at it.

   The report was "there are THREE different bars in this view … three
   different views is kind of insane". It was worse than insane: it was three
   answers to "how am I doing" that could not all be right at once.

   WHAT THIS BAR SAYS INSTEAD — one question, one answer.

   The ladder is the thing the learner actually moves along, so the ladder is
   the bar. FOUR rungs, always all four, left to right in decreasing support:
   Lesson, Faded, Worked example, Solo. The rung you are on is named ABOVE the
   track, in words, and lit under it.

   ONE BAR, CUT — not four bars in a row, which is how this was drawn until
   2026-08-22. Each rung had its own bordered track and its own fill, and four
   meters side by side read as four quantities when the learner is asking one:
   how far into this concept am I. Now a single rounded track runs the width of
   the strip, the fill crosses it end to end, and the rung boundaries are
   chevron SEAMS punched out of it — sideways Vs pointing right, in the panel's
   own colour, so the bar reads as four arrow segments of one track. The fill
   crossing a seam is the promotion.

   The percentage rides ABOVE the fill's leading edge and says what it is
   measuring: "20% understanding of concept array reshape". A bare number over
   a bar is the thing the four old bars were each doing, and it is why nobody
   could tell them apart.

   🔴 THERE IS NO FIFTH SECTION, and "Integrated" was one for two days.

   `solo` is the top rung — `kc_graph.LADDER_STAGES` has four names and the
   promotion arithmetic reads them back. (Reaching it is not the same as being
   done: `kc_is_learned` wants the BKT posterior over the concept or its whole
   pool served, which is why the bar tops out at 75% — see `_overall`.)
   `ladder_integrated` is not a rung and the backend says so
   twice in as many files (`practice_schemas.py`, `lessons.is_integrated`): it
   is a property of the PROBLEM — this one happens to use the concept beside
   others already taught — computed at serve time and stored nowhere, while the
   record keeps saying `solo`.

   Drawn as a dot that lights per question, that was fine. Drawn as the fifth
   section of a track that fills left to right, it is not: Solo went `is-done`
   and Integrated `is-active` on an integrated problem and both snapped back on
   the next one, so the far end of a monotone progress bar flickered while the
   learner's rung never moved — and the bar showed a section still to clear at
   the exact moment the system had already marked the concept learned. It is a
   chip beside the rung name now, which is what it always was.

   Difficulty survives as one line of text underneath, which is all it ever
   deserved: it is an input to question SELECTION, not a measure of the
   learner.

   🔴 WHAT FILLS THE CURRENT SECTION — and what it must not claim.

   The rung moves on the Wilson LOWER bound of this concept's own recent
   attempts, against `PROMOTE_AT` (mirrored from `app/kc_graph.py` PROMOTE_LO).
   So the partial fill is `bound / threshold` and nothing else. Two rules come
   out of that and both are load-bearing:

     - A rung with no threshold gets NO partial fill. `lesson` is left by
       reading the page and `solo` is the top of the per-concept ladder, so a
       progress fill on either would be a promise about a promotion that no
       number drives.
     - There are TWO routes out of a rung and the fill takes whichever is
       further along: the Wilson bound against `PROMOTE_AT`, or a run of
       `_PROMOTE_STREAK` correct answers, which `kc_estimate` now sends as
       `streak` / `streak_needed`. Taking the lower of the two would leave the
       bar at a third while the very next correct answer promoted.
   ================================================================ */

const StageLadder = (() => {
  "use strict";

  /* Display order. Left to right is decreasing support.

     🔴 FADED AND WORKED USED TO BE THE WRONG WAY ROUND, and the error was not
     cosmetic. The backend's `faded` rung serves a fill-in-the-blank drill and
     was displayed as "Worked — solve beside a worked example you can still
     see"; its `partial` rung serves a drill with a solved example above it and
     was displayed as "Faded — most of the solution is written". Each named the
     other one's rung, so the screen promised support that was not on the page.
     The order below is the one the course actually teaches. */
  const STAGES = [
    { id: "lesson", label: "Lesson", blurb: "Read the explanation and run the examples." },
    { id: "faded", label: "Faded", blurb: "Most of the solution is written — supply the rest." },
    { id: "example", label: "Worked example", blurb: "Read the solved example above it, then write this one yourself." },
    { id: "solo", label: "Solo", blurb: "No scaffold. You have earned it." },
  ];

  /* The ⓘ beside each rung name. Keyed by DISPLAYED rung; the copy lives in
     `infotips-registry.js` under these exact keys, and `watch.py`'s
     `check_infotips` reads this file as one of its anchor sources so a rung
     with no copy (or copy with no rung) fails there rather than opening an
     empty panel in front of a learner. */
  const INFO_KEY = {
    lesson: "ladder.lesson",
    faded: "ladder.faded",
    example: "ladder.example",
    solo: "ladder.solo",
  };

  /* Backend stage -> displayed rung. The server's `worked` is the lesson screen
     (it is the rung at which `LessonGate` takes over and no drill is served);
     its `faded` is the blank-filling rung and its `partial` is the
     read-an-example rung. Renaming those in the backend rewrites every
     learner's stored `kc_ladder[kc].attempts[].stage`, so the display is
     corrected first and the state migration follows separately. This mapping is
     the only place the two vocabularies meet — do not scatter it.

     🔴 THE THIRD RUNG'S ID IS `example`, NOT `worked`, and that is the point.
     The word `worked` is TAKEN: it is the backend's name for the lesson
     screen. If the rung the learner sees as "Worked example" were also called
     `worked` internally, this table would have to decide which sense a caller
     meant and would silently pick one — a caller passing display vocabulary
     would land on Lesson. Every key below now means exactly one thing, so
     `normalizeStage` is total and a mistake here is impossible rather than
     merely unlikely. The LABEL is still "Worked example"; only the id differs.
  */
  const STAGE_ALIASES = {
    worked: "lesson",
    faded: "faded",
    partial: "example",
    solo: "solo",
    independent: "solo",
    // Already-new vocabulary passes through untouched, so this file needs no
    // edit on the day the backend switches over.
    lesson: "lesson",
    example: "example",
    // NOT a rung of its own — see the header. Mapped to `solo` rather than
    // dropped so the table stays total: `ladder_integrated` rides on a solo
    // record, so a caller that still passes the word lands on the rung the
    // record actually says instead of on `null`, which draws no track at all.
    integrated: "solo",
  };

  /* Where the next rung starts, as a Wilson LOWER bound.

     Keyed by DISPLAYED rung, which is why the names look shifted against
     `PROMOTE_LO`: the backend's `partial` is this file's `example`.

     ⚠️ A copy of a backend constant. `practice/watch.py` reads both and fails
     if they drift, because a section drawn as half full against the wrong
     threshold is worse than one drawn empty: the learner fills it and nothing
     happens. */
  const PROMOTE_AT = {
    faded: 0.34,   // → worked example  (backend PROMOTE_LO.faded)
    example: 0.51, // → solo            (backend PROMOTE_LO.partial)
  };

  const esc = (value) =>
    String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  const _el = (id) => document.getElementById(id);
  const _index = (stage) => STAGES.findIndex((s) => s.id === stage);

  /* The concept currently on screen. `stage` is cached rather than passed
     around because a promotion can land mid-screen with nothing else in hand,
     and the caption is redrawn from it long after `show` returned. */
  let current = { kc: null, title: null, stage: null, support: undefined,
                  bound: null, streak: null, streakNeeded: null, estStage: null,
                  integrated: false };
  /* The difficulty caption's two numbers, held so a mid-screen promotion can
     redraw the line without them being re-fetched — and so an aim that arrives
     before a rating (or the other way round) does not blank the other half. */
  let aimValue = null;
  let problemValue = null;
  let extraNote = "";

  /* Own-property lookup, not a plain index: the alias table is an object
     literal, so `__proto__` and `constructor` would resolve through the
     prototype and return something truthy for a stage no backend ever sends.
     The whole point of the table is that an unknown stage becomes null. */
  const normalizeStage = (stage) =>
    (Object.prototype.hasOwnProperty.call(STAGE_ALIASES, stage) && STAGE_ALIASES[stage]) || null;

  /* The Wilson lower bound this concept's record currently stands at.
     `ladder_estimate.ci` is [lower, upper] from `kc_graph.kc_estimate`; the
     ladder promotes on the lower one, so that is the only half drawn. */
  const _boundOf = (estimate) => {
    const ci = estimate && estimate.ci;
    return Array.isArray(ci) && Number.isFinite(ci[0]) ? ci[0] : null;
  };

  /* The run of correct answers standing right now, and how long a run has to
     be to promote on its own (`kc_graph._PROMOTE_STREAK`). Read from the
     payload rather than hardcoded: this is the only one of the two routes the
     learner can watch move, so a client that guessed the length wrong would
     draw a bar that fills at the wrong rate. */
  const _streakOf = (estimate) => {
    const n = estimate && estimate.streak;
    const need = estimate && estimate.streak_needed;
    return {
      streak: Number.isFinite(n) ? n : null,
      streakNeeded: Number.isFinite(need) && need > 0 ? need : null,
      /* WHICH rung this run was counted against (`kc_estimate` sends its own
         `stage`). The backend scopes the run to the rung the learner is on, so
         a run of three at `faded` reads as zero the moment it promotes — and
         this readout deliberately keeps showing the old rung until the next
         question. Without the rung the number came from, that zero would be
         drawn as "no progress on the rung you are looking at", which is the
         opposite of what happened. */
      estStage: normalizeStage(estimate && estimate.stage),
    };
  };

  /* Jump to this concept on the map.

     🔴 `window.deltaFocusConceptGraphKc` and not an event. The graph defines
     that function (`concept-graph/lesson-graph.js`) and it is what waits for
     the tab's data AND its layout before moving the viewport; a CustomEvent
     has no listener anywhere and would open the tab on whatever node was
     already centred — the button appearing to work while doing half its job.
     `practice/graph-jump.js` calls the same function the same way. The rAF is
     the tab switch: focusing before the graph is on screen measures a hidden
     canvas. */
  const _openGraph = (kc) => {
    if (!kc) return;
    if (typeof switchTab === "function") switchTab("knowledge-graph");
    requestAnimationFrame(() => {
      if (typeof window.deltaFocusConceptGraphKc === "function") {
        window.deltaFocusConceptGraphKc(kc);
      }
    });
  };

  /* How far through the CURRENT section, 0..1. Null where the section has no
     threshold — see the header: those rungs are not left by clearing a number,
     and a fill on them would be a claim no arithmetic backs.

     TWO routes out of a rung, and the backend takes whichever arrives first
     (`kc_graph._stage_from`): the Wilson lower bound over the last twenty
     attempts clearing `PROMOTE_LO`, or a run of `_PROMOTE_STREAK` correct
     answers on its own. So the section fills to whichever is FURTHER along —
     the bar has to be the shorter of the two distances left, or it would sit
     at a third while the very next correct answer promotes.

     The run is the half a learner can watch move. The window average barely
     shifts on one answer out of twenty and, on a concept carrying old misses,
     can hold still for a dozen questions; a run advances by a third of the
     section every time they get one right, and goes back to nothing when they
     do not — which is also exactly what the rung does. */
  const _progress = () => {
    const threshold = PROMOTE_AT[current.stage];
    if (!threshold) return null;
    const shown = _index(current.stage);
    /* Which rung the estimate in hand describes. Equal to the rung on screen
       on every normal render; ahead of it for exactly as long as one graded
       answer, because `setProgress` moves the fill and not the rung. */
    const at = current.estStage ? _index(current.estStage) : shown;
    // The answer just given cleared this rung. Draw it FULL: its own fill is
    // now computed against a rung the learner has left, and letting it fall
    // back to the window average would run the bar BACKWARDS on a correct
    // answer — the one thing a progress bar may never do.
    if (at > shown) return 1;
    const parts = [];
    if (Number.isFinite(current.bound)) parts.push(current.bound / threshold);
    // The run promotes out of the rung it was counted against, so it is this
    // section's progress only while the two agree. Behind (a miss demoted the
    // concept mid-question) the window average carries the fill on its own.
    if (at === shown && Number.isFinite(current.streak) && current.streakNeeded) {
      parts.push(current.streak / current.streakNeeded);
    }
    if (!parts.length) return null;
    return Math.max(0, Math.min(1, Math.max(...parts)));
  };

  /* How far along the WHOLE ladder, 0..1 — the number the bar draws and the
     callout says as a percentage.

     Rungs behind the learner are cleared, the rung they are on is worth
     `_progress()` of its own width, and the ladder has `STAGES.length` of
     them. Equal widths are what lets the chevron seams sit at fixed quarters
     and still mean something: the fill crossing a seam IS the promotion.

     🔴 IT TOPS OUT AT (STAGES.length - 1) / STAGES.length, i.e. 75% on Solo,
     and that is not an off-by-one. `_progress()` returns null on Solo because
     no threshold sits above it — and `kc_is_learned` does NOT fire on arrival
     at Solo either: it wants the BKT posterior over the concept, or its whole
     question pool served. Neither number is in this payload, so the last
     quarter is the honest empty one. Filling it on arrival would tell the
     learner they were done with a concept the queue is still going to serve
     them. */
  const _overall = () => {
    const active = _index(current.stage);
    if (active < 0) return null;
    const partial = _progress();
    return (active + (partial === null ? 0 : partial)) / STAGES.length;
  };

  /* The seams: one per boundary BETWEEN rungs, so `STAGES.length - 1` of them
     at fixed fractions. Injected rather than written into index.html so a
     fifth rung — if the ladder ever grows one — cannot leave three cuts on a
     five-part bar. */
  const _seamsHtml = () =>
    STAGES.slice(1)
      .map((s, i) => {
        const at = ((i + 1) / STAGES.length) * 100;
        return `<span class="stage-ladder-seam" style="left:${at.toFixed(4)}%"></span>`;
      })
      .join("");

  /* One rung's cell: the name, and the ⓘ that explains it.

     🔴 THE DOT IS HAND-WRITTEN, not left to infotips.js to inject. Its scanner
     mints a dot beside any `[data-dd-info]` anchor — but it skips elements
     that ARE dots (`.dd-info`), and it only sweeps the ones it generated
     itself, so writing the button here is supported and is the only way the
     two label layers can stay pixel-identical: an asynchronously injected dot
     would appear in one layer and not the other, and the clip would then show
     two different label widths through the same window.

     `interactive` is false for the clipped copy. Same box, same glyph, no key
     — without a `data-dd-info` the scanner does not see it at all, and the
     layer is `aria-hidden` + `pointer-events: none` so the copy is invisible
     to both the reader and the mouse. */
  const _cellHtml = (s, i, active, interactive) => {
    const state = i < active ? "is-done" : i === active ? "is-active" : "is-todo";
    const title = `Step ${i + 1} of ${STAGES.length} — ${s.label}. ${s.blurb}`;
    const dot = interactive
      ? `<button class="dd-info stage-seg-info" type="button" data-dd-info="${esc(INFO_KEY[s.id])}"` +
        ` aria-expanded="false" aria-label="What is ${esc(s.label.toLowerCase())}?">i</button>`
      : '<span class="dd-info stage-seg-info" aria-hidden="true">i</span>';
    return (
      `<li class="stage-seg ${state}" data-stage="${esc(s.id)}"` +
      (interactive ? ` title="${esc(title)}"` : "") +
      (interactive && i === active ? ' aria-current="step"' : "") +
      `><span class="stage-seg-label">${esc(s.label)}</span>${dot}</li>`
    );
  };

  const _trackHtml = (interactive) => {
    const active = _index(current.stage);
    const cls = interactive ? "stage-ladder-track--base" : "stage-ladder-track--on";
    return (
      `<ol class="stage-ladder-track ${cls}"` +
      (interactive ? ' aria-label="Scaffold stage"' : ' aria-hidden="true"') +
      ">" +
      STAGES.map((s, i) => _cellHtml(s, i, active, interactive)).join("") +
      "</ol>"
    );
  };

  /* The rung names sit INSIDE the track, and a name inside a bar that fills
     underneath it has a contrast problem no single colour solves: at 38% the
     word "Faded" is half over the accent and half over the empty track, and in
     the light theme those two grounds are on opposite sides of black.

     So the whole row is drawn TWICE, stacked. The base layer is coloured for
     the EMPTY track; the copy on top is coloured for the FILL and clipped to
     exactly the filled width, so every pixel of every glyph is painted in the
     colour that suits the ground behind that pixel. `--dd-ladder-pct` on the
     bar is what the clip reads, so the two cannot drift apart — the same trick
     the level pill uses for its label (styles/xp.css).

     A cheaper version — one layer, colour chosen per label from whether its
     CENTRE is inside the fill — was the first attempt and is wrong exactly
     where it matters: the fill's leading edge spends most of its life crossing
     a label, which is the moment that label becomes unreadable. */
  const _renderMeter = () => {
    const meter = _el("stage-ladder-meter");
    if (!meter) return;
    const overall = _overall();
    meter.hidden = overall === null;
    if (overall === null) return;

    const pct = overall * 100;
    const bar = _el("stage-ladder-bar");
    if (bar) {
      /* Rebuilt only when the RUNG changes, which is what the labels depend
         on. Within one rung a graded answer moves the width and nothing else,
         and a fill element replaced on every reading is born at its final
         width — `transition: width` then has no value to animate from and the
         one moment the bar must visibly move is a jump cut. */
      const sig = `${current.stage}|${STAGES.length}`;
      if (bar.dataset.ladderSig !== sig) {
        bar.innerHTML =
          '<span class="stage-ladder-fill"></span>' +
          _seamsHtml() +
          _trackHtml(true) +
          _trackHtml(false);
        bar.dataset.ladderSig = sig;
      }
      const width = `${pct.toFixed(2)}%`;
      /* ONE number, read by both the fill's width and the clip on the
         on-accent label layer. Two separate writes are how a label ends up
         painted for a ground the fill has not reached yet. */
      bar.style.setProperty("--dd-ladder-pct", width);
      const fill = bar.querySelector(".stage-ladder-fill");
      if (fill) fill.style.width = width;
      bar.setAttribute("aria-valuenow", String(Math.round(pct)));
      bar.setAttribute("aria-valuetext", `${Math.round(pct)}% understanding`);
    }

    _renderCallout(pct);
  };

  /* The reading, as a notch UNDER the bar pointing up at the seam it
     describes: "38% understanding of concept array reshape".

     It hung ABOVE the bar first. Under it is better for one reason that is
     not taste: above, the label sits between the topbar and the track and
     reads as a heading for the whole strip; under, with an arrow on it, it
     is unmistakably a pointer at one position on one bar.

     Centred on the fill edge, except within a hair of either end, and then
     nudged back inside the strip by measurement — the thresholds are a guess
     at where a label runs out of room and cannot be more than a guess,
     because the label carries a concept NAME. */
  const _renderCallout = (pct) => {
    const meter = _el("stage-ladder-meter");
    const callout = _el("stage-ladder-callout");
    if (!meter || !callout) return;

    const textEl = _el("stage-ladder-callout-text");
    const name = current.title;
    if (textEl) {
      textEl.innerHTML =
        `<b>${Math.round(pct)}%</b> understanding` + (name ? " of concept" : "");
    }
    /* A KC-less item has no concept to name, so the clause is dropped rather
       than filled with a "This problem" fallback — "38% understanding of
       concept This problem" is not a sentence. */
    const kcBtn = _el("stage-ladder-kc");
    if (kcBtn) kcBtn.hidden = !name;

    /* What the rung asks of the learner used to be a whole row of the strip.
       It is the ⓘ on the rung's own name now — but the case where the page
       does NOT carry the scaffold the rung promises still has to be said out
       loud, because the learner is about to look for it. */
    const noteEl = _el("stage-ladder-note");
    if (noteEl) {
      const stage = STAGES[_index(current.stage)];
      const parts = [];
      if (stage && current.support === false) {
        parts.push(NO_SUPPORT_BLURB[stage.id] ||
          "The scaffold for this rung is not on the page — write it unaided.");
      }
      /* `competency-bar.js`'s TOPIC-level BKT reading. It had the difficulty
         caption to live in; that row is gone, so it lands here. The caller
         still owes it its own scope — a bare percentage next to this one is
         the one thing it must never be. */
      if (extraNote) parts.push(extraNote);
      noteEl.textContent = parts.join(" · ");
      noteEl.hidden = !parts.length;
    }

    callout.style.left = `${pct.toFixed(2)}%`;
    const shift = pct <= 8 ? 0 : pct >= 92 ? 100 : 50;
    callout.style.transform = `translateX(-${shift}%)`;

    const room = meter.getBoundingClientRect();
    const box = callout.getBoundingClientRect();
    let dx = 0;
    if (room.width && box.width) {
      dx = box.left < room.left
        ? room.left - box.left
        : box.right > room.right
          ? room.right - box.right
          : 0;
      if (dx) callout.style.transform = `translateX(calc(-${shift}% + ${dx.toFixed(2)}px))`;
    }

    _pushNotchAside(callout);
  };

  /* The notch between the panels carries the session controls and sits on the
     same line as this callout, centred on the seam between them. When the
     reading lands there, the reading wins and the notch steps aside.

     🔴 The one place this module touches anything outside the strip, and it is
     deliberate: the callout's position is not knowable from CSS — it follows a
     percentage that follows a learner — so the collision can only be resolved
     by whoever just placed it. `practice/notch-menu.js` owns everything else
     about that control. */
  const _pushNotchAside = (callout) => {
    const notch = document.getElementById("practice-notch");
    if (!notch) return;
    notch.style.transform = "";
    const at = notch.getBoundingClientRect();
    // Hidden with the split on the setup screen: nothing to move.
    if (!at.width) return;
    const box = callout.getBoundingClientRect();
    const GAP = 10;
    const clashes =
      box.bottom > at.top && box.top < at.bottom &&
      box.right + GAP > at.left && box.left - GAP < at.right;
    if (!clashes) return;
    const split = notch.parentElement
      ? notch.parentElement.getBoundingClientRect()
      : null;
    const right = box.right + GAP - at.left;   // move the notch right of the label
    const left = box.left - GAP - at.right;    // or left of it
    // Whichever side keeps the notch on screen; right unless it would run off.
    const dx = split && at.right + right > split.right ? left : right;
    notch.style.transform = `translate(calc(-50% + ${dx.toFixed(2)}px), -50%)`;
  };

  /* The rung's own promise, withdrawn when the page does not keep it.

     When the drill on screen carries neither blanks nor an example, the rung's
     ⓘ would describe a scaffold that is not there. The rung still stands — it
     is where the record puts them — so the name is kept and only the promise
     is withdrawn, in the callout's note. */
  const NO_SUPPORT_BLURB = {
    faded: "This one came through with no blanks — write it unaided.",
    example: "No solved example was available for this one — write it unaided.",
  };

  /* "Integrated" — a chip in the callout, not a section of the track. Gated on
     `solo` as well as on the flag: the backend only sets `ladder_integrated`
     on a solo record, and a chip claiming several concepts at once over a
     drill that still carries blanks would be the readout promising something
     the page is not doing. */
  const _renderNow = () => {
    const flagEl = _el("stage-ladder-flag");
    if (!flagEl) return;
    const integrated = !!current.integrated && current.stage === "solo";
    flagEl.textContent = integrated ? "Integrated" : "";
    flagEl.hidden = !integrated;
    flagEl.title = integrated
      ? "This problem uses the concept alongside others you have already been taught."
      : "";
  };

  /* 🪦 THE DIFFICULTY CAPTION IS GONE — 2026-08-22, at Seth's request.

     It read "Aiming at difficulty 38.0 · this problem is rated 22", then for
     one afternoon just "38.0 mastery" pinned to the right of the strip. Both
     are off the screen now: the strip is the bar, the bar's own reading, and
     nothing else. The rating of the item on screen was never a measure of the
     learner, and the aim names a queue mechanic rather than anything the
     learner is doing.

     `setDifficulty` still RECORDS both numbers and `show()` still replaces
     them unconditionally, which is what keeps a stale one from leaking onto
     the next screen if either is ever asked for again — `watch.py` holds that
     assignment in place. Nothing renders them.

     `extraNote` outlived them: it is `competency-bar.js`'s topic-level BKT
     reading and it now rides in the callout's note — see `_renderCallout`.

     CHIP FIRST below. The Integrated chip lives INSIDE the callout, and the
     callout is measured — after it is written — to keep it inside the strip
     and off the notch. A chip added after that measurement changes the width
     the placement was computed from. */
  const _render = () => {
    _renderNow();
    _renderMeter();
  };

  /* Show the readout for one concept.

     `stage` accepts either vocabulary. An unrecognised stage draws no sections
     rather than guessing a position — showing the learner the wrong rung is
     worse than showing them none. */
  const show = ({ kc, title, stage, difficulty, target, support, estimate,
                 eyebrow, integrated } = {}) => {
    const host = _el("stage-ladder");
    if (!host) return;
    /* A new concept invalidates the caption's mastery clause. `competency-bar.js`
       writes that clause from the BKT posterior for the concept's TOPIC — not
       a per-concept number, which is why it says so — and nothing in the
       readout distinguished it from the rest of the caption, so moving to
       another concept carried the previous one's reading across. Cleared
       on a KC CHANGE only: the same concept re-renders on every question, and
       blanking it there would drop the reading until the next graded answer.
       The competency bar republishes on its own events, so the clause comes
       back for the new concept as soon as there is one to state. */
    if ((kc || null) !== current.kc) extraNote = "";
    current = {
      kc: kc || null,
      /* The name the callout says out loud ("… understanding of concept array
         reshape"). Held rather than re-read off the button, because the
         button's text falls back to "This problem" on a KC-less item and the
         callout has to be able to tell that case apart and say nothing. */
      title: title || kc || null,
      stage: normalizeStage(stage),
      support,
      bound: _boundOf(estimate),
      ..._streakOf(estimate),
      /* Per-QUESTION, and re-read on every one: the previous problem's
         integration says nothing about this one, and the record underneath
         both of them says `solo` either way. */
      integrated: !!integrated,
    };
    /* The pair is REPLACED, not merged. A `show()` means a different concept
       is on screen, and both numbers describe the item served — not the
       readout. The lesson screen passes neither, so merging left the previous
       question's aim standing over a page with no problem on it at all
       ("Aiming at difficulty 40.0", nothing to aim at). The practice path
       re-supplies them in the SAME synchronous render: `ui.js` calls
       `LadderUI.decorate` — which lands here — and `setTargetDifficultyInitial`
       fifty lines further down the same function. */
    problemValue = Number.isFinite(difficulty) ? difficulty : null;
    aimValue = Number.isFinite(target) ? target : null;

    /* "PyTorch tensors · Concept 2 of 3" — where this concept sits in the
       lesson it belongs to. Only the lesson screen passes one; the practice
       queue has no such sequence to report, so the slot stays empty there
       rather than inventing a position. */
    const eyebrowEl = _el("stage-ladder-eyebrow");
    if (eyebrowEl) {
      eyebrowEl.textContent = eyebrow || "";
      eyebrowEl.hidden = !eyebrow;
    }

    const kcBtn = _el("stage-ladder-kc");
    if (kcBtn) {
      kcBtn.textContent = title || kc || "This problem";
      kcBtn.dataset.kc = kc || "";
      kcBtn.disabled = !kc;
      kcBtn.title = kc ? `Open “${kc}” in the knowledge graph` : "";
      kcBtn.onclick = () => _openGraph(kcBtn.dataset.kc);
    }
    /* Unhidden BEFORE the render, not after. `_renderMeter` measures the
       callout to keep it inside the strip, and a measurement taken while the
       section is still `display: none` comes back all zeroes — so the first
       question of a session would draw its label uncorrected. */
    host.classList.remove("hidden");
    _render();
  };

  /* A fresh reading for the concept already on screen, after a graded answer.

     🔴 The FILL moves and the RUNG does not. A promotion earned by the answer
     just given is real, but the problem is still on screen behind this
     readout — relabelling it "Worked example" while a faded drill sits under
     it is the same lie the four old bars were telling, in one bar. The rung
     catches up on the next render, which is where the next problem actually
     comes from that rung.

     Only `submit-local-eval` returns a fresh estimate (the Colab verdict and
     the einops fallback); `/api/practice/feedback` does not carry one, so the
     normal graded path still updates at the next question. Sending it from
     there is a backend change, deliberately not made in a UI pass. */
  const setProgress = (estimate) => {
    const bound = _boundOf(estimate);
    if (bound === null) return;
    current.bound = bound;
    /* The run, not just the average. This is the call that lands the moment an
       answer is graded, and the run is the part of the section that visibly
       moves on it. */
    Object.assign(current, _streakOf(estimate));
    _render();
  };

  /* The difficulty caption, from `bars.js`. Called on every render and after
     every graded answer, which is why it is separate from `show`: the aim
     moves between questions and the rung does not. */
  const setDifficulty = (problem, aim) => {
    if (problem !== undefined) problemValue = Number.isFinite(problem) ? problem : null;
    if (aim !== undefined) aimValue = Number.isFinite(aim) ? aim : null;
  };

  /* An extra clause on the caption. The knowledge-graph focus flow uses it for
     the TOPIC-level BKT posterior that ends its loop, which used to be a whole
     second bar with its own thresholds — see the header. Text, deliberately:
     this screen gets one track and the ladder has it. The wording is the
     caller's job and the caller owes it a scope: a bare percentage labelled
     "this concept" is the one thing this clause must never be. */
  const setNote = (text) => {
    extraNote = text || "";
    _renderMeter();
  };

  /* No ladder context on this question — a diagnostic probe, a KC-less item,
     or the guest queue, which serves straight from the local bank and has no
     ladder at all. The whole readout goes: with no concept there is no rung,
     and a difficulty caption on its own is the tail wagging the dog. */
  const hide = () => {
    const host = _el("stage-ladder");
    if (host) host.classList.add("hidden");
    current = { kc: null, title: null, stage: null, support: undefined,
                bound: null, streak: null, streakNeeded: null, estStage: null,
                integrated: false };
    aimValue = null;
    problemValue = null;
    extraNote = "";
  };

  /* The callout's clamp is measured in PIXELS against a strip whose width is a
     percentage of the window, so a resize invalidates it — the label keeps its
     percentage position and loses its correction, which at the ends is the
     difference between "inside the strip" and "half off it". Re-placed on
     resize, once per frame: the ladder itself does not change, so this reads
     the same numbers and only moves the label. */
  let _resizeQueued = false;
  window.addEventListener("resize", () => {
    if (_resizeQueued || !current.stage) return;
    _resizeQueued = true;
    requestAnimationFrame(() => {
      _resizeQueued = false;
      if (current.stage) _renderMeter();
    });
  });

  const activeKc = () => current.kc;

  return {
    show,
    hide,
    setProgress,
    setDifficulty,
    setNote,
    activeKc,
    normalizeStage,
    STAGES,
    // Exported so `practice/watch.py` can assert the mirror of the backend's
    // promotion thresholds from outside the module.
    PROMOTE_AT,
  };
})();

window.StageLadder = StageLadder;
