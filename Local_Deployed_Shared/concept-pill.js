/* ================================================================
   CONCEPT PILL — the concept under test, in the topbar, as a bar that fills
   ================================================================

   Seth, 2026-08-27: "for the concept that's being tested it should be on the
   top bar instead of on the left. And it should have like a progress bar as it
   gets filled up for that thing. Kind of like how you have the level thing that
   gets filled up as you level up. It should essentially be a pill at the top
   that gets filled up as you make progress. And it won't have the other
   complicated thing where it has the different stages."

   So this is the level pill's twin, and deliberately so — same chip, same
   fill-from-the-left, same two stacked text layers. What it names is the
   concept the current question is about; what it fills with is how far into
   that concept the learner is.

   ── WHAT IT IS NOT ────────────────────────────────────────────────
   It is not the stage ladder moved upstairs. The ladder is one track cut into
   four rungs by chevron seams, with the rung names drawn inside it, each with
   its own ⓘ — four labelled sections a learner has to read before the bar
   means anything. That is the "complicated thing with the different stages",
   and none of it comes up here: the pill is a name and a fraction.

   The ladder itself is NOT deleted. `practice/stage-ladder.js` still runs, and
   it is where this chip's number comes from — see below. What moved is where
   the concept is READ, not where it is computed.

   ── WHERE THE NUMBER COMES FROM, and why not from here ────────────
   🔴 THIS FILE COMPUTES NOTHING. It listens for `dd-concept-progress`, which
   `practice/stage-ladder.js` fires at the end of every render, and draws the
   `pct` it is handed. That is the whole contract, and it is the contract on
   purpose: the fraction is a Wilson lower bound against a promotion threshold,
   plus a promotion streak, over four rungs, with a ceiling that is not 100 —
   arithmetic that already exists, in one place, mirrored from the backend and
   asserted by `practice/watch.py`. A second implementation up here would be a
   second answer to "how far in am I", and the ladder's own header is a long
   note about what happens when this screen has more than one of those.

   🔴 NULL IS NOT ZERO. `pct: null` means there is no reading for this concept
   — an unknown rung, a KC-less item — and the chip draws an empty track and
   says so in its tooltip rather than reporting no progress, which is a claim
   about the learner.

   🔴 THE CEILING IS 75% AND THAT IS NOT A BUG. `_overall()` in the ladder tops
   out at (rungs - 1) / rungs because arriving at the Solo rung is not the same
   as being done with the concept — `kc_is_learned` wants the BKT posterior or
   the whole question pool served, and neither number is in this payload. The
   pill inherits the ceiling untouched. Rescaling it to 100 here would fill the
   chip to the brim at the exact moment the queue is still going to serve this
   concept again.

   ── THE TWO TEXT LAYERS ───────────────────────────────────────────
   Same trap the level pill documents at length in styles/xp.css: one text
   colour cannot be legible on both a near-transparent chip and a saturated
   accent fill across three themes. So the label is drawn TWICE in one grid
   cell — the base layer in the empty track's colours, the copy on top in
   `--on-accent`, clipped to exactly the filled width. Both read the same
   `--dd-concept-pct` custom property this file writes, so the clip edge and
   the fill edge cannot drift apart.

   Which is also why this file writes the SAME STRING into both label nodes and
   never one of them: two layers with different text are two different glyph
   widths under one clip, and the seam lands in the middle of a letter that is
   not there on the other layer.
   ================================================================ */

(function () {
  "use strict";

  const _el = (id) => document.getElementById(id);

  /* Held so a repeat of the same NAME is not a repeat write. The pill is on
     screen through every graded answer and the concept changes far less often
     than the reading does.

     🔴 THERE IS NO `shownPct` BESIDE THIS, and there was one for about an hour.
     It looked like the same optimisation — skip the write when the reading has
     not moved — and it is not, because `pct` has a THIRD state: `null`, meaning
     there is no reading. `hide()` resets the cache to `null` too, so
     `pct !== shownPct` is FALSE on the first unmeasured concept after a hide,
     and the whole block is skipped: the chip keeps the PREVIOUS concept's fill
     width, the previous `aria-valuenow`, and never gets `is-unmeasured`. Caught
     in the browser: an unknown rung rendered as 33% of a concept it had nothing
     to do with, while the tooltip — written unconditionally, below — correctly
     said there was no reading. A cache whose "empty" value is also a legal
     value cannot guard anything.

     Nothing is lost by dropping it. Writing a custom property to the value it
     already holds starts no transition, so the fill does not stutter; only the
     text write needed a guard, and the text has no null state. */
  let shownTitle = null;

  /* The last fraction actually written, held for ONE decision and no other:
     animate forwards, snap backwards. It is not a write guard — see the note
     above for what happened the last time this file cached a value whose
     "empty" state was also a legal state — and `_paint` still writes the
     property unconditionally every time.

     🔴 A BAR THAT SLIDES BACKWARDS READS AS BEING PUNISHED. Measured
     2026-08-28: answering a question wrong and rating it drove the pill from
     25% to 0% as a 600ms drain, 151px sliding away to nothing right after the
     learner pressed a button — the concept looked deleted. The reading itself
     is honest (a missed drill can drop a rung, and a new concept genuinely
     starts near zero), so the number is not softened; only the SLIDE is
     dropped, and the lower value appears at once instead. `xp.js` reached the
     same conclusion for the level pill and snaps for the same reason. */
  let shownPct = null;

  /* A concept name is prose and can run to six words. The chip clips with an
     ellipsis (CSS) rather than wrapping — the topbar is 44px tall and a second
     line would push the bar's whole height — so the full name has to be
     reachable some other way, and the tooltip is it. */
  const _tooltip = (title, pct) => {
    if (pct === null) return `${title} — no reading for this concept yet.`;
    return (
      `${title} — ${Math.round(pct)}% of the way through this concept. ` +
      "The bar fills as your answers raise the tutor's estimate; it stops " +
      "short of full because reaching the unscaffolded rung is not the same " +
      "as being finished with the concept."
    );
  };

  /* The last reading the ladder published, held so the chip can be redrawn
     when the SCREEN changes underneath it without the ladder having rendered.
     See `_onScreen` — the chip has two reasons to be up, and only one of them
     arrives as an event. */
  let last = null;

  /* 🔴 IS THE QUESTION THIS CHIP NAMES ACTUALLY ON THE SCREEN?

     The ladder publishes per QUESTION and knows nothing about tabs or about
     the idle screen, and it renders once in the background at load — timer.js
     `start()` says so out loud: "nothing about the one rendered in the
     background at init is recorded". Drawn on that alone, the chip named a
     concept for a question nobody had asked for yet: Seth, 2026-08-27, "when I
     first joined the page the top bar wasn't there and then it took like a
     second later before the top bar appeared". It also stayed up across the
     Notebooks and Account tabs, naming a concept nothing on screen was about.

     The two facts below are the same two `styles/practice/timer.css` uses to
     hide the ladder card itself (`#page-practice.session-idle .stage-ladder`),
     which is why this is a read and not a second opinion:

       `.hidden`       — another tab is up; the practice page is not.
       `.session-idle` — the practice page is up, but between blocks: the
                         question split is display:none and the idle dial has
                         the screen. timer.js adds it on pause/finish and
                         removes it on start/resume; diagnostic-page.js sets it
                         from `practiceHoldsQuestion()`. A lesson page removes
                         it too (lessons.js), which is correct — a lesson is a
                         concept on screen.

     🔴 A CSS RULE CANNOT DO THIS. The chip is in the topbar, a sibling of every
     page, so no selector rooted at #page-practice reaches it. */
  const _onScreen = () => {
    const page = _el("page-practice");
    if (!page) return false;
    return (
      !page.classList.contains("hidden") && !page.classList.contains("session-idle")
    );
  };

  const _render = (detail) => {
    last = detail || null;
    _paint();
  };

  function _paint() {
    const host = _el("dd-concept");
    if (!host) return;

    const detail = last;
    const title = (detail && (detail.title || detail.kc)) || null;
    /* No concept, or no question on screen to have one: the readout goes, it
       does not go blank. An empty chip in the topbar is a control the learner
       will try to click.

       🔴 `shownTitle` IS NOT CLEARED HERE. It guards the two label writes, and
       the labels are still holding the right string while the chip is away —
       clearing it would make every tab switch rewrite two nodes for no change.
       A title that changed while the chip was down still differs from it and
       still writes. */
    if (!title || !_onScreen()) {
      host.classList.add("hidden");
      return;
    }

    const raw = detail ? detail.pct : null;
    const pct = Number.isFinite(raw) ? Math.max(0, Math.min(100, raw)) : null;

    const conceptChanged = title !== shownTitle;
    if (conceptChanged) {
      /* BOTH layers, same string, same call — see the header. */
      const base = _el("dd-concept-label");
      const on = _el("dd-concept-label-on");
      if (base) base.textContent = title;
      if (on) on.textContent = title;
      shownTitle = title;
    }

    /* Snap, don't slide, for the two moves that are not progress: a drop, and
       a different concept (whose fill has nothing to do with the one on
       screen — the label has already been swapped above, so an animated slide
       would be the OLD concept's bar draining under the NEW concept's name,
       which is what it looked like in the browser). Everything else animates.
       The class is removed again in the same call, after a forced reflow, so
       the next move — which is normally forwards — still transitions. */
    const nextPct = pct === null ? 0 : pct;
    const snap =
      conceptChanged || (shownPct !== null && nextPct < shownPct - 0.01);
    if (snap) host.classList.add("dd-concept--snap");
    shownPct = nextPct;

    /* UNCONDITIONAL — see the note on the cache above. */
    /* ONE property, read by the fill's width AND by the clip on the on-accent
       layer. Two separate writes are how a label ends up painted for a ground
       the fill has not reached. */
    host.style.setProperty("--dd-concept-pct", `${nextPct.toFixed(2)}%`);
    if (snap) {
      // The width has to LAND while the transition is off; without the reflow
      // the browser coalesces the add and the remove and the slide happens
      // anyway. Same trick, same reason, as the level-up snap in xp.js.
      void host.offsetWidth;
      host.classList.remove("dd-concept--snap");
    }
    host.classList.toggle("is-unmeasured", pct === null);
    /* `aria-valuenow` on a `progressbar` with no value is what `aria-valuetext`
       is for; a missing reading is stated, not implied by a zero. */
    if (pct === null) {
      host.removeAttribute("aria-valuenow");
      host.setAttribute("aria-valuetext", "No reading yet");
    } else {
      host.setAttribute("aria-valuenow", String(Math.round(pct)));
      host.setAttribute("aria-valuetext", `${Math.round(pct)}% of this concept`);
    }

    host.title = _tooltip(title, pct);
    host.classList.remove("hidden");
  }

  window.addEventListener("dd-concept-progress", (e) => _render(e.detail));

  /* The other half of `_onScreen`: a tab switch and a pause change nothing
     about the reading, so the ladder never fires for either, and without this
     the chip would keep whatever it had when the last question rendered.
     Cheap enough to be unconditional — one attribute filter on one element,
     and `_paint` writes only what changed. */
  if (typeof MutationObserver === "function") {
    const page = _el("page-practice");
    if (page) {
      new MutationObserver(_paint).observe(page, {
        attributes: true,
        attributeFilter: ["class"],
      });
    }
  }

  /* 🔴 NO BOOT RENDER, and nothing read off the DOM at load. The ladder fires
     this event on every one of its renders, including the first one of a
     session, so the chip's first state is the first real reading. Seeding it
     from `#question-number` at load — the obvious shortcut, since that heading
     holds the same name — would put a concept in the topbar before there is a
     question under it, on a page the learner has not started. */
})();
