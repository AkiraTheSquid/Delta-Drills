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

  const _render = (detail) => {
    const host = _el("dd-concept");
    if (!host) return;

    const title = (detail && (detail.title || detail.kc)) || null;
    /* No concept on screen: the readout goes, it does not go blank. An empty
       chip in the topbar is a control the learner will try to click. */
    if (!title) {
      host.classList.add("hidden");
      shownTitle = null;
      return;
    }

    const raw = detail ? detail.pct : null;
    const pct = Number.isFinite(raw) ? Math.max(0, Math.min(100, raw)) : null;

    if (title !== shownTitle) {
      /* BOTH layers, same string, same call — see the header. */
      const base = _el("dd-concept-label");
      const on = _el("dd-concept-label-on");
      if (base) base.textContent = title;
      if (on) on.textContent = title;
      shownTitle = title;
    }

    /* UNCONDITIONAL — see the note on the cache above. */
    /* ONE property, read by the fill's width AND by the clip on the on-accent
       layer. Two separate writes are how a label ends up painted for a ground
       the fill has not reached. */
    host.style.setProperty("--dd-concept-pct", `${(pct === null ? 0 : pct).toFixed(2)}%`);
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
  };

  window.addEventListener("dd-concept-progress", (e) => _render(e.detail));

  /* 🔴 NO BOOT RENDER, and nothing read off the DOM at load. The ladder fires
     this event on every one of its renders, including the first one of a
     session, so the chip's first state is the first real reading. Seeding it
     from `#question-number` at load — the obvious shortcut, since that heading
     holds the same name — would put a concept in the topbar before there is a
     question under it, on a page the learner has not started. */
})();
