/* ================================================================
   MASTERY-BAR.JS — the mastery bar's markup, in one place.

   WHAT THIS IS
     fill = the estimate, shaded band = its confidence interval, ticks = the
     thresholds the engine acts on. It was written inside `lesson-graph.js` as
     a private `_masteryBar`, which was fine while the Knowledge Graph dock was
     the only thing drawing one. The Course content tab draws the same bar per
     section, so the markup moved here rather than being typed twice.

     Only the MARKUP moved. The two other halves of this bar were already
     shared and are untouched:
       * the interval maths — `kc_interval.js` (`window.DeltaKcInterval`)
       * the styling        — `.kg2-dock-*` in `styles/how-it-works.css`
     The class names are deliberately still `kg2-dock-*`. They are what the
     stylesheet keys on, and renaming them to something surface-neutral would
     be a rename of every rule for no behaviour change.

   MEASURED vs INFERRED
     `measured: false` means no graded attempt has landed on this thing, so the
     interval is nearly the whole scale. It is drawn in a flatter hatch with no
     end caps, because a nearly-full-width stripe drawn like a measurement
     reads as a very confident wide one — the opposite of what it means.

   GATES ARE OPT-IN
     `gates: true` draws the 85%/95% ticks and the scale row beneath. They are
     the thresholds for ONE concept, so they belong on the Knowledge Graph dock
     and NOT on an aggregate: "85% unlocks" under a bar averaged over a whole
     ARENA section names a gate that section has no such thing as. Callers
     drawing an aggregate leave `gates` off and get a bare track.
   ================================================================ */

(function (root, factory) {
  const api = factory();
  if (root) root.DeltaMasteryBar = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : null, function () {
  "use strict";

  const UNLOCK_T = 0.85;
  const MASTERY_T = 0.95;
  const UNKNOWN_COLOR = "#6b7280";

  const clamp01 = (n) => Math.max(0, Math.min(1, n));

  /** The red→blue ramp the graph nodes use, so a bar and its node agree. */
  function masteryColor(r, unknownColor) {
    if (!Number.isFinite(r)) return unknownColor || UNKNOWN_COLOR;
    const t = clamp01(r);
    const lo = [214, 72, 72], hi = [59, 130, 246];  // #d64848 → #3b82f6
    const c = lo.map((v, i) => Math.round(v + (hi[i] - v) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  /**
   * Build one bar.
   *
   * value    0–1 estimate, or non-finite for "not yet estimated" (grey, no fill)
   * ci       [lo, hi] in 0–1, or null/undefined to draw no band
   * measured false ⇒ the band is inferred, not observed (see header)
   * gates    draw the 85%/95% ticks + scale row (single concept only)
   *
   * Returns an HTML string. The band keeps a 0.5% floor so a very tight
   * interval stays visible rather than collapsing to nothing.
   */
  function render(opts) {
    const o = opts || {};
    const r = o.value;
    const w = Number.isFinite(r) ? clamp01(r) * 100 : 0;
    const ci = Array.isArray(o.ci) && o.ci.length === 2 ? o.ci : null;

    const band = ci
      ? `<span class="kg2-dock-ci${o.measured ? "" : " is-inferred"}" ` +
        `style="left:${clamp01(ci[0]) * 100}%;width:${Math.max(0.5, (clamp01(ci[1]) - clamp01(ci[0])) * 100)}%"></span>`
      : "";

    const ticks = o.gates
      ? `<span class="kg2-dock-gate" style="left:${UNLOCK_T * 100}%"></span>` +
        `<span class="kg2-dock-gate is-mastery" style="left:${MASTERY_T * 100}%"></span>`
      : "";

    const scale = o.gates
      ? `<div class="kg2-dock-scale"><span>0%</span><span>85% unlocks</span><span>95% mastered</span></div>`
      : "";

    return (
      `<div class="kg2-dock-track">` +
        `<div class="kg2-dock-fill" style="width:${w}%;background:${masteryColor(r, o.unknownColor)}"></div>` +
        band +
        ticks +
      `</div>` +
      scale
    );
  }

  return { render, masteryColor, UNLOCK_T, MASTERY_T };
});
