/* ================================================================
   PER-KC CONFIDENCE INTERVAL — how wide the mastery band should be.

   THE BUG THIS EXISTS TO FIX
   --------------------------
   The graph used to draw the band from `subtopic_states[key].n`: the number of
   graded attempts anywhere in the *lesson*. A subtopic holds many KCs, so a
   concept nobody had ever been tested on inherited its lesson-mates' attempt
   count and drew a band as tight as a concept with real evidence behind it.
   Zero evidence is the WIDEST state on the map, not the tightest.

   WHAT REPLACES IT
   ----------------
   An effective sample size that is per-concept:

       n_eff = n_direct + n_prior

   `n_direct` — the share of graded attempts that actually bear on THIS concept.
   Nothing anywhere records a literal per-KC attempt count, so it is attributed
   from the two things that ARE measured per concept:

       n_direct = n_subtopic x covered_w x specificity

     * `n_subtopic` — attempts in the lesson. The pool being attributed.
     * `covered_w`  — fraction of this KC's crosswalk atom weight that has ANY
                      observation behind it (backend `kc_graph.kc_mastery`, sent
                      on /api/practice/kc-lattice). This is the term that kills
                      the inheritance bug: a KC none of whose atoms were ever
                      touched scores 0 no matter how much its lesson-mates
                      practised, so it cannot borrow their band.
     * `specificity` — how much of that evidence is about this concept rather
                      than its siblings. The crosswalk publishes exactly this as
                      `reliability` (= sum of share*spec; `shared_with` =
                      1/reliability - 1 is documented as "effective sibling KCs
                      drawing on the same evidence"). With no crosswalk row, fall
                      back to 1/siblings-in-the-subtopic, which the dock already
                      calls the honest denominator.

   `n_prior` — what a structural prior (subtopic inheritance, or the difficulty-
   shifted extrapolation from the learner's overall level) is worth in units of
   graded attempts. Derived, not picked:

     A predictor correlating with the truth at rho explains rho^2 of its
     variance, so it cuts the estimate's variance to (1 - rho^2) of the
     no-information variance. For a Bernoulli quantity, an estimate resting on
     n0 pseudo-observations on top of a uniform prior has variance
     p(1-p)/(n0 + 1) — n0 = 0 recovering the ignorance variance p(1-p). Setting

         p(1-p)(1 - rho^2) = p(1-p)/(n0 + 1)   =>   n0 = rho^2 / (1 - rho^2)

     Seth's figure for how well this class of structural prior tracks actual
     student performance is rho ~ 0.10-0.15. Taking the GENEROUS end, rho = 0.15
     gives n0 = 0.0230 — about one fortieth of a single graded attempt. That is
     the point: the prior is worth keeping (it beats a blank map) but it must
     never buy a band comparable to observation, and it is swamped the moment one
     real attempt lands.

   INVARIANT (enforced by `kcInterval`, asserted by the node harness)
   ------------------------------------------------------------------
   A KC with less than one attempt's worth of direct evidence never draws a band
   narrower than a KC with exactly one real graded attempt. The algebra already
   guarantees it for any n_prior < 0.47, but it is clamped explicitly rather than
   left to depend on a constant somebody may retune.

   Loads in the browser as `window.DeltaKcInterval`; also `module.exports` so the
   test harness can run it under node without a DOM.
   ================================================================ */
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.DeltaKcInterval = api;
})(typeof window !== "undefined" ? window : null, function () {
  "use strict";

  var Z95 = 1.96;

  // How well a structural prior (inheritance / extrapolation) tracks real
  // performance. Top of Seth's stated 0.10-0.15 range, i.e. the reading most
  // favourable to the prior — the conclusion holds a fortiori at 0.10.
  var PRIOR_R = 0.15;

  // rho^2 / (1 - rho^2). See the header derivation. ~= 0.0230 attempts.
  var PRIOR_N = (PRIOR_R * PRIOR_R) / (1 - PRIOR_R * PRIOR_R);

  /* 95% Wilson score interval. Wilson rather than the normal approximation
     because n is small and p sits near the edges, where the normal interval runs
     off [0,1] and understates uncertainty. n may be fractional here: it is an
     effective sample size, not a count of rows. */
  function wilsonParts(p, n, z) {
    z = Number.isFinite(z) ? z : Z95;
    if (!Number.isFinite(p) || !Number.isFinite(n) || n <= 0) return null;
    var zz = z * z;
    var d = 1 + zz / n;
    return {
      centre: (p + zz / (2 * n)) / d,
      half: (z * Math.sqrt((p * (1 - p)) / n + zz / (4 * n * n))) / d,
    };
  }

  function wilson(p, n, z) {
    var w = wilsonParts(p, n, z);
    if (!w) return null;
    return [Math.max(0, w.centre - w.half), Math.min(1, w.centre + w.half)];
  }

  function wilsonHalf(p, n, z) {
    var w = wilsonParts(p, n, z);
    return w ? w.half : NaN;
  }

  // The band one real graded attempt earns, at its widest (p = 0.5): ~0.445.
  // This is the floor for anything with less evidence than that.
  var ONE_ATTEMPT_HALF = wilsonHalf(0.5, 1);

  /* Attempts bearing on THIS concept. See the header for each term.
     `ev` = {nSub, coveredW, reliability, siblings}. */
  function directEvidenceN(ev) {
    ev = ev || {};
    var coveredW = Number.isFinite(ev.coveredW) ? Math.max(0, Math.min(1, ev.coveredW)) : 0;
    if (coveredW <= 0) return 0;

    var nSub = Number.isFinite(ev.nSub) ? Math.max(0, ev.nSub) : 0;
    // covered_w > 0 means the server has at least one observation behind this
    // KC's atoms. If the subtopic pool is missing (a state shape that carries
    // atom posteriors but no per-subtopic counter) claim the minimum that is
    // consistent with that — one attempt, before the discounts — rather than
    // reporting zero evidence for a learner who plainly has some.
    if (nSub <= 0) nSub = 1;

    var spec;
    if (Number.isFinite(ev.reliability) && ev.reliability > 0) {
      spec = Math.min(1, ev.reliability);
    } else if (Number.isFinite(ev.siblings) && ev.siblings > 0) {
      spec = 1 / ev.siblings;
    } else {
      spec = 1;
    }
    return nSub * coveredW * spec;
  }

  /* The band for one KC.
     `o` = {r, nDirect, spreadHalf?}
       r          - the point estimate being drawn (any source).
       nDirect    - from directEvidenceN(); 0 when nothing graded touched this KC.
       spreadHalf - optional extra widening: for an extrapolated estimate, how
                    varied the learner's own results are. Only ever widens.
     Returns {ci:[lo,hi], half, nEff, nDirect, measured, floored}. */
  function kcInterval(o) {
    o = o || {};
    var p = Number.isFinite(o.r) ? Math.max(0, Math.min(1, o.r)) : 0.5;
    var nDirect = Number.isFinite(o.nDirect) ? Math.max(0, o.nDirect) : 0;
    var nEff = nDirect + PRIOR_N;

    var w = wilsonParts(p, nEff);
    var centre = w ? w.centre : 0.5;
    var half = w ? w.half : 0.5;

    if (Number.isFinite(o.spreadHalf)) half = Math.max(half, o.spreadHalf);

    // The invariant. Anything short of one real attempt's worth of direct
    // evidence is at least as uncertain as one real attempt at its worst.
    var measured = nDirect >= 1;
    var floored = false;
    if (!measured && half < ONE_ATTEMPT_HALF) { half = ONE_ATTEMPT_HALF; floored = true; }

    return {
      ci: [Math.max(0, centre - half), Math.min(1, centre + half)],
      half: half,
      nEff: nEff,
      nDirect: nDirect,
      measured: measured,
      floored: floored,
    };
  }

  return {
    Z95: Z95,
    PRIOR_R: PRIOR_R,
    PRIOR_N: PRIOR_N,
    ONE_ATTEMPT_HALF: ONE_ATTEMPT_HALF,
    wilson: wilson,
    wilsonHalf: wilsonHalf,
    directEvidenceN: directEvidenceN,
    kcInterval: kcInterval,
  };
});
