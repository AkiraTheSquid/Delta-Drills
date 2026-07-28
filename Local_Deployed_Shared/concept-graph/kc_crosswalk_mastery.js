/* Read a KC's mastery out of the backend's per-ATOM BKT posteriors.
 *
 * The graph names concepts with KC ids (`numpy.dtype-astype`, from
 * kc_registry.json) and the practice backend tracks belief per atom
 * (`argmax-prediction`, from question_atom_tags.jsonl). The two id spaces are
 * disjoint — measured overlap is zero — so `atom_mastery[kc]` never hits and
 * every node fell through to the lesson-subtopic average or to a projection.
 *
 * kc_atom_crosswalk.json joins them through the shared question bank. The join
 * is many-to-few, though: 63 KCs map onto 55 atoms, and umbrella atoms like
 * `einops-rearrange` are tagged across a dozen KCs. Reading such an atom as
 * "this concept's mastery" would hand a dozen sibling KCs one identical number
 * and present a topic average as a per-node measurement. So the crosswalk
 * tiers each KC, and this module reads ONLY the `measured` tier (20 of 63).
 * Topic proxies deliberately return null and keep falling back — a coarser
 * estimate honestly labelled beats a precise-looking one that isn't.
 *
 * See docs/plan-graph-estimator-rev2.md, Slice 0.
 */
(function () {
  "use strict";

  var crosswalk = null;

  // A KC's weight is spread over several atoms and the learner will usually
  // have practised only some of them. Renormalizing over the atoms that DO
  // have a posterior is right in principle, but it degrades: with one minor
  // atom answered, renormalizing turns a 6%-weight sliver into the whole
  // measurement. Require the covered weight to be at least a simple majority
  // before calling the result this KC's mastery. Author default, not sourced.
  var MIN_COVERED_W = 0.5;

  /** Load the crosswalk once. Safe to call repeatedly; failure is non-fatal. */
  function loadKcCrosswalk(url) {
    if (crosswalk) return Promise.resolve(crosswalk);
    return fetch(url || "concept-graph/kc_atom_crosswalk.json", { cache: "no-cache" })
      .then(function (r) { return r.json(); })
      .then(function (j) { crosswalk = j && j.kcs ? j : null; return crosswalk; })
      .catch(function () { crosswalk = null; return null; });
  }

  /** True when this KC has atoms specific enough to measure it on its own. */
  function isMeasuredKc(kc) {
    var row = crosswalk && crosswalk.kcs ? crosswalk.kcs[kc] : null;
    return !!(row && row.tier === "measured");
  }

  /* Weighted mean of the atom posteriors standing in for this KC.
   *
   * `decay` is passed in rather than reimplemented so this stays on the same
   * forgetting curve as the rest of the graph: decaying per ATOM (each by its
   * own last-seen timestamp) rather than decaying the blended number is the
   * point — an old atom should contribute an old belief, not drag a fresh one
   * down through an averaged timestamp.
   *
   * Returns {r, ts, atoms, coveredW} or null when the KC is a topic proxy,
   * has no atom posteriors, or has too little of its weight covered.
   */
  function kcCrosswalkReadiness(kc, atomMastery, atomLastTs, decay) {
    if (!atomMastery || !isMeasuredKc(kc)) return null;
    var row = crosswalk.kcs[kc];
    var sumW = 0, sumWR = 0, newest = null, used = [];

    (row.atoms || []).forEach(function (pair) {
      var raw = atomMastery[pair.a];
      if (!Number.isFinite(raw)) return;
      var ts = atomLastTs ? atomLastTs[pair.a] : null;
      var r = typeof decay === "function" ? decay(raw, ts) : raw;
      if (!Number.isFinite(r)) return;
      sumW += pair.w;
      sumWR += pair.w * r;
      used.push(pair.a);
      var t = ts ? Date.parse(ts) : NaN;
      if (Number.isFinite(t) && (newest === null || t > newest)) newest = t;
    });

    if (!used.length || sumW < MIN_COVERED_W) return null;
    return {
      r: Math.max(0, Math.min(1, sumWR / sumW)),
      ts: newest === null ? null : new Date(newest).toISOString(),
      atoms: used,
      coveredW: sumW,
    };
  }

  /** For the dock panel: how many sibling KCs lean on the same evidence. */
  function kcCrosswalkInfo(kc) {
    var row = crosswalk && crosswalk.kcs ? crosswalk.kcs[kc] : null;
    if (!row) return null;
    return { tier: row.tier, reliability: row.reliability, sharedWith: row.shared_with };
  }

  window.loadKcCrosswalk = loadKcCrosswalk;
  window.isMeasuredKc = isMeasuredKc;
  window.kcCrosswalkReadiness = kcCrosswalkReadiness;
  window.kcCrosswalkInfo = kcCrosswalkInfo;
})();
