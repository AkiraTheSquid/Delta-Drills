// Atom-level readiness — BKT-only. EWMA fully removed.
//
// Readiness for an atom IS its per-atom Bayesian Knowledge Tracing posterior
// P(known), decay-adjusted to now. The posterior is maintained server-side
// (bkt_mastery.py) from graded attempts on the atom + encompassing FIRe credit,
// and arrives in `adaptiveStateJson.atom_mastery` / `.atom_last_ts` via
// /api/practice/state. There is no per-subtopic EWMA bridge anymore: an atom
// with no posterior yet returns the caller's fallback (un-practiced), and bank
// questions — now atom-tagged — are what raise these posteriors.
//
// Returns a number in [0, 1] (same convention as before).

(function () {
  "use strict";

  // Mirror of backend bkt_mastery.py constants (decay target + half-life).
  const BKT_P_INIT = 0.10;
  const BKT_HALF_LIFE_DAYS = 14.0;

  const _readAdaptiveState = () => {
    if (typeof adaptiveStateJson !== "string" || !adaptiveStateJson) return null;
    try { return JSON.parse(adaptiveStateJson); } catch (_) { return null; }
  };

  // Decay a posterior toward p_init by elapsed-time half-life (forgetting).
  // Mirrors bkt_mastery.decay() so a read between attempts agrees with the
  // server's next write.
  const _decayBkt = (L, lastTs) => {
    if (!Number.isFinite(L)) return null;
    if (!lastTs) return L;
    const prev = Date.parse(lastTs);
    if (!Number.isFinite(prev)) return L;
    const elapsedDays = Math.max(0, (Date.now() - prev) / 86400000);
    const factor = Math.pow(0.5, elapsedDays / BKT_HALF_LIFE_DAYS);
    return BKT_P_INIT + (L - BKT_P_INIT) * factor;
  };

  // Decayed BKT posterior for an atom, or null if none recorded.
  const _bktReadiness = (state, atomId) => {
    const m = state && state.atom_mastery;
    if (!m || typeof m !== "object") return null;
    const raw = Number(m[atomId]);
    if (!Number.isFinite(raw)) return null;
    const ts = (state.atom_last_ts && state.atom_last_ts[atomId]) || null;
    const v = _decayBkt(raw, ts);
    return v === null ? null : Math.max(0, Math.min(1, v));
  };

  /**
   * computeAtomReadiness(atomId, fallback = 0)
   * Returns the learner's BKT mastery P(known) for the atom in [0,1], or
   * `fallback` if the atom has no posterior yet (un-practiced).
   */
  window.computeAtomReadiness = (atomId, fallback) => {
    const fb = Number.isFinite(Number(fallback)) ? Number(fallback) : 0;
    if (typeof atomId !== "string" || !atomId) return fb;
    const state = _readAdaptiveState();
    const bkt = _bktReadiness(state, atomId);
    return bkt === null ? fb : bkt;
  };

  /**
   * computeAtomReadinessBatch(atomIds, fallback = 0) → { atomId: readiness }.
   */
  window.computeAtomReadinessBatch = (atomIds, fallback) => {
    const out = {};
    if (!Array.isArray(atomIds)) return out;
    atomIds.forEach((id) => {
      out[id] = window.computeAtomReadiness(id, fallback);
    });
    return out;
  };

  /**
   * Diagnostic — `window._atomReadinessDiagnostic()`.
   * Counts atoms with a BKT posterior vs none, over the loaded concept graph.
   */
  window._atomReadinessDiagnostic = () => {
    const g = window.CONCEPT_GRAPH_V5_V2;
    const state = _readAdaptiveState();
    const m = (state && state.atom_mastery) || {};
    const withPosterior = Object.keys(m).length;
    if (!g || !Array.isArray(g.concepts)) {
      return { with_posterior: withPosterior, note: "CONCEPT_GRAPH_V5_V2 not loaded" };
    }
    let inGraph = 0;
    const ids = new Set(g.concepts.map((c) => c && c.id).filter(Boolean));
    Object.keys(m).forEach((id) => { if (ids.has(id)) inGraph += 1; });
    return {
      total_atoms: g.concepts.length,
      with_posterior: withPosterior,
      with_posterior_in_graph: inGraph,
      no_posterior: g.concepts.length - inGraph,
    };
  };
})();
