/* The server's own reading of a concept, and a check that it exists.
 *
 * Two bugs met here and produced one symptom: a knowledge graph that answered
 * the same thing forever, however much the learner practised.
 *
 * 1. The graph computed every concept's mastery in the browser, out of
 *    `localStorage.adaptive_state_<email>`. Backend mode never writes that key
 *    — `saveAdaptiveState` is only reached by the Pyodide paths, and a
 *    signed-in learner runs on the server engine — so the read either found
 *    nothing or, worse, found a stale `adaptive_state_guest` blob from before
 *    they signed in and reported month-old numbers as current.
 * 2. The backend could not measure anything either: `kc_atom_crosswalk.json`
 *    is gitignored and was never COPY'd into the Fly image, so `kc_mastery`
 *    returned the bare prior with tier `unmapped` for all 63 concepts.
 *
 * So the browser asked a question it had no data to answer, and the server had
 * the data but its answer was a constant. Both are fixed; this module is where
 * the fix lives on the client side.
 *
 * The order is deliberate. `/api/practice/kc-lattice` is the SAME code that
 * gates practice (`kc_graph.kc_report`), so taking its number first means the
 * colour on a bubble and the decision the queue makes about that bubble cannot
 * disagree. The browser's own crosswalk read stays underneath it for guests
 * and offline, where there is no server to ask.
 *
 * What is NOT taken from the server: a `topic-proxy` concept's mastery. The
 * server computes one for every KC, but for 40 of the 63 the atoms behind it
 * are coarser than the concept, so it is the topic's number wearing the
 * concept's name. Those keep falling through to the labelled subtopic
 * fallback, exactly as `kc_crosswalk_mastery.js` already decided for the same
 * reason.
 */
(function () {
  "use strict";

  var NOTE_ID = "kg-lattice-note";

  /* This concept's own measured mastery, or null to fall through.
   *
   * `lattice` is the parsed /api/practice/kc-lattice body (null for guests),
   * `state` the freshest learner model the page holds — in backend mode the
   * in-memory /api/practice/state snapshot, offline the persisted engine
   * state. `decay` is passed in so a client-side read stays on the same
   * forgetting curve as everything else the graph draws.
   *
   * Returns the same shape `kcReadinessInfo` returns: {r, source, coveredW}.
   */
  function kcLatticeReadiness(kc, lattice, state, decay) {
    var row = lattice && lattice.kcs ? lattice.kcs[kc] : null;
    // `evidenced` is the server's own covered-weight test (MIN_COVERED_W), so
    // one lightly-weighted atom cannot stand in for a whole concept — the same
    // majority-of-weight rule the browser applies, decided once on the side
    // that has the complete picture.
    if (row && row.tier === "measured" && row.evidenced && Number.isFinite(row.mastery)) {
      return {
        r: Math.max(0, Math.min(1, row.mastery)),
        source: "atom",
        coveredW: Number.isFinite(row.covered_w) ? row.covered_w : 1,
        server: true,
      };
    }
    if (state && state.atom_mastery && typeof window.kcCrosswalkReadiness === "function") {
      var x = window.kcCrosswalkReadiness(kc, state.atom_mastery, state.atom_last_ts, decay);
      if (x) return { r: x.r, source: "atom", via: x.atoms, ts: x.ts, coveredW: x.coveredW };
    }
    return null;
  }

  /* Accept a lattice response, and say so on screen when it is hollow.
   *
   * A lattice whose every row is `unmapped` is not an outage — it answers 200,
   * it names all 63 concepts, and it reports each of them at the starting
   * prior. Drawn, that is 63 identical bubbles: indistinguishable from a
   * learner who has done nothing, which is what production looked like for a
   * month. The graph has to say that the numbers are not measurements rather
   * than paint them and let them be read as results.
   *
   * Returns the lattice (or null), so the caller can assign straight through.
   */
  function kcLatticeNote(data) {
    var lattice = data && data.kcs ? data : null;
    var ids = lattice ? Object.keys(lattice.kcs) : [];
    var mapped = 0;
    ids.forEach(function (id) {
      var row = lattice.kcs[id];
      if (row && row.tier && row.tier !== "unmapped") mapped += 1;
    });
    _renderNote(ids.length > 0 && mapped === 0);
    return lattice;
  }

  function _renderNote(degraded) {
    var host = document.getElementById("kg-cy");
    var note = document.getElementById(NOTE_ID);
    if (!degraded) {
      if (note && note.parentNode) note.parentNode.removeChild(note);
      return;
    }
    if (!host || !host.parentNode) return;
    if (!note) {
      note = document.createElement("div");
      note.id = NOTE_ID;
      note.style.cssText =
        "margin:0 0 8px;padding:8px 12px;border-radius:8px;font-size:13px;line-height:1.45;" +
        "background:rgba(227,33,44,.12);border:1px solid rgba(227,33,44,.45);color:#f4d7d9;";
      host.parentNode.insertBefore(note, host);
    }
    note.textContent =
      "The server can't measure any concept right now — its concept-to-atom " +
      "join is missing, so every bubble below is showing the starting prior " +
      "instead of your practice. Your answers are still being recorded.";
  }

  window.kcLatticeReadiness = kcLatticeReadiness;
  window.kcLatticeNote = kcLatticeNote;
})();
