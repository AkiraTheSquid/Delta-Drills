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
 * A `topic-proxy` concept's mastery is not taken as this concept's own
 * measurement. The server computes one for every KC, but for 40 of the 63 the
 * atoms behind it are coarser than the concept, so it is the topic's number
 * wearing the concept's name. Those keep falling through to the labelled
 * subtopic fallback, exactly as `kc_crosswalk_mastery.js` already decided for
 * the same reason.
 *
 * They do NOT fall all the way through to grey, though, and that was the
 * second bug this file has now met. `kcTopicReadiness` below is the floor: a
 * TOPIC-GRAIN reading, labelled as one, for any concept the server says it has
 * evidence for. Without it, a learner who had just taken the placement test saw
 * 40 of 63 concepts drawn "no estimate" while the queue held 0.02 for every one
 * of them and locked the lattice accordingly — the graph and the gate
 * disagreeing about the same learner, which is exactly what taking the server's
 * number first was supposed to prevent.
 *
 * Placement is why that case exists at all. `diagnostic.finish()` seeds a
 * per-atom posterior from the per-AREA ability estimate but creates no
 * attempts, no `subtopic_states` and no ladder rows, so every fallback that
 * counts attempts — the subtopic average, and `_extrapolated`'s
 * `learnerAbility()` — has nothing to count and returns null. The seeded
 * posteriors are real belief the engine acts on; they are just not observations
 * of any single concept. Hence: shown, and never called a measurement.
 *
 * The gate is the server's own `evidenced` flag (covered_w >= MIN_COVERED_W),
 * NOT "did placement run". A learner who has never answered anything has no
 * atom posteriors, so covered_w is 0, so this returns null and the map stays
 * honestly grey. Nothing here can resurrect the 63-identical-bubbles bug that
 * `kcLatticeNote` exists to catch.
 */
(function () {
  "use strict";

  var NOTE_ID = "kg-lattice-note";

  /* The parsed /api/practice/kc-lattice body, and the placement status.
   *
   * Cached HERE rather than in lesson-graph.js because why-graph.js — the map
   * on "Why this app exists" — borrows lesson-graph's reader but never runs its
   * build(), which is where the fetch used to live. The landing map therefore
   * asked for a reading with `lattice` permanently null and got the offline
   * answer for a signed-in learner. One cache, both surfaces.
   */
  var _lattice = null;
  var _placement = null;      // {completed, completedAt, probes} — cosmetic only
  var _latticeReq = null;
  var _placementReq = null;

  /* Both reads below are made by DEFERRED scripts (lesson-graph, why-graph),
     which run as soon as the document is parsed — and that is BEFORE
     practice/init.js has awaited DDGuest.ensure(). A signed-out visitor
     therefore had no token yet, so these two went out with no Authorization
     header at all and FastAPI's bearer dependency answered 403 (not 401 —
     401 is a bad token, 403 is no header).

     That was never only console noise. `_latticeReq` memoizes the FIRST
     attempt, so one 403 at boot pinned the knowledge graph to its offline
     client-side fallback for the rest of the page load, on exactly the
     surfaces this module exists to keep on the server's number. The learner
     saw the map disagree with the queue — the bug the comment at the top of
     this file describes, arriving by a different road.

     So wait for the session first. `DDGuest.ensure()` is memoized and is the
     same promise init.js awaits, so this joins the existing round trip rather
     than adding one; when it resolves false (backend unreachable) the fetch
     still goes out and still falls back, exactly as before. Everything is
     optional-chained: with guest-session.js absent this is the old behaviour. */
  function _sessionReady() {
    if (window.DDPracticeModeReady) return Promise.resolve();
    var ensure = window.DDGuest && window.DDGuest.ensure;
    if (typeof ensure !== "function") return Promise.resolve();
    try {
      return Promise.resolve(window.DDGuest.ensure()).catch(function () {});
    } catch (_) {
      return Promise.resolve();
    }
  }

  function _fetch(url) {
    return _sessionReady().then(function () {
      var fn = typeof window.apiFetch === "function" ? window.apiFetch : window.fetch;
      return fn(url);
    });
  }

  /* Fetch the report once and hand it to `kcLatticeNote`. Returns the lattice
   * (null for guests/offline, which is the fallback path, not an error). */
  function loadKcLattice(force) {
    if (_latticeReq && !force) return _latticeReq;
    _latticeReq = _fetch("/api/practice/kc-lattice")
      .then(function (res) {
        if (!res || !res.ok) return null;
        return res.json();
      })
      .then(function (data) { return setKcLattice(data); })
      .catch(function () { _lattice = null; return null; });
    return _latticeReq;
  }

  /* Adopt a report someone else fetched (lesson-graph refreshes it on every
   * graded attempt). Runs the degraded-join check, so the notice fires exactly
   * once per body however the body arrived. */
  function setKcLattice(data) {
    _lattice = kcLatticeNote(data);
    return _lattice;
  }

  function getKcLattice() { return _lattice; }

  /* Whether the learner has finished the placement test. Used ONLY for copy —
   * which words the map puts on screen — never to decide a number. A failure
   * leaves it null and the copy falls back to its pre-placement wording. */
  function loadPlacementStatus(force) {
    if (_placementReq && !force) return _placementReq;
    _placementReq = _fetch("/api/practice/diagnostic/status")
      .then(function (res) { return res && res.ok ? res.json() : null; })
      .then(function (d) {
        _placement = d
          ? {
              completed: !!d.completed_at,
              completedAt: d.completed_at || null,
              probes: Number.isFinite(d.probes_done) ? d.probes_done : 0,
            }
          : null;
        return _placement;
      })
      .catch(function () { _placement = null; return null; });
    return _placementReq;
  }

  function placementStatus() { return _placement; }

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

  /* The server's number for this concept at TOPIC grain — the floor under every
   * other reading, and the last thing tried before "no estimate".
   *
   * Same field, same endpoint, same `kc_graph.kc_report` as above; what differs
   * is the CLAIM. `kcLatticeReadiness` only speaks for a `measured`-tier
   * concept, where the crosswalk says the atoms behind it separate it from its
   * topic-mates, and calls that reading "atom". This one speaks for any concept
   * the server has evidence for and calls the reading "topic", because for a
   * topic-proxy the same number is shared with its siblings. Callers must
   * surface that: the dock tags it `topic-level`, the bubble stays dashed and
   * washed out, and `_isMeasured` is unaffected — nothing here can make a
   * concept look measured.
   *
   * `coveredW: 0` is deliberate and load-bearing. It means "no evidence
   * SPECIFIC to this concept", which is the honest reading of a number shared
   * across a topic, and it keeps the confidence band at its widest. Note that
   * `_evidence` in lesson-graph.js prefers the server's own `covered_w` when
   * the lattice has a row, so the band is sized by the server either way; this
   * field is what a caller without a lattice row would fall back to.
   *
   * Ordering, for whoever adds the next source: this must stay BELOW the
   * subtopic average. A subtopic number rests on the learner's own graded
   * attempts at LESSON grain, which is finer than a topic and made of
   * observations; this one can rest on nothing but a placement seed. It stays
   * ABOVE `_extrapolated`, which is a projection from the learner's overall
   * level and observes this concept's topic not at all.
   */
  function kcTopicReadiness(kc, lattice) {
    var src = lattice || _lattice;
    var row = src && src.kcs ? src.kcs[kc] : null;
    if (!row || !row.evidenced || !Number.isFinite(row.mastery)) return null;
    return {
      r: Math.max(0, Math.min(1, row.mastery)),
      source: "topic",
      coveredW: 0,
      tier: row.tier || null,
      server: true,
    };
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
  window.kcTopicReadiness = kcTopicReadiness;
  window.kcLatticeNote = kcLatticeNote;
  window.loadKcLattice = loadKcLattice;
  window.setKcLattice = setKcLattice;
  window.getKcLattice = getKcLattice;
  window.loadPlacementStatus = loadPlacementStatus;
  window.kcPlacementStatus = placementStatus;
})();
