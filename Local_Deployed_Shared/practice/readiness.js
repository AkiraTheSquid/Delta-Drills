/* ================================================================
   READINESS — one number for "how far through the map am I?"

   WHAT IT IS
     The share of the 63 concepts in lessons/kc_registry.json that the
     learner has taken past the engine's MASTERY threshold (0.95 — the
     same constant the ARENA unlock and the concept graph act on). That
     is completion of the graph, which is what the pause screen says:
     "N% ready for the ARENA curriculum".

   🔴 WHY IT DOES NOT USE THE CONCEPT GRAPH'S OWN COUNT
     concept-graph/lesson-graph.js holds `kcById` and could answer this
     directly — but it only populates it when the Knowledge Graph tab
     BUILDS, which on the Practice tab never happens. Reading it there
     returns 0 of 0 and renders a confident "100%" or a NaN. So the KC
     LIST is read from the registry here (11 KB, fetched once per page)
     and only the per-concept READING is delegated, to
     `window.deltaKcReadinessInfo` — that function is graph-independent:
     it goes to `computeAtomReadiness`, then the persisted learner state,
     then the lattice, then the subtopic estimate, in that order.

   BORROWED IS NOT MASTERED
     `mastered` counts only concepts whose reading is the learner's own
     (`source: "atom"`). A subtopic average or an extrapolation can sit
     above the threshold without this concept ever having been attempted;
     those come back as `borrowed` and are never in the percentage.

   NOT MEASURED IS NOT ZERO
     A learner who has answered nothing has no evidence, not evidence of
     nothing, and `pct` for them is 0 either way — but `measured` says
     which of those it is, so the caller can print "not started yet"
     instead of a 0% that reads like a grade.
   ================================================================ */

window.PracticeReadiness = (() => {
  /* 0.95 = mastered. Mirrors concept-graph/lesson-graph.js's MASTERY_T and
     the engine behind it; the 0.85 unlock gate is a different question
     (may I attempt what comes after this) and is not what completion means. */
  const MASTERY_T = 0.95;
  const REGISTRY_URL = "lessons/kc_registry.json";

  let kcIds = null;
  let inFlight = null;

  const _loadKcIds = () => {
    if (kcIds) return Promise.resolve(kcIds);
    if (inFlight) return inFlight;
    inFlight = fetch(REGISTRY_URL, { cache: "no-cache" })
      .then((r) => (r.ok ? r.json() : null))
      .then((reg) => {
        const rows = Array.isArray(reg?.kcs) ? reg.kcs : [];
        kcIds = rows.map((k) => k?.id).filter(Boolean);
        return kcIds;
      })
      .catch(() => {
        /* No list, no number. Left NULL rather than cached as [] so a
           connection that comes back is retried instead of answering
           "0 of 0" for the rest of the page's life. */
        return null;
      })
      .finally(() => {
        inFlight = null;
      });
    return inFlight;
  };

  /* {r, source} or null. The SOURCE comes back with the number because
     lesson-graph.js's own contract says it must: "a subtopic number is NOT a
     per-concept measurement, and an extrapolated one is not a measurement at
     all. Presenting either as one overclaims." Every path that is real
     per-concept evidence — in-memory posterior, persisted atom mastery, the
     server lattice's `measured`+`evidenced` tier, the browser crosswalk —
     labels itself "atom". Everything else is borrowed from the lesson or
     guessed. */
  const _readOne = (kc) => {
    try {
      const info = window.deltaKcReadinessInfo?.(kc);
      const r = info?.r;
      if (!Number.isFinite(r)) return null;
      return { r, source: info?.source || "none" };
    } catch (_) {
      return null;
    }
  };

  /* {pct, mastered, total, measured} — or null when the registry could not
     be read at all, which the caller must render as "unknown", never as 0%. */
  const read = async () => {
    /* 🔴 THE READER MAY NOT EXIST YET. lesson-graph.js — which installs
       `window.deltaKcReadinessInfo` — is a DEFER script, so it does not run
       until the parser is finished, while this file is a plain classic script
       that runs the moment it is reached. A read that happens in between finds
       no reader, scores every concept as unreadable, and paints a confident
       0%. Unknown, not zero: the caller renders "—" and session-idle.js paints
       again on DOMContentLoaded, which fires AFTER every defer script. */
    if (typeof window.deltaKcReadinessInfo !== "function") return null;
    const ids = await _loadKcIds();
    if (!ids || !ids.length) return null;
    let mastered = 0;
    let measured = 0;
    let borrowed = 0;
    for (const kc of ids) {
      const hit = _readOne(kc);
      if (!hit) continue;
      const { r, source } = hit;
      /* Any real reading above the floor counts as evidence the learner has
         touched the map. `> 0` and not `>= 0`: the fallbacks return 0 for a
         concept with no attempts, and treating that as measured would tell
         someone who has answered nothing that they have started. */
      if (r > 0) measured += 1;
      if (r < MASTERY_T) continue;
      /* 🔴 MASTERED IS ATOM-ONLY. A subtopic rate of 0.96 says the LESSON went
         well, not that this concept was ever attempted, and an extrapolated
         number was never a measurement — counting either would let a headline
         "ready for ARENA" percentage be built out of readings that are not
         about these concepts. Counted separately instead, so the caller can
         say how many are only nearly-known. */
      if (source === "atom") mastered += 1;
      else borrowed += 1;
    }
    const total = ids.length;
    return {
      pct: total ? Math.round((mastered / total) * 100) : 0,
      mastered,
      borrowed,
      total,
      measured: measured > 0,
    };
  };

  return { read, MASTERY_T };
})();
