/* ================================================================
   READINESS — THE ONE "% ready for the ARENA curriculum" IN THE APP

   🔴 WHY THIS FILE IS THE ONLY PLACE THE NUMBER IS COMPUTED
     Two surfaces said "ready for ARENA" and meant two different things
     (Seth, 2026-08-23: "the placement test display of how ready you are
     for ARENA ... is different from practice thing before you enter the
     practice ... it needs to be the same"):

       Practice idle dial   share of the 63 concepts at r >= 0.95, and
                            only from the learner's OWN evidence
       Placement results    mean of (theta - 20) / 80 over the NINE areas
                            the test scores

     They cannot agree, and not by a little. `diagnostic.py` seeds a
     finished placement at SEED_MASTERY_CAP = 0.92, which is BELOW the
     0.95 mastery threshold — so a learner walked off a placement that
     had just told them "45% ready" onto a Practice tab reading 0%. The
     number nobody could act on was the one the app repeated twice.

     Both now call `read()`. practice/placement-results.js renders it
     beside what the test itself measured; practice/session-idle.js paints
     it in the dial. There is one definition and one reader.

   WHAT THE NUMBER IS
     The learner's MEAN readiness across the concepts in
     lessons/kc_registry.json — the model's own P(known) per concept,
     averaged over the map ARENA needs. "How much of the map do you know",
     which is the question both screens were asking.

     It is NOT the count of mastered concepts any more. That count is a
     completion tally: it sits at 0 through everything a new learner does
     (a placement cannot reach 0.95, and neither can two good sessions),
     so it answered "have you finished" on a screen asking "where are
     you". It is still reported — `mastered` — as the detail line, which
     is where a tally belongs.

   WHERE EACH CONCEPT'S NUMBER COMES FROM
     `window.deltaKcReadinessInfo` (concept-graph/lesson-graph.js), the
     same ladder the knowledge graph and the landing map read:
     in-memory posterior → persisted atom mastery → the server lattice →
     the subtopic average → an extrapolation. Every reading carries its
     SOURCE, and the sources are not equal: `atom` is the learner's own
     evidence about that concept, everything else is borrowed from the
     lesson, the topic or the learner's overall level. The mean uses every
     reading the model has — that is what makes it move on the day of a
     placement — and `own` says how much of it is the learner's own, so a
     caller can say which it is instead of implying all of it is measured.

   🔴 THE LATTICE HAS TO BE FETCHED HERE
     `kcReadinessInfo` reads the server's per-concept report out of
     lesson-graph.js's own `lattice` variable, which is filled by its
     `build()` — i.e. only when the Knowledge Graph tab is opened. On
     Practice and on the placement page it was permanently null, so every
     server-side reading (including everything a placement seeds) fell
     through to the offline path and the dial under-reported. `read()`
     refreshes it through `window.deltaRefreshKcLattice`, which is the
     sanctioned door: calling `loadKcLattice` directly fills the shared
     cache while lesson-graph's own `lattice` stays null, and that is the
     variable the `atom`-tier read is passed.

   NOT MEASURED IS NOT ZERO
     A learner who has answered nothing has no evidence, not evidence of
     nothing. There is no reading to average, so `read()` reports
     `known: 0` and the callers print "not started yet" rather than a 0%
     that reads like a grade. A registry or a reader that cannot be
     reached at all comes back NULL, which is "unknown", never 0%.
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

  /* 🔴 THE READER MAY NOT EXIST YET. lesson-graph.js — which installs
     `window.deltaKcReadinessInfo` — is a DEFER script, so it does not run
     until the parser is finished, while this file is a plain classic script
     that runs the moment it is reached. A read that happens in between finds
     no reader and would score every concept as unreadable.

     So the first read WAITS for DOMContentLoaded, which is the first moment
     every defer script is guaranteed to have run, instead of answering
     "unknown" to a caller that painted a fraction of a second too early.
     After that the answer is immediate either way. */
  const _reader = () =>
    typeof window.deltaKcReadinessInfo === "function"
      ? Promise.resolve(true)
      : document.readyState === "loading"
        ? new Promise((resolve) => {
            document.addEventListener(
              "DOMContentLoaded",
              () => resolve(typeof window.deltaKcReadinessInfo === "function"),
              { once: true },
            );
          })
        : Promise.resolve(false);

  /* The server's per-concept report, fetched at most once per STAMP.

     The stamp is the caller's answer to "has the model moved on the server
     since the last read" — placement-results.js passes the placement's
     `completed_at`, so finishing a test forces one fresh fetch and every
     later render reuses it. Without that the results card would read a
     lattice fetched before the seeding ran and report the learner's old
     level on the screen announcing their new one.

     Two requests per stamp (the lattice and the placement status), not two
     per paint: the idle dial repaints on every pause and this must not turn
     into a poll. A failure resolves to null — the reader falls back to the
     offline ladder, which is a worse number, not a broken one. */
  /* 🔴 A SET OF STAMPS SEEN, not the LAST stamp. The two callers alternate —
     the results card renders on every `delta:practice-state-changed` with the
     placement's `completed_at`, the idle dial repaints on every pause with no
     stamp at all — so "has the stamp changed since last time" is true on every
     single call between them, and the page would refetch the lattice forever,
     ping-ponging between two keys. A stamp already fetched is already covered.

     The retry exists for the other failure: `deltaRefreshKcLattice` resolves
     whether or not the report arrived (`refreshLattice` swallows its own
     errors and leaves the lattice null), so a page loaded while the backend
     was down would otherwise hold that empty answer until a reload. Throttled,
     because a guest legitimately has no lattice and must not turn a repaint
     into a poll. */
  const RETRY_MS = 60000;
  const fetched = new Set();
  let latticeReq = null;
  let latticeAt = 0;
  const _latticeLoaded = () => {
    try {
      return !!window.getKcLattice?.();
    } catch (_) {
      return false;
    }
  };
  const _ensureLattice = (stamp) => {
    const key = stamp == null ? "" : String(stamp);
    if (typeof window.deltaRefreshKcLattice !== "function") return Promise.resolve(null);
    const covered = latticeReq && fetched.has(key);
    const retry = covered && !_latticeLoaded() && Date.now() - latticeAt > RETRY_MS;
    if (covered && !retry) return latticeReq;
    fetched.add(key);
    latticeAt = Date.now();
    latticeReq = window.deltaRefreshKcLattice().catch(() => null);
    return latticeReq;
  };

  /* {r, source} or null. The SOURCE comes back with the number because
     lesson-graph.js's own contract says it must: "a subtopic number is NOT a
     per-concept measurement, and an extrapolated one is not a measurement at
     all. Presenting either as one overclaims." Every path that is real
     per-concept evidence — in-memory posterior, persisted atom mastery, the
     server lattice's `measured`+`evidenced` tier, the browser crosswalk —
     labels itself "atom". Everything else is borrowed from the lesson, the
     topic (which is what a placement seeds) or the learner's overall level. */
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

  /* {pct, known, own, mastered, total} — or null when there is no reader or
     no registry at all, which the caller must render as "unknown", never 0%.

     `opts.stamp` — see `_ensureLattice`. Pass the placement's completed_at
     from a results screen; leave it off everywhere else. */
  const read = async (opts) => {
    if (!(await _reader())) return null;
    await _ensureLattice(opts && opts.stamp);
    const ids = await _loadKcIds();
    if (!ids || !ids.length) return null;

    let sum = 0;
    let known = 0;
    let own = 0;
    let mastered = 0;
    for (const kc of ids) {
      const hit = _readOne(kc);
      if (!hit) continue;
      const { r, source } = hit;
      sum += Math.max(0, Math.min(1, r));
      known += 1;
      /* `own` is the learner's own evidence about THIS concept. A placement
         seed, a lesson average and an extrapolation are all real numbers the
         queue acts on — they belong in the mean — but none of them is a
         measurement of this concept, and a caller that presents the whole
         figure as measured overclaims on the learner's behalf. */
      if (source !== "atom") continue;
      own += 1;
      if (r >= MASTERY_T) mastered += 1;
    }

    const total = ids.length;
    return {
      /* The mean over the WHOLE map, not over the concepts that happen to
         have a reading: a learner with one strong concept and 62 blanks is
         not 90% ready, and averaging only what is known says they are. An
         unread concept contributes 0, which is what "no evidence" is worth
         to a readiness figure. */
      pct: total ? Math.round((sum / total) * 100) : 0,
      known,
      own,
      mastered,
      total,
    };
  };

  /* The one line that goes UNDER the figure, wherever the figure is drawn.

     A bare percentage with no denominator reads as a grade, and the two
     screens describing the same number in two different vocabularies is the
     other half of the bug this file exists to close. So the words live here
     too: how much of the figure is the learner's own evidence, and the
     mastered tally after it — a completion count, in the place a completion
     count belongs. Nothing measured says so in words rather than printing a
     0 that reads as a mark. */
  const detail = (info) => {
    if (!info) return "";
    if (!info.known) return `${info.total} concepts to go — none measured yet`;
    /* The state a learner is in the moment they finish the placement: every
       concept has a number and not one of them was measured on its own.
       "0 of 63 concepts measured directly" is true and reads as a failure
       report on the test they just took; this says the same thing forwards. */
    if (!info.own) return `estimated across ${info.total} concepts — none measured directly yet`;
    const own = `${info.own} of ${info.total} concepts measured directly`;
    return info.mastered ? `${own} · ${info.mastered} mastered` : own;
  };

  return { read, detail, MASTERY_T };
})();
