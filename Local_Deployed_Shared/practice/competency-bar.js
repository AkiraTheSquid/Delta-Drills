/* ================================================================
   CONCEPT MASTERY — knowledge-graph focus flow, headless

   Lives in the ?lesson=<kc>&embed=1 iframe. Tracks the learner's BKT
   posterior for THIS concept's subtopic and reports the mastery crossing back
   to the parent graph.

   It used to DRAW that posterior: a full-width track with two gate marks
   (0.85 unlock, 0.95 mastery), a phase title, and a tween. On the focus screen
   that made three progress readouts at once — the concept strip, the target
   difficulty bar, and this — for three quantities a learner has no reason to
   know are different. The stage ladder is the one bar now, so the DOM here is
   gone and the number is a clause of its caption.

   What is NOT gone, and must not be:
     `_emitGateCrossed` dispatches `competency:gate-crossed` and posts
     `delta:kc-mastered` to the parent. `concept-graph/lesson-graph.js` listens
     for that message to close the overlay and recolour the node. Without it a
     mastered concept stays open and grey and the learner is stuck in the loop.

   The two thresholds are real engine constants, still exported:
     UNLOCK_THRESHOLD  0.85 — prereq cleared, dependents unlock
     MASTERY_THRESHOLD 0.95 — concept mastered, leave the loop
   ================================================================ */

const CompetencyBar = (() => {
  const UNLOCK_THRESHOLD = 0.85;
  const MASTERY_THRESHOLD = 0.95;

  // Backend and local mode name the same subtopic differently ("Numpy: Core
  // array literacy" vs "Core array literacy"), and BKT is keyed by whatever
  // question.subtopic says — so hold both and match on either.
  let targetSubtopics = [];
  let currentMastery = null;
  let gateAlreadyCrossed = false;
  let bound = false;

  const _clamp01 = (v) => Math.max(0, Math.min(1, v));

  // Backend mode keeps the subtopic posterior on the served question
  // (p_current) and on every feedback response; local/supabase mode keeps it
  // in the adaptive state blob. Try the live snapshot first, then the blob.
  const _readMastery = () => {
    for (const key of targetSubtopics) {
      const live = window.__subtopicMastery && window.__subtopicMastery[key];
      if (Number.isFinite(live)) return live;
    }
    for (const key of targetSubtopics) {
      try {
        const p = getEwmaFromAdaptiveState(key);
        if (Number.isFinite(p)) return p;
      } catch (_err) { /* try the next key */ }
    }
    return null;
  };

  /* The caption clause. Named for what the number gates rather than shown as a
     bare percentage: "62%" beside a difficulty reading was most of why the old
     screen read as several competing scores.

     🔴 IT SAYS "TOPIC", AND IT USED TO SAY "THIS CONCEPT". The number is the
     BKT posterior for the concept's SUBTOPIC — `_readMastery` reads
     `__subtopicMastery` or the subtopic's EWMA, and there is no per-concept
     mastery on this page at all. Labelling it per-concept made it a topic
     average wearing a per-node label, which the technical spec bans by name:
     any surface showing a concept's mastery owes the crosswalk TIER and the
     COVERAGE beside it, because `m_k` alone cannot tell "measured at 0.6" from
     "borrowed 0.6 from the neighbourhood". This is the borrowed kind. Naming
     the scope honestly is the cheap half of that rule; sending tier and
     coverage down to the practice payload is the other half and is a backend
     change, deliberately not made in a display pass.

     The number is still the right one to show: it is what actually ends the
     focus loop (`_emitGateCrossed` fires off this same posterior), so the
     learner is watching the thing that closes the overlay. Only its name was
     wrong. */
  const _pushNote = () => {
    if (!window.StageLadder) return;
    if (!Number.isFinite(currentMastery)) {
      window.StageLadder.setNote("");
      return;
    }
    const pct = Math.round(_clamp01(currentMastery) * 100);
    window.StageLadder.setNote(currentMastery >= MASTERY_THRESHOLD
      ? "this topic is mastered"
      : `this topic is ${pct}% mastered ` +
        `(${Math.round(MASTERY_THRESHOLD * 100)}% ends the loop)`);
  };

  // Fires once per page. The parent graph listens for the postMessage to close
  // the overlay and recolour the node; the same-window event is what the
  // in-frame "next concept" prompt hangs off.
  const _emitGateCrossed = (mastery) => {
    if (gateAlreadyCrossed) return;
    gateAlreadyCrossed = true;
    const detail = { kc: window.__kcFocusId || null, subtopic: targetSubtopics[0] || null, mastery };
    window.dispatchEvent(new CustomEvent("competency:gate-crossed", { detail }));
    try {
      if (window.parent && window.parent !== window) {
        window.parent.postMessage({ type: "delta:kc-mastered", ...detail }, window.location.origin);
      }
    } catch (_err) {
      /* cross-origin parent — in-frame event still fired */
    }
  };

  const _onFeedbackUpdate = (e) => {
    const { subtopic, pBefore, pAfter } = e.detail || {};
    if (!targetSubtopics.includes(subtopic) || !Number.isFinite(pAfter)) return;
    // Cache for backend mode, where the adaptive blob isn't the source of truth.
    window.__subtopicMastery = window.__subtopicMastery || {};
    window.__subtopicMastery[subtopic] = pAfter;
    // `pBefore` is the posterior as it stood before THIS answer; falling back
    // to the last value we held keeps the crossing test honest when the
    // feedback payload omits it.
    const from = Number.isFinite(pBefore) ? pBefore : currentMastery;
    currentMastery = pAfter;
    _pushNote();
    if (!(from >= MASTERY_THRESHOLD) && pAfter >= MASTERY_THRESHOLD) _emitGateCrossed(pAfter);
  };

  /** @param {string|string[]} subtopics BKT key(s) for the focused KC. */
  const init = (subtopics) => {
    targetSubtopics = (Array.isArray(subtopics) ? subtopics : [subtopics]).filter(Boolean);
    currentMastery = _readMastery();
    if (!bound) {
      window.addEventListener("competency:feedback-update", _onFeedbackUpdate);
      bound = true;
    }
    return targetSubtopics.length > 0;
  };

  /** The ladder tells us which tier it just served. The rung is the stage
   *  ladder's to draw — this only takes the chance to re-state the mastery
   *  clause, which `StageLadder.hide()` drops on a KC-less question. */
  const setPhaseKind = (_kind) => {
    _pushNote();
  };

  /** Called when the lesson pages finish and graded practice starts. */
  const beginPractice = () => {
    currentMastery = _readMastery() ?? currentMastery;
    _pushNote();
  };

  return {
    init,
    beginPractice,
    setPhaseKind,
    UNLOCK_THRESHOLD,
    MASTERY_THRESHOLD,
    get currentMastery() {
      return currentMastery;
    },
  };
})();

window.CompetencyBar = CompetencyBar;
