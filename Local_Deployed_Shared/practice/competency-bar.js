/* ================================================================
   COMPETENCY BAR — single-KC practice progress (concept-graph maximize)

   Lives in the ?lesson=<kc>&embed=1 iframe. Shows the learner's BKT
   posterior for THIS concept's subtopic, animates it on every graded
   attempt, and reports the mastery crossing back to the parent graph.

   Two marks on the track, both real engine thresholds:
     left  (0.85) UNLOCK_THRESHOLD  — prereq cleared, dependents unlock
     right (0.95) MASTERY_THRESHOLD — concept mastered, leave the loop

   NOTE: the left mark is drawn as "unlocks next", NOT as a diagnostic
   probe trigger. Mid-loop prerequisite probing needs per-concept
   uncertainty, which atom_mastery does not expose (SD lives only inside
   the placement diagnostic's per-area posterior). That is engine work.

   Phase bands mirror the ERE tiers used by arena-unlock.js so the whole
   app agrees on what "faded" means: <0.40 worked, <0.75 faded, else full.
   ================================================================ */

const CompetencyBar = (() => {
  const UNLOCK_THRESHOLD = 0.85;
  const MASTERY_THRESHOLD = 0.95;
  const FADED_CEIL = 0.75; // ERE: ≥ this ⇒ independent problems

  let container = null;
  // Backend and local mode name the same subtopic differently ("Numpy: Core
  // array literacy" vs "Core array literacy"), and BKT is keyed by whatever
  // question.subtopic says — so hold both and match on either.
  let targetSubtopics = [];
  let currentMastery = null;
  let phase = "lesson"; // "lesson" | "faded" | "independent"
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

  const _phaseFor = (p) => {
    if (!Number.isFinite(p)) return "faded";
    if (p < FADED_CEIL) return "faded";
    return "independent";
  };

  const PHASE_TEXT = {
    lesson: ["Lesson", "Read the worked explanation — nothing graded yet."],
    faded: ["Faded drills", "Scaffolded problems — some of the work is done for you."],
    guided: ["Guided problems", "You write all of it — a hint is there if you want it."],
    independent: ["Independent problems", "No scaffolding — solve it end to end."],
  };

  const _renderPhase = () => {
    if (!container) return;
    const [title, sub] = PHASE_TEXT[phase] || PHASE_TEXT.faded;
    const titleEl = container.querySelector(".competency-phase-title");
    const subEl = container.querySelector(".competency-phase-sub");
    if (titleEl) titleEl.textContent = title;
    if (subEl) subEl.textContent = sub;
    const wrap = container.querySelector(".competency-bar-wrap");
    if (wrap) wrap.dataset.phase = phase;
  };

  const _renderLabel = () => {
    const el = container && container.querySelector(".competency-bar-value");
    if (!el) return;
    el.textContent = Number.isFinite(currentMastery)
      ? `${Math.round(currentMastery * 100)}%`
      : "—";
  };

  const _build = () => {
    if (!container) return;
    const fill = Number.isFinite(currentMastery) ? _clamp01(currentMastery) : 0;
    container.innerHTML = `
      <div class="competency-bar-wrap" data-phase="${phase}">
        <div class="competency-phase">
          <span class="competency-phase-title"></span>
          <span class="competency-phase-sub"></span>
        </div>
        <div class="competency-bar-row">
          <div class="competency-bar-track" role="progressbar"
               aria-label="Mastery of this concept" aria-valuemin="0" aria-valuemax="100">
            <div class="competency-bar-fill" style="width:${fill * 100}%"></div>
            <div class="competency-bar-gate is-unlock" style="left:${UNLOCK_THRESHOLD * 100}%">
              <span class="competency-gate-tip">Unlocks what comes next (85%)</span>
            </div>
            <div class="competency-bar-gate is-mastery" style="left:${MASTERY_THRESHOLD * 100}%">
              <span class="competency-gate-tip">Mastered — back to the map (95%)</span>
            </div>
          </div>
          <div class="competency-bar-value"></div>
        </div>
      </div>
    `;
    _renderPhase();
    _renderLabel();
  };

  // rAF lerp old→new. performance.now() (not Date.now) so the tween tracks
  // frame time; the final frame is written explicitly so a dropped rAF can
  // never leave the fill short of the true value.
  const _animateFill = (oldP, newP, duration = 600) => {
    const fillEl = container && container.querySelector(".competency-bar-fill");
    if (!fillEl || !Number.isFinite(newP)) return;
    // First graded attempt on a fresh concept has no prior posterior — sweep
    // from empty rather than snapping, so the learner sees the bar move.
    const from = Number.isFinite(oldP)
      ? oldP
      : Number.isFinite(currentMastery) ? currentMastery : 0;
    const started = performance.now();

    const step = (now) => {
      const t = Math.min(1, (now - started) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const at = from + (newP - from) * eased;
      fillEl.style.width = `${_clamp01(at) * 100}%`;
      const track = container.querySelector(".competency-bar-track");
      if (track) track.setAttribute("aria-valuenow", String(Math.round(_clamp01(at) * 100)));
      if (t < 1) {
        requestAnimationFrame(step);
        return;
      }
      fillEl.style.width = `${_clamp01(newP) * 100}%`;
      currentMastery = newP;
      _renderLabel();
      if (from < MASTERY_THRESHOLD && newP >= MASTERY_THRESHOLD) _emitGateCrossed(newP);
    };
    requestAnimationFrame(step);
  };

  // Fires once per bar instance. The parent graph listens for the postMessage
  // to close the overlay and recolour the node; the same-window event is what
  // the in-frame "next concept" prompt hangs off.
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
    _animateFill(Number.isFinite(pBefore) ? pBefore : currentMastery, pAfter);
  };

  /** @param {string|string[]} subtopics BKT key(s) for the focused KC. */
  const init = (subtopics) => {
    targetSubtopics = (Array.isArray(subtopics) ? subtopics : [subtopics]).filter(Boolean);
    container = document.getElementById("competency-bar-container");
    if (!container) return false;
    currentMastery = _readMastery();
    phase = "lesson";
    _build();
    container.classList.remove("hidden");
    if (!bound) {
      window.addEventListener("competency:feedback-update", _onFeedbackUpdate);
      bound = true;
    }
    return true;
  };

  /** The ladder tells the bar which tier it just served — the phase label
   *  reports what the learner is ACTUALLY looking at, not what their mastery
   *  band predicts they should be looking at (those disagree once the faded
   *  items run out and independent problems start). */
  const setPhaseKind = (kind) => {
    const next = Object.prototype.hasOwnProperty.call(PHASE_TEXT, kind) && kind !== "lesson"
      ? kind
      : phase;
    if (next === phase) return;
    phase = next;
    _renderPhase();
  };

  /** Called when the lesson pages finish and graded practice starts. */
  const beginPractice = () => {
    currentMastery = _readMastery() ?? currentMastery;
    phase = _phaseFor(currentMastery);
    _renderPhase();
    _renderLabel();
    const fillEl = container && container.querySelector(".competency-bar-fill");
    if (fillEl && Number.isFinite(currentMastery)) {
      fillEl.style.width = `${_clamp01(currentMastery) * 100}%`;
    }
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
    get phase() {
      return phase;
    },
  };
})();

window.CompetencyBar = CompetencyBar;
