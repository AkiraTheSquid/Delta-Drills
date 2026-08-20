/* ================================================================
   PRACTICE ADAPTIVE STATE — load/save helpers
   ================================================================ */

let adaptiveStateJson = null; // JSON string of UserPracticeState

function emitAdaptiveStateChanged() {
  window.dispatchEvent(
    new CustomEvent("delta:adaptive-state-changed", {
      detail: { hasState: typeof adaptiveStateJson === "string" && !!adaptiveStateJson },
    })
  );
}

async function syncAdaptiveWeightsToPracticePreferences() {
  if (!adaptiveStateJson || typeof buildEffectiveWeightsFromSubtopics !== "function") return;

  const bank = await loadQuestionsBank();
  const pyodide = await initPyodide();
  if (!bank || !pyodide || !practiceEngineLoaded) return;

  const descriptors = bank.map((q) => ({ topic: q.topic || "", subtopic: q.subtopic }));
  const effectiveWeights = buildEffectiveWeightsFromSubtopics(descriptors);
  const api = pyodide.globals.get("engine_api");
  adaptiveStateJson = api.set_custom_weights(adaptiveStateJson, JSON.stringify(effectiveWeights));
  await saveAdaptiveState();
}

async function loadAdaptiveState() {
  const email = typeof authEmail === "string" && authEmail.trim() ? authEmail.trim() : "guest";
  let loadedExistingState = false;

  if (practiceMode === "supabase") {
    const sbState = await loadPracticeStateFromSupabase(email);
    if (sbState) {
      adaptiveStateJson = JSON.stringify(sbState);
      loadedExistingState = true;
    }
  }

  // Try localStorage
  const localKey = `adaptive_state_${email}`;
  if (!loadedExistingState) {
    const saved = localStorage.getItem(localKey);
    if (saved) {
      adaptiveStateJson = saved;
      loadedExistingState = true;
    }
  }

  // Init fresh state via engine
  if (!loadedExistingState) {
    const pyodide = await initPyodide();
    if (pyodide && practiceEngineLoaded) {
      const api = pyodide.globals.get("engine_api");
      adaptiveStateJson = api.init_state(email);
    } else {
      adaptiveStateJson = null;
    }
  }

  // Apply a self-reported level chosen BEFORE this state existed (the
  // selector writes localStorage immediately; the engine state may not
  // have been initialized yet at click time).
  await syncSelfReportIntoEngineState();

  await syncAdaptiveWeightsToPracticePreferences();
  emitAdaptiveStateChanged();
}

async function syncSelfReportIntoEngineState() {
  if (!adaptiveStateJson || practiceMode === "backend") return;
  const saved = localStorage.getItem(_selfReportStorageKey());
  if (!saved) return;
  try {
    const current = JSON.parse(adaptiveStateJson)?.self_reported_level ?? null;
    const savedLevel = ["beginner", "strong"].includes(saved) ? saved : null;
    if (current === savedLevel) return;
    const pyodide = await initPyodide();
    if (pyodide && practiceEngineLoaded) {
      const api = pyodide.globals.get("engine_api");
      adaptiveStateJson = api.set_self_reported_level(adaptiveStateJson, saved);
      await saveAdaptiveState();
    }
  } catch (err) {
    console.warn("[practice] self-report sync error:", err);
  }
}

// Backend-mode hydration: backend mode skips the Pyodide engine, so
// `adaptiveStateJson` stays null and the concept-graph atom-readiness
// bridge (computeAtomReadiness) sees no signal. Fetch the per-subtopic
// snapshot from /api/practice/state and bare-assign it so the bridge can
// read state.subtopic_states[sub].baseline like in supabase/local mode.
//
// NB: must be bare assignment, not `window.adaptiveStateJson = ...` —
// adaptiveStateJson is a module-scope `let` and only the bare lvalue
// updates that binding; window.x = creates a shadow property the
// readers above never see.
async function loadBackendAdaptiveState() {
  if (typeof apiFetch !== "function") return false;
  try {
    const res = await apiFetch("/api/practice/state");
    if (res.status === 401) {
      handleExpiredToken();
      return false;
    }
    if (!res.ok) {
      console.warn("[practice] /api/practice/state failed:", res.status);
      return false;
    }
    const snapshot = await res.json();
    // Heal a server-side self-report lost to a state wipe/reset: localStorage
    // still remembers the learner's choice, the (fresh) server state doesn't.
    // Push it back up so the prior survives — without this, a wiped account
    // silently reverts to the default prior while the UI button stays lit.
    const savedLevel = localStorage.getItem(_selfReportStorageKey());
    if (
      snapshot.self_reported_level == null &&
      ["beginner", "strong"].includes(savedLevel)
    ) {
      try {
        const putRes = await apiFetch("/api/practice/self-report", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ level: savedLevel }),
        });
        if (putRes.ok) snapshot.self_reported_level = savedLevel;
      } catch (_) { /* non-fatal — selector click still works */ }
    }
    adaptiveStateJson = JSON.stringify(snapshot);
    // Expose raw per-atom BKT posteriors so the ARENA "Score updates" panel
    // can snapshot a before-value and animate the delta after a rating POST.
    window.__atomMastery = snapshot.atom_mastery || {};
    // Refresh the unified per-atom unlock sets (backend = single source of
    // truth; the shipped v2 graph lacks the v3 prereq edges). Non-fatal.
    try {
      const gres = await apiFetch("/api/practice/atom-gates");
      if (gres.ok) {
        const g = await gres.json();
        window.__atomGates = {
          ready: new Set(g.ready || []),
          mastered: new Set(g.mastered || []),
          threshold: g.threshold,
        };
      }
    } catch (_) { /* offline / unauth — drills fall back to readiness */ }
    emitAdaptiveStateChanged();
    return true;
  } catch (err) {
    console.warn("[practice] loadBackendAdaptiveState error:", err);
    return false;
  }
}

async function saveAdaptiveState() {
  if (!adaptiveStateJson) return;
  const email = typeof authEmail === "string" && authEmail.trim() ? authEmail.trim() : "guest";
  const localKey = `adaptive_state_${email}`;

  // Always save to localStorage as backup
  localStorage.setItem(localKey, adaptiveStateJson);

  // Also save to Supabase if in supabase mode
  if (practiceMode === "supabase") {
    const stateObj = JSON.parse(adaptiveStateJson);
    await savePracticeStateToSupabase(email, stateObj);
  }
  emitAdaptiveStateChanged();
}

function getTargetDifficultyFromAdaptiveState(subtopic) {
  if (!adaptiveStateJson || !subtopic) return null;
  try {
    const state = JSON.parse(adaptiveStateJson);
    const subState = state?.subtopic_states?.[subtopic];
    const value = subState?.target_difficulty;
    return Number.isFinite(value) ? value : null;
  } catch (_err) {
    return null;
  }
}

// Returns true if the subtopic is still in the cold-start calibration phase (n < 3).
function isColdStart(subtopic, overrideN) {
  const n = Number.isFinite(overrideN) ? overrideN : (() => {
    if (!adaptiveStateJson || !subtopic) return 0;
    try {
      const state = JSON.parse(adaptiveStateJson);
      return state?.subtopic_states?.[subtopic]?.n ?? 0;
    } catch (_err) {
      return 0;
    }
  })();
  return n < 3;
}

// Returns the 1-based index of the current cold-start question (1, 2, or 3).
function coldStartIndex(subtopic, overrideN) {
  if (!isColdStart(subtopic, overrideN)) return null;
  const n = Number.isFinite(overrideN) ? overrideN : (() => {
    if (!adaptiveStateJson || !subtopic) return 0;
    try {
      const state = JSON.parse(adaptiveStateJson);
      return state?.subtopic_states?.[subtopic]?.n ?? 0;
    } catch (_err) {
      return 0;
    }
  })();
  return n + 1;
}

function getEwmaFromAdaptiveState(subtopic) {
  if (!adaptiveStateJson || !subtopic) return null;
  try {
    const state = JSON.parse(adaptiveStateJson);
    const subState = state?.subtopic_states?.[subtopic];
    const value = subState?.p;
    return Number.isFinite(value) ? value : null;
  } catch (_err) {
    return null;
  }
}

/* ================================================================
   SELF-REPORTED EXPERIENCE LEVEL — seeds where the adaptive queue
   starts (a prior, not a placement; answers overrule it quickly).
   Backend mode: PUT /api/practice/self-report (BKT p_init prior).
   Offline/Supabase mode: Pyodide engine staircase seed.
   ================================================================ */

function _selfReportStorageKey() {
  const email = typeof authEmail === "string" && authEmail.trim() ? authEmail.trim() : "guest";
  return `self_report_level_${email}`;
}

async function setSelfReportedLevel(level) {
  const normalized = ["beginner", "strong"].includes(level) ? level : "default";
  localStorage.setItem(_selfReportStorageKey(), normalized);

  if (practiceMode === "backend" && typeof apiFetch === "function") {
    try {
      const res = await apiFetch("/api/practice/self-report", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ level: normalized }),
      });
      if (!res.ok) console.warn("[practice] self-report save failed:", res.status);
    } catch (err) {
      console.warn("[practice] self-report save error:", err);
    }
  } else if (adaptiveStateJson) {
    try {
      const pyodide = await initPyodide();
      if (pyodide && practiceEngineLoaded) {
        const api = pyodide.globals.get("engine_api");
        adaptiveStateJson = api.set_self_reported_level(adaptiveStateJson, normalized);
        await saveAdaptiveState();
      }
    } catch (err) {
      console.warn("[practice] self-report engine error:", err);
    }
  }
  return normalized;
}

function initSelfReportControls() {
  const row = document.getElementById("self-report-row");
  if (!row) return;
  const note = document.getElementById("self-report-note");
  const buttons = Array.from(row.querySelectorAll(".self-report-btn"));

  const NOTE_BY_LEVEL = {
    beginner: "Starting at the easiest problems — answering well moves you up fast.",
    strong: "Starting at harder problems — a miss steps back down, no harm done.",
    default: "Starting in the middle — your answers take it from there.",
  };

  const highlight = (level) => {
    buttons.forEach((b) => b.classList.toggle("active", b.dataset.level === level));
  };

  const saved = localStorage.getItem(_selfReportStorageKey());
  if (saved) highlight(saved);

  buttons.forEach((btn) => {
    btn.addEventListener("click", async () => {
      const level = await setSelfReportedLevel(btn.dataset.level);
      highlight(level);
      if (note) {
        note.textContent = NOTE_BY_LEVEL[level] || "";
        note.classList.remove("hidden");
      }
      // Cold-start prior only. Do not replace a Practice question from this
      // control; next diagnostic probe consumes the new prior.
      window.DiagnosticPage?.refresh();
    });
  });
}

document.addEventListener("DOMContentLoaded", initSelfReportControls);
