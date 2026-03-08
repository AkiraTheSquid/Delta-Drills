/* ================================================================
   PRACTICE ADAPTIVE STATE — load/save helpers
   ================================================================ */

let adaptiveStateJson = null; // JSON string of UserPracticeState

async function loadAdaptiveState() {
  const email = typeof authEmail === "string" && authEmail.trim() ? authEmail.trim() : "guest";

  if (practiceMode === "supabase") {
    const sbState = await loadPracticeStateFromSupabase(email);
    if (sbState) {
      adaptiveStateJson = JSON.stringify(sbState);
      return;
    }
  }

  // Try localStorage
  const localKey = `adaptive_state_${email}`;
  const saved = localStorage.getItem(localKey);
  if (saved) {
    adaptiveStateJson = saved;
    return;
  }

  // Init fresh state via engine
  const pyodide = await initPyodide();
  if (pyodide && practiceEngineLoaded) {
    const api = pyodide.globals.get("engine_api");
    adaptiveStateJson = api.init_state(email);
  } else {
    adaptiveStateJson = null;
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
