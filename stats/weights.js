/* ================================================================
   STATS WEIGHTS — persistence + backend sync
   ================================================================ */

const WEIGHTS_KEY = "delta_drills_weights";

const loadWeights = () => {
  try {
    return JSON.parse(localStorage.getItem(WEIGHTS_KEY)) || { topics: {}, subtopics: {} };
  } catch (_) {
    return { topics: {}, subtopics: {} };
  }
};

const saveWeights = (w) => {
  localStorage.setItem(WEIGHTS_KEY, JSON.stringify(w));
};

const pushWeightsToBackend = async (areas) => {
  if (typeof apiFetch !== "function" || !authToken) return;

  const weights = {};
  areas.forEach((area) => {
    area.subareas.forEach((sub) => {
      weights[sub.id] = area.weight * sub.weightShare;
    });
  });

  try {
    await apiFetch("/api/practice/weights", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ weights }),
    });
  } catch (_) {
    // Non-critical: best-effort
  }
};
