/* ================================================================
   STATS WEIGHTS — persistence + backend sync
   ================================================================ */

const WEIGHTS_KEY = "delta_drills_weights";

const defaultWeights = () => ({
  topics: {},
  subtopics: {},
  topicEnabled: {},
  subtopicEnabled: {},
});

const normalizeWeights = (raw) => {
  const defaults = defaultWeights();
  if (!raw || typeof raw !== "object") return defaults;
  return {
    topics: raw.topics && typeof raw.topics === "object" ? raw.topics : {},
    subtopics: raw.subtopics && typeof raw.subtopics === "object" ? raw.subtopics : {},
    topicEnabled: raw.topicEnabled && typeof raw.topicEnabled === "object" ? raw.topicEnabled : {},
    subtopicEnabled:
      raw.subtopicEnabled && typeof raw.subtopicEnabled === "object" ? raw.subtopicEnabled : {},
  };
};

const loadWeights = () => {
  try {
    return normalizeWeights(JSON.parse(localStorage.getItem(WEIGHTS_KEY)));
  } catch (_) {
    return defaultWeights();
  }
};

const saveWeights = (w) => {
  localStorage.setItem(WEIGHTS_KEY, JSON.stringify(normalizeWeights(w)));
};

const isTopicEnabled = (topicName, weights = loadWeights()) => weights.topicEnabled[topicName] !== false;

const isSubtopicEnabled = (subtopicId, topicName, weights = loadWeights()) => {
  if (!isTopicEnabled(topicName, weights)) return false;
  return weights.subtopicEnabled[subtopicId] !== false;
};

const setTopicEnabled = (topicName, enabled, weights = loadWeights()) => {
  const next = normalizeWeights(weights);
  next.topicEnabled[topicName] = enabled !== false;
  return next;
};

const setSubtopicEnabled = (subtopicId, enabled, weights = loadWeights()) => {
  const next = normalizeWeights(weights);
  next.subtopicEnabled[subtopicId] = enabled !== false;
  return next;
};

const sumValues = (values) => values.reduce((sum, value) => sum + value, 0);

const buildEffectiveWeightsFromSubtopics = (subtopics, weights = loadWeights()) => {
  const normalized = normalizeWeights(weights);
  const topicEntries = new Map();

  subtopics.forEach((entry) => {
    if (!entry?.topic || !entry?.subtopic) return;
    if (!topicEntries.has(entry.topic)) {
      topicEntries.set(entry.topic, []);
    }
    topicEntries.get(entry.topic).push(entry);
  });

  const topicBaseWeights = new Map();
  topicEntries.forEach((entries, topicName) => {
    if (!isTopicEnabled(topicName, normalized)) return;
    const customTopicPct = normalized.topics[topicName];
    const fallbackTopicWeight = entries.every((entry) => Number.isFinite(entry.weight))
      ? sumValues(entries.map((entry) => entry.weight))
      : 1;
    topicBaseWeights.set(topicName, customTopicPct != null ? customTopicPct / 100 : fallbackTopicWeight);
  });

  const totalTopicWeight = sumValues(Array.from(topicBaseWeights.values()));
  const resolvedTopicWeights = new Map();
  topicBaseWeights.forEach((baseWeight, topicName) => {
    resolvedTopicWeights.set(topicName, totalTopicWeight > 0 ? baseWeight / totalTopicWeight : 0);
  });

  const effectiveWeights = {};
  topicEntries.forEach((entries, topicName) => {
    const enabledEntries = entries.filter((entry) =>
      isSubtopicEnabled(entry.subtopic, topicName, normalized)
    );
    const subtopicBaseWeights = enabledEntries.map((entry) => {
      const customSubPct = normalized.subtopics[entry.subtopic];
      return customSubPct != null ? customSubPct / 100 : 1;
    });
    const totalSubtopicWeight = sumValues(subtopicBaseWeights);
    const topicWeight = resolvedTopicWeights.get(topicName) || 0;

    entries.forEach((entry) => {
      effectiveWeights[entry.subtopic] = 0;
    });

    enabledEntries.forEach((entry, index) => {
      const normalizedShare = totalSubtopicWeight > 0 ? subtopicBaseWeights[index] / totalSubtopicWeight : 0;
      effectiveWeights[entry.subtopic] = topicWeight * normalizedShare;
    });
  });

  return effectiveWeights;
};

const buildResolvedWeightState = (subtopics, weights = loadWeights()) => {
  const normalized = normalizeWeights(weights);
  const effectiveWeights = buildEffectiveWeightsFromSubtopics(subtopics, normalized);
  const topicWeights = {};
  const subtopicShares = {};
  const topicTotals = {};

  subtopics.forEach((entry) => {
    if (!entry?.topic || !entry?.subtopic) return;
    topicTotals[entry.topic] = (topicTotals[entry.topic] || 0) + (effectiveWeights[entry.subtopic] || 0);
  });

  subtopics.forEach((entry) => {
    if (!entry?.topic || !entry?.subtopic) return;
    topicWeights[entry.topic] = topicTotals[entry.topic] || 0;
    const topicWeight = topicTotals[entry.topic] || 0;
    subtopicShares[entry.subtopic] =
      topicWeight > 0 ? (effectiveWeights[entry.subtopic] || 0) / topicWeight : 0;
  });

  return {
    effectiveWeights,
    topicWeights,
    subtopicShares,
  };
};

const pushWeightsToBackend = async (areas) => {
  if (typeof apiFetch !== "function" || !authToken) return;

  const weights = {};
  areas.forEach((area) => {
    area.subareas.forEach((sub) => {
      weights[sub.id] = sub.effectiveWeight;
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
