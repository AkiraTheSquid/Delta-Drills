/* ================================================================
   STATS DATA — transform + fetch
   ================================================================ */

const calcDiffMult = (p) => {
  if (p <= 0.85) return 0.5 + 0.5 * Math.pow(p / 0.85, 1.8);
  return Math.min(2.5, 1 + Math.pow((p - 0.85) / 0.15, 2.5));
};

const COLD_START_TARGETS = [25, 50, 75];
const COLD_START_PRIORITY_LR = 200;

const estimateLearningRateFromState = (subState) => {
  const n = Number.isFinite(subState?.n) ? subState.n : 0;
  const history = Array.isArray(subState?.history) ? subState.history : [];
  if (n < COLD_START_TARGETS.length || history.length < 2) {
    return COLD_START_PRIORITY_LR;
  }

  const alpha = 1 - Math.exp(-0.3);
  let smoothedRate = null;

  for (let i = 1; i < history.length; i += 1) {
    const curr = history[i];
    const prev = history[i - 1];
    const currPerf = Number.isFinite(curr?.baseline_after) ? curr.baseline_after : 0;
    const prevPerf = Number.isFinite(prev?.baseline_after) ? prev.baseline_after : 0;
    const delta = currPerf - prevPerf;
    smoothedRate = smoothedRate == null ? delta : alpha * delta + (1 - alpha) * smoothedRate;
  }

  return smoothedRate == null ? 0.5 : smoothedRate;
};

const getAdaptiveStateSnapshot = () => {
  if (typeof adaptiveStateJson !== "string" || !adaptiveStateJson) return null;
  try {
    return JSON.parse(adaptiveStateJson);
  } catch (_) {
    return null;
  }
};

const buildItemsFromAdaptiveState = async () => {
  const state = getAdaptiveStateSnapshot();
  if (!state || typeof loadQuestionsBank !== "function") return null;

  const bank = await loadQuestionsBank();
  if (!Array.isArray(bank) || !bank.length) return null;

  const descriptors = new Map();
  bank.forEach((question) => {
    if (!question?.subtopic) return;
    if (descriptors.has(question.subtopic)) return;
    descriptors.set(question.subtopic, {
      subtopic: question.subtopic,
      topic: question.topic || question.subtopic.split(":")[0].trim(),
    });
  });

  const items = [];
  const descriptorList = Array.from(descriptors.values());
  const defaultWeight = descriptorList.length ? 1 / descriptorList.length : 0;
  const effectiveWeights = buildEffectiveWeightsFromSubtopics(descriptorList);

  descriptorList.forEach((entry) => {
    const subState = state?.subtopic_states?.[entry.subtopic] || {};
    const learningRate = estimateLearningRateFromState(subState);
    const questionsAnswered = Number.isFinite(subState?.n) ? subState.n : 0;
    const baseline = Number.isFinite(subState?.baseline) ? subState.baseline : 0;
    const p = Number.isFinite(subState?.p) ? subState.p : 0.5;
    const currentDifficulty = Number.isFinite(subState?.target_difficulty)
      ? subState.target_difficulty
      : (questionsAnswered < COLD_START_TARGETS.length ? COLD_START_TARGETS[questionsAnswered] : 25);
    const weight =
      state?.custom_weights && Number.isFinite(state.custom_weights[entry.subtopic])
        ? state.custom_weights[entry.subtopic]
        : defaultWeight;
    const gradient = (effectiveWeights[entry.subtopic] || weight) * learningRate;

    items.push({
      subtopic: entry.subtopic,
      topic: entry.topic,
      questions_answered: questionsAnswered,
      current_difficulty: currentDifficulty,
      weight,
      learning_rate: learningRate,
      gradient,
      baseline,
      p,
    });
  });

  return items;
};

const buildAreas = (items, weights) => {
  const normalizedWeights = normalizeWeights(weights);
  const resolved = buildResolvedWeightState(items, normalizedWeights);
  const getTopicDisplayPct = (topicName) => {
    const configured = normalizedWeights.topics[topicName];
    if (Number.isFinite(configured)) return configured;
    return Math.round((resolved.topicWeights[topicName] || 0) * 100);
  };
  const getSubtopicDisplayPct = (subtopicId) => {
    const configured = normalizedWeights.subtopics[subtopicId];
    if (Number.isFinite(configured)) return configured;
    return Math.round((resolved.subtopicShares[subtopicId] || 0) * 100);
  };
  // Group subtopics by topic
  const topicMap = new Map();
  items.forEach((item) => {
    const topicName = item.topic || item.subtopic.split(":")[0].trim();
    if (!topicMap.has(topicName)) topicMap.set(topicName, []);
    const colonIdx = item.subtopic.indexOf(":");
    const label = colonIdx >= 0 ? item.subtopic.slice(colonIdx + 2) : item.subtopic;
    topicMap.get(topicName).push({ ...item, label });
  });

  const areas = [];
  let rank = 1;

  topicMap.forEach((subtopics, topicName) => {
    const n = subtopics.length;
    const topicWeightFraction = resolved.topicWeights[topicName] || 0;
    const topicDisplayPct = getTopicDisplayPct(topicName);
    const topicEnabled = isTopicEnabled(topicName, normalizedWeights);

    const subareas = subtopics
      .slice()
      .sort((a, b) => b.gradient - a.gradient)
      .map((st) => {
        const enabled = isSubtopicEnabled(st.subtopic, topicName, normalizedWeights);
        const subShareFraction = resolved.subtopicShares[st.subtopic] || 0;
        const subDisplayPct = getSubtopicDisplayPct(st.subtopic);
        const effectiveWeight = resolved.effectiveWeights[st.subtopic] || 0;
        const gradient = effectiveWeight * st.learning_rate;

        return {
          id: st.subtopic,
          label: st.label,
          topicName,
          enabled,
          weightShare: subShareFraction,
          effectiveWeight,
          displayPct: subDisplayPct,
          currentScore: Math.min(100, st.baseline),
          learningRate: st.learning_rate,
          delta: gradient,
          solved: st.questions_answered,
          currentDifficulty: st.current_difficulty,
          p: st.p,
          targetDifficulty: st.current_difficulty,
          difficultyMultiplier: calcDiffMult(st.p),
        };
      });

    const topicSolved = subtopics.reduce((s, st) => s + st.questions_answered, 0);
    const avgBaseline = n ? subtopics.reduce((s, st) => s + st.baseline, 0) / n : 0;
    const avgLr = n ? subtopics.reduce((s, st) => s + st.learning_rate, 0) / n : 0;
    const topicDelta = subareas.length > 0 ? Math.max(...subareas.map((s) => s.delta)) : 0;
    const avgP = n ? subtopics.reduce((s, st) => s + st.p, 0) / n : 0;
    const avgTargetDiff =
      n ? subtopics.reduce((s, st) => s + st.current_difficulty, 0) / n : 0;

    areas.push({
      id: topicName.toLowerCase().replace(/\s+/g, "-"),
      rank: rank++,
      area: topicName,
      enabled: topicEnabled,
      weight: topicWeightFraction,
      displayPct: topicDisplayPct,
      currentScore: Math.min(100, avgBaseline),
      learningRate: avgLr,
      solved: topicSolved,
      subareas,
      p: avgP,
      targetDifficulty: avgTargetDiff,
      difficultyMultiplier: calcDiffMult(avgP),
    });
  });

  return areas;
};

let rawSubtopicsCache = null;
let statsData = [];

const fetchAndBuild = async () => {
  const adaptiveFallback = async () => {
    const items = await buildItemsFromAdaptiveState();
    if (!items?.length) return null;
    rawSubtopicsCache = items;
    return buildAreas(items, loadWeights());
  };

  if (typeof apiFetch !== "function" || !authToken) return adaptiveFallback();

  let items;
  try {
    const res = await apiFetch("/api/practice/subtopics");
    if (!res.ok) return adaptiveFallback();
    items = await res.json();
  } catch (_) {
    return adaptiveFallback();
  }

  rawSubtopicsCache = items;
  const areas = buildAreas(items, loadWeights());
  // Don't block stats rendering on a best-effort weight sync.
  pushWeightsToBackend(areas);
  return areas;
};
