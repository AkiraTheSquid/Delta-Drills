/* ================================================================
   STATS DATA — transform + fetch
   ================================================================ */

const calcDiffMult = (p) => {
  if (p <= 0.85) return 0.5 + 0.5 * Math.pow(p / 0.85, 1.8);
  return Math.min(2.5, 1 + Math.pow((p - 0.85) / 0.15, 2.5));
};

const buildAreas = (items, weights) => {
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

    // Server-derived topic weight (sum of the uniform per-subtopic weights)
    const serverTopicWeight = subtopics.reduce((s, st) => s + st.weight, 0);

    // Custom topic weight (percentage, e.g. 70 for 70%)
    const customTopicPct = weights.topics[topicName];
    const topicWeightFraction = customTopicPct != null ? customTopicPct / 100 : serverTopicWeight;
    const topicDisplayPct =
      customTopicPct != null ? customTopicPct : Math.round(serverTopicWeight * 100);

    const defaultSubSharePct = n > 0 ? 100 / n : 100;

    const subareas = subtopics
      .slice()
      .sort((a, b) => b.gradient - a.gradient)
      .map((st) => {
        const customSubPct = weights.subtopics[st.subtopic];
        const subShareFraction = customSubPct != null ? customSubPct / 100 : 1 / (n || 1);
        const subDisplayPct =
          customSubPct != null ? customSubPct : Math.round(defaultSubSharePct);

        // Recalculate gradient with possibly-overridden weights
        const effectiveWeight = topicWeightFraction * subShareFraction;
        const gradient = effectiveWeight * st.learning_rate;

        return {
          id: st.subtopic,
          label: st.label,
          topicName,
          weightShare: subShareFraction,
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
  if (typeof apiFetch !== "function" || !authToken) return null;

  let items;
  try {
    const res = await apiFetch("/api/practice/subtopics");
    if (!res.ok) return null;
    items = await res.json();
  } catch (_) {
    return null;
  }

  rawSubtopicsCache = items;
  const areas = buildAreas(items, loadWeights());
  // Push weights to backend on initial load so stored custom weights take effect
  await pushWeightsToBackend(areas);
  return areas;
};
