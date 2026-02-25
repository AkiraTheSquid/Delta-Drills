const statsTableBody = document.getElementById("stats-table-body");
const statsTabs = document.querySelectorAll(".stats-tab");
const statsPanels = document.querySelectorAll("[data-stats-panel]");
const graphContainer = document.getElementById("stats-graph");
const graphRangeButtons = document.querySelectorAll("[data-graph-range]");

// Tracks which area IDs are currently expanded — survives re-renders
const openAreas = new Set();

// ---------------------------------------------------------------------------
// Custom weight persistence (localStorage)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Difficulty multiplier — mirrors the backend formula in adaptive.py
// ---------------------------------------------------------------------------

const calcDiffMult = (p) => {
  if (p <= 0.85) return 0.5 + 0.5 * Math.pow(p / 0.85, 1.8);
  return Math.min(2.5, 1 + Math.pow((p - 0.85) / 0.15, 2.5));
};

// ---------------------------------------------------------------------------
// Data transform: raw API items → area objects used by the renderer
// Applies custom weights on top of the server's uniform defaults.
// ---------------------------------------------------------------------------

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
    const topicWeightFraction =
      customTopicPct != null ? customTopicPct / 100 : serverTopicWeight;
    const topicDisplayPct =
      customTopicPct != null
        ? customTopicPct
        : Math.round(serverTopicWeight * 100);

    const defaultSubSharePct = n > 0 ? 100 / n : 100;

    const subareas = subtopics
      .slice()
      .sort((a, b) => b.gradient - a.gradient)
      .map((st) => {
        const customSubPct = weights.subtopics[st.subtopic];
        const subShareFraction =
          customSubPct != null ? customSubPct / 100 : 1 / (n || 1);
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
    const avgBaseline = n
      ? subtopics.reduce((s, st) => s + st.baseline, 0) / n
      : 0;
    const avgLr = n
      ? subtopics.reduce((s, st) => s + st.learning_rate, 0) / n
      : 0;
    const topicDelta =
      subareas.length > 0 ? Math.max(...subareas.map((s) => s.delta)) : 0;
    const avgP = n ? subtopics.reduce((s, st) => s + st.p, 0) / n : 0;
    const avgTargetDiff = n
      ? subtopics.reduce((s, st) => s + st.current_difficulty, 0) / n
      : 0;

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

// ---------------------------------------------------------------------------
// Push effective weights to backend so selection algorithm uses them
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Fetch + cache
// ---------------------------------------------------------------------------

let rawSubtopicsCache = null;

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

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

let statsData = [];

// Sets a number input's width to exactly fit its digit count
const fitInputWidth = (input) => {
  input.style.width = Math.max(1, String(input.value).length) + "ch";
};

const renderStatsTable = () => {
  if (!statsTableBody) return;

  // Snapshot which areas are currently open before destroying DOM
  statsTableBody.querySelectorAll("[data-area-toggle]").forEach((btn) => {
    if (btn.dataset.open === "true") openAreas.add(btn.getAttribute("data-area-toggle"));
    else openAreas.delete(btn.getAttribute("data-area-toggle"));
  });

  statsTableBody.innerHTML = "";

  if (!statsData.length) {
    statsTableBody.innerHTML =
      '<tr><td colspan="9" style="text-align:center;padding:1.5rem;color:var(--color-muted)">No data yet — start practising to see your stats.</td></tr>';
    return;
  }

  const maxDelta = statsData.reduce((maxArea, area) => {
    const subMax = area.subareas.reduce((max, s) => Math.max(max, s.delta), 0);
    return Math.max(maxArea, subMax);
  }, 0);

  statsData.forEach((area) => {
    const areaDelta = area.subareas.reduce((max, s) => Math.max(max, s.delta), 0);
    const areaDeltaWidth = maxDelta > 0 ? (areaDelta / maxDelta) * 100 : 0;

    const areaRow = document.createElement("tr");
    areaRow.className = "stats-row stats-row-top";
    areaRow.innerHTML = `
      <td class="stats-col-toggle">
        <button class="stats-toggle" type="button" data-area-toggle="${area.id}">▸</button>
      </td>
      <td class="stats-col-check">
        <input type="checkbox" class="stats-check" checked />
      </td>
      <td>${area.rank}</td>
      <td class="stats-col-area">${area.area}</td>
      <td class="stats-col-weight">
        <input type="number" class="weight-input"
               data-weight-topic="${area.area}"
               value="${area.displayPct}"
               min="0" max="999" step="1" />%
      </td>
      <td class="stats-col-score">
        <div class="stats-bar">
          <div class="stats-bar-track">
            <div class="stats-bar-fill" style="width: ${area.currentScore}%"></div>
          </div>
          <span class="stats-bar-value">${area.currentScore.toFixed(0)}/100</span>
        </div>
      </td>
      <td class="stats-col-solved">${area.solved}</td>
      <td class="stats-col-lr">${area.learningRate.toFixed(2)}</td>
      <td class="stats-col-delta">
        <div class="stats-bar">
          <div class="stats-bar-track">
            <div class="stats-bar-fill stats-bar-fill-delta" style="width: ${areaDeltaWidth}%"></div>
          </div>
          <span class="stats-bar-value">${areaDelta.toFixed(3)}</span>
        </div>
      </td>
    `;
    statsTableBody.appendChild(areaRow);

    area.subareas.forEach((sub, index) => {
      const subDeltaWidth = maxDelta > 0 ? (sub.delta / maxDelta) * 100 : 0;
      const subRow = document.createElement("tr");
      subRow.className = "stats-row stats-subrow hidden";
      subRow.dataset.subareaFor = area.id;
      subRow.innerHTML = `
        <td class="stats-col-toggle"></td>
        <td class="stats-col-check">
          <input type="checkbox" class="stats-check" checked />
        </td>
        <td>${area.rank}.${index + 1}</td>
        <td class="stats-col-area stats-subarea">${sub.label}</td>
        <td class="stats-col-weight">
          <input type="number" class="weight-input"
                 data-weight-subtopic="${sub.id}"
                 data-parent-topic="${area.area}"
                 value="${sub.displayPct}"
                 min="0" max="999" step="1" />% × ${area.displayPct}%
        </td>
        <td class="stats-col-score">
          <div class="stats-bar">
            <div class="stats-bar-track">
              <div class="stats-bar-fill" style="width: ${sub.currentScore}%"></div>
            </div>
            <span class="stats-bar-value">${sub.currentScore.toFixed(0)}/100</span>
          </div>
        </td>
        <td class="stats-col-solved">${sub.solved}</td>
        <td class="stats-col-lr">${sub.learningRate.toFixed(2)}</td>
        <td class="stats-col-delta">
          <div class="stats-bar">
            <div class="stats-bar-track">
              <div class="stats-bar-fill stats-bar-fill-delta" style="width: ${subDeltaWidth}%"></div>
            </div>
            <span class="stats-bar-value">${sub.delta.toFixed(3)}</span>
          </div>
        </td>
      `;
      statsTableBody.appendChild(subRow);
    });
  });

  // --- Toggle expand/collapse ---
  statsTableBody.querySelectorAll("[data-area-toggle]").forEach((btn) => {
    // Restore previously open areas
    const areaId = btn.getAttribute("data-area-toggle");
    if (openAreas.has(areaId)) {
      btn.dataset.open = "true";
      btn.textContent = "▾";
      statsTableBody.querySelectorAll(`[data-subarea-for="${areaId}"]`).forEach((row) => {
        row.classList.remove("hidden");
      });
    }

    btn.addEventListener("click", () => {
      const isOpen = btn.dataset.open === "true";
      btn.dataset.open = isOpen ? "false" : "true";
      btn.textContent = isOpen ? "▸" : "▾";
      if (isOpen) openAreas.delete(areaId); else openAreas.add(areaId);
      statsTableBody.querySelectorAll(`[data-subarea-for="${areaId}"]`).forEach((row) => {
        row.classList.toggle("hidden", isOpen);
      });
    });
  });

  // --- Editable weight inputs ---
  statsTableBody.querySelectorAll(".weight-input").forEach((input) => {
    // Size the input to exactly fit its current digits
    fitInputWidth(input);

    // Resize as the user types
    input.addEventListener("input", () => fitInputWidth(input));

    // Enter blurs without doing anything else (prevents toggle button activation)
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        input.blur();
      }
    });

    input.addEventListener("change", () => {
      const val = Math.max(0, Number(input.value) || 0);
      input.value = val;
      fitInputWidth(input);

      const weights = loadWeights();
      const topicKey = input.dataset.weightTopic;
      const subtopicKey = input.dataset.weightSubtopic;

      if (topicKey) {
        weights.topics[topicKey] = val;
      } else if (subtopicKey) {
        weights.subtopics[subtopicKey] = val;
      }

      saveWeights(weights);

      // Rebuild from cache (no re-fetch) and re-render
      if (rawSubtopicsCache) {
        statsData = buildAreas(rawSubtopicsCache, weights);
        renderStatsTable();
        pushWeightsToBackend(statsData);
      }
    });
  });
};

// ---------------------------------------------------------------------------
// Advanced tab renderer
// ---------------------------------------------------------------------------

const openAreasAdv = new Set();

const renderAdvancedTable = () => {
  const body = document.getElementById("adv-table-body");
  if (!body) return;

  // Snapshot which areas are currently open before destroying DOM
  body.querySelectorAll("[data-adv-toggle]").forEach((btn) => {
    if (btn.dataset.open === "true") openAreasAdv.add(btn.getAttribute("data-adv-toggle"));
    else openAreasAdv.delete(btn.getAttribute("data-adv-toggle"));
  });

  body.innerHTML = "";

  if (!statsData.length) {
    body.innerHTML =
      '<tr><td colspan="11" style="text-align:center;padding:1.5rem;color:var(--color-muted)">No data yet — start practising to see your stats.</td></tr>';
    return;
  }

  const maxDelta = statsData.reduce((maxArea, area) => {
    const subMax = area.subareas.reduce((max, s) => Math.max(max, s.delta), 0);
    return Math.max(maxArea, subMax);
  }, 0);

  statsData.forEach((area) => {
    const areaDelta = area.subareas.reduce((max, s) => Math.max(max, s.delta), 0);
    const areaDeltaWidth = maxDelta > 0 ? (areaDelta / maxDelta) * 100 : 0;

    const areaRow = document.createElement("tr");
    areaRow.className = "stats-row stats-row-top";
    areaRow.innerHTML = `
      <td class="stats-col-toggle">
        <button class="stats-toggle" type="button" data-adv-toggle="${area.id}">▸</button>
      </td>
      <td class="stats-col-check">
        <input type="checkbox" class="stats-check" checked />
      </td>
      <td class="stats-col-rank">${area.rank}</td>
      <td class="stats-col-area">${area.area}</td>
      <td class="stats-col-score">
        <div class="stats-bar">
          <div class="stats-bar-track">
            <div class="stats-bar-fill" style="width: ${area.currentScore}%"></div>
          </div>
          <span class="stats-bar-value">${area.currentScore.toFixed(0)}/100</span>
        </div>
      </td>
      <td class="stats-col-solved">${area.solved}</td>
      <td class="stats-col-lr">${area.learningRate.toFixed(2)}</td>
      <td class="stats-col-delta">
        <div class="stats-bar">
          <div class="stats-bar-track">
            <div class="stats-bar-fill stats-bar-fill-delta" style="width: ${areaDeltaWidth}%"></div>
          </div>
          <span class="stats-bar-value">${areaDelta.toFixed(3)}</span>
        </div>
      </td>
      <td class="stats-col-p">${(area.p * 100).toFixed(1)}%</td>
      <td class="stats-col-target">${area.targetDifficulty.toFixed(1)}</td>
      <td class="stats-col-mult">${area.difficultyMultiplier.toFixed(2)}×</td>
    `;
    body.appendChild(areaRow);

    area.subareas.forEach((sub, index) => {
      const subDeltaWidth = maxDelta > 0 ? (sub.delta / maxDelta) * 100 : 0;
      const subRow = document.createElement("tr");
      subRow.className = "stats-row stats-subrow hidden";
      subRow.dataset.advSubareaFor = area.id;
      subRow.innerHTML = `
        <td class="stats-col-toggle"></td>
        <td class="stats-col-check">
          <input type="checkbox" class="stats-check" checked />
        </td>
        <td class="stats-col-rank">${area.rank}.${index + 1}</td>
        <td class="stats-col-area stats-subarea">${sub.label}</td>
        <td class="stats-col-score">
          <div class="stats-bar">
            <div class="stats-bar-track">
              <div class="stats-bar-fill" style="width: ${sub.currentScore}%"></div>
            </div>
            <span class="stats-bar-value">${sub.currentScore.toFixed(0)}/100</span>
          </div>
        </td>
        <td class="stats-col-solved">${sub.solved}</td>
        <td class="stats-col-lr">${sub.learningRate.toFixed(2)}</td>
        <td class="stats-col-delta">
          <div class="stats-bar">
            <div class="stats-bar-track">
              <div class="stats-bar-fill stats-bar-fill-delta" style="width: ${subDeltaWidth}%"></div>
            </div>
            <span class="stats-bar-value">${sub.delta.toFixed(3)}</span>
          </div>
        </td>
        <td class="stats-col-p">${(sub.p * 100).toFixed(1)}%</td>
        <td class="stats-col-target">${sub.targetDifficulty.toFixed(1)}</td>
        <td class="stats-col-mult">${sub.difficultyMultiplier.toFixed(2)}×</td>
      `;
      body.appendChild(subRow);
    });
  });

  // Toggle expand/collapse
  body.querySelectorAll("[data-adv-toggle]").forEach((btn) => {
    const areaId = btn.getAttribute("data-adv-toggle");
    if (openAreasAdv.has(areaId)) {
      btn.dataset.open = "true";
      btn.textContent = "▾";
      body.querySelectorAll(`[data-adv-subarea-for="${areaId}"]`).forEach((row) => {
        row.classList.remove("hidden");
      });
    }
    btn.addEventListener("click", () => {
      const isOpen = btn.dataset.open === "true";
      btn.dataset.open = isOpen ? "false" : "true";
      btn.textContent = isOpen ? "▸" : "▾";
      if (isOpen) openAreasAdv.delete(areaId); else openAreasAdv.add(areaId);
      body.querySelectorAll(`[data-adv-subarea-for="${areaId}"]`).forEach((row) => {
        row.classList.toggle("hidden", isOpen);
      });
    });
  });
};

const loadAndRenderStats = async () => {
  if (statsTableBody) {
    statsTableBody.innerHTML =
      '<tr><td colspan="9" style="text-align:center;padding:1.5rem;color:var(--color-muted)">Loading…</td></tr>';
  }
  const data = await fetchAndBuild();
  statsData = data || [];
  renderStatsTable();
  renderAdvancedTable();
};

statsTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = tab.dataset.statsTab;
    statsTabs.forEach((t) => t.classList.toggle("active", t === tab));
    statsPanels.forEach((panel) => {
      panel.classList.toggle("hidden", panel.dataset.statsPanel !== target);
    });
  });
});

// Refresh stats whenever the Statistics page tab is clicked
document.querySelectorAll(".tab[data-tab='statistics']").forEach((tab) => {
  tab.addEventListener("click", loadAndRenderStats);
});

renderStatsTable();
loadAndRenderStats();

// ---------------------------------------------------------------------------
// Grade history graph
// ---------------------------------------------------------------------------

const gradeSeries = [
  { date: "2026-02-01", grade: 62 },
  { date: "2026-02-02", grade: 65 },
  { date: "2026-02-03", grade: 68 },
  { date: "2026-02-04", grade: 60 },
  { date: "2026-02-05", grade: 72 },
  { date: "2026-02-06", grade: 70 },
  { date: "2026-02-07", grade: 74 },
  { date: "2026-02-08", grade: 76 },
  { date: "2026-02-09", grade: 71 },
  { date: "2026-02-10", grade: 78 },
  { date: "2026-02-11", grade: 80 },
  { date: "2026-02-12", grade: 77 },
  { date: "2026-02-13", grade: 82 },
  { date: "2026-02-14", grade: 79 },
  { date: "2026-02-15", grade: 84 },
];

const parseDate = (value) => new Date(`${value}T00:00:00Z`);

const groupByRange = (range) => {
  if (range === "day") {
    return gradeSeries.map((point) => ({
      label: point.date.slice(5),
      grade: point.grade,
    }));
  }

  const buckets = new Map();
  gradeSeries.forEach((point) => {
    const date = parseDate(point.date);
    let key = "";
    if (range === "week") {
      const day = date.getUTCDay() || 7;
      const monday = new Date(date);
      monday.setUTCDate(date.getUTCDate() - day + 1);
      key = `${monday.getUTCFullYear()}-${String(monday.getUTCMonth() + 1).padStart(2, "0")}-${String(monday.getUTCDate()).padStart(2, "0")}`;
    } else {
      key = `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, "0")}`;
    }
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(point.grade);
  });

  return Array.from(buckets.entries()).map(([label, values]) => ({
    label: range === "week" ? label.slice(5) : label,
    grade: Math.round(values.reduce((sum, v) => sum + v, 0) / values.length),
  }));
};

const renderGraph = (range) => {
  if (!graphContainer) return;
  const data = groupByRange(range);
  const width = graphContainer.clientWidth || 640;
  const height = 220;
  const padding = 24;
  const maxGrade = 100;
  const minGrade = 0;
  const xStep = data.length > 1 ? (width - padding * 2) / (data.length - 1) : 0;

  const points = data.map((point, index) => {
    const x = padding + index * xStep;
    const ratio = (point.grade - minGrade) / (maxGrade - minGrade);
    const y = height - padding - ratio * (height - padding * 2);
    return { x, y, label: point.label, grade: point.grade };
  });

  const path = points
    .map((point, index) => `${index === 0 ? "M" : "L"}${point.x.toFixed(1)},${point.y.toFixed(1)}`)
    .join(" ");

  graphContainer.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" class="stats-graph-svg">
      <defs>
        <linearGradient id="statsLine" x1="0" x2="1">
          <stop offset="0%" stop-color="rgba(80,129,255,0.9)" />
          <stop offset="100%" stop-color="rgba(80,129,255,0.4)" />
        </linearGradient>
      </defs>
      <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" class="stats-graph-axis" />
      <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" class="stats-graph-axis" />
      <path d="${path}" fill="none" stroke="url(#statsLine)" stroke-width="3" stroke-linecap="round" />
      ${points
        .map(
          (point) => `
        <circle cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="4" class="stats-graph-point" />
        <text x="${point.x.toFixed(1)}" y="${height - 6}" class="stats-graph-label">${point.label}</text>
      `
        )
        .join("")}
    </svg>
  `;
};

graphRangeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const range = button.dataset.graphRange;
    graphRangeButtons.forEach((btn) => btn.classList.toggle("active", btn === button));
    renderGraph(range);
  });
});

renderGraph("day");
