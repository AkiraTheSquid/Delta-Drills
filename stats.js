const statsTableBody = document.getElementById("stats-table-body");
const statsTabs = document.querySelectorAll(".stats-tab");
const statsPanels = document.querySelectorAll("[data-stats-panel]");
const graphContainer = document.getElementById("stats-graph");
const graphRangeButtons = document.querySelectorAll("[data-graph-range]");

const statsData = [
  {
    id: "numpy",
    rank: 1,
    area: "NumPy",
    weight: 0.12,
    currentScore: 72,
    learningRate: 0.42,
    subareas: [
      { id: "core", label: "Core array literacy", weightShare: 0.25, currentScore: 68, learningRate: 0.31, delta: 0.037 },
      { id: "indexing", label: "Indexing and selection", weightShare: 0.25, currentScore: 74, learningRate: 0.45, delta: 0.054 },
      { id: "vector", label: "Vectorization and broadcasting", weightShare: 0.25, currentScore: 70, learningRate: 0.39, delta: 0.047 },
      { id: "applied", label: "Applied patterns and advanced", weightShare: 0.25, currentScore: 65, learningRate: 0.51, delta: 0.061 },
    ],
  },
];

const renderStatsTable = () => {
  if (!statsTableBody) return;
  statsTableBody.innerHTML = "";

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
      <td>${(area.weight * 100).toFixed(0)}%</td>
      <td class="stats-col-score">
        <div class="stats-bar">
          <div class="stats-bar-track">
            <div class="stats-bar-fill" style="width: ${area.currentScore}%"></div>
          </div>
          <span class="stats-bar-value">${area.currentScore}%</span>
        </div>
      </td>
      <td>${area.learningRate.toFixed(2)}</td>
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

    const rankedSubareas = [...area.subareas].sort((a, b) => b.delta - a.delta);
    rankedSubareas.forEach((sub, index) => {
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
        <td>${(sub.weightShare * 100).toFixed(0)}% × ${(area.weight * 100).toFixed(0)}%</td>
        <td class="stats-col-score">
          <div class="stats-bar">
            <div class="stats-bar-track">
              <div class="stats-bar-fill" style="width: ${sub.currentScore}%"></div>
            </div>
            <span class="stats-bar-value">${sub.currentScore}%</span>
          </div>
        </td>
        <td>${sub.learningRate.toFixed(2)}</td>
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

  statsTableBody.querySelectorAll("[data-area-toggle]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const areaId = btn.getAttribute("data-area-toggle");
      const isOpen = btn.dataset.open === "true";
      btn.dataset.open = isOpen ? "false" : "true";
      btn.textContent = isOpen ? "▸" : "▾";
      statsTableBody.querySelectorAll(`[data-subarea-for="${areaId}"]`).forEach((row) => {
        row.classList.toggle("hidden", isOpen);
      });
    });
  });
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

renderStatsTable();

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
    if (!buckets.has(key)) {
      buckets.set(key, []);
    }
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
