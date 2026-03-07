/* ================================================================
   STATS GRAPH — grade history
   ================================================================ */

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
    .map((point, index) =>
      `${index === 0 ? "M" : "L"}${point.x.toFixed(1)},${point.y.toFixed(1)}`
    )
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

const initGraphControls = () => {
  graphRangeButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const range = button.dataset.graphRange;
      graphRangeButtons.forEach((btn) => btn.classList.toggle("active", btn === button));
      renderGraph(range);
    });
  });

  renderGraph("day");
};
