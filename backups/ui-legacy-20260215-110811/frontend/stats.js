const statsTableBody = document.getElementById("stats-table-body");
const statsTabs = document.querySelectorAll(".stats-tab");
const statsPanels = document.querySelectorAll("[data-stats-panel]");

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

  statsData.forEach((area) => {
    const maxDelta = area.subareas.reduce((max, s) => Math.max(max, s.delta), 0);
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
      <td>${area.area}</td>
      <td>${(area.weight * 100).toFixed(0)}%</td>
      <td>${area.currentScore}%</td>
      <td>${area.learningRate.toFixed(2)}</td>
      <td>${maxDelta.toFixed(3)}</td>
    `;
    statsTableBody.appendChild(areaRow);

    area.subareas.forEach((sub) => {
      const subRow = document.createElement("tr");
      subRow.className = "stats-row stats-subrow hidden";
      subRow.dataset.subareaFor = area.id;
      subRow.innerHTML = `
        <td class="stats-col-toggle"></td>
        <td class="stats-col-check">
          <input type="checkbox" class="stats-check" checked />
        </td>
        <td>—</td>
        <td class="stats-subarea">${sub.label}</td>
        <td>${(sub.weightShare * 100).toFixed(0)}% × ${(area.weight * 100).toFixed(0)}%</td>
        <td>${sub.currentScore}%</td>
        <td>${sub.learningRate.toFixed(2)}</td>
        <td>${sub.delta.toFixed(3)}</td>
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
