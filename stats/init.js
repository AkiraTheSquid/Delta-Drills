/* ================================================================
   STATS INIT — bootstrap
   ================================================================ */

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

const showStatsPanel = (target = "areas") => {
  statsTabs.forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.statsTab === target);
  });
  statsPanels.forEach((panel) => {
    const isActive = panel.dataset.statsPanel === target;
    panel.classList.toggle("hidden", !isActive);
    panel.hidden = !isActive;
    panel.style.display = isActive ? "block" : "none";
  });
};

const initStats = () => {
  statsTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.statsTab;
      showStatsPanel(target);
    });
  });

  // Refresh stats whenever the Statistics page tab is clicked
  document.querySelectorAll(".tab[data-tab='statistics']").forEach((tab) => {
    tab.addEventListener("click", () => {
      showStatsPanel("areas");
      loadAndRenderStats();
    });
  });

  renderStatsTable();
  loadAndRenderStats();
  showStatsPanel("areas");
  initGraphControls();
};
