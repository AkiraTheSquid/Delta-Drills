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

const initStats = () => {
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
  initGraphControls();
};
