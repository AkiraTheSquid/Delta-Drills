/* ================================================================
   STATS INIT — bootstrap
   ================================================================ */

const loadAndRenderStats = async () => {
  if (statsTableBody) {
    statsTableBody.innerHTML =
      '<tr><td colspan="9" style="text-align:center;padding:1.5rem;color:var(--color-muted)">Loading…</td></tr>';
  }
  try {
    await hydrateWeightsFromSupabase();
    const data = await fetchAndBuild();
    statsData = data || [];
    renderStatsTable();
    renderAdvancedTable();
  } catch (err) {
    console.warn("[stats] failed to load:", err);
    statsData = [];
    renderStatsTable();
    renderAdvancedTable();
  }
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

const shouldAutoRefreshStats = () => {
  const page = document.getElementById("page-statistics");
  return !statsData.length || !!page && !page.classList.contains("hidden");
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
  showStatsPanel("areas");
  initGraphControls();

  // Practice state and auth can finish initializing after the first stats
  // render. Refresh when those sources become available.
  window.addEventListener("delta:adaptive-state-changed", () => {
    if (shouldAutoRefreshStats()) loadAndRenderStats();
  });
  window.addEventListener("delta:auth-state-changed", () => {
    if (shouldAutoRefreshStats()) loadAndRenderStats();
  });

  if (document.getElementById("page-statistics") && !document.getElementById("page-statistics").classList.contains("hidden")) {
    loadAndRenderStats();
  }
};
