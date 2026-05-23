/* ================================================================
   STATS INIT — bootstrap
   ================================================================ */

let statsLoadedOnce = false;
let statsNeedsRefresh = true;
let statsLoadPromise = null;
let statsRefreshScheduled = false;

const renderStatsLoadingState = () => {
  if (!statsTableBody || statsData.length) return;
  statsTableBody.innerHTML =
    '<tr><td colspan="9" style="text-align:center;padding:1.5rem;color:var(--color-muted)">Loading…</td></tr>';
};

const loadAndRenderStats = async () => {
  if (statsLoadPromise) return statsLoadPromise;

  renderStatsLoadingState();
  statsLoadPromise = (async () => {
    try {
      const data = await fetchAndBuild();
      statsData = data || [];
      statsLoadedOnce = true;
      statsNeedsRefresh = false;
      renderStatsTable();
      renderAdvancedTable();
      renderPredictedTable();
      return statsData;
    } catch (err) {
      console.warn("[stats] failed to load:", err);
      if (!statsLoadedOnce) {
        statsData = [];
        renderStatsTable();
        renderAdvancedTable();
        renderPredictedTable();
      }
      return statsData;
    } finally {
      statsLoadPromise = null;
    }
  })();

  return statsLoadPromise;
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
  return !!page && !page.classList.contains("hidden");
};

const scheduleStatsRefresh = () => {
  if (!statsNeedsRefresh && statsLoadedOnce) return;
  if (statsRefreshScheduled) return;
  statsRefreshScheduled = true;
  window.setTimeout(() => {
    statsRefreshScheduled = false;
    loadAndRenderStats();
  }, 0);
};

const initStats = () => {
  statsTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.statsTab;
      showStatsPanel(target);
      // Predicted table is sourced from the ARENA registry, not the stats
      // backend — render it on demand so it's populated even when the
      // stats fetch hasn't completed.
      if (target === "predicted") renderPredictedTable();
    });
  });

  // Refresh stats whenever the Statistics page tab is clicked
  document.querySelectorAll(".tab[data-tab='statistics']").forEach((tab) => {
    tab.addEventListener("click", () => {
      showStatsPanel("areas");
      if (statsLoadedOnce) {
        renderStatsTable();
        renderAdvancedTable();
        renderPredictedTable();
      } else {
        renderStatsLoadingState();
      }
      scheduleStatsRefresh();
    });
  });

  renderStatsLoadingState();
  showStatsPanel("areas");
  initGraphControls();
  scheduleStatsRefresh();

  // Practice state and auth can finish initializing after the first stats
  // render. Refresh when those sources become available.
  window.addEventListener("delta:adaptive-state-changed", () => {
    statsNeedsRefresh = true;
    if (shouldAutoRefreshStats()) scheduleStatsRefresh();
  });
  window.addEventListener("delta:practice-state-changed", () => {
    statsNeedsRefresh = true;
    if (shouldAutoRefreshStats()) scheduleStatsRefresh();
  });
  window.addEventListener("delta:auth-state-changed", () => {
    statsNeedsRefresh = true;
    if (shouldAutoRefreshStats()) scheduleStatsRefresh();
  });
};
