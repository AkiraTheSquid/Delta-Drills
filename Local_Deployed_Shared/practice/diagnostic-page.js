/* Dedicated top-nav Diagnostic page. Practice DOM moves here only while the
   placement is active, so one editor implementation serves two distinct tabs
   without ever exposing #page-practice beneath #page-diagnostic. */
const DiagnosticPage = (() => {
  const byId = (id) => document.getElementById(id);
  let running = false;

  const setPracticeTabDisabled = (disabled) => {
    document.querySelectorAll('.tab[data-tab="practice"]').forEach((tab) => {
      tab.disabled = disabled;
      tab.setAttribute("aria-disabled", String(disabled));
    });
  };

  const moveWorkspace = (active) => {
    const page = byId("page-diagnostic");
    const practicePage = byId("page-practice");
    const host = byId("diagnostic-workspace-host");
    const home = byId("practice-workspace-home");
    const workspace = document.querySelector(".practice-container");
    if (!page || !practicePage || !host || !home || !workspace) return;

    running = !!active;
    page.classList.toggle("diagnostic-running", running);
    setPracticeTabDisabled(running);
    if (running) {
      host.appendChild(workspace);
      host.classList.remove("hidden");
      practicePage.classList.add("hidden");
      practicePage.classList.remove("session-idle");
      return;
    }

    home.insertAdjacentElement("afterend", workspace);
    host.classList.add("hidden");
    practicePage.classList.add("session-idle");
  };

  const render = (status) => {
    const statusEl = byId("diagnostic-status");
    const priorEl = byId("self-report-row");
    const continueEl = byId("diagnostic-practice-btn");
    const startEl = byId("placement-start-btn");
    const resultsEl = byId("diagnostic-results");
    if (!status) {
      if (statusEl) statusEl.textContent = "Sign in to run placement.";
      priorEl?.classList.add("hidden");
      continueEl?.classList.add("hidden");
      startEl?.classList.add("hidden");
      resultsEl?.classList.add("hidden");
      moveWorkspace(false);
      return;
    }

    const done = Number(status.probes_done) || 0;
    const budget = Number(status.budget) || 14;
    if (statusEl) {
      statusEl.textContent = status.active
        ? `Placement active · ${done} of at most ${budget} probes complete`
        : status.completed_at
          ? `Placement complete · ${done} probes`
          : "Placement not started.";
    }
    priorEl?.classList.toggle("hidden", !status.can_set_prior);
    const hasProbeOnScreen = !!window.PracticeAPI?.currentQuestion?.diagnostic_active;
    continueEl?.classList.toggle("hidden", !status.active || hasProbeOnScreen);
    if (startEl) {
      startEl.textContent = status.completed_at
        ? "Retake placement diagnostic"
        : "Take placement diagnostic";
      startEl.classList.toggle("hidden", status.active);
      startEl.disabled = false;
    }
    resultsEl?.classList.toggle("hidden", !status.completed_at || status.active);
    moveWorkspace(!!status.active);
  };

  const refresh = async () => {
    try { render(await PracticeAPI.diagnosticStatus()); }
    catch (_) { render(null); }
  };

  byId("diagnostic-practice-btn")?.addEventListener("click", () => {
    window.dispatchEvent(new CustomEvent("delta:diagnostic-next"));
  });
  window.addEventListener("delta:practice-state-changed", refresh);

  // Switching elsewhere hides #page-diagnostic through app.js. Workspace
  // remains owned by Diagnostic until placement finishes.
  const leave = () => {};
  return { refresh, leave, isRunning: () => running };
})();
window.DiagnosticPage = DiagnosticPage;
if (!document.getElementById("page-diagnostic")?.classList.contains("hidden")) {
  DiagnosticPage.refresh();
}
