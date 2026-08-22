/* Dedicated top-nav Diagnostic page. Practice DOM moves here only while the
   placement is active AND this page is the one on screen, so one editor
   implementation serves two distinct tabs without ever exposing
   #page-practice beneath #page-diagnostic.

   THE ON-SCREEN HALF OF THAT CONDITION IS LOAD-BEARING. It used to key on the
   placement alone, which broke the Practice tab outright for anyone with an
   unfinished placement:
     - the Practice tab was marked `disabled`, and no `:disabled` style exists
       anywhere, so the tab looked normal and simply ate the click; and
     - the workspace stayed parented to the hidden #page-diagnostic while
       #page-practice was force-hidden, so even a programmatic switch landed
       on an empty page.
   Worse, `render()` also runs on `delta:practice-state-changed` — which fires
   on every question load — so a learner who did reach Practice had the
   workspace yanked back out from under them a moment later.
   The Practice tab is never disabled now: leaving a placement mid-probe is a
   legitimate move, and the probe is reloaded from the Diagnostic page. */
const DiagnosticPage = (() => {
  const byId = (id) => document.getElementById(id);
  let running = false;

  // A previous build could leave `disabled` latched on the tab. Clear it once
  // at load so an upgrade doesn't inherit a dead Practice tab.
  const enablePracticeTab = () => {
    document.querySelectorAll('.tab[data-tab="practice"]').forEach((tab) => {
      tab.disabled = false;
      tab.removeAttribute("aria-disabled");
    });
  };

  const diagnosticOnScreen = () =>
    byId("page-diagnostic")?.classList.contains("hidden") === false;

  // Mirrors app.js's own check: the practice page owes its setup panel only
  // when nothing holds the question. Releasing the workspace must not stomp a
  // running/paused session or a lesson back to the setup screen.
  const practiceHoldsQuestion = () =>
    document.body.classList.contains("lesson-mode") ||
    (typeof PracticeSession !== "undefined" &&
      (PracticeSession.isActive?.() || PracticeSession.hasPausedSession?.()));

  // Idempotent on purpose: re-inserting a node that is already in place tears
  // its subtree out of the document and back in, which would reload any live
  // editor/iframe inside the workspace. Only move when it is in the wrong home.
  const syncWorkspace = () => {
    const page = byId("page-diagnostic");
    const practicePage = byId("page-practice");
    const host = byId("diagnostic-workspace-host");
    const home = byId("practice-workspace-home");
    const workspace = document.querySelector(".practice-container");
    if (!page || !practicePage || !host || !home || !workspace) return;

    const hosted = running && diagnosticOnScreen();
    page.classList.toggle("diagnostic-running", hosted);
    if (hosted) {
      if (workspace.parentElement !== host) host.appendChild(workspace);
      host.classList.remove("hidden");
      return;
    }

    if (workspace.previousElementSibling !== home) {
      home.insertAdjacentElement("afterend", workspace);
    }
    host.classList.add("hidden");
    practicePage.classList.toggle("session-idle", !practiceHoldsQuestion());
  };

  const moveWorkspace = (active) => {
    running = !!active;
    syncWorkspace();
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

  // Switching elsewhere hides #page-diagnostic through app.js; the workspace
  // goes home with it, so Practice renders whether or not the placement is
  // finished. Coming back re-claims it (see refresh → render → syncWorkspace).
  const leave = () => syncWorkspace();

  /* app.js calls leave() on every tab switch, but page visibility is the real
     signal and it is not ours to depend on: watch the class instead, so the
     workspace follows the Diagnostic page even if some other route (a solo
     deep link, a future nav) shows or hides it without telling us. */
  const watched = byId("page-diagnostic");
  if (watched && typeof MutationObserver === "function") {
    new MutationObserver(() => syncWorkspace())
      .observe(watched, { attributes: true, attributeFilter: ["class"] });
  }

  enablePracticeTab();
  return { refresh, leave, isRunning: () => running };
})();
window.DiagnosticPage = DiagnosticPage;
if (!document.getElementById("page-diagnostic")?.classList.contains("hidden")) {
  DiagnosticPage.refresh();
}
