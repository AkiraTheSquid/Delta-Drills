/* Dedicated top-nav Placement test page (the route, the ids and the backend
   endpoints all still say `diagnostic` — only the learner-facing words changed).
   Practice DOM moves here only while the placement is active AND this page is
   the one on screen, so one editor
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

  /* The ONE writer of the placement start button. events.js has its own reason
     to refresh it (the click handler, and the state-changed sweep) and used to
     carry a second copy of the label + visibility rules; two copies of a label
     is how a button flickers between two names on refresh. */
  const renderStartButton = (status, el = byId("placement-start-btn")) => {
    if (!el) return;
    el.textContent = status?.completed_at
      ? "Retake the placement test"
      : "Take the placement test";
    el.disabled = false;
    const off = !!status?.active;
    el.classList.toggle("hidden", off);
    /* The .placement-cta wrapper has to go with it. infotips.js already mirrors
       `.hidden` onto the dot, so nothing is left VISIBLE — but an empty
       inline-flex item still counts in .diagnostic-actions' `gap`, which shunted
       "Load next placement question" 12px off the card's left edge for the whole
       test. Measured, not guessed. */
    el.parentElement?.classList.toggle("hidden", off);
  };

  /* How far through the test you are, in the same visual language as the
     scaffolding bar on Practice: one continuous track that fills as probes
     land. `budget` is a CEILING — the placement stops as soon as the estimate
     is confident, which can be as early as `min_probes` — so the count says
     "of at most" and the tick marks the earliest possible finish. A bar that
     implied a fixed length would be wrong for most runs, since most stop early. */
  const renderProgress = (status) => {
    const host = byId("placement-progress");
    if (!host) return;
    const show = !!status?.active;
    host.classList.toggle("hidden", !show);
    if (!show) return;

    const budget = Math.max(1, Number(status.budget) || 14);
    const done = Math.min(budget, Math.max(0, Number(status.probes_done) || 0));
    const minProbes = Number(status.min_probes) || 0;

    const fill = byId("placement-progress-fill");
    if (fill) fill.style.width = `${(done / budget) * 100}%`;
    const tick = byId("placement-progress-tick");
    if (tick) {
      const usable = minProbes > 0 && minProbes < budget;
      tick.classList.toggle("hidden", !usable);
      if (usable) {
        tick.style.left = `${(minProbes / budget) * 100}%`;
        tick.title = `Can finish from ${minProbes} questions`;
      }
    }
    const count = byId("placement-progress-count");
    if (count) count.textContent = `${done} of at most ${budget}`;
    host.setAttribute("aria-valuenow", String(done));
    host.setAttribute("aria-valuemax", String(budget));
  };

  const render = (status) => {
    const statusEl = byId("diagnostic-status");
    const priorEl = byId("self-report-row");
    const continueEl = byId("diagnostic-practice-btn");
    const startEl = byId("placement-start-btn");
    const resultsEl = byId("diagnostic-results");
    if (!status) {
      if (statusEl) statusEl.textContent = "Sign in to take the placement test.";
      priorEl?.classList.add("hidden");
      continueEl?.classList.add("hidden");
      startEl?.classList.add("hidden");
      resultsEl?.classList.add("hidden");
      renderProgress(null);
      moveWorkspace(false);
      return;
    }

    const done = Number(status.probes_done) || 0;
    const budget = Number(status.budget) || 14;
    if (statusEl) {
      statusEl.textContent = status.active
        ? `Placement test in progress · ${done} of at most ${budget} questions answered`
        : status.completed_at
          ? `Placement test complete · ${done} questions`
          : "Placement test not started.";
    }
    renderProgress(status);
    priorEl?.classList.toggle("hidden", !status.can_set_prior);
    // Same trap as notebook-view.js documents: `PracticeAPI` is a top-level
    // const, so it is NOT on `window` and this read was always undefined —
    // which left "Load next placement question" on screen underneath the probe
    // it would replace. Script-scope binding first, window as the fallback.
    const _papi = typeof PracticeAPI !== "undefined" ? PracticeAPI : window.PracticeAPI;
    const hasProbeOnScreen = !!_papi?.currentQuestion?.diagnostic_active;
    continueEl?.classList.toggle("hidden", !status.active || hasProbeOnScreen);
    renderStartButton(status, startEl);
    /* This file owns WHETHER the results card shows; placement-results.js owns
       what is in it. Fill before unhiding so the card never flashes the shape
       of the previous placement, and fill only when it is actually going to be
       shown — a mid-test status carries live estimates that would read as a
       finished result. */
    const showResults = !!status.completed_at && !status.active;
    if (showResults) window.PlacementResults?.render(status);
    resultsEl?.classList.toggle("hidden", !showResults);
    moveWorkspace(!!status.active);
  };

  /* An unanswerable status call is not the same thing as "not signed in".

     `render(null)` writes "Sign in to take the placement test." and hides
     every button — which is the truth for a visitor with no token, and a lie
     for a learner whose backend is simply restarting. They got a page with no
     start button, no continue button, and an instruction that does not apply
     to them, with nothing to click and no way to know it was temporary.

     So a failed call says so, and tries again. Every attempt after the first
     is silent about failing until it gives up, because the common case is one
     redeploy's worth of downtime and the page fixes itself. */
  const UNREACHABLE_RETRIES = 4;
  const UNREACHABLE_BACKOFF_MS = 2000;
  let retryTimer = null;
  /* Every refresh belongs to a generation, and only the newest may paint.
     Clearing the timer stops a SCHEDULED retry; it cannot stop one already
     waiting on the network, and that one is the problem — it resolves after a
     newer refresh has already rendered and puts the page back to a state the
     server has since moved past. */
  let generation = 0;

  const renderUnreachable = (attempt) => {
    const statusEl = byId("diagnostic-status");
    if (!statusEl) return;
    statusEl.textContent = attempt < UNREACHABLE_RETRIES
      ? "Checking your placement status…"
      : "Couldn't reach the server. Reload the page to try again.";
  };

  const refresh = async (attempt = 0) => {
    clearTimeout(retryTimer);
    retryTimer = null;
    const mine = ++generation;
    let status;
    try {
      status = await PracticeAPI.diagnosticStatus();
    } catch (_) {
      status = { unavailable: true, httpStatus: 0 };
    }
    if (mine !== generation) return;

    if (status && status.unavailable) {
      /* A 401 that survived apiFetch is USUALLY a real signed-out state: a
         guest's token is refreshed in place there, so reaching here means
         there was nothing left to refresh with. Retrying that forever would
         show an outage message to someone who just needs to sign in.

         Usually — but the same 401 comes back when the silent re-login could
         not run: the backend was unreachable, or it is inside its 30s
         cooldown. This browser still holds a guest password, so telling that
         learner to sign in is both wrong and sticky (nothing re-renders until
         they change tabs). DDGuest.canRecover() separates the two, and the
         outage copy's advice — reload — is what actually fixes the other
         case, because a reload mints a fresh guest.

         401 only. A 403 is the backend refusing what this account is allowed
         to do, and a new token for the same account changes nothing. */
      const recoverable = status.httpStatus === 401 && window.DDGuest?.canRecover?.() === true;
      if ((status.httpStatus === 401 || status.httpStatus === 403) && !recoverable) {
        render(null);
        return;
      }
      renderUnreachable(attempt);
      if (attempt < UNREACHABLE_RETRIES) {
        retryTimer = setTimeout(() => refresh(attempt + 1), UNREACHABLE_BACKOFF_MS * (attempt + 1));
      }
      return;
    }
    render(status);
  };

  byId("diagnostic-practice-btn")?.addEventListener("click", () => {
    window.dispatchEvent(new CustomEvent("delta:diagnostic-next"));
  });
  // Called with an Event, which must not land in `attempt`.
  window.addEventListener("delta:practice-state-changed", () => refresh());

  // Switching elsewhere hides #page-diagnostic through app.js; the workspace
  // goes home with it, so Practice renders whether or not the placement is
  // finished. Coming back re-claims it (see refresh → render → syncWorkspace).
  /* Leaving the page cancels the retry chain. A pending refresh that lands
     after app.js has moved the workspace home would call render() →
     moveWorkspace() and haul it back under a page nobody is looking at.
     Bumping the generation invalidates the in-flight one as well as the
     scheduled one. */
  const leave = () => {
    clearTimeout(retryTimer);
    retryTimer = null;
    generation += 1;
    syncWorkspace();
  };

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
  return { refresh, leave, renderStartButton, isRunning: () => running };
})();
window.DiagnosticPage = DiagnosticPage;
if (!document.getElementById("page-diagnostic")?.classList.contains("hidden")) {
  DiagnosticPage.refresh();
}
