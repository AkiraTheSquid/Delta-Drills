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
   🔴 THE PRACTICE TAB IS LOCKED AGAIN — DELIBERATELY, AND DIFFERENTLY.
   Seth, 2026-08-23: "whenever you're jumping in between the placement test and
   the practice ... it has a tendency to think that you are in placement test
   mode whenever you click on the practice, which is not the case. I think you
   should make it such that whenever you do the placement test, it locks you out
   of the practice. But the reverse is not true."

   He is describing a real thing, not a misreading. Leaving mid-probe sent the
   workspace home to #page-practice with the PROBE still in it — same editor,
   same "I don't know yet" button, the placement's clock ticking on it — and
   PracticeAPI.currentQuestion is a single global, so there was no second
   question to put there. Practice WAS the placement test, wearing the Practice
   tab's name. A block that starts from there would then be built on a probe.

   So an active placement takes the Practice tab out of reach, and the lock is
   the fix for the exact three failures the old one caused:
     - it is VISIBLE — `.tab:disabled` is styled now (styles/layout.css), and
       the tab carries a `title` saying why. The old lock had no style at all,
       so the tab looked normal and silently ate the click;
     - it is ROUTED — app.js `switchTab` sends "practice" to "diagnostic" while
       the lock is on, so [data-goto-tab] and a restored tab name land on the
       page that explains itself instead of on a hidden one;
     - it is RELEASED by the same status read that set it, on every
       `delta:practice-state-changed`, so finishing (or abandoning, from the
       backend's side) unlocks without a reload.

   NOT the reverse. The Placement tab stays reachable from a running practice
   session — that page is where you read what the test is and choose to start
   it, and taking that away is how you get a learner who cannot find out. */
const DiagnosticPage = (() => {
  const byId = (id) => document.getElementById(id);
  let running = false;

  /* The one writer of the lock. `disabled` is what actually stops the click
     (app.js binds the handler to the button, and a disabled button fires no
     click event); `aria-disabled` and the title are what make it legible.
     Called with `false` on load, so a build that latched the flag and a
     placement that ended while the tab was closed both clear on the next
     status read rather than inheriting a dead Practice tab. */
  const LOCK_WHY =
    "Finish the placement test first — it uses the same workspace, so Practice " +
    "would show you the placement question.";
  const setPracticeLock = (locked) => {
    document.querySelectorAll('.tab[data-tab="practice"]').forEach((tab) => {
      tab.disabled = !!locked;
      if (locked) {
        tab.setAttribute("aria-disabled", "true");
        tab.title = LOCK_WHY;
      } else {
        tab.removeAttribute("aria-disabled");
        tab.removeAttribute("title");
      }
    });
    /* Locking the tab the learner is standing on would leave them there with
       no way off. Cannot normally happen — the test starts from this page —
       but a status that turns active underneath an open Practice tab (another
       device, a resumed placement) is exactly the case the old code got wrong
       by hiding #page-practice and leaving them on it. */
    /* 🔴 `switchTab` is a top-level `const` in app.js, so it is NOT
       `window.switchTab` — a classic script's top-level const never becomes a
       window property, and `window.switchTab?.()` would have been a silent
       no-op forever. Same trap this file already documents for `PracticeAPI`.
       app.js is loaded before this file, so the script-scope binding is
       there; the typeof guard is for a page that loads one without the
       other. */
    if (locked && document.querySelector(".tab.active")?.dataset.tab === "practice" &&
        typeof switchTab === "function") {
      switchTab("diagnostic");
    }
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

    /* 🔴 A PROBE ON SCREEN, not merely a placement in progress. Self-review
       caught this the moment the overview started hiding on `running` alone:
       `#diagnostic-practice-btn` ("Load next placement question") lives INSIDE
       #diagnostic-overview, and it is shown exactly when the test is active and
       no probe is up — which was now also exactly when the overview was hidden.
       A learner between probes, or one whose first probe failed to load, got a
       blank page with the workspace hosted over it and nothing to click.

       So `hosted` asks for the question too. With no probe the workspace goes
       home to the (hidden) practice page and this one renders as itself: the
       status line, the progress bar and the button that loads the next probe.
       `PracticeAPI` is a top-level const, not a window property — this file
       documents that trap twice already. */
    const _api = typeof PracticeAPI !== "undefined" ? PracticeAPI : window.PracticeAPI;
    const probeOnScreen = !!_api?.currentQuestion?.diagnostic_active;
    const hosted = running && diagnosticOnScreen() && probeOnScreen;
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

  /* `running` and the Practice lock are the SAME fact — an active placement
     owns the workspace, and the lock is what stops a second tab claiming it —
     so they are written together and there is no path that sets one without
     the other. Lock last, and after `syncWorkspace`: it can switch tabs, and
     switching tabs re-enters this file through app.js. */
  const moveWorkspace = (active) => {
    running = !!active;
    syncWorkspace();
    setPracticeLock(running);
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

  /* How long the test will take, said BEFORE it starts. Seth, 2026-08-23:
     "it should also make it more clear to the user the amount of time that it
     will approximately take before they jump into the placement test."

     🔴 A RANGE, NOT A NUMBER, and the range is the backend's. `budget` is a
     CEILING and `min_probes` the earliest possible finish — most runs stop
     short — so a single figure is wrong for almost every learner, and the two
     ends are exactly the two the estimator already publishes. The answer clock
     is the only part that is bounded (`PLACEMENT_ANSWER_SECS`, 2:00 a probe);
     reviewing a graded probe is untimed, which is why this says "about" and
     leads with the question count, the part that is actually fixed.

     Read from the status payload every time rather than cached at load: the
     budget is a server-side policy and this page is the only thing that quotes
     it. */
  const renderLength = (status) => {
    const el = byId("placement-length");
    if (!el || !status) return;
    const secs = Number(window.PlacementTimer?.secondsPerQuestion?.()) || 120;
    const budget = Math.max(1, Number(status.budget) || 14);
    const minProbes = Math.min(budget, Math.max(0, Number(status.min_probes) || 0));
    const mins = (n) => Math.max(1, Math.round((n * secs) / 60));
    const each = `${String(Math.floor(secs / 60)).padStart(2, "0")}:${String(secs % 60).padStart(2, "0")}`;
    /* Without a usable floor there is only a ceiling, and saying "about 28
       minutes" for a test that usually ends less than half way through is the
       overclaim this whole line exists to avoid. */
    const span = minProbes > 0 && minProbes < budget
      ? `about ${mins(minProbes)}–${mins(budget)} minutes`
      : `up to about ${mins(budget)} minutes`;
    const count = minProbes > 0 && minProbes < budget
      ? `${minProbes}–${budget} questions`
      : `up to ${budget} questions`;
    el.innerHTML =
      `Plan for <strong>${span}</strong> — ${count}, <strong>${each}</strong> each. ` +
      `It stops as soon as it is confident, so most runs are shorter than the top of that range.`;
  };

  /* What the notch tab says while a probe is up. The placement page's own
     status line and progress bar are hidden the moment the workspace is hosted
     here (styles/practice/diagnostic.css) — a probe on screen IS the practice
     screen — so this is where the count goes instead: the tab's tooltip and its
     screen-reader phase, which is the same surface a practice session uses to
     say which phase it is in. Empty when nothing is running, so notch-menu.js
     can fall back to the session's own words. */
  let lastStatus = null;
  const progressLabel = () => {
    if (!running || !lastStatus) return "";
    const budget = Math.max(1, Number(lastStatus.budget) || 14);
    const done = Math.min(budget, Math.max(0, Number(lastStatus.probes_done) || 0));
    return `Placement question ${done + 1} of at most ${budget}`;
  };

  const render = (status) => {
    lastStatus = status || null;
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
    renderLength(status);
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
  /* And re-place the workspace IMMEDIATELY on the same event. `refresh()` is a
     network round trip, and the fact that decides where the workspace belongs —
     is a probe on screen — is already true locally by the time this fires.
     Waiting for the status call meant the first probe rendered under the
     placement page's own header card for as long as the request took, then
     jumped. `syncWorkspace` is idempotent and re-parents only when the node is
     in the wrong home, so running it twice per event costs nothing. */
  window.addEventListener("delta:practice-state-changed", () => syncWorkspace());

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

  /* Start unlocked. Nothing is known about the placement until the first
     status read, and a Practice tab that is dead until the network answers is
     worse than one that locks a moment later — the lock exists to stop a probe
     being mistaken for practice, and there is no probe on screen yet. */
  setPracticeLock(false);
  return { refresh, leave, renderStartButton, isRunning: () => running, progressLabel };
})();
window.DiagnosticPage = DiagnosticPage;
if (!document.getElementById("page-diagnostic")?.classList.contains("hidden")) {
  DiagnosticPage.refresh();
}
