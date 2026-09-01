/* THE PLACEMENT TEST, on #page-placement.

   The route, the ids and the backend endpoints all still say `diagnostic` —
   only the learner-facing words changed, and then where they are drawn.

   🔴 IT HAS ITS OWN PAGE AGAIN (Seth, 2026-09-01: "we can probably just keep
   the interface for the diagnostic ... separate, and then it only gets
   displayed whenever you click on the drop-down one and you go to it
   specifically"). It was merged into the Learner Home on 2026-08-24 and that
   merge is what is undone here. The page is #page-placement — NOT the retired
   #page-diagnostic id, which nothing may look for again.

   WHAT IS ON WHICH PAGE, because that split is the whole design:

     #page-placement — the overview card, its results, and the workspace host.
     Everything about a test taken once. It has no tab in the strip; the account
     menu row, the welcome fork's right arm and /diagnostic reach it.

     #page-practice — the readiness dial, the clock picker and the AREA BARS.
     The screen a learner opens every day. This file still writes the bars from
     the same status payload (renderAreas), because they are a daily readout of
     what the test measured, not part of the offer to sit it.

   WHAT THE SPLIT BRINGS BACK:

     The practice/placement redirect, in app.js. There is one
     `.practice-container` and one `PracticeAPI.currentQuestion`, so with two
     pages a route to Practice mid-probe would drag the probe onto the Learner
     Home and render it under that page's name. Asking for Practice while
     `running` lands on the placement instead. It is a redirect, NOT the old
     tab-disabling `setPracticeLock` — a disabled tab has no :disabled style, so
     it looks live and eats the click.

     The finish. A placement that completes releases the workspace and leaves
     the learner on this page's results. It never silently starts ordinary
     practice; continuing from here is a separate learner action.

   WHAT IT KEEPS:

     The workspace host. `.practice-container` moves into
     #diagnostic-workspace-host while a probe is on screen and back to
     #practice-workspace-home when it is not. `syncWorkspace` is idempotent —
     re-inserting a node that is already in place would tear a live editor out
     of the document and back in.

     The on-screen half of `hosted`: a status call that lands while the learner
     is reading some other tab must not haul the workspace anywhere. */
const DiagnosticPage = (() => {
  const byId = (id) => document.getElementById(id);
  let running = false;

  /* 🔴 `setPracticeLock` USED TO BE HERE, and it is deleted rather than left
     unwired. It disabled `.tab[data-tab="practice"]` for the length of a
     placement, because Practice and the placement test shared one editor and
     one `PracticeAPI.currentQuestion` — so opening Practice mid-test showed the
     PROBE under the Practice tab's name (Seth, 2026-08-23).

     They are ONE TAB now (Seth, 2026-08-24), and a tab cannot be locked against
     itself: the probe and the practice question are the same workspace on the
     same page, which is what the lock was trying to fake. app.js lost the
     matching `tabName === "practice" -> "diagnostic"` redirect in the same
     change. `moveWorkspace` still writes `running`, which is what
     `isRunning()` and `progressLabel()` read. */

  /* THE PAGE THE PLACEMENT IS ON. #page-diagnostic until 2026-08-24, then the
     Learner Home, and #page-placement since 2026-09-01. The name of this helper
     has outlived all three because what it asks has not changed: is the surface
     that owns the placement on screen? */
  const diagnosticOnScreen = () =>
    byId("page-placement")?.classList.contains("hidden") === false;

  // Mirrors app.js's own check: the practice page owes its setup panel only
  // when nothing holds the question. Releasing the workspace must not stomp a
  // running/paused session or a lesson back to the setup screen.
  /* A block is not the only thing that owns the question. The `?lesson=<kc>`
     KC drill (practice/lessons.js -> practice/kc-practice.js) is DELIBERATELY
     sessionless — "No session quota or timer", its own comment — and it clears
     `session-idle` itself so the learner can see what it serves. */
  const practiceHoldsQuestion = () =>
    document.body.classList.contains("lesson-mode") ||
    window.KcPractice?.isActive?.() === true ||
    (typeof PracticeSession !== "undefined" &&
      (PracticeSession.isActive?.() || PracticeSession.hasPausedSession?.()));

  // Idempotent on purpose: re-inserting a node that is already in place tears
  // its subtree out of the document and back in, which would reload any live
  // editor/iframe inside the workspace. Only move when it is in the wrong home.
  const syncWorkspace = () => {
    /* 🔴 TWO PAGES, TWO NAMES — and they are different elements again. `page`
       is the placement page, which carries `.diagnostic-running`
       (styles/practice/diagnostic.css hangs the overview's visibility off it);
       `practicePage` is the Learner Home, which carries `.session-idle`
       (timer.js owns that one). They were the same element between 2026-08-24
       and 2026-09-01 and the two names were kept through the merge precisely so
       this split would not have to re-derive which class belongs to whom. */
    const page = byId("page-placement");
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
    /* Read BEFORE the write below: the idle screen is restored only for a
       workspace that is coming HOME, and "was it hosted a moment ago" is the
       only record of that. */
    const wasHosted = page.classList.contains("diagnostic-running");
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
    /* 🔴 ONLY ON THE WAY HOME. This line puts the Practice page back to its
       idle screen after a placement hands the workspace back — that is the
       whole of its job. It used to run on EVERY call, and `syncWorkspace` is
       wired to `delta:practice-state-changed`, which the felt-difficulty
       rating fires (api.js sendFeedback). So a learner solving a question
       outside a block — the `?lesson=<kc>` KC drill is exactly that — rated
       the problem they had just solved and watched the question disappear
       under the readiness dial. Seth, 2026-08-23: "after you solve a problem
       ... it essentially goes back to the original page that exits out of what
       you're currently doing."

       Nothing is lost by the gate: every OTHER way a block ends already writes
       the class itself (timer.js `pause()` and `finish()` both add it, `start()`
       and `_resumeCore()` both remove it), so this was never the writer for any
       of them — it was a placement-release step running on every state change. */
    if (wasHosted) {
      practicePage.classList.toggle("session-idle", !practiceHoldsQuestion());
    }
  };

  /* 🔴 THE CARD IS NOT ON THE LEARNER HOME AT ALL ANY MORE (Seth, 2026-09-01:
     keep the placement interface "separate ... it only gets displayed whenever
     you click on the drop-down one and you go to it specifically"). It lives on
     #page-placement, which a learner reaches on purpose.

     THAT REPLACED A VISIBILITY RULE, and deleting the rule is the point. From
     2026-08-31 the card sat on the Learner Home and was hidden once the
     placement was COMPLETE (`#page-practice.placement-taken`), with a
     `forceShow` override the account-menu row set for one visit so a learner
     could read their results and press Retake. Both are gone: a page you
     navigated to must show what it is for, every time, in every state — a
     placement page that hides its own card on arrival is the "nothing to click"
     dead end that override existed to patch.

     WHAT SURVIVES is the sentence under the area bars, because it is the one
     thing on the Learner Home that pointed AT the card. It said "below", and
     below is not where the card is from any page. */
  let lastStatus = null;

  const AREAS_NOTE_UNTAKEN =
    "Each bar is this area's estimated readiness. Take the placement test from " +
    "Account and Settings to measure them instead of assuming them.";
  const AREAS_NOTE_TAKEN =
    "Each bar is this area's estimated readiness, measured by your placement " +
    "test. Retake it any time from Account and Settings.";

  const syncOverviewVisibility = (status) => {
    const taken = !!status?.completed_at && !status.active;
    const note = byId("learner-areas-note");
    if (note) note.textContent = taken ? AREAS_NOTE_TAKEN : AREAS_NOTE_UNTAKEN;
  };

  /* `running` is the one fact this file publishes about a placement, and
     `syncWorkspace` is the only thing that acts on it. (There was a Practice
     tab lock written alongside it until the two tabs were merged — see the note
     above `diagnosticOnScreen`.) */
  const moveWorkspace = (active) => {
    running = !!active;
    syncWorkspace();
  };

  /* The ONE writer of the placement start button. events.js has its own reason
     to refresh it (the click handler, and the state-changed sweep) and used to
     carry a second copy of the label + visibility rules; two copies of a label
     is how a button flickers between two names on refresh. */
  /* THE ACCOUNT-MENU ROW SAYS THE SAME THING THIS BUTTON DOES.

     Seth, 2026-08-31: the retake has to be reachable from the account menu, not
     only from this card. The row is a route (account-menu.js scrolls the card up
     and flashes the button — it never calls /diagnostic/start), so the only
     thing it needs from here is its LABEL, and that label has to come from the
     same status payload the button reads. Two independent copies of "Take" vs
     "Retake" is how a menu ends up offering a retake of a test nobody has sat.

     🔴 THREE STATES, not the button's two. The button HIDES while a placement
     is active — the card shows "Load next placement question" instead — but a
     menu row that vanishes mid-test is exactly the disappearance the row exists
     to fix ("that way you can ALWAYS retake"). So the row stays and says
     Resume, which is what clicking it does: route back to the live test. */
  const syncMenuLabel = (status) => {
    const el = byId("account-menu-placement-label");
    if (!el) return;
    el.textContent = status?.active
      ? "Resume the placement test"
      : status?.completed_at
        ? "Retake the placement test"
        : "Take the placement test";
  };

  const renderStartButton = (status, el = byId("placement-start-btn")) => {
    // Before the early return: the row is on the topbar, on every page, and it
    // is labelled whether or not this card is in the document.
    syncMenuLabel(status);
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
      ? `about <strong>${mins(minProbes)}–${mins(budget)} min</strong>`
      : `about <strong>${mins(budget)} min</strong>`;
    const count = minProbes > 0 && minProbes < budget
      ? `<strong>${minProbes}–${budget}</strong> questions`
      : `<strong>${budget}</strong> questions`;
    /* Chips, not sentences: `.placement-chip` (styles/practice/diagnostic.css)
       is the unit the whole page states a fact in now. The "stops early" chip
       is what keeps the range honest — the top of it is a ceiling almost no
       run reaches — and it is four words instead of a clause. */
    el.innerHTML =
      `<span class="placement-chip">${count}</span>` +
      `<span class="placement-chip"><strong>${each}</strong> each</span>` +
      `<span class="placement-chip">${span}</span>` +
      `<span class="placement-chip">stops early when confident</span>`;
  };

  /* What the notch tab says while a probe is up. The placement page's own
     status line and progress bar are hidden the moment the workspace is hosted
     here (styles/practice/diagnostic.css) — a probe on screen IS the practice
     screen — so this is where the count goes instead: the tab's tooltip and its
     screen-reader phase, which is the same surface a practice session uses to
     say which phase it is in. Empty when nothing is running, so notch-menu.js
     can fall back to the session's own words. */
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
    const skipEl = byId("placement-skip-btn");
    const resultsEl = byId("diagnostic-results");

    /* 🔴 "SKIP FOR NOW" IS OFFERED ONLY WHEN THERE IS SOMETHING TO SKIP.

       While a placement is ACTIVE the backend's next-question endpoint serves
       PROBES — there is no practice stream running beside a live test — so
       "go to practice" mid-test cannot deliver what it says. Measured: with the
       test open, leaving to the Learner Home fetched question 486 with
       `diagnostic_active: true`, restarted the probe clock on it and left it
       ticking behind the idle surface, where it would have expired into a
       recorded miss on a probe the learner never saw.

       So the door is on the two states it is honest in: not started (the
       welcome fork's arm lands a first-time learner here and "no thanks" has to
       work) and completed (they came to read results). Mid-test the card offers
       "Load next placement question", which is the only thing that is true. */
    const showSkip = !status?.active;
    if (!status) {
      if (statusEl) statusEl.textContent = "Sign in to take the placement test.";
      priorEl?.classList.add("hidden");
      continueEl?.classList.add("hidden");
      startEl?.classList.add("hidden");
      // A signed-out visitor gets no test and must still be able to leave.
      skipEl?.classList.remove("hidden");
      resultsEl?.classList.add("hidden");
      // No status is not "you finished one": a stale "Retake the placement
      // test" in the account menu is a claim about this learner's record.
      syncMenuLabel(null);
      renderProgress(null);
      window.PlacementResults?.renderAreas([]);
      moveWorkspace(false);
      // Unknown record ≠ taken: keep the card up for a signed-out visitor.
      syncOverviewVisibility(null);
      return;
    }

    const done = Number(status.probes_done) || 0;
    const budget = Number(status.budget) || 14;
    if (statusEl) {
      statusEl.textContent = status.active
        ? `In progress · ${done} of at most ${budget} answered`
        : status.completed_at
          ? `Complete · ${done} questions`
          : "Not started";
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
    skipEl?.classList.toggle("hidden", !showSkip);
    renderStartButton(status, startEl);
    /* This file owns WHETHER the results card shows; placement-results.js owns
       what is in it. Fill before unhiding so the card never flashes the shape
       of the previous placement, and fill only when it is actually going to be
       shown — a mid-test status carries live estimates that would read as a
       finished result. */
    /* 🔴 THE AREA BARS ARE NOT PART OF THE RESULTS CARD ANY MORE. They used to
       be drawn only inside `render(status)` below, which runs only when a
       placement is COMPLETE — so the one screen a learner opens every day named
       nothing and reported "0% ready" (Seth, 2026-08-24: "it should display the
       information about einops, numpy, and einsum to be learned").

       `areas` comes back populated from the very first status call on a new
       account — theta is the prior, `probes` is 0 — and the rows carry their own
       "not probed" marking and ± band, so drawing them early claims nothing the
       test has not measured. Drawn on EVERY status, including a mid-test one:
       watching an area move as you answer is the point of having it there. */
    window.PlacementResults?.renderAreas(Array.isArray(status.areas) ? status.areas : []);

    const showResults = !!status.completed_at && !status.active;
    if (showResults) {
      window.PlacementTimer?.stop?.();
      window.PlacementResults?.render(status);
    }
    resultsEl?.classList.toggle("hidden", !showResults);
    moveWorkspace(!!status.active);
    syncOverviewVisibility(status);
    /* Completion stays here. The completed-state branch stops the probe clock;
       `moveWorkspace(false)` releases the final probe and reveals the result
       card above. Do not `switchTab("practice")`: that made
       a timeout look like it had aborted placement, and it hid the post-submit
       controls before the learner could understand why no next probe existed. */
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

  // Switching elsewhere hides #page-practice through app.js; the workspace
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
     workspace follows the pages even if some other route (a solo deep link, a
     future nav) shows or hides one without telling us.

     🔴 BOTH PAGES, since the split. `hosted` is a function of the PLACEMENT
     page being on screen, and the workspace's home is on the PRACTICE page —
     watching only one of them means a change to the other moves nothing until
     something else happens to call syncWorkspace. */
  ["page-placement", "page-practice"].forEach((id) => {
    const watched = byId(id);
    if (watched && typeof MutationObserver === "function") {
      new MutationObserver(() => syncWorkspace())
        .observe(watched, { attributes: true, attributeFilter: ["class"] });
    }
  });

  return { refresh, leave, renderStartButton, isRunning: () => running, progressLabel };
})();
window.DiagnosticPage = DiagnosticPage;

/* 🔴 THE MODE IS NOT DECIDED YET WHEN THIS FILE RUNS.

   `PracticeAPI.diagnosticStatus()` returns null — not a status, not a failure,
   NULL — whenever `practiceMode !== "backend"`, and `practiceMode` starts as
   "local" and is only set by `detectPracticeMode()` inside `initPractice()`,
   which lives in practice/init.js and is loaded AFTER this file. Refreshing at
   parse time therefore asked a question whose answer was already fixed: null,
   which `render(null)` paints as "Sign in to take the placement test." with the
   area bars and every button hidden — at a signed-in learner, on a backend that
   was answering 200 the whole time.

   This did not bite while the placement lived on its own tab, because the boot
   guard read #page-diagnostic and that page is hidden at load, so this never
   ran. The 2026-08-24 merge pointed the guard at #page-practice, which IS the
   landing page — so the broken call became the FIRST thing every visitor got.

   So: wait for the signal init.js fires once the mode is real. The interim copy
   is the "checking" one, never the signed-out one, because at this point we do
   not yet know which is true. The timer is for a page that loads this without
   init.js (a solo route, a test harness) — it must still end up rendering
   something rather than sitting on "Checking…" forever. */
(function bootDiagnosticPage() {
  /* 🔴 EITHER PAGE. The guard asks "is a surface this file writes on screen?",
     and since 2026-09-01 there are two: the Learner Home (area bars) and
     #page-placement (the card). Left reading only the Home, a learner who
     landed straight on the placement — /diagnostic, or the welcome fork's right
     arm on a first load — got no boot refresh at all and a card stuck on
     "Loading placement status…" until they navigated away and back. */
  const _onScreen = ["page-practice", "page-placement"].some(
    (id) => document.getElementById(id)?.classList.contains("hidden") === false
  );
  if (!_onScreen) return;
  const statusEl = document.getElementById("diagnostic-status");
  if (statusEl) statusEl.textContent = "Checking your placement status…";
  /* 🔴 THE FALLBACK MUST NOT LATCH. An earlier version set one `fired` flag from
     both paths, so a slow `DDGuest.ensure()` — it is a network round trip, and
     eight seconds is not impossible on a cold backend — let the timer refresh
     FIRST, while `practiceMode` was still "local". That paints the signed-out
     copy, and latching meant the real `delta:practice-mode-ready` that arrived a
     second later was dropped: the page stayed wrong until something else
     refreshed it. The event is authoritative and always gets its turn; only IT
     may close the door. The double refresh that costs is harmless — `refresh()`
     bumps `generation`, so the later call is the one that paints. */
  let ready = false;
  const go = () => {
    if (ready) return;
    ready = true;
    DiagnosticPage.refresh();
  };
  window.addEventListener("delta:practice-mode-ready", go, { once: true });
  // Already past us (another entry point called initPractice first).
  if (window.DDPracticeModeReady === true) go();
  else setTimeout(() => { if (!ready) DiagnosticPage.refresh(); }, 8000);
})();
