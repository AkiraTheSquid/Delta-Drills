/* ================================================================
   COLAB_FOCUS.JS — show one problem, and skin the page.

   WHAT MAKES THIS POSSIBLE
     The generated notebooks are nbformat 4.5 and every cell carries a
     `metadata.id` we mint, which Colab renders as the DOM id `cell-<id>`:

       dd-setup            imports + DD_LESSON_ID
       dd-lesson-<id>      the lesson header
       dd-kp-<slug>        one teaching cell per knowledge component
       dd-q<n>-example     a solved example, ABOVE the problem  ─┐
       dd-q<n>-example-code  …and its answer                     │
       dd-q<n>             a problem's header                    ├─ the group
       dd-q<n>-hints       its hint block                        │  focus keeps
       dd-q<n>-code        its answer cell                      ─┘

     So no notebook change was needed to hook onto — the anchors have been there
     since the panel needed them for jumping. This file only adds classes; the
     hiding itself is CSS (`colab_dd.css`).

   WHY THE EXAMPLE IS NAMED AFTER THE PROBLEM, NOT AFTER ITSELF
     A stage-2 problem is a PAIR: a solved example you read, then a problem that
     is the same move on different specifics, which you solve. Both halves have
     to be on screen together, and the example is built from a different bank
     question — so left to itself it would carry that question's number and
     focus would hide it the moment the learner was routed to its twin.

     The generator therefore mints it as `dd-q<problem>-example`: the anchor
     names the problem the scaffold BELONGS TO. Nothing here had to change for
     that to work, because `problemOf` already groups on the number, and that is
     the point — the pairing is a fact the notebook states, not one this file
     infers. The tempting alternative is a DOM heuristic ("also keep the cells
     immediately above the target"), which is correct until the day a segment's
     prose or the previous problem's answer happens to sit there, and then it
     puts unrelated content under a heading the learner has been told to trust.
     Explicit beats adjacent.

   WHICH PROBLEM IS "THE" PROBLEM
     The URL says. The app's "Open in Colab ↗" builds
     `…ipynb#scrollTo=dd-q123`, so the fragment already names the target and no
     messaging between the app, the panel and this script is needed for the
     common case. Colab rewrites that fragment as the student moves around,
     which is fine: every cell in a focused group shares the group's number, so
     clicking within the problem re-resolves to the same problem.

   THE RULE THAT KEEPS THIS SAFE
     Hide nothing unless a target actually resolved to cells that exist. Focus
     with no match would blank the notebook, and a blank notebook is
     indistinguishable from a broken page — on ARENA's own 458 notebooks, which
     are nbformat 4.2 with no ids at all, that is the DEFAULT case. There, focus
     stays inert and says so.
   ================================================================ */

(() => {
  const CELL = "div.cell";
  const STORE_KEY = "dd_colab_view";
  // `gemini` is "let Colab's AI complete your code", so OFF is the default and
  // the checkbox is unticked — see `pushGemini`.
  const DEFAULTS = { theme: true, focus: true, gemini: false, collapsed: false };

  // Cells that focus never hides. The checker cell defines `dd_check`, so
  // hiding it leaves a problem that cannot be checked — the kind of break that
  // is silent until something below raises NameError and reads as broken
  // starter code rather than as a missing prerequisite. The setup cell is here
  // for the same reason it always was, even though the theme now hides it: the
  // two mechanisms are separate, and `dd-out-of-focus` is the one that would
  // also take it away in an unthemed notebook.
  const ALWAYS_VISIBLE = /^dd-(?:setup|checker)(?:$|[^a-z0-9])/i;

  // The setup cell, which is exempt from focus (above) and then hidden by the
  // theme anyway (`colab_dd.css`). It assigns `DD_LESSON_ID` and nothing else;
  // `colab.js`'s `identify` reads that off the rendered text, and text in a
  // `display: none` cell is still text. Nothing needs it RUN, so the learner
  // does not need to see a cell whose only job is to be read by a program.
  const SETUP = /^dd-setup(?:$|[^a-z0-9])/i;

  // The worked half of a stage-2 pair. Styling only: these cells are already in
  // focus with their problem (the anchor names it), but nothing on screen would
  // otherwise say that the code cell holding a full solution is the one the
  // learner is meant to READ. Two prompts and two code cells in a row, all
  // equally styled, is an invitation to start solving the example.
  const EXAMPLE = /^dd-q(\d+)-example(?:$|[^0-9])/i;

  // A problem's answer cell. Hidden until the learner submits — see `reveal`.
  // There is no toggle for this and there should not be: an "always show
  // solutions" switch is a switch for turning the exercise off.
  const SOLUTION = /^dd-q(\d+)-solution$/i;

  // A problem's checker cell, and the line `dd_check` prints into it.
  //
  // This is the whole reporting channel. The notebook has no way to call the
  // app — a beacon would need a token pasted into the notebook, and a cell's
  // rich output is sandboxed away from the page anyway. But stdout renders as
  // plain text in THIS document, so the summary line the learner reads is also
  // the line this script reads. `scripts/watch.py` grades the pair together so
  // the wording and this pattern cannot drift apart.
  const CHECK = /^dd-q(\d+)-check$/i;
  const RESULT = /(✅|❌) Problem (\d+) — /g;

  let settings = { ...DEFAULTS };
  let lastReport = { target: null, inFocus: 0, total: 0 };

  // Problem numbers whose solution has been unlocked this page-load. Not
  // persisted on purpose: reopening a notebook is how you get a clean run at a
  // problem, and a remembered unlock would hand you the answer before you
  // started.
  const revealed = new Set();

  // problem -> the last result line seen for it, so a re-render does not
  // re-report a grade the app already has.
  const lastSeen = new Map();
  let seeded = false;

  // ── Reading the notebook ───────────────────────────────────────────

  /** The `metadata.id` behind a rendered cell, or "" for an un-id'd one. */
  function cellAnchor(cell) {
    const id = cell.id || "";
    if (id.startsWith("cell-")) return id.slice(5);
    const inner = cell.querySelector('[id^="cell-dd-"]');
    return inner ? inner.id.slice(5) : "";
  }

  /**
   * The problem number an anchor belongs to, or null.
   *
   * The trailing boundary matters: a bare prefix test would put `dd-q123-code`
   * AND `dd-q12` in the same group, so problem 12 would drag problem 123's
   * cells on screen with it.
   */
  function problemOf(anchor) {
    const m = /^dd-q(\d+)(?:$|[^0-9])/.exec(anchor || "");
    return m ? m[1] : null;
  }

  /** The problem the URL is pointing at, or null. */
  function targetProblem() {
    const hash = decodeURIComponent(location.hash || "").replace(/^#/, "");
    const value = /(?:^|[?&])scrollTo=([^&]+)/.exec(hash);
    const raw = value ? value[1] : hash;
    return problemOf(String(raw).replace(/^cell-/, ""));
  }

  /**
   * The problem in focus, which is NOT the same as the problem in the URL.
   *
   * Colab rewrites `#scrollTo=` to whatever cell is at the top of the viewport
   * as you scroll — that is its own deep-linking feature, not something we ask
   * for. So scrolling up to re-read the checker, or down past the last cell of
   * the problem, rewrites the fragment to `dd-checker` or `dd-setup`, which
   * belong to no problem. Read straight, that says "no target", focus switches
   * off mid-thought and the whole notebook unfolds under the cursor — the
   * learner scrolled two lines and lost their place.
   *
   * So the fragment is a way to CHANGE the target, not the target itself: a
   * fragment naming a different problem switches to it, and anything else
   * leaves the current one alone. The panel routes by rewriting the fragment,
   * so the one path that must keep working still does.
   */
  let sticky = null;
  function focusTarget() {
    const fromUrl = targetProblem();
    if (fromUrl) sticky = fromUrl;
    return sticky;
  }

  // ── Applying it ────────────────────────────────────────────────────

  /**
   * The verdict `dd_check` printed in a cell, or null if it has not run.
   *
   * `textContent`, not `innerText`: this runs on every mutation of a notebook
   * that mutates constantly, and innerText forces layout per cell. The cell's
   * SOURCE is in there too (`dd_check(480)`), which is exactly why the pattern
   * requires the printed prefix — source alone can never match.
   *
   * Last match wins: a cell re-run replaces its output, but Colab can have both
   * the old and the new node in the DOM for a frame.
   */
  function verdictIn(cell) {
    const text = cell.textContent || "";
    if (!text) return null;
    RESULT.lastIndex = 0;
    let found = null;
    let m;
    while ((m = RESULT.exec(text)) !== null) found = m;
    return found ? { problem: found[2], correct: found[1] === "✅", line: found[0] } : null;
  }

  /**
   * Report every check that has finished since the last pass.
   *
   * Two things happen to a result: the answer cell for that problem opens
   * (running the check IS the submit, in the notebook), and the panel is told,
   * so the rail marks the problem right or wrong without the learner clicking
   * a verdict for something they just measured.
   */
  function harvest(cells) {
    cells.forEach((cell) => {
      const anchor = cellAnchor(cell);
      const check = CHECK.exec(anchor);
      if (!check) return;
      const verdict = verdictIn(cell);
      if (!verdict || verdict.problem !== check[1]) return;
      if (lastSeen.get(check[1]) === verdict.line) return;
      lastSeen.set(check[1], verdict.line);
      // The first pass only records what is already on screen. A notebook
      // reopened with its outputs saved would otherwise replay every grade in
      // it the moment the page loads.
      if (!seeded) return;
      revealed.add(check[1]);
      report(check[1], verdict.correct);
    });
    seeded = true;
  }

  function report(problem, correct) {
    try {
      chrome.runtime.sendMessage(
        { type: "dd:check-result", problem, correct },
        () => void chrome.runtime.lastError, // no panel open is not an error
      );
    } catch (_) {
      /* the extension can be mid-reload; the notebook still works */
    }
  }

  /**
   * Tell the MAIN-world script whether Gemini may complete the learner's code.
   *
   * Colab turns "Show AI-powered inline completions" on by default, and on a
   * Delta Drills notebook what it completes is the answer — grey shadow text
   * appearing ahead of the caret, on a problem the course has just decided the
   * learner cannot do yet.
   *
   * The suppression itself cannot happen here. A content script is in the
   * isolated world and cannot see `window.monaco`, so `colab_no_ai.js` runs in
   * the MAIN world and this file only says what the policy is. It cannot be CSS
   * either: hiding the ghost text leaves Tab accepting a suggestion the learner
   * cannot see, which is strictly worse than showing it.
   *
   * Two events rather than one carrying a flag: their `detail` would be an
   * object minted in this world, and the event NAME is a string, which always
   * crosses. Dispatched only on a real change, because this runs on every
   * mutation of a notebook that mutates constantly.
   *
   * Recorded only once the dispatch has actually happened, and never allowed to
   * throw. `apply` runs inside the boot try/catch, so an exception here does not
   * surface as an error — it aborts the rest of the pass and sends `start` round
   * a second time, leaving two MutationObservers and two hashchange listeners on
   * a notebook that looks fine. Failing quietly and retrying on the next pass is
   * the only version of this that cannot do that.
   */
  let geminiPushed = null;
  function pushGemini(suppress) {
    if (geminiPushed === suppress) return;
    try {
      document.dispatchEvent(new CustomEvent(suppress ? "dd:gemini-off" : "dd:gemini-on"));
      geminiPushed = suppress;
    } catch (_) {
      /* no CustomEvent here (a test host, or mid-reload); retry next pass */
    }
  }

  function apply() {
    const root = document.documentElement;
    root.classList.toggle("dd-theme", Boolean(settings.theme));
    // No toggle. The answer is hidden until the learner has submitted, full
    // stop; `reveal` and a finished check are the only two ways past it.
    root.classList.add("dd-hide-solutions");

    const cells = Array.from(document.querySelectorAll(CELL));
    // Observe the fragment on EVERY pass, including while focus is off, and only
    // then decide whether to use it. Gating the observation instead would let
    // the sticky value go stale: route to problem 118 with focus off, scroll (so
    // Colab rewrites the fragment to a cell belonging to no problem), turn focus
    // back on — and it would hide 118 to show 117, the problem before last.
    const seen = focusTarget();
    const target = settings.focus ? seen : null;

    // Before tagging, not after: a check that just finished unlocks its answer
    // cell, and the tagging pass below is what puts that on screen.
    harvest(cells);

    // Tag first, count second, and only then decide whether to hide anything.
    let inFocus = 0;
    let ourCells = 0;
    cells.forEach((cell) => {
      const anchor = cellAnchor(cell);
      if (anchor.startsWith("dd-")) ourCells += 1;
      const always = ALWAYS_VISIBLE.test(anchor);
      const mine = target !== null && problemOf(anchor) === target;
      if (mine) inFocus += 1;
      cell.classList.toggle("dd-always-visible", always);
      cell.classList.toggle("dd-setup-cell", SETUP.test(anchor));
      cell.classList.toggle("dd-out-of-focus", Boolean(target) && !mine && !always);
      cell.classList.toggle("dd-in-focus", mine);
      cell.classList.toggle("dd-example", EXAMPLE.test(anchor));
      const solution = SOLUTION.exec(anchor);
      cell.classList.toggle("dd-solution", Boolean(solution));
      cell.classList.toggle("dd-solution-shown", Boolean(solution) && revealed.has(solution[1]));
    });

    // The guard. Focus is only ON when the target resolved to at least one real
    // cell in THIS notebook — otherwise every cell would carry dd-out-of-focus
    // and the page would go blank.
    const live = Boolean(target) && inFocus > 0;
    root.classList.toggle("dd-focus", live);
    if (!live) {
      cells.forEach((cell) => cell.classList.remove("dd-out-of-focus"));
    }

    // Gemini is only suppressed on a notebook that is OURS — one carrying at
    // least one `dd-` anchor. The extension is installed for Delta Drills, but
    // the browser it is installed in opens other people's Colab notebooks, and
    // silently disabling a Google feature on those is not a thing an extension
    // should do for a tutor that is not even open. Same reasoning as the CSS:
    // nothing of ours applies until a cell of ours is on the page.
    const suppressAi = ourCells > 0 && !settings.gemini;
    root.classList.toggle("dd-no-ai", suppressAi);
    pushGemini(suppressAi);

    lastReport = { target, inFocus, total: cells.length };
    paintNote();
  }

  // ── Unlocking an answer ────────────────────────────────────────────

  /**
   * Show problem `n`'s solution cell, from now until the page reloads.
   *
   * The signal comes from the panel, on the click that records how it went —
   * "then and only then it shows you the solution … below what you typed".
   * A notebook cell cannot do this itself: Colab renders every cell's output in
   * a sandboxed iframe, so CSS or JS emitted by the notebook cannot reach a
   * sibling cell. Only a content script can, which is why this lives here and
   * not in `scripts/colab_grader.py`.
   *
   * Reached through `colab.js`'s message switch (one protocol surface), which
   * finds this on the shared isolated-world global.
   */
  function reveal(problem) {
    const n = String(problem == null ? "" : problem).replace(/^dd-q/, "");
    if (!/^\d+$/.test(n)) return { ok: false, reason: "bad-problem" };
    revealed.add(n);
    apply();
    const shown = document.querySelectorAll(".cell.dd-solution-shown").length;
    return { ok: true, problem: n, shown };
  }

  // `rescan` is `apply` under the name that says why anything outside this file
  // would call it: to re-read the notebook after something changed in it.
  window.__ddFocus = { reveal, rescan: () => apply() };

  // ── The toggle ─────────────────────────────────────────────────────

  let panel = null;

  function paintNote() {
    if (!panel) return;
    const note = panel.querySelector(".dd-note");
    if (!note) return;
    if (!settings.focus) {
      note.textContent = "Showing the whole notebook.";
      return;
    }
    if (!lastReport.target) {
      note.textContent = "No problem in the URL — open one from the app "
        + "(the link carries #scrollTo=dd-q…). Showing everything.";
      return;
    }
    if (!lastReport.inFocus) {
      note.textContent = `Problem ${lastReport.target} is not in this notebook. `
        + "Showing everything.";
      return;
    }
    note.textContent = `Problem ${lastReport.target}: `
      + `${lastReport.inFocus} of ${lastReport.total} cells.`;
  }

  function buildToggle() {
    if (panel || !document.body) return;
    panel = document.createElement("div");
    panel.id = "dd-colab-toggle";
    panel.innerHTML = `
      <button class="dd-handle" type="button" title="Collapse">Delta Drills ▾</button>
      <label class="dd-row"><input type="checkbox" data-dd="focus" /> Only this problem</label>
      <label class="dd-row"><input type="checkbox" data-dd="theme" /> Delta Drills theme</label>
      <label class="dd-row" title="Colab's Gemini writes the answer ahead of your caret in grey. Off while you are practising."><input type="checkbox" data-dd="gemini" /> Gemini autocomplete</label>
      <div class="dd-note"></div>
    `;
    panel.querySelectorAll("input[data-dd]").forEach((input) => {
      input.checked = Boolean(settings[input.dataset.dd]);
      input.addEventListener("change", () => {
        settings = { ...settings, [input.dataset.dd]: input.checked };
        save();
        apply();
      });
    });
    panel.querySelector(".dd-handle").addEventListener("click", () => {
      settings = { ...settings, collapsed: !settings.collapsed };
      save();
      paintCollapsed();
    });
    document.body.appendChild(panel);
    paintCollapsed();
    paintNote();
  }

  function paintCollapsed() {
    if (!panel) return;
    panel.classList.toggle("dd-collapsed", Boolean(settings.collapsed));
    const handle = panel.querySelector(".dd-handle");
    if (handle) handle.textContent = settings.collapsed ? "Delta Drills ▸" : "Delta Drills ▾";
  }

  function save() {
    try {
      chrome.storage.local.set({ [STORE_KEY]: settings });
    } catch (_) {
      /* storage can be unavailable mid-reload; the session still works */
    }
  }

  // ── Wiring ─────────────────────────────────────────────────────────

  function start() {
    buildToggle();
    apply();

    // Colab mounts cells long after load and re-renders them on expand, on
    // scroll and on run. Re-tagging on mutation is what keeps a cell from
    // appearing unstyled — debounced, because a running notebook mutates
    // constantly and this walks every cell.
    let queued = false;
    const observer = new MutationObserver(() => {
      if (queued) return;
      queued = true;
      setTimeout(() => {
        queued = false;
        buildToggle();
        apply();
      }, 200);
    });
    observer.observe(document.documentElement, { childList: true, subtree: true });

    // The app opens the next problem by changing only the fragment, which is
    // not a navigation — without this the notebook would keep showing the
    // previous problem.
    window.addEventListener("hashchange", apply);
  }

  try {
    chrome.storage.local.get(STORE_KEY, (stored) => {
      settings = { ...DEFAULTS, ...((stored && stored[STORE_KEY]) || {}) };
      start();
    });
  } catch (_) {
    start();
  }
})();
