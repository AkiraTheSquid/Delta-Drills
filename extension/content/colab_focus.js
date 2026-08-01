/* ================================================================
   COLAB_FOCUS.JS — show one problem, and skin the page.

   WHAT MAKES THIS POSSIBLE
     The generated notebooks are nbformat 4.5 and every cell carries a
     `metadata.id` we mint, which Colab renders as the DOM id `cell-<id>`:

       dd-setup            imports + DD_LESSON_ID
       dd-lesson-<id>      the lesson header
       dd-kp-<slug>        one teaching cell per knowledge component
       dd-q<n>             a problem's header  ─┐
       dd-q<n>-hints       its hint block       ├─ the group focus keeps
       dd-q<n>-code        its answer cell     ─┘

     So no notebook change was needed to hook onto — the anchors have been there
     since the panel needed them for jumping. This file only adds classes; the
     hiding itself is CSS (`colab_dd.css`).

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
  const DEFAULTS = { theme: true, focus: true, collapsed: false, solutions: false };

  // Cells that are never hidden. The setup cell holds the imports and
  // DD_LESSON_ID and the checker cell defines `dd_check`: hide either and the
  // problem's own cells die on NameError, which reads as broken starter code
  // rather than as a missing prerequisite.
  const ALWAYS_VISIBLE = /^dd-(?:setup|checker)(?:$|[^a-z0-9])/i;

  // A problem's answer cell. Hidden until the learner says how it went — see
  // `reveal` below.
  const SOLUTION = /^dd-q(\d+)-solution$/i;

  let settings = { ...DEFAULTS };
  let lastReport = { target: null, inFocus: 0, total: 0 };

  // Problem numbers whose solution has been unlocked this page-load. Not
  // persisted on purpose: reopening a notebook is how you get a clean run at a
  // problem, and a remembered unlock would hand you the answer before you
  // started.
  const revealed = new Set();

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

  // ── Applying it ────────────────────────────────────────────────────

  function apply() {
    const root = document.documentElement;
    root.classList.toggle("dd-theme", Boolean(settings.theme));
    root.classList.toggle("dd-hide-solutions", !settings.solutions);

    const cells = Array.from(document.querySelectorAll(CELL));
    const target = settings.focus ? targetProblem() : null;

    // Tag first, count second, and only then decide whether to hide anything.
    let inFocus = 0;
    cells.forEach((cell) => {
      const anchor = cellAnchor(cell);
      const always = ALWAYS_VISIBLE.test(anchor);
      const mine = target !== null && problemOf(anchor) === target;
      if (mine) inFocus += 1;
      cell.classList.toggle("dd-always-visible", always);
      cell.classList.toggle("dd-out-of-focus", Boolean(target) && !mine && !always);
      cell.classList.toggle("dd-in-focus", mine);
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

  window.__ddFocus = { reveal };

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
      <label class="dd-row"><input type="checkbox" data-dd="solutions" /> Show every solution</label>
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
