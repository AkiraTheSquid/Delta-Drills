/* ================================================================
   ARENA NOTEBOOK VIEW — a Courses section, opened IN the app
   ================================================================

   WHAT THIS IS

   The Courses tab is the ARENA curriculum, and every section row used to be a
   link OUT: `<a target="_blank">` at Google Colab, with `courses-fork-gate.js`
   offering to repoint it at the student's own ARENA_3.0 fork first.

   Seth, 2026-09-01: "whenever you click those links for the courses ... it
   won't actually take you to the Google Colab. It will instead stay inside of
   the app, and it will have an app version of those arena notebooks."

   So a section row opens here instead. `scripts/compile_arena_notebooks.py`
   rewrites the upstream `.ipynb` into the same cell JSON the lesson notebooks
   use, this file draws it, and the CSS is the notebook surface's own — the
   `.nbv-*` classes from styles/practice/notebook-view.css — because "like our
   other notebooks thing" is the whole brief.

   HOW IT DIFFERS FROM practice/notebook-view.js

   That file renders DELTA DRILLS lessons, and everything specific to it is
   about grading: a checker cell holding the test cases, `dd_check(n)` cells,
   and a beacon that posts the verdict line to the engine. This surface has
   NONE of that, and the absence is deliberate rather than unfinished:

     🔴 NOTHING HERE REACHES THE ENGINE. These are upstream's exercises, not
     our question bank. There is no question id to record against, no mastery
     claim that could be derived from "the learner ran a cell", and a surface
     that silently fed the estimator from someone else's notebook would move a
     learner's ladder on evidence the ladder does not model. Reading ARENA and
     practising Delta Drills are two different activities; only one of them is
     graded, and this is the other one.

   What IS shared is what a cell IS: `LessonNotebook.runSource`, one
   `_delta_cell` harness, one kernel session per notebook.

   THE FOUR ROLES the compiler mints:

     prose     markdown. Upstream's HTML (<img>, <a>, <details>, <code>) was
               turned back into markdown at compile time, because the app's
               one renderer escapes tags on purpose.
     details   a disclosure — every ARENA hint and every ARENA solution. Closed:
               a hint read before the attempt replaces the thinking it was
               written to prompt. (The lesson notebooks show their solutions
               OPEN, and that is right there for a reason that does not hold
               here: those solutions answer a problem the app itself set.)
     code      runnable.
     magic     a Colab setup cell (`%pip install …`, `!git clone …`). A line
               magic is a SyntaxError to `exec`, which is what the kernel runs,
               so this is drawn read-only with the reason. 🔴 Without that, the
               FIRST cell of every ARENA notebook is a Run button that answers
               with a SyntaxError, which reads as a broken app.
   ================================================================ */

const ArenaNotebookView = (() => {
  const DIR = "lessons/notebooks/";
  const FILE = (slug) => `${DIR}arena-${encodeURIComponent(slug)}.json`;
  const INDEX = `${DIR}arena-index.json`;
  /* One Python session per section, and switching sections starts a new one.
     Same reasoning as the lesson notebooks: the cells below start from cell 1
     either way, so a name surviving from another notebook could only ever be a
     silent wrong answer. */
  const CONTEXT = (slug) => `arena:${slug}`;

  const esc = (value) =>
    String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

  /* The app's one markdown renderer, the same call the lesson notebook view
     makes. `headingLevels` for the same reason too: this is a whole ARENA
     section — chapter heading, section headings, exercise headings — and the
     depths are the only thing saying which contains which. */
  const md = (text) => {
    const render = window.LessonGate && window.LessonGate.renderMarkdown;
    if (!render) return `<pre>${esc(text)}</pre>`;
    return render(text, { headingLevels: true });
  };

  const _fetchJson = async (path) => {
    // no-cache for the same reason the lesson notebooks use it: a disk-cached
    // notebook shows an OLD compile after a recompile, with no visible error.
    const res = await fetch(path, { cache: "no-cache" });
    if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
    return res.json();
  };

  let index = null;
  /* The notebook on screen. 🔴 A run is asynchronous and the learner is not:
     they can press Back, or open another section, while a cell is executing.
     Anything read AFTER an await must come from the notebook the run BELONGS
     to — not from this variable, which by then may be a different section or
     null. The same trap `notebook-view.js` documents, and the same answer. */
  let current = null;

  /* ---------- cells ---------------------------------------------------- */

  /* Where a cell's source comes from at Run time: carried ON THE NODE, never
     read back out of the DOM. 🔴 `innerText` is defined in terms of LAYOUT, so
     a cell inside a collapsed <details> returns the empty string — which runs
     an empty program and reports success. Learned on the lesson notebooks
     (practice/watch_notebook.py pins it there); it is true here for the same
     reason, and ARENA puts code inside disclosures constantly. */
  const _sourceOf = (node) => {
    if (node._ddSource != null) return node._ddSource;
    const code = node.querySelector(".nbv-src code");
    return ((code && (code.innerText || code.textContent)) || "").replace(/ /g, " ");
  };

  const _codeCell = (cell) => {
    const el = document.createElement("section");
    el.className = "nbv-cell nbv-code";
    el.dataset.role = cell.role;
    el.dataset.cellId = cell.id;
    el.id = `arena-${cell.id}`;
    el._ddSource = String(cell.src || "").replace(/\s+$/, "");
    el.innerHTML =
      '<div class="nbv-gutter">' +
      '<button type="button" class="nbv-run" title="Run this cell">▶</button>' +
      '<span class="nbv-count" aria-hidden="true"></span>' +
      "</div>" +
      '<div class="nbv-body">' +
      '<pre class="nbv-src"><code contenteditable="plaintext-only" spellcheck="false">' +
      esc(el._ddSource) +
      "</code></pre>" +
      '<pre class="nbv-out hidden"></pre>' +
      "</div>";
    return el;
  };

  const _magicCell = (cell) => {
    const el = document.createElement("section");
    el.className = "nbv-cell nbv-code arena-nb-magic";
    el.dataset.role = cell.role;
    el.id = `arena-${cell.id}`;
    el.innerHTML =
      '<div class="nbv-gutter"><span class="arena-nb-magic-mark" aria-hidden="true">⚙</span></div>' +
      '<div class="nbv-body">' +
      '<p class="nbv-checker-note">⚙ <strong>Colab setup</strong> — this cell uses ' +
      "notebook magics (<code>%pip</code>, <code>!</code>), which the app's Python " +
      "session cannot run. It is here because it is part of the notebook; install " +
      "what it names in your own environment.</p>" +
      '<pre class="nbv-src"><code>' +
      esc(String(cell.src || "").replace(/\s+$/, "")) +
      "</code></pre>" +
      "</div>";
    return el;
  };

  const _mdCell = (cell) => {
    const el = document.createElement("section");
    el.className = "nbv-cell nbv-md";
    el.dataset.role = cell.role;
    el.id = `arena-${cell.id}`;
    el.innerHTML = md(cell.src);
    return el;
  };

  /* Upstream's <details>. Rendered as a real disclosure rather than passed to
     the markdown renderer, which escapes HTML — the tags would print. Closed;
     see the header. Code inside one is prose, not a runnable cell: an ARENA
     solution block is written to be READ next to your own attempt, and lifting
     it into a Run button would put an answer one click from the exercise. */
  const _detailsCell = (cell) => {
    const el = document.createElement("details");
    el.className = "nbv-cell nbv-hints arena-nb-details";
    el.dataset.role = cell.role;
    el.id = `arena-${cell.id}`;
    const head = document.createElement("summary");
    head.textContent = cell.summary || "Show";
    const body = document.createElement("div");
    body.className = "nbv-md";
    body.innerHTML = md(cell.src);
    el.appendChild(head);
    el.appendChild(body);
    return el;
  };

  const _cellNode = (cell) => {
    switch (cell.role) {
      case "code":
        return _codeCell(cell);
      case "magic":
        return _magicCell(cell);
      case "details":
        return _detailsCell(cell);
      default:
        return _mdCell(cell);
    }
  };

  /* ---------- running -------------------------------------------------- */

  /* 🔴 `state.host` is ONE element re-filled per notebook, so holding a state
     does not make a write through it safe — only the notebook on screen may
     paint. Identical guard to notebook-view.js::_banner. */
  const _banner = (message, kind = "info", state = current) => {
    if (state !== current) return;
    const bar = state && state.host.querySelector(".nbv-banner");
    if (!bar) return;
    bar.className = `nbv-banner nbv-banner-${kind}`;
    bar.textContent = message || "";
    bar.classList.toggle("hidden", !message);
  };

  /* The kernel restarted between two clicks — it idled out, it was evicted, the
     box was redeployed. Every name the learner bound is gone. Nothing is
     replayed: the lesson view rebuilds its checker because that cell is
     infrastructure rather than the learner's work, and this notebook has no
     equivalent — its setup cells are upstream's imports, which only the learner
     knows they meant to have run. */
  const _onFresh = (state) => {
    if (state === current) {
      state.host.querySelectorAll(".nbv-cell.has-run").forEach((cell) => {
        cell.classList.add("is-stale");
      });
    }
    _banner(
      "The Python session restarted — anything you had defined is gone. " +
        "Re-run the imports at the top.",
      "warn",
      state,
    );
  };

  const _runCell = async (node) => {
    const button = node.querySelector(".nbv-run");
    const out = node.querySelector(".nbv-out");
    const count = node.querySelector(".nbv-count");
    if (!button || !out) return;
    // Captured BEFORE the first await — see the note on `current`.
    const state = current;
    if (!state) return;
    if (!window.DeltaKernel || !window.DeltaKernel.available()) {
      out.classList.remove("hidden");
      out.classList.add("is-error");
      out.textContent =
        "Running cells needs an account — the notebook keeps a Python session " +
        "on the server, and a session needs someone to belong to.";
      return;
    }

    button.disabled = true;
    node.classList.add("is-running");
    node.classList.remove("is-stale");
    if (count) count.textContent = "[*]";
    out.classList.remove("hidden", "is-error");
    out.textContent = "";

    let failed = false;
    try {
      let result = await window.LessonNotebook.runSource(_sourceOf(node), {
        context: CONTEXT(state.id),
        name: `<${node.dataset.cellId || node.id.replace(/^arena-/, "")}>`,
      });
      if (!result) {
        result = {
          text: "The kernel is not available right now. Try again in a moment.",
          failed: true,
        };
      } else if (result.fresh) {
        _onFresh(state);
      }
      failed = !!result.failed;
      out.textContent = result.text || (failed ? "" : "✓ ran successfully");
      out.classList.toggle("is-error", failed);
    } catch (err) {
      failed = true;
      out.textContent = `Error: ${err.message}`;
      out.classList.add("is-error");
    }

    state.runSeq += 1;
    node.classList.remove("is-running");
    node.classList.add("has-run");
    node.classList.toggle("has-failed", failed);
    if (count) count.textContent = `[${state.runSeq}]`;
    button.disabled = false;
  };

  /* ---------- the notebook screen -------------------------------------- */

  /* 🔴 THERE IS NO "JUMP TO…" DROPDOWN ANY MORE (Seth, 2026-09-02). The jump
     list is `practice/arena-notebook-nav.js`: a rail down the left edge that
     is ticks until you put the mouse in the gutter and titles when you do,
     built from the headings in the RENDERED page rather than from the cell
     source. Reading the DOM is what lets it carry every heading — the select
     only ever listed the first heading of each prose cell, so a cell that
     opened a section and then started a subsection contributed one row. */

  const _host = () => document.getElementById("arena-notebook-host");
  /* The tab's page element, not the mount. The contents rail is parked here
     rather than in the host so that `host.innerHTML = …` on the next notebook
     does not take it with it, and so that hiding the page hides the rail. */
  const _page = () => document.getElementById("page-arena-notebook");

  /* The route back. `switchTab` is a top-level `const` in app.js and therefore
     NOT a property of `window` — the trap this tree documents for `PracticeAPI`
     and `PracticeSession` too. This file is a classic script loaded after
     app.js in the same document, so it shares that scope and calls it by name;
     the typeof guard is for a page that loads this without app.js. */
  const _backToCourses = () => {
    if (typeof switchTab === "function") switchTab("courses");
    else if (typeof window.switchTab === "function") window.switchTab("courses");
  };

  const _headerHtml = (nb) =>
    '<div class="nbv-toolbar">' +
    '<button type="button" class="nbv-back">← The course</button>' +
    `<span class="nbv-title">${esc(nb.number ? `${nb.number} ${nb.title}` : nb.title)}</span>` +
    '<button type="button" class="nbv-restart" title="Throw the Python session away">' +
    "Restart session</button>" +
    "</div>" +
    '<div class="nbv-banner hidden"></div>' +
    '<header class="arena-nb-head">' +
    `<div class="arena-nb-chapter">${esc(nb.chapter || "ARENA Curriculum")}</div>` +
    `<h1 class="arena-nb-title">${esc(nb.number ? `${nb.number} — ${nb.title}` : nb.title)}</h1>` +
    (nb.desc ? `<p class="arena-nb-desc">${esc(nb.desc)}</p>` : "") +
    '<p class="arena-nb-origin">This is Callum McDougall\'s ARENA notebook, ' +
    `rendered here instead of in Colab — <code>${esc(nb.notebook_path || "")}</code> ` +
    `from <code>${esc(nb.edition || "ARENA")}</code>. Nothing you do on this page is ` +
    "graded; the drills that are graded are on the Learner Home.</p>" +
    "</header>";

  const _render = (nb, host) => {
    const state = { id: nb.id, title: nb.title, host, runSeq: 0 };
    current = state;
    host.innerHTML = _headerHtml(nb);

    const body = document.createElement("div");
    body.className = "nbv-cells";
    const fragment = document.createDocumentFragment();
    nb.cells.forEach((cell) => {
      const node = _cellNode(cell);
      if (node) fragment.appendChild(node);
    });
    body.appendChild(fragment);
    host.appendChild(body);

    // One listener for the whole notebook rather than one per Run button — a
    // 300-cell page should not pay for a handler per cell.
    body.addEventListener("click", (event) => {
      const button = event.target.closest(".nbv-run");
      if (!button) return;
      const node = button.closest(".nbv-cell");
      if (node) _runCell(node);
    });
    // An edit updates the source the node carries. `innerText` is accurate here
    // because a cell being typed into is by definition on screen.
    body.addEventListener("input", (event) => {
      const code = event.target.closest(".nbv-src code");
      if (!code) return;
      const node = code.closest(".nbv-cell");
      if (node) node._ddSource = (code.innerText || "").replace(/ /g, " ");
    });

    host.querySelector(".nbv-back").onclick = () => _backToCourses();
    host.querySelector(".nbv-restart").onclick = async () => {
      await window.DeltaKernel?.reset();
      state.runSeq = 0;
      host.querySelectorAll(".nbv-cell.has-run").forEach((cell) => cell.classList.add("is-stale"));
      _banner("Session thrown away. Re-run the imports before anything below them.", "warn", state);
    };

    if (!window.DeltaKernel || !window.DeltaKernel.available()) {
      _banner(
        "You are reading this signed out, so the Run buttons have nothing to run. " +
          "Sign in and the whole notebook shares one live Python session.",
        "warn",
        state,
      );
    }

    /* The contents rail and the scroll memory, in that order: the rail measures
       heading offsets, and restoring the scroll position first would have it
       measure them mid-jump. Both are optional-chained — a build without either
       file is a notebook with no rail and no memory, not a blank page. */
    window.ArenaNotebookNav?.mount(_page(), host, nb.number ? `${nb.number} ${nb.title}` : nb.title);
    window.ArenaNotebookState?.bind(nb.id);
  };

  /* ---------- entry points --------------------------------------------- */

  const _fail = (host, message) => {
    host.innerHTML =
      `<p class="nbv-error">${esc(message)}</p>` +
      '<p><button type="button" class="nbv-back">← The course</button></p>';
    host.querySelector(".nbv-back").onclick = () => _backToCourses();
  };

  /* Open one section. Called by courses.js on a section click and by the
     `?arena=<slug>` deep link. Returns false when there is nothing to open, so
     the caller can decide what to do instead of assuming a page appeared. */
  const open = async (slug) => {
    const host = _host();
    if (!host || !slug) return false;

    /* 🔴 REOPENING THE NOTEBOOK YOU ARE ALREADY IN DOES NOT REBUILD IT (Seth,
       2026-09-02: "if you go back to that tab, it will stay at that location
       that you were at before"). This used to re-fetch and re-render on every
       click of the section row, which threw away the whole session: the cells
       you had edited, the outputs you had run, the disclosures you had opened,
       and the position you were reading at. The kernel is untouched by a
       re-render — `CONTEXT(slug)` is the same session — so the page came back
       LOOKING empty while your names were still bound, which is worse than
       either honest answer. Nothing here is stale: the notebook JSON is a
       compiled artifact of this build. */
    if (current && current.id === slug && host.querySelector(".nbv-cells")) {
      if (typeof switchTab === "function") switchTab("arena-notebook");
      else if (typeof window.switchTab === "function") window.switchTab("arena-notebook");
      window.ArenaNotebookNav?.refresh();
      window.ArenaNotebookState?.restore();
      return true;
    }

    // Leaving a different notebook: take its position with us AND let go of it,
    // before the DOM that the reading is relative to is replaced by a loading
    // paragraph. Found by codex, 2026-09-02.
    window.ArenaNotebookState?.suspend();
    if (typeof switchTab === "function") switchTab("arena-notebook");
    else if (typeof window.switchTab === "function") window.switchTab("arena-notebook");
    current = null;
    window.ArenaNotebookNav?.destroy();
    host.innerHTML = '<p class="nbv-loading">Loading the notebook…</p>';
    let nb;
    try {
      nb = await _fetchJson(FILE(slug));
    } catch (err) {
      /* 🔴 THE COMPILE STEP IS THE USUAL CAUSE, so say so. These notebooks are
         built from `Local_Deployed_Shared/content/`, which is gitignored — a
         checkout without it, or a tree where the compiler has never run, has
         the Courses tab and none of its notebooks. A bare "not found" sends
         the reader looking for a broken link that is not there. */
      _fail(
        host,
        `That ARENA notebook has not been compiled here (${slug}). ` +
          "Run: python3 scripts/compile_arena_notebooks.py",
      );
      console.warn("[arena-notebook] notebook unavailable:", err);
      return false;
    }
    _render(nb, host);
    /* Top of the notebook unless this browser remembers a position in it —
       which it does after a reload, or a visit yesterday. `bind` in _render
       has already named the slug this reads. */
    window.scrollTo({ top: 0 });
    window.ArenaNotebookState?.restore();
    return true;
  };

  /* The compiled index, for anything that wants to know which sections exist
     without opening one. Cached: courses.js asks once per click to decide
     whether a row can be opened in-app at all. */
  const sections = async () => {
    if (index) return index;
    try {
      index = (await _fetchJson(INDEX)).sections || [];
    } catch (err) {
      console.warn("[arena-notebook] index unavailable:", err);
      index = [];
    }
    return index;
  };

  return { open, sections };
})();

window.ArenaNotebook = ArenaNotebookView;

/* `?arena=<slug>` opens a section directly — the same shape as the lesson
   notebooks' `?notebook=<id>`, and what a link to "0.1 Ray Tracing" pasted
   into a chat should do. */
(function () {
  const requested = new URLSearchParams(location.search).get("arena");
  if (!requested) return;
  const start = () => ArenaNotebookView.open(requested);
  if (document.readyState === "loading") {
    window.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
