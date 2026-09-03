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
  const EDITS_KEY = (slug) => `dd_arena_cells:${slug}`;
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

  const _renderMath = (root) => {
    if (!root || typeof window.renderMathInElement !== "function") return;
    try {
      window.renderMathInElement(root, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true },
        ],
        throwOnError: false,
      });
    } catch (_) {
      // Malformed upstream math stays readable as source.
    }
  };

  const _readSavedCells = (nb) => {
    try {
      const saved = JSON.parse(localStorage.getItem(EDITS_KEY(nb.id)) || "null");
      if (saved?.version !== 1 || !Array.isArray(saved.cells)) return nb.cells;
      return saved.cells.filter(
        (cell) => cell && cell.id && ["prose", "code", "magic", "details"].includes(cell.role),
      );
    } catch (_) {
      return nb.cells;
    }
  };

  const _removeSavedCells = (slug) => {
    try {
      localStorage.removeItem(EDITS_KEY(slug));
    } catch (_) {}
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
    _addCellTools(el);
    return el;
  };

  const _magicCell = (cell) => {
    const el = document.createElement("section");
    el.className = "nbv-cell nbv-code arena-nb-magic";
    el.dataset.role = cell.role;
    el.dataset.cellId = cell.id;
    el.id = `arena-${cell.id}`;
    el._ddSource = String(cell.src || "").replace(/\s+$/, "");
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
    el.dataset.cellId = cell.id;
    el.id = `arena-${cell.id}`;
    el._ddMarkdown = String(cell.src || "");
    el.innerHTML =
      '<div class="arena-nb-md-rendered" title="Double-click to edit this text"></div>' +
      '<textarea class="arena-nb-md-editor hidden" spellcheck="true" ' +
      'aria-label="Markdown cell source"></textarea>';
    _paintMarkdown(el);
    _addCellTools(el);
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
    el.dataset.cellId = cell.id;
    el.id = `arena-${cell.id}`;
    el._ddMarkdown = String(cell.src || "");
    const head = document.createElement("summary");
    head.textContent = cell.summary || "Show";
    const body = document.createElement("div");
    body.className = "nbv-md";
    body.innerHTML = md(cell.src);
    el.appendChild(head);
    el.appendChild(body);
    return el;
  };

  function _paintMarkdown(node) {
    const rendered = node.querySelector(".arena-nb-md-rendered");
    const editor = node.querySelector(".arena-nb-md-editor");
    if (!rendered || !editor) return;
    rendered.innerHTML = md(node._ddMarkdown || "");
    editor.value = node._ddMarkdown || "";
    _renderMath(rendered);
  }

  function _addCellTools(node) {
    const code = node.dataset.role === "code";
    const tools = document.createElement("div");
    tools.className = "arena-nb-cell-tools";
    tools.setAttribute("aria-label", "Cell actions");
    tools.innerHTML =
      '<button type="button" data-cell-action="insert-code" title="Add code cell below">+ Code</button>' +
      '<button type="button" data-cell-action="insert-prose" title="Add text cell below">+ Text</button>' +
      '<button type="button" data-cell-action="up" title="Move cell up">↑</button>' +
      '<button type="button" data-cell-action="down" title="Move cell down">↓</button>' +
      `<button type="button" data-cell-action="convert" title="Change cell type">${code ? "Text" : "Code"}</button>` +
      '<button type="button" data-cell-action="delete" title="Delete cell">×</button>';
    node.appendChild(tools);
  }

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

  const _cellRecord = (node) => ({
    id: node.dataset.cellId || node.id.replace(/^arena-/, ""),
    role: node.dataset.role || "prose",
    src:
      node.dataset.role === "prose" || node.dataset.role === "details"
        ? node._ddMarkdown || ""
        : _sourceOf(node),
    ...(node.dataset.role === "details"
      ? { summary: node.querySelector(":scope > summary")?.textContent || "Show" }
      : {}),
  });

  /* 🔴 AN UNTOUCHED NOTEBOOK IS NEVER WRITTEN. `_backToCourses` persists on the
     way out, so without this flag merely OPENING a section stored a full copy
     of it — and an ARENA notebook is up to 656 cells. Read all 32 and the 5 MB
     origin quota is gone, at which point `localStorage.setItem` throws and the
     learner is told their edits could not be saved on a notebook they only
     read. `dirty` is set by the two input handlers and by `_afterCellChange`,
     which is every path that can actually change a cell. */
  const _persistCells = (state = current) => {
    if (!state?.body || !state.dirty) return;
    clearTimeout(state.persistTimer);
    state.persistTimer = 0;
    const cells = Array.from(state.body.children)
      .filter((node) => node.matches(".nbv-cell"))
      .map(_cellRecord);
    try {
      localStorage.setItem(EDITS_KEY(state.id), JSON.stringify({ version: 1, cells }));
    } catch (_) {
      _banner("Notebook edits could not be saved in this browser.", "warn", state);
    }
  };

  const _queuePersist = (state = current) => {
    if (!state?.body) return;
    state.dirty = true;
    clearTimeout(state.persistTimer);
    state.persistTimer = setTimeout(() => _persistCells(state), 250);
  };

  const _newCell = (role) => ({
    id: `user-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    role,
    src: role === "code" ? "# Write Python here\n" : "Double-click to edit this text.",
  });

  /* 🔴 OPENING AN EDITOR IS `_beginMarkdownEdit`, ALWAYS. This used to unhide
     the textarea itself, which skipped the one line that matters —
     `_ddEditBefore`. Escape then restored `node._ddEditBefore || ""`, so
     cancelling out of a cell you had just inserted did not undo the edit, it
     BLANKED the cell. */
  const _focusCell = (node) => {
    if (node.dataset.role === "code") {
      node.querySelector(".nbv-src code")?.focus();
      return;
    }
    _beginMarkdownEdit(node);
    node.querySelector(".arena-nb-md-editor")?.select();
  };

  const _afterCellChange = (state, focusNode = null) => {
    state.dirty = true;
    _persistCells(state);
    window.ArenaNotebookNav?.refresh({ rebuild: true });
    if (focusNode) requestAnimationFrame(() => _focusCell(focusNode));
  };

  const _handleCellAction = (button, state) => {
    const node = button.closest(".nbv-cell");
    if (!node || state !== current) return;
    const action = button.dataset.cellAction;
    if (action === "insert-code" || action === "insert-prose") {
      const fresh = _cellNode(_newCell(action === "insert-code" ? "code" : "prose"));
      node.after(fresh);
      _afterCellChange(state, fresh);
      return;
    }
    if (action === "up" || action === "down") {
      const sibling = action === "up" ? node.previousElementSibling : node.nextElementSibling;
      if (!sibling?.matches(".nbv-cell")) return;
      if (action === "up") sibling.before(node);
      else sibling.after(node);
      _afterCellChange(state);
      return;
    }
    if (action === "convert") {
      const role = node.dataset.role === "code" ? "prose" : "code";
      const replacement = _cellNode({
        id: node.dataset.cellId,
        role,
        src: node.dataset.role === "prose" ? node._ddMarkdown : _sourceOf(node),
      });
      node.replaceWith(replacement);
      _afterCellChange(state, replacement);
      return;
    }
    if (action === "delete") {
      if (!window.confirm("Delete this cell? Reset edits restores the compiled notebook.")) return;
      node.remove();
      _afterCellChange(state);
    }
  };

  const _beginMarkdownEdit = (node) => {
    const editor = node.querySelector(".arena-nb-md-editor");
    const rendered = node.querySelector(".arena-nb-md-rendered");
    if (!editor || !rendered) return;
    node._ddEditBefore = node._ddMarkdown || "";
    editor.value = node._ddMarkdown || "";
    node.classList.add("is-editing");
    rendered.classList.add("hidden");
    editor.classList.remove("hidden");
    editor.focus();
  };

  const _finishMarkdownEdit = (node, state, save = true) => {
    const editor = node.querySelector(".arena-nb-md-editor");
    const rendered = node.querySelector(".arena-nb-md-rendered");
    if (!editor || !rendered || !node.classList.contains("is-editing")) return;
    if (save) node._ddMarkdown = editor.value;
    else node._ddMarkdown = node._ddEditBefore || "";
    delete node._ddEditBefore;
    node.classList.remove("is-editing");
    editor.classList.add("hidden");
    rendered.classList.remove("hidden");
    _paintMarkdown(node);
    if (save) {
      _afterCellChange(state);
      return;
    }
    /* 🔴 ESCAPE HAS TO REACH STORAGE, NOT JUST THE NODE. Every keystroke runs
       `_queuePersist`, so a cancel that arrives more than 250ms after the
       first character is cancelling text that is ALREADY in localStorage —
       restoring `_ddMarkdown` in memory and stopping there means the next
       reload brings the abandoned edit back. `_persistCells` clears the
       pending timer and writes the restored cells, and it still no-ops on a
       notebook nothing has ever written. Found by codex, 2026-09-03. */
    _persistCells(state);
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
      window.ArenaNotebookNav?.syncCompletion();
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
    window.ArenaNotebookNav?.syncCompletion();
  };

  /* ---------- the notebook screen -------------------------------------- */

  /* There is no "Jump to…" dropdown. `arena-notebook-nav.js` builds a plain
     contents tree from every rendered h1-h4, reveals it across the full left
     gutter, tracks the current heading, and derives completed sections from
     successful code-cell runs. Reading rendered headings matters: one prose
     cell can open several nested sections. */

  const _host = () => document.getElementById("arena-notebook-host");
  /* The tab's page element, not the mount. The contents tree is parked here
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
    '<button type="button" class="arena-nb-reset-edits" title="Restore compiled cells">' +
    "Reset edits</button>" +
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
    const state = {
      id: nb.id,
      title: nb.title,
      host,
      body: null,
      nb,
      persistTimer: 0,
      dirty: false,
      runSeq: 0,
    };
    current = state;
    host.innerHTML = _headerHtml(nb);

    const body = document.createElement("div");
    body.className = "nbv-cells";
    state.body = body;
    const fragment = document.createDocumentFragment();
    _readSavedCells(nb).forEach((cell) => {
      const node = _cellNode(cell);
      if (node) fragment.appendChild(node);
    });
    body.appendChild(fragment);
    host.appendChild(body);

    // One listener for the whole notebook rather than one per Run button — a
    // 300-cell page should not pay for a handler per cell.
    body.addEventListener("click", (event) => {
      const action = event.target.closest("[data-cell-action]");
      if (action) {
        _handleCellAction(action, state);
        return;
      }
      const button = event.target.closest(".nbv-run");
      if (!button) return;
      const node = button.closest(".nbv-cell");
      if (node) _runCell(node);
    });
    // An edit updates the source the node carries. `innerText` is accurate here
    // because a cell being typed into is by definition on screen.
    body.addEventListener("input", (event) => {
      const markdown = event.target.closest(".arena-nb-md-editor");
      if (markdown) {
        const node = markdown.closest(".nbv-cell");
        if (node) {
          node._ddMarkdown = markdown.value;
          _queuePersist(state);
        }
        return;
      }
      const code = event.target.closest(".nbv-src code");
      if (!code) return;
      const node = code.closest(".nbv-cell");
      if (node) {
        node._ddSource = (code.innerText || "").replace(/ /g, " ");
        /* 🔴 A GREEN SECTION IS A CLAIM ABOUT THE CODE THAT IS THERE NOW.
           Running a cell marks it `has-run` and the contents tree turns the
           section green with a check; typing into it afterwards left the
           check standing over code that had never been executed. `is-stale`
           is the class the Restart-session path already uses for exactly this
           — "ran once, no longer describes what you see" — and
           `syncCompletion` refuses to count a stale cell. Found by codex,
           2026-09-03. */
        if (node.classList.contains("has-run") && !node.classList.contains("is-stale")) {
          node.classList.add("is-stale");
          window.ArenaNotebookNav?.syncCompletion();
        }
        _queuePersist(state);
      }
    });
    body.addEventListener("dblclick", (event) => {
      const rendered = event.target.closest(".arena-nb-md-rendered");
      if (rendered) _beginMarkdownEdit(rendered.closest(".nbv-cell"));
    });
    body.addEventListener("focusout", (event) => {
      const editor = event.target.closest(".arena-nb-md-editor");
      if (editor) _finishMarkdownEdit(editor.closest(".nbv-cell"), state, true);
      else if (event.target.closest(".nbv-src code")) _persistCells(state);
    });
    body.addEventListener("keydown", (event) => {
      const editor = event.target.closest(".arena-nb-md-editor");
      if (!editor) return;
      if (event.key === "Escape") {
        event.preventDefault();
        _finishMarkdownEdit(editor.closest(".nbv-cell"), state, false);
      } else if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        _finishMarkdownEdit(editor.closest(".nbv-cell"), state, true);
      }
    });

    _renderMath(body);

    host.querySelector(".nbv-back").onclick = () => _backToCourses();
    host.querySelector(".arena-nb-reset-edits").onclick = () => {
      if (!window.confirm("Restore the compiled notebook and discard your cell edits?")) return;
      clearTimeout(state.persistTimer);
      _removeSavedCells(nb.id);
      _render(nb, host);
      window.scrollTo({ top: 0 });
    };
    host.querySelector(".nbv-restart").onclick = async () => {
      await window.DeltaKernel?.reset();
      state.runSeq = 0;
      host.querySelectorAll(".nbv-cell.has-run").forEach((cell) => cell.classList.add("is-stale"));
      window.ArenaNotebookNav?.syncCompletion();
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

    /* The contents tree and scroll memory, in that order: the tree measures
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
    _persistCells(current);
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
