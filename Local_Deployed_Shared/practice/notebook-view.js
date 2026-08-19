/* ================================================================
   LESSON NOTEBOOK VIEW — a whole lesson, as a notebook, in the app
   ================================================================

   WHAT THIS IS

   The Colab edition's answer to "where does the learner read and work" is a
   notebook: one file per lesson, every fence a runnable cell, every problem
   followed by a checker, all of it against one live Python session. This is
   that same notebook, rendered in the default app against the kernel from
   `app/kernel_runner.py`.

   IT IS THE SAME NOTEBOOK, NOT A SECOND ONE. `scripts/compile_web_notebooks.py`
   calls `generate_colab_notebooks.build_notebook` — the function that compiles
   the .ipynb files — and writes its cells out as JSON. Same order, same ids,
   same `dd_check` cases. Nothing about content is decided in this file; it
   decides how a cell LOOKS and what happens when you press Run.

   HOW IT DIFFERS FROM practice/notebook.js

   `notebook.js` serves the lesson GATE: one concept, its fences turned into
   cells, and a stateless prefix replay when no kernel is available. That
   replay is O(cells) per click, which a six-cell concept can afford.

   This surface is a whole lesson — up to 656 cells — so the replay is not a
   fallback here, it is the thing pass 1 existed to abolish. Without a kernel
   this view READS and does not run, and says so. The two share exactly one
   thing, `LessonNotebook.runSource`, which is what a cell IS.

   THE FIVE ROLES

   The compiler labels every cell, from the id grammar `colab_cells.py` mints:

     setup     names the lesson for the Chrome extension. Nothing to run, and
               nothing to read — not rendered here at all.
     checker   defines `dd_check` and carries this lesson's cases as 80 KB of
               base64. Runnable, never readable: printing it would be printing
               the answer key above the problems.
     problem   the header a problem is anchored on.
     hints     a raw <details> block, which the shared markdown renderer would
               escape — unwrapped into a real disclosure element here.
     check     `dd_check(<n>)`. The line it prints is the only thing on this
               screen that reaches the engine.
     solution  the answer. Behind a disclosure, closed.

   WHAT REACHES THE ENGINE

   The verdict `dd_check` prints, and nothing else — the same line, matched by
   the same pattern, that `content/colab_focus.js` reads off a Colab cell. It
   is recorded ONCE per problem per visit. A learner debugging a failing drill
   presses Run repeatedly, and counting each press as an attempt would let the
   act of iterating — which is the correct way to work — drive their own
   mastery estimate down.
*/

const LessonNotebookView = (() => {
  const DIR = "lessons/notebooks/";
  /* One session per lesson. Switching notebooks restarts the kernel, which is
     right: the cells below start from cell 1 either way, so a name surviving
     from another lesson could only be a silent wrong answer. */
  const CONTEXT = (id) => `nb:${id}`;
  /* The line `dd_check` prints. Identical to `content/colab_focus.js`'s RESULT
     pattern on purpose — one verdict format, read the same way on both
     editions. The em dash is part of it: the SOURCE `dd_check(480)` is in the
     cell too, and matching a bare number would find that first. */
  const VERDICT = /(✅|❌) Problem (\d+) — /;

  const esc = (value) =>
    String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

  /* The app's one markdown renderer. Shared with the lesson page, the ladder's
     inline example and lessons/viewer.html, so a paragraph reads the same
     wherever it is shown. `headingLevels` is the one thing this caller asks
     for that the others do not — see the note on `md` in practice/lessons.js. */
  const md = (text) => {
    const render = window.LessonGate && window.LessonGate.renderMarkdown;
    if (!render) return `<pre>${esc(text)}</pre>`;
    return render(text, { headingLevels: true });
  };

  const _fetchJson = async (path) => {
    // Same reason as the lesson gate's fetch: a disk-cached notebook silently
    // shows an OLD lesson after a recompile, with no visible error.
    const res = await fetch(path, { cache: "no-cache" });
    if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
    return res.json();
  };

  /* `<!-- dd:dd-q481 -->` is Colab's navigation fallback: the panel searches
     rendered text for it when Colab drops cell ids. Nothing here needs it, and
     the shared renderer escapes HTML — so left in, it would print the comment
     to the learner as if it were part of the lesson. */
  const _stripMarkers = (src) => String(src || "").replace(/<!--[\s\S]*?-->/g, "").trim();

  /* `#@title …` is a Colab form directive that renders as the cell's heading
     THERE and as a stray comment anywhere else. Stripped for display only; the
     source that runs keeps it, because what runs should be what was compiled. */
  const _stripTitle = (src) => String(src || "").replace(/^#@title[^\n]*\n/, "");

  let manifest = null;
  /* The notebook on screen: `{ id, title, host, checkerSource, checkerRan,
     runSeq }`. 🔴 A run is asynchronous and the learner is not: they can press
     Back, or open another lesson, while a cell is still executing. Anything
     read AFTER an await must come from the notebook the run BELONGS to — not
     from this variable, which by then may be a different lesson or null. */
  let current = null;

  /* ---------- cell DOM ------------------------------------------------- */

  /* Where a cell's source comes from at Run time.

     Carried ON THE NODE, never read back out of the rendered DOM. 🔴 That is
     not a style preference — `innerText` is defined in terms of LAYOUT, so a
     cell inside a collapsed `<details>` (every solution, every hints block)
     returns the empty string. Reading it there ran an empty program and
     reported "✓ ran successfully": the learner opened a solution, pressed Run,
     was told it worked, and `dd_check` below still said `solve` is not defined.
     `_ddSource` is seeded at build time and updated on every edit, so it is
     right whether the cell is on screen or not.

     `textContent` is the fallback rather than `innerText` for the same reason —
     it does not care whether the node is displayed. */
  const _sourceOf = (node) => {
    if (node._ddSource != null) return node._ddSource;
    const code = node.querySelector(".nbv-src code");
    return ((code && (code.innerText || code.textContent)) || "").replace(/ /g, " ");
  };

  const _codeCell = (cell, { editable = true, source = null } = {}) => {
    const el = document.createElement("section");
    el.className = "nbv-cell nbv-code";
    el.dataset.role = cell.role;
    el.id = `nbv-${cell.id}`;
    el.dataset.cellId = cell.id;
    if (cell.q) el.dataset.q = String(cell.q);
    el._ddSource = source != null ? source : _stripTitle(cell.src).replace(/\s+$/, "");
    el.innerHTML =
      '<div class="nbv-gutter">' +
      '<button type="button" class="nbv-run" title="Run this cell">▶</button>' +
      '<span class="nbv-count" aria-hidden="true"></span>' +
      "</div>" +
      '<div class="nbv-body">' +
      (editable
        ? '<pre class="nbv-src"><code contenteditable="plaintext-only" spellcheck="false">' +
          esc(_stripTitle(cell.src).replace(/\s+$/, "")) +
          "</code></pre>"
        : "") +
      '<pre class="nbv-out hidden"></pre>' +
      "</div>";
    return el;
  };

  const _mdCell = (cell) => {
    const el = document.createElement("section");
    el.className = "nbv-cell nbv-md";
    el.dataset.role = cell.role;
    el.id = `nbv-${cell.id}`;
    if (cell.q) el.dataset.q = String(cell.q);
    el.innerHTML = md(_stripMarkers(cell.src));
    return el;
  };

  /* A `<details>` the compiler authored as literal HTML. Unwrapped rather than
     passed through: the shared renderer escapes HTML, so the tags would print. */
  const _detailsCell = (cell, summary, body, extraClass) => {
    const el = document.createElement("details");
    el.className = `nbv-cell ${extraClass}`;
    el.dataset.role = cell.role;
    el.id = `nbv-${cell.id}`;
    if (cell.q) el.dataset.q = String(cell.q);
    const head = document.createElement("summary");
    head.textContent = summary;
    el.appendChild(head);
    el.appendChild(body);
    return el;
  };

  const _hintsCell = (cell) => {
    const inner = String(cell.src || "")
      .replace(/^\s*<details>\s*/i, "")
      .replace(/<summary>[\s\S]*?<\/summary>/i, "")
      .replace(/<\/details>\s*$/i, "")
      .trim();
    const body = document.createElement("div");
    body.className = "nbv-md";
    body.innerHTML = md(inner);
    return _detailsCell(cell, "Hints", body, "nbv-hints");
  };

  /* The answer, closed. Runnable once open, exactly as it is in Colab — the
     comment the compiler put at the top of it explains that running it rebinds
     `solve`, which is a thing a learner may legitimately want to do. */
  const _solutionCell = (cell) => {
    const body = _codeCell(cell);
    body.classList.add("nbv-solution-body");
    // The <details> around it takes the cell's id — two elements answering to
    // one id would make `getElementById` (the concept jump, the beacon's status
    // line) pick whichever came first.
    body.id = `nbv-${cell.id}-body`;
    return _detailsCell(cell, `💡 Show the solution — problem ${cell.q}`, body, "nbv-solution");
  };

  const _checkerCell = (cell) => {
    const el = _codeCell(cell, { editable: false, source: cell.src });
    el.classList.add("nbv-checker");
    el.querySelector(".nbv-body").insertAdjacentHTML(
      "afterbegin",
      '<p class="nbv-checker-note">🔧 <strong>Checker</strong> — run this once, ' +
        "before any <code>dd_check()</code> below. It loads the same test cases " +
        "the tutor grades with.</p>",
    );
    return el;
  };

  const _cellNode = (cell) => {
    switch (cell.role) {
      case "setup":
        return null; // names the lesson for the extension; nothing to read or run
      case "checker":
        return _checkerCell(cell);
      case "solution":
        return _solutionCell(cell);
      case "hints":
        return _hintsCell(cell);
      case "code":
      case "check":
        return _codeCell(cell);
      default:
        return _mdCell(cell);
    }
  };

  /* ---------- running -------------------------------------------------- */

  /* 🔴 `state.host` is `#notebooks-host` — ONE element, re-filled per notebook,
     never a per-notebook node. So holding a notebook's state does not make a
     write through it safe: a run that lands after the learner opened another
     lesson would paint ITS banner and stale ITS cells. Only the notebook on
     screen may write. (The cell nodes a late run touches are the ones it was
     started from, which innerHTML has already discarded — those are harmless.) */
  const _banner = (message, kind = "info", state = current) => {
    if (state !== current) return;
    const bar = state && state.host.querySelector(".nbv-banner");
    if (!bar) return;
    bar.className = `nbv-banner nbv-banner-${kind}`;
    bar.textContent = message || "";
    bar.classList.toggle("hidden", !message);
  };

  /* The kernel restarted between two clicks — it idled out, it was evicted for
     another learner, the box was redeployed. Every name the learner bound is
     gone, and saying nothing would leave them reading a NameError from a cell
     that worked a minute ago.

     The one thing rebuilt without being asked is the checker. It is
     infrastructure rather than the learner's work, it is a single cell, and
     without it every `dd_check` below reads as a broken app rather than as a
     lost session. Their own cells are NOT replayed: there can be 656 of them,
     and which ones they meant to have run is not something this file knows. */
  const _onFresh = async (state) => {
    if (state === current) {
      state.host.querySelectorAll(".nbv-cell.has-run").forEach((cell) => {
        cell.classList.add("is-stale");
      });
    }
    state.recorded = new Set();
    if (!state.checkerRan) {
      _banner("The Python session restarted — anything you had defined is gone.", "warn", state);
      return;
    }
    const restored = await window.LessonNotebook.runSource(state.checkerSource, {
      context: CONTEXT(state.id),
      name: "<checker>",
      echo: false,
    });
    _banner(
      restored && !restored.failed
        ? "The Python session restarted. The checker is loaded again; re-run your own cells."
        : "The Python session restarted — anything you had defined is gone.",
      "warn",
      state,
    );
  };

  /* 🔴 `PracticeAPI` is a top-level `const` in a classic script, so it lives on
     the global LEXICAL scope and is NOT a property of `window`. Reaching for it
     as `window.PracticeAPI` reads `undefined`, and the only symptom was a
     console warning while every verdict quietly failed to reach the engine —
     the notebook looked like it was grading and was recording nothing. Read it
     by name, with `typeof` so this file still loads if api.js ever does not. */
  const _practiceApi = () =>
    typeof PracticeAPI !== "undefined" ? PracticeAPI : window.PracticeAPI;

  /* The verdict, and only the verdict, reaches the engine. Once per problem —
     see the header. */
  const _beacon = async (state, node, text) => {
    const match = VERDICT.exec(String(text || ""));
    if (!match) return;
    const qid = Number(match[2]);
    const correct = match[1] === "✅";
    if (state.recorded.has(qid)) return;
    state.recorded.add(qid);
    try {
      await _practiceApi().recordLocalEval(qid, correct);
      const status = node.querySelector(".nbv-count");
      if (status) status.title = `Recorded: problem ${qid} ${correct ? "correct" : "incorrect"}`;
    } catch (err) {
      // A lost POST costs one attempt, not the run the learner just did. Let
      // it be retried by allowing the next press through.
      state.recorded.delete(qid);
      console.warn("[notebook-view] could not record the attempt:", err);
    }
  };

  const _runCell = async (node) => {
    const button = node.querySelector(".nbv-run");
    const out = node.querySelector(".nbv-out");
    const count = node.querySelector(".nbv-count");
    if (!button || !out) return;
    /* The notebook this run belongs to, captured BEFORE the first await. Read
       `current` after one and a slow cell finishing just as the learner opens
       another lesson would mark THAT lesson's checker as loaded, and a later
       restart would replay this lesson's checker into its session. */
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

    const source = _sourceOf(node);
    let failed = false;
    try {
      let result = await window.LessonNotebook.runSource(source, {
        context: CONTEXT(state.id),
        name: `<${node.dataset.cellId || node.id.replace(/^nbv-/, "")}>`,
      });
      // A kernel that was available a moment ago and is not now. One retry
      // after rebuilding the checker would be guessing; say what happened.
      if (!result) {
        result = { text: "The kernel is not available right now. Try again in a moment.", failed: true };
      } else if (result.fresh) {
        await _onFresh(state);
      }
      failed = !!result.failed;
      const checks = window.LessonNotebook.checkCount(source);
      out.textContent =
        !failed && checks && !/\S/.test(result.text.replace(/^✓.*$/gm, ""))
          ? `✓ ${checks} check${checks === 1 ? "" : "s"} passed`
          : result.text;
      out.classList.toggle("is-error", failed);
      if (node.dataset.role === "check" && !failed) await _beacon(state, node, result.text);
      if (node.dataset.role === "checker" && !failed) {
        state.checkerRan = true;
        _banner("", "info", state);
      }
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

  /* Jump targets: every problem, and every heading a cell opens with. A 656-cell
     notebook without one is a scrollbar and a hope. */
  const _tocOptions = (cells) =>
    cells
      .map((cell) => {
        if (cell.role === "problem") return { id: cell.id, label: `Problem ${cell.q}` };
        const heading = _stripMarkers(cell.src).match(/^(#{1,4})\s+(.+)$/m);
        if (cell.t === "md" && heading) {
          const depth = heading[1].length;
          return { id: cell.id, label: `${"  ".repeat(Math.max(0, depth - 1))}${heading[2]}` };
        }
        return null;
      })
      .filter(Boolean);

  const _toolbarHtml = (nb, options) =>
    '<div class="nbv-toolbar">' +
    '<button type="button" class="nbv-back">← All notebooks</button>' +
    `<span class="nbv-title">${esc(nb.title)}</span>` +
    '<select class="nbv-toc" aria-label="Jump to a section">' +
    '<option value="">Jump to…</option>' +
    options.map((o) => `<option value="${esc(o.id)}">${esc(o.label)}</option>`).join("") +
    "</select>" +
    '<button type="button" class="nbv-restart" title="Throw the Python session away">' +
    "Restart session</button>" +
    "</div>" +
    '<div class="nbv-banner hidden"></div>';

  const _render = (nb, host) => {
    const checker = nb.cells.find((c) => c.role === "checker");
    const state = {
      id: nb.id,
      title: nb.title,
      host,
      checkerSource: checker ? checker.src : "",
      checkerRan: false,
      runSeq: 0,
      /* Problems whose verdict has already been recorded — see the header on
         why this is once-per-problem. It belongs to the NOTEBOOK rather than to
         this file, so a cell finishing after the learner opened another lesson
         cannot write into the set the new lesson is using. */
      recorded: new Set(),
    };
    current = state;
    host.innerHTML = _toolbarHtml(nb, _tocOptions(nb.cells));
    const body = document.createElement("div");
    body.className = "nbv-cells";
    const fragment = document.createDocumentFragment();
    nb.cells.forEach((cell) => {
      const node = _cellNode(cell);
      if (node) fragment.appendChild(node);
    });
    body.appendChild(fragment);
    host.appendChild(body);

    // One listener for the whole notebook rather than 400 — a 656-cell page
    // should not pay for a handler per Run button.
    body.addEventListener("click", (event) => {
      const button = event.target.closest(".nbv-run");
      if (!button) return;
      const node = button.closest(".nbv-cell");
      if (node) _runCell(node);
    });
    // An edit updates the source the node carries. `innerText` is accurate
    // here because a cell being typed into is by definition on screen — and
    // this is the only moment it is read, so a collapsed cell never is.
    body.addEventListener("input", (event) => {
      const code = event.target.closest(".nbv-src code");
      if (!code) return;
      const node = code.closest(".nbv-cell");
      if (node) node._ddSource = (code.innerText || "").replace(/ /g, " ");
    });
    host.querySelector(".nbv-back").onclick = () => showList();
    host.querySelector(".nbv-toc").onchange = (event) => {
      const target = event.target.value && document.getElementById(`nbv-${event.target.value}`);
      if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
      event.target.value = "";
    };
    host.querySelector(".nbv-restart").onclick = async () => {
      await window.DeltaKernel?.reset();
      state.checkerRan = false;
      state.runSeq = 0;
      host.querySelectorAll(".nbv-cell.has-run").forEach((cell) => cell.classList.add("is-stale"));
      state.recorded = new Set();
      _banner(
        "Session thrown away. Run the checker cell again before checking anything.",
        "warn",
        state,
      );
    };

    if (!window.DeltaKernel || !window.DeltaKernel.available()) {
      _banner(
        "You are reading this signed out, so the Run buttons have nothing to run. " +
          "Sign in and the whole notebook shares one live Python session.",
        "warn",
        state,
      );
    }
  };

  /* ---------- the index screen ----------------------------------------- */

  const _cardHtml = (entry) =>
    `<button type="button" class="nbv-card" data-id="${esc(entry.id)}">` +
    `<span class="nbv-card-title">${esc(entry.title)}</span>` +
    `<span class="nbv-card-meta">${entry.cells} cells · ${entry.questions.length} problems</span>` +
    `<span class="nbv-card-id">${esc(entry.id)}</span>` +
    "</button>";

  const _host = () => document.getElementById("notebooks-host");

  const showList = async () => {
    const host = _host();
    if (!host) return;
    current = null;
    host.innerHTML = '<p class="nbv-loading">Loading notebooks…</p>';
    try {
      if (!manifest) manifest = await _fetchJson(`${DIR}manifest.json`);
    } catch (err) {
      host.innerHTML =
        '<p class="nbv-error">The notebooks could not be loaded. Reload to try again.</p>';
      console.warn("[notebook-view] manifest unavailable:", err);
      return;
    }
    host.innerHTML =
      '<div class="nbv-intro"><h2>Lesson notebooks</h2>' +
      "<p>The whole lesson, every block runnable, all of it in one live Python " +
      "session — the same notebooks the Colab edition publishes.</p></div>" +
      '<div class="nbv-cards">' +
      manifest.lessons.map(_cardHtml).join("") +
      "</div>";
    host.querySelector(".nbv-cards").addEventListener("click", (event) => {
      const card = event.target.closest(".nbv-card");
      if (card) open(card.dataset.id);
    });
  };

  const open = async (lessonId) => {
    const host = _host();
    if (!host) return false;
    host.innerHTML = '<p class="nbv-loading">Loading the notebook…</p>';
    let nb;
    try {
      nb = await _fetchJson(`${DIR}${encodeURIComponent(lessonId)}.json`);
    } catch (err) {
      host.innerHTML =
        `<p class="nbv-error">No notebook for "${esc(lessonId)}".</p>` +
        '<p><button type="button" class="nbv-back">← All notebooks</button></p>';
      host.querySelector(".nbv-back").onclick = () => showList();
      console.warn("[notebook-view] notebook unavailable:", err);
      return false;
    }
    _render(nb, host);
    window.scrollTo({ top: 0 });
    return true;
  };

  return { showList, open };
})();

window.LessonNotebookView = LessonNotebookView;

/* Entry points.

   The tab renders the index the first time it is shown, and `?notebook=<id>`
   opens one directly — the same deep link the Colab edition has, pointed at
   this edition instead. */
(function () {
  let listed = false;
  const requested = new URLSearchParams(location.search).get("notebook");

  const show = () => {
    document.querySelectorAll(".tab").forEach((tab) =>
      tab.classList.toggle("active", tab.dataset.tab === "notebooks"));
    document.querySelectorAll(".page").forEach((page) =>
      page.classList.toggle("hidden", page.id !== "page-notebooks"));
  };

  const start = () => {
    document.querySelectorAll('.tab[data-tab="notebooks"]').forEach((tab) => {
      tab.addEventListener("click", () => {
        if (listed) return;
        listed = true;
        LessonNotebookView.showList();
      });
    });
    if (requested) {
      listed = true;
      show();
      LessonNotebookView.open(requested);
    }
  };

  if (document.readyState === "loading") {
    window.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
