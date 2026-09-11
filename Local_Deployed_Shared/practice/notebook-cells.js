/* ================================================================
   NOTEBOOK CELLS — the one definition of what a cell LOOKS like

   WHY THIS FILE EXISTS

   Three surfaces in this app show runnable code inside prose:

     practice/notebook.js       the lesson GATE — one concept's fences, turned
                                into cells inside the practice panel
     practice/notebook-view.js  the Notebooks tab — a whole compiled lesson
     practice/arena-notebook.js an ARENA section, opened in the app

   They already shared what a cell IS: `LessonNotebook.runSource`, one
   `_delta_cell` harness, one markdown renderer, and (since 2026-09-09) one
   editor in practice/notebook-code-edit.js. What they did NOT share was what a
   cell LOOKS like. The lesson gate built `.nb-cell` — a Run BAR under the code
   with a text button and a status line — and the other two built `.nbv-cell`,
   a Run GUTTER down the left with a ▶. Two rectangles, two stylesheets, two
   copies of the run-state painting, and a learner who moved from a lesson to
   the notebook of that same lesson saw the same code in two different frames.

   Seth, 2026-09-10: "I want the lessons in notebook style format rather than
   what they currently are", and on reusing this renderer rather than restyling
   the old one: "it is better to reuse something rather than creating another
   surface with more bugs. I want them to be the same thing."

   So: ONE builder, ONE set of class names (`.nbv-*` — the notebook's, because
   two of the three surfaces already spoke it), ONE run-state painter. 🪦 The
   `.nb-cell` family is gone; styles/practice/notebook.css keeps only the
   lesson page's LAYOUT.

   WHAT THIS FILE IS NOT

   It does not run anything and it does not know what a kernel is. The three
   surfaces genuinely differ in how a cell RUNS — the lesson gate replays every
   cell above the one you clicked when there is no kernel, the notebook view
   refuses to run at all without one, because replaying a prefix of 656 cells
   is the precise thing the kernel was built to abolish. That difference is
   real and stays with the callers. What is shared here is the DOM, the class
   names and the three lines of state painting that were copied between them.
   ================================================================ */

const DeltaNotebookCells = (() => {
  "use strict";

  const esc = (value) =>
    String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

  /* Checks belong to the runnable source, but never to the editable example:
     the learner edits and re-runs the code above the marker, and the
     assertions below it go along for the ride without ever being on screen.

     🔴 THE OWNER OF THIS MOVED HERE. It was in practice/notebook.js and
     notebook-view.js reached across for it as `LessonNotebook.splitChecks`,
     which put a load-order dependency between two files that otherwise had
     none. It is a statement about what a cell's source IS, so it belongs with
     the thing that builds cells. `LessonNotebook.splitChecks` still exists and
     delegates here — it is a published name and the ARENA sheets call it. */
  const CHECK_MARKER = "\n# Hidden checks\n";
  const splitChecks = (source) => {
    const text = String(source || "");
    const at = text.indexOf(CHECK_MARKER);
    return at < 0
      ? { code: text, checks: "" }
      : { code: text.slice(0, at).trimEnd(), checks: text.slice(at) };
  };

  /* ---------- building ------------------------------------------------- */

  /* One code cell.

     `source` is split on the marker above and the two halves are carried ON
     THE NODE — `_ddSource` for what is shown and editable, `_ddChecks` for
     what runs with it but is never rendered.

     🔴 `_ddSource` IS THE SOURCE OF TRUTH, NOT THE DOM. Read the text back out
     of the rendered cell and a cell inside a collapsed `<details>` — every
     solution, every hints block — returns the empty string, because
     `innerText` is defined in terms of LAYOUT. That shipped once: the learner
     opened a solution, pressed Run, was told "✓ ran successfully", and the
     check below still said `solve` is not defined. `readSource` below reads
     the property first and the DOM only as a fallback.

     The `<pre><code contenteditable="plaintext-only">` shape is not cosmetic
     either: it is exactly what practice/notebook-code-edit.js attaches to, so
     a cell built here gets the tokeniser, the ghost completion, Tab/Shift+Tab,
     its own undo stack and Ctrl/Shift+Enter from the same file on all three
     surfaces. Change the markup here and the selector constants at the top of
     that file have to change with it. */
  const codeCell = ({
    source = "",
    id = "",
    role = "code",
    q = null,
    index = null,
    editable = true,
    /* 🔴 THE ELEMENT ID IS PREFIXED, AND THE PREFIX IS THE SURFACE'S. The
       notebook view mints `nbv-<cell id>` and jumps to it by that name from
       its contents `<select>` and its beacon; the ARENA page mints
       `arena-<cell id>` and its own saved-output store and run harness key off
       that expression. Two surfaces, two id namespaces, one builder — so the
       prefix is an argument rather than something this file decides. */
    idPrefix = "nbv-",
    // The lesson gate numbers its cells `In [3]` the way a Jupyter prompt
    // does; the notebook view, which shows hundreds of them beside a 52px
    // gutter, numbers them `[3]`. Same counter, different room for it.
    countPrefix = "",
  } = {}) => {
    const el = document.createElement("section");
    el.className = "nbv-cell nbv-code";
    el.dataset.role = role;
    if (id) {
      el.id = `${idPrefix}${id}`;
      el.dataset.cellId = id;
    }
    if (q != null) el.dataset.q = String(q);
    if (index != null) el.dataset.nbIndex = String(index);
    el.dataset.countPrefix = countPrefix;
    const parts = splitChecks(String(source == null ? "" : source));
    el._ddSource = parts.code;
    el._ddChecks = parts.checks;
    el.innerHTML =
      '<div class="nbv-gutter">' +
      '<button type="button" class="nbv-run" title="Run this cell" aria-label="Run this cell">▶</button>' +
      '<span class="nbv-count" aria-hidden="true"></span>' +
      "</div>" +
      '<div class="nbv-body">' +
      (editable
        ? '<pre class="nbv-src"><code contenteditable="plaintext-only" spellcheck="false">' +
          esc(el._ddSource) +
          "</code></pre>"
        : "") +
      // Empty until a run says something about itself ("running cells 1–4",
      // "restoring cells 1–7"). `:empty` hides it, so a cell that never has
      // anything to say never reserves a line for it.
      '<p class="nbv-status" aria-live="polite"></p>' +
      '<pre class="nbv-out hidden"></pre>' +
      "</div>";
    return el;
  };

  /* A markdown cell. `html` is already rendered — this file has no opinion
     about markdown, and the app has exactly one renderer. */
  const mdCell = (html, { id = "", role = "prose", q = null, idPrefix = "nbv-" } = {}) => {
    const el = document.createElement("section");
    el.className = "nbv-cell nbv-md";
    el.dataset.role = role;
    if (id) el.id = `${idPrefix}${id}`;
    if (q != null) el.dataset.q = String(q);
    el.innerHTML = String(html == null ? "" : html);
    return el;
  };

  /* A disclosure cell — hints, a solution. `body` is an element. */
  const detailsCell = ({
    summary,
    body,
    extraClass = "",
    id = "",
    role = "",
    q = null,
    open = false,
    idPrefix = "nbv-",
  }) => {
    const el = document.createElement("details");
    el.className = `nbv-cell ${extraClass}`.trim();
    if (role) el.dataset.role = role;
    if (id) el.id = `${idPrefix}${id}`;
    if (q != null) el.dataset.q = String(q);
    if (open) el.open = true;
    const head = document.createElement("summary");
    head.textContent = summary;
    el.appendChild(head);
    el.appendChild(body);
    return el;
  };

  /* ---------- reading -------------------------------------------------- */

  /* What this cell will hand the runtime.

     TWO READERS, AND WHICH ONE WINS DEPENDS ON WHETHER THE EDITOR IS LOADED.

     With practice/notebook-code-edit.js attached — the normal case on all
     three surfaces — `_ddSource` is authoritative. It is seeded at build time
     and rewritten on every edit, and it has to be preferred over the DOM
     because `innerText` is defined in terms of LAYOUT: a cell inside a
     collapsed `<details>` (every solution, every hints block) reads back as
     the empty string. That shipped once — the learner opened a solution,
     pressed Run, was told "✓ ran successfully", and the check below still said
     `solve` is not defined.

     🔴 WITHOUT THAT FILE, THE DOM WINS, and that is not a stylistic fallback —
     it is the only thing keeping an edit honoured. Nothing else writes
     `_ddSource`, so preferring it unconditionally would mean a learner typing
     into a cell, pressing Run, and watching the ORIGINAL source execute, with
     their own edit still on screen above the output. The lesson gate used to
     read the DOM and only the DOM, so this is exactly the behaviour it had.

     NBSP goes back to a space either way: a contenteditable substitutes one
     for a space it thinks would collapse, and Python does not accept it. */
  const readSource = (node) => {
    if (!node) return "";
    const checks = node._ddChecks || "";
    const code = node.querySelector(".nbv-src code");
    const read = window.DeltaNotebookCode && window.DeltaNotebookCode.readText;
    const edited = typeof read === "function";
    if (node._ddSource != null && (edited || !code)) return node._ddSource + checks;
    if (!code) return checks;
    const text = edited ? read(code) : code.innerText || code.textContent;
    return String(text || "").replace(/\u00a0/g, " ") + checks;
  };

  /* ---------- run-state painting --------------------------------------- */

  /* The three lines that were copied between notebook.js and notebook-view.js,
     in one place. They paint the cell; they do not decide anything. */

  const parts = (node) => ({
    button: node.querySelector(".nbv-run"),
    out: node.querySelector(".nbv-out"),
    count: node.querySelector(".nbv-count"),
    status: node.querySelector(".nbv-status"),
  });

  const setStatus = (node, message) => {
    const status = node && node.querySelector(".nbv-status");
    if (status) status.textContent = message || "";
  };

  /* The cell is about to run: button out of reach, output cleared and shown,
     counter to `[*]`. Returns the cell's parts so the caller does not look
     them up a second time. */
  const begin = (node, { status = "" } = {}) => {
    const p = parts(node);
    if (p.button) p.button.disabled = true;
    node.classList.add("is-running");
    node.classList.remove("is-stale");
    if (p.count) p.count.textContent = `${node.dataset.countPrefix || ""}[*]`;
    if (p.status) p.status.textContent = status || "";
    if (p.out) {
      p.out.classList.remove("hidden", "is-error");
      p.out.textContent = "";
    }
    return p;
  };

  /* The run is over, whatever happened. `seq` is the caller's counter — the
     numbers say what ran and in what order, like a notebook's In[] prompt,
     rather than how many times each cell has been clicked, so the counter
     belongs to the PAGE and not to the cell. */
  const finish = (node, { text = "", failed = false, seq = null } = {}) => {
    const p = parts(node);
    if (p.out) {
      p.out.textContent = text;
      p.out.classList.toggle("is-error", !!failed);
    }
    node.classList.remove("is-running");
    node.classList.add("has-run");
    node.classList.toggle("has-failed", !!failed);
    if (p.count && seq != null) {
      p.count.textContent = `${node.dataset.countPrefix || ""}[${seq}]`;
    }
    if (p.status) p.status.textContent = "";
    if (p.button) p.button.disabled = false;
  };

  /* The run never started — no kernel, or the surface refuses to run this
     cell. Says why in the output slot and leaves the cell unmarked: nothing
     ran, so `has-run` would be a lie. */
  const refuse = (node, message) => {
    const p = parts(node);
    if (p.out) {
      p.out.classList.remove("hidden");
      p.out.classList.add("is-error");
      p.out.textContent = message;
    }
    node.classList.remove("is-running");
    if (p.button) p.button.disabled = false;
    if (p.status) p.status.textContent = "";
  };

  return {
    esc,
    splitChecks,
    codeCell,
    mdCell,
    detailsCell,
    readSource,
    begin,
    finish,
    refuse,
    setStatus,
  };
})();

window.DeltaNotebookCells = DeltaNotebookCells;
