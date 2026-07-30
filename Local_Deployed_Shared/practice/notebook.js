/* ================================================================
   LESSON NOTEBOOK — runnable code cells inside the explanation

   The lesson screen used to be the practice split: prose on the left, one
   editor on the right holding the whole worked example. That layout asks the
   learner to hold a mapping in their head — "the paragraph I am reading
   corresponds to *some* part of the block over there" — and the mapping gets
   harder exactly as the example gets longer.

   A notebook removes the mapping. Each code block sits where the prose that
   explains it sits, and each one runs on its own, so "what does this line
   actually do" is answered by pressing Run beside the line rather than by
   scrolling a second panel and deleting the parts you are not asking about.

   WHICH BLOCKS BECOME CELLS

   Only the ones inside the worked example. A lesson's explanation also
   contains fenced code, but those are illustrations, not programs — snippets
   like `t.arange(6).reshape(2, 3)   # → [[0, 1, 2], [3, 4, 5]]`, written to be
   read beside the sentence that explains them. They name no imports and often
   are not even whole statements, so a Run button on one is a button that is
   guaranteed to produce a traceback. Measured across the current content:
   122 segments carry 122 worked-example blocks, every one of which opens with
   its own import and runs standalone, against 13 illustrative blocks in the
   explanation bodies. The split is not a heuristic about the text; it is the
   structural distinction the lesson format already makes.

   HOW STATE WORKS ACROSS CELLS

   Neither runtime keeps a session between calls: the backend runs each request
   in a fresh fork, and the Pyodide preamble resets stdout each time. So a cell
   that says `flat = t.arange(9)` and a later one that says `flat.reshape(3, 3)`
   would fail on its own.

   Running a cell therefore executes every cell above it as well, concatenated,
   and shows only the output. That gives ordinary notebook semantics on a
   stateless runner. The cost is honest and worth naming: editing an early cell
   changes what the later ones see, exactly as it would in Jupyter, and a slow
   early cell is paid for again by every cell below it. The alternative — a
   persistent per-learner interpreter — is a server-side session to build,
   expire and secure, which is not worth it for optional experimentation.

   Every worked example in the bank today holds exactly one block, so this
   never actually fires. It is here because the moment a lesson is authored
   with two, the alternative is a second block that fails on a name the first
   one defined — a failure the author would have no reason to expect.

   WHAT RUNS, AND FOR WHOM

   Delegated wholesale to `DeltaRunner.runSnippet`, which owns the
   backend-vs-Pyodide decision for the whole app. The consequence worth
   knowing: the lessons are torch since the July dialect conversion, and
   Pyodide cannot import torch, so cells genuinely execute only for signed-in
   learners. Guests get the runner's plain explanation rather than a
   ModuleNotFoundError, and the lesson still reads correctly — the code is
   printed with its expected results either way.
   ================================================================ */

const LessonNotebook = (() => {
  "use strict";

  const esc = (value) =>
    String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

  /* One cell's markup. The code is rendered as plain text rather than an
     editable field: these are the lesson's examples, and an edit box invites
     the learner to lose the example they were given. `contenteditable` on the
     <code> keeps experimentation possible without turning the page into a
     form — the text is read back at Run time, so an edit is honoured. */
  const cellHtml = (code, index) =>
    '<div class="nb-cell" data-nb-index="' + index + '">' +
    '<div class="nb-cell-code">' +
    '<pre><code contenteditable="plaintext-only" spellcheck="false">' +
    esc(code) +
    "</code></pre>" +
    "</div>" +
    '<div class="nb-cell-bar">' +
    '<button type="button" class="nb-run">Run</button>' +
    '<span class="nb-status" aria-live="polite"></span>' +
    "</div>" +
    '<pre class="nb-out hidden"></pre>' +
    "</div>";

  const _codeOf = (cell) => {
    const node = cell.querySelector(".nb-cell-code code");
    return node ? node.innerText.replace(/ /g, " ") : "";
  };

  /* Every cell up to and including this one, joined. See the note above on why
     the whole prefix is re-run rather than only the cell that was clicked. */
  const _programUpTo = (cells, index) =>
    cells
      .slice(0, index + 1)
      .map(_codeOf)
      .join("\n\n");

  const _runCell = async (cells, index) => {
    const cell = cells[index];
    const button = cell.querySelector(".nb-run");
    const status = cell.querySelector(".nb-status");
    const out = cell.querySelector(".nb-out");
    if (!button || !out) return;

    button.disabled = true;
    button.textContent = "Running…";
    status.textContent = index > 0 ? `running cells 1–${index + 1}` : "";
    out.classList.remove("hidden", "is-error");
    out.textContent = "";

    try {
      const result = await window.DeltaRunner.runSnippet(_programUpTo(cells, index), {
        question: window.LessonGate?.activeQuestion || null,
        onStatus: (message) => {
          if (message) out.textContent = message;
        },
      });
      out.textContent = result.text;
      out.classList.toggle("is-error", !!result.failed);
    } catch (err) {
      out.textContent = "Error: " + err.message;
      out.classList.add("is-error");
    }

    status.textContent = "";
    button.disabled = false;
    button.textContent = "Run";
  };

  /* Turn the fenced code blocks inside `host` into runnable cells.

     `host` is the worked-example container, NOT the whole page — see the note
     above on why the explanation's illustrative blocks stay static.

     Operates on rendered DOM rather than on markdown so there is exactly one
     markdown renderer in the app: the lesson body, the ladder's inline example
     and the standalone viewer all keep producing the same `<pre><code>`, and
     this decides which of those become interactive. */
  const mount = (host) => {
    if (!host || !window.DeltaRunner) return 0;
    const blocks = Array.from(host.querySelectorAll("pre > code")).filter(
      (node) => !node.closest(".nb-cell"),
    );
    blocks.forEach((node, index) => {
      const pre = node.parentElement;
      const wrapper = document.createElement("div");
      wrapper.innerHTML = cellHtml(node.textContent, index);
      pre.replaceWith(wrapper.firstElementChild);
    });

    const cells = Array.from(host.querySelectorAll(".nb-cell"));
    cells.forEach((cell, index) => {
      const button = cell.querySelector(".nb-run");
      if (button) button.onclick = () => _runCell(cells, index);
    });
    return cells.length;
  };

  return { mount };
})();

window.LessonNotebook = LessonNotebook;
