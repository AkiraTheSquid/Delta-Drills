/* Colab-like practice notebook. Cells share DeltaKernel/Pyodide state; each
   runs independently, owns output, survives session pause, joins for grading. */
const DeltaNotebook = (() => {
  "use strict";

  const host = document.getElementById("practice-notebook");
  const cellsHost = document.getElementById("notebook-cells");
  const addBtn = document.getElementById("notebook-add-cell");
  const primary = document.querySelector('.notebook-cell[data-cell-id="1"]');
  let nextId = 2;
  let localExecCount = 0;

  const cells = () => Array.from(cellsHost?.querySelectorAll(".notebook-cell") || []);
  const editorOf = (cell) => cell?.querySelector(".notebook-cell-editor");
  const outputOf = (cell) => cell?.querySelector(".output-area");
  const outputShellOf = (cell) => cell?.querySelector("[data-cell-output]");
  const runOf = (cell) => cell?.querySelector(".notebook-cell-run");
  const execOf = (cell) => cell?.querySelector(".notebook-cell-exec");

  /* A cell is as tall as its code. No ceiling: a Colab cell grows and the
     NOTEBOOK scrolls (`.practice-notebook` is the `overflow-y: auto` one),
     which is not the same thing as a cell that stops at 420px and scrolls
     inside itself. The inner scrollbar was the worse of the two — it hides
     how much code there is, it steals the wheel from the pane, and the line
     the learner is typing on can sit under the fold of a box that has room
     to grow. The 96px floor stays: an empty cell still has to be a target
     you can click.

     `height: auto` first is not optional — `scrollHeight` on an element
     with an explicit height is that height, so without the reset a cell can
     grow but never shrink back. */
  const resize = (editor) => {
    if (!editor) return;
    if (editor.__deltaHBorders === undefined) {
      /* `scrollHeight` is content+padding and EXCLUDES the border, but
         base.css puts the whole app in `box-sizing: border-box`, so the
         height we assign has to include it. Setting height = scrollHeight
         therefore left every auto-sized cell 2px short of its own content —
         which is why a cell that had just been measured to fit still showed
         an inner scrollbar. Read once per element: the border width is a
         constant of the stylesheet, and this runs on every keystroke. */
      const box = getComputedStyle(editor);
      editor.__deltaHBorders = box.boxSizing === "border-box"
        ? (parseFloat(box.borderTopWidth) || 0) + (parseFloat(box.borderBottomWidth) || 0)
        : 0;
    }
    editor.style.height = "auto";
    editor.style.height = `${Math.max(96, editor.scrollHeight + editor.__deltaHBorders)}px`;
  };

  const markRun = (cell, result = {}) => {
    const exec = execOf(cell);
    if (!exec) return;
    const n = Number(result.execCount) || ++localExecCount;
    exec.textContent = `[${n}]`;
    exec.setAttribute("aria-label", `Executed as cell ${n}`);
    exec.classList.toggle("notebook-cell-exec--error", !!result.failed);
    cell.dataset.executed = "1";
    if (result.fresh) {
      cells().filter((other) => other !== cell).forEach((other) => {
        other.classList.toggle("notebook-cell--stale", other.dataset.executed === "1");
      });
    }
    cell.classList.remove("notebook-cell--stale");
  };

  const runCell = async (cell) => {
    const editor = editorOf(cell);
    const output = outputOf(cell);
    const outputShell = outputShellOf(cell);
    const button = runOf(cell);
    if (!editor || !output || !button || !window.DeltaRunner) return;
    outputShell?.classList.remove("hidden");
    button.disabled = true;
    button.textContent = "…";
    output.textContent = "Running…";
    try {
      const question = window.LessonGate?.activeQuestion || window.PracticeAPI?.currentQuestion;
      let result = await window.LessonNotebook?.runSource(editor.value, {
        context: "practice-editor",
        name: `<cell ${cell.dataset.cellId}>`,
        echo: true,
      });
      if (!result) {
        result = await window.DeltaRunner.runSnippet(editor.value, {
          question,
          source: editor.value,
          onStatus: (message) => { if (message) output.textContent = message; },
        });
      }
      output.textContent = result.text;
      markRun(cell, result);
      if (runtimeStatus && Number(result.execCount)) {
        runtimeStatus.textContent = result.fresh
          ? "Persistent runtime started"
          : `Persistent runtime · ${result.execCount} cells`;
      }
      if (cell === primary && !result.failed && result.pyodide) {
        await window.DeltaRunner.renderRunOutputVisual(result.pyodide, question);
      }
    } catch (err) {
      output.textContent = `Error: ${err.message || String(err)}`;
      markRun(cell, { failed: true });
    } finally {
      button.disabled = false;
      button.textContent = "▶";
    }
  };

  const focusNext = (cell, create = false) => {
    const list = cells();
    const at = list.indexOf(cell);
    let next = list[at + 1];
    if (!next && create) next = addCell("");
    editorOf(next)?.focus();
  };

  const bindCell = (cell) => {
    if (!cell || cell.dataset.notebookBound === "1") return;
    cell.dataset.notebookBound = "1";
    const editor = editorOf(cell);
    const button = runOf(cell);
    window.DeltaRunner?.installCodeEditorKeys(editor);
    editor?.addEventListener("input", () => resize(editor));
    /* Typing is not the only thing that changes how tall the code is.
         * `delta-editor-value-set` — runner.js announces the writes that
           fire no event of their own: ui.js loading a question's starter,
           events.js resetting, timer.js restoring a draft, lessons.js
           loading example code. Without this a long starter opens in a 96px
           box, which is the bug this whole change is about.
         * width — the pane narrows (the drawer, a window resize, the Chrome
           side panel) and the same code wraps onto more lines. Guarded on
           WIDTH so the height we just set cannot feed the observer back into
           itself. */
    editor?.addEventListener("delta-editor-value-set", () => resize(editor));
    if (editor && typeof ResizeObserver === "function") {
      let lastWidth = 0;
      new ResizeObserver(() => {
        const width = editor.clientWidth;
        if (width === lastWidth) return;
        lastWidth = width;
        resize(editor);
      }).observe(editor);
    }
    editor?.addEventListener("keydown", async (event) => {
      if (event.key !== "Enter" || (!event.shiftKey && !event.altKey && !event.ctrlKey && !event.metaKey)) return;
      event.preventDefault();
      await runCell(cell);
      if (event.altKey) focusNext(cell, true);
      else if (event.shiftKey) focusNext(cell, false);
    });
    if (button?.id !== "run-btn") button?.addEventListener("click", () => runCell(cell));
    cell.querySelector(".notebook-cell-delete")?.addEventListener("click", () => {
      if (cells().length <= 1) return;
      cell.remove();
      host?.dispatchEvent(new Event("input", { bubbles: true }));
    });
    resize(editor);
  };

  function addCell(code = "", saved = null) {
    if (!cellsHost) return null;
    const id = Number(saved?.id) || nextId++;
    nextId = Math.max(nextId, id + 1);
    const cell = document.createElement("section");
    cell.className = "notebook-cell";
    cell.dataset.cellId = String(id);
    cell.innerHTML = `
      <div class="notebook-cell-gutter">
        <button class="notebook-cell-run" type="button" aria-label="Run cell ${id}">▶</button>
        <span class="notebook-cell-exec" aria-label="Not run">[ ]</span>
      </div>
      <div class="notebook-cell-main">
        <div class="notebook-cell-actions">
          <span>Code</span>
          <button class="notebook-cell-delete" type="button" aria-label="Delete cell ${id}">×</button>
        </div>
        <textarea class="code-editor notebook-cell-editor" spellcheck="false" aria-label="Code cell ${id}"></textarea>
        <div class="notebook-cell-output hidden" data-cell-output>
          <div class="output-header">Output</div>
          <pre class="output-area"></pre>
        </div>
      </div>`;
    editorOf(cell).value = code;
    cellsHost.appendChild(cell);
    bindCell(cell);
    if (saved?.output) {
      outputOf(cell).textContent = saved.output;
      outputShellOf(cell).classList.remove("hidden");
    }
    if (saved?.execLabel) execOf(cell).textContent = saved.execLabel;
    return cell;
  }

  const reset = (code, { addScratch = true } = {}) => {
    if (!primary) return;
    cells().slice(1).forEach((cell) => cell.remove());
    const editor = editorOf(primary);
    editor.value = code || "";
    outputOf(primary).textContent = "";
    outputShellOf(primary)?.classList.add("hidden");
    execOf(primary).textContent = "[ ]";
    primary.dataset.executed = "";
    primary.classList.remove("notebook-cell--stale");
    nextId = 2;
    resize(editor);
    if (addScratch) addCell("");
  };

  const serialize = () => ({
    version: 1,
    cells: cells().map((cell) => ({
      id: Number(cell.dataset.cellId),
      code: editorOf(cell)?.value || "",
      output: outputOf(cell)?.textContent || "",
      execLabel: execOf(cell)?.textContent || "[ ]",
    })),
  });

  const restore = (draft) => {
    if (typeof draft === "string") {
      reset(draft);
      return;
    }
    if (!draft?.cells?.length) return;
    reset(draft.cells[0].code, { addScratch: false });
    if (draft.cells[0].output) {
      outputOf(primary).textContent = draft.cells[0].output;
      outputShellOf(primary)?.classList.remove("hidden");
    }
    if (draft.cells[0].execLabel) execOf(primary).textContent = draft.cells[0].execLabel;
    draft.cells.slice(1).forEach((saved) => addCell(saved.code, saved));
  };

  const submissionCode = () => cells()
    .map((cell, index) => {
      const code = editorOf(cell)?.value.trimEnd() || "";
      return code ? `# --- cell ${index + 1} ---\n${code}` : "";
    })
    .filter(Boolean)
    .join("\n\n");

  bindCell(primary);
  addBtn?.addEventListener("click", () => editorOf(addCell(""))?.focus());
  if (outputArea && outputShellOf(primary)) {
    new MutationObserver(() => {
      outputShellOf(primary).classList.toggle("hidden", !outputArea.textContent);
    }).observe(outputArea, { childList: true, characterData: true, subtree: true });
  }

  return { addCell, markRun, reset, restore, runCell, serialize, submissionCode };
})();

window.DeltaNotebook = DeltaNotebook;
