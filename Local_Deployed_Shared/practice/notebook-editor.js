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

  /* 🔴 THE LEARNER'S CELLS, WHICH IS NOT THE SAME AS EVERY CELL ON SCREEN.
     The reference solution is appended as a real `.notebook-cell` below them
     (showSolution), so that it sits under the code you typed and runs like any
     other cell. Everything that treats a cell as the learner's WORK has to
     skip it, and this one selector is how:

       submissionCode()  — otherwise Submit posts the answer key as the answer
       serialize()       — otherwise the saved draft carries the solution into
                           the next session, and restore() re-lays it out as
                           an ordinary editable cell with a delete button
       reset()           — `cells().slice(1)` is what clears the notebook
                           between questions; clearSolution() handles this one
       markRun()         — staleness is about the learner's chain of cells

     A marker attribute rather than the class, because `.notebook-cell` is what
     the CSS grid and every existing handler key on and the solution cell wants
     all of that. */
  const cells = () =>
    Array.from(cellsHost?.querySelectorAll(
      ".notebook-cell:not([data-solution-cell]):not([data-example-cell])",
    ) || []);
  const solutionCell = () => cellsHost?.querySelector("[data-solution-cell]") || null;
  const exampleCells = () =>
    Array.from(cellsHost?.querySelectorAll("[data-example-cell]") || []);
  /* Where the learner's own cells stop and the graded feedback begins: the
     failed-case block (parented here by ui.js::renderFailedTests) and the
     answer both live at the tail of this list, and anything the learner adds
     belongs above them. querySelector returns the FIRST match in document
     order, so this is the boundary whichever of the two came first, and null
     — i.e. "append" — before a grade. */
  const feedbackBoundary = () =>
    cellsHost?.querySelector("#failed-tests-block, [data-solution-cell]") || null;
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
    /* Above the graded feedback, never below it. A learner who keeps
       experimenting after a wrong grade calls this while the tail of the list
       is already [failed cases, answer], and a plain append would bury their
       new scratch cell under both. The boundary is the FIRST of the two, not
       the solution cell alone — inserting before only the answer still drops
       the new cell below the failure report and breaks the intended reading
       order (your work → what failed → what it should have been).
       insertBefore(null) === appendChild, so an ungraded notebook is
       unchanged. */
    cellsHost.insertBefore(cell, feedbackBoundary());
    bindCell(cell);
    if (saved?.output) {
      outputOf(cell).textContent = saved.output;
      outputShellOf(cell).classList.remove("hidden");
    }
    if (saved?.execLabel) execOf(cell).textContent = saved.execLabel;
    return cell;
  }

  /* The reference answer, as a cell under the learner's own.

     Seth, 2026-08-24: "it needs to render BELOW the code you typed" — the
     solution has always been written to `#solution-code`, a dead <pre> at the
     BOTTOM of the left rail, under the question, the worked example and the
     Next problem button. It rendered; nobody scrolled that far, so in practice
     the app did not show you the answer. Putting it where your code is means
     you read the two side by side, and because it is a real cell you can run
     it and see what it prints instead of taking its word for it.

     Editable on purpose, exactly like Colab's: poking at the reference answer
     to see what breaks is the point. Nothing here is graded — the marker
     attribute keeps it out of submissionCode() — so an edit costs nothing. */
  const showSolution = (code) => {
    if (!cellsHost || !code) return null;
    /* 🔴 EXISTS IS NOT VISIBLE. #notebook-cells stays in the DOM on surfaces
       that hide the whole right pane — a torch drill routed out to Colab, the
       Colab edition, an idle session. Appending there would put the answer
       somewhere nobody can scroll to AND set dd-solution-in-notebook, which
       hides the left-rail copy that IS on screen: the learner would end up
       with no answer at all, which is the exact bug this feature fixes.
       Refusing here leaves the class unset, so the rail fallback survives.

       🔴 CLEAR BEFORE REFUSING. Unset is not the same as never set: a visible
       question that graded wrong leaves the class on, and if the NEXT question
       hides the pane an early return would keep the suppression while the only
       copy of the answer sits in a pane nobody can see — the same
       nothing-anywhere failure, arrived at from the other direction. */
    if (!cellsHost.getClientRects().length) {
      clearSolution();
      return null;
    }
    let cell = solutionCell();
    if (!cell) {
      cell = document.createElement("section");
      cell.className = "notebook-cell notebook-cell--solution";
      cell.dataset.solutionCell = "1";
      cell.dataset.cellId = "solution";
      cell.innerHTML = `
      <div class="notebook-cell-gutter">
        <button class="notebook-cell-run" type="button" aria-label="Run the solution">▶</button>
        <span class="notebook-cell-exec" aria-label="Not run">[ ]</span>
      </div>
      <div class="notebook-cell-main">
        <div class="notebook-cell-actions notebook-cell-actions--solution">
          <span>💡 Solution — the answer this was graded against</span>
        </div>
        <textarea class="code-editor notebook-cell-editor" spellcheck="false"
                  aria-label="Reference solution"></textarea>
        <div class="notebook-cell-output hidden" data-cell-output>
          <div class="output-header">Output</div>
          <pre class="output-area"></pre>
        </div>
      </div>`;
      /* 🔴 APPEND BEFORE BIND, the same order addCell uses, and for a reason
         that is invisible if you get it wrong. bindCell ends in resize(), and
         resize memoises the editor's border height ONCE per element from
         getComputedStyle. On a DETACHED node that read returns defaults, so
         `boxSizing === "border-box"` is false, the borders are recorded as 0,
         and every later resize lands 2px short of the content — a permanent
         hairline scrollbar on the answer, forever, because the memo is never
         recomputed. */
      cellsHost.appendChild(cell);
      bindCell(cell);
    }
    const editor = editorOf(cell);
    editor.value = code;
    /* The source just changed underneath any run the learner did on the
       previous question's answer, so the old output and its [n] marker now
       describe code that is no longer in the box. Clear both rather than
       leave a result sitting under source that did not produce it. */
    outputOf(cell).textContent = "";
    outputShellOf(cell)?.classList.add("hidden");
    execOf(cell).textContent = "[ ]";
    cell.dataset.executed = "";
    cell.classList.remove("notebook-cell--stale");
    // Re-appended every time, not inserted once: keeps the answer last even
    // if something else touched the list between grades.
    cellsHost.appendChild(cell);
    resize(editor);
    /* Tells basic-mode.css that the answer is already on screen, so the left
       rail's copy of it can stay hidden. 🔴 The rail copy is NOT dead code —
       it is the fallback for every question that has no notebook to append to
       (a torch drill routed to Colab hides the whole right panel; so does the
       Colab edition). Without this class the two would both show. */
    document.body.classList.add("dd-solution-in-notebook");
    return cell;
  };

  const clearSolution = () => {
    solutionCell()?.remove();
    document.body.classList.remove("dd-solution-in-notebook");
  };

  /* Scroll the notebook pane down to the answer.

     🔴 APPENDING IS NOT SHOWING. The cells above the solution are as tall as
     the code the learner just wrote, so the answer lands below the fold of
     `.practice-notebook` and the screen looks exactly like it did before the
     grade — which is the tester's original complaint arriving a second time
     ("it needs to automatically scroll down ... rather than not realizing
     that it's there"). The scroll is what turns an appended cell into a
     delivered one.

     Scrolls THAT pane, by arithmetic, rather than calling scrollIntoView:
     scrollIntoView walks every scrollable ancestor, so it also moves the
     document and the left rail underneath a learner who did not ask for it.

     The 56px lead-in is deliberate — it leaves the tail of the failed-case
     block on screen above the solution header, so the read stays "these
     cases failed → here is what it should have been" instead of dropping
     the learner straight onto an answer with no cause. */
  const SOLUTION_SCROLL_LEAD_IN = 56;
  /* `instant` for a RESTORE. Animating a scroll on a pane the learner has not
     looked at yet — one that was rebuilt a frame ago by a reload or the Resume
     button — animates from a position that was never on screen, so it reads as
     the page settling rather than as the app taking them somewhere. After a
     live submit the motion is the point: it says the grade moved you. */
  const scrollToSolution = ({ instant = false, retries = 0 } = {}) => {
    const cell = solutionCell();
    if (!cell || !cellsHost?.getClientRects().length) return false;
    const pane = cell.closest(".practice-notebook") || host;
    if (!pane || !pane.getClientRects().length) return false;
    const inPane = () => {
      const c = cell.getBoundingClientRect(), p = pane.getBoundingClientRect();
      return c.top < p.bottom && c.bottom > p.top;
    };
    const offset = cell.getBoundingClientRect().top - pane.getBoundingClientRect().top;
    const top = Math.max(0, pane.scrollTop + offset - SOLUTION_SCROLL_LEAD_IN);
    /* Honour the OS setting: a long smooth scroll is one of the motions
       `prefers-reduced-motion` exists for, and jumping there still shows it. */
    const reduced = instant || window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
    pane.scrollTo({ top, behavior: reduced ? "auto" : "smooth" });
    /* 🔴 A RESTORE SCROLLS INTO A PANE THAT IS STILL SETTLING, and a scroll
       past the current scrollHeight is CLAMPED, not queued — so it silently
       lands at 0. Resuming a paused review does three things after this runs:
       it un-hides the session status row, re-lays the restored cells, and only
       then is the pane as tall as its content. Observed on prod: the answer was
       in the notebook, correctly ordered, and the pane sat at scrollTop 0.
       Re-aim while the cell is still out of view rather than guess one delay. */
    if (retries > 0) {
      setTimeout(() => {
        if (!inPane()) scrollToSolution({ instant, retries: retries - 1 });
      }, 120);
    }
    return true;
  };

  /* The worked example's code, as runnable cells ABOVE the learner's own.

     The tester (2026-08-24): "have the python snippets that are currently in
     the left column WITHIN the code editor". Reading the example in the rail
     and typing in the editor are two different places; putting the same
     snippets at the top of the notebook means they can be run against the
     live kernel and copied from in place. Same non-learner-cell contract as
     the solution cell: `data-example-cell` keeps these out of submissionCode()
     (Submit must never post the scaffold as the answer), serialize() (a draft
     must not fossilise them — ladder.js re-supplies them per question), and
     reset()'s learner-cell sweep. Prose between the snippets stays in the rail
     — the cells are the runnable half, not a second copy of the lesson.

     `exampleSources` is module state, not just DOM, because reset() runs on
     every question render AND on a draft restore, and both need to re-lay the
     cells for the CURRENT question without waiting for another KP fetch.
     ladder.js clears it at the top of every decorate, so a question without an
     example cannot inherit the previous question's cells. */
  let exampleSources = [];

  const renderExamples = () => {
    exampleCells().forEach((cell) => cell.remove());
    if (!cellsHost || !primary || !exampleSources.length) return;
    // Same rule as showSolution: a hidden pane means the rail copy is the one
    // on screen — appending here would help nobody.
    if (!cellsHost.getClientRects().length) return;
    exampleSources.forEach((code, index) => {
      const n = index + 1;
      const label = exampleSources.length > 1
        ? `📖 Worked example ${n} of ${exampleSources.length} — run it, copy from it; your solution goes below`
        : "📖 Worked example — run it, copy from it; your solution goes below";
      const cell = document.createElement("section");
      cell.className = "notebook-cell notebook-cell--example";
      cell.dataset.exampleCell = "1";
      cell.dataset.cellId = `example-${n}`;
      cell.innerHTML = `
      <div class="notebook-cell-gutter">
        <button class="notebook-cell-run" type="button" aria-label="Run worked example ${n}">▶</button>
        <span class="notebook-cell-exec" aria-label="Not run">[ ]</span>
      </div>
      <div class="notebook-cell-main">
        <div class="notebook-cell-actions notebook-cell-actions--example">
          <span>${label}</span>
        </div>
        <textarea class="code-editor notebook-cell-editor" spellcheck="false"
                  aria-label="Worked example ${n}"></textarea>
        <div class="notebook-cell-output hidden" data-cell-output>
          <div class="output-header">Output</div>
          <pre class="output-area"></pre>
        </div>
      </div>`;
      editorOf(cell).value = code;
      // Attach before bind — bindCell's resize memoises border widths from
      // getComputedStyle, which returns defaults on a detached node (the same
      // 2px-short trap showSolution documents).
      cellsHost.insertBefore(cell, primary);
      bindCell(cell);
    });
  };

  const showExamples = (codes) => {
    exampleSources = (Array.isArray(codes) ? codes : [])
      .map((code) => String(code || "").trim())
      .filter(Boolean)
      .slice(0, 8);
    renderExamples();
  };

  const clearExamples = () => {
    exampleSources = [];
    exampleCells().forEach((cell) => cell.remove());
  };

  const reset = (code, { addScratch = true } = {}) => {
    if (!primary) return;
    clearSolution();
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
    // Re-lay the current question's example cells: a draft restore reaches
    // here through restore(), after ladder.js may already have supplied them,
    // and the sweep above must not cost the learner the scaffold.
    renderExamples();
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

  return {
    addCell, markRun, reset, restore, runCell, serialize, submissionCode,
    showSolution, clearSolution, scrollToSolution, showExamples, clearExamples,
  };
})();

window.DeltaNotebook = DeltaNotebook;
