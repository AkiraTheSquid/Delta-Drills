/* ================================================================
   PRACTICE RUNNER — Pyodide run button
   ================================================================ */

let pyodideInstance = null;
let pyodideLoading = false;
let pyodidePackagePromise = null;
let arenaNumbersPromise = null;

const normalizeOutput = (value) => (value || "").replace(/\r\n/g, "\n").trim();

function hideOutputVisual() {
  outputVisual.classList.add("hidden");
  outputVisualCanvas.classList.add("hidden");
  outputVisualNote.textContent = "";
  const ctx = outputVisualCanvas.getContext("2d");
  ctx.clearRect(0, 0, outputVisualCanvas.width || 1, outputVisualCanvas.height || 1);
}

async function renderRunOutputVisual(pyodide, question) {
  if (!question?.supports_visual_output || question?.submission_mode !== "function") {
    hideOutputVisual();
    return;
  }
  // Parameterized visual questions take real arguments — render from the
  // canonical case's call (e.g. solve(img); the starter defines the fixture
  // vars at module level, so they exist in the user's globals). Zero-param
  // legacy questions fall back to solve().
  const visualCall =
    (Array.isArray(question.test_cases) && question.test_cases[0]?.call) || "solve()";
  try {
    const payload = await pyodide.runPythonAsync(`
import json
import numpy as np

def _delta_to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_delta_to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_delta_to_jsonable(v) for v in value]
    return value

_delta_output_value = ${visualCall}
json.dumps(_delta_to_jsonable(_delta_output_value))
`);
    const parsed = JSON.parse(payload);
    window.renderDeltaArrayToCanvas(outputVisualCanvas, parsed);
    outputVisual.classList.remove("hidden");
    outputVisualNote.textContent = "Rendered from solve().";
  } catch (err) {
    hideOutputVisual();
  }
}

async function initPyodide() {
  if (pyodideInstance) return pyodideInstance;
  if (pyodideLoading) {
    while (pyodideLoading) {
      await new Promise((r) => setTimeout(r, 100));
    }
    return pyodideInstance;
  }
  pyodideLoading = true;
  try {
    pyodideInstance = await loadPyodide();
    await pyodideInstance.loadPackage("numpy");
  } catch (e) {
    console.warn("[runner] Pyodide failed to load:", e);
  }
  pyodideLoading = false;
  return pyodideInstance;
}

async function ensurePyodidePracticePackages({ needsEinops = false } = {}) {
  const pyodide = await initPyodide();
  if (!pyodide) return null;
  if (!needsEinops || pyodide.__deltaEinopsReady) return pyodide;
  if (!pyodidePackagePromise) {
    pyodidePackagePromise = (async () => {
      await pyodide.loadPackage("micropip");
      await pyodide.runPythonAsync(`
import micropip
await micropip.install("einops")
import einops
`);
      pyodide.__deltaEinopsReady = true;
    })().catch((err) => {
      pyodidePackagePromise = null;
      throw err;
    });
  }
  await pyodidePackagePromise;
  return pyodide;
}

async function ensureArenaNumbersInPyodide() {
  const pyodide = await initPyodide();
  if (!pyodide) return null;
  if (pyodide.__deltaArenaNumbersReady) return pyodide;
  if (!arenaNumbersPromise) {
    arenaNumbersPromise = (async () => {
      const res = await fetchArenaNumbersAsset();
      const bytes = new Uint8Array(await res.arrayBuffer());
      pyodide.FS.writeFile("/delta_numbers.npy", bytes);
      pyodide.__deltaArenaNumbersReady = true;
    })().catch((err) => {
      arenaNumbersPromise = null;
      throw err;
    });
  }
  await arenaNumbersPromise;
  return pyodide;
}

// Runtime contract for the in-browser Pyodide preamble is documented in
// practice/RUNTIME_CONTRACT.md. Per-question dependency declarations live in
// `runtime_dependencies` / `runtime_unmet_dependencies` on each question entry
// in arena_prereqs_structured.json. Keep this function in sync with that doc
// when adding/removing injected globals.
async function buildPyodidePreamble(
  question = window.LessonGate?.activeQuestion || PracticeAPI?.currentQuestion,
) {
  const needsEinops = questionNeedsEinops(question);
  const needsArenaArray = questionNeedsArenaArray(question);
  await ensurePyodidePracticePackages({ needsEinops });
  if (needsArenaArray) {
    await ensureArenaNumbersInPyodide();
  }

  // Pull fixtures from the test's setup_code so user code sees variables
  // defined by the question (e.g. hwcs, list_of_tensors). Skip
  // expected_setup_code on purpose — that block constructs the canonical
  // answer and would let `solve()` cheat by reading it.
  const testCase = Array.isArray(question?.test_cases) ? question.test_cases[0] : null;
  const testSetup = (testCase?.setup_code || "").trim();

  return `
import sys
from io import StringIO
sys.stdout = StringIO()
sys.stderr = StringIO()
import numpy as np
np.random.seed(0)
${needsEinops ? "import einops\nfrom einops import einsum, rearrange, reduce, repeat" : ""}
def display_array_as_img(*args, **kwargs):
    return None
${needsArenaArray ? "arr = np.load('/delta_numbers.npy')" : ""}
${testSetup}
`;
}

// Tab inside the code editor indents instead of jumping to the Run button.
// Tab => insert FOUR SPACES (or indent every selected line);
// Shift+Tab => dedent one Python indent.
// Accessibility escape hatch: press Escape first, then Tab moves focus out of
// the editor as usual (so keyboard-only users are never trapped).
//
// The indent unit is spaces, not "\t". Every starter in the bank is indented
// with 4 spaces — a scan of all 449 questions finds ZERO tab characters in any
// starter_code, answer_code or question_text. So a Tab keypress used to drop a
// real tab into space-indented code, and Python answered with
// `TabError: inconsistent use of tabs and spaces in indentation`. The learner
// had done nothing wrong; the editor manufactured the error. Matching the bank
// is the fix, and it has to stay matched: if starters are ever re-indented,
// change this with them.
const EDITOR_SPACE_INDENT_WIDTH = 4;
const EDITOR_INDENT = " ".repeat(EDITOR_SPACE_INDENT_WIDTH);
// Lines that end a block. After one of these, the next line dedents a level —
// nothing can follow `return` at the same depth inside the same suite.
const EDITOR_DEDENT_AFTER = /^\s*(return|pass|break|continue|raise)\b/;

/* ASSIGNING `editor.value` FIRES NOTHING, AND FIVE MODULES DO IT.

   `ui.js` prefills a question's starter code, `events.js` resets between
   questions, `timer.js` restores a saved draft, `lessons.js` loads example
   code, `notebook-editor.js` seeds a new cell — none of them dispatch an
   event, because a textarea does not need one. Two features now do: the
   syntax overlay has to repaint, and the cell has to re-measure its height.

   So the property is shadowed ON THE ELEMENT, delegating to the native
   descriptor and then announcing itself. `delta-editor-value-set` does NOT
   bubble and is NOT an `input` event on purpose — it is a repaint signal,
   not a learner edit, and the notebook's draft autosave listens for `input`
   on the whole pane.

   🔴 THIS IS THE ONLY PLACE THAT MAY PATCH `value` ON A CODE EDITOR. A
   second patch installed later would capture the PROTOTYPE descriptor, not
   this one, so its setter would bypass this announcement entirely and
   whichever feature got here first would silently stop updating. Anything
   that needs to know a value was written listens for the event. */
function announceValueWrites(editor) {
  const native = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value");
  Object.defineProperty(editor, "value", {
    configurable: true,
    get() { return native.get.call(this); },
    set(v) {
      native.set.call(this, v);
      this.dispatchEvent(new Event("delta-editor-value-set"));
    },
  });
}

function installCodeEditorKeys(editor) {
  if (!editor || editor.dataset.deltaEditorKeys === "1") return;
  editor.dataset.deltaEditorKeys = "1";
  /* Every code cell in the app reaches this function exactly once — the one
     in index.html below, and every cell notebook-editor.js mints — which is
     why the editor's shared plumbing is hung here rather than at five call
     sites that would each have to remember.

     Order is load-bearing: the announcement has to be in place before
     anything that listens for it, and the highlighter builds the
     `.code-surface` wrapper and the overlay that draws the ghost, which the
     completion layer needs. Both layers are optional (`?.`): if either
     script is missing the editor is exactly the plain textarea it was
     before, keys and all. */
  announceValueWrites(editor);
  window.DeltaCodeHighlight?.attach(editor);
  window.DeltaCodeComplete?.attach(editor);
  let editorTabEscapes = false;
  editor.addEventListener("keydown", (e) => {
    // Enter keeps the indent you are already at, so the learner does not have
    // to re-tab into a function body on every single line. `:` opens a suite so
    // the next line goes one level deeper; a block-ending statement closes one.
    // Only fires on a bare Enter with no selection — Shift/Ctrl/Alt+Enter and
    // Enter-over-a-selection keep their normal behaviour.
    if (
      e.key === "Enter" &&
      !e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey &&
      editor.selectionStart === editor.selectionEnd
    ) {
      const el = editor;
      const at = el.selectionStart;
      const lineStart = el.value.lastIndexOf("\n", at - 1) + 1;
      const line = el.value.slice(lineStart, at);
      let indent = (line.match(/^[ \t]*/) || [""])[0];
      // Only the code BEFORE the caret decides the next indent — pressing Enter
      // mid-line splits it, and the trailing half is what moves down.
      const codeBefore = line.replace(/#.*$/, "").trimEnd();
      if (codeBefore.endsWith(":")) {
        indent += EDITOR_INDENT;
      } else if (EDITOR_DEDENT_AFTER.test(line) && indent.length >= EDITOR_INDENT.length) {
        indent = indent.slice(0, indent.length - EDITOR_INDENT.length);
      }
      if (indent) {
        e.preventDefault();
        const insert = "\n" + indent;
        el.value = el.value.slice(0, at) + insert + el.value.slice(el.selectionEnd);
        el.selectionStart = el.selectionEnd = at + insert.length;
        el.dispatchEvent(new Event("input", { bubbles: true }));
      }
      editorTabEscapes = false;
      return;
    }
    if (e.key === "Escape") {
      // Arm a one-shot "let Tab leave the field" so keyboard users can escape.
      editorTabEscapes = true;
      return;
    }
    if (e.key !== "Tab") {
      editorTabEscapes = false;
      return;
    }
    if (editorTabEscapes) {
      // Let this Tab move focus normally, then re-arm capture.
      editorTabEscapes = false;
      return;
    }
    e.preventDefault();
    const el = editor;
    const value = el.value;
    const start = el.selectionStart;
    const end = el.selectionEnd;
    const selectsMultipleLines = value.slice(start, end).includes("\n");

    if (!e.shiftKey && !selectsMultipleLines) {
      // Simple caret indent: insert the indent unit at the caret.
      el.value = value.slice(0, start) + EDITOR_INDENT + value.slice(end);
      el.selectionStart = el.selectionEnd = start + EDITOR_INDENT.length;
      return;
    }

    // Block (de)indent: operate on every line the selection touches.
    const lineStart = value.lastIndexOf("\n", start - 1) + 1;
    const block = value.slice(lineStart, end);
    const lines = block.split("\n");
    let newBlock;
    let removedFirst = 0;
    let removedTotal = 0;
    if (e.shiftKey) {
      newBlock = lines
        .map((line, i) => {
          let cut = 0;
          const leadingSpaces = line.match(/^ +/)?.[0].length || 0;
          if (line.startsWith(EDITOR_INDENT)) cut = EDITOR_INDENT.length;
          else if (leadingSpaces) cut = Math.min(leadingSpaces, EDITOR_SPACE_INDENT_WIDTH);
          if (i === 0) removedFirst = cut;
          removedTotal += cut;
          return line.slice(cut);
        })
        .join("\n");
    } else {
      newBlock = lines.map((line) => EDITOR_INDENT + line).join("\n");
    }
    el.value = value.slice(0, lineStart) + newBlock + value.slice(end);
    if (e.shiftKey) {
      el.selectionStart = Math.max(lineStart, start - removedFirst);
      el.selectionEnd = Math.max(el.selectionStart, end - removedTotal);
    } else {
      el.selectionStart = start + EDITOR_INDENT.length;
      el.selectionEnd = end + EDITOR_INDENT.length * lines.length;
    }
  });
}

installCodeEditorKeys(codeEditor);

/* `TORCH_IMPORT_RE`, `needsTorchRuntime` and `TORCH_UNAVAILABLE` live in ui.js
   so api.js can share them — the submit path needs exactly this rule, and a
   second copy of the regex here is how the two drifted apart in the first
   place (Run refused torch politely; Submit crashed on it). */
const TORCH_IMPORT = TORCH_IMPORT_RE;

/* Execute one block of Python and report what happened.

   The single place that decides WHERE code runs, so the editor's Run button
   and the lesson notebook's per-cell Run buttons cannot drift apart on that
   question. The rules, in order:

     * A lesson's example code and einops code stay in the browser — neither is
       being graded, and there is no reason to spend a backend round trip on
       experimentation.
     * ...unless it is torch, which Pyodide cannot import AT ALL. That covers
       most of the bank since the July conversion, so torch code goes to the
       backend fork runner even though the two rules above would keep it local.
       The code is sniffed as well as the question, so a learner who types
       `import torch` themselves still lands on a runtime that can execute it.
     * A guest has no backend to fall back to. Say so plainly rather than
       letting Pyodide answer torch with a bare ModuleNotFoundError.

   Returns { text, failed, blocked, pyodide }. `text` is always something worth
   showing. `pyodide` is the instance when the run happened locally, so a
   caller that wants to read variables back out of it (the visual renderer) can.
*/
async function runSnippet(code, { question = null, onStatus = null, source = null } = {}) {
  const say = (message) => {
    if (typeof onStatus === "function") onStatus(message);
  };
  /* `source` is the learner's code when `code` is a program that WRAPS it.
     The lesson notebook compiles its cells into one harness that passes each
     cell as a Python string literal, so the sniff below reads `_delta_cell("
     import torch as t\n...")` — the import is inside a quoted one-liner and
     the regex, which anchors to a real line start, cannot see it. Detecting
     torch on generated scaffolding rather than on what the learner wrote is
     how a torch cell talks itself onto Pyodide. */
  const isTorch = questionIsTorch(question) || TORCH_IMPORT.test(source || code || "");
  /* Torch only runs on the backend. A session that booted without a token
     (backend cold for one second at load time -> DDGuest.ensure() memoized a
     failure) used to be stranded: every torch Run hit the refusal below for
     the rest of the page load. This click is a fresh chance — provision now,
     upgrade the mode, and let the normal backend path handle the run. A real
     signed-in user demoted by a 401 is deliberately NOT re-provisioned; that
     would silently turn them into their old guest (practiceRealUserDemoted,
     practice/mode.js). */
  if (isTorch && practiceMode !== "backend" && !practiceRealUserDemoted) {
    say("Connecting to the practice backend...");
    const provisioned = await window.DDGuest?.retryProvision?.();
    if (provisioned) upgradePracticeModeToBackend();
    say("");
  }
  let useLocalPyodide =
    practiceMode !== "backend" ||
    ((!!window.LessonGate?.activeQuestion || questionNeedsEinops(question)) && !isTorch);

  if (practiceMode === "backend" && !useLocalPyodide) {
    const kernel = window.DeltaKernel;
    if (kernel?.available()) {
      say("Running in persistent runtime...");
      const cell = await kernel.runCell({
        code,
        filename: "<practice cell>",
        context: "practice-editor",
        timeout: 30,
      });
      if (cell.busy) {
        return { text: "Runtime busy — wait for current cell or restart it.", failed: true, blocked: false, pyodide: null };
      }
      if (!cell.unavailable) {
        if (runtimeStatus) runtimeStatus.textContent = cell.fresh
          ? "Persistent runtime started"
          : `Persistent runtime · ${cell.execCount} cells`;
        const stdout = normalizeOutput(cell.stdout);
        const stderr = normalizeOutput(cell.stderr);
        return {
          text: stdout || stderr || "✓ Ran successfully (state kept)",
          failed: !cell.ok || !!stderr,
          blocked: false,
          pyodide: null,
          execCount: cell.execCount,
          fresh: cell.fresh,
        };
      }
    }
    try {
      const res = await apiFetch("/api/practice/run-code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      if (res.status === 401) {
        handleExpiredToken();
        useLocalPyodide = true; // fall back to in-browser Pyodide
      } else if (!res.ok) {
        const detail = await res.text();
        return { text: detail || "Failed to run code.", failed: true, blocked: false, pyodide: null };
      } else {
        const data = await res.json();
        const stdout = normalizeOutput(data.stdout);
        const stderr = normalizeOutput(data.stderr);
        return {
          text: stdout || stderr || "✓ Ran successfully (no printed output)",
          failed: !!stderr,
          blocked: false,
          pyodide: null,
        };
      }
    } catch (_fetchErr) {
      useLocalPyodide = true; // backend unreachable
    }
  }

  if (useLocalPyodide && isTorch) {
    return { text: TORCH_UNAVAILABLE, failed: true, blocked: true, pyodide: null };
  }

  say("Loading Python...");
  const pyodide = await initPyodide();
  if (!pyodide) {
    return { text: "Failed to load Python.", failed: true, blocked: false, pyodide: null };
  }
  say("");

  const preamble = await buildPyodidePreamble(question);
  pyodide.runPython(preamble);
  try {
    pyodide.runPython(code);
    const stdout = normalizeOutput(pyodide.runPython("sys.stdout.getvalue()"));
    const stderr = normalizeOutput(pyodide.runPython("sys.stderr.getvalue()"));
    let text = stdout || "";
    if (stderr) text += (text ? "\n" : "") + stderr;
    return {
      text: text || "✓ Ran successfully (no printed output)",
      failed: !!stderr,
      blocked: false,
      pyodide,
    };
  } catch (pyErr) {
    const stderr = normalizeOutput(pyodide.runPython("sys.stderr.getvalue()"));
    return { text: stderr || pyErr.message, failed: true, blocked: false, pyodide };
  } finally {
    pyodide.runPython("sys.stdout = sys.__stdout__\nsys.stderr = sys.__stderr__\n");
  }
}

window.DeltaRunner = { runSnippet, installCodeEditorKeys, renderRunOutputVisual };

if (runtimeStatus) {
  runtimeStatus.textContent = window.DeltaKernel?.available()
    ? "Persistent runtime · state kept"
    : "Browser runtime";
}

if (runtimeResetBtn) {
  runtimeResetBtn.classList.toggle("hidden", !window.DeltaKernel?.available());
  runtimeResetBtn.addEventListener("click", async () => {
    runtimeResetBtn.disabled = true;
    if (runtimeStatus) runtimeStatus.textContent = "Restarting runtime…";
    const restarted = await window.DeltaKernel?.reset();
    if (runtimeStatus) runtimeStatus.textContent = restarted
      ? "Runtime restarted · state cleared"
      : "Could not restart runtime";
    runtimeResetBtn.disabled = false;
  });
}

runBtn.addEventListener("click", async () => {
  if (window.DeltaNotebook) {
    await window.DeltaNotebook.runCell(runBtn.closest(".notebook-cell"));
    return;
  }
  const idleLabel = runBtn.textContent;
  runBtn.disabled = true;
  runBtn.textContent = "…";
  outputArea.closest("[data-cell-output]")?.classList.remove("hidden");
  outputArea.textContent = "";
  hideOutputVisual();

  try {
    // During an inline lesson the editor holds optional runnable worked code.
    const runQuestion = window.LessonGate?.activeQuestion || PracticeAPI?.currentQuestion;
    const result = await runSnippet(codeEditor.value, {
      question: runQuestion,
      onStatus: (message) => {
        outputArea.textContent = message;
      },
    });
    outputArea.textContent = result.text;
    window.DeltaNotebook?.markRun(document.querySelector('.notebook-cell[data-cell-id="1"]'), result);
    if (!result.failed && result.pyodide) {
      await renderRunOutputVisual(result.pyodide, runQuestion);
    } else {
      hideOutputVisual();
    }
  } catch (e) {
    outputArea.textContent = "Error: " + e.message;
  }

  runBtn.disabled = false;
  runBtn.textContent = idleLabel;
});
