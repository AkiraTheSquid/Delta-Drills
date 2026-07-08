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

_delta_output_value = solve()
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
  outputArea.textContent = "Loading Python...";
  try {
    pyodideInstance = await loadPyodide();
    await pyodideInstance.loadPackage("numpy");
    outputArea.textContent = "";
  } catch (e) {
    outputArea.textContent = "Failed to load Python: " + e.message;
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
async function buildPyodidePreamble(question = PracticeAPI?.currentQuestion) {
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
// Tab => insert a real tab character (or indent every selected line);
// Shift+Tab => dedent one tab or up to one Python indent of spaces.
// Accessibility escape hatch: press Escape first, then Tab moves focus out of
// the editor as usual (so keyboard-only users are never trapped).
let _editorTabEscapes = false;
const EDITOR_INDENT = "\t";
const EDITOR_SPACE_INDENT_WIDTH = 4;
if (codeEditor) {
  codeEditor.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      // Arm a one-shot "let Tab leave the field" so keyboard users can escape.
      _editorTabEscapes = true;
      return;
    }
    if (e.key !== "Tab") {
      _editorTabEscapes = false;
      return;
    }
    if (_editorTabEscapes) {
      // Let this Tab move focus normally, then re-arm capture.
      _editorTabEscapes = false;
      return;
    }
    e.preventDefault();
    const el = codeEditor;
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

runBtn.addEventListener("click", async () => {
  runBtn.disabled = true;
  runBtn.textContent = "Running...";
  outputArea.textContent = "";
  hideOutputVisual();

  try {
    let actualOutput = "";
    let runFailed = false;

    let useLocalPyodide =
      practiceMode !== "backend" || questionNeedsEinops(PracticeAPI?.currentQuestion);

    if (practiceMode === "backend" && !useLocalPyodide) {
      try {
        const res = await apiFetch("/api/practice/run-code", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ code: codeEditor.value }),
        });
        if (res.status === 401) {
          handleExpiredToken();
          useLocalPyodide = true; // fall back to in-browser Pyodide
        } else if (!res.ok) {
          const detail = await res.text();
          outputArea.textContent = detail || "Failed to run code.";
          runFailed = true;
        } else {
          const data = await res.json();
          const stdout = normalizeOutput(data.stdout);
          const stderr = normalizeOutput(data.stderr);
          actualOutput = stdout;
          outputArea.textContent = stdout || stderr || "(No output)";
          if (stderr) {
            runFailed = true;
          }
        }
      } catch (_fetchErr) {
        // Backend unreachable — fall back to in-browser Pyodide
        useLocalPyodide = true;
      }
    }

    if (useLocalPyodide) {
      const pyodide = await initPyodide();
      if (!pyodide) {
        runBtn.disabled = false;
        runBtn.textContent = "Run";
        return;
      }

      const preamble = await buildPyodidePreamble(PracticeAPI?.currentQuestion);
      pyodide.runPython(preamble);

      try {
        pyodide.runPython(codeEditor.value);
        const stdout = normalizeOutput(pyodide.runPython("sys.stdout.getvalue()"));
        const stderr = normalizeOutput(pyodide.runPython("sys.stderr.getvalue()"));
        actualOutput = stdout;
        let output = stdout || "";
        if (stderr) {
          output += (output ? "\n" : "") + stderr;
          runFailed = true;
        }
        outputArea.textContent = output || "(No output)";
        if (!runFailed) {
          await renderRunOutputVisual(pyodide, PracticeAPI?.currentQuestion);
        }
      } catch (pyErr) {
        const stderr = normalizeOutput(pyodide.runPython("sys.stderr.getvalue()"));
        outputArea.textContent = stderr || pyErr.message;
        runFailed = true;
        hideOutputVisual();
      } finally {
        // Reset stdout/stderr
        pyodide.runPython(`
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
`);
      }
    }

  } catch (e) {
    outputArea.textContent = "Error: " + e.message;
  }

  runBtn.disabled = false;
  runBtn.textContent = "Run";
});
