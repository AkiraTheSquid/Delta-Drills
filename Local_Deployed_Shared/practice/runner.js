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

async function buildPyodidePreamble(question = PracticeAPI?.currentQuestion) {
  const needsEinops = questionNeedsEinops(question);
  const needsArenaArray = questionNeedsArenaArray(question);
  await ensurePyodidePracticePackages({ needsEinops });
  if (needsArenaArray) {
    await ensureArenaNumbersInPyodide();
  }

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
`;
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
