/* ================================================================
   PYODIDE BOOT — the interpreter, and the one place code may run
   ================================================================

   Was `runner.js`. On 2026-07-31 practice stopped running the learner's code:
   there is no editor, no Run button and no in-app grading, because problems are
   worked in the Colab notebook they were compiled into and the result comes
   back through the two self-report buttons. What is left here is everything
   that still genuinely needs a Python interpreter in the page:

     * `initPyodide` — the adaptive engine for guest/local mode is Python.
       `engine.js` loads `practice_engine.py` into THIS instance, so removing
       the bootstrap along with the runner would have taken guest-mode mastery
       tracking with it. It is not part of the grading path and never was.
     * `buildPyodidePreamble` — `visuals.js` runs a question's canonical
       solution to draw the target image. That is problem context shown next to
       the prompt, not output from anything the learner typed.
     * `runSnippet` / `window.DeltaRunner` — the per-cell Run buttons inside
       LESSON prose (`notebook.js`). Teaching surface, not practice; a lesson
       that cannot run its own examples stops being a lesson.

   Nothing here grades. If you find yourself adding a comparison against an
   expected answer, it belongs in `api.js` (backend) — see practice/README.md.
*/

let pyodideInstance = null;
let pyodideLoading = false;
let pyodidePackagePromise = null;
let arenaNumbersPromise = null;

const normalizeOutput = (value) => (value || "").replace(/\r\n/g, "\n").trim();

async function initPyodide() {
  if (pyodideInstance) return pyodideInstance;
  if (pyodideLoading) {
    while (pyodideLoading) {
      await new Promise((r) => setTimeout(r, 100));
    }
    return pyodideInstance;
  }
  pyodideLoading = true;
  // Load status used to be written into the editor's output pane. That pane is
  // gone, and there is no longer any surface where "Loading Python..." makes
  // sense to the learner: the engine loads in the background before the first
  // question, and a lesson cell reports its own status through `onStatus`.
  // A failure must still be visible SOMEWHERE, so it goes to the console and
  // the null return propagates to callers, which all handle it.
  try {
    pyodideInstance = await loadPyodide();
    await pyodideInstance.loadPackage("numpy");
  } catch (e) {
    console.error("[pyodide] failed to load:", e);
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

const TORCH_IMPORT = /(^|\n)\s*(import\s+torch\b|from\s+torch[\s.])/;

const TORCH_UNAVAILABLE =
  "This code uses PyTorch, which can't run in the browser sandbox. " +
  "Open it in Colab (Show Answer / the solution notebook) to run it, " +
  "or sign in to use the full runner.";

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
async function runSnippet(code, { question = null, onStatus = null } = {}) {
  const say = (message) => {
    if (typeof onStatus === "function") onStatus(message);
  };
  const isTorch = questionIsTorch(question) || TORCH_IMPORT.test(code || "");
  let useLocalPyodide =
    practiceMode !== "backend" ||
    ((!!window.LessonGate?.activeQuestion || questionNeedsEinops(question)) && !isTorch);

  if (practiceMode === "backend" && !useLocalPyodide) {
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

window.DeltaRunner = { runSnippet };
