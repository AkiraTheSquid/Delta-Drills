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

/* Everything from here down is gone (2026-07-31, second pass).

   `ensurePyodidePracticePackages` (einops), `ensureArenaNumbersInPyodide`
   (the /delta_numbers.npy fixture), `buildPyodidePreamble`, `normalizeOutput`,
   the torch-detection constants and `runSnippet` / `window.DeltaRunner` all
   existed to execute Python that a learner would read or write in the page:
   lesson cells, and the canonical solution behind a question's target image.
   Both surfaces moved into the Colab notebook, so nothing calls them.

   What is left above is the interpreter itself, and it is here for exactly one
   reason: the guest/local adaptive engine is Python. `engine.js` loads
   `practice_engine.py` into this instance and `adaptive.js` / `api.js` drive it
   to update mastery when a session is not backed by the server. That path never
   ran learner code and is not affected by any of this. */
