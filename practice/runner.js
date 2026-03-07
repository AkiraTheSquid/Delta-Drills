/* ================================================================
   PRACTICE RUNNER — Pyodide run button
   ================================================================ */

let pyodideInstance = null;
let pyodideLoading = false;

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

runBtn.addEventListener("click", async () => {
  runBtn.disabled = true;
  runBtn.textContent = "Running...";
  outputArea.textContent = "";

  try {
    let actualOutput = "";
    let runFailed = false;

    let useLocalPyodide = practiceMode !== "backend";

    if (practiceMode === "backend") {
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

      // Redirect stdout to capture print output
      pyodide.runPython(`
import sys
from io import StringIO
sys.stdout = StringIO()
sys.stderr = StringIO()
import numpy as np
np.random.seed(0)
`);

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
      } catch (pyErr) {
        const stderr = normalizeOutput(pyodide.runPython("sys.stderr.getvalue()"));
        outputArea.textContent = stderr || pyErr.message;
        runFailed = true;
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
