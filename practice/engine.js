/* ================================================================
   PRACTICE ENGINE — Pyodide init for adaptive engine
   ================================================================ */

let practiceEngineLoaded = false;

async function loadPracticeEngine(pyodide) {
  if (practiceEngineLoaded) return;
  try {
    const res = await fetch("practice_engine.py");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const source = await res.text();
    pyodide.runPython(source);
    practiceEngineLoaded = true;
    console.log("[practice] engine loaded in Pyodide");
  } catch (e) {
    console.error("[practice] failed to load practice_engine.py:", e);
  }
}
