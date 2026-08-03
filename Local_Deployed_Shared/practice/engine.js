/* ================================================================
   PRACTICE ENGINE — Pyodide init for adaptive engine
   ================================================================ */

let practiceEngineLoaded = false;

async function loadPracticeEngine(pyodide) {
  if (practiceEngineLoaded) return;
  try {
    // Cache-busted: bump when practice_engine.py changes, or the browser
    // keeps running a stale engine (set_self_reported_level was missing).
    const res = await fetch("practice_engine.py?v=4");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const source = await res.text();
    pyodide.runPython(source);
    practiceEngineLoaded = true;
    console.log("[practice] engine loaded in Pyodide");
  } catch (e) {
    console.error("[practice] failed to load practice_engine.py:", e);
  }
}
