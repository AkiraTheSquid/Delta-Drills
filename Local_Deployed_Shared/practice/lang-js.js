/* ================================================================
   LANG-JS.JS — answer a LeetCode drill in JavaScript or Java instead of
   Python.

   Seth, 2026-09-26: "make it such that you can utilize javascript for the
   leetcode problems". 2026-09-28: "she meant java instead of javascript …
   maybe it can be a third option" — so Java joined as the third.

   The bank is Python. Nothing is re-authored: the backend derives a starter
   and translates the drill's own test calls
   (This-Directory-Only/backend/app/leetcode_js.py, leetcode_java.py), then
   grades in node or the JVM. This file is the editor side:

     - a Python | JavaScript | Java toggle in the notebook toolbar, shown only
       on a LeetCode drill in backend mode (both run on the server; a guest
       in local Pyodide mode has nowhere to run them);
     - the switch itself: keeps what you typed in each language for this page
       load, swaps in the starter from GET /api/practice/starter/{id}?language=;
     - the cell ▶ (POST /api/practice/run-js: console output; /run-java:
       compiles, runs a `main` if there is one);
     - `languageFor(q)`, which /check (test-check.js) and /submit (api.js)
       send so the server grades the code in the language it is written in.

   🔴 THE EDITOR'S LANGUAGE LIVES IN ONE PLACE: `#practice-notebook
   [data-code-lang]`. DeltaNotebook.reset() writes it (python unless told
   otherwise), serialize() carries it into the saved draft, restore() puts it
   back. So a question change, a draft restore and a toggle can never leave
   JS code in the editor labelled as Python, which would grade JS under the
   Python harness and record a wrong answer.

   The preference ("I answer LeetCode in Java") is remembered per browser; a
   drill the server cannot carry in a language greys that button (JS: two
   ints past 2^53; Java: one past 2^63).
   ================================================================ */
const DeltaLang = (() => {
  "use strict";

  const JS = "javascript";
  const JAVA = "java";
  const PY = "python";
  const OTHER = [JS, JAVA]; // graded on the server, never by Pyodide
  const RUN_PATH = { [JS]: "/api/practice/run-js", [JAVA]: "/api/practice/run-java" };
  const PREF_KEY = "dd-leetcode-lang";
  const host = document.getElementById("practice-notebook");
  const toolbar = host?.querySelector(".notebook-toolbar");
  const starters = new Map(); // `${question_id}:${lang}` -> Promise<{supported, starter, reason}>
  const drafts = new Map(); // `${question_id}:${lang}` -> serialized notebook
  /* Reference answers in Java, {question_id: code}, each one checked against
     the drill's cases by scripts/validate_java_solutions.py. Loaded once, on
     the first LeetCode drill. */
  let javaSolutions = null;
  const loadJavaSolutions = () => {
    if (!javaSolutions) {
      javaSolutions = fetch("lessons/leetcode/solutions_java.json?v=1")
        .then((res) => (res.ok ? res.json() : {}))
        .then((rows) => {
          javaSolutions.rows = rows;
          // An answer shown before the file landed went up in Python.
          if (editorLang() === JAVA) window.DeltaNotebook?.refreshSolution?.();
          return rows;
        })
        .catch(() => { javaSolutions = null; return {}; });
    }
    return javaSolutions;
  };
  let current = null;

  const readPref = () => {
    try {
      const v = localStorage.getItem(PREF_KEY);
      return OTHER.includes(v) ? v : PY;
    } catch (_) { return PY; }
  };
  const writePref = (lang) => {
    try { localStorage.setItem(PREF_KEY, lang); } catch (_) { /* private window: per-load only */ }
  };
  const isLeetcode = (q) => !!q && q.topic === "LeetCode" && q.submission_mode === "function";
  const backendMode = () => typeof practiceMode !== "undefined" && practiceMode === "backend";
  const editorLang = () => (OTHER.includes(host?.dataset.codeLang) ? host.dataset.codeLang : PY);
  const sameQuestion = (a, b) => !!a && !!b && String(a.question_id) === String(b.question_id);

  /* What /check and /submit are told. JS/Java only when the editor holds it
     AND the question being graded is the LeetCode drill on screen. */
  const languageFor = (q) => (editorLang() !== PY && isLeetcode(q) && sameQuestion(q, current) ? editorLang() : PY);

  /* ---------------------------------------------------------------- toggle */
  const style = document.createElement("style");
  style.textContent = `
    .lang-toggle { display: inline-flex; gap: 0; margin-right: 8px; border: 1px solid var(--border, #ccc); border-radius: 6px; overflow: hidden; }
    .lang-toggle[hidden] { display: none; }
    .lang-toggle button { border: 0; border-radius: 0; padding: 2px 10px; background: transparent; color: inherit; font: inherit; font-size: 12px; cursor: pointer; }
    .lang-toggle button[aria-pressed="true"] { background: var(--accent, #2563eb); color: #fff; }
    .lang-toggle button:disabled { opacity: 0.45; cursor: not-allowed; }`;
  document.head.appendChild(style);

  const toggle = document.createElement("div");
  toggle.className = "lang-toggle";
  toggle.setAttribute("role", "group");
  toggle.setAttribute("aria-label", "Answer language");
  toggle.hidden = true;
  const button = (lang, label) => {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.lang = lang;
    b.textContent = label;
    b.addEventListener("click", () => switchTo(lang));
    toggle.appendChild(b);
    return b;
  };
  const buttons = { [PY]: button(PY, "Python"), [JS]: button(JS, "JavaScript"), [JAVA]: button(JAVA, "Java") };
  const LABEL = { [JS]: "JavaScript", [JAVA]: "Java" };
  toolbar?.insertBefore(toggle, toolbar.firstChild);

  /* `lang`/`info`: what the server said about one language on this drill. */
  const paint = (lang, info) => {
    const now = editorLang();
    for (const [l, b] of Object.entries(buttons)) b.setAttribute("aria-pressed", String(l === now));
    if (lang && info) {
      buttons[lang].disabled = !info.supported && now !== lang;
      buttons[lang].title = info.supported ? "" : `Not available in ${LABEL[lang]}: ${info.reason || "unsupported"}`;
    }
  };

  const fetchStarter = (q, lang) => {
    const key = `${q.question_id}:${lang}`;
    if (!starters.has(key)) {
      const p = apiFetch(`/api/practice/starter/${encodeURIComponent(q.question_id)}?language=${lang}`)
        .then((res) => (res.ok ? res.json() : { supported: false, reason: `server said ${res.status}` }))
        .catch(() => {
          starters.delete(key); // a network blip must not grey the toggle for the whole load
          return { supported: false, reason: "could not reach the server" };
        });
      starters.set(key, p);
    }
    return starters.get(key);
  };

  /* ---------------------------------------------------------------- switch */
  async function switchTo(target) {
    const q = current;
    const nb = window.DeltaNotebook;
    if (!q || !nb || editorLang() === target) return;
    let next = drafts.get(`${q.question_id}:${target}`) || null;
    if (!next && target !== PY) {
      const info = await fetchStarter(q, target);
      if (!sameQuestion(q, current) || !info.supported) { paint(target, info); return; }
      if (editorLang() === target) return; // a second click landed while the first was fetching
      next = { version: 1, lang: target, cells: [{ id: 1, code: info.starter }, { id: 2, code: "" }] };
    }
    drafts.set(`${q.question_id}:${editorLang()}`, nb.serialize());
    if (next) nb.restore(next);
    else nb.reset(q.starter_code || "", { lang: PY });
    writePref(target);
    window.DeltaTestCheck?.hide?.();
    paint();
  }

  /* Called by ui.js right after DeltaNotebook.reset() puts a question's
     (Python) starter in the editor. */
  function onQuestion(q) {
    current = q;
    const offer = isLeetcode(q) && backendMode();
    toggle.hidden = !offer;
    paint();
    if (!offer) return;
    loadJavaSolutions();
    const pythonStarter = q.starter_code || "";
    const pref = readPref();
    for (const lang of OTHER) {
      fetchStarter(q, lang).then((info) => {
        if (!sameQuestion(q, current)) return;
        paint(lang, info);
        /* Honour the remembered preference only while the editor still holds
           the untouched Python starter: a draft restored in between (timer.js)
           is the learner's own work, in whatever language it carries. */
        const untouched = editorLang() === PY &&
          (window.DeltaNotebook?.serialize()?.cells?.[0]?.code || "") === pythonStarter;
        if (info.supported && pref === lang && untouched) switchTo(lang);
      });
    }
  }

  /* ---------------------------------------------------------------- cell ▶ */
  async function runCell(code) {
    const lang = editorLang();
    const res = await apiFetch(RUN_PATH[lang] || RUN_PATH[JS], {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // Java: the drill's cases run for what they print when there is no `main`.
      body: JSON.stringify({ code, question_id: current?.question_id ?? null }),
    });
    if (res.status === 401) {
      handleExpiredToken();
      return { text: `Signed out — ${LABEL[lang] || "this language"} runs on the server. Sign in again to run it.`, failed: true };
    }
    if (!res.ok) return { text: (await res.text()) || `Run failed (${res.status}).`, failed: true };
    const data = await res.json();
    const text = [data.stdout, data.stderr].map((s) => String(s || "").trim()).filter(Boolean).join("\n");
    const quiet = lang === JAVA
      ? "(compiled — nothing printed; System.out.println in your method shows here. ▶ also checks the test cases below)"
      : "(no console output — ▶ also checks the test cases below)";
    return { text: text || quiet, failed: !data.success };
  }

  /* The answer to show for the drill on screen, given the Python one: the
     Java answer when the editor holds Java and one exists, else `python`
     unchanged. Synchronous — the file was fetched on the question's arrival. */
  const solutionFor = (python) => {
    if (editorLang() !== JAVA || !current) return python;
    return javaSolutions?.rows?.[String(current.question_id)] || python;
  };

  /* isServerEditor: the editor holds JS or Java, which run on the server,
     never in the Python kernel. */
  const isServerEditor = () => editorLang() !== PY;
  return { JS, JAVA, PY, OTHER, editorLang, isServerEditor, isJsEditor: isServerEditor, languageFor, onQuestion, runCell, solutionFor, switchTo };
})();

window.DeltaLang = DeltaLang;
