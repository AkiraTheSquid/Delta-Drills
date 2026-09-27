/* ================================================================
   LANG-JS.JS — answer a LeetCode drill in JavaScript instead of Python.

   Seth, 2026-09-26: "make it such that you can utilize javascript for the
   leetcode problems".

   The bank is Python. Nothing is re-authored: the backend derives a JS
   starter and translates the drill's own test calls
   (This-Directory-Only/backend/app/leetcode_js.py), then grades the JS in
   node. This file is the editor side:

     - a Python | JavaScript toggle in the notebook toolbar, shown only on a
       LeetCode drill in backend mode (JS runs on the server; a guest in
       local Pyodide mode has nowhere to run it);
     - the switch itself: keeps what you typed in the other language for this
       page load, swaps in the JS starter from GET /api/practice/js-starter;
     - the cell ▶ for JS (POST /api/practice/run-js, console output only);
     - `languageFor(q)`, which /check (test-check.js) and /submit (api.js)
       send so the server grades the code in the language it is written in.

   🔴 THE EDITOR'S LANGUAGE LIVES IN ONE PLACE: `#practice-notebook
   [data-code-lang]`. DeltaNotebook.reset() writes it (python unless told
   otherwise), serialize() carries it into the saved draft, restore() puts it
   back. So a question change, a draft restore and a toggle can never leave
   JS code in the editor labelled as Python, which would grade JS under the
   Python harness and record a wrong answer.

   The preference ("I answer LeetCode in JS") is remembered per browser; a
   drill the server cannot carry in JS (two ints past 2^53) greys the toggle.
   ================================================================ */
const DeltaLang = (() => {
  "use strict";

  const JS = "javascript";
  const PY = "python";
  const PREF_KEY = "dd-leetcode-lang";
  const host = document.getElementById("practice-notebook");
  const toolbar = host?.querySelector(".notebook-toolbar");
  const starters = new Map(); // question_id -> Promise<{supported, starter, reason}>
  const drafts = new Map(); // `${question_id}:${lang}` -> serialized notebook
  let current = null;

  const readPref = () => {
    try { return localStorage.getItem(PREF_KEY) === JS ? JS : PY; } catch (_) { return PY; }
  };
  const writePref = (lang) => {
    try { localStorage.setItem(PREF_KEY, lang); } catch (_) { /* private window: per-load only */ }
  };
  const isLeetcode = (q) => !!q && q.topic === "LeetCode" && q.submission_mode === "function";
  const backendMode = () => typeof practiceMode !== "undefined" && practiceMode === "backend";
  const editorLang = () => (host?.dataset.codeLang === JS ? JS : PY);
  const sameQuestion = (a, b) => !!a && !!b && String(a.question_id) === String(b.question_id);

  /* What /check and /submit are told. JS only when the editor holds JS AND
     the question being graded is the LeetCode drill on screen. */
  const languageFor = (q) => (editorLang() === JS && isLeetcode(q) && sameQuestion(q, current) ? JS : PY);

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
  const pyBtn = button(PY, "Python");
  const jsBtn = button(JS, "JavaScript");
  toolbar?.insertBefore(toggle, toolbar.firstChild);

  const paint = (info) => {
    const lang = editorLang();
    pyBtn.setAttribute("aria-pressed", String(lang === PY));
    jsBtn.setAttribute("aria-pressed", String(lang === JS));
    if (info) {
      jsBtn.disabled = !info.supported && lang !== JS;
      jsBtn.title = info.supported ? "" : `Not available in JavaScript: ${info.reason || "unsupported"}`;
    }
  };

  const fetchStarter = (q) => {
    const id = q.question_id;
    if (!starters.has(id)) {
      const p = apiFetch(`/api/practice/js-starter/${encodeURIComponent(id)}`)
        .then((res) => (res.ok ? res.json() : { supported: false, reason: `server said ${res.status}` }))
        .catch(() => {
          starters.delete(id); // a network blip must not grey the toggle for the whole load
          return { supported: false, reason: "could not reach the server" };
        });
      starters.set(id, p);
    }
    return starters.get(id);
  };

  /* ---------------------------------------------------------------- switch */
  async function switchTo(target) {
    const q = current;
    const nb = window.DeltaNotebook;
    if (!q || !nb || editorLang() === target) return;
    let next = drafts.get(`${q.question_id}:${target}`) || null;
    if (!next && target === JS) {
      const info = await fetchStarter(q);
      if (!sameQuestion(q, current) || !info.supported) { paint(info); return; }
      next = { version: 1, lang: JS, cells: [{ id: 1, code: info.starter }, { id: 2, code: "" }] };
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
    const pythonStarter = q.starter_code || "";
    fetchStarter(q).then((info) => {
      if (!sameQuestion(q, current)) return;
      paint(info);
      /* Honour the remembered preference only while the editor still holds
         the untouched Python starter: a draft restored in between (timer.js)
         is the learner's own work, in whatever language it carries. */
      const untouched = editorLang() === PY &&
        (window.DeltaNotebook?.serialize()?.cells?.[0]?.code || "") === pythonStarter;
      if (info.supported && readPref() === JS && untouched) switchTo(JS);
    });
  }

  /* ---------------------------------------------------------------- cell ▶ */
  async function runCell(code) {
    const res = await apiFetch("/api/practice/run-js", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    });
    if (res.status === 401) {
      handleExpiredToken();
      return { text: "Signed out — JavaScript runs on the server. Sign in again to run it.", failed: true };
    }
    if (!res.ok) return { text: (await res.text()) || `Run failed (${res.status}).`, failed: true };
    const data = await res.json();
    const text = [data.stdout, data.stderr].map((s) => String(s || "").trim()).filter(Boolean).join("\n");
    return { text: text || "(no console output — ▶ also checks the test cases below)", failed: !data.success };
  }

  return { JS, PY, editorLang, isJsEditor: () => editorLang() === JS, languageFor, onQuestion, runCell, switchTo };
})();

window.DeltaLang = DeltaLang;
