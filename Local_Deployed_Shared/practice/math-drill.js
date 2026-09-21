/* ================================================================
   MATH DRILL — multiple-choice (non-code) questions on the practice page
   ================================================================

   A math question is a bank row with `submission_mode: "mc"`: the learner
   works it on paper and picks the choice that matches. Spec:
   docs/spec-math-mc-backbone.md. This module owns everything that differs
   from a coding drill, so the four files that render and submit questions
   each carry a one-line hook and nothing else:

     ui.js       renderQuestion  → DeltaMath.mount(q)      choices in, editor out
     events.js   submit          → DeltaMath.selectedKey() the answer IS the key
                                   DeltaMath.hasPick()/nudge() no pick, no submit
                 verdict         → DeltaMath.showSolution(q, result)
     api.js      local grading   → DeltaMath.gradeLocal(q, key)
     lessons.js  lesson page     → DeltaMath.render(root)  the KaTeX pass

   The KaTeX pass is the same call ui.js and arena-notebook.js make; it lives
   here as well so a lesson page and a solution block render math the same way
   the question prompt does. `renderMarkdown` is the app's one markdown
   renderer (practice/lessons.js) — a solution is markdown with math in it,
   not code, so it never goes near a code cell.

   🔴 `html.dd-math-mc` is the switch. styles/practice/math-drill.css hides the
   editor column, the Run check and the code solution under it, and mount()
   toggles it per QUESTION — a coding drill after a math one gets its editor
   back because mount() runs for every rendered question, not only math ones. */

(function () {
  "use strict";

  const CLASS = "dd-math-mc";
  const CHOICES_ID = "math-choices";
  const SOLUTION_ID = "math-solution";
  let current = null;

  const isMC = (q) => !!(q && q.submission_mode === "mc" && Array.isArray(q.choices) && q.choices.length);

  const esc = (value) =>
    String(value == null ? "" : value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  /* The KaTeX pass. auto-render.min.js loads `defer`, so it is usually ready
     by the time anything renders; without it the source stays readable. */
  const render = (root) => {
    if (!root || typeof window.renderMathInElement !== "function") return;
    try {
      window.renderMathInElement(root, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true },
        ],
        throwOnError: false,
      });
    } catch (_) {
      /* malformed LaTeX — leave the raw text */
    }
  };

  const markdown = (text) => {
    const md = window.LessonGate && window.LessonGate.renderMarkdown;
    return md ? md(String(text || "")) : `<p>${esc(text)}</p>`;
  };

  /* The choice list, as a sibling of #question-text so LadderUI.decorate
     (which rewrites #question-text) never wipes it. Created once, reused. */
  const _choicesEl = () => {
    let el = document.getElementById(CHOICES_ID);
    if (el) return el;
    const anchor = document.getElementById("question-text");
    if (!anchor || !anchor.parentNode) return null;
    el = document.createElement("div");
    el.id = CHOICES_ID;
    el.className = "math-choices hidden";
    el.setAttribute("role", "radiogroup");
    el.setAttribute("aria-label", "Answer choices");
    anchor.parentNode.insertBefore(el, anchor.nextSibling);
    return el;
  };

  const _solutionEl = () => {
    let el = document.getElementById(SOLUTION_ID);
    if (el) return el;
    const pre = document.getElementById("solution-code");
    if (!pre || !pre.parentNode) return null;
    el = document.createElement("div");
    el.id = SOLUTION_ID;
    el.className = "math-solution hidden";
    pre.parentNode.insertBefore(el, pre.nextSibling);
    return el;
  };

  const _submitBtn = () => document.getElementById("practice-submit-btn");

  const selectedKey = () => {
    const picked = document.querySelector(`#${CHOICES_ID} input[type=radio]:checked`);
    return picked ? String(picked.value) : "";
  };

  /* Submit before a pick. 🔴 NOT the `disabled` attribute: the answer clock
     force-submits by `practiceSubmitBtn.click()` and skips a disabled button
     (timer.js `_forceSubmitOrAdvance` → next question, nothing recorded), and
     an expiry is a MISS, not a skip (2026-09-20). So the button stays live,
     looks inert (`needs-pick`, math-drill.css), and the submit handler
     refuses a learner's click with no pick (`hasPick`) while the clock's
     click goes through with an empty key — graded as a miss by
     grading.grade_choice, logged as a timeout. */
  const NEEDS_PICK = "needs-pick";
  const _setNeedsPick = (on) => {
    const btn = _submitBtn();
    if (!btn) return;
    btn.classList.toggle(NEEDS_PICK, on);
    if (on) btn.setAttribute("aria-disabled", "true");
    else btn.removeAttribute("aria-disabled");
  };
  const hasPick = () => selectedKey() !== "";

  /* A learner's click with nothing picked: say so, on the list. */
  const nudge = () => {
    const list = document.getElementById(CHOICES_ID);
    if (!list) return;
    list.classList.remove("is-nudged");
    void list.offsetWidth; /* restart the animation */
    list.classList.add("is-nudged");
  };

  /* Called for EVERY rendered question (ui.js). Non-math: take the switch off
     and hide the list. Math: draw the choices and hold Submit until one is
     picked. */
  const mount = (q) => {
    const on = isMC(q);
    current = on ? q : null;
    document.documentElement.classList.toggle(CLASS, on);
    const list = _choicesEl();
    const sol = _solutionEl();
    if (sol) {
      sol.classList.add("hidden");
      sol.innerHTML = "";
    }
    _setNeedsPick(false);
    if (!list) return;
    list.classList.remove("is-nudged");
    if (!on) {
      list.classList.add("hidden");
      list.innerHTML = "";
      return;
    }
    const name = `math-choice-${q.question_id}`;
    list.innerHTML = q.choices
      .map((c) => {
        const key = esc(c.key);
        return (
          `<label class="math-choice">` +
          `<input type="radio" name="${name}" value="${key}">` +
          `<span class="math-choice-key">${key}</span>` +
          `<span class="math-choice-text">${esc(c.text)}</span>` +
          `</label>`
        );
      })
      .join("");
    list.classList.remove("hidden");
    render(list);
    _setNeedsPick(true);
    list.querySelectorAll("input[type=radio]").forEach((input) => {
      input.addEventListener("change", () => {
        if (current !== q) return;
        list.classList.remove("is-nudged");
        _setNeedsPick(false);
      });
    });
  };

  /* After the verdict: mark the picked and keyed choices, freeze the list,
     and put the authored solution (markdown + LaTeX) where the code answer
     would go. `result.expected_output` is the keyed letter — the server
     returns it only now (practice_schemas: not on the served question). */
  const showSolution = (q, result) => {
    if (!isMC(q)) return false;
    const key = String((result && result.expected_output) || q.correct_choice || "").trim().toUpperCase();
    const picked = selectedKey().toUpperCase();
    const list = document.getElementById(CHOICES_ID);
    if (list) {
      list.querySelectorAll(".math-choice").forEach((label) => {
        const input = label.querySelector("input");
        const value = input ? String(input.value).toUpperCase() : "";
        if (input) input.disabled = true;
        label.classList.toggle("is-correct", value === key);
        label.classList.toggle("is-picked", value === picked);
        label.classList.toggle("is-wrong", value === picked && picked !== key);
      });
    }
    const sol = _solutionEl();
    if (!sol) return true;
    const body = q.solution_md || (result && result.solution_md) || "";
    sol.innerHTML = body ? markdown(body) : `<p class="math-solution-empty">Answer: ${esc(key)}</p>`;
    sol.classList.remove("hidden");
    render(sol);
    return true;
  };

  /* Local (no-backend) grading — the key compare, same as grading.grade_choice. */
  const gradeLocal = (q, key) => {
    const picked = String(key || "").trim().toUpperCase();
    const expected = String((q && q.correct_choice) || "").trim().toUpperCase();
    const correct = !!picked && picked === expected;
    return { correct, actual_output: picked, expected_output: expected, failed_tests: [] };
  };

  window.DeltaMath = {
    isMC,
    active: () => current !== null,
    render,
    mount,
    selectedKey,
    hasPick,
    nudge,
    showSolution,
    gradeLocal,
  };
})();
