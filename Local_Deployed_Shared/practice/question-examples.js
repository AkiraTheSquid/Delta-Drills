/* ================================================================
   QUESTION EXAMPLES — "for this input, this output" under the prompt
   ================================================================

   Seth, 2026-08-28, while testing the ndarray ladder: "on the left below the
   question, it still needs to have like multiple examples of expected, like
   for input, what the expected output would be for the problem for all of the
   problems … It should also have examples of incorrect output given the input."

   Two blocks, from two different sources, and the difference matters:

     * CORRECT rows are DERIVED, never authored. Every graded question already
       carries `test_cases[*].call` and `test_cases[*].expected_expr` — the
       exact pairs the grader will compare. Rendering those cannot drift from
       grading, because it IS grading's input, so this block lights up for the
       whole bank with no authoring at all.

     * WRONG rows are AUTHORED, per question, in `wrong_examples` (curated
       overrides → export → backend → payload). There is no honest way to
       derive them: a mutated correct answer is as likely to be a SECOND
       correct answer as a near miss, and the part that teaches is the `why`,
       which only a human knows. A question with none simply shows no second
       block — this is opt-in content, not a missing feature.

   Both are rendered read-only. Nothing here is executed, and nothing here is
   part of the submission — the learner's code still has to produce these.
*/

(function () {
  "use strict";

  // Enough to show the SHAPE of the mapping (one case is a coincidence, three
  // is a rule) without turning the prompt into a wall. Seth's standing note on
  // this content: "they can't be too long."
  const MAX_CORRECT_ROWS = 4;
  const MAX_WRONG_ROWS = 3;

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  /* What the learner should read as "the input".

     A grader case is a `setup_code` fixture plus a `call` expression, and the
     call very often names a variable the fixture bound (`solve(x)`), so the
     call ALONE says nothing about what went in. What is rendered is therefore
     the pair, REPL-style: the fixture lines the call actually references, then
     the call itself.

     The call is shown whole, wrapper and all. A case like
     `(lambda r: (r[0].tolist(), r[1]))(solve(x))` exists because the return
     value is a tensor and cannot be compared directly — and its
     `expected_expr` is the expected value of THAT WHOLE EXPRESSION, not of
     `solve(x)`. Digging the bare call out of the wrapper would pair an input
     with an output that does not belong to it, which is worse than a noisy
     line. */
  const ASSIGN_RE = /^([A-Za-z_]\w*)\s*=[^=]/;
  const DEF_RE = /^(?:def|class)\s+([A-Za-z_]\w*)/;

  /* Grader plumbing, not input. The LeetCode harness builds a case's
     arguments with `tree_node([2, 1, 3])` / `list_node([...])` and compares
     with `is_same_*`; the call already names the builder with its literal,
     which reads as LeetCode writes the case. Listing the builder's 20-line
     body under every case buried the input (Seth on q60151, 2026-09-29).
     `_dd_*` is the grader's own instrumentation (out= view capture). */
  const HARNESS_HELPERS = new Set([
    "tree_node", "list_node", "list_node_cycle", "is_same_tree", "is_same_list",
  ]);
  const isPlumbing = (name, text) =>
    HARNESS_HELPERS.has(name) || /^_(?:lc|from|dd)_/.test(name || "") || /\b_dd_/.test(text);

  /* Top-level statements, each with its indented body attached and its
     indentation KEPT — trimming line by line flattened every `def`/`class`
     fixture into unreadable, unrunnable text. */
  function statements(setup) {
    const out = [];
    let depth = 0; // open brackets in the current statement (strings ignored)
    for (const raw of String(setup || "").split("\n")) {
      if (!raw.trim()) continue;
      const prev = out.length ? out[out.length - 1] : null;
      // A body line, a line inside an open bracket (`x = [\n1,\n]`), or the
      // def/class a decorator sits on all belong to the statement above.
      if (prev !== null && (/^\s/.test(raw) || depth > 0 || /^@/.test(prev.split("\n").pop()))) {
        out[out.length - 1] += "\n" + raw.replace(/\s+$/, "");
      } else {
        out.push(raw.trim());
        depth = 0;
      }
      for (const ch of raw) depth += "([{".includes(ch) ? 1 : ")]}".includes(ch) ? -1 : 0;
      if (depth < 0) depth = 0;
    }
    return out;
  }

  function inputText(testCase) {
    const call = String(testCase?.call || "").trim();
    if (!call) return "";
    const lines = [];
    for (const stmt of statements(testCase?.setup_code)) {
      if (stmt.startsWith("#")) continue;
      if (/^import\b|^from\b/.test(stmt)) continue;
      const name = (stmt.match(DEF_RE) || stmt.match(ASSIGN_RE) || [])[1];
      if (isPlumbing(name, stmt)) continue;
      lines.push(stmt);
    }
    /* Keep the fixture lines this call actually reads — a shared setup block
       can bind several names, and listing the unused ones as "the input" is a
       lie by inclusion.

       🔴 Walked BACKWARDS, and each kept line widens what counts as read. A
       fixture can be built in steps (`x = […]` then `y = build(x)` with the
       call on `y`); a single forward pass keeping only names the CALL mentions
       drops `x` and renders a snippet that does not run. (codex, 2026-08-28.) */
    let wanted = call;
    const kept = [];
    for (let i = lines.length - 1; i >= 0; i -= 1) {
      const bound = lines[i].match(DEF_RE) || lines[i].match(ASSIGN_RE);
      if (bound && !new RegExp(`\\b${bound[1]}\\b`).test(wanted)) continue;
      kept.unshift(lines[i]);
      wanted += "\n" + lines[i];
    }
    return kept.concat(call).join("\n");
  }

  function correctRows(question) {
    const cases = Array.isArray(question?.test_cases) ? question.test_cases : [];
    const rows = [];
    for (const testCase of cases) {
      const expected = String(testCase?.expected_expr ?? "").trim();
      const input = inputText(testCase);
      // A case with no literal expectation grades by some other route (a
      // property assertion, an image compare). It has no pair to show, so it
      // is skipped rather than rendered as an empty arrow.
      if (!expected || !input) continue;
      rows.push({ input, output: expected });
    }
    return rows;
  }

  function wrongRows(question, fallbackInput) {
    const authored = Array.isArray(question?.wrong_examples) ? question.wrong_examples : [];
    const rows = [];
    for (const item of authored) {
      const output = String(item?.output ?? "").trim();
      if (!output) continue;
      rows.push({
        // An entry with no input of its own means "same input as the first
        // graded case" — which is how a near miss is normally authored, and it
        // keeps the two blocks describing the SAME call instead of two calls
        // the learner has to line up by eye.
        input: String(item?.call || item?.input || "").trim() || fallbackInput || "",
        output,
        why: String(item?.why || "").trim(),
      });
    }
    return rows;
  }

  /* THE `why` WAITS FOR THE GRADE. A near miss's explanation says what the
     wrong output got wrong — "without the added axis, (k,) * (3,) only
     broadcasts when k == 3" — which, read BEFORE answering, is the method.
     Seth on q821, 2026-09-21: "shouldn't be giving hints". The wrong OUTPUT
     stays on screen from the start (that is the "examples of incorrect
     output" he asked for on 2026-08-28); the sentence under it renders
     `hidden` and is shown when practice/concept-mask.js announces the grade
     landed (`dd-concept-mask`, masked → false) — the same moment the concept's
     name comes back. */
  function rowHtml(row, kind) {
    const mark = kind === "wrong" ? "✗" : "→";
    const why = row.why
      ? `<div class="qex-why" hidden>${esc(row.why)}</div>`
      : "";
    return (
      `<div class="qex-row qex-${kind}">` +
      `<code class="qex-in">${esc(row.input)}</code>` +
      `<span class="qex-arrow" aria-hidden="true">${mark}</span>` +
      `<code class="qex-out">${esc(row.output)}</code>` +
      why +
      `</div>`
    );
  }

  function blockHtml(title, note, rows, kind, hiddenCount) {
    const more = hiddenCount > 0
      ? `<div class="qex-more">+${hiddenCount} more case${hiddenCount === 1 ? "" : "s"} the grader runs.</div>`
      : "";
    return (
      `<div class="qex-block qex-block-${kind}">` +
      `<div class="qex-title">${esc(title)}</div>` +
      (note ? `<div class="qex-note">${esc(note)}</div>` : "") +
      rows.map((row) => rowHtml(row, kind)).join("") +
      more +
      `</div>`
    );
  }

  function render(question) {
    const host = document.getElementById("question-examples");
    if (!host) return;
    host.innerHTML = "";
    host.classList.add("hidden");
    if (!question) return;

    const correct = correctRows(question);
    const wrong = wrongRows(question, correct.length ? correct[0].input : "");
    if (!correct.length && !wrong.length) return;

    const shown = correct.slice(0, MAX_CORRECT_ROWS);
    const blocks = [];
    if (shown.length) {
      blocks.push(
        blockHtml(
          "For this input, return this",
          "These are the graded cases. Your function has to produce the right-hand side for every one of them.",
          shown,
          "correct",
          correct.length - shown.length,
        ),
      );
    }
    if (wrong.length) {
      blocks.push(
        blockHtml(
          "Not this",
          "Close-looking output that the grader rejects.",
          wrong.slice(0, MAX_WRONG_ROWS),
          "wrong",
          0,
        ),
      );
    }
    host.innerHTML = blocks.join("");
    host.classList.remove("hidden");
  }

  function revealWhy() {
    const host = document.getElementById("question-examples");
    if (!host) return;
    host.querySelectorAll(".qex-why[hidden]").forEach((el) => { el.hidden = false; });
  }

  window.addEventListener("dd-concept-mask", (ev) => {
    if (ev && ev.detail && ev.detail.masked === false) revealWhy();
  });

  window.QuestionExamples = { render, revealWhy };
})();
