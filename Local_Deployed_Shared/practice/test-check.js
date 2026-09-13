/* ================================================================
   TEST CHECK — every ▶ tells you which test cases pass, before Submit.

   Seth, 2026-09-13: "when you run the code, it tells you which test cases
   passed and which test cases failed, even if you didn't submit it, so that
   you get immediate feedback on what you did right versus what you did
   wrong ... it should of course still show you the output for the different
   test cases for the inputs as well as the output of the original code cell."

   So a Run now does two things. The cell still runs where it always ran
   (the persistent kernel, the backend fork, or Pyodide for a guest) and its
   printed output lands under the cell exactly as before. THEN the notebook's
   joined cells — `DeltaNotebook.submissionCode()`, the same text Submit
   posts — go through the same test harness Submit grades with, and every
   case comes back ✓/✗ with the call, what it returned and what was expected.

   🔴 THE VERDICT IS THE GRADER'S, NOT A COPY OF IT. Backend mode posts to
   `/api/practice/check`, which is `grade_submission` with the verdict
   thrown away (same `run_function_tests`, same tolerance rules, records
   nothing). A guest runs `runPyodideTests` below, which is the ONE Pyodide
   harness — `api.js::submitAnswer` calls this same function to grade, so
   what ▶ says and what Submit says cannot drift apart. A second copy of the
   harness is how Run and Submit disagreed about torch in the first place.

   🔴 NOTHING HERE IS EVIDENCE. No attempt, no ladder outcome, no pending
   slot. Pressing ▶ fifty times costs the learner nothing but the wait.

   Owned here, not in notebook-editor.js: that file is the cells, this is
   the grade-shaped feedback under them, and it parents itself the way
   ui.js::renderFailedTests does — under the learner's cells, above the
   Submit verdict and the answer.
   ================================================================ */
const DeltaTestCheck = (() => {
  "use strict";

  const BLOCK_ID = "run-tests-block";
  const CLIP = 220;
  let seq = 0;

  const clip = (s) => {
    const t = String(s == null ? "" : s);
    return t.length > CLIP ? t.slice(0, CLIP) + "…" : t;
  };
  /* A harness that never reached the cases answers with the whole traceback
     — a temp-file path and a line number that mean nothing to the learner.
     The last line is the error, and it is the only one worth the space. */
  const lastLine = (s) => {
    const lines = String(s == null ? "" : s).trim().split("\n").filter((l) => l.trim());
    return /Traceback/.test(lines[0] || "") ? lines[lines.length - 1] : String(s);
  };
  const esc = (s) => String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const questionKey = (q) =>
    (typeof stableQuestionId === "function" ? stableQuestionId(q) : "") || String(q?.question_id ?? "");

  /* ---------------------------------------------------------------
     The Pyodide harness. Moved here from api.js verbatim on 2026-09-13 so
     that Submit (api.js) and Run (this file) execute ONE harness. Mirrors
     the shape of code_runner.run_function_tests: seed, setup, call,
     assert, expected-setup, expected, compare. It does NOT carry the
     backend's float tolerance or torch branches — Pyodide never sees
     torch (needsTorchRuntime routes it away), and this equality is what
     a guest's Submit has always been graded by.
     --------------------------------------------------------------- */
  const harness = (testCases) => {
    const testsJsonLiteral = JSON.stringify(JSON.stringify(testCases));
    return `
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
    if isinstance(value, dict):
        return {k: _delta_to_jsonable(v) for k, v in value.items()}
    return value

def _delta_equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_delta_equal(x, y) for x, y in zip(a, b))
    return bool(a == b)

_delta_results = []
for _delta_case in json.loads(${testsJsonLiteral}):
    try:
        if _delta_case.get("setup_code"):
            np.random.seed(0)
            exec(_delta_case["setup_code"], globals())
        _delta_actual = eval(_delta_case["call"], globals())
        if _delta_case.get("assert_code"):
            exec(_delta_case["assert_code"], dict(globals(), result=_delta_actual))
        _delta_expected_setup = _delta_case.get("expected_setup_code") or _delta_case.get("setup_code")
        if _delta_expected_setup:
            np.random.seed(0)
            exec(_delta_expected_setup, globals())
        _delta_expected = eval(_delta_case["expected_expr"], globals())
        _delta_results.append({
            "passed": bool(_delta_equal(_delta_actual, _delta_expected)),
            "actual": repr(_delta_to_jsonable(_delta_actual)),
            "expected": repr(_delta_to_jsonable(_delta_expected)),
            "error": "",
        })
    except Exception as _delta_exc:
        _delta_results.append({
            "passed": False,
            "actual": "",
            "expected": "",
            "error": f"{type(_delta_exc).__name__}: {_delta_exc}",
        })
json.dumps(_delta_results)
`;
  };

  /* Run the question's function tests on the local Pyodide instance.

     Returns { results, output } on a harness that completed (`results` is
     one row per case, in bank order) or { results: null, output } when the
     learner's code itself failed to run — `output` is then the traceback,
     which is the message worth showing. Always restores sys.stdout/stderr:
     the preamble redirects both. */
  async function runPyodideTests(question, userCode) {
    const pyodide = await initPyodide();
    if (!pyodide) return { results: null, output: "Failed to load Python." };
    const preamble = await buildPyodidePreamble(question);
    pyodide.runPython(preamble);
    try {
      pyodide.runPython(userCode);
      const resultJson = pyodide.runPython(harness(question.test_cases));
      const results = JSON.parse(resultJson);
      return { results, output: pyodide.runPython("sys.stdout.getvalue()").trim() };
    } catch (e) {
      return { results: null, output: pyodide.runPython("sys.stderr.getvalue()").trim() || e.message };
    } finally {
      pyodide.runPython("sys.stdout = sys.__stdout__\nsys.stderr = sys.__stderr__");
    }
  }

  /* Where would this code grade, and how? The routing is api.js's own
     (`requiresLocalPyodide`, guarded by watch_invariants.py) restated: the
     backend grades everything except the numpy/visual einops drills, whose
     helpers only the local preamble defines. A 401 demotes to local mode
     and falls through; a backend that cannot be reached is reported as
     such rather than silently re-graded on a weaker Python. */
  async function check(question, code) {
    const cases = Array.isArray(question?.test_cases) ? question.test_cases : [];
    const withCalls = (tests) => tests.map((t, i) => ({ ...t, call: t.call || cases[i]?.call || "" }));
    const requiresLocalPyodide =
      questionNeedsEinops(question) && !needsTorchRuntime(question, code);

    if (practiceMode === "backend" && !requiresLocalPyodide) {
      let res;
      try {
        res = await apiFetch("/api/practice/check", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question_id: question.question_id, user_code: code }),
        });
      } catch (_err) {
        return { error: "Could not reach the practice backend to check the test cases." };
      }
      if (res.status === 401) {
        handleExpiredToken();
      } else if (!res.ok) {
        return { error: (await res.text()) || `Test check failed (${res.status}).` };
      } else {
        const data = await res.json();
        if (!data.supported) return { unsupported: true };
        return { tests: withCalls(data.tests || []), correct: !!data.correct };
      }
    }

    if (!(question?.submission_mode === "function" && cases.length)) return { unsupported: true };
    if (needsTorchRuntime(question, code)) return { error: TORCH_UNAVAILABLE };
    const run = await runPyodideTests(question, code);
    if (!run.results) {
      return {
        correct: false,
        tests: withCalls([{ passed: false, actual: "", expected: "", error: run.output }]),
      };
    }
    return { tests: withCalls(run.results), correct: run.results.every((t) => t.passed) };
  }

  /* ---------------------------------------------------------------
     Rendering. One block, looked up by id, hidden between questions and
     re-parented on every render (the notebook rebuilds the cells around
     it — see ui.js::renderFailedTests for the same lesson). It sits
     UNDER the learner's cells and ABOVE the Submit verdict and the
     answer; notebook-editor.js::feedbackBoundary keeps new cells above it.
     --------------------------------------------------------------- */
  function block() {
    const host = document.getElementById("notebook-cells");
    if (!host || !host.getClientRects().length) return null;
    let el = document.getElementById(BLOCK_ID);
    if (!el) {
      el = document.createElement("div");
      el.id = BLOCK_ID;
      el.className = "run-tests-block";
    }
    host.insertBefore(el, host.querySelector("#failed-tests-block, [data-solution-cell]"));
    return el;
  }

  function hide() {
    document.getElementById(BLOCK_ID)?.classList.add("hidden");
  }

  /* Hide the block unless it belongs to this question — a re-render of the
     same question (a tab switch) keeps it. */
  function hideUnless(question) {
    const el = document.getElementById(BLOCK_ID);
    if (el && el.dataset.questionId !== questionKey(question)) el.classList.add("hidden");
  }

  function paint(el, question, { title, tone, body, note }) {
    el.classList.remove("run-tests-block--pass", "run-tests-block--fail", "run-tests-block--pending", "hidden");
    if (tone) el.classList.add(`run-tests-block--${tone}`);
    el.dataset.questionId = questionKey(question);
    el.innerHTML =
      `<div class="run-tests-title">${esc(title)}</div>` +
      (body ? `<pre class="run-tests-body">${body}</pre>` : "") +
      (note ? `<div class="run-tests-note">${esc(note)}</div>` : "");
  }

  function render(question, outcome) {
    const el = block();
    if (!el) return;
    if (outcome.unsupported) { hide(); return; }
    if (outcome.error) {
      paint(el, question, {
        title: "Test cases could not be checked",
        tone: "fail",
        body: esc(clip(outcome.error)),
      });
      return;
    }
    const cases = Array.isArray(question?.test_cases) ? question.test_cases : [];
    const total = cases.length || outcome.tests.length;
    const passed = outcome.tests.filter((t) => t.passed).length;
    const rows = [];
    for (let i = 0; i < total; i++) {
      const t = outcome.tests[i];
      const call = esc(clip((t && t.call) || cases[i]?.call || `case ${i + 1}`));
      if (!t) {
        // The harness stopped before this case (the code itself failed to
        // run): the first row carries the traceback, the rest never ran.
        rows.push(`<span class="run-test--fail">✗</span> ${call}\n    not reached`);
      } else if (t.error) {
        rows.push(`<span class="run-test--fail">✗</span> ${call}\n    error: ${esc(clip(lastLine(t.error)))}`);
      } else if (t.passed) {
        rows.push(`<span class="run-test--pass">✓</span> ${call}\n    got:      ${esc(clip(t.actual))}`);
      } else {
        rows.push(`<span class="run-test--fail">✗</span> ${call}\n    expected: ${esc(clip(t.expected))}\n    got:      ${esc(clip(t.actual))}`);
      }
    }
    const all = outcome.correct && passed === total;
    paint(el, question, {
      title: `${passed} of ${total} test case${total === 1 ? "" : "s"} pass${all ? " — Submit when ready" : ""} · not submitted`,
      tone: all ? "pass" : "fail",
      body: rows.join("\n"),
      note: "Checked as a fresh run of all your cells, the same way Submit grades. Nothing is recorded until you Submit.",
    });
  }

  /* Called by notebook-editor.js after every learner-cell run. Not awaited
     there — the cell's own output and the ▶ button do not wait on this.
     A newer run supersedes an older check (`seq`), and a check that lands
     after the question changed paints nothing. */
  async function afterRun(question, code) {
    const my = ++seq;
    if (!question || !String(code || "").trim()) { hide(); return; }
    const el = block();
    if (el) paint(el, question, { title: "Checking test cases…", tone: "pending" });
    let outcome;
    try {
      outcome = await check(question, code);
    } catch (err) {
      outcome = { error: err?.message || String(err) };
    }
    if (my !== seq) return;
    const _papi = typeof PracticeAPI !== "undefined" ? PracticeAPI : window.PracticeAPI;
    const current = window.LessonGate?.activeQuestion || _papi?.currentQuestion;
    if (current && current !== question) return;
    /* A Submit (the learner's, or the clock's) that landed while this check
       was in flight already painted the real verdict; a dry check arriving
       after it must not paint over the grade. */
    if (typeof practiceFeedbackArea !== "undefined" && practiceFeedbackArea &&
        !practiceFeedbackArea.classList.contains("hidden")) return;
    render(question, outcome);
  }

  return { afterRun, check, render, hide, hideUnless, runPyodideTests };
})();

window.DeltaTestCheck = DeltaTestCheck;
