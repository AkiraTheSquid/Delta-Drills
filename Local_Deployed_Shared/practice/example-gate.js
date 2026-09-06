/* ================================================================
   EXAMPLE GATE — a worked example that pops up in front of a drill

   The backend decides whether THIS drill opens behind an example
   (app/example_schedule.py → `question.ladder_example.show`): on the Faded
   rung after a miss, on the Solo rung at widening intervals, on the
   Integrated rung once on entry and then never. This file is the screen:
   the example's code, runnable and editable in the same panel the lesson
   uses, and a button that hands over to the drill. The learner reads and
   runs; they do not solve — "you don't do the coding, you just see the code"
   (Seth, 2026-08-30).

   Which example: the drill's own ```python worked``` fence when its KP
   authored one (solo items carry `worked_example_code`); otherwise the
   segment whose `faded_items` own the drill; otherwise the KP's last
   segment, its most complete example. Same choice `ladder.js` made when the
   example sat beside the drill — it now sits in front of it instead, and
   only when the schedule says so.

   Runs AFTER LessonGate.maybeShow and LadderUI.maybeShowWorked in
   events.js, so a first-contact lesson is never followed by its own example
   twice. Skipped on the Colab edition, where the example lives in the
   notebook beside the problem.
   ================================================================ */
const ExampleGate = (() => {
  "use strict";

  const esc = (value) =>
    String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  const _el = (id) => document.getElementById(id);

  const WHY_LINE = {
    after_miss: "That one did not land — here is the move again. Read it, run it, change it, then try the next one from memory.",
    scheduled: "An example first. Read it, run it, change it — then write the next one yourself.",
  };

  /* The example for this drill: its own fence, else its segment's. */
  const _pick = (kp, questionId) => {
    const qid = Number(questionId);
    const own = (kp.solo_items || kp.applied_items || []).find(
      (item) => item && Number(item.question_id) === qid && String(item.worked_example_code || "").trim(),
    );
    if (own) {
      return {
        title: kp.title,
        markdown: "```python\n" + String(own.worked_example_code).trim() + "\n```",
        code: String(own.worked_example_code).trim(),
      };
    }
    const segments = kp.segments && kp.segments.length ? kp.segments : null;
    if (!segments) {
      return {
        title: kp.title,
        markdown: kp.worked_example_markdown || "",
        code: null,
      };
    }
    const owns = (seg) =>
      (seg.faded_items || []).some((item) => item && Number(item.question_id) === qid);
    const seg = segments.find(owns) || segments[segments.length - 1];
    return {
      title: seg.title || kp.title,
      markdown: seg.worked_example_markdown || "",
      code: seg.worked_example_code || null,
    };
  };

  const _firstFence = (markdown) => {
    const match = /```(?:python|py)?[^\n]*\n([\s\S]*?)```/.exec(String(markdown || ""));
    return match ? match[1].trim() : "";
  };

  const _cleanup = () => {
    document.body.classList.remove("lesson-mode");
    if (window.DDFeedbackPanel) {
      window.DDFeedbackPanel.setLessonContext(null);
      window.DDFeedbackPanel.closeLesson();
    }
    const fallback = typeof DEFAULT_EDITOR_CODE === "string" ? DEFAULT_EDITOR_CODE : "";
    if (window.DeltaNotebook) window.DeltaNotebook.reset(fallback, { addScratch: false });
    else if (_el("code-editor")) _el("code-editor").value = fallback;
    const out = _el("output-area");
    if (out) out.textContent = "";
  };

  /* Show the example for `question` if the server scheduled one. Resolves
     true when the screen was taken over (and `onDone` will be called from the
     button), false when the drill should render straight away. */
  const maybeShow = async (question, onDone) => {
    if (question?.attempt_first) return false;
    try {
      const plan = question && question.ladder_example;
      const kc = question && question.ladder_kc;
      if (!plan || !plan.show || !kc) return false;
      if (question.diagnostic_active) return false;
      if (window.DDColab && typeof window.DDColab.active === "function" && window.DDColab.active()) {
        return false;
      }
      if (!window.LessonGate || typeof window.LessonGate.getKpEntry !== "function") return false;
      const found = await window.LessonGate.getKpEntry(kc);
      if (!found || !found.kp) return false;
      const picked = _pick(found.kp, question.question_id);
      if (!String(picked.markdown || "").trim()) return false;

      const host = _el("question-text");
      const editor = _el("code-editor");
      if (!host || !editor) return false;
      const md = window.LessonGate.renderMarkdown || ((text) => "<pre>" + esc(text) + "</pre>");

      document.body.classList.add("lesson-mode");
      _el("page-practice")?.classList.remove("session-idle");

      const stageLabel = question.ladder_stage || "";
      host.innerHTML =
        `<h2 class="lesson-kp-title" id="lesson-title" tabindex="-1">Example — ${esc(picked.title)}</h2>` +
        `<p class="ladder-stage-callout">${esc(WHY_LINE[plan.why] || WHY_LINE.scheduled)}</p>` +
        '<div class="lesson-worked nb-scope"><h3>Worked example</h3>' +
        md(picked.markdown) +
        '<p class="lesson-example-note">Run any block to see it execute. Edit it if you want to ' +
        "check something. The problem comes next.</p></div>" +
        '<div class="lesson-actions"><button type="button" class="primary" id="example-continue-btn">' +
        "Now you try →</button></div>";

      if (window.LessonNotebook && typeof window.LessonNotebook.mount === "function") {
        window.LessonNotebook.mount(host, `${kc}#example#${question.question_id}`);
      }
      const code = picked.code || _firstFence(picked.markdown);
      if (window.DeltaNotebook) window.DeltaNotebook.reset(code, { addScratch: false });
      else editor.value = code;
      const out = _el("output-area");
      if (out) out.textContent = "";
      if (window.DDFeedbackPanel) {
        window.DDFeedbackPanel.setLessonContext({
          kc,
          title: picked.title,
          questionId: question.question_id,
        });
      }
      if (window.StageLadder && typeof window.StageLadder.show === "function") {
        window.StageLadder.show({
          kc,
          title: question.ladder_kc_title || kc,
          eyebrow: "Example",
          stage: stageLabel,
          estimate: question.ladder_estimate || null,
        });
      }
      host.scrollTop = 0;
      window.scrollTo({ top: 0 });
      // Reported back on submit (api.js `example_shown`): the server scheduled
      // this example, but only the client knows it was actually drawn.
      question.ladder_example_shown = true;

      const button = _el("example-continue-btn");
      let advancing = false;
      button.onclick = () => {
        if (advancing) return;
        advancing = true;
        button.disabled = true;
        _cleanup();
        onDone();
      };
      return true;
    } catch (err) {
      console.warn("[example-gate] continuing without the example:", err);
      _cleanup();
      return false;
    }
  };

  return { maybeShow, _pick };
})();

window.ExampleGate = ExampleGate;
