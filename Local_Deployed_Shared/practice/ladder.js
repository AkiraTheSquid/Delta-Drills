/* ================================================================
   LADDER UI — the expertise-reversal rungs, on screen

   The backend decides WHICH rung a concept sits on (app/kc_graph.py:
   worked -> faded -> partial -> solo, promoted on a Wilson lower bound and
   demoted on a miss). This file is the part the learner sees:

     1. pointing the page-wide concept topbar (practice/concept-topbar.js) at
        whatever concept and rung the current card is on;
     2. the worked example itself at the `worked` rung, before any question;
     3. that same worked example kept ON SCREEN through the faded and partial
        rungs, beside the problem.

   (3) is the load-bearing one. Renkl & Atkinson's completion problems are
   completion problems because the example is still visible — the learner is
   filling in a solution they can see the shape of. Show the example, take it
   away, then ask for the missing step and you have not built a scaffold, you
   have built a recall test with extra ceremony. The example only disappears at
   `solo`, which is the rung that is supposed to be unsupported.

   The lesson content comes from `lessons_structured.json` through LessonGate,
   which already loads and caches it — no second copy of the KP records, and no
   new payload on the question response.
   ================================================================ */

const LadderUI = (() => {
  "use strict";

  // Rungs on which the worked example stays beside the problem. `solo` is
  // absent on purpose: the whole point of that rung is that support is gone.
  const SUPPORTED_STAGES = new Set(["faded", "partial"]);

  const STAGE_BLURB = {
    faded: "Most of the solution is written for you — supply the last step.",
    partial: "Read the example above, then write this one yourself.",
    solo: "No scaffold on this one. You have earned it.",
  };

  // Quotes matter here: every one of these values is interpolated into a
  // double-quoted HTML attribute (title=, data-kc=), and a concept title
  // carrying an apostrophe-free `"` would close the attribute early and
  // scramble the header. Escaping both quote forms costs nothing.
  const esc = (value) =>
    String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  const _kcOf = (q) => (q && q.ladder_kc) || null;
  const _stageOf = (q) => (q && q.ladder_stage) || null;

  /* ---------- worked-example acknowledgement --------------------------- */

  /* Tell the backend the example was read, and take back the re-staged
     starter for the question already on screen.

     Returns the server's response, or null when the call could not be made.
     A null is not fatal: the learner still saw the example, and the concept
     simply stays on the `worked` rung until a call succeeds. Failing OPEN in
     the other direction — pretending the promotion happened — would hand out a
     faded starter the server does not believe in, and the next question would
     silently drop back to the example again. */
  const noteWorkedSeen = async (kc, questionId) => {
    if (!kc || typeof apiFetch !== "function" || practiceMode !== "backend") return null;
    try {
      const res = await apiFetch("/api/practice/worked-seen", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          kc,
          question_id: Number.isFinite(questionId) ? questionId : null,
        }),
      });
      if (res.status === 401) {
        handleExpiredToken();
        return null;
      }
      if (!res.ok) return null;
      return await res.json();
    } catch (err) {
      console.warn("[ladder] could not record worked example:", err);
      return null;
    }
  };

  /* Acknowledge the example AND re-stage the question in one step.

     These have to happen together. The moment the server credits the example,
     the concept is on `faded`, and the question object the client is holding
     was cut for `worked` — same prompt, unfaded starter, and `decorate` will
     not attach the example because `worked` is not a supported stage. Render
     it as-is and the learner gets the one card the whole ladder was built to
     avoid: a solo problem dressed as a scaffolded one, immediately after being
     told they were about to get support.

     So: await, then mutate in place, so whatever renders next renders the rung
     the learner is actually on. */
  const applyWorkedSeen = async (question, kc) => {
    const res = await noteWorkedSeen(kc, question && question.question_id);
    if (!res || !question) return false;
    question.ladder_stage = res.ladder_stage || question.ladder_stage;
    if (res.ladder_estimate) question.ladder_estimate = res.ladder_estimate;
    // A null starter_code means "the question's own starter is right for this
    // rung" — a one-statement body has no honest half to hide.
    if (res.starter_code) question.starter_code = res.starter_code;
    return true;
  };

  /* Credit every KP a lesson screen just taught.

     A first-encounter gate can teach several KPs in one sitting, but at most
     one of them is the concept the pending question is staged on. That one is
     awaited and applied; the rest only need their counter moved, so they go
     out in parallel and nothing waits on them. */
  const creditTaught = async (kcs, question) => {
    const taught = [...new Set((kcs || []).filter(Boolean))];
    if (!taught.length) return;
    const staged = _kcOf(question);
    await Promise.all(
      taught.filter((kc) => kc !== staged).map((kc) => noteWorkedSeen(kc, null)),
    );
    if (staged && taught.includes(staged)) await applyWorkedSeen(question, staged);
  };

  /* The `worked` rung: teach before asking. FIRST CONTACT ONLY.

     The backend reaches this rung only when a concept has never been taught
     (kc_graph._stage_from returns it on `worked_seen == 0` and floors every
     demotion at `faded`), so getting here means the learner has genuinely not
     read this page before. That is deliberate: replaying a lesson after a miss
     is the system telling a learner they never read something they did read,
     and because the demotion re-derived from the last attempt it repeated
     before every question until they happened to answer correctly. Support
     after a miss now comes back as the example beside the problem, which is
     what `decorate` attaches on the supported rungs.

     The backend still attaches a question to this rung — it has to pick
     something to hand back — so rather than discarding it (which would burn a
     question out of a pool that is only two deep for some concepts) we show
     the example over the top of it. LessonGate credits and re-stages the
     question on its way out, so `onDone` renders the faded rung. */
  const maybeShowWorked = async (question, onDone) => {
    const kc = _kcOf(question);
    if (_stageOf(question) !== "worked" || !kc) return false;
    if (!window.LessonGate || typeof window.LessonGate.showLesson !== "function") return false;
    if (question.diagnostic_active) return false;

    const shown = await window.LessonGate.showLesson(kc, onDone, question);
    if (!shown) {
      // No KP page for this concept (unauthored lesson, or content that failed
      // to load). Credit the rung anyway and let the question through.
      //
      // This does record an example the learner never read, which is a real
      // cost — but the alternative is worse and permanent: leaving the concept
      // on `worked` parks it on a rung whose screen does not exist, so every
      // future question on it re-attempts the same missing page and the
      // learner can never reach a drill. Advancing is the recoverable failure;
      // a miss simply knocks them back down and they try again.
      await applyWorkedSeen(question, kc);
      return false;
    }
    return true;
  };

  /* ---------- per-card decoration -------------------------------------- */

  /* Put served KC, rung, support, estimate, question rating, queue aim into
     single page-wide ladder. */
  const _syncTopbar = (question) => {
    const bar = window.StageLadder;
    if (!bar) return;
    const kc = _kcOf(question);
    const stage = _stageOf(question);
    const target = typeof getTargetDifficultyForQuestion === "function"
      ? getTargetDifficultyForQuestion(question)
      : question?.target_difficulty;
    if (!kc || !stage) {
      bar.hide();
      return;
    }
    bar.show({
      kc,
      title: question.ladder_kc_title || kc,
      difficulty: question.difficulty,
      target,
      stage,
      integrated: !!question.ladder_integrated,
      support: question.ladder_support !== false,
      estimate: question.ladder_estimate || null,
    });
  };

  const _exampleHtml = (kp, seg, stage) => {
    const render =
      (window.LessonGate && window.LessonGate.renderMarkdown) ||
      ((text) => "<pre>" + esc(text) + "</pre>");
    const blurb = STAGE_BLURB[stage] || "";
    /* The blurb sits OUTSIDE the <details>, as a callout above it, since
       2026-08-24: as a muted line inside the example it was the single most
       important sentence on the card ("most of the solution is written for
       you") rendered least visibly, and a tester read the faded rung as
       write-it-from-scratch — "I almost didn't see this." */
    return (
      (blurb ? `<p class="ladder-stage-callout">${esc(blurb)}</p>` : "") +
      '<details class="ladder-example" open>' +
      `<summary>Worked example — ${esc(seg.title || kp.title)}</summary>` +
      '<div class="ladder-example-body">' +
      render(seg.worked_example_markdown) +
      "</div></details>"
    );
  };

  /* The example's code fences, in order, for the notebook editor. Prose stays
     in the rail; these become runnable cells above the learner's own
     (DeltaNotebook.showExamples). The fence regex tolerates an info string
     after the language ("```python title=x") but takes only real fences — an
     indented code block has no reliable boundary in this markdown. */
  const _exampleSnippets = (markdown) => {
    const codes = [];
    const fence = /```(?:python|py)?[^\n]*\n([\s\S]*?)```/g;
    let match;
    while ((match = fence.exec(String(markdown || "")))) {
      const code = match[1].trim();
      if (code) codes.push(code);
    }
    return codes;
  };

  /* Pick the segment whose example matches the problem being asked.

     29 of the 63 KPs hold several segments, each with its own example. Showing
     segment 1's example next to segment 3's problem is worse than showing
     none — it invites the learner to map a solution onto a problem it does not
     fit. A segment's `faded_items` name the questions it owns, so use that
     when it answers; otherwise fall back to the last segment, which is the
     KP's most complete example. */
  const _segmentFor = (kp, questionId) => {
    const segments = kp.segments && kp.segments.length ? kp.segments : null;
    if (!segments) {
      return {
        title: kp.title,
        worked_example_markdown: kp.worked_example_markdown,
      };
    }
    const owns = (seg) =>
      (seg.faded_items || []).some(
        (item) => item && Number(item.question_id) === Number(questionId),
      );
    return segments.find(owns) || segments[segments.length - 1];
  };

  // Guards against a stale async insert: by the time the KP JSON arrives the
  // learner may already be on the next question.
  let _decorateToken = 0;

  const decorate = (question) => {
    const host = document.getElementById("question-text");
    if (!host || !question) return;
    const token = ++_decorateToken;
    const kc = _kcOf(question);
    const stage = _stageOf(question);

    /* Synchronously, before any early return below: the previous question's
       example cells must not survive onto a question that attaches none (a
       solo rung, a KC-less item, a concept with no KP page). ui.js calls
       DeltaNotebook.reset AFTER decorate in the same task, and the KP fetch
       resolves after both, so the insert can never be wiped by the reset. */
    window.DeltaNotebook?.clearExamples?.();

    _syncTopbar(question);
    if (!kc || !stage) return;

    /* THE COLAB EDITION DOES NOT GET THE EXAMPLE HERE.

       On that deploy the problem lives in the notebook, and so does its worked
       example: `scripts/generate_colab_notebooks.py` emits the solved twin as
       cells directly ABOVE the problem's header, anchored `dd-q<n>-example` so
       `colab_focus.js` keeps the pair on screen together. Scroll up from the
       problem and there is the example; scroll down and there is the problem
       that is the same move on different specifics.

       Rendering a second copy into this rail would put the same content on
       screen twice, at two different widths, with the sidebar's copy being the
       one nobody asked for. The instruction was explicit — the example goes
       above the problem in Colab, not in the sidebar.

       `dd-no-notebook` is not an exception so much as the same rule read the
       other way: ui.js sets it for the ~75 questions with no published cell to
       route to. For those the rail IS the whole screen, there is no notebook
       holding the example, and dropping it here would delete the scaffold
       rather than relocate it. */
    if (
      window.DDColab
      && typeof window.DDColab.active === "function"
      && window.DDColab.active()
      && !document.documentElement.classList.contains("dd-no-notebook")
    ) {
      return;
    }

    if (!SUPPORTED_STAGES.has(stage)) return;
    if (!window.LessonGate || typeof window.LessonGate.getKpEntry !== "function") return;

    window.LessonGate.getKpEntry(kc)
      .then((found) => {
        if (!found || token !== _decorateToken) return;
        const seg = _segmentFor(found.kp, question.question_id);
        if (!seg || !seg.worked_example_markdown) return;
        host.insertAdjacentHTML("beforeend", _exampleHtml(found.kp, seg, stage));
        // The same example's code, runnable in the editor (tester ask,
        // 2026-08-24). showExamples handles the pane-hidden case itself.
        window.DeltaNotebook?.showExamples?.(_exampleSnippets(seg.worked_example_markdown));
      })
      .catch((err) => console.warn("[ladder] example unavailable:", err));
  };

  return {
    decorate,
    maybeShowWorked,
    applyWorkedSeen,
    creditTaught,
    noteWorkedSeen,
    SUPPORTED_STAGES,
  };
})();

window.LadderUI = LadderUI;
