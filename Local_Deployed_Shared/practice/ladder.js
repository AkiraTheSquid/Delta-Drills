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
    partial: "Half of the solution is written for you — finish it.",
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

  /* The `worked` rung: teach before asking.

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

  /* The in-card header this file used to build — concept name, rung, interval —
     now lives in the page-wide topbar (practice/concept-topbar.js), which does
     not scroll away and is shared with the lesson screen. Rendering both would
     put the concept and its estimate on screen twice, three inches apart, with
     the card's copy going stale the moment a submit updated the other. */
  const _syncTopbar = (question) => {
    const bar = window.ConceptTopbar;
    if (!bar) return;
    const kc = _kcOf(question);
    const stage = _stageOf(question);
    if (!kc || !stage) {
      // No ladder context on this question (a diagnostic probe, or a KC-less
      // item). Hiding is the honest move: leaving the previous concept's bar up
      // would label this problem with a concept it has nothing to do with.
      bar.hide();
      return;
    }
    bar.show({
      kc,
      title: question.ladder_kc_title || kc,
      // Just "Concept". The rung itself is named by the dots, which speak the
      // learner-facing vocabulary; labelling it a second time here in the
      // backend's vocabulary would put two different names for one rung side
      // by side ("Fill in the rest" beside a dot reading "Faded").
      eyebrow: "Concept",
      stage,
      estimate: question.ladder_estimate || null,
    });
  };

  const _exampleHtml = (kp, seg, stage) => {
    const render =
      (window.LessonGate && window.LessonGate.renderMarkdown) ||
      ((text) => "<pre>" + esc(text) + "</pre>");
    const blurb = STAGE_BLURB[stage] || "";
    return (
      '<details class="ladder-example" open>' +
      `<summary>Worked example — ${esc(seg.title || kp.title)}</summary>` +
      (blurb ? `<p class="ladder-example-blurb">${esc(blurb)}</p>` : "") +
      '<div class="ladder-example-body">' +
      render(seg.worked_example_markdown) +
      "</div></details>"
    );
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

    _syncTopbar(question);
    if (!kc || !stage) return;

    if (!SUPPORTED_STAGES.has(stage)) return;
    if (!window.LessonGate || typeof window.LessonGate.getKpEntry !== "function") return;

    window.LessonGate.getKpEntry(kc)
      .then((found) => {
        if (!found || token !== _decorateToken) return;
        const seg = _segmentFor(found.kp, question.question_id);
        if (!seg || !seg.worked_example_markdown) return;
        host.insertAdjacentHTML("beforeend", _exampleHtml(found.kp, seg, stage));
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
