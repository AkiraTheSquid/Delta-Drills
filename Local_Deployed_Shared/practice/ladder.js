/* ================================================================
   LADDER UI — the expertise-reversal rungs, on screen

   The backend decides WHICH rung a concept sits on (app/kc_graph.py:
   worked -> faded -> partial -> solo, promoted on a Wilson lower bound and
   demoted on a miss). This file is the part the learner sees:

     1. pointing the page-wide concept topbar (practice/concept-topbar.js) at
        whatever concept and rung the current card is on;
     2. gating the `worked` rung — handing off to LessonGate so the learner
        reads the example before the first question on that concept, and
        crediting the rung when they have.

   Rendering the example is no longer part of it (2026-07-31). Renkl &
   Atkinson's completion problems are completion problems because the example
   stays visible while the learner fills in the missing step, and that property
   is preserved by the notebook's own layout: `generate_colab_notebooks.py`
   emits each concept as prose → worked example → that segment's faded problems,
   in one file, in that order. The example is a few cells above the problem the
   whole time — and unlike the panel's old copy, it can be run.

   The KP records still come from `lessons_structured.json` through LessonGate,
   which already loads and caches them — no second copy, and no new payload on
   the question response.
   ================================================================ */

const LadderUI = (() => {
  "use strict";

  /* `SUPPORTED_STAGES`, `STAGE_BLURB` and `esc` were here — the rung list the
     panel kept an example visible on, the one-line description of each rung,
     and the attribute escaper that rendered them. All three belonged to the
     in-panel example, which the notebook now carries. */

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
    // `getTargetDifficultyForQuestion` already resolves the backend field, the
    // local adaptive state and the item's own rating in that order, so the
    // guest path gets a sensible answer without a second code path here.
    const target =
      typeof getTargetDifficultyForQuestion === "function"
        ? getTargetDifficultyForQuestion(question)
        : question.target_difficulty;
    if (!kc || !stage) {
      // No ladder context on this question — a diagnostic probe, a KC-less
      // item, or the guest queue, which serves straight from the local bank and
      // has no ladder at all. The concept half of the strip must not survive
      // that: leaving the previous concept's name and rung up would label this
      // problem with a concept it has nothing to do with.
      //
      // The difficulty half is not a claim about a concept, so it stays. Drop
      // it here and a guest would have no difficulty readout anywhere on the
      // page, which is the one number that is meaningful for every problem the
      // app can serve.
      if (Number.isFinite(question.difficulty)) {
        bar.show({ difficulty: question.difficulty, target });
      } else {
        bar.hide();
      }
      return;
    }
    bar.show({
      kc,
      title: question.ladder_kc_title || kc,
      // This item's own rating, and where the adaptive queue is aiming — the
      // number that moves between questions.
      difficulty: question.difficulty,
      target,
      // Just "Concept". The rung itself is named by the dots, which speak the
      // learner-facing vocabulary; labelling it a second time here in the
      // backend's vocabulary would put two different names for one rung side
      // by side ("Fill in the rest" beside a dot reading "Faded").
      eyebrow: "Concept",
      stage,
      estimate: question.ladder_estimate || null,
    });
  };

  /* `_exampleHtml` and `_segmentFor` were here. They inserted the KP's worked
     example into #question-text beside faded and partial problems, picking the
     segment whose example matched the question being asked.

     Both are gone (2026-07-31), and the scaffold they existed to provide is
     NOT: the generated notebook lays each concept out as prose, then its worked
     example, then that segment's faded problems, in that order in one file. The
     example is therefore still on screen above the problem while the learner
     completes it — which is what makes a completion problem a completion
     problem — and it is there as runnable cells rather than a static excerpt.
     Keeping a second copy in the panel would have shown the example twice, and
     the panel's copy could not be run. */

  const decorate = (question) => {
    if (!question) return;
    // The topbar is the whole of it now: which concept, which rung, where the
    // difficulty is aimed. The worked example that used to be inserted here
    // lives in the notebook, above this problem's cell.
    _syncTopbar(question);
  };

  return {
    decorate,
    maybeShowWorked,
    applyWorkedSeen,
    creditTaught,
    noteWorkedSeen,
  };
})();

window.LadderUI = LadderUI;
