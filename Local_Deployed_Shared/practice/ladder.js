/* ================================================================
   LADDER UI — the expertise-reversal rungs, on screen

   The backend decides WHICH rung a concept sits on (app/kc_graph.py:
   worked -> faded -> partial -> solo, promoted on a Wilson lower bound and
   demoted on a miss). This file is the part the learner sees:

     1. pointing the page-wide concept topbar (practice/concept-topbar.js) at
        whatever concept and rung the current card is on;
     2. the worked example itself at the `worked` rung, before any question —
        which is a LESSON screen (LessonGate.showLesson), not a decoration on
        a drill card.

   🪦 NO EXAMPLE IS ATTACHED TO A DRILL. `decorate` used to hang the KP
   segment's worked example off `#question-text` as a `<details
   class="ladder-example">` and lay its code fences in as runnable cells above
   the learner's own (`DeltaNotebook.showExamples`). Both are deleted, and both
   must stay deleted — `watch_example_gate.py` asserts it.

   The rungs it did that on shrank twice before dying. First `faded` came off
   (2026-08-28): the example is the segment's worked solution and the faded
   starter is that same solution with the blanks cut into it, so side by side
   the example was an answer key. Seth, then: "the example on the left
   completely gives away the scaffolded answer." Then `partial` came off
   (2026-08-30) in favour of a scheduled popup, which left `SUPPORTED_STAGES`
   an empty set and this whole path unreachable. The popup
   (`practice/example-gate.js`) is gone too as of 2026-09-10 — it wrote the
   example into the learner's OWN primary cell, and Seth met it on q198 as "a
   bunch of unrelated code" that Submit then graded as his answer.

   Where the examples are: on the Lesson rung, which the learner reads
   immediately before the faded rung opens. "We don't put worked examples in
   the problems anymore. They have their own lessons" (Seth, 2026-09-10).

   The lesson content comes from `lessons_structured.json` through LessonGate,
   which already loads and caches it — no second copy of the KP records, and no
   new payload on the question response.
   ================================================================ */

const LadderUI = (() => {
  "use strict";

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

  // What the learner is told when their own rung ran out of unseen problems
  // and the queue reached down a rung for something they have not done. Named
  // by the rung it reached TO, because that is the one on the screen.
  const GAP_FROM_LABEL = {
    faded: "fill-in-the-blank",
    partial: "solo",
    solo: "integrated",
    unranked: "unsorted",
  };

  /* The banner for a spent rung.

     Shown instead of silently looking like a demotion: the strip still names
     the rung the learner has EARNED (that is what their record says), while
     the problem in front of them came from a lower one because the earned rung
     has nothing left they have not already answered. Saying so is the whole
     difference between "the app went backwards" and "you have finished these".
     The server never repeats a solved problem — see
     prioritization.narrow_to_next_kc. */
  const _gapHtml = (gap) => {
    if (!gap || !gap.served_from) return "";
    /* Served from ANOTHER CONCEPT: the lattice's head had nothing left to
       serve (a rung with no drills written, or every drill answered), so the
       queue moved to the next concept on the frontier instead of stopping
       (questions_router.next_question, 2026-09-09). `kc_title` here names the
       concept that ran dry, not the one on screen. */
    if (gap.served_from === "other_concept") {
      const dry = gap.kc_title || gap.kc || "the next concept";
      return (
        '<p class="ladder-stage-callout ladder-gap-callout">'
        + `Nothing new is written yet for “${esc(dry)}” at your rung, so this `
        + "problem is from another concept on your frontier. Ask Claude for "
        + "more drills on it."
        + "</p>"
      );
    }
    const spent = GAP_FROM_LABEL[gap.stage] || gap.stage;
    const from = GAP_FROM_LABEL[gap.served_from] || gap.served_from;
    return (
      '<p class="ladder-stage-callout ladder-gap-callout">'
      + `You have done every ${esc(spent)} problem written for this concept`
      + (gap.seen ? ` (all ${esc(gap.seen)})` : "")
      + `. Nothing is being repeated — this one is from the ${esc(from)} rung `
      + "until more are written. Ask Claude for more drills on this concept."
      + "</p>"
    );
  };

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
    if (question.diagnostic_active || question.attempt_first) return false;

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

  /* Per-card decoration is now callouts ONLY.

     🪦 `_exampleHtml`, `_exampleSnippets`, `_segmentFor` and the KP fetch they
     fed were deleted 2026-09-10. They put the KP segment's worked example into
     `#question-text` as `<details class="ladder-example">` and its code fences
     into the notebook as runnable cells above the learner's own. Seth: "we
     don't put worked examples in the problems anymore. They have their own
     lessons." Unreachable since `SUPPORTED_STAGES` emptied on 2026-08-30, so
     nothing on screen changes — but the code being live was how the rail copy
     kept coming back. `watch_example_gate.py` is the ratchet.

     What survives is the pair of lines a learner needs ABOUT their own card and
     cannot read anywhere else: "you have not read the lesson yet", and "your
     rung is spent, this one came from a lower one". Neither is content.

     The async KP fetch went with the example, and with it `_decorateToken` —
     there is no longer anything here that resolves after the next question may
     have rendered, so there is nothing to stale-guard. */
  const decorate = (question) => {
    const host = document.getElementById("question-text");
    if (!host || !question) return;

    _syncTopbar(question);
    if (!_kcOf(question) || !_stageOf(question)) return;

    /* A spent rung is worth saying on EVERY card it affects. This used to sit
       above a Colab early-return and a supported-stage check, both of which
       existed only to gate the example; with the example gone the callouts are
       unconditional, which is what they always should have been. */
    if (question.attempt_first) {
      host.insertAdjacentHTML("beforeend", '<p class="ladder-stage-callout">First attempt — lesson not yet reviewed.</p>');
    }
    const gapHtml = _gapHtml(question.ladder_gap);
    if (gapHtml) host.insertAdjacentHTML("beforeend", gapHtml);
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
