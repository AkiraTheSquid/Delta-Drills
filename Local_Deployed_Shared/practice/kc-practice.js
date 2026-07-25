/* ================================================================
   SINGLE-KC PRACTICE LADDER — the ?lesson=<kc> flow

   Reached from the Knowledge Graph's "Practice ⤢" overlay. Replaces the
   old preview dead-end ("Demo lesson complete") with the real expertise-
   reversal ladder that lessons_structured.json has always carried but
   nothing ever served:

     lesson pages  →  faded_items  →  independent_items  →  focused queue

   Scaffolding is chosen by the learner's BKT posterior for this KC's
   subtopic, using the same ERE bands as arena-unlock.js (< 0.75 ⇒ faded).
   Faded items are REAL bank questions re-served with a blanked starter, so
   grading, BKT updates and FIRe credit all run through the normal submit
   path — nothing here is a parallel scoring system.

   Once every ladder item is used, isActive() goes false and the normal
   adaptive queue takes over, still pinned to this subtopic by
   window.__kcFocusSubtopic (see questions.js).
   ================================================================ */

const KcPractice = (() => {
  const FADED_CEIL = 0.75; // ERE band shared with arena-unlock.js

  let active = false;
  let kcId = null;
  let subtopicKeys = []; // see _resolveSubtopicKeys
  let queue = [];
  let served = 0;

  // The two modes disagree on what a "subtopic" string is, and BKT is keyed by
  // whatever `question.subtopic` says:
  //   backend  → composite, "Numpy: Core array literacy" (questions.py prefixes
  //              the topic so Numpy/Einops subtopics stay distinct)
  //   local    → bare, "Core array literacy" (questions.json keeps the
  //              composite separately as subtopic_key)
  // Carrying BOTH and matching on either keeps the bar and the focus filter
  // correct in each mode without branching on practiceMode everywhere.
  const _resolveSubtopicKeys = (lesson, sampleBankQ) => {
    const keys = [];
    const push = (v) => { if (v && !keys.includes(v)) keys.push(v); };
    push(sampleBankQ && sampleBankQ.subtopic);
    push(sampleBankQ && sampleBankQ.subtopic_key);
    push(lesson && lesson.subtopic_key);
    return keys;
  };

  // Composite form — what the backend's focus_subtopic expects.
  const _compositeKey = () => subtopicKeys.find((k) => k.includes(": ")) || subtopicKeys[0] || null;

  const _mastery = () => {
    for (const key of subtopicKeys) {
      const live = window.__subtopicMastery && window.__subtopicMastery[key];
      if (Number.isFinite(live)) return live;
    }
    for (const key of subtopicKeys) {
      try {
        const p = getEwmaFromAdaptiveState(key);
        if (Number.isFinite(p)) return p;
      } catch (_err) { /* try the next key */ }
    }
    return null;
  };

  // Faded first for a learner who is still below the independent band; above
  // it, go straight to unscaffolded problems and keep faded items in reserve
  // (they are still useful practice, just not the entry point).
  const _buildQueue = (kp) => {
    const faded = (kp.faded_items || [])
      .filter((it) => Number.isFinite(it?.question_id))
      .map((it) => ({ kind: "faded", questionId: it.question_id, starter: it.starter_code || null }));
    const independent = (kp.independent_items || [])
      .filter((id) => Number.isFinite(id))
      .map((id) => ({ kind: "independent", questionId: id, starter: null }));
    const p = _mastery();
    const scaffoldFirst = !Number.isFinite(p) || p < FADED_CEIL;
    return scaffoldFirst ? [...faded, ...independent] : [...independent, ...faded];
  };

  // Any bank record for this KP — used only to read its subtopic naming.
  const _sampleBankQuestion = (kp) => {
    if (typeof getQuestionFromBank !== "function") return null;
    const ids = [
      ...(kp.faded_items || []).map((it) => it && it.question_id),
      ...(kp.independent_items || []),
    ].filter((id) => Number.isFinite(id));
    for (const id of ids) {
      const q = getQuestionFromBank(id);
      if (q) return q;
    }
    return null;
  };

  const _hydrate = (item) => {
    const bankQ = typeof getQuestionFromBank === "function" ? getQuestionFromBank(item.questionId) : null;
    if (!bankQ) return null;
    const overrides = {};
    // Only override the starter when the faded item actually supplies one —
    // an empty string would wipe the question's own scaffold.
    if (item.kind === "faded" && item.starter) overrides.starter_code = item.starter;
    return buildPracticeQuestionFromBank(bankQ, overrides);
  };

  /** Start the ladder for a KC. Returns false when the KC has no lesson data. */
  const start = async (kc) => {
    if (!window.LessonGate || typeof window.LessonGate.getKpEntry !== "function") return false;
    const entry = await window.LessonGate.getKpEntry(kc);
    if (!entry) return false;
    await loadQuestionsBank();

    kcId = kc;
    // Keys must be resolved BEFORE _buildQueue — the faded/independent
    // ordering reads mastery, which is looked up by those keys.
    subtopicKeys = _resolveSubtopicKeys(entry.lesson, _sampleBankQuestion(entry.kp));
    queue = _buildQueue(entry.kp);
    served = 0;
    active = queue.length > 0;

    window.__kcFocusId = kc;
    window.__kcFocusSubtopics = subtopicKeys;
    window.__kcFocusSubtopic = _compositeKey();
    // The preview flag suppressed renderQuestion so a late resume couldn't
    // clobber the lesson. Graded practice starts now, so it has to come off.
    window.__lessonDemoOnly = false;

    if (window.CompetencyBar) {
      window.CompetencyBar.init(subtopicKeys);
      window.CompetencyBar.beginPractice();
    }
    return true;
  };

  /** Next ladder item, or null once the ladder is spent (queue takes over). */
  const nextQuestion = async () => {
    while (served < queue.length) {
      const item = queue[served++];
      const q = _hydrate(item);
      if (q) {
        q.ladder_kind = item.kind;
        if (window.CompetencyBar) window.CompetencyBar.setPhaseKind(item.kind);
        return q;
      }
      // Item points at a question the bank no longer has (curated exclusion or
      // a regen drop) — skip it rather than dead-ending the ladder.
      console.warn("[kc-practice] ladder item missing from bank:", item.questionId);
    }
    active = false;
    return null;
  };

  const isActive = () => active && !!queue.length;

  return {
    start,
    nextQuestion,
    isActive,
    get kc() { return kcId; },
    get subtopicKeys() { return subtopicKeys.slice(); },
  };
})();

window.KcPractice = KcPractice;
