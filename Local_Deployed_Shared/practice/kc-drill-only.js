/* ================================================================
   DRILL-ONLY CONCEPTS — Practice ⤢ on a lesson-less course (LeetCode)

   The ?lesson=<kc> ladder (practice/kc-practice.js) is built from the KP in
   lessons_structured.json, and the LeetCode Patterns course has none: every
   `leetcode.*` concept is drills only. So Practice ⤢ on a LeetCode bubble
   said "No lesson found" and served nothing.

   This builds the ladder entry for such a concept from the bank itself: every
   question whose `leetcode_kc` is this concept, easiest first
   (difficulty_score, then id). `drillOnly` tells kc-practice.js the two ways
   this ladder differs:

     • nothing sits behind it. An ARENA ladder hands over to the adaptive queue
       pinned to the KP's subtopic, which is a handful of sibling concepts.
       Every LeetCode drill shares ONE subtopic ("LeetCode: Patterns"), so
       that hand-over would be the whole course — not "this concept only".
       A spent drill-only ladder ends instead (KcPractice.drillOnlyDone).
     • a drill served before is not dropped: it comes back once, after every
       unseen one, so a second visit to a concept is a review rather than an
       instant dead end.

   No scoring here — grading, BKT and FIRe run through the normal submit path.
   ================================================================ */

const KcDrillOnly = (() => {
  const _score = (q) => {
    const d = Number(q.difficulty_score);
    return Number.isFinite(d) ? d : 0;
  };

  /** Ladder entry in LessonGate.getKpEntry's shape, or null when no bank
   *  question targets `kc`. */
  const entry = async (kc) => {
    if (!kc || typeof loadQuestionsBank !== "function") return null;
    const bank = await loadQuestionsBank();
    if (!Array.isArray(bank)) return null;
    const rows = bank
      .filter((q) => q && q.leetcode_kc === kc && Number.isFinite(Number(q.id)))
      .sort((a, b) => _score(a) - _score(b) || Number(a.id) - Number(b.id));
    if (!rows.length) return null;
    const first = rows[0];
    return {
      drillOnly: true,
      kp: {
        kc,
        title: first.concept_title || kc,
        independent_items: rows.map((q) => Number(q.id)),
      },
      lesson: { subtopic_key: first.subtopic_key || null, title: first.concept_title || kc },
    };
  };

  return { entry };
})();

window.KcDrillOnly = KcDrillOnly;
