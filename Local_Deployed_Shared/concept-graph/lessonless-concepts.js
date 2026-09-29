/* ================================================================
   LESSON-LESS CONCEPT PAGES — what a LeetCode bubble opens

   lesson-graph.js renders a clicked bubble from its lessons_structured.json
   KP, and the LeetCode Patterns course has no KPs: it is drills only. So a
   `leetcode.*` bubble opened nothing — no description, no Practice ⤢.

   This reads lessons/leetcode/concepts.json (written by
   This-Directory-Only/scripts/leetcode_course/export_concepts.py from
   concept_notes.md + the exported drills) and hands lesson-graph.js a
   KP-shaped record per concept: `concept_markdown` is the note plus a
   "Practise it" block naming the concept's drills. `drill_only` marks it so
   nothing mistakes it for a taught lesson.

   Practice ⤢ on such a bubble opens ?lesson=<kc> like any other; the
   practice side (practice/kc-drill-only.js) serves that concept's drills
   and nothing else.

   Optional by design: a missing file costs the pages, never the graph.
   ================================================================ */
(() => {
  "use strict";

  const SOURCES = ["lessons/leetcode/concepts.json"];
  const ORDER = ["easy", "medium", "hard"];

  const _practiseBlock = (c) => {
    if (!c.drills) return "#### Practise it\n\nNo drills are written for this concept yet.";
    const counts = c.by_difficulty || {};
    const labels = [...ORDER, ...Object.keys(counts).filter((k) => !ORDER.includes(k))];
    const split = labels.filter((k) => counts[k]).map((k) => `${counts[k]} ${k}`).join(" · ");
    const lines = [
      "#### Practise it",
      "",
      `${c.drills} drill${c.drills === 1 ? "" : "s"} on this concept (${split}). ` +
        "Press **Practice ⤢** to work through them one concept at a time, easiest first.",
      "",
    ];
    (c.examples || []).forEach((ex) => {
      lines.push(`- ${ex.title}${ex.difficulty ? ` *(${ex.difficulty})*` : ""}`);
    });
    if (c.drills > (c.examples || []).length) lines.push(`- …and ${c.drills - c.examples.length} more`);
    return lines.join("\n");
  };

  const _asKp = (kc, c) => ({
    kc,
    title: c.title,
    concept_markdown: `${c.description_markdown || ""}\n\n${_practiseBlock(c)}`,
    drill_only: true,
  });

  let loading = null;
  const load = () => {
    if (loading) return loading;
    loading = Promise.all(SOURCES.map((url) =>
      fetch(url, { cache: "no-cache" }).then((r) => (r.ok ? r.json() : null)).catch(() => null)))
      .then((docs) => {
        const out = {};
        docs.forEach((doc) => {
          Object.entries((doc && doc.kcs) || {}).forEach(([kc, c]) => { out[kc] = _asKp(kc, c); });
        });
        // Nothing loaded (offline, a redeploy mid-fetch): let the next call retry.
        if (docs.every((d) => !d)) loading = null;
        return out;
      });
    return loading;
  };

  window.DDLessonlessConcepts = { load };
})();
