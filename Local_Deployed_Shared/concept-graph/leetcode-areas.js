/* concept-graph/leetcode-areas.js — the LeetCode course's AREAS.
 *
 * Seth, 2026-10-01: "for the leetcode graph it actually displays the
 * sections for the color coding so that users can see the differences for
 * the different areas". Until then every `leetcode.*` concept was one
 * section ("clc", one lime), so Sections mode painted the whole course one
 * colour. These are the course's topic families, each in its own hue.
 *
 * The registry carries no grouping (all 40 concepts sit in lesson lc-1), so
 * the grouping lives here, by concept id. A `leetcode.*` id missing from it
 * falls into "Other LeetCode" rather than vanishing.
 *
 * Read by lesson-graph.js (`_sectionOf`, SECTION_ORDER) and mirrored by
 * graph-views.js's fallback. Plain script, BEFORE both.
 *
 *   window.DeltaLeetcodeAreas.of(kc)  → the area object, or null for a
 *                                        concept that is not LeetCode's
 *   window.DeltaLeetcodeAreas.list    → every area, in course order
 *
 * Area objects have the section shape: {id, label, color, order}. `order`
 * sorts them after ARENA and prep, before Delta Drills (graph-views.js
 * sectionOrder: courses are 200+). Colours avoid the yellow (#ffd23f) the map
 * keeps for its path highlight and "next up". */
(function () {
  "use strict";

  const AREAS = [
    ["lca", "Arrays & hashing", "#6fd08c", ["arrays-strings", "hash-maps", "prefix-sums"]],
    ["lcp", "Pointers & windows", "#b8dc6e", ["two-pointers", "sliding-window", "cyclic-sort"]],
    ["lcl", "Linked lists", "#4fd1c5", ["linked-lists", "fast-slow-pointers", "in-place-reversal"]],
    ["lcs", "Stacks", "#f2a65a", ["stack", "monotonic-stack"]],
    ["lcb", "Sorting & searching", "#5cc8f0", ["sorting-algorithms", "binary-search"]],
    ["lci", "Intervals & greedy", "#c9a27e", ["merge-intervals", "greedy"]],
    ["lcm", "Bits & maths", "#f0a3a3", ["bit-manipulation", "math"]],
    ["lct", "Trees & tries", "#a78bfa", ["binary-trees", "tree-bfs", "tree-dfs", "bst", "trie"]],
    ["lch", "Heaps", "#f07cc8", ["top-k-elements", "two-heaps", "k-way-merge"]],
    ["lcg", "Graphs", "#6f9cf0", ["graphs", "islands", "topological-sort", "union-find", "shortest-paths", "mst"]],
    ["lcr", "Recursion & backtracking", "#d9a0f5", ["recursion", "subsets", "backtracking"]],
    ["lcd", "Dynamic programming", "#ef6f6f", ["dp-1d", "knapsack-dp", "dp-grid", "dp-strings", "dp-lis", "dp-intervals"]],
  ];
  const OTHER = { id: "lcz", label: "Other LeetCode", color: "#b0b4c0", order: 200.99 };

  const list = [];
  const byKc = {};
  AREAS.forEach(([id, label, color, slugs], i) => {
    const area = { id, label, color, order: 200 + (i + 1) / 100 };
    list.push(area);
    slugs.forEach((slug) => { byKc["leetcode." + slug] = area; });
  });
  list.push(OTHER);

  const of = (kc, lessonId) => {
    const id = typeof kc === "string" ? kc : "";
    if (byKc[id]) return byKc[id];
    if (id.startsWith("leetcode.") || /^lc-/.test(lessonId || "")) return OTHER;
    return null;
  };

  window.DeltaLeetcodeAreas = { of, list };
})();
