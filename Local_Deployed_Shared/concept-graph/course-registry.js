/* concept-graph/course-registry.js — the courses the Courses tab
   (courses.js) and the Knowledge Graph view (graph-views.js) both read.

   Kept in lockstep BY HAND with the backend's
   This-Directory-Only/backend/app/course_registry.py — two small, rarely-
   changing lists, and nothing else in this repo shares a registry file
   across the frontend/backend boundary either. `milestoneKcs()` on the
   "arena" entry is async (it reads lessons/arena_exercise_kcs.json, the same
   file diagnostic.py's `_arena_links()` reads server-side); every other
   field is a plain literal.

   docs/spec-multi-course-catalog.md. */
(function () {
  "use strict";

  // Ids must match app/course_registry.py::DELTA_DRILLS_KCS exactly; titles
  // are lessons/kc_registry.json's, for the course's lesson list.
  const DELTA_DRILLS_TITLES = {
    "deltadrills.mastery": "What “mastery” means in this app",
    "deltadrills.spaced-repetition": "Why drills come back: spaced repetition",
    "deltadrills.why-drills": "Why drills, instead of just reading answers",
  };
  const DELTA_DRILLS_KCS = Object.keys(DELTA_DRILLS_TITLES);

  let arenaKcs = null;      // Set, once resolved successfully
  let arenaLoading = null;
  const loadArenaKcs = () => {
    if (arenaKcs) return Promise.resolve(arenaKcs);
    if (arenaLoading) return arenaLoading;
    arenaLoading = fetch("lessons/arena_exercise_kcs.json", { cache: "no-cache" })
      .then((r) => {
        if (!r.ok) throw new Error("arena_exercise_kcs.json " + r.status);
        return r.json();
      })
      .then((raw) => {
        const set = new Set();
        Object.entries(raw || {}).forEach(([slug, exercises]) => {
          if (slug.charAt(0) === "_" || !exercises || typeof exercises !== "object") return;
          Object.values(exercises).forEach((row) => {
            if (row && row.kc) set.add(row.kc);
          });
        });
        arenaKcs = set;
        arenaLoading = null;
        return set;
      })
      .catch((err) => {
        // Don't cache a failure as "no ARENA concepts" — that would filter
        // the course to nothing forever. Let the next call retry.
        arenaLoading = null;
        throw err;
      });
    return arenaLoading;
  };

  // LeetCode Patterns: its concepts are whatever leetcode.* rows the exporter
  // (This-Directory-Only/scripts/leetcode_course/export_app.py) wrote into
  // lessons/kc_registry.json — the backend reads the same prefix, so the two
  // registries cannot disagree on membership.
  let leetcode = null;      // {kcs: Set, titles: {}}, once resolved successfully
  let leetcodeLoading = null;
  const loadLeetcode = () => {
    if (leetcode) return Promise.resolve(leetcode);
    if (leetcodeLoading) return leetcodeLoading;
    leetcodeLoading = fetch("lessons/kc_registry.json", { cache: "no-cache" })
      .then((r) => {
        if (!r.ok) throw new Error("kc_registry.json " + r.status);
        return r.json();
      })
      .then((raw) => {
        const kcs = new Set();
        const titles = {};
        (raw?.kcs || []).forEach((kc) => {
          if (kc && typeof kc.id === "string" && kc.id.startsWith("leetcode.")) {
            kcs.add(kc.id);
            titles[kc.id] = kc.title || kc.id;
          }
        });
        leetcode = { kcs, titles };
        leetcodeLoading = null;
        return leetcode;
      })
      .catch((err) => {
        leetcodeLoading = null;  // same as ARENA: never cache a failure
        throw err;
      });
    return leetcodeLoading;
  };

  const COURSES = [
    {
      id: "arena",
      label: "ARENA",
      eyebrow: "AI safety, hands-on",
      detailKind: "arena",
      // Its concepts are the main graph whether or not this is on; on, a
      // share of drills comes from its own exercises before the gate.
      toggleLabel: "Mix its exercises into practice",
      milestoneKcs: loadArenaKcs,
    },
    {
      id: "delta-drills",
      label: "Delta Drills",
      eyebrow: "What this app is, and the math behind it",
      detailKind: "lesson-list",
      toggleLabel: "Add to practice",
      milestoneKcs: () => Promise.resolve(new Set(DELTA_DRILLS_KCS)),
      kcTitle: (kc) => DELTA_DRILLS_TITLES[kc] || kc,
    },
    {
      id: "leetcode",
      label: "LeetCode Patterns",
      eyebrow: "Coding-interview problems, by pattern",
      detailKind: "lesson-list",
      intro:
        "The Grokking coding-interview patterns as a graph: two pointers, sliding window, trees, graphs, heaps, DP and more, each with real LeetCode problems you solve in Python against hidden test cases. Open a pattern to work it on the Knowledge Graph, scoped to just this course.",
      toggleLabel: "Add to practice",
      milestoneKcs: () => loadLeetcode().then((c) => c.kcs),
      kcTitle: (kc) => (leetcode && leetcode.titles[kc]) || kc,
    },
  ];

  window.DeltaCourseRegistry = {
    list: () => COURSES,
    get: (id) => COURSES.find((c) => c.id === id) || null,
  };
})();
