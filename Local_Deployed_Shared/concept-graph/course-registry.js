/* concept-graph/course-registry.js — the two courses the Courses tab
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
  ];

  window.DeltaCourseRegistry = {
    list: () => COURSES,
    get: (id) => COURSES.find((c) => c.id === id) || null,
  };
})();
