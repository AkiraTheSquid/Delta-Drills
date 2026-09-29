/* course-builder/builder-data.js — what the course builder's graph and lesson
   view read: the concept registry, each concept's lesson, and which concepts
   a course covers.

   The same sources the Knowledge Graph tab reads (concept-graph/lesson-graph.js):
   lessons/kc_registry.json for the nodes and prerequisite edges,
   lessons/lessons_structured.json for the lessons, and
   concept-graph/lessonless-concepts.js for the concept pages of a course with
   no lessons (LeetCode). A course's concepts are the same set the KG course
   dropdown shows (concept-graph/graph-views.js): its milestone concepts from
   concept-graph/course-registry.js plus every prerequisite under them.

   Loaded once, on the builder's first open; a failed load is not cached, so
   the next open retries. */
(function () {
  "use strict";

  let data = null;     // { kcs, parents, children, content, lessons } — all keyed by id
  let loading = null;

  const fetchJson = (url) =>
    fetch(url, { cache: "no-cache" }).then((r) => {
      if (!r.ok) throw new Error(`${url} ${r.status}`);
      return r.json();
    });

  const load = () => {
    if (data) return Promise.resolve(data);
    if (loading) return loading;
    loading = Promise.all([
      fetchJson("lessons/kc_registry.json"),
      fetchJson("lessons/lessons_structured.json"),
    ])
      .then(async ([registry, structured]) => {
        const kcs = {}, parents = {}, children = {}, content = {}, lessons = {};
        (registry.lessons || []).forEach((l) => { lessons[l.id] = l.title || l.id; });
        (registry.kcs || []).forEach((k) => {
          kcs[k.id] = k;
          parents[k.id] = (k.prereqs || []).slice();
          children[k.id] = children[k.id] || [];
        });
        Object.values(kcs).forEach((k) => {
          (k.prereqs || []).forEach((p) => { (children[p] = children[p] || []).push(k.id); });
        });
        (structured.lessons || []).forEach((l) => (l.kps || []).forEach((kp) => { content[kp.kc] = kp; }));
        try {
          const pages = window.DDLessonlessConcepts ? await window.DDLessonlessConcepts.load() : {};
          Object.entries(pages || {}).forEach(([kc, kp]) => { if (kcs[kc] && !content[kc]) content[kc] = kp; });
        } catch (_) { /* a lesson-less course then shows "no lesson yet" */ }
        data = { kcs, parents, children, content, lessons };
        loading = null;
        return data;
      })
      .catch((err) => {
        loading = null;
        throw err;
      });
    return loading;
  };

  // Seeds plus every prerequisite beneath them.
  const closure = (seeds) => {
    const out = new Set(seeds);
    const stack = [...seeds];
    while (stack.length) {
      (data.parents[stack.pop()] || []).forEach((p) => {
        if (!out.has(p) && data.kcs[p]) { out.add(p); stack.push(p); }
      });
    }
    return out;
  };

  // The concepts of one course; "" = every concept in the registry.
  const conceptsFor = async (courseId) => {
    await load();
    if (!courseId) return new Set(Object.keys(data.kcs));
    const course = window.DeltaCourseRegistry && window.DeltaCourseRegistry.get(courseId);
    if (!course) return new Set(Object.keys(data.kcs));
    const set = await course.milestoneKcs();
    return closure([...(set || [])].filter((kc) => data.kcs[kc]));
  };

  window.DDBuilderData = {
    load,
    conceptsFor,
    kc: (id) => (data && data.kcs[id]) || null,
    lesson: (id) => (data && data.content[id]) || null,
    lessonTitle: (lessonId) => (data && data.lessons[lessonId]) || "",
    parents: (id) => (data && data.parents[id]) || [],
    children: (id) => (data && data.children[id]) || [],
  };
})();
