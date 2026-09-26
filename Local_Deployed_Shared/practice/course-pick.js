/* ================================================================
   COURSE PICK — "Which courses do you want to study?"

   Seth, 2026-09-25: "when they go through the onboarding that is originally
   set up, it should have a single prompt question that asks them which
   courses they want to include for study. this is an intermediate step
   between the 'take me to diagnostic' and viewing diagnostic controls".

   So the welcome fork's right arm lands on #page-course-pick, not on the
   placement: one question, one tile per course
   (concept-graph/course-registry.js), and Continue. Continue posts the
   answer to POST /api/practice/study-courses and then opens #page-placement,
   whose set-up questions (practice/placement-wizard.js) now offer only the
   picked courses' areas.

   What the answer does is the backend's (app/course_registry.py::course_off):
   a course left out has its concepts switched off everywhere — practice, the
   placement, the Knowledge Graph — ARENA's whole graph included. The Courses
   tab changes it later; its toggles and this page read the same state.

   Like #page-placement this is a page with no tab: reached by name
   ([data-goto-tab="course-pick"]) and by nothing else.
   ================================================================ */
(function initCoursePick() {
  const byId = (id) => document.getElementById(id);
  const grid = byId("course-pick-grid");
  const go = byId("course-pick-continue");
  const note = byId("course-pick-note");
  if (!grid || !go) return;

  /* The line under each course name on this page. The Courses tab's own
     eyebrows are written for a catalogue card; a first-time learner choosing
     what to study needs to know what they would be practising. */
  const BLURB = {
    arena: "Python, NumPy, PyTorch, einops and the math the ARENA AI-safety curriculum assumes, then ARENA's own exercises.",
    leetcode: "Coding-interview problems by pattern: two pointers, sliding window, trees, graphs, heaps, dynamic programming.",
    "delta-drills": "Three short lessons on how this app works: what mastery means, spaced repetition, why drills.",
  };

  let chosen = null;    // Set of course ids; null until the saved state is read
  let saving = false;
  let touched = false;  // a tile clicked: a late load must not undo it

  const courses = () => (window.DeltaCourseRegistry ? window.DeltaCourseRegistry.list() : []);

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  const say = (text, warn) => {
    if (!note) return;
    note.textContent = text || "";
    note.classList.toggle("is-warning", !!warn);
  };

  const paint = () => {
    grid.textContent = "";
    if (!chosen) {
      grid.appendChild(el("p", "placement-step-loading", "Loading courses…"));
      go.disabled = true;
      return;
    }
    courses().forEach((course) => {
      const on = chosen.has(course.id);
      const btn = el("button", "placement-focus-tile");
      btn.type = "button";
      btn.dataset.course = course.id;
      btn.setAttribute("aria-pressed", String(on));
      btn.classList.toggle("is-selected", on);
      btn.appendChild(el("span", "placement-focus-check"));
      const body = el("span", "placement-focus-body");
      body.appendChild(el("span", "placement-focus-name", course.label));
      body.appendChild(el("span", "placement-focus-blurb", BLURB[course.id] || course.eyebrow || ""));
      btn.appendChild(body);
      btn.addEventListener("click", () => {
        if (saving) return;
        touched = true;
        if (chosen.has(course.id)) chosen.delete(course.id);
        else chosen.add(course.id);
        paint();
      });
      grid.appendChild(btn);
    });
    go.disabled = saving || !chosen.size;
    if (!saving) say(chosen.size ? "" : "Pick at least one course.", !chosen.size);
  };

  /* The learner's current answer: what they picked before, or, never asked,
     what is on today (ARENA, plus any course switched on in the Courses tab).
     Offline or signed out, ARENA alone — the app's default. */
  const load = async () => {
    let picked = null;
    try {
      const res = await apiFetch("/api/practice/course-shares");
      if (res.ok) {
        const data = await res.json();
        picked = (data.courses || []).filter((r) => r.studied).map((r) => r.course);
      }
    } catch (_) {
      picked = null;
    }
    if (touched && chosen) return;
    chosen = new Set(picked && picked.length ? picked : ["arena"]);
    paint();
  };

  const toPlacement = () => {
    window.PlacementWizard?.coursesChanged?.();
    switchTab("placement");
  };

  go.addEventListener("click", async () => {
    if (saving || !chosen || !chosen.size) return;
    saving = true;
    go.disabled = true;
    say("Saving…");
    try {
      const res = await apiFetch("/api/practice/study-courses", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ courses: [...chosen] }),
      });
      if (!res.ok) throw new Error(String(res.status));
    } catch (_) {
      saving = false;
      paint();
      say("Couldn't save that. Check your connection and try again.", true);
      return;
    }
    saving = false;
    say("");
    paint();
    // Every surface that reads which concepts are on: the graph, the
    // Courses tab's toggles, the readiness dial.
    try { await window.deltaRefreshKcLattice?.(); } catch (_) {}
    window.dispatchEvent(new CustomEvent("delta:courses-changed"));
    window.dispatchEvent(new CustomEvent("delta:adaptive-state-changed"));
    toPlacement();
  });

  /* Read once the backend mode is settled (a first-time visitor's guest
     account is minted by then), and again each time the page opens, so a
     change made on the Courses tab in between shows here. */
  const onOpen = () => {
    if (!byId("page-course-pick")?.classList.contains("hidden")) load();
  };
  window.addEventListener("delta:practice-mode-ready", () => { if (!saving) onOpen(); });
  if (window.DDPracticeModeReady) onOpen();
  document.querySelectorAll('[data-goto-tab="course-pick"]').forEach((b) =>
    b.addEventListener("click", () => {
      touched = false;
      setTimeout(load, 0);
    }));
  paint();
})();
