/* concept-graph/course-graph.js — a small, still picture of each course's
 * graph on its Courses-tab card, with a Maximize button that opens the real
 * Knowledge Graph scoped to that course (Seth, 2026-09-29: "each course …
 * a graph preview … maximize … opens the full graph for that course", which
 * replaced the graph's own course menus).
 *
 *   What it draws  — the course's concepts plus every prerequisite under them,
 *                    the same closure graph-views.js scopes the full graph to,
 *                    so the preview and the maximized graph hold the same
 *                    nodes. Coloured by area (`window.deltaKcSection`, the
 *                    header colours), laid out bottom-up like the real one.
 *   What it is NOT — a second graph to read: no labels, no pan/zoom, no
 *                    selection, no mastery. Clicking anywhere on it (or the
 *                    ⤢ button) is Maximize.
 *
 *   API  window.DDCourseGraphPreview.mount(el, courseId)  — idempotent per el
 *        window.DDCourseGraphPreview.open(courseId)       — Maximize
 *
 * courses.js builds the cards; until it calls mount() itself, this file
 * mounts onto every `.course-catalog-card` it sees in #courses-list-view,
 * matching the card's title to a course-registry label. */
(function () {
  "use strict";

  const REGISTRY_URL = "lessons/kc_registry.json";
  const FALLBACK = "#9aa3b2";
  const live = new Map(); // host element -> {cy, ro, courseId}

  let regLoad = null;
  const loadRegistry = () => {
    regLoad ??= fetch(REGISTRY_URL, { cache: "force-cache" })
      .then((r) => { if (!r.ok) throw new Error("kc_registry.json " + r.status); return r.json(); })
      .then((raw) => {
        const parents = {};
        (raw.kcs || []).forEach((k) => { parents[k.id] = (k.prereqs || []).slice(); });
        return parents;
      })
      .catch((err) => { regLoad = null; throw err; });
    return regLoad;
  };

  // Same closure as graph-views.js: the course's concepts and every ancestor.
  const closure = (seeds, parents) => {
    const out = new Set(seeds.filter((kc) => parents[kc]));
    const stack = [...out];
    while (stack.length) {
      (parents[stack.pop()] || []).forEach((p) => {
        if (parents[p] && !out.has(p)) { out.add(p); stack.push(p); }
      });
    }
    return out;
  };

  const sectionColor = (kc) => {
    const s = typeof window.deltaKcSection === "function" ? window.deltaKcSection(kc) : null;
    return (s && s.color) || FALLBACK;
  };

  const open = (courseId) => {
    if (typeof window.deltaSetKgCourseFilter === "function") window.deltaSetKgCourseFilter(courseId);
    // app.js's switchTab is a script-global const, not a window property.
    if (typeof switchTab === "function") switchTab("knowledge-graph");
    else if (typeof window.switchTab === "function") window.switchTab("knowledge-graph");
  };

  const fit = (cy) => { cy.resize(); cy.fit(undefined, 10); };

  const drop = (el) => {
    const it = live.get(el);
    if (!it) return;
    try { it.ro && it.ro.disconnect(); } catch (_) {}
    try { it.cy && it.cy.destroy(); } catch (_) {}
    it.wrap?.remove(); // a remount onto the same host starts from an empty one
    live.delete(el);
  };
  // Cards are rebuilt wholesale (courses.js renderList); an instance whose
  // card left the page is torn down the next time anything mounts.
  const sweep = () => { live.forEach((_, el) => { if (!el.isConnected) drop(el); }); };

  const mount = (el, courseId) => {
    if (!el || !courseId) return;
    sweep();
    if (live.has(el) && live.get(el).courseId === courseId) return;
    drop(el);
    const course = window.DeltaCourseRegistry && window.DeltaCourseRegistry.get(courseId);
    if (!course || typeof window.cytoscape !== "function") return;

    const wrap = document.createElement("div");
    wrap.className = "ccg-preview";
    const canvas = document.createElement("div");
    canvas.className = "ccg-canvas";
    canvas.setAttribute("aria-hidden", "true");
    const max = document.createElement("button");
    max.type = "button";
    max.className = "ccg-max";
    max.title = "Open this course's graph";
    max.setAttribute("aria-label", `Open the ${course.label} graph`);
    max.textContent = "⤢";
    const status = document.createElement("span");
    status.className = "ccg-status";
    status.textContent = "Loading graph…";
    wrap.append(canvas, status, max);
    el.appendChild(wrap);
    wrap.addEventListener("click", (e) => { e.stopPropagation(); open(courseId); });

    const entry = { cy: null, ro: null, courseId, wrap };
    live.set(el, entry);

    Promise.all([loadRegistry(), Promise.resolve(course.milestoneKcs())]).then(([parents, set]) => {
      if (live.get(el) !== entry) return;
      const kcs = closure([...(set instanceof Set ? set : new Set(set || []))], parents);
      if (!kcs.size) { status.textContent = "No concepts yet"; return; }
      const elements = [];
      kcs.forEach((kc) => elements.push({ data: { id: kc, c: sectionColor(kc) } }));
      let ei = 0;
      kcs.forEach((kc) => (parents[kc] || []).forEach((p) => {
        if (kcs.has(p)) elements.push({ data: { id: "e" + ei++, source: p, target: kc } });
      }));
      const cy = window.cytoscape({
        container: canvas,
        elements,
        // fit() clamps to this and centres, so a three-concept course is not
        // blown up to fill the card.
        maxZoom: 1,
        userZoomingEnabled: false, userPanningEnabled: false,
        boxSelectionEnabled: false, autoungrabify: true, autounselectify: true,
        style: [
          { selector: "node", style: { width: 14, height: 14, "background-color": "data(c)", "border-width": 0 } },
          { selector: "edge", style: { width: 1, "line-color": "rgba(150,150,170,0.45)", "curve-style": "straight" } },
        ],
        // The plugin, not graphlib: window.dagre can exist without cytoscape-dagre.
        layout: window.cytoscapeDagre
          ? { name: "dagre", rankDir: "BT", nodeSep: 12, rankSep: 40, animate: false, fit: true, padding: 10 }
          : { name: "grid", fit: true, padding: 10 },
      });
      entry.cy = cy;
      status.textContent = `${kcs.size} concept${kcs.size === 1 ? "" : "s"}`;
      // The Courses page may be hidden when this runs; fit once it has a size.
      if (typeof ResizeObserver === "function") {
        entry.ro = new ResizeObserver(() => { if (canvas.offsetWidth) fit(cy); });
        entry.ro.observe(canvas);
      }
    }).catch(() => {
      if (live.get(el) === entry) status.textContent = "Graph unavailable";
    });
  };

  // Area colours come from lesson-graph.js's section map, which may land after
  // a preview was drawn: repaint every live preview when it does.
  const repaint = () => live.forEach((it) => {
    if (!it.cy) return;
    it.cy.nodes().forEach((n) => n.data("c", sectionColor(n.id())));
  });
  ["delta:kg-sections-ready", "delta:practice-target-graph-ready"].forEach((ev) => window.addEventListener(ev, repaint));

  /* ---- self-mount onto the Courses cards (until courses.js calls mount) ---- */
  const courseForCard = (card) => {
    if (card.dataset.course) return card.dataset.course;
    const title = card.querySelector(".course-catalog-card-title");
    const reg = window.DeltaCourseRegistry;
    if (!title || !reg) return null;
    const hit = reg.list().find((c) => c.label === title.textContent.trim());
    return hit ? hit.id : null;
  };
  const scan = (root) => {
    root.querySelectorAll(".course-catalog-card").forEach((card) => {
      const body = card.querySelector(".course-catalog-card-text") || card;
      if (body.querySelector(".ccg-preview")) return;
      const id = courseForCard(card);
      if (id) mount(body, id);
    });
  };
  const boot = () => {
    const list = document.getElementById("courses-list-view");
    if (!list) return;
    scan(list);
    // Sweep first: a rebuild with no mountable card still frees the old ones.
    new MutationObserver(() => { sweep(); scan(list); }).observe(list, { childList: true, subtree: true });
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  window.DDCourseGraphPreview = { mount, open };
})();
