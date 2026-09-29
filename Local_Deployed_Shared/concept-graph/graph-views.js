/* concept-graph/graph-views.js — how much of the Knowledge Graph a learner sees.
 *
 * lesson-graph.js draws every concept, every time. This file layers TWO views
 * over that one graph (a segment in kg-toolbar.js's bar switches them):
 *
 *   complete  — the whole graph, laid out with each SECTION as one cluster
 *               (kg-look.js `routeLayout`'s `groupBy`), so kg-sections.js can
 *               hang one header over each area.
 *   condensed — one bubble per ARENA section (−1.0 Python, −1.1 arrays,
 *               0.0, 0.1, …) with the number of prerequisite links between
 *               sections on the edges. Tapping a section opens it into its
 *               concepts, in place, inside a labelled frame.
 *
 * and ONE scope: a COURSE (`window.deltaSetKgCourseFilter`). The Courses tab's
 * Maximize (course-graph.js) sets it, so the full-page graph shows that
 * course's own concepts plus everything they build on. There is no picker on
 * the graph any more (Seth, 2026-09-29: "we don't have to add … a menu to
 * isolate it to whichever curriculum"). The Adaptive horizon view and the
 * Chapters filter went with that cleanup; hiding an area is now the Filter's
 * per-area practice switch (kg-toolbar.js), which greys it out instead.
 *
 * 🔴 The graph stays lesson-graph.js's. Complete operates on ITS Cytoscape
 * instance (`window.deltaConceptGraphCy()`) under the same contract
 * instructor-graph-edit.js honours. The course scope is the only `cy.remove`
 * on that instance (collection kept, `.restore()`d before the next view).
 *
 * Condensed is a SECOND, private Cytoscape instance in an overlay over the
 * same box. Section nodes have no lesson, no lattice row and no learner
 * model, and lesson-graph.js's tap handler would try to render one; giving
 * them their own canvas keeps that file's assumptions true (every node id is
 * a KC). Tapping a concept inside an opened section hands off to
 * `window.deltaFocusConceptGraphKc`, so the side panel lights up exactly as
 * it does from the map.
 *
 * Section membership is asked of lesson-graph.js when it exports it
 * (`window.deltaKcSection`); until then the same rule is mirrored from
 * `lessons/arena_exercise_kcs.json` — see `_sectionFallback`. */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const VIEW_KEY = "dd_kg_view";
  const COURSE_KEY = "dd_kg_course";
  const MODES = ["complete", "condensed"];

  /* ---------------- section taxonomy ---------------------------------- */
  // Prep tiers are ours; ARENA's sections come from the exercise map's slugs.
  // Labels here are the FALLBACK only — the real labels and colours are
  // lesson-graph.js's ARENA_SECTIONS, read through `window.deltaKcSection`
  // when that export exists.
  const PREP_PYTHON = { id: "sm10", label: "Section −1.0 — Python", color: "#dfae74", order: 0 };
  const PREP_ARRAYS = { id: "sm11", label: "Section −1.1 — arrays, einops, tensors", color: "#b0b4c0", order: 1 };
  const PREP_MATH = { id: "sm12", label: "Section −1.2 — the maths underneath", color: "#f0a3a3", order: 2 };
  const LATER = { id: "sNN", label: "Later ARENA sections", color: "#e0709a", order: 99 };
  // The standalone courses, as lesson-graph.js's COURSE_LEETCODE / COURSE_DELTA
  // (2026-09-25): before, their concepts fell through to −1.1.
  const COURSE_LEETCODE = { id: "clc", label: "LeetCode Patterns — standalone course", color: "#b8dc6e", order: 200 };
  const COURSE_DELTA = { id: "cdd", label: "Delta Drills — how the app works", color: "#c4a8f0", order: 201 };
  const SLUG_COLORS = { "0-0": "#4f9fe0", "0-1": "#bb7de8", "0-2": "#e8a765" };
  const _slugSection = (slug) => {
    const m = /^(\d+)-(\d+)$/.exec(slug);
    if (!m) return LATER;
    return { id: "s" + m[1] + m[2], label: `Section ${m[1]}.${m[2]}`, color: SLUG_COLORS[slug] || LATER.color,
             order: 10 + Number(m[1]) * 10 + Number(m[2]) };
  };

  let arenaSlugByKc = {};        // kc -> notebook slug, earliest wins
  const loadArenaMap = () =>
    fetch("lessons/arena_exercise_kcs.json", { cache: "no-cache" })
      .then((r) => (r.ok ? r.json() : null))
      .then((map) => {
        arenaSlugByKc = {};
        if (map && typeof map === "object") {
          Object.keys(map).sort().forEach((slug) => {
            if (slug.charAt(0) === "_") return;
            const ex = map[slug];
            if (!ex || typeof ex !== "object") return;
            Object.keys(ex).forEach((fn) => {
              const kc = ex[fn] && ex[fn].kc;
              if (kc && !arenaSlugByKc[kc]) arenaSlugByKc[kc] = slug;
            });
          });
        }
      })
      .catch(() => {});

  const _sectionFallback = (kc, lessonId) => {
    const slug = arenaSlugByKc[kc];
    if (slug) return _slugSection(slug);
    if (String(kc).startsWith("leetcode.") || /^lc-/.test(lessonId || "")) return COURSE_LEETCODE;
    if (String(kc).startsWith("deltadrills.") || /^dd-/.test(lessonId || "")) return COURSE_DELTA;
    if (String(kc).startsWith("python.") || /^py-/.test(lessonId || "")) return PREP_PYTHON;
    if (String(kc).startsWith("math.") || /^ma-/.test(lessonId || "")) return PREP_MATH;
    return PREP_ARRAYS;
  };
  const sectionOf = (kc) => {
    if (typeof window.deltaKcSection === "function") {
      const s = window.deltaKcSection(kc);
      if (s && s.id) return s;
    }
    return _sectionFallback(kc, nodeLesson[kc]);
  };
  const sectionOrder = (s) => {
    if (Number.isFinite(s.order)) return s.order;
    // lesson-graph.js's objects carry no order; derive one from the id.
    // "sm10"/"sm11" are the prep chapters (−1.0, −1.1) and sort first;
    // "s00", "s01", … follow numerically. Anything else goes last.
    const m = /^s(m?)(\d+)$/.exec(s.id || "");
    if (!m) return 9999;
    return (m[1] ? 0 : 1000) + Number(m[2]);
  };

  /* ---------------- state --------------------------------------------- */
  let cy = null;                 // lesson-graph.js's instance
  let ccy = null;                // the condensed instance (ours)
  let condensedLay = null;       // its running layout, stopped before a rebuild
  // "adaptive" (pre-2026-09-29) is gone; a stored one reads as complete.
  let mode = "complete";
  let courseFilter = null;       // a course-registry id, or null = every concept
  let courseVisible = null;      // Set: that course's milestones + prereq closure, or null while loading/off
  const courseMilestones = {};   // course id -> Set, resolved lazily from course-registry.js
  let openSections = new Set();  // condensed: sections opened into concepts
  let removed = null;            // cy collection of everything we took off the canvas
  const parents = {}, children = {}, nodeLesson = {}, nodeLabel = {};
  let allKcs = [];
  let panel = null, hintEl = null;
  let origFocus = null;          // lesson-graph.js's deltaFocusConceptGraphKc, pre-wrap
  let applying = false;

  try { const m = localStorage.getItem(VIEW_KEY); if (MODES.includes(m)) mode = m; } catch (_) {}
  try { courseFilter = localStorage.getItem(COURSE_KEY) || null; } catch (_) {}

  const persist = () => {
    try {
      localStorage.setItem(VIEW_KEY, mode);
      if (courseFilter) localStorage.setItem(COURSE_KEY, courseFilter);
      else localStorage.removeItem(COURSE_KEY);
    } catch (_) {}
  };
  // The page head (index.html's course title) and kg-sections.js follow the
  // scope; the detail is the course id or null.
  const paintCourseTitle = () => {
    const el = document.getElementById("kg-course-title");
    if (!el) return;
    const course = courseFilter && window.DeltaCourseRegistry && window.DeltaCourseRegistry.get(courseFilter);
    el.textContent = courseFilter ? (course ? course.label : courseFilter) : "All courses";
  };
  const announceCourse = () => {
    paintCourseTitle();
    window.dispatchEvent(new CustomEvent("delta:kg-course-changed", { detail: { course: courseFilter } }));
  };

  /* ---------------- course scope (concept-graph/course-registry.js) ---- */
  // A takeOff on every view: ONE course's own concepts plus their
  // prerequisite closure, so picking a course roots the graph on it without a
  // second rendering path. `courseVisible` is null while off OR while that
  // course's milestone set hasn't resolved yet (course-registry.js's ARENA
  // lookup is a fetch) — null reads as "don't hide anything" (fail open).
  const resolveCourseMilestones = (id, course) => {
    if (courseMilestones[id]) return Promise.resolve(courseMilestones[id]);
    // Registry script not loaded yet (or blocked) is a TRANSIENT failure —
    // reject so the caller can fail open, not "this course has no concepts".
    if (!course) return Promise.reject(new Error("course registry not loaded"));
    return Promise.resolve(course.milestoneKcs()).then((set) => {
      courseMilestones[id] = set instanceof Set ? set : new Set(set || []);
      return courseMilestones[id];
    });
  };
  const applyCourseFilter = () => {
    if (!courseFilter) { courseVisible = null; applyView(); return; }
    const requested = courseFilter;
    const course = window.DeltaCourseRegistry && window.DeltaCourseRegistry.get(requested);
    // A ONCE-valid id the registry no longer knows (e.g. stale localStorage
    // after a course was removed) resets to every concept instead of silently
    // filtering the graph to nothing. A registry that hasn't loaded yet is
    // different (see resolveCourseMilestones) and fails open below.
    if (window.DeltaCourseRegistry && !course) {
      courseFilter = null;
      courseVisible = null;
      persist();
      announceCourse();
      applyView();
      return;
    }
    resolveCourseMilestones(requested, course).then((set) => {
      if (courseFilter !== requested) return; // filter moved on while this load was in flight
      const seeds = [...set].filter((kc) => allKcs.includes(kc));
      courseVisible = closure(seeds);
      applyView();
    }).catch(() => {
      // Load failed — fail open: show everything rather than filter to nothing
      // or keep the PREVIOUS course's subset under the new selection.
      if (courseFilter !== requested) return;
      courseVisible = null;
      applyView();
    });
  };
  const courseHidden = (kc) => !!(courseFilter && courseVisible && !courseVisible.has(kc));

  // The Courses tab's Maximize (course-graph.js) and "View course"
  // (courses.js) call this to land on the graph already scoped to that
  // course. Safe before the graph exists: init() applies `courseFilter`.
  window.deltaSetKgCourseFilter = (id) => {
    // Another course: whatever was selected may not be on its graph.
    if ((id || null) !== courseFilter) window.DeltaKgCore?.deselect?.();
    courseFilter = id || null;
    courseVisible = null;
    persist();
    announceCourse();
    if (cy) applyCourseFilter();
  };

  const readiness = (kc) => {
    if (typeof window.deltaKcReadinessInfo !== "function") return NaN;
    const info = window.deltaKcReadinessInfo(kc);
    return info && Number.isFinite(info.r) ? info.r : NaN;
  };

  /* ---------------- main-instance plumbing ---------------------------- */
  const restoreAll = () => {
    if (!cy) return;
    if (removed && removed.length) removed.restore();
    removed = cy.collection();
  };
  const takeOff = (eles) => {
    if (!eles || !eles.length) return;
    removed = removed.union(cy.remove(eles));
  };
  // Every prerequisite, transitively, of the given ids — plus the ids.
  const closure = (ids) => {
    const out = new Set(ids); const stack = [...ids];
    while (stack.length) {
      (parents[stack.pop()] || []).forEach((p) => { if (!out.has(p)) { out.add(p); stack.push(p); } });
    }
    return out;
  };
  // A removed element keeps its classes and misses every selection change
  // while it is off the canvas, so after any restore the selection chain is
  // repainted from scratch: the same classes lesson-graph.js's selectNode
  // sets (hl-strong on the pick, hl on its ancestors + chain edges, faded on
  // the rest), or none when nothing is selected. `id` is read BEFORE the
  // view takes elements off, so a selected node that is hidden by the view
  // simply leaves no chain behind.
  const selectedId = () => { const n = cy.nodes(".hl-strong"); return n.length ? n[0].id() : null; };
  const reselect = (id) => {
    cy.batch(() => {
      cy.elements().removeClass("faded hl hl-strong");
      if (!id || !cy.getElementById(id).length) return;
      const path = closure([id]);
      cy.nodes().forEach((n) => n.addClass(path.has(n.id()) ? (n.id() === id ? "hl-strong" : "hl") : "faded"));
      cy.edges().forEach((e) => e.addClass(path.has(e.source().id()) && path.has(e.target().id()) ? "hl" : "faded"));
    });
  };

  let mainLay = null;            // the running main layout, stopped before the next
  const layoutMain = (fitEles, opts) => {
    const o = opts || {};
    // A view switch inside the previous layout's 320 ms would otherwise leave
    // its layoutstop fit racing this one around the old subset.
    if (mainLay) { try { mainLay.stop(); } catch (_) {} mainLay = null; }
    cy.stop(true);
    // Without the shortcut links kg-look.js hides: dagre orders each rank to
    // cut crossings, and a hidden edge it still counts only costs it moves.
    // kg-look.js runs dagre itself so the edges can follow the routes it
    // computes (see routeLayout there); cytoscape-dagre is the fallback.
    // Grouped by section: each area is one cluster, so its header
    // (kg-sections.js) sits over one block instead of a scatter.
    const look = window.DeltaKgLook;
    const dagreOpts = {
      name: window.cytoscapeDagre ? "dagre" : "cose",
      rankDir: "BT", nodeSep: 26, rankSep: o.rankSep || 150, edgeSep: 12,
      animate: o.animate !== false, animationDuration: 320, animationEasing: "ease-out",
      fit: false, padding: 40, nodeDimensionsIncludeLabels: true,
      groupBy: (kc) => sectionOf(kc).id,
      // kg-look.js refines the layout off-thread and calls this when the
      // nodes have glided to their final places (not if the learner has
      // panned or zoomed since).
      refit: () => { if (mainLay === lay) fitTo(fitEles, o.pad); },
    };
    const lay = (look && look.routeLayout(cy, dagreOpts)) ||
      (look ? look.layoutEles(cy) : cy.elements()).layout(dagreOpts);
    lay.one("layoutstop", () => { if (mainLay === lay) fitTo(fitEles, o.pad); });
    mainLay = lay;
    lay.run();
  };
  // Fit, but never so close that three bubbles fill the screen.
  const fitTo = (eles, pad) => {
    const target = eles && eles.length ? eles : cy.elements();
    if (!target.length) return;
    cy.animate({ fit: { eles: target, padding: pad == null ? 60 : pad } }, {
      duration: 260,
      complete: () => { if (cy.zoom() > 1.3) cy.animate({ zoom: { level: 1.3, position: target.boundingBox ? bbCenter(target) : undefined } }, { duration: 160 }); },
    });
  };
  const bbCenter = (eles) => {
    const bb = eles.boundingBox();
    return { x: (bb.x1 + bb.x2) / 2, y: (bb.y1 + bb.y2) / 2 };
  };

  /* ---------------- condensed ------------------------------------------ */
  const ensureCondensedContainer = () => {
    let el = $("kg-cy-condensed");
    if (el) return el;
    const graph = document.querySelector(".kg2-graph");
    if (!graph) return null;
    el = document.createElement("div");
    el.id = "kg-cy-condensed";
    el.className = "kgv-condensed";
    el.hidden = true;
    graph.insertBefore(el, $("kg-cy").nextSibling);
    return el;
  };

  // A condensed concept's label sits under its circle on the pane, so it takes
  // the pane's text colour; re-read on a theme switch.
  let kcInk = "#c8cdd8";
  const readKcInk = () => { const g = document.querySelector(".kg2-graph"); if (g) kcInk = getComputedStyle(g).color || kcInk; };
  window.addEventListener("delta:theme-changed", () => { readKcInk(); if (ccy) ccy.style().update(); });
  const kcLook = () => (window.DeltaKgLook ? window.DeltaKgLook.node(0.9, () => kcInk) : {
    "shape": "round-rectangle", "label": "data(label)", "width": "label", "height": "label", "padding": "10px",
    "text-wrap": "wrap", "text-max-width": "110px", "text-valign": "center", "text-halign": "center",
    "font-size": 11.5, "font-weight": 600, "color": "#15151f",
  });
  const edgeInk = () => (window.DeltaKgLook ? window.DeltaKgLook.edgeInk() : "#8f9bb8");
  // The condensed canvas's stylesheet. A function: the concept rule follows
  // the node look (kg-look.js), and a look switch swaps the whole sheet.
  const condensedSheet = () => [
    { selector: "node.sec", style: {
        "shape": "round-rectangle", "background-color": "data(color)", "background-opacity": 0.95,
        "label": "data(label)", "text-wrap": "wrap", "text-max-width": "190px", "text-valign": "center",
        "text-halign": "center", "font-size": 13, "font-weight": 700, "line-height": 1.35, "color": "#15151f",
        "width": "label", "height": "label", "padding": "22px", "border-width": 2,
        "border-color": "rgba(21,21,31,0.35)",
    }},
    { selector: "node.sec-open", style: {
        "shape": "round-rectangle", "background-color": "data(color)", "background-opacity": 0.16,
        "border-width": 2, "border-color": "data(color)", "label": "data(label)", "text-valign": "top",
        "text-halign": "center", "text-margin-y": -8, "font-size": 13, "font-weight": 700, "color": "#15151f",
        "padding": "24px",
    }},
    // Concepts inside an opened section: the main canvas's node look
    // (kg-look.js — labelled box or dot), a touch smaller.
    { selector: "node.kc", style: Object.assign(kcLook(), {
        "background-color": "data(color)", "border-width": 1.5, "border-color": "rgba(21,21,31,0.35)",
    })},
    { selector: "node.kc[!measured]", style: { "background-opacity": 0.45, "border-style": "dashed" } },
    { selector: "edge", style: {
        "curve-style": "bezier", "target-arrow-shape": "triangle", "line-color": () => edgeInk(),
        "target-arrow-color": () => edgeInk(), "width": 1.4, "arrow-scale": 0.9, "opacity": 0.7,
    }},
    { selector: "edge.agg", style: {
        "width": (e) => Math.min(9, 1.5 + e.data("count") * 0.7), "label": "data(label)",
        "font-size": 12, "font-weight": 700, "color": "#15151f", "text-background-color": "#fff",
        "text-background-opacity": 0.9, "text-background-padding": "3px", "text-background-shape": "round-rectangle",
        "opacity": 0.8,
    }},
    { selector: "node:active, node.sec:selected", style: { "overlay-opacity": 0.08 } },
  ];
  const kcColor = (kc) => {
    if (typeof window.deltaKcMasteryColor === "function") return window.deltaKcMasteryColor(readiness(kc));
    return "#9aa3b2";
  };
  const kcMeasured = (kc) => typeof window.deltaKcIsMeasured === "function" ? !!window.deltaKcIsMeasured(kc) : false;

  const condensedElements = () => {
    const secs = {};            // id -> { meta, kcs: [] }
    allKcs.forEach((kc) => {
      if (courseHidden(kc)) return;
      const s = sectionOf(kc);
      if (!secs[s.id]) secs[s.id] = { meta: s, kcs: [] };
      secs[s.id].kcs.push(kc);
    });
    const secOf = (kc) => { const s = sectionOf(kc); return secs[s.id] ? s.id : null; };
    const els = [];
    const internal = {};
    Object.keys(secs).forEach((sid) => { internal[sid] = 0; });
    // Count internal links first so a closed section can say how dense it is.
    allKcs.forEach((kc) => (parents[kc] || []).forEach((p) => {
      const a = secOf(p), b = secOf(kc);
      if (a && b && a === b) internal[a] += 1;
    }));
    Object.keys(secs).forEach((sid) => {
      const { meta, kcs } = secs[sid];
      const open = openSections.has(sid);
      els.push({ data: {
        id: "sec:" + sid, kind: open ? "section-open" : "section", sid,
        label: open ? `${meta.label} — tap the frame to close`
                    : `${meta.label}\n${kcs.length} concept${kcs.length === 1 ? "" : "s"} · ${internal[sid]} link${internal[sid] === 1 ? "" : "s"} inside`,
        color: meta.color, count: kcs.length,
      }, classes: open ? "sec-open" : "sec" });
      if (open) kcs.forEach((kc) => els.push({ data: {
        id: kc, kind: "kc", parent: "sec:" + sid, label: nodeLabel[kc] || kc,
        color: kcColor(kc), measured: kcMeasured(kc),
      }, classes: "kc" }));
    });
    // Edges: concept→concept inside an open section, and between two open
    // sections; everything else stays ONE counted edge between the section
    // nodes (an open section's frame is a node too). Fanning 26 links from a
    // closed section onto 26 children made the cluster unreadable.
    const agg = {};
    allKcs.forEach((kc) => (parents[kc] || []).forEach((p) => {
      const a = secOf(p), b = secOf(kc);
      if (!a || !b) return;
      const detail = openSections.has(a) && openSections.has(b);
      const src = detail ? p : "sec:" + a;
      const tgt = detail ? kc : "sec:" + b;
      if (src === tgt) return;
      const key = src + "→" + tgt;
      if (!agg[key]) agg[key] = { source: src, target: tgt, count: 0, kind: (src === p && tgt === kc) ? "link" : "agg" };
      agg[key].count += 1;
    }));
    Object.keys(agg).forEach((key, i) => {
      const e = agg[key];
      els.push({ data: { id: "ce" + i, source: e.source, target: e.target, count: e.count, kind: e.kind,
                         label: e.kind === "agg" ? String(e.count) : "" },
                 classes: e.kind });
    });
    return { els };
  };

  // dagre throughout. The vendored cytoscape-dagre (2.5.0) builds its graph
  // `compound: true` and `setParent`s children, so an opened section lays out
  // as a cluster inside the same bottom-up ranking as the closed ones.
  // 🔴 NOT fcose: with a compound parent + relativePlacementConstraint it
  // spun the page's main thread forever (2026-09-12) — the tab froze hard
  // enough that CDP could no longer attach to the browser.
  const condensedLayout = () => {
    const lay = ccy.layout({ name: window.cytoscapeDagre ? "dagre" : "cose", rankDir: "BT",
      nodeSep: 40, rankSep: 110, edgeSep: 16, animate: true, animationDuration: 320, fit: false, padding: 50 });
    // An opened section is what the learner asked to look at: fit to it (and
    // any other open one), not to the whole map around it.
    lay.one("layoutstop", () => {
      const open = ccy.nodes(":parent");
      ccy.animate({ fit: { eles: open.length ? open : ccy.elements(), padding: open.length ? 40 : 50 } }, { duration: 240 });
    });
    return lay;
  };

  const buildCondensed = () => {
    const el = ensureCondensedContainer();
    if (!el || typeof cytoscape === "undefined") return;
    el.hidden = false;
    readKcInk();
    const { els } = condensedElements();
    if (ccy) {
      // Nothing may still be animating what is about to be removed.
      if (condensedLay) { try { condensedLay.stop(); } catch (_) {} condensedLay = null; }
      ccy.stop(true); ccy.elements().stop(true);
      ccy.elements().remove(); ccy.add(els);
    }
    else {
      ccy = cytoscape({
        container: el, elements: els, wheelSensitivity: 0.25, minZoom: 0.1, maxZoom: 3,
        style: condensedSheet(),
        layout: { name: "preset" },
      });
      ccy.on("tap", "node.sec, node.sec-open", (evt) => {
        // A tap on a concept INSIDE an open section bubbles up to the section
        // node; that tap is the concept's, not a request to close the frame.
        if (evt.target.data("kind") === "kc") return;
        const sid = evt.target.data("sid");
        if (openSections.has(sid)) openSections.delete(sid); else openSections.add(sid);
        // Off the tap tick: rebuilding INSIDE the handler removes the very
        // node Cytoscape is still finishing the tap on (style hints on a
        // removed element → "reading 'index'").
        setTimeout(buildCondensed, 0);
      });
      ccy.on("tap", "node.kc", (evt) => {
        // Straight to lesson-graph.js's own focus, NOT the wrapper below: the
        // learner stays in the condensed view and gets the side panel for
        // that concept.
        evt.stopPropagation();
        const kc = evt.target.id();
        const f = origFocus || window.deltaFocusConceptGraphKc;
        if (typeof f === "function") f(kc);
      });
      window.addEventListener("resize", () => { if (ccy && mode === "condensed") ccy.resize(); });
    }
    ccy.resize();
    condensedLay = condensedLayout();
    condensedLay.run();
    const open = openSections.size;
    setHint(open
      ? "Tap an open section to close it, a concept to read its lesson."
      : "One bubble per section; edge numbers are prerequisite links between them. Tap a section to open it.");
  };

  const showCondensed = (on) => {
    const el = ensureCondensedContainer();
    const main = $("kg-cy");
    if (el) el.hidden = !on;
    if (main) main.classList.toggle("kgv-behind", on);
    if (!on && cy) cy.resize();
  };

  /* ---------------- apply ---------------------------------------------- */
  const applyView = () => {
    if (!cy || applying) return;
    applying = true;
    try {
      restoreAll();
      const selId = selectedId();
      takeOff(cy.nodes().filter((n) => courseHidden(n.id())));
      // By id, not under `panel`: kg-toolbar.js moves the segment into its bar.
      document.querySelectorAll("#kg-view-seg [data-view]").forEach((b) => {
        const on = b.dataset.view === mode;
        b.classList.toggle("active", on);
        b.setAttribute("aria-pressed", on ? "true" : "false");
      });
      const graphEl = document.querySelector(".kg2-graph");
      if (graphEl) graphEl.dataset.kgView = mode;
      if (mode === "condensed") {
        showCondensed(true);
        buildCondensed();
      } else {
        showCondensed(false);
        layoutMain(cy.nodes(), { rankSep: 150, pad: 36 });
        setHint("");
        reselect(selId);
      }
      persist();
      window.dispatchEvent(new CustomEvent("delta:kg-view-changed", { detail: { mode } }));
    } finally { applying = false; }
  };

  /* ---------------- the view segment ---------------------------------- */
  // Built here, moved into the bar by kg-toolbar.js (#kg-view-seg,
  // #kg-view-hint are looked up by id).
  const setHint = (html) => { if (hintEl) hintEl.innerHTML = html; };

  const buildPanel = () => {
    const graph = document.querySelector(".kg2-graph");
    if (!graph || $("kg-view-panel")) { panel = $("kg-view-panel"); return; }
    panel = document.createElement("div");
    panel.id = "kg-view-panel";
    panel.className = "kgv-panel";
    panel.setAttribute("role", "group");
    panel.setAttribute("aria-label", "Graph view");
    panel.innerHTML =
      '<div class="kgv-seg" id="kg-view-seg">' +
        '<button type="button" data-view="complete" title="Every concept, grouped by area">Complete</button>' +
        '<button type="button" data-view="condensed" title="One bubble per area">Condensed</button>' +
      "</div>" +
      '<div class="kgv-hint" id="kg-view-hint"></div>';
    graph.appendChild(panel);
    graph.classList.add("has-kgv");
    hintEl = $("kg-view-hint");
    panel.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => {
      if (mode === b.dataset.view) return;
      mode = b.dataset.view;
      applyView();
    }));
  };

  /* ---------------- wiring --------------------------------------------- */
  const snapshotGraph = () => {
    allKcs = [];
    cy.nodes().forEach((n) => {
      const id = n.id();
      allKcs.push(id);
      nodeLesson[id] = n.data("lesson");
      nodeLabel[id] = n.data("label");
      parents[id] = parents[id] || [];
      children[id] = children[id] || [];
    });
    cy.edges().forEach((e) => {
      const s = e.data("source"), t = e.data("target");
      (parents[t] = parents[t] || []).push(s);
      (children[s] = children[s] || []).push(t);
    });
  };

  const init = () => {
    cy = typeof window.deltaConceptGraphCy === "function" ? window.deltaConceptGraphCy() : null;
    if (!cy || panel) return !!panel;
    removed = cy.collection();
    snapshotGraph();
    buildPanel();
    // Jumping to a concept from the Practice tab must find it on the canvas:
    // out of the condensed overlay, and out of a course scope that hides it.
    const orig = window.deltaFocusConceptGraphKc;
    if (typeof orig === "function") {
      origFocus = orig;
      window.deltaFocusConceptGraphKc = (kc) => {
        if (kc && parents[kc]) {
          let again = false;
          if (courseHidden(kc)) { courseFilter = null; courseVisible = null; announceCourse(); again = true; }
          if (mode === "condensed") { mode = "complete"; again = true; }
          if (again) applyView();
        }
        return orig(kc);
      };
    }
    // Section labels/colours come from lesson-graph.js when it exports them;
    // otherwise the map has to be read before the grouping is right.
    loadArenaMap().then(() => { if (typeof window.deltaKcSection !== "function") applyView(); });
    if (courseFilter) applyCourseFilter(); else applyView();
    announceCourse();
    return true;
  };

  const boot = () => {
    paintCourseTitle();
    if (init()) return;
    window.addEventListener("delta:practice-target-graph-ready", () => init(), { once: true });
    let tries = 0;
    const tick = () => { if (!init() && tries++ < 200) setTimeout(tick, 250); };
    tick();
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  window.deltaKgView = {
    mode: () => mode,
    set: (m) => { if (MODES.includes(m)) { mode = m; applyView(); } },
    // The course scope (a course-registry id, or null for every concept).
    course: () => courseFilter,
    // Same view, laid out again. A node-look switch (lesson-graph.js calls
    // this on `delta:kg-look-changed`): the condensed sheet is replaced
    // whole, since patching node.kc would leave the old look's properties
    // under the new rule, then the current view is laid out again.
    relayout: () => { if (ccy) ccy.style(condensedSheet()); applyView(); },
    condensed: () => ccy,
    // Every area in the course scope, in chapter order, with its concepts:
    // [{id, label, color, kcs:[…]}]. kg-toolbar.js's Filter lists these;
    // kg-sections.js heads them.
    sections: () => {
      const by = {};
      allKcs.forEach((kc) => {
        if (courseHidden(kc)) return;
        const s = sectionOf(kc);
        if (!by[s.id]) by[s.id] = { id: s.id, label: s.label, color: s.color, order: sectionOrder(s), kcs: [] };
        by[s.id].kcs.push(kc);
      });
      return Object.values(by).sort((a, b) => a.order - b.order);
    },
  };
})();
