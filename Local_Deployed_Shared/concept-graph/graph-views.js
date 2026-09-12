/* concept-graph/graph-views.js — how much of the Knowledge Graph a learner sees.
 *
 * lesson-graph.js draws every concept, every time. For a learner that is the
 * wrong first picture: 64 bubbles with three of them relevant today. This file
 * layers THREE views over that one graph and a settings card in the
 * bottom-left corner of the canvas to switch between them:
 *
 *   adaptive  — start zoomed in on the learner's KNOWLEDGE FRONTIER: the
 *               concepts the tutor can serve now (unlocked, not yet learned).
 *               Only those bubbles are on the canvas. Tapping one reveals its
 *               direct prerequisites BELOW it (fan-in, blue) and the concepts
 *               it unlocks ABOVE it (fan-out, orange); tapping a revealed node
 *               reveals its neighbours in turn, so the map grows outward from
 *               what the learner is actually studying.
 *   condensed — one bubble per ARENA section (−1.0 Python, −1.1 arrays,
 *               0.0, 0.1, …) with the number of prerequisite links between
 *               sections on the edges. Tapping a section opens it into its
 *               concepts, in place, inside a labelled frame.
 *   complete  — the whole graph, exactly as lesson-graph.js has always drawn it.
 *
 * The card also folds out a Chapters list to hide whole sections from all
 * three views.
 *
 * 🔴 The graph stays lesson-graph.js's. Adaptive and complete operate on ITS
 * Cytoscape instance (`window.deltaConceptGraphCy()`) under the same contract
 * instructor-graph-edit.js honours: every hidden element is one `cy.remove`
 * whose collection is kept, and everything is `.restore()`d before the next
 * view or a mode switch — a remove is the only way to make dagre lay out the
 * SUBSET, and a bypass style on a node's border would outrank the gate-state
 * classes (locked / frontier / next-up) that file paints. Fan colours are
 * therefore `underlay-*` on nodes (a halo, not the border) and a bypass on the
 * incident EDGES only, and both are cleared when the view resets.
 *
 * Condensed is a SECOND, private Cytoscape instance in an overlay over the
 * same box. Section nodes have no lesson, no lattice row and no learner
 * model, and lesson-graph.js's tap handler would try to render one; giving
 * them their own canvas keeps that file's assumptions true (every node id is
 * a KC). Tapping a concept inside an opened section hands off to
 * `window.deltaFocusConceptGraphKc`, so the lesson pane and the learner-model
 * dock — both lesson-graph.js's — light up exactly as they do from the map.
 *
 * Section membership is asked of lesson-graph.js when it exports it
 * (`window.deltaKcSection`); until then the same rule is mirrored from
 * `lessons/arena_exercise_kcs.json` — see `_sectionFallback`. */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const VIEW_KEY = "dd_kg_view";
  const CHAPTERS_OFF_KEY = "dd_kg_chapters_off";
  const CHAPTERS_OPEN_KEY = "dd_kg_chapters_open";
  const MODES = ["adaptive", "condensed", "complete"];
  // Mirrors of lesson-graph.js's gates, used ONLY by the guest/offline mirror
  // of the frontier — a signed-in learner gets the server's `state`.
  const UNLOCK_T = 0.85, LEARNED_T = 0.95;
  // Fan colours. Blue = what this needs (fan-in, drawn below); orange = what
  // this unlocks (fan-out, drawn above). Neither is the gold lesson-graph.js
  // uses for the selected chain nor the red of a plain prerequisite arrow.
  const FAN_IN = "#3d8bfd", FAN_OUT = "#f28c28";

  /* ---------------- section taxonomy ---------------------------------- */
  // Prep tiers are ours; ARENA's sections come from the exercise map's slugs.
  // Labels here are the FALLBACK only — the real labels and colours are
  // lesson-graph.js's ARENA_SECTIONS, read through `window.deltaKcSection`
  // when that export exists.
  const PREP_PYTHON = { id: "sm10", label: "Section −1.0 — Python", color: "#dfae74", order: 0 };
  const PREP_ARRAYS = { id: "sm11", label: "Section −1.1 — arrays, einops, tensors", color: "#b0b4c0", order: 1 };
  const LATER = { id: "sNN", label: "Later ARENA sections", color: "#e0709a", order: 99 };
  const SLUG_COLORS = { "0-0": "#4f9fe0", "0-1": "#bb7de8", "0-2": "#e8a765" };
  const _slugSection = (slug) => {
    const m = /^(\d+)-(\d+)$/.exec(slug);
    if (!m) return LATER;
    return { id: "s" + m[1] + m[2], label: `Section ${m[1]}.${m[2]}`, color: SLUG_COLORS[slug] || LATER.color,
             order: 10 + Number(m[1]) * 10 + Number(m[2]) };
  };

  let arenaSlugByKc = {};        // kc -> notebook slug, earliest wins
  let arenaMapLoaded = false;
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
          arenaMapLoaded = true;
        }
      })
      .catch(() => {});

  const _sectionFallback = (kc, lessonId) => {
    const slug = arenaSlugByKc[kc];
    if (slug) return _slugSection(slug);
    if (String(kc).startsWith("python.") || /^py-/.test(lessonId || "")) return PREP_PYTHON;
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
  let mode = "adaptive";
  let chaptersOff = new Set();
  let expanded = new Set();      // adaptive: KCs whose neighbourhood is revealed
  let openSections = new Set();  // condensed: sections opened into concepts
  let removed = null;            // cy collection of everything we took off the canvas
  let fanStyled = null;          // cy collection carrying our fan bypass styles
  const parents = {}, children = {}, nodeLesson = {}, nodeLabel = {};
  let allKcs = [];
  let panel = null, hintEl = null, chaptersEl = null;
  let origFocus = null;          // lesson-graph.js's deltaFocusConceptGraphKc, pre-wrap
  let applying = false;

  try { const m = localStorage.getItem(VIEW_KEY); if (MODES.includes(m)) mode = m; } catch (_) {}
  try {
    const off = JSON.parse(localStorage.getItem(CHAPTERS_OFF_KEY) || "[]");
    if (Array.isArray(off)) chaptersOff = new Set(off.filter((x) => typeof x === "string"));
  } catch (_) {}

  const persist = () => {
    try {
      localStorage.setItem(VIEW_KEY, mode);
      localStorage.setItem(CHAPTERS_OFF_KEY, JSON.stringify([...chaptersOff]));
    } catch (_) {}
  };

  /* ---------------- the learner's frontier ---------------------------- */
  const readiness = (kc) => {
    if (typeof window.deltaKcReadinessInfo !== "function") return NaN;
    const info = window.deltaKcReadinessInfo(kc);
    return info && Number.isFinite(info.r) ? info.r : NaN;
  };
  // Server truth first (`state` from kc_graph.kc_report — the same gate the
  // practice queue uses); the browser mirror only for a guest with no report.
  // `frontierDone` is set when the learner has evidence and NOTHING is left
  // on the frontier (everything unlocked is learned) — the view then shows
  // the summit (concepts nothing depends on) and the hint says why, instead
  // of relabelling mastered roots as "practise now". The roots are only the
  // answer when there is no evidence at all.
  let frontierDone = false;
  const frontierSet = () => {
    const lattice = typeof window.getKcLattice === "function" ? window.getKcLattice() : null;
    const out = new Set();
    let learnedAny = false;
    if (lattice && lattice.kcs && Object.keys(lattice.kcs).length) {
      allKcs.forEach((kc) => {
        const row = lattice.kcs[kc];
        if (!row) return;
        if (row.state === "frontier") out.add(kc);
        if (row.state === "learned") learnedAny = true;
      });
      if (!out.size && lattice.next_kc && parents[lattice.next_kc]) out.add(lattice.next_kc);
    } else {
      allKcs.forEach((kc) => {
        const r = readiness(kc);
        if (Number.isFinite(r) && r >= LEARNED_T) { learnedAny = true; return; }
        const locked = (parents[kc] || []).some((p) => {
          const pr = readiness(p);
          return !(Number.isFinite(pr) && pr >= UNLOCK_T);
        });
        if (!locked) out.add(kc);
      });
    }
    frontierDone = !out.size && learnedAny;
    if (!out.size) {
      allKcs.forEach((kc) => {
        const edges = frontierDone ? children[kc] : parents[kc];
        if (!(edges || []).length) out.add(kc);
      });
    }
    return out;
  };

  /* ---------------- main-instance plumbing ---------------------------- */
  // Only OUR bypass properties come off: lesson-graph.js's recolor() sets
  // background-color / background-opacity / border-style as bypasses too,
  // and a bare removeStyle() would strip an inferred node's dashed border
  // and a disabled node's grey along with the halo.
  const FAN_PROPS = "line-color target-arrow-color width opacity underlay-color underlay-opacity underlay-padding";
  const clearFans = () => {
    if (fanStyled && fanStyled.length) fanStyled.forEach((e) => e.removeStyle(FAN_PROPS));
    fanStyled = null;
  };
  const restoreAll = () => {
    if (!cy) return;
    clearFans();
    if (removed && removed.length) removed.restore();
    removed = cy.collection();
  };
  const takeOff = (eles) => {
    if (!eles || !eles.length) return;
    removed = removed.union(cy.remove(eles));
  };
  // Bring nodes back, then every edge whose endpoints are both on the canvas.
  const bringBack = (ids) => {
    const want = new Set(ids);
    const nodes = removed.filter((e) => e.isNode() && want.has(e.id()));
    if (nodes.length) { nodes.restore(); removed = removed.difference(nodes); }
    const edges = removed.filter((e) => e.isEdge() &&
      cy.getElementById(e.data("source")).length && cy.getElementById(e.data("target")).length);
    if (edges.length) { edges.restore(); removed = removed.difference(edges); }
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
      const path = new Set([id]); const stack = [id];
      while (stack.length) {
        (parents[stack.pop()] || []).forEach((p) => { if (!path.has(p)) { path.add(p); stack.push(p); } });
      }
      cy.nodes().forEach((n) => n.addClass(path.has(n.id()) ? (n.id() === id ? "hl-strong" : "hl") : "faded"));
      cy.edges().forEach((e) => e.addClass(path.has(e.source().id()) && path.has(e.target().id()) ? "hl" : "faded"));
    });
  };
  const chapterHidden = (kc) => chaptersOff.has(sectionOf(kc).id);

  let mainLay = null;            // the running main layout, stopped before the next
  const layoutMain = (fitEles, opts) => {
    const o = opts || {};
    // A view switch inside the previous layout's 320 ms would otherwise leave
    // its layoutstop fit racing this one around the old subset.
    if (mainLay) { try { mainLay.stop(); } catch (_) {} mainLay = null; }
    cy.stop(true);
    const lay = cy.layout({
      name: window.cytoscapeDagre ? "dagre" : "cose",
      rankDir: "BT", nodeSep: 26, rankSep: o.rankSep || 150, edgeSep: 12,
      animate: o.animate !== false, animationDuration: 320, animationEasing: "ease-out",
      fit: false, padding: 40,
    });
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

  /* ---------------- adaptive ------------------------------------------ */
  const adaptiveVisible = () => {
    const vis = new Set(frontierSet());
    expanded.forEach((kc) => {
      vis.add(kc);
      (parents[kc] || []).forEach((p) => vis.add(p));
      (children[kc] || []).forEach((c) => vis.add(c));
    });
    return new Set([...vis].filter((kc) => !chapterHidden(kc)));
  };

  const paintFans = () => {
    clearFans();
    fanStyled = cy.collection();
    expanded.forEach((kc) => {
      const node = cy.getElementById(kc);
      if (!node.length) return;
      node.incomers("edge").forEach((e) => {
        e.style({ "line-color": FAN_IN, "target-arrow-color": FAN_IN, "width": 3, "opacity": 1 });
        fanStyled = fanStyled.union(e);
        const src = e.source();
        if (!expanded.has(src.id())) {
          src.style({ "underlay-color": FAN_IN, "underlay-opacity": 0.28, "underlay-padding": 7 });
          fanStyled = fanStyled.union(src);
        }
      });
      node.outgoers("edge").forEach((e) => {
        e.style({ "line-color": FAN_OUT, "target-arrow-color": FAN_OUT, "width": 3, "opacity": 1 });
        fanStyled = fanStyled.union(e);
        const tgt = e.target();
        if (!expanded.has(tgt.id())) {
          tgt.style({ "underlay-color": FAN_OUT, "underlay-opacity": 0.28, "underlay-padding": 7 });
          fanStyled = fanStyled.union(tgt);
        }
      });
    });
    // lesson-graph.js fades everything off the selected node's prerequisite
    // chain; the fan is the point of this view, so it stays lit.
    fanStyled.removeClass("faded");
  };

  const applyAdaptive = (focusKc) => {
    const vis = adaptiveVisible();
    // Frontier alone and nothing to show — fall back to the whole map rather
    // than an empty canvas, and say so.
    const hidden = cy.nodes().filter((n) => !vis.has(n.id()));
    takeOff(hidden);
    bringBack([...vis]);
    const frontier = frontierSet();
    const fitEles = focusKc && cy.getElementById(focusKc).length
      ? cy.getElementById(focusKc).closedNeighborhood()
      : cy.nodes();
    layoutMain(fitEles, { rankSep: 110, pad: focusKc ? 80 : 60 });
    const n = [...frontier].filter((kc) => !chapterHidden(kc)).length;
    const tapHint = `Tap one to reveal what it <span class="kgv-in">needs</span> (below) and what it <span class="kgv-out">unlocks</span> (above).`;
    setHint(frontierDone
      ? `Nothing left on your frontier — everything unlocked is learned. Showing the summit. ${tapHint}`
      : n
        ? `Your frontier: <b>${n}</b> concept${n === 1 ? "" : "s"} you can practise now. ${tapHint}`
        : "Nothing on your frontier in the chapters shown.");
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

  const kcColor = (kc) => {
    if (typeof window.deltaKcMasteryColor === "function") return window.deltaKcMasteryColor(readiness(kc));
    return "#9aa3b2";
  };
  const kcMeasured = (kc) => typeof window.deltaKcIsMeasured === "function" ? !!window.deltaKcIsMeasured(kc) : false;

  const condensedElements = () => {
    const secs = {};            // id -> { meta, kcs: [] }
    allKcs.forEach((kc) => {
      const s = sectionOf(kc);
      if (chaptersOff.has(s.id)) return;
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
        style: [
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
          { selector: "node.kc", style: {
              "shape": "round-rectangle", "background-color": "data(color)", "label": "data(label)",
              "width": "label", "height": "label", "padding": "10px", "text-wrap": "wrap", "text-max-width": "110px",
              "text-valign": "center", "text-halign": "center", "font-size": 11.5, "font-weight": 600, "color": "#15151f",
              "border-width": 1.5, "border-color": "rgba(21,21,31,0.35)",
          }},
          { selector: "node.kc[!measured]", style: { "background-opacity": 0.45, "border-style": "dashed" } },
          { selector: "edge", style: {
              "curve-style": "bezier", "target-arrow-shape": "triangle", "line-color": "#e3212c",
              "target-arrow-color": "#e3212c", "width": 1.6, "arrow-scale": 1.1,
          }},
          { selector: "edge.agg", style: {
              "width": (e) => Math.min(9, 1.5 + e.data("count") * 0.7), "label": "data(label)",
              "font-size": 12, "font-weight": 700, "color": "#15151f", "text-background-color": "#fff",
              "text-background-opacity": 0.9, "text-background-padding": "3px", "text-background-shape": "round-rectangle",
              "line-color": "#c53a42", "target-arrow-color": "#c53a42",
          }},
          { selector: "node:active, node.sec:selected", style: { "overlay-opacity": 0.08 } },
        ],
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
        // learner stays in the condensed view and gets the lesson pane and
        // the learner-model dock for that concept.
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
  const applyView = (opts) => {
    if (!cy || applying) return;
    applying = true;
    try {
      const o = opts || {};
      restoreAll();
      const selId = selectedId();
      // Chapter filter applies to every view.
      takeOff(cy.nodes().filter((n) => chapterHidden(n.id())));
      panel.querySelectorAll("[data-view]").forEach((b) => {
        const on = b.dataset.view === mode;
        b.classList.toggle("active", on);
        b.setAttribute("aria-pressed", on ? "true" : "false");
      });
      panel.dataset.view = mode;
      if (mode === "condensed") {
        showCondensed(true);
        buildCondensed();
      } else {
        showCondensed(false);
        if (mode === "adaptive") applyAdaptive(o.focus);
        else {
          layoutMain(cy.nodes(), { rankSep: 150, pad: 36 });
          setHint("Everything. Tap a concept for its prerequisite chain.");
        }
        reselect(selId);
        if (mode === "adaptive") paintFans();   // after reselect: the fan stays lit
      }
      persist();
    } finally { applying = false; }
  };

  /* ---------------- the settings card --------------------------------- */
  const setHint = (html) => { if (hintEl) hintEl.innerHTML = html; };

  const buildChapters = () => {
    if (!chaptersEl) return;
    const seen = {};
    allKcs.forEach((kc) => { const s = sectionOf(kc); if (!seen[s.id]) seen[s.id] = { meta: s, n: 0 }; seen[s.id].n += 1; });
    const list = Object.values(seen).sort((a, b) => sectionOrder(a.meta) - sectionOrder(b.meta));
    chaptersEl.innerHTML = list.map(({ meta, n }) =>
      `<label class="kgv-chapter"><input type="checkbox" data-sid="${meta.id}"${chaptersOff.has(meta.id) ? "" : " checked"}>` +
      `<span class="kgv-swatch" style="background:${meta.color}"></span>` +
      `<span class="kgv-chapter-label">${esc(meta.label)}</span><span class="kgv-chapter-n">${n}</span></label>`).join("") +
      (arenaMapLoaded || typeof window.deltaKcSection === "function" ? "" :
        '<div class="kgv-warn">Section map unavailable — every concept reads as prep.</div>');
    chaptersEl.querySelectorAll("input[data-sid]").forEach((cb) => cb.addEventListener("change", () => {
      if (cb.checked) chaptersOff.delete(cb.dataset.sid); else chaptersOff.add(cb.dataset.sid);
      applyView();
    }));
  };

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const buildPanel = () => {
    const graph = document.querySelector(".kg2-graph");
    if (!graph || $("kg-view-panel")) { panel = $("kg-view-panel"); return; }
    panel = document.createElement("div");
    panel.id = "kg-view-panel";
    panel.className = "kgv-panel";
    panel.setAttribute("role", "group");
    panel.setAttribute("aria-label", "Graph view");
    let chaptersOpen = false;
    try { chaptersOpen = localStorage.getItem(CHAPTERS_OPEN_KEY) === "1"; } catch (_) {}
    panel.innerHTML =
      '<div class="kgv-head"><span class="kgv-title">View</span>' +
        '<button type="button" class="kgv-reset" id="kg-view-reset" title="Collapse what you expanded">Reset</button></div>' +
      '<div class="kgv-seg" id="kg-view-seg">' +
        '<button type="button" data-view="adaptive" title="Only your knowledge frontier; tap to expand">Adaptive</button>' +
        '<button type="button" data-view="condensed" title="One bubble per section">Condensed</button>' +
        '<button type="button" data-view="complete" title="Every concept">Complete</button>' +
      "</div>" +
      '<div class="kgv-hint" id="kg-view-hint"></div>' +
      '<details class="kgv-chapters"' + (chaptersOpen ? " open" : "") + '><summary>Chapters</summary>' +
        '<div class="kgv-chapter-list" id="kg-view-chapters"></div></details>';
    graph.appendChild(panel);
    graph.classList.add("has-kgv");
    hintEl = $("kg-view-hint");
    chaptersEl = $("kg-view-chapters");
    panel.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => {
      if (mode === b.dataset.view) return;
      mode = b.dataset.view;
      applyView();
    }));
    $("kg-view-reset").addEventListener("click", () => {
      expanded.clear(); openSections.clear();
      applyView();
    });
    const det = panel.querySelector("details.kgv-chapters");
    det.addEventListener("toggle", () => { try { localStorage.setItem(CHAPTERS_OPEN_KEY, det.open ? "1" : "0"); } catch (_) {} });
    buildChapters();
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

  const onMainTap = (evt) => {
    if (mode !== "adaptive") return;
    const kc = evt.target.id();
    if (expanded.has(kc)) {
      // Already open: nothing new to reveal, but lesson-graph.js just faded
      // everything off the chain — relight the fans.
      paintFans();
      return;
    }
    expanded.add(kc);
    applyView({ focus: kc });
  };

  const init = () => {
    cy = typeof window.deltaConceptGraphCy === "function" ? window.deltaConceptGraphCy() : null;
    if (!cy || panel) return !!panel;
    removed = cy.collection();
    snapshotGraph();
    buildPanel();
    cy.on("tap", "node", onMainTap);
    // A graded attempt can move the frontier; recolor() is when the numbers
    // settle. Re-apply only if the visible set actually changed — a layout on
    // every repaint would make the map jump under the learner.
    let lastVis = "";
    window.addEventListener("delta:kc-readiness-changed", () => {
      if (mode !== "adaptive" || applying) return;
      const vis = [...adaptiveVisible()].sort().join("|");
      if (vis === lastVis) return;
      lastVis = vis;
      applyView();
    });
    lastVis = [...adaptiveVisible()].sort().join("|");
    // Jumping to a concept from the Practice tab must find it on the canvas.
    const orig = window.deltaFocusConceptGraphKc;
    if (typeof orig === "function") {
      origFocus = orig;
      window.deltaFocusConceptGraphKc = (kc) => {
        if (kc && parents[kc]) {
          if (mode === "condensed") {
            // An outside jump (Practice's "See in knowledge graph") lands on
            // the real map, opened around that concept.
            mode = "adaptive"; expanded.add(kc); applyView({ focus: kc });
          } else if (mode === "adaptive" && !cy.getElementById(kc).length) {
            expanded.add(kc); applyView({ focus: kc });
          }
        }
        const r = orig(kc);
        // orig's selectNode fades everything off the prerequisite chain, and
        // it ran AFTER paintFans here (the tap path runs them the other way).
        if (mode === "adaptive") paintFans();
        return r;
      };
    }
    // Section labels/colours come from lesson-graph.js when it exports them;
    // otherwise the map has to be read before the chapter list is right.
    loadArenaMap().then(() => { buildChapters(); if (mode !== "complete") applyView(); });
    applyView();
    return true;
  };

  const boot = () => {
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
    reset: () => { expanded.clear(); openSections.clear(); applyView(); },
    frontier: () => [...frontierSet()],
    condensed: () => ccy,
  };
})();
