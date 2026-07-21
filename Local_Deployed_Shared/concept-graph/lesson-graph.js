/* ================================================================
   LESSON KNOWLEDGE GRAPH — interactive Cytoscape.js embed for the
   Knowledge Graph tab.

   Built from scratch over the EXISTING lesson content:
     - lessons/kc_registry.json    → 64 knowledge components (KCs) + their
                                      `prereqs` (the edges) + lesson/topic.
     - lessons/lessons_structured.json → per-KC teaching content
                                      (concept / worked example / misconceptions).

   Each bubble is one KC, coloured by lesson, laid out bottom-up so
   prerequisites sit beneath what they unlock. Click a bubble to:
     - light up its full prerequisite chain, and
     - render that KC's learning content in the left pane.
   The left pane's "Practice" (maximize) button hands off to the real
   Delta-Drills practice screen via window.LessonGate.showLesson(kc).

   Superseded concept-graph/graph-viz.js (the old ARENA 205-atom graph),
   which is no longer wired into index.html.

   Built on demand via window.deltaInitConceptGraph() — called by app.js
   switchTab() when the tab opens (Cytoscape can't size a display:none
   container, so we defer until the tab is visible).
   ================================================================ */
(() => {
  "use strict";

  // One pastel per lesson (grouped by topic hue) so black labels stay legible.
  const LESSON_COLORS = {
    "np-1": "#cbe8f7", "np-2": "#aedaf0", "np-3": "#8fcbe9", "np-4": "#72bde1",
    "eo-1": "#cdf2d6", "eo-2": "#a9e6b8", "eo-3": "#88db9c",
    "es-1": "#e6d2ff", "es-2": "#d3b4ff",
  };
  const FALLBACK = "#dddddd";
  const ACCENT = "#ffd23f"; // prerequisite-path highlight

  const $ = (id) => document.getElementById(id);
  const lessonColor = (lid) => LESSON_COLORS[lid] || FALLBACK;

  let cy = null;
  let building = false;
  let kcById = {};        // id -> {id,lesson,topic,title,prereqs}
  let contentByKc = {};   // id -> kp block from lessons_structured
  let lessonMeta = {};    // lesson id -> {topic,title,subtopic_key}
  let parentsOf = {};     // id -> [prereq ids]
  let childrenOf = {};    // id -> [dependent ids]
  let selectedKc = null;

  /* ---------------- tiny markdown renderer ----------------------------- */
  const esc = (v) =>
    String(v ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  const inline = (v) =>
    esc(v)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>");

  const md = (text, { renderCode = true } = {}) => {
    if (!text) return "";
    const lines = String(text).split("\n");
    const out = [];
    let i = 0, list = null, para = [];
    const flushPara = () => { if (para.length) { out.push("<p>" + inline(para.join(" ")) + "</p>"); para = []; } };
    const flushList = () => { if (list) { out.push("</" + list + ">"); list = null; } };
    while (i < lines.length) {
      const line = lines[i];
      const fence = line.match(/^```(.*)$/);
      if (fence) {
        flushPara(); flushList();
        const buf = []; i++;
        while (i < lines.length && !/^```/.test(lines[i])) { buf.push(lines[i]); i++; }
        i++;
        if (renderCode) out.push("<pre><code>" + esc(buf.join("\n")) + "</code></pre>");
        continue;
      }
      const heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) { flushPara(); flushList(); out.push("<h4>" + inline(heading[2]) + "</h4>"); i++; continue; }
      const item = line.match(/^(\s*)([-*]|\d+\.)\s+(.*)$/);
      if (item) {
        flushPara();
        const kind = /\d+\./.test(item[2]) ? "ol" : "ul";
        if (list !== kind) { flushList(); out.push("<" + kind + ">"); list = kind; }
        let t = item[3]; i++;
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*]|\d+\.)\s/.test(lines[i])) { t += " " + lines[i].trim(); i++; }
        out.push("<li>" + inline(t) + "</li>");
        continue;
      }
      if (!line.trim()) { flushPara(); flushList(); i++; continue; }
      para.push(line.trim()); i++;
    }
    flushPara(); flushList();
    return out.join("\n");
  };

  /* ---------------- graph helpers -------------------------------------- */
  const ancestors = (id) => {
    const seen = new Set(), out = [];
    const q = [...(parentsOf[id] || [])];
    q.forEach((p) => seen.add(p));
    while (q.length) {
      const cur = q.shift();
      out.push(cur);
      for (const p of (parentsOf[cur] || [])) if (!seen.has(p)) { seen.add(p); q.push(p); }
    }
    return out;
  };

  const chipLink = (id) => {
    const kc = kcById[id];
    if (!kc) return "";
    return `<button class="kg2-chip" data-goto="${esc(id)}" title="${esc(kc.title)}">
      <span class="kg2-chip-dot" style="background:${lessonColor(kc.lesson)}"></span>${esc(kc.title)}</button>`;
  };

  /* ---------------- left content pane ---------------------------------- */
  const setPlaceholder = () => {
    selectedKc = null;
    const btn = $("kg-maximize");
    if (btn) btn.hidden = true;
    if ($("kg-info-meta")) $("kg-info-meta").innerHTML = "";
    if ($("kg-info-body"))
      $("kg-info-body").innerHTML =
        `<div class="kg2-placeholder"><strong>Click a bubble</strong> to open its lesson here.<br><br>
         You'll see what the skill teaches and its worked example, and the whole
         prerequisite chain lights up on the graph. Use <strong>Practice ⤢</strong>
         to jump into the full practice screen for that skill.</div>`;
  };

  const renderContent = (id) => {
    const kc = kcById[id];
    const kp = contentByKc[id];
    if (!kc || !kp) return;
    selectedKc = id;
    const lm = lessonMeta[kc.lesson] || {};

    const meta = $("kg-info-meta");
    if (meta)
      meta.innerHTML =
        `<span class="kg2-meta-topic"><span class="kg2-chip-dot" style="background:${lessonColor(kc.lesson)}"></span>${esc(kc.topic)}</span>
         <span class="kg2-meta-lesson">${esc(lm.title || kc.lesson)}</span>`;

    const btn = $("kg-maximize");
    if (btn) { btn.hidden = false; btn.dataset.kc = id; }

    const parents = parentsOf[id] || [];
    const kids = childrenOf[id] || [];
    let html = `<h2 class="kg2-title">${esc(kp.title || kc.title)}</h2>`;
    html += `<div class="kg2-concept">${md(kp.concept_markdown)}</div>`;
    if (kp.worked_example_markdown)
      html += `<div class="kg2-worked"><h3>Worked example</h3>${md(kp.worked_example_markdown)}</div>`;
    if (kp.misconceptions_markdown)
      html += `<div class="kg2-watch"><h3>Watch out</h3>${md(kp.misconceptions_markdown)}</div>`;

    html += `<div class="kg2-nav">`;
    html += `<div class="kg2-nav-col"><h4>Prerequisites (${parents.length})</h4>` +
      (parents.length ? parents.map(chipLink).join("") : `<span class="kg2-nav-empty">Foundation skill — none.</span>`) + `</div>`;
    html += `<div class="kg2-nav-col"><h4>Unlocks (${kids.length})</h4>` +
      (kids.length ? kids.map(chipLink).join("") : `<span class="kg2-nav-empty">Nothing downstream yet.</span>`) + `</div>`;
    html += `</div>`;

    const body = $("kg-info-body");
    body.innerHTML = html;
    body.scrollTop = 0;
    body.querySelectorAll("[data-goto]").forEach((b) =>
      b.addEventListener("click", () => selectNode(b.getAttribute("data-goto"))));
  };

  /* ---------------- selection + highlight ------------------------------ */
  const selectNode = (id) => {
    if (!cy) return;
    const node = cy.getElementById(id);
    if (!node || node.empty()) return;
    const path = new Set([id, ...ancestors(id)]);
    cy.batch(() => {
      cy.elements().removeClass("hl hl-strong").addClass("faded");
      cy.nodes().forEach((el) => {
        if (path.has(el.id())) el.removeClass("faded").addClass(el.id() === id ? "hl-strong" : "hl");
      });
      cy.edges().forEach((e) => {
        if (path.has(e.source().id()) && path.has(e.target().id())) e.removeClass("faded").addClass("hl");
      });
    });
    renderContent(id);
  };

  const resetView = () => {
    if (cy) cy.elements().removeClass("faded hl hl-strong");
    setPlaceholder();
  };

  /* ---------------- maximize: focused practice page (own iframe) -------- */
  // Maximize opens the practice view for the KC as its OWN separate page — a
  // full-screen overlay hosting index.html?lesson=<kc>&embed=1 in an iframe
  // (embed=1 hides the app chrome so no tabs show). This is deliberately a
  // duplicate instance, NOT a tab switch — the graph stays live underneath, so
  // Minimize drops the learner right back onto the same node with its lesson.
  let overlay = null;

  const closeMaximize = () => {
    if (!overlay) return;
    overlay.classList.add("hidden");
    document.body.classList.remove("kg-maxi-open");
    const frame = overlay.querySelector("#kg-maxi-frame");
    if (frame) frame.src = "about:blank"; // tear down the embedded app (pyodide/audio)
    // Back to the workflow: the node is still selected and its lesson is still
    // on the left — just re-centre it so focus returns cleanly.
    if (selectedKc && cy) {
      const n = cy.getElementById(selectedKc);
      if (n && !n.empty()) cy.animate({ center: { eles: n } }, { duration: 220 });
    }
  };

  const ensureOverlay = () => {
    if (overlay) return;
    overlay = document.createElement("div");
    overlay.id = "kg-maxi";
    overlay.className = "kg-maxi hidden";
    overlay.innerHTML =
      '<div class="kg-maxi-bar">' +
        '<span class="kg-maxi-title" id="kg-maxi-title"></span>' +
        '<button type="button" class="kg-maxi-min" id="kg-maxi-min" title="Back to the graph">⤡ Minimize</button>' +
      "</div>" +
      '<iframe class="kg-maxi-frame" id="kg-maxi-frame" title="Practice"></iframe>';
    document.body.appendChild(overlay);
    overlay.querySelector("#kg-maxi-min").addEventListener("click", closeMaximize);
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && overlay && !overlay.classList.contains("hidden")) closeMaximize();
    });
  };

  const openMaximize = (kc) => {
    if (!kc) return;
    ensureOverlay();
    const kcObj = kcById[kc];
    const title = overlay.querySelector("#kg-maxi-title");
    if (title) title.textContent = kcObj ? kcObj.title : "Practice";
    const frame = overlay.querySelector("#kg-maxi-frame");
    frame.src = "index.html?lesson=" + encodeURIComponent(kc) + "&embed=1";
    overlay.classList.remove("hidden");
    document.body.classList.add("kg-maxi-open");
  };

  /* ---------------- legend --------------------------------------------- */
  const buildLegend = () => {
    const el = $("kg-legend");
    if (!el) return;
    const seen = [];
    Object.values(kcById).forEach((k) => { if (!seen.includes(k.lesson)) seen.push(k.lesson); });
    seen.sort();
    el.innerHTML = seen.map((lid) => {
      const lm = lessonMeta[lid] || {};
      return `<span class="kg2-li"><span class="kg2-li-dot" style="background:${lessonColor(lid)}"></span>${esc(lm.title || lid)}</span>`;
    }).join("");
  };

  /* ---------------- build ---------------------------------------------- */
  async function build() {
    if (cy || building) return;
    const container = $("kg-cy");
    if (!container || typeof cytoscape === "undefined") return;
    building = true;

    try { if (window.cytoscapeDagre) cytoscape.use(window.cytoscapeDagre); } catch (_) {}

    let registry, structured;
    try {
      [registry, structured] = await Promise.all([
        fetch("lessons/kc_registry.json", { cache: "no-cache" }).then((r) => r.json()),
        fetch("lessons/lessons_structured.json", { cache: "no-cache" }).then((r) => r.json()),
      ]);
    } catch (e) {
      if ($("kg-status")) $("kg-status").textContent = "Couldn't load the lesson graph data.";
      building = false;
      return;
    }

    (registry.lessons || []).forEach((l) => { lessonMeta[l.id] = l; });
    (registry.kcs || []).forEach((k) => {
      kcById[k.id] = k;
      parentsOf[k.id] = [...(k.prereqs || [])];
      childrenOf[k.id] = childrenOf[k.id] || [];
    });
    Object.values(kcById).forEach((k) => {
      (k.prereqs || []).forEach((p) => {
        if (!childrenOf[p]) childrenOf[p] = [];
        childrenOf[p].push(k.id);
      });
    });
    (structured.lessons || []).forEach((l) => l.kps.forEach((kp) => { contentByKc[kp.kc] = kp; }));

    buildLegend();

    const elements = [];
    Object.values(kcById).forEach((k) => {
      elements.push({ data: { id: k.id, label: k.title, lesson: k.lesson } });
    });
    let ei = 0;
    Object.values(kcById).forEach((k) => {
      (k.prereqs || []).forEach((p) => {
        if (kcById[p]) elements.push({ data: { id: "e" + (ei++), source: p, target: k.id } });
      });
    });

    cy = cytoscape({
      container,
      elements,
      wheelSensitivity: 0.25,
      minZoom: 0.1, maxZoom: 3,
      style: [
        { selector: "node", style: {
            "background-color": (n) => lessonColor(n.data("lesson")),
            "shape": "round-rectangle",
            "label": "data(label)",
            "width": "label", "height": "label", "padding": "13px",
            "text-wrap": "wrap", "text-max-width": "120px",
            "text-valign": "center", "text-halign": "center",
            "font-size": 13, "font-weight": 600, "color": "#15151f",
            "border-width": 1, "border-color": "rgba(0,0,0,.28)",
            "transition-property": "opacity, border-width, border-color", "transition-duration": "120ms",
        }},
        { selector: "edge", style: {
            "curve-style": "bezier", "width": 1.8, "line-color": "#e3212c",
            "target-arrow-shape": "triangle", "target-arrow-color": "#e3212c", "arrow-scale": 0.8, "opacity": 0.9,
        }},
        { selector: ".faded", style: { "opacity": 0.1 } },
        { selector: "node.hl", style: { "opacity": 1, "border-width": 3, "border-color": ACCENT, "z-index": 50 } },
        { selector: "node.hl-strong", style: { "opacity": 1, "border-width": 5, "border-color": ACCENT, "font-size": 12, "z-index": 99 } },
        { selector: "edge.hl", style: { "opacity": 1, "width": 3, "line-color": ACCENT, "target-arrow-color": ACCENT, "z-index": 60 } },
      ],
      layout: { name: window.cytoscapeDagre ? "dagre" : "cose",
        rankDir: "BT", nodeSep: 26, rankSep: 150, edgeSep: 12, animate: false, fit: true, padding: 40 },
    });

    if ($("kg-status")) $("kg-status").style.display = "none";
    cy.on("tap", "node", (evt) => selectNode(evt.target.id()));
    cy.on("tap", (evt) => { if (evt.target === cy) resetView(); });

    if ($("kg-fit")) $("kg-fit").onclick = () => cy.fit(undefined, 36);
    const maxBtn = $("kg-maximize");
    if (maxBtn) maxBtn.onclick = () => openMaximize(maxBtn.dataset.kc);

    setPlaceholder();
    building = false;
  }

  window.deltaInitConceptGraph = function () {
    if (cy) { cy.resize(); cy.fit(undefined, 36); return; }
    let tries = 0;
    const tick = () => { build(); if (!cy && tries++ < 80) setTimeout(tick, 120); };
    tick();
  };
})();
