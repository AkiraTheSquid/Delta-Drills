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

  /* ---- mastery colouring (BKT posterior → red↔blue, gray = no estimate) ---- */
  const BKT_P_INIT = 0.10, BKT_HALF_LIFE_DAYS = 14.0;
  const UNKNOWN_COLOR = "#5b5b70";       // no estimate yet
  let colorMode = "mastery";             // "mastery" | "lesson"

  // Persisted engine state (guest: adaptive_state_guest) so the graph shows
  // mastery even before Practice has been opened this session.
  const _persistedState = () => {
    try {
      const email = (typeof authEmail === "string" && authEmail.trim()) ? authEmail.trim() : "guest";
      const raw = localStorage.getItem(`adaptive_state_${email}`) || localStorage.getItem("adaptive_state_guest");
      return raw ? JSON.parse(raw) : null;
    } catch (_) { return null; }
  };
  const _decay = (L, ts) => {
    if (!Number.isFinite(L)) return NaN;
    if (!ts) return L;
    const prev = Date.parse(ts);
    if (!Number.isFinite(prev)) return L;
    const days = Math.max(0, (Date.now() - prev) / 86400000);
    return BKT_P_INIT + (L - BKT_P_INIT) * Math.pow(0.5, days / BKT_HALF_LIFE_DAYS);
  };
  // Decayed BKT posterior for a KC in [0,1], or NaN when there is no estimate.
  // Prefers the app's in-memory computation; falls back to persisted state.
  const kcReadiness = (kc) => {
    if (typeof window.computeAtomReadiness === "function") {
      // NB: computeAtomReadiness coerces a non-finite fallback to 0, so we use
      // -1 as the "no posterior" sentinel (valid readiness is [0,1]). r >= 0 is
      // a real in-memory estimate; -1 falls through to the persisted read.
      const r = window.computeAtomReadiness(kc, -1);
      if (r >= 0) return r;
    }
    const s = _persistedState();
    const raw = s && s.atom_mastery ? s.atom_mastery[kc] : undefined;
    if (!Number.isFinite(raw)) return NaN;
    return _decay(raw, s.atom_last_ts ? s.atom_last_ts[kc] : null);
  };
  const kcLastTs = (kc) => {
    const s = _persistedState();
    return s && s.atom_last_ts ? s.atom_last_ts[kc] : null;
  };
  // red (low) → muted purple → blue (high); gray for no estimate.
  const masteryColor = (r) => {
    if (!Number.isFinite(r)) return UNKNOWN_COLOR;
    const t = Math.max(0, Math.min(1, r));
    const lo = [214, 72, 72], hi = [59, 130, 246];  // #d64848 → #3b82f6
    const c = lo.map((v, i) => Math.round(v + (hi[i] - v) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  };
  const nodeColor = (kc) =>
    colorMode === "mastery" ? masteryColor(kcReadiness(kc)) : lessonColor((kcById[kc] || {}).lesson);
  const masteryBand = (r) => {
    if (!Number.isFinite(r)) return "Not yet estimated";
    if (r < 0.30) return "Just starting";
    if (r < 0.60) return "Learning";
    if (r < 0.85) return "Proficient";
    return "Strong";
  };
  const relTime = (ts) => {
    if (!ts) return null;
    const t = Date.parse(ts); if (!Number.isFinite(t)) return null;
    const s = Math.max(0, (Date.now() - t) / 1000);
    if (s < 90) return "just now";
    const m = s / 60; if (m < 90) return `${Math.round(m)} min ago`;
    const h = m / 60; if (h < 36) return `${Math.round(h)} h ago`;
    return `${Math.round(h / 24)} d ago`;
  };

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

  /* ---------------- mastery handoff: iframe → graph -------------------- */
  // The embedded practice page posts `delta:kc-mastered` when the competency
  // bar crosses 0.95. Sequence: refresh the learner state the iframe just
  // wrote → drop back to the map → animate the node red→blue → offer the next
  // concept. The iframe stays open for ~900ms so the bar visibly reaches the
  // gate before the overlay closes.
  const MASTERED_HOLD_MS = 900;
  const NODE_ANIM_MS = 1100;

  // The iframe is a second app instance: it writes mastery to localStorage (or
  // the backend), but THIS window's in-memory adaptiveStateJson — what
  // computeAtomReadiness reads — is stale until we re-read it. Without this the
  // node recolours to its old value and the animation lands on the wrong blue.
  const _refreshLearnerState = async () => {
    if (typeof practiceMode !== "undefined" && practiceMode === "backend" &&
        typeof loadBackendAdaptiveState === "function") {
      try { await loadBackendAdaptiveState(); return; } catch (_) { /* fall through */ }
    }
    try {
      const email = (typeof authEmail === "string" && authEmail.trim()) ? authEmail.trim() : "guest";
      const raw = localStorage.getItem(`adaptive_state_${email}`) || localStorage.getItem("adaptive_state_guest");
      // Bare assignment on purpose — adaptiveStateJson is a module-scope `let`
      // shared across the practice scripts; window.x = would shadow it.
      if (raw) adaptiveStateJson = raw;
    } catch (_) { /* keep the stale value rather than blanking it */ }
  };

  const _parseRgb = (css) => {
    const m = String(css || "").match(/rgba?\((\d+)[,\s]+(\d+)[,\s]+(\d+)/);
    return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : null;
  };

  const _animateNodeColor = (kc, fromCss, toCss) => {
    if (!cy) return;
    const node = cy.getElementById(kc);
    if (!node || node.empty()) return;
    const from = _parseRgb(fromCss);
    const to = _parseRgb(toCss);
    if (!from || !to) { node.style("background-color", toCss); return; }
    const started = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - started) / NODE_ANIM_MS);
      const eased = t * t * (3 - 2 * t);
      const c = from.map((v, i) => Math.round(v + (to[i] - v) * eased));
      node.style("background-color", `rgb(${c[0]},${c[1]},${c[2]})`);
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };

  // Next concept worth practising: a direct dependent of the mastered KC whose
  // OTHER prerequisites are already cleared (0.85 = the engine's unlock
  // threshold), least-mastered first. Returns null when nothing downstream is
  // ready — the learner picks their own next node instead.
  const _nextConcept = (kc) => {
    const ready = (id) => {
      const r = kcReadiness(id);
      return Number.isFinite(r) && r >= 0.85;
    };
    const candidates = (childrenOf[kc] || []).filter((id) => {
      const r = kcReadiness(id);
      if (Number.isFinite(r) && r >= 0.95) return false; // already mastered
      return (parentsOf[id] || []).every((p) => p === kc || ready(p));
    });
    if (!candidates.length) return null;
    return candidates.sort((a, b) => {
      const ra = kcReadiness(a), rb = kcReadiness(b);
      return (Number.isFinite(ra) ? ra : 0) - (Number.isFinite(rb) ? rb : 0);
    })[0];
  };

  let masteredToast = null;

  const _dismissToast = () => {
    if (masteredToast) { masteredToast.remove(); masteredToast = null; }
  };

  const _showMasteredToast = (kc, nextKc) => {
    _dismissToast();
    const kcObj = kcById[kc] || {};
    const nextObj = nextKc ? (kcById[nextKc] || {}) : null;
    masteredToast = document.createElement("div");
    masteredToast.className = "kg2-mastered-toast";
    masteredToast.innerHTML =
      `<div class="kg2-mastered-title">Mastered — ${esc(kcObj.title || kc)}</div>` +
      `<div class="kg2-mastered-body">${nextObj
        ? `This unlocks <strong>${esc(nextObj.title || nextKc)}</strong>.`
        : "Nothing downstream is waiting on it — pick any bubble to keep going."}</div>` +
      '<div class="kg2-mastered-actions">' +
        (nextObj ? '<button type="button" class="kg2-mastered-go">Practice it next</button>' : "") +
        '<button type="button" class="kg2-mastered-stay">Stay on the map</button>' +
      "</div>";
    document.body.appendChild(masteredToast);
    const go = masteredToast.querySelector(".kg2-mastered-go");
    if (go) go.addEventListener("click", () => {
      _dismissToast();
      selectNode(nextKc);
      openMaximize(nextKc);
    });
    masteredToast.querySelector(".kg2-mastered-stay").addEventListener("click", _dismissToast);
  };

  const _onKcMastered = async (kc) => {
    if (!kc || !kcById[kc]) return;
    const beforeCss = cy ? cy.getElementById(kc).style("background-color") : null;
    await _refreshLearnerState();
    setTimeout(() => {
      // Select BEFORE closing: closeMaximize re-centres on the selected node,
      // and the point of this moment is watching THIS node change colour.
      selectNode(kc);
      closeMaximize();
      if (cy) {
        const n = cy.getElementById(kc);
        if (n && !n.empty()) {
          cy.animate({ center: { eles: n }, zoom: Math.max(cy.zoom(), 1.1) }, { duration: 320 });
        }
      }
      const afterCss = nodeColor(kc);
      if (colorMode === "mastery") _animateNodeColor(kc, beforeCss, afterCss);
      // Every OTHER node also moved (FIRe credit reaches prerequisites), so
      // repaint the rest once the focused animation has finished.
      setTimeout(recolor, NODE_ANIM_MS + 60);
      _showMasteredToast(kc, _nextConcept(kc));
    }, MASTERED_HOLD_MS);
  };

  window.addEventListener("message", (e) => {
    // Same-origin only: the practice iframe is served from this app.
    if (e.origin !== window.location.origin) return;
    if (!e.data || e.data.type !== "delta:kc-mastered") return;
    _onKcMastered(e.data.kc);
  });

  /* ---------------- legend (mode-aware) + recolour --------------------- */
  const buildLegend = () => {
    const el = $("kg-legend");
    if (!el) return;
    if (colorMode === "mastery") {
      el.classList.add("kg2-legend-mastery");
      el.innerHTML =
        '<span class="kg2-li"><span class="kg2-li-dot" style="background:' + UNKNOWN_COLOR + '"></span>No estimate</span>' +
        '<span class="kg2-li kg2-li-scale"><span>less</span><span class="kg2-scale-bar"></span><span>more mastered</span></span>';
    } else {
      el.classList.remove("kg2-legend-mastery");
      const seen = [];
      Object.values(kcById).forEach((k) => { if (!seen.includes(k.lesson)) seen.push(k.lesson); });
      seen.sort();
      el.innerHTML = seen.map((lid) => {
        const lm = lessonMeta[lid] || {};
        return `<span class="kg2-li"><span class="kg2-li-dot" style="background:${lessonColor(lid)}"></span>${esc(lm.title || lid)}</span>`;
      }).join("");
    }
  };

  const recolor = () => {
    if (!cy) return;
    cy.batch(() => cy.nodes().forEach((n) => n.style("background-color", nodeColor(n.id()))));
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
            "background-color": (n) => nodeColor(n.id()),
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

    /* ---- colour-mode toggle (Mastery ↔ Lessons) ---- */
    const controls = document.querySelector(".kg2-controls");
    if (controls && !$("kg-colormode")) {
      const seg = document.createElement("div");
      seg.className = "kg2-seg";
      seg.id = "kg-colormode";
      seg.innerHTML =
        '<button type="button" data-mode="mastery" class="active">Mastery</button>' +
        '<button type="button" data-mode="lesson">Lessons</button>';
      controls.insertBefore(seg, controls.firstChild);
      seg.querySelectorAll("button").forEach((b) =>
        b.addEventListener("click", () => {
          colorMode = b.dataset.mode;
          seg.querySelectorAll("button").forEach((x) => x.classList.toggle("active", x === b));
          buildLegend();
          recolor();
        }));
    }

    /* ---- hover popup: per-KC learner-model data ---- */
    let tip = $("kg-tip");
    if (!tip) {
      tip = document.createElement("div");
      tip.id = "kg-tip";
      tip.className = "kg2-tip hidden";
      (document.querySelector(".kg2-graph") || document.body).appendChild(tip);
    }
    const moveTip = (rpos) => { tip.style.left = (rpos.x + 16) + "px"; tip.style.top = (rpos.y + 16) + "px"; };
    const showTip = (kc, rpos) => {
      const k = kcById[kc];
      if (!k) return;
      const r = kcReadiness(kc);
      const pct = Number.isFinite(r) ? Math.round(r * 100) + "%" : "—";
      const parents = parentsOf[kc] || [];
      const ready = parents.filter((p) => { const pr = kcReadiness(p); return Number.isFinite(pr) && pr >= 0.85; }).length;
      const last = relTime(kcLastTs(kc));
      tip.innerHTML =
        `<div class="kg2-tip-title">${esc(k.title)}</div>` +
        `<div class="kg2-tip-row"><span>Mastery estimate</span><strong style="color:${masteryColor(r)}">${pct}</strong></div>` +
        `<div class="kg2-tip-row"><span>State</span><span>${masteryBand(r)}</span></div>` +
        (last ? `<div class="kg2-tip-row"><span>Last practiced</span><span>${last}</span></div>` : "") +
        (parents.length
          ? `<div class="kg2-tip-row"><span>Prerequisites ready</span><span>${ready}/${parents.length}</span></div>`
          : `<div class="kg2-tip-row"><span>Prerequisites</span><span>none — foundation</span></div>`) +
        `<div class="kg2-tip-meta">${esc(k.topic)} · ${esc((lessonMeta[k.lesson] || {}).title || k.lesson)}</div>`;
      moveTip(rpos);
      tip.classList.remove("hidden");
    };
    cy.on("mouseover", "node", (evt) => showTip(evt.target.id(), evt.target.renderedPosition()));
    cy.on("mousemove", "node", (evt) => moveTip(evt.target.renderedPosition()));
    cy.on("mouseout", "node", () => tip.classList.add("hidden"));
    cy.on("pan zoom", () => tip.classList.add("hidden"));

    // Recolour when the learner model changes (a graded attempt updates BKT).
    window.addEventListener("delta:adaptive-state-changed", recolor);

    buildLegend();
    recolor();
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
