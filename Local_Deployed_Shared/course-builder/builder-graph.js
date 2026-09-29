/* course-builder/builder-graph.js — the course builder's graph: the Knowledge
   Graph tab's map with nothing around it.

   Its own Cytoscape instance in the builder's right pane, never the KG tab's
   (lesson-graph.js owns that one and its lesson pane, gate state and
   instructor contract). What makes it the SAME graph is borrowed, not
   copied: the node look and edge rules from concept-graph/kg-look.js, the
   mastery colour and readiness reading from lesson-graph.js's read-only
   exports (`deltaKcMasteryColor`, `deltaKcReadinessInfo`,
   `deltaKcMasteryBand`), the server's gate state from kc_lattice_read.js's
   `getKcLattice`, and dagre's bottom-up rows with the KG's spacing.

   A tap on a node opens a small card beside it: the concept's mastery,
   gate state and neighbours, plus the two things the builder does with a
   node — hand it to the chat, or read its lesson. The builder
   (course-builder.js) supplies those two actions. */
(function () {
  "use strict";

  const D = () => window.DDBuilderData;
  const LOOK = () => window.DeltaKgLook || null;
  const HL = "#ffd23f"; // the KG's own highlight gold (lesson-graph.js ACCENT)

  let cy = null;
  let host = null;       // the pane the canvas + card live in
  let canvas = null;
  let card = null;
  let actions = { addToChat: () => {}, viewLesson: () => {} };
  let shownFor = null;   // the course id the canvas was built for
  let wanted = 0;        // bumped per show(); an older call that resolves late is dropped
  let ink = "#c8cdd8";

  const esc = (v) => String(v == null ? "" : v).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  const readiness = (kc) => {
    const info = typeof window.deltaKcReadinessInfo === "function" ? window.deltaKcReadinessInfo(kc) : null;
    return info || { r: NaN, source: "none" };
  };
  const colorOf = (kc) =>
    typeof window.deltaKcMasteryColor === "function" ? window.deltaKcMasteryColor(readiness(kc).r) : "#5b5b70";
  const latticeRow = (kc) => {
    const l = typeof window.getKcLattice === "function" ? window.getKcLattice() : null;
    return (l && l.kcs && l.kcs[kc]) || null;
  };

  const sheet = () => [
    { selector: "node", style: Object.assign(
      LOOK() ? LOOK().node(1, () => ink) : {
        shape: "round-rectangle", label: "data(label)", width: "label", height: "label", padding: "11px",
        "text-wrap": "wrap", "text-max-width": "120px", "text-valign": "center", "text-halign": "center",
        "font-size": 12.5, "font-weight": 600, color: "#15151f",
      },
      { "background-color": (n) => colorOf(n.id()) }) },
    ...(LOOK() ? LOOK().edgeRules() : [{ selector: "edge", style: {
      "curve-style": "bezier", width: 1.4, "line-color": "#8f9bb8",
      "target-arrow-shape": "triangle", "target-arrow-color": "#8f9bb8", "arrow-scale": 0.8,
    } }]),
    { selector: ".cb-faded", style: { opacity: 0.12 } },
    { selector: "node.cb-chain", style: { opacity: 1, "border-width": 3, "border-color": HL, "z-index": 50 } },
    { selector: "node.cb-picked", style: { opacity: 1, "border-width": 5, "border-color": HL, "z-index": 99 } },
    { selector: "edge.cb-chain", style: { opacity: 1, width: 3, "line-color": HL, "target-arrow-color": HL, "z-index": 60, display: "element" } },
    { selector: "node.cb-in-chat", style: { "outline-width": 5, "outline-color": HL, "outline-opacity": 0.45 } },
  ];

  const layout = () => {
    if (!cy) return;
    const eles = LOOK() ? LOOK().layoutEles(cy) : cy.elements();
    eles.layout({
      name: window.cytoscapeDagre ? "dagre" : "cose",
      rankDir: "BT", nodeSep: 26, rankSep: 150, edgeSep: 12,
      animate: false, fit: true, padding: 40, nodeDimensionsIncludeLabels: true,
    }).run();
  };

  /* ---------------- the node card ------------------------------------ */
  const closeCard = () => {
    if (card) card.hidden = true;
    if (cy) cy.elements().removeClass("cb-faded cb-chain cb-picked");
  };

  const ancestors = (id) => {
    const out = new Set();
    const stack = [id];
    while (stack.length) D().parents(stack.pop()).forEach((p) => { if (!out.has(p)) { out.add(p); stack.push(p); } });
    return out;
  };

  const highlight = (id) => {
    const chain = new Set([id, ...ancestors(id)]);
    cy.batch(() => {
      cy.elements().removeClass("cb-chain cb-picked").addClass("cb-faded");
      cy.nodes().forEach((n) => {
        if (chain.has(n.id())) n.removeClass("cb-faded").addClass(n.id() === id ? "cb-picked" : "cb-chain");
      });
      cy.edges().forEach((e) => {
        if (chain.has(e.source().id()) && chain.has(e.target().id())) e.removeClass("cb-faded").addClass("cb-chain");
      });
    });
  };

  const STATE_WORDS = {
    learned: "Learned", frontier: "Ready to practise", locked: "Locked — prerequisites first",
    disabled: "Switched off",
  };

  /* A small amount of diagnostics, not the KG's whole dock: how well it is
     known and on what evidence, whether practice will serve it, and its
     place in the graph. */
  const cardHtml = (id) => {
    const kc = D().kc(id) || { id, title: id };
    const info = readiness(id);
    const r = info.r;
    const pct = Number.isFinite(r) ? Math.round(r * 100) : null;
    const band = typeof window.deltaKcMasteryBand === "function" ? window.deltaKcMasteryBand(r) : "";
    const src = { atom: "measured", subtopic: "from its lesson", extrapolated: "projected", topic: "from its topic" }[info.source] || "";
    const row = latticeRow(id);
    const state = row && STATE_WORDS[row.state];
    const nP = D().parents(id).length, nC = D().children(id).length;
    const hasLesson = !!D().lesson(id);
    return `
      <div class="cb-card-head">
        <div class="cb-card-title">${esc(kc.title || id)}</div>
        <button type="button" class="cb-card-x" data-act="close" aria-label="Close">×</button>
      </div>
      <code class="cb-card-id">${esc(id)}</code>
      <div class="cb-card-meter" aria-label="Mastery ${pct == null ? "not estimated" : pct + "%"}">
        <span class="cb-card-meter-fill" style="width:${pct == null ? 0 : pct}%;background:${colorOf(id)}"></span>
      </div>
      <dl class="cb-card-stats">
        <div><dt>Mastery</dt><dd>${pct == null ? esc(band || "Not yet estimated")
          : `${pct}%${band ? ` · ${esc(band)}` : ""}${src ? ` <span class="cb-card-dim">(${src})</span>` : ""}`}</dd></div>
        ${state ? `<div><dt>Practice</dt><dd>${esc(state)}</dd></div>` : ""}
        <div><dt>Graph</dt><dd>${nP} prerequisite${nP === 1 ? "" : "s"} · unlocks ${nC}</dd></div>
        <div><dt>Lesson</dt><dd>${esc(D().lessonTitle(kc.lesson) || kc.topic || "—")}</dd></div>
      </dl>
      <div class="cb-card-actions">
        <button type="button" class="cb-btn cb-btn-ghost" data-act="chat">＋ Add to AI chat</button>
        <button type="button" class="cb-btn cb-btn-solid" data-act="lesson" ${hasLesson ? "" : "disabled title=\"No lesson written for this concept yet\""}>View lesson</button>
      </div>`;
  };

  // Beside the node, flipped to whichever side has room, kept inside the pane.
  const placeCard = (node) => {
    const p = node.renderedPosition();
    const bb = node.renderedBoundingBox();
    const W = host.clientWidth, H = host.clientHeight;
    const cw = card.offsetWidth, ch = card.offsetHeight;
    const gap = 14;
    let x = bb.x2 + gap;
    if (x + cw > W - 8) x = bb.x1 - gap - cw;
    x = Math.max(8, Math.min(W - cw - 8, x));
    const y = Math.max(8, Math.min(H - ch - 8, p.y - ch / 2));
    card.style.left = `${x}px`;
    card.style.top = `${y}px`;
  };

  const openCard = (id) => {
    const node = cy.getElementById(id);
    if (!node || node.empty()) return;
    highlight(id);
    card.innerHTML = cardHtml(id);
    card.dataset.kc = id;
    card.hidden = false;
    placeCard(node);
    card.querySelector('[data-act="chat"]').focus({ preventScroll: true });
  };

  /* Keyboard: the canvas has no focusable nodes, so the arrow keys walk
     them in reading order (top to bottom, then left to right), centring
     each one and opening its card — whose buttons Tab then reaches. */
  const step = (dir) => {
    if (!cy || !cy.nodes().length) return;
    const nodes = cy.nodes().sort((a, b) => (a.position("y") - b.position("y")) || (a.position("x") - b.position("x")));
    const cur = card.hidden ? -1 : nodes.toArray().findIndex((n) => n.id() === card.dataset.kc);
    const i = cur < 0 ? (dir > 0 ? 0 : nodes.length - 1) : (cur + dir + nodes.length) % nodes.length;
    const n = nodes[i];
    cy.center(n);
    openCard(n.id());
  };

  const onCardClick = (e) => {
    const act = e.target.closest("[data-act]")?.dataset.act;
    const id = card.dataset.kc;
    if (!act || !id) return;
    if (act === "close") closeCard();
    else if (act === "chat") {
      actions.addToChat(id);
      markInChat(id, true);
      const b = e.target.closest("button");
      b.textContent = "✓ In the chat";
      b.disabled = true;
    } else if (act === "lesson") {
      closeCard();
      actions.viewLesson(id);
    }
  };

  const markInChat = (id, on) => {
    if (!cy) return;
    const n = cy.getElementById(id);
    if (n && n.length) n.toggleClass("cb-in-chat", on);
  };

  /* ---------------- build / show ------------------------------------- */
  const mount = (el, acts) => {
    host = el;
    actions = Object.assign(actions, acts || {});
    canvas = document.createElement("div");
    canvas.className = "cb-graph-canvas";
    card = document.createElement("div");
    card.className = "cb-card";
    card.setAttribute("role", "dialog");
    card.setAttribute("aria-label", "Concept details");
    card.hidden = true;
    card.addEventListener("click", onCardClick);
    host.append(canvas, card);
    host.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !card.hidden) { closeCard(); host.focus({ preventScroll: true }); return; }
      const dir = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      if (dir) { e.preventDefault(); step(dir); }
    });
    window.addEventListener("delta:kg-look-changed", () => { if (cy) { cy.style(sheet()); layout(); } });
    window.addEventListener("delta:theme-changed", () => { readInk(); if (cy) cy.style().update(); });
    window.addEventListener("delta:adaptive-state-changed", () => { if (cy) cy.style().update(); });
  };

  const readInk = () => { if (host) ink = getComputedStyle(host).color || ink; };

  /* One course's graph. Rebuilt whole on a course change rather than
     filtered in place: kg-look.js's shortcut marking remembers the edges it
     first saw, so a canvas reused across courses would keep the old set.
     Resolves true (shown), false (no Cytoscape) or null (a later call won). */
  const show = async (courseId, inChat) => {
    if (!host || typeof cytoscape === "undefined") return false;
    try { if (window.cytoscapeDagre) cytoscape.use(window.cytoscapeDagre); } catch (_) { /* already registered */ }
    const mine = ++wanted;
    const ids = await D().conceptsFor(courseId || "");
    if (mine !== wanted) return null; // superseded by a later show()
    if (cy && shownFor === (courseId || "")) { cy.resize(); return true; }
    if (cy) { cy.destroy(); cy = null; }
    closeCard();
    readInk();
    const elements = [];
    ids.forEach((id) => elements.push({ data: { id, label: (D().kc(id) || {}).title || id } }));
    let n = 0;
    ids.forEach((id) => {
      const enc = (D().kc(id) || {}).encompassing || {};
      D().parents(id).forEach((p) => {
        if (!ids.has(p)) return;
        const w = typeof enc[p] === "number" ? enc[p] : 0;
        elements.push({ data: { id: `cbe${n++}`, source: p, target: id, w, kind: w > 0 ? "encompassing" : "prereq" } });
      });
    });
    cy = cytoscape({ container: canvas, elements, style: sheet(), wheelSensitivity: 0.25, minZoom: 0.1, maxZoom: 3 });
    if (LOOK()) { LOOK().markShortcuts(cy); LOOK().curve(cy); }
    layout();
    shownFor = courseId || "";
    cy.on("tap", "node", (evt) => openCard(evt.target.id()));
    cy.on("tap", (evt) => { if (evt.target === cy) closeCard(); });
    cy.on("pan zoom", () => { if (!card.hidden) closeCard(); });
    (inChat || []).forEach((id) => markInChat(id, true));
    // The server's gate state + mastery arrive after the first paint; repaint
    // when they do rather than hold the canvas empty for them.
    Promise.resolve(window.deltaRefreshKcLattice?.()).then(() => cy && cy.style().update()).catch(() => {});
    return true;
  };

  const resize = () => { if (cy) { cy.resize(); } };
  const fit = () => { if (cy) { cy.resize(); cy.fit(undefined, 36); } };

  window.DDBuilderGraph = { mount, show, resize, fit, closeCard, markInChat, count: () => (cy ? cy.nodes().length : 0) };
})();
