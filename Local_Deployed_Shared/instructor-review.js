/* INSTRUCTOR REVIEW — what instructor feedback mode actually SHOWS.
 *
 * One page (#page-instructor-review), two doors (Seth, 2026-08-24: "they
 * should have the choice to choose between providing feedback through the
 * graph, and providing feedback through the questions themselves"):
 *
 *   QUESTIONS  browse the whole bank topic → subtopic → question, each one
 *              shown WITH its canonical solution — an instructor reviews
 *              content, they don't grind drills to reach it. Flags post to
 *              the existing /api/practice/problem-feedback endpoint, so an
 *              actionable flag from an allowlisted account still queues the
 *              repair runner exactly as an in-practice flag does.
 *   GRAPH      the same concept graph the Knowledge Graph tab draws, but its
 *              OWN cytoscape instance (no learner-mastery overlay, no shared
 *              state with graph-viz.js): tap an edge or a node to flag it,
 *              or propose a missing edge by tapping two concepts. Posts to
 *              /api/practice/graph-feedback (append-only log; no automated
 *              repair behind it — edge changes reshape the unlock lattice).
 *
 * Data comes from the two files the app already ships: questions_structured
 * .json (the bank, canonical_solution included) and concept-graph/
 * graph-viz.json. Both are fetched lazily on the first door click, never at
 * boot — learners pay nothing for a surface they never open.
 *
 * Submission plumbing mirrors practice/api.js reportProblem: best-effort via
 * window.apiFetch (app.js), and on ANY failure — offline, 401, apiFetch not
 * there — the entry falls back to a localStorage queue so nothing is lost.
 * The question flags reuse the SAME queue key practice uses
 * ("problem_feedback_queue"); graph flags get their own ("graph_feedback_
 * queue") because their shape is different.
 *
 * Leaving the mode leaves the page: instructor-mode.js dispatches
 * dd-instructor-mode-changed, and if the flag goes OFF while this page is
 * showing we click the hidden #ir-exit-goto proxy (data-goto-tab="practice")
 * — switchTab is a top-level const in app.js, deliberately not on window, and
 * proxy buttons are the sanctioned way in. */
(() => {
  "use strict";

  const page = document.getElementById("page-instructor-review");
  if (!page) return;

  const el = (id) => document.getElementById(id);
  const esc = (s) =>
    String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");

  /* ── view switching inside the page ─────────────────────────────── */
  const views = { chooser: el("ir-chooser"), questions: el("ir-questions"), graph: el("ir-graph") };
  const backBtn = el("ir-back");
  const show = (name) => {
    Object.entries(views).forEach(([k, v]) => v && v.classList.toggle("hidden", k !== name));
    if (backBtn) backBtn.classList.toggle("hidden", name === "chooser");
  };
  if (backBtn) backBtn.addEventListener("click", () => show("chooser"));

  /* ── shared submit plumbing ─────────────────────────────────────── */
  const queueLocally = (key, entry) => {
    try {
      const q = JSON.parse(localStorage.getItem(key) || "[]");
      q.push({ ...entry, timestamp: new Date().toISOString() });
      localStorage.setItem(key, JSON.stringify(q));
    } catch (_) { /* storage blocked — best effort only */ }
  };
  const post = async (path, body, queueKey) => {
    const fetcher = window.apiFetch;
    if (typeof fetcher === "function") {
      try {
        const res = await fetcher(path, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (res.ok) return { sent: true };
      } catch (_) { /* fall through to the queue */ }
    }
    queueLocally(queueKey, body);
    return { sent: false };
  };
  const flashStatus = (node, sent) => {
    if (!node) return;
    node.textContent = sent ? "sent ✓" : "saved locally ✓";
    node.classList.remove("hidden");
  };

  /* ══ QUESTIONS ═══════════════════════════════════════════════════ */
  const QUESTION_TAGS = [
    ["broken", "🚩 Broken"],
    ["unclear", "😕 Unclear"],
    ["wrong_image", "🖼 Wrong image"],
    ["good", "👍 Good"],
  ];
  let bank = null; // [{id, curriculum, exercise}]
  let activeTopic = null;

  const loadBank = async () => {
    if (bank) return bank;
    const res = await fetch("questions_structured.json");
    bank = await res.json();
    return bank;
  };

  const questionCard = (q) => {
    const c = q.curriculum || {};
    const ex = q.exercise || {};
    const art = document.createElement("article");
    art.className = "ir-q";
    art.innerHTML = `
      <header class="ir-q-head">
        <span class="ir-q-id">#${esc(q.id)}</span>
        <span class="ir-q-diff">${esc(c.difficulty_label || "?")} · ${esc(c.difficulty_score ?? "?")}</span>
        <span class="ir-q-sub">${esc(c.subtopic || "")}</span>
      </header>
      <div class="ir-q-text">${esc(ex.question_text || "")}</div>
      <details class="ir-q-details"><summary>Starter code</summary><pre>${esc(ex.starter_code || "—")}</pre></details>
      <details class="ir-q-details" open><summary>Canonical solution</summary><pre>${esc(ex.canonical_solution || "—")}</pre></details>
      <details class="ir-q-details"><summary>Expected output</summary><pre>${esc(ex.expected_output || "—")}</pre></details>
      <div class="ir-q-flags">
        ${QUESTION_TAGS.map(([tag, label]) => `<button type="button" class="ghost ir-flag" data-tag="${tag}">${label}</button>`).join("")}
        <span class="ir-q-status hidden"></span>
      </div>
      <textarea class="ir-q-note" rows="1" maxlength="5000"
        placeholder="Optional note — what's wrong, or what's good, about this problem…"></textarea>`;
    const note = art.querySelector(".ir-q-note");
    const status = art.querySelector(".ir-q-status");
    art.querySelectorAll(".ir-flag").forEach((btn) => {
      btn.addEventListener("click", async () => {
        btn.disabled = true;
        const { sent } = await post(
          "/api/practice/problem-feedback",
          { question_id: q.id, tag: btn.dataset.tag, note: note.value.trim(), correct: null },
          "problem_feedback_queue"
        );
        btn.disabled = false;
        flashStatus(status, sent);
      });
    });
    return art;
  };

  const renderSubtopic = (subKey) => {
    const list = el("ir-qlist");
    list.innerHTML = "";
    bank
      .filter((q) => (q.curriculum || {}).subtopic_key === subKey)
      .forEach((q) => list.appendChild(questionCard(q)));
  };

  const renderTopic = (topic) => {
    activeTopic = topic;
    el("ir-topics").querySelectorAll(".ir-pill").forEach((p) =>
      p.classList.toggle("ir-pill--on", p.dataset.topic === topic));
    const subs = new Map(); // subtopic_key -> {label, n}
    bank.forEach((q) => {
      const c = q.curriculum || {};
      if (c.topic !== topic) return;
      const cur = subs.get(c.subtopic_key) || { label: c.subtopic, n: 0 };
      cur.n += 1;
      subs.set(c.subtopic_key, cur);
    });
    const box = el("ir-subtopics");
    box.innerHTML = "";
    subs.forEach(({ label, n }, key) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "ir-pill ir-pill--sub";
      b.dataset.sub = key;
      b.textContent = `${label} (${n})`;
      b.addEventListener("click", () => {
        box.querySelectorAll(".ir-pill").forEach((p) => p.classList.toggle("ir-pill--on", p === b));
        renderSubtopic(key);
      });
      box.appendChild(b);
    });
    el("ir-qlist").innerHTML =
      '<p class="ir-hint">Pick a subtopic to list its questions.</p>';
  };

  const openQuestions = async () => {
    show("questions");
    try {
      await loadBank();
    } catch (_) {
      el("ir-qlist").innerHTML = '<p class="ir-hint">Couldn’t load the question bank.</p>';
      return;
    }
    const topicsBox = el("ir-topics");
    if (!topicsBox.childElementCount) {
      const counts = new Map();
      bank.forEach((q) => {
        const t = (q.curriculum || {}).topic || "Other";
        counts.set(t, (counts.get(t) || 0) + 1);
      });
      counts.forEach((n, topic) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "ir-pill";
        b.dataset.topic = topic;
        b.textContent = `${topic} (${n})`;
        b.addEventListener("click", () => renderTopic(topic));
        topicsBox.appendChild(b);
      });
    }
    if (!activeTopic) el("ir-qlist").innerHTML = '<p class="ir-hint">Pick a topic above.</p>';
  };

  /* ══ GRAPH ═══════════════════════════════════════════════════════ */
  const EDGE_TAGS = [
    ["wrong_direction", "Points the wrong way"],
    ["wrong_type", "Wrong kind of edge"],
    ["should_not_exist", "Shouldn't exist"],
    ["good", "👍 Good edge"],
  ];
  const NODE_TAGS = [
    ["mislabeled", "Mislabeled"],
    ["wrong_topic", "Filed under the wrong topic"],
    ["should_not_exist", "Shouldn't exist"],
    ["good", "👍 Good concept"],
  ];
  const MISSING_TAGS = [
    ["prereq", "Should be a prerequisite edge"],
    ["enc", "Should be an encompassing edge"],
  ];

  let cy = null;
  let selection = null; // {kind, source, target, edge_type}
  let missingArm = null; // null | "first" | node-id awaiting second tap
  let chosenTag = null;

  const setPanelHint = (t) => { const h = el("ir-panel-hint"); if (h) h.textContent = t; };

  const openForm = (title, tags) => {
    chosenTag = null;
    el("ir-form").classList.remove("hidden");
    el("ir-form-target").innerHTML = title;
    const box = el("ir-form-tags");
    box.innerHTML = "";
    tags.forEach(([tag, label]) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "ghost ir-flag";
      b.textContent = label;
      b.addEventListener("click", () => {
        chosenTag = tag;
        box.querySelectorAll(".ir-flag").forEach((p) => p.classList.toggle("ir-flag--on", p === b));
      });
      box.appendChild(b);
    });
    el("ir-form-note").value = "";
    const s = el("ir-form-status");
    if (s) s.classList.add("hidden");
  };

  const selectEdge = (e) => {
    missingArm = null;
    const enc = !!e.data("enc");
    selection = { kind: "edge", source: e.data("source"), target: e.data("target"), edge_type: enc ? "enc" : "prereq" };
    openForm(
      `<strong>${esc(e.data("source"))}</strong> → <strong>${esc(e.data("target"))}</strong><br>` +
        `<span class="ir-hint">${enc ? "encompassing" : "prerequisite"} edge</span>`,
      EDGE_TAGS
    );
  };

  const selectNode = (n) => {
    if (missingArm === "first") {
      missingArm = n.id();
      setPanelHint(`First concept: ${n.data("label")}. Now tap the SECOND concept (the one it should point to).`);
      return;
    }
    if (missingArm && missingArm !== n.id()) {
      selection = { kind: "missing_edge", source: missingArm, target: n.id(), edge_type: null };
      missingArm = null;
      openForm(
        `Missing edge: <strong>${esc(selection.source)}</strong> → <strong>${esc(selection.target)}</strong>`,
        MISSING_TAGS
      );
      setPanelHint("Pick which kind of edge it should be, add a note, send.");
      return;
    }
    selection = { kind: "node", source: n.id(), target: null, edge_type: null };
    openForm(
      `<strong>${esc(n.data("label"))}</strong><br><span class="ir-hint">${esc(n.data("topic") || "")} · ${esc(n.data("family") || "")}</span>`,
      NODE_TAGS
    );
  };

  const buildGraph = async () => {
    const status = el("ir-graph-status");
    let data;
    try {
      const res = await fetch("concept-graph/graph-viz.json?v=5");
      data = await res.json();
    } catch (_) {
      if (status) status.textContent = "Couldn't load the graph data.";
      return;
    }
    try {
      if (window.cytoscapeFcose) cytoscape.use(window.cytoscapeFcose);
    } catch (_) { /* already registered */ }
    cy = cytoscape({
      container: el("ir-cy"),
      elements: [
        ...data.nodes.map((n) => ({ data: { id: n.id, label: n.label, topic: n.topic, family: n.family } })),
        ...data.edges.map((e, i) => ({
          data: { id: `e${i}`, source: e.source, target: e.target, enc: e.enc ? 1 : 0 },
          classes: e.enc ? "enc" : "prereq",
        })),
      ],
      wheelSensitivity: 0.2,
      style: [
        { selector: "node", style: { "background-color": "#8ab4f8", width: 14, height: 14, label: "data(label)", color: "#c0c0d0", "font-size": 8, "text-wrap": "wrap", "text-max-width": "80px", "text-opacity": 0.75 } },
        { selector: "edge.prereq", style: { width: 2, "curve-style": "straight", "line-color": "#e3212c", "target-arrow-shape": "triangle", "target-arrow-color": "#e3212c", "arrow-scale": 0.7 } },
        { selector: "edge.enc", style: { width: 1, "curve-style": "straight", "line-color": "#5a5a86", "line-style": "dashed", opacity: 0.55 } },
        { selector: ":selected", style: { "border-width": 3, "border-color": "#ffffff", "line-color": "#ffffff", "target-arrow-color": "#ffffff", opacity: 1 } },
      ],
      layout: window.cytoscapeFcose
        ? { name: "fcose", animate: false, randomize: true, nodeRepulsion: 6500, idealEdgeLength: 55, fit: true, padding: 28 }
        : { name: "cose", animate: false, fit: true, padding: 28 },
    });
    cy.on("tap", "edge", (evt) => selectEdge(evt.target));
    cy.on("tap", "node", (evt) => selectNode(evt.target));
    if (status) status.classList.add("hidden");
  };

  const openGraph = () => {
    show("graph");
    // Cytoscape can't size a display:none container — build only now, once
    // the pane is actually visible (same deferral graph-viz.js documents).
    if (!cy && typeof cytoscape !== "undefined") requestAnimationFrame(buildGraph);
    else if (cy) requestAnimationFrame(() => cy.resize());
  };

  el("ir-missing-edge").addEventListener("click", () => {
    missingArm = "first";
    el("ir-form").classList.add("hidden");
    setPanelHint("Tap the FIRST concept (the prerequisite / the part).");
  });

  el("ir-form-send").addEventListener("click", async () => {
    if (!selection || !chosenTag) {
      flashStatus(el("ir-form-status"), false);
      el("ir-form-status").textContent = "pick a flag first";
      return;
    }
    const body = {
      kind: selection.kind,
      source: selection.source,
      target: selection.target,
      edge_type: selection.kind === "missing_edge" ? chosenTag : selection.edge_type,
      tag: selection.kind === "missing_edge" ? "proposed" : chosenTag,
      note: el("ir-form-note").value.trim(),
    };
    const btn = el("ir-form-send");
    btn.disabled = true;
    const { sent } = await post("/api/practice/graph-feedback", body, "graph_feedback_queue");
    btn.disabled = false;
    flashStatus(el("ir-form-status"), sent);
  });

  /* ── doors + mode exit ──────────────────────────────────────────── */
  el("ir-door-questions").addEventListener("click", openQuestions);
  el("ir-door-graph").addEventListener("click", openGraph);

  document.addEventListener("dd-instructor-mode-changed", (e) => {
    const on = !!(e.detail && e.detail.on);
    if (!on && !page.classList.contains("hidden")) {
      const exit = el("ir-exit-goto");
      if (exit) exit.click();
    }
  });
})();
