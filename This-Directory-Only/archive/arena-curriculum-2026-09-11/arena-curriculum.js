/* ARENA's reading/exercise graph. These nodes never enter the mastery engine. */
(() => {
  const page = document.querySelector("#page-knowledge-graph .kg-container");
  if (!page) return;
  const skills = page.querySelector(".kg2-wrap");
  const switcher = document.createElement("nav");
  switcher.className = "arena-graph-modes";
  switcher.setAttribute("aria-label", "Graph view");
  switcher.innerHTML = '<button type="button" aria-pressed="true">Practice skills</button>' +
    '<button type="button" aria-pressed="false">ARENA curriculum</button>';
  page.prepend(switcher);
  const panel = document.createElement("section");
  panel.className = "arena-curriculum";
  panel.hidden = true;
  panel.innerHTML = '<div class="arena-curriculum-controls">' +
    '<label>Chapter <select aria-label="ARENA chapter"></select></label>' +
    '<button type="button" class="arena-curriculum-overview">Section map</button>' +
    '<button type="button" class="arena-curriculum-fit">Fit</button></div>' +
    '<p class="arena-curriculum-caption">Follow ARENA’s section and exercise order. ' +
    'Arrows show sequence; preparation links show supporting concepts.</p>' +
    '<div class="arena-curriculum-layout"><div class="arena-curriculum-canvas" aria-label="Curriculum graph"></div>' +
    '<aside class="arena-curriculum-detail" aria-live="polite"></aside></div>';
  page.append(panel);
  const detail = panel.querySelector("aside");
  const chapter = panel.querySelector("select");
  let data, cy, loading, selectedSection = null;
  const text = (tag, value, parent = detail) => {
    const el = document.createElement(tag);
    el.textContent = value;
    parent.append(el);
    return el;
  };
  const button = (label, action, parent = detail) => {
    const el = text("button", label, parent);
    el.type = "button";
    el.onclick = action;
    return el;
  };
  const setMode = async (arena) => {
    skills.hidden = arena;
    panel.hidden = !arena;
    switcher.querySelectorAll("button").forEach((el, i) => el.setAttribute("aria-pressed", String(i === (arena ? 1 : 0))));
    if (!arena) {
      window.deltaConceptGraphCy?.()?.resize();
      return;
    }
    try {
      await load();
      if (!panel.hidden) draw(selectedSection);
    } catch (err) {
      detail.replaceChildren();
      text("p", `Curriculum unavailable: ${err.message}`);
      button("Retry", () => setMode(true));
    }
  };
  switcher.children[0].onclick = () => setMode(false);
  switcher.children[1].onclick = () => setMode(true);
  const load = async () => {
    if (data) return;
    if (loading) return loading;
    loading = (async () => {
      const response = await fetch("lessons/notebooks/arena-curriculum.json");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const graph = await response.json();
      if (!Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) throw new Error("Invalid curriculum index");
      data = graph;
      const chapters = [...new Set(data.nodes.filter((n) => n.kind === "section").map((n) => n.chapter))];
      chapters.forEach((name) => chapter.add(new Option(name, name)));
    })().finally(() => { loading = null; });
    return loading;
  };
  const openExercise = (node) => window.ArenaNotebook.open(node.section, node.id);
  const show = (node) => {
    detail.replaceChildren();
    text("h2", node.title.replace(/`/g, ""));
    if (node.description) text("p", node.description);
    if (node.kind === "section") {
      text("p", `${node.exercise_count} exercise tasks · setup and earlier work stay in the notebook.`);
      button("Explore exercises", () => draw(node.section));
      button("Open lesson", () => window.ArenaNotebook.open(node.section));
    } else if (node.kind === "exercise") {
      button("Open exercise →", () => openExercise(node));
      const previous = data.edges.find((e) => e.target === node.id && e.relation === "exercise_order");
      const prev = data.nodes.find((n) => n.id === previous?.source);
      if (prev?.kind === "exercise") button(`Previous: ${prev.title.replace(/`/g, "")}`, () => openExercise(prev));
      const topicEdge = data.edges.find((e) => e.target === node.id && e.relation === "topic_exercise");
      const topic = data.nodes.find((n) => n.id === topicEdge?.source);
      if (topic) {
        text("h3", "Lesson context");
        button(topic.title, () => show(topic));
      }
      const prep = data.edges.filter((e) => e.target === node.id && ["prepares", "exercises_concept"].includes(e.relation));
      if (prep.length) {
        text("h3", "Preparation");
        const list = document.createElement("ul");
        detail.append(list);
        prep.forEach((edge) => {
          const concept = data.nodes.find((n) => n.id === edge.source);
          if (!concept) return;
          const li = document.createElement("li");
          list.append(li);
          button(concept.title, () => show(concept), li);
        });
      } else text("p", "Use the preceding lesson for preparation. Dedicated drill mapping is not available yet.");
    } else if (node.kind === "topic") {
      text("p", "Authored lesson topic containing these exercises.");
      const children = data.edges.filter((e) => e.source === node.id && e.relation === "topic_exercise")
        .map((e) => data.nodes.find((n) => n.id === e.target)).filter(Boolean);
      children.forEach((exercise) => button(exercise.title.replace(/`/g, ""), () => openExercise(exercise)));
    } else if (node.kind === "kc") {
      button("Open preparation lesson", async () => {
        await setMode(false);
        window.deltaFocusConceptGraphKc?.(node.kc);
      });
    } else {
      text("p", "Supporting concept from ARENA exercise annotations. This node does not award mastery.");
    }
  };
  const draw = (section) => {
    if (!window.cytoscape) throw new Error("Graph renderer unavailable");
    if (window.cytoscapeDagre) window.cytoscape.use(window.cytoscapeDagre);
    selectedSection = section;
    let nodes;
    if (section) {
      nodes = data.nodes.filter((n) => n.section === section);
      // Show the exercise sequence first. Concepts appear in the detail pane
      // on selection instead of turning every section into a dense hairball.
    } else nodes = data.nodes.filter((n) => n.kind === "section" && n.chapter === chapter.value);
    const ids = new Set(nodes.map((n) => n.id));
    const edges = data.edges.filter((e) => ids.has(e.source) && ids.has(e.target));
    cy?.destroy();
    cy = window.cytoscape({
      container: panel.querySelector(".arena-curriculum-canvas"),
      elements: [
        ...nodes.map((n) => ({ data: { ...n, label: n.title.replace(/`/g, "") } })),
        ...edges.map((e, i) => ({ data: { ...e, id: `sequence-${i}` } })),
      ],
      style: [
        { selector: "node", style: { label: "data(label)", "text-wrap": "wrap", "text-max-width": 180,
          "font-size": 13, width: 200, height: 62, shape: "roundrectangle",
          "background-color": "#cbe5f6", color: "#172b3a", "text-valign": "center" } },
        { selector: 'node[kind = "exercise"]', style: { "background-color": "#e0e9dc" } },
        { selector: 'node[kind = "topic"]', style: { "background-color": "#f2dfb6", width: 180, height: 52 } },
        { selector: ":selected", style: { "border-width": 3, "border-color": "#2463a1" } },
        { selector: "edge", style: { width: 2, "line-color": "#8b9bab", "target-arrow-color": "#8b9bab",
          "target-arrow-shape": "triangle", "curve-style": "bezier" } },
      ],
      layout: { name: window.cytoscapeDagre ? "dagre" : "cose", rankDir: "TB", nodeSep: 24, rankSep: 28, fit: true, padding: 28 },
      minZoom: 0.08, maxZoom: 2,
    });
    cy.on("tap", "node", (event) => show(event.target.data()));
    detail.replaceChildren();
    if (section) {
      const parent = nodes.find((n) => n.kind === "section");
      text("h2", parent?.title || "Exercises");
      text("p", "Choose an exercise. Earlier prompts, answers, and setup remain above it.");
      const list = document.createElement("ol");
      detail.append(list);
      nodes.filter((n) => n.kind === "exercise").forEach((node) => {
        const item = document.createElement("li");
        list.append(item);
        button(node.title.replace(/`/g, ""), () => { cy.$id(node.id).select(); show(node); }, item);
      });
    } else {
      text("h2", chapter.value);
      text("p", "Choose a section to see its exercises or open its lesson.");
      nodes.forEach((node) => button(node.title, () => show(node)));
      if (chapter.selectedIndex === 0) button("Prepare with Python, NumPy, einops and tensors", () => setMode(false));
    }
  };
  chapter.onchange = () => draw(null);
  panel.querySelector(".arena-curriculum-overview").onclick = () => data && draw(null);
  panel.querySelector(".arena-curriculum-fit").onclick = () => cy?.fit(undefined, 28);
  let lastWidth = 0;
  new ResizeObserver(() => {
    const width = panel.clientWidth;
    if (!width || !cy) return;
    cy.resize();
    if (width !== lastWidth) cy.fit(undefined, 28);
    lastWidth = width;
  }).observe(panel);
})();
