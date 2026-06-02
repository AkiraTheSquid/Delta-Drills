/* ================================================================
   CONCEPT GRAPH VIZ — interactive Cytoscape.js embed for the
   Knowledge Graph tab. Loads concept-graph/graph-viz.json (exported by
   This-Directory-Only/scripts/export_graph_viz.py), colours nodes by
   family, draws the structural prerequisite backbone bright red and the
   encompassing links faint/dashed.

   Built on demand via window.deltaInitConceptGraph() — called by app.js
   switchTab() when the Knowledge Graph tab opens. (Cytoscape can't size
   its container while the page is display:none, so we defer the build
   until the tab is actually visible.)
   ================================================================ */
(() => {
  "use strict";

  const FAMILY_COLORS = {
    "Fundamentals": "#f4d35e",
    "CNNs": "#4cc9f0",
    "Ray Tracing": "#b5179e",
    "Backprop & Autograd": "#80ed99",
    "Optimization & Training": "#ff7b54",
    "Generative": "#c77dff",
    "Distributed": "#ff5d8f",
    "Other": "#8a8a99",
  };
  const ACCENT = "#e3212c"; // structural prerequisite edges

  let cy = null;
  let building = false;

  const buildLegend = (families) => {
    const el = document.getElementById("concept-graph-legend");
    if (!el) return;
    el.innerHTML = families
      .map(
        (f) =>
          `<span class="cg-legend-item"><span class="cg-legend-dot" style="background:${
            FAMILY_COLORS[f] || FAMILY_COLORS.Other
          }"></span>${f}</span>`
      )
      .join("");
  };

  const setInfo = (html) => {
    const el = document.getElementById("concept-graph-info");
    if (el) el.innerHTML = html;
  };

  async function build() {
    if (cy || building) return;
    const container = document.getElementById("concept-graph-cy");
    if (!container) return;
    if (typeof cytoscape === "undefined") return; // CDN not loaded yet
    building = true;

    let layoutName = "cose";
    try {
      if (window.cytoscapeFcose) {
        cytoscape.use(window.cytoscapeFcose);
        layoutName = "fcose";
      }
    } catch (_) {
      /* already registered */
    }

    let data;
    try {
      const res = await fetch("concept-graph/graph-viz.json?v=4");
      data = await res.json();
    } catch (e) {
      setInfo("Couldn't load the graph data.");
      building = false;
      return;
    }

    const families = [...new Set(data.nodes.map((n) => n.family))].sort();
    buildLegend(families);

    const elements = [
      ...data.nodes.map((n) => ({
        data: { id: n.id, label: n.label, topic: n.topic, family: n.family },
      })),
      ...data.edges.map((e, i) => ({
        data: { id: `e${i}`, source: e.source, target: e.target, enc: e.enc ? 1 : 0 },
        classes: e.enc ? "enc" : "prereq",
      })),
    ];

    cy = cytoscape({
      container,
      elements,
      wheelSensitivity: 0.2,
      style: [
        {
          selector: "node",
          style: {
            "background-color": (n) => FAMILY_COLORS[n.data("family")] || FAMILY_COLORS.Other,
            width: 14,
            height: 14,
            label: "",
            "border-width": 0,
            "transition-property": "opacity, width, height",
            "transition-duration": "120ms",
          },
        },
        {
          selector: "edge",
          style: { "curve-style": "straight", "target-arrow-shape": "none", opacity: 0.9 },
        },
        {
          selector: "edge.enc",
          style: { width: 0.8, "line-color": "#5a5a86", "line-style": "dashed", opacity: 0.55 },
        },
        {
          selector: "edge.prereq",
          style: {
            width: 2,
            "line-color": ACCENT,
            "target-arrow-shape": "triangle",
            "target-arrow-color": ACCENT,
            "arrow-scale": 0.7,
          },
        },
        { selector: ".faded", style: { opacity: 0.08 } },
        {
          selector: "node.hl",
          style: { width: 22, height: 22, "border-width": 2, "border-color": "#ffffff", "z-index": 99 },
        },
        { selector: "edge.hl", style: { opacity: 1, width: 2.5, "line-color": "#ffffff", "line-style": "solid" } },
        {
          selector: "node.named",
          style: { label: "data(label)", color: "#e0e0e0", "font-size": 9, "text-wrap": "wrap", "text-max-width": "90px", "text-background-color": "#16213e", "text-background-opacity": 0.7, "text-background-padding": 2 },
        },
      ],
      layout: {
        name: layoutName,
        animate: false,
        randomize: true,
        quality: "default",
        nodeRepulsion: 6500,
        idealEdgeLength: 55,
        nodeSeparation: 80,
        packComponents: true,
        fit: true,
        padding: 28,
      },
    });

    cy.on("tap", "node", (evt) => {
      const n = evt.target;
      const hood = n.closedNeighborhood();
      cy.elements().removeClass("hl named").addClass("faded");
      hood.removeClass("faded").addClass("hl");
      hood.nodes().addClass("named");
      setInfo(`<strong>${n.data("label")}</strong><span class="cg-info-topic">${n.data("topic")}</span>`);
    });
    cy.on("tap", (evt) => {
      if (evt.target === cy) {
        cy.elements().removeClass("faded hl named");
        setInfo("Click any skill to trace what it connects to.");
      }
    });

    setInfo("Click any skill to trace what it connects to.");

    const fitBtn = document.getElementById("concept-graph-fit");
    if (fitBtn) fitBtn.onclick = () => cy.fit(undefined, 28);
    const toggle = document.getElementById("concept-graph-toggle-enc");
    if (toggle)
      toggle.onchange = () => cy.edges(".enc").style("display", toggle.checked ? "element" : "none");

    building = false;
  }

  // Called when the Knowledge Graph tab opens. Builds the first time (now that
  // the container is visible); on later opens just re-fits to the new size.
  window.deltaInitConceptGraph = function () {
    if (cy) {
      cy.resize();
      cy.fit(undefined, 28);
      return;
    }
    let tries = 0;
    const tick = () => {
      build();
      if (!cy && tries++ < 80) setTimeout(tick, 150);
    };
    tick();
  };
})();
