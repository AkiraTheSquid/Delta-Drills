/* ================================================================
   RICH CELL OUTPUT — draw a kernel's display_data under a cell
   ================================================================

   A notebook cell can answer with more than text: `fig.show()` is a plotly
   figure, `display(df)` is an HTML table, matplotlib is a PNG. The Modal
   kernel (backend app/modal_kernel.py) forwards those as mimebundles in
   `outputs[]`; the fork kernel has no display channel and sends none. This
   is the one place they are turned into DOM, for both notebook surfaces
   (notebook-view.js and arena-notebook.js).

   One bundle → one node, by the richest mimetype the page can draw. Plotly's
   own JSON wins over its HTML twin: the HTML form carries a <script> that
   would need eval, the JSON form is data for a library loaded once, lazily,
   from the CDN the rest of index.html already trusts (cdn.jsdelivr.net).

   HTML and SVG output come from the learner's own kernel — the trust Jupyter
   extends it. Active content is still stripped (scripts, on* handlers,
   javascript: URLs): a package the learner pip-installed is an author too,
   and a table does not need any of it.
   ================================================================ */

const DeltaCellOutputs = (() => {
  const PLOTLY_SRC = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js";
  let plotlyLoading = null;

  const _plotly = () => {
    if (window.Plotly) return Promise.resolve(window.Plotly);
    if (!plotlyLoading) {
      plotlyLoading = new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = PLOTLY_SRC;
        s.onload = () => resolve(window.Plotly);
        s.onerror = () => {
          plotlyLoading = null;
          reject(new Error("plotly.js did not load"));
        };
        document.head.appendChild(s);
      });
    }
    return plotlyLoading;
  };

  /* Drop what can run: <script>, inline handlers, javascript: URLs. Tables,
     styles, images and SVG shapes survive untouched. */
  const _sanitized = (markup, mime) => {
    const doc = new DOMParser().parseFromString(markup, mime);
    doc.querySelectorAll("script").forEach((s) => s.remove());
    doc.querySelectorAll("*").forEach((el) => {
      for (const attr of [...el.attributes]) {
        const name = attr.name.toLowerCase();
        const value = String(attr.value).replace(/\s+/g, "").toLowerCase();
        if (name.startsWith("on") || ((name === "href" || name === "src" || name === "xlink:href") && value.startsWith("javascript:"))) {
          el.removeAttribute(attr.name);
        }
      }
    });
    const root = mime === "image/svg+xml" ? doc.documentElement : doc.body;
    return root ? root.innerHTML !== undefined && mime !== "image/svg+xml" ? root.innerHTML : root.outerHTML : "";
  };

  const _note = (text) => {
    const p = document.createElement("div");
    p.className = "nbv-rich-note";
    p.textContent = text;
    return p;
  };

  const _one = (bundle) => {
    const node = document.createElement("div");
    node.className = "nbv-rich-item";
    const plotly = bundle["application/vnd.plotly.v1+json"];
    if (plotly) {
      let fig;
      try {
        fig = typeof plotly === "string" ? JSON.parse(plotly) : plotly;
      } catch (err) {
        node.appendChild(_note(`(figure not drawn: ${err.message})`));
        return node;
      }
      node.classList.add("nbv-rich-plotly");
      _plotly()
        .then((P) => P.newPlot(node, fig.data || [], fig.layout || {},
          Object.assign({ responsive: true, displayModeBar: false }, fig.config || {})))
        .catch((err) => node.replaceChildren(_note(`(figure not drawn: ${err.message})`)));
      return node;
    }
    if (bundle["text/html"]) {
      node.innerHTML = _sanitized(bundle["text/html"], "text/html");
      return node;
    }
    if (bundle["image/png"]) {
      const img = document.createElement("img");
      img.src = `data:image/png;base64,${String(bundle["image/png"]).replace(/\s+/g, "")}`;
      img.alt = bundle["text/plain"] || "figure";
      node.appendChild(img);
      return node;
    }
    if (bundle["image/svg+xml"]) {
      node.innerHTML = _sanitized(bundle["image/svg+xml"], "image/svg+xml");
      return node;
    }
    if (bundle["text/markdown"] && window.LessonGate?.renderMarkdown) {
      node.innerHTML = window.LessonGate.renderMarkdown(bundle["text/markdown"]);
      return node;
    }
    if (bundle["text/plain"] != null) {
      const pre = document.createElement("pre");
      pre.className = "nbv-rich-plain";
      pre.textContent = bundle["text/plain"];
      node.appendChild(pre);
      return node;
    }
    return null;
  };

  /* Append every drawable bundle under `container` (a cell's .nbv-out). The
     caller has already written the cell's text; this never clears it. */
  const render = (container, outputs) => {
    if (!container || !Array.isArray(outputs) || !outputs.length) return 0;
    const wrap = document.createElement("div");
    wrap.className = "nbv-rich";
    let drawn = 0;
    for (const bundle of outputs) {
      // One bad bundle must not turn a cell that ran into a cell that failed.
      let node = null;
      try {
        node = bundle && _one(bundle);
      } catch (err) {
        node = _note(`(output not drawn: ${err.message})`);
      }
      if (node) {
        wrap.appendChild(node);
        drawn += 1;
      }
    }
    if (drawn) container.appendChild(wrap);
    return drawn;
  };

  return { render };
})();

window.DeltaCellOutputs = DeltaCellOutputs;
