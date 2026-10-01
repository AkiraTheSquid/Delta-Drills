/* ================================================================
   LESSON FIGURES — interactive diagrams inside lesson prose.

   WHAT THIS IS
     A lesson page is markdown. Some ideas read better as a picture you can
     poke than as a paragraph: Seth on math.barycentric-coordinates
     (2026-09-29, lesson feedback on q50018): "seeing a generated svg graph
     would be helpful for this. it should look good. it will be for viewing
     the triangle and being able to click and see the values of the points".

   HOW A PAGE ASKS FOR ONE
     A fence whose info string is `figure <name>`:

         ```figure barycentric
         Interactive figure: drag P around triangle ABC and read its weights.
         ```

     The body is the fallback text for every surface that is not this app
     (the Colab export keeps any non-`python` fence as markdown, so it prints
     there as a short note) — the figure itself never reads it.

   HOW IT GETS ON SCREEN — owns no renderer
     Same trick as jargon.js: both lesson renderers carry the info string on
     the rendered element (`<pre data-fence="figure barycentric">` —
     practice/lessons.js for every fence, concept-graph/lesson-graph.js for
     `figure` fences only), and a debounced MutationObserver here swaps each
     such <pre> for its figure. A name with no builder below is left as the
     plain text block, which is the honest fallback.

     It is never a runnable cell: practice/notebook.js runs `python` fences
     only, and the validators execute nothing else.
   ================================================================ */

(function initLessonFigures() {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const round2 = (x) => Math.round(x * 100) / 100;
  // -0.00 reads as a sign error to someone checking a weight's sign.
  const fmt = (x) => {
    const r = round2(x);
    return (Object.is(r, -0) ? 0 : r).toFixed(2);
  };

  function el(tag, attrs, parent) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, String(v));
    if (parent) parent.appendChild(node);
    return node;
  }

  /* ---------------- barycentric: the maths, kept pure ------------------ */

  /* u, v with P = A + u(B − A) + v(C − A), by Cramer's rule on the 2×2
     system of the two edge columns. The figure's triangle is fixed and
     nondegenerate, so the determinant is never zero. */
  function barycentric(p, a, b, c) {
    const e1x = b[0] - a[0], e1y = b[1] - a[1];
    const e2x = c[0] - a[0], e2y = c[1] - a[1];
    const rx = p[0] - a[0], ry = p[1] - a[1];
    const det = e1x * e2y - e2x * e1y;
    const u = (rx * e2y - e2x * ry) / det;
    const v = (e1x * ry - rx * e1y) / det;
    return { u, v, w: [1 - u - v, u, v] };
  }

  function pointAt(u, v, a, b, c) {
    return [a[0] + u * (b[0] - a[0]) + v * (c[0] - a[0]),
            a[1] + u * (b[1] - a[1]) + v * (c[1] - a[1])];
  }

  /* The sentence under the numbers: inside, on which edge, or which bound
     failed. Judged on the ROUNDED weights, so the words agree with the
     digits printed beside them. */
  function verdict(u, v) {
    const U = round2(u), V = round2(v), W = round2(1 - U - V);
    const fails = [];
    if (U < 0) fails.push("u < 0, so the weight on B is negative");
    if (V < 0) fails.push("v < 0, so the weight on C is negative");
    if (W < 0) fails.push("u + v > 1, so the weight on A is negative");
    if (fails.length) {
      const para = U <= 1 && V <= 1 && U >= 0 && V >= 0;
      return {
        inside: false,
        text: "Outside: " + fails.join("; ") + "." +
          (para ? " It is still in the parallelogram, which only bounds u and v one at a time." : ""),
      };
    }
    const on = [];
    if (V === 0) on.push("edge AB (v = 0)");
    if (U === 0) on.push("edge AC (u = 0)");
    if (W === 0) on.push("edge BC (u + v = 1)");
    if (on.length >= 2) {
      const name = U === 0 && V === 0 ? "A" : V === 0 ? "B" : "C";
      return { inside: true, text: `On vertex ${name}: all the weight is on ${name}, where ${on.join(" and ")} meet.` };
    }
    if (on.length === 1) return { inside: true, text: "On the boundary: " + on[0] + ". Boundaries count as inside." };
    return { inside: true, text: "Inside: all three weights are nonnegative, and they add to one." };
  }

  /* ---------------- barycentric: the figure ---------------------------- */

  // Math coordinates (y up). A non-right, non-isosceles triangle so no
  // weight looks special by symmetry; the parallelogram A, B, B+C−A, C fits.
  const A = [0.5, 0.5], B = [5.3, 1.1], C = [1.7, 3.9];
  const SCALE = 58, PAD = 26;
  const X0 = -0.2, X1 = 6.9, Y0 = -0.1, Y1 = 4.9;
  const W_PX = (X1 - X0) * SCALE + 2 * PAD;
  const H_PX = (Y1 - Y0) * SCALE + 2 * PAD;
  const sx = (x) => PAD + (x - X0) * SCALE;
  const sy = (y) => PAD + (Y1 - y) * SCALE;
  const fromScreen = (px, py) => [X0 + (px - PAD) / SCALE, Y1 - (py - PAD) / SCALE];
  const P = (pt) => `${sx(pt[0]).toFixed(1)},${sy(pt[1]).toFixed(1)}`;

  const PRESETS = [
    { label: "u = v = 0.6 (the example below)", u: 0.6, v: 0.6 },
    { label: "centre (⅓, ⅓)", u: 1 / 3, v: 1 / 3 },
    { label: "on edge BC", u: 0.5, v: 0.5 },
    { label: "vertex B", u: 1, v: 0 },
  ];

  let figureCount = 0;

  function buildBarycentric() {
    const fig = document.createElement("figure");
    fig.className = "dd-figure dd-figure-bary";

    const svg = el("svg", {
      viewBox: `0 0 ${W_PX.toFixed(0)} ${H_PX.toFixed(0)}`,
      role: "img",
      tabindex: "0",
      "aria-label": "Triangle ABC. Click or drag to move point P; arrow keys nudge it.",
    });
    fig.appendChild(svg);

    // Marker ids are document-global: the same page can be mounted in the
    // practice column and the graph pane at once, so each figure gets its own.
    const uid = "dd-bary-" + (++figureCount);
    const defs = el("defs", {}, svg);
    for (const [id, cls] of [[uid + "-ah-b", "ink-b"], [uid + "-ah-c", "ink-c"]]) {
      const m = el("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 4.2, markerHeight: 4.2, orient: "auto-start-reverse" }, defs);
      el("path", { d: "M0,0 L10,5 L0,10 z", class: cls }, m);
    }

    const BC = [B[0] + C[0] - A[0], B[1] + C[1] - A[1]];
    el("polygon", { points: [A, B, BC, C].map(P).join(" "), class: "para" }, svg);
    // Labelled high in the half the sum bound cuts away, clear of where the
    // lesson's own example (u = v = 0.6) puts P.
    const cut = pointAt(0.8, 0.75, A, B, C);
    const paraLabel = el("text", { x: sx(cut[0]) - 20, y: sy(cut[1]), class: "para-label", "text-anchor": "middle" }, svg);
    paraLabel.textContent = "0 ≤ u ≤ 1, 0 ≤ v ≤ 1 alone";
    const paraLabel2 = el("text", { x: sx(cut[0]) - 20, y: sy(cut[1]) + 14, class: "para-label", "text-anchor": "middle" }, svg);
    paraLabel2.textContent = "would also allow this half";
    el("polygon", { points: [A, B, C].map(P).join(" "), class: "tri" }, svg);

    // Edge names at each edge's midpoint, pushed outward from the triangle.
    const centroid = [(A[0] + B[0] + C[0]) / 3, (A[1] + B[1] + C[1]) / 3];
    const edgeLabel = (p, q, text) => {
      const mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2;
      const dx = mx - centroid[0], dy = my - centroid[1];
      const n = Math.hypot(dx, dy) || 1;
      const t = el("text", { x: sx(mx + (dx / n) * 0.34), y: sy(my + (dy / n) * 0.34) + 4, class: "edge-label", "text-anchor": "middle" }, svg);
      t.textContent = text;
    };
    edgeLabel(A, B, "v = 0");
    edgeLabel(A, C, "u = 0");
    edgeLabel(B, C, "u + v = 1");

    // The edge walk: from A along u(B − A), then v(C − A), arriving at P.
    const legU = el("line", { class: "leg leg-u", "marker-end": `url(#${uid}-ah-b)` }, svg);
    const legV = el("line", { class: "leg leg-v", "marker-end": `url(#${uid}-ah-c)` }, svg);

    const vertex = (pt, name, cls, dx, dy) => {
      el("circle", { cx: sx(pt[0]), cy: sy(pt[1]), r: 5, class: `vtx ${cls}` }, svg);
      const t = el("text", { x: sx(pt[0]) + dx, y: sy(pt[1]) + dy, class: `vtx-label ${cls}`, "text-anchor": "middle" }, svg);
      t.textContent = name;
    };
    vertex(A, "A", "ink-a", -12, 16);
    vertex(B, "B", "ink-b", 14, 14);
    vertex(C, "C", "ink-c", -14, -6);

    const dot = el("circle", { r: 7, class: "pt" }, svg);
    const ptLabel = el("text", { class: "pt-label" }, svg);
    ptLabel.textContent = "P";

    const read = document.createElement("div");
    read.className = "dd-bary-read";
    read.innerHTML =
      '<div class="dd-bary-uv"><span>u = <b data-k="u"></b></span><span>v = <b data-k="v"></b></span></div>' +
      '<div class="dd-bary-weights">' +
      ["a", "b", "c"].map((k) =>
        `<div class="dd-bary-w" data-w="${k}"><span class="dd-bary-name ink-${k}">${k.toUpperCase()}</span>` +
        `<span class="dd-bary-bar"><span class="dd-bary-fill"></span></span><b class="dd-bary-num"></b></div>`).join("") +
      "</div>" +
      '<div class="dd-bary-sum"></div>' +
      '<div class="dd-bary-verdict" role="status" aria-live="polite"></div>';
    fig.appendChild(read);

    const chips = document.createElement("div");
    chips.className = "dd-bary-presets";
    for (const p of PRESETS) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = p.label;
      b.addEventListener("click", () => set(p.u, p.v));
      chips.appendChild(b);
    }
    fig.appendChild(chips);

    const cap = document.createElement("figcaption");
    cap.textContent = "Click or drag anywhere to move P. The arrows are the walk from A: u of the way along AB, then v of the way along AC.";
    fig.appendChild(cap);

    let cur = { u: 0.25, v: 0.35 };
    function set(u, v) {
      cur = { u, v };
      const p = pointAt(u, v, A, B, C);
      const mid = pointAt(u, 0, A, B, C);
      legU.setAttribute("x1", sx(A[0])); legU.setAttribute("y1", sy(A[1]));
      legU.setAttribute("x2", sx(mid[0])); legU.setAttribute("y2", sy(mid[1]));
      legV.setAttribute("x1", sx(mid[0])); legV.setAttribute("y1", sy(mid[1]));
      legV.setAttribute("x2", sx(p[0])); legV.setAttribute("y2", sy(p[1]));
      legU.style.display = Math.abs(u) < 0.02 ? "none" : "";
      legV.style.display = Math.abs(v) < 0.02 ? "none" : "";
      dot.setAttribute("cx", sx(p[0])); dot.setAttribute("cy", sy(p[1]));
      ptLabel.setAttribute("x", sx(p[0]) + 10); ptLabel.setAttribute("y", sy(p[1]) - 10);

      const w = [1 - u - v, u, v];
      const verdictNow = verdict(u, v);
      fig.classList.toggle("is-outside", !verdictNow.inside);
      read.querySelector('[data-k="u"]').textContent = fmt(u);
      read.querySelector('[data-k="v"]').textContent = fmt(v);
      ["a", "b", "c"].forEach((k, i) => {
        const row = read.querySelector(`[data-w="${k}"]`);
        const val = round2(w[i]);
        row.classList.toggle("neg", val < 0);
        row.querySelector(".dd-bary-num").textContent = fmt(w[i]);
        // Bars run from a centre zero line: right for positive, left for
        // negative, full half-width at |w| = 1.5.
        const fill = row.querySelector(".dd-bary-fill");
        const frac = Math.min(Math.abs(w[i]) / 1.5, 1) * 50;
        fill.style.width = frac.toFixed(1) + "%";
        fill.style.left = (val < 0 ? 50 - frac : 50).toFixed(1) + "%";
      });
      read.querySelector(".dd-bary-sum").textContent =
        `P = ${fmt(w[0])}·A + ${fmt(w[1])}·B + ${fmt(w[2])}·C   (weights sum to ${fmt(w[0] + w[1] + w[2])})`;
      read.querySelector(".dd-bary-verdict").textContent = verdictNow.text;
    }

    function fromEvent(ev) {
      const box = svg.getBoundingClientRect();
      const px = ((ev.clientX - box.left) / box.width) * W_PX;
      const py = ((ev.clientY - box.top) / box.height) * H_PX;
      const [x, y] = fromScreen(px, py);
      const { u, v } = barycentric([x, y], A, B, C);
      // Snap to the hundredth that is printed, so an edge or a vertex is
      // reachable by hand and the verdict matches the digits.
      set(round2(u), round2(v));
    }
    let dragging = false;
    svg.addEventListener("pointerdown", (ev) => {
      dragging = true;
      try { svg.setPointerCapture(ev.pointerId); } catch (_) { /* old browsers */ }
      fromEvent(ev);
      ev.preventDefault();
    });
    svg.addEventListener("pointermove", (ev) => { if (dragging) fromEvent(ev); });
    const stop = () => { dragging = false; };
    svg.addEventListener("pointerup", stop);
    svg.addEventListener("pointercancel", stop);
    svg.addEventListener("keydown", (ev) => {
      const step = ev.shiftKey ? 0.1 : 0.01;
      const d = { ArrowRight: [step, 0], ArrowLeft: [-step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] }[ev.key];
      if (!d) return;
      ev.preventDefault();
      set(round2(cur.u + d[0]), round2(cur.v + d[1]));
    });

    set(cur.u, cur.v);
    return fig;
  }

  const BUILDERS = { barycentric: buildBarycentric };

  /* ---------------- mounting ------------------------------------------- */

  function mountAll(root) {
    (root || document).querySelectorAll('pre[data-fence^="figure"]').forEach((pre) => {
      const name = (pre.getAttribute("data-fence") || "").split(/\s+/)[1];
      const build = BUILDERS[name];
      if (!build || !pre.parentNode) return;
      pre.replaceWith(build());
    });
  }

  if (typeof document !== "undefined" && typeof MutationObserver !== "undefined") {
    let queued = false;
    const schedule = () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; mountAll(document); });
    };
    // A figure's own readout rewrites itself on every drag frame; those
    // mutations cannot add a fence, so they do not trigger a rescan.
    const outsideFigures = (r) => !(r.target instanceof Element && r.target.closest(".dd-figure"));
    new MutationObserver((records) => { if (records.some(outsideFigures)) schedule(); })
      .observe(document.documentElement, { childList: true, subtree: true });
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => mountAll(document));
    else mountAll(document);
  }

  /* Other figures live in their own files (lesson-figure-*.js, loaded after
     this one) and register into BUILDERS with these shared helpers. */
  const api = { barycentric, pointAt, verdict, mountAll, BUILDERS, el, fmt, round2 };
  if (typeof window !== "undefined") window.LessonFigures = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})();
