/* ================================================================
   LESSON FIGURE `independence` — two triangle edges, dependent or not.

   Seth on math.barycentric-coordinates (2026-10-01, lesson feedback on
   q50019): the "nondegenerate triangle" popup linked to linear combinations,
   but that page "doesn't have an obvious connection to nondegenerate
   triangle … say the triangle is nondegenerate if the vectors making up its
   edges are linearly independent … degenerate if their cross product is
   zero … making it such that you can change the angle … to see linear
   dependence versus independence."

   A KP asks for it with a ```figure independence``` fence (see
   lesson-figures.js for how fences become figures). Edge B − A is fixed; the
   learner drags C, or turns the angle slider, and reads the 2D cross product,
   the triangle's area and whether the edges are independent.

   Loaded AFTER lesson-figures.js; registers into its BUILDERS.
   ================================================================ */

(function initIndependenceFigure() {
  "use strict";

  const LF = typeof window !== "undefined" ? window.LessonFigures : null;
  const shared = LF || (typeof module !== "undefined" ? require("./lesson-figures.js") : null);
  const { el, fmt } = shared;

  /* ---------------- the maths, kept pure ------------------------------- */

  // Exact zero only: a slider at 0° or 180° sets C − A to an exact multiple
  // of B − A (no trig), and a drag snaps to a 0.1 grid, so dependence is
  // reachable by hand and never claimed for a merely thin triangle.
  const EPS = 1e-9;

  function cross(e1, e2) {
    return e1[0] * e2[1] - e1[1] * e2[0];
  }

  /* What the two edges are: independent (a real triangle), dependent (flat),
     or the second edge is zero (C on A). `k` is the multiple when dependent. */
  function classify(e1, e2) {
    const c = cross(e1, e2);
    const area = Math.abs(c) / 2;
    if (Math.hypot(e2[0], e2[1]) < EPS) return { kind: "zero", cross: c, area, k: 0 };
    // B − A itself zero: B sits on A, flat with no multiple to name.
    if (Math.hypot(e1[0], e1[1]) < EPS) return { kind: "zero-b", cross: c, area, k: 0 };
    if (Math.abs(c) < EPS) {
      const k = (e1[0] * e2[0] + e1[1] * e2[1]) / (e1[0] * e1[0] + e1[1] * e1[1]);
      return { kind: "dependent", cross: c, area: 0, k };
    }
    return { kind: "independent", cross: c, area, k: null };
  }

  function verdictText(r) {
    if (r.kind === "zero") {
      return "Degenerate: C − A is the zero vector, so C sits on A and the " +
        "triangle is just the segment AB. A zero vector is always dependent. Area 0.";
    }
    if (r.kind === "zero-b") {
      return "Degenerate: B − A is the zero vector, so B sits on A and the " +
        "triangle is just the segment AC. A zero vector is always dependent. Area 0.";
    }
    if (r.kind === "dependent") {
      return `Degenerate: C − A = ${fmt(r.k)}·(B − A), so A, B and C sit on one line. ` +
        "The edges are linearly dependent, they span only that line, and the " +
        "cross product and the area are both 0.";
    }
    return "Nondegenerate: neither edge is a multiple of the other, so they are " +
      "linearly independent and span the whole plane. The cross product is not " +
      `0, and the triangle has area ${fmt(r.area)}.`;
  }

  /* ---------------- the figure ----------------------------------------- */

  const A = [0, 0];
  const E1 = [3, 1];                       // B − A, fixed
  const B = [A[0] + E1[0], A[1] + E1[1]];
  const SCALE = 52, PAD = 20;
  const X0 = -3.6, X1 = 4.6, Y0 = -2.6, Y1 = 3.4;
  const W_PX = (X1 - X0) * SCALE + 2 * PAD;
  const H_PX = (Y1 - Y0) * SCALE + 2 * PAD;
  const sx = (x) => PAD + (x - X0) * SCALE;
  const sy = (y) => PAD + (Y1 - y) * SCALE;
  const fromScreen = (px, py) => [X0 + (px - PAD) / SCALE, Y1 - (py - PAD) / SCALE];
  const P = (pt) => `${sx(pt[0]).toFixed(1)},${sy(pt[1]).toFixed(1)}`;
  const E1_ANGLE = Math.atan2(E1[1], E1[0]);
  const E1_LEN = Math.hypot(E1[0], E1[1]);
  const snap = (x) => Math.round(x * 10) / 10;
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

  /* C − A at `deg` degrees counter-clockwise from B − A, length `len`. The
     two flat angles are built as exact multiples of B − A so the cross
     product is exactly zero there, not 1e-16. */
  function edgeAtAngle(deg, len) {
    const d = ((deg % 360) + 360) % 360;
    if (d === 0 || d === 180) {
      const k = (d === 0 ? 1 : -1) * len / E1_LEN;
      return [k * E1[0], k * E1[1]];
    }
    const t = E1_ANGLE + (d * Math.PI) / 180;
    return [len * Math.cos(t), len * Math.sin(t)];
  }

  const PRESETS = [
    { label: "60° apart", deg: 60 },
    { label: "nearly flat (5°)", deg: 5 },
    { label: "flat (0°)", deg: 0 },
    { label: "flat, pointing back (180°)", deg: 180 },
  ];

  let figureCount = 0;

  function buildIndependence() {
    const fig = document.createElement("figure");
    fig.className = "dd-figure dd-figure-indep";

    const svg = el("svg", {
      viewBox: `0 0 ${W_PX.toFixed(0)} ${H_PX.toFixed(0)}`,
      role: "img",
      tabindex: "0",
      "aria-label": "Triangle ABC with edge B − A fixed. Click or drag to move C; arrow keys nudge it.",
    });
    fig.appendChild(svg);

    const uid = "dd-indep-" + (++figureCount);
    const defs = el("defs", {}, svg);
    for (const [id, cls] of [[uid + "-ah-1", "ink-1"], [uid + "-ah-2", "ink-2"]]) {
      const m = el("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 4.2, markerHeight: 4.2, orient: "auto-start-reverse" }, defs);
      el("path", { d: "M0,0 L10,5 L0,10 z", class: cls }, m);
    }

    // The span of B − A alone: the line every dependent C must land on.
    const far = 12;
    el("line", {
      x1: sx(A[0] - far * E1[0]), y1: sy(A[1] - far * E1[1]),
      x2: sx(A[0] + far * E1[0]), y2: sy(A[1] + far * E1[1]),
      class: "span-line",
    }, svg);
    // Below the line (y = x / 3 there), so the dashes never cross the words.
    const spanLabel = el("text", { x: sx(-3.3), y: sy(-1.55), class: "span-label" }, svg);
    spanLabel.textContent = "span of B − A";

    const para = el("polygon", { class: "para" }, svg);
    const tri = el("polygon", { class: "tri" }, svg);
    el("line", { x1: sx(A[0]), y1: sy(A[1]), x2: sx(B[0]), y2: sy(B[1]), class: "leg leg-1", "marker-end": `url(#${uid}-ah-1)` }, svg);
    const leg2 = el("line", { x1: sx(A[0]), y1: sy(A[1]), class: "leg leg-2", "marker-end": `url(#${uid}-ah-2)` }, svg);
    const leg1Label = el("text", { x: sx(1.7), y: sy(0.25) + 16, class: "leg-label ink-1", "text-anchor": "middle" }, svg);
    leg1Label.textContent = "B − A";
    const leg2Label = el("text", { class: "leg-label ink-2", "text-anchor": "middle" }, svg);
    leg2Label.textContent = "C − A";

    const vertex = (pt, name, dx, dy) => {
      el("circle", { cx: sx(pt[0]), cy: sy(pt[1]), r: 5, class: "vtx" }, svg);
      const t = el("text", { x: sx(pt[0]) + dx, y: sy(pt[1]) + dy, class: "vtx-label", "text-anchor": "middle" }, svg);
      t.textContent = name;
    };
    vertex(A, "A", -12, 16);
    vertex(B, "B", 14, 14);
    const cDot = el("circle", { r: 8, class: "pt" }, svg);
    const cLabel = el("text", { class: "pt-label" }, svg);
    cLabel.textContent = "C";

    const controls = document.createElement("label");
    controls.className = "dd-indep-angle";
    controls.innerHTML =
      '<span>Turn C − A from B − A</span>' +
      '<input type="range" min="0" max="359" step="1" />' +
      '<b class="dd-indep-deg"></b>';
    fig.appendChild(controls);
    const slider = controls.querySelector("input");
    const degOut = controls.querySelector(".dd-indep-deg");

    const read = document.createElement("div");
    read.className = "dd-indep-read";
    read.innerHTML =
      '<div class="dd-indep-edges"></div>' +
      '<div class="dd-indep-cross"></div>' +
      '<div class="dd-indep-area"></div>' +
      '<div class="dd-indep-verdict" role="status" aria-live="polite"></div>';
    fig.appendChild(read);
    // Looked up once: render() runs on every drag frame.
    const out = {
      edges: read.querySelector(".dd-indep-edges"),
      cross: read.querySelector(".dd-indep-cross"),
      area: read.querySelector(".dd-indep-area"),
      verdict: read.querySelector(".dd-indep-verdict"),
    };

    const chips = document.createElement("div");
    chips.className = "dd-bary-presets";
    for (const p of PRESETS) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = p.label;
      b.addEventListener("click", () => setAngle(p.deg));
      chips.appendChild(b);
    }
    fig.appendChild(chips);

    const cap = document.createElement("figcaption");
    cap.textContent = "Drag C, or turn the angle. The dashed parallelogram's area is the cross product; " +
      "the triangle is half of it. Line C up with the dashed span of B − A and both drop to zero.";
    fig.appendChild(cap);

    let e2 = edgeAtAngle(60, 2.4);

    function angleOf(v) {
      if (Math.hypot(v[0], v[1]) < EPS) return 0;
      const d = (Math.atan2(cross(E1, v), E1[0] * v[0] + E1[1] * v[1]) * 180) / Math.PI;
      return Math.round((d + 360) % 360) % 360;
    }

    function render() {
      const C = [A[0] + e2[0], A[1] + e2[1]];
      const D = [B[0] + e2[0], B[1] + e2[1]];
      para.setAttribute("points", [A, B, D, C].map(P).join(" "));
      tri.setAttribute("points", [A, B, C].map(P).join(" "));
      leg2.setAttribute("x2", sx(C[0])); leg2.setAttribute("y2", sy(C[1]));
      leg2.style.display = Math.hypot(e2[0], e2[1]) < 0.15 ? "none" : "";
      leg2Label.setAttribute("x", sx(e2[0] * 0.5) - 14);
      leg2Label.setAttribute("y", sy(e2[1] * 0.5) - 8);
      cDot.setAttribute("cx", sx(C[0])); cDot.setAttribute("cy", sy(C[1]));
      cLabel.setAttribute("x", sx(C[0]) + 11); cLabel.setAttribute("y", sy(C[1]) - 11);

      const r = classify(E1, e2);
      const flat = r.kind !== "independent";
      fig.classList.toggle("is-flat", flat);
      const deg = angleOf(e2);
      slider.value = String(deg);
      degOut.textContent = deg + "°";
      out.edges.textContent =
        `B − A = (${fmt(E1[0])}, ${fmt(E1[1])})    C − A = (${fmt(e2[0])}, ${fmt(e2[1])})`;
      out.cross.textContent =
        `cross product (B − A) × (C − A) = ${fmt(E1[0])}·${fmt(e2[1])} − ${fmt(E1[1])}·${fmt(e2[0])} = ${fmt(flat ? 0 : r.cross)}`;
      out.area.textContent =
        `triangle area = |cross product| ÷ 2 = ${fmt(r.area)}`;
      out.verdict.textContent = verdictText(r);
    }

    function setAngle(deg) {
      const len = Math.hypot(e2[0], e2[1]) || 2.4;
      e2 = edgeAtAngle(deg, len);
      render();
    }

    function setC(x, y) {
      e2 = [snap(clamp(x, X0 + 0.1, X1 - 0.1)) - A[0], snap(clamp(y, Y0 + 0.1, Y1 - 0.1)) - A[1]];
      render();
    }

    slider.addEventListener("input", () => setAngle(Number(slider.value)));

    function fromEvent(ev) {
      const box = svg.getBoundingClientRect();
      const px = ((ev.clientX - box.left) / box.width) * W_PX;
      const py = ((ev.clientY - box.top) / box.height) * H_PX;
      const [x, y] = fromScreen(px, py);
      setC(x, y);
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
      const d = { ArrowRight: [0.1, 0], ArrowLeft: [-0.1, 0], ArrowUp: [0, 0.1], ArrowDown: [0, -0.1] }[ev.key];
      if (!d) return;
      ev.preventDefault();
      setC(A[0] + e2[0] + d[0], A[1] + e2[1] + d[1]);
    });

    render();
    return fig;
  }

  const api = { cross, classify, verdictText, edgeAtAngle };
  if (LF) {
    LF.BUILDERS.independence = buildIndependence;
    LF.independence = api;
    // A lesson already on screen before this file loaded still gets its figure.
    LF.mountAll(document);
  }
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})();
