/* concept-graph/kg-tune.js — sliders for the layout optimizer's knobs, on the
 * real Knowledge Graph (Seth, 2026-09-25: "give me hyperparameters that I can
 * slide around until it looks good").
 *
 * Off unless asked for: `?kgtune=1` turns it on in this browser (remembered),
 * `?kgtune=0` turns it off. kg-toolbar.js loads this file only then, so a
 * learner never downloads it.
 *
 * What it does:
 *   - a ⚙ button in the toolbar opens a panel of sliders, one per knob of
 *     kg-layout.js (`place` moves the nodes, `route` draws the edges);
 *   - releasing a slider stores the value (`dd_kg_tune`, this browser only)
 *     and lays the graph out again; kg-look.js reads the values through
 *     `window.DeltaKgTune.opts()` and the layout cache keys on them, so a
 *     setting you return to is instant;
 *   - when the new layout lands it measures what is drawn: edge crossings,
 *     edges running a lane beside an unrelated one, the gap from each node to
 *     its nearest neighbour, and how long the optimizer took;
 *   - "Copy values" copies the knobs you moved, to paste back to Claude so
 *     they become everyone's defaults.
 */
(function () {
  "use strict";

  const KEY = "dd_kg_tune";
  const read = () => { try { return JSON.parse(localStorage.getItem(KEY) || "{}") || {}; } catch (_) { return {}; } };
  const write = (v) => { try { localStorage.setItem(KEY, JSON.stringify(v)); } catch (_) {} };
  let over = { place: {}, route: {} };
  window.DeltaKgTune = {
    opts: () => ({ place: Object.assign({}, over.place), route: Object.assign({}, over.route) }),
  };

  // [stage, key, label, min, max, step, what it does]
  const KNOBS = [
    ["Nodes", [
      ["place", "sepX", "Side gap", 20, 320, 5, "px between two boxes side by side"],
      ["place", "sepY", "Stack gap", 20, 320, 5, "px between two boxes one above the other"],
      ["place", "gapY", "Level gap", 50, 340, 5, "px a prerequisite sits below what it unlocks"],
      ["place", "wCross", "Uncross nodes", 0, 800, 10, "how hard nodes move to stop edges crossing"],
      ["place", "wLen", "Short edges", 0, 1, 0.02, "pull connected nodes together"],
      ["place", "wGrav", "Pack to centre", 0, 0.1, 0.002, "pull every node toward the middle"],
      ["place", "gap", "Piece gap", 40, 600, 10, "px between unconnected pieces of the graph"],
    ]],
    ["Edges", [
      ["route", "cell", "Lane width", 8, 28, 2, "grid px: parallel edges sit at least this far apart"],
      ["route", "lane", "Keep apart", 0, 12, 0.5, "cost of running beside an unrelated edge"],
      ["route", "lane2", "…two lanes over", 0, 2, 0.05, "× Keep apart, one lane further out"],
      ["route", "laneRel", "Merge fans", 0, 6, 0.25, "cost of a fan running side by side instead of merging"],
      ["route", "cross", "Avoid crossing", 0, 400, 10, "cost of an edge crossing another"],
      ["route", "bend", "Fewer bends", 0, 10, 0.5, "cost of each turn"],
      ["route", "radius", "Corner radius", 0, 140, 5, "px of rounding at a turn"],
      ["route", "down", "No doubling back", 0, 3, 0.1, "cost of an edge running downward"],
      ["route", "near", "Clear the boxes", 0, 4, 0.1, "cost of hugging a node"],
    ]],
  ];

  // Saved values: only known knobs, only finite numbers inside the slider's
  // range (a stale or hand-edited entry would desynchronise panel and layout).
  (() => {
    const saved = read();
    KNOBS.forEach(([, rows]) => rows.forEach(([stage, key, , min, max]) => {
      const v = saved && saved[stage] && saved[stage][key];
      if (typeof v === "number" && isFinite(v) && v >= min && v <= max) over[stage][key] = v;
    }));
  })();

  const $ = (sel, root) => (root || document).querySelector(sel);
  const el = (tag, cls, html) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  };
  const defaults = () => (window.DeltaKgLayout && window.DeltaKgLayout.defaults
    ? window.DeltaKgLayout.defaults() : { place: {}, route: {} });
  const fmt = (v, step) => (step < 1 ? (+v).toFixed(String(step).split(".")[1].length) : String(Math.round(v)));
  const cyNow = () => (typeof window.deltaConceptGraphCy === "function" ? window.deltaConceptGraphCy() : null);

  /* ---------------- measuring what is drawn ----------------------------- */
  // The curve Cytoscape draws for an unbundled bezier through `cps`: from the
  // source, quadratics ending at the midpoints of consecutive control points.
  const curve = (s, t, cps) => {
    const out = [];
    let cur = s;
    for (let i = 0; i <= cps.length; i++) {
      const last = i >= cps.length - 1;
      const end = last ? t : { x: (cps[i].x + cps[i + 1].x) / 2, y: (cps[i].y + cps[i + 1].y) / 2 };
      const ctl = i < cps.length ? cps[i] : { x: (cur.x + end.x) / 2, y: (cur.y + end.y) / 2 };
      for (let f = i ? 1 : 0; f <= 8; f++) {
        const a = f / 8, b = 1 - a;
        out.push({ x: b * b * cur.x + 2 * b * a * ctl.x + a * a * end.x, y: b * b * cur.y + 2 * b * a * ctl.y + a * a * end.y });
      }
      cur = end;
      if (last) break;
    }
    return out;
  };
  const measure = (cy) => {
    const routes = cy.__kgRoutes || {};
    const polys = [];
    cy.edges().filter((e) => e.visible()).forEach((e) => {
      const r = routes[e.id()];
      const P = curve(e.source().position(), e.target().position(), r ? r.cps : []);
      let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
      P.forEach((p) => { x1 = Math.min(x1, p.x); y1 = Math.min(y1, p.y); x2 = Math.max(x2, p.x); y2 = Math.max(y2, p.y); });
      polys.push({ s: e.source().id(), t: e.target().id(), P, x1, y1, x2, y2 });
    });
    const o = (p, q, r) => Math.sign((q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x));
    const X = (a, b, c, d) => o(a, b, c) * o(a, b, d) < 0 && o(c, d, a) * o(c, d, b) < 0;
    let crossings = 0;
    for (let i = 0; i < polys.length; i++) for (let j = i + 1; j < polys.length; j++) {
      const A = polys[i], B = polys[j];
      if (A.s === B.s || A.s === B.t || A.t === B.s || A.t === B.t) continue;
      if (A.x2 < B.x1 || B.x2 < A.x1 || A.y2 < B.y1 || B.y2 < A.y1) continue;
      for (let a = 0; a + 1 < A.P.length; a++) for (let b = 0; b + 1 < B.P.length; b++) {
        if (X(A.P[a], A.P[a + 1], B.P[b], B.P[b + 1])) crossings++;
      }
    }
    const boxes = cy.nodes().filter((n) => n.visible()).map((n) => n.boundingBox({ includeLabels: true }));
    const gaps = boxes.map((a, i) => {
      let m = Infinity;
      boxes.forEach((b, j) => {
        if (i === j) return;
        m = Math.min(m, Math.hypot(Math.max(0, b.x1 - a.x2, a.x1 - b.x2), Math.max(0, b.y1 - a.y2, a.y1 - b.y2)));
      });
      return m;
    }).sort((a, b) => a - b);
    const st = cy.__kgLayoutStats || {};
    return {
      crossings,
      closeCells: st.closeCells,
      gapMed: boxes.length > 1 ? Math.round(gaps[gaps.length >> 1]) : null,
      gapMin: boxes.length > 1 ? Math.round(gaps[0]) : null,
      ms: st.cached ? 0 : (st.ms || (st.placeMs || 0) + (st.routeMs || 0)),
      cached: !!st.cached,
    };
  };

  /* ---------------- the panel ------------------------------------------- */
  let panel = null, statusEl = null, startedAt = 0, timer = 0;
  const painters = [];
  const say = (html) => { if (statusEl) statusEl.innerHTML = html; };
  const running = () => {
    startedAt = Date.now();
    clearInterval(timer);
    const tick = () => {
      const s = (Date.now() - startedAt) / 1000;
      if (s > 30) { clearInterval(timer); say("No answer after 30 s — this view may not use the optimizer (try Complete)."); return; }
      say(`<span class="kgtu-spin"></span>Laying out… ${s.toFixed(1)} s`);
    };
    tick();
    timer = setInterval(tick, 200);
  };
  const relayout = () => {
    write(over);
    running();
    if (window.deltaKgView && typeof window.deltaKgView.relayout === "function") window.deltaKgView.relayout();
  };
  const moved = () => {
    const d = defaults(), out = {};
    ["place", "route"].forEach((k) => Object.keys(over[k]).forEach((key) => {
      if (over[k][key] !== d[k][key]) (out[k] = out[k] || {})[key] = over[k][key];
    }));
    return out;
  };

  const build = () => {
    const graph = $(".kg2-graph.has-kgt");
    const bar = graph && $(".kgt-bar", graph);
    if (!graph || !bar || panel) return !!panel;
    const d = defaults();

    const btn = el("button", "kgt-btn kgt-icon kgtu-btn",
      '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 4h10M3 8h10M3 12h10"/><circle cx="6" cy="4" r="1.6"/><circle cx="10.5" cy="8" r="1.6"/><circle cx="5" cy="12" r="1.6"/></svg><span class="kgt-sr">Layout tuning</span>');
    btn.type = "button";
    btn.title = "Layout tuning (?kgtune=0 hides this)";
    btn.setAttribute("aria-expanded", "false");
    const look = $(".kgt-look", bar);
    bar.insertBefore(btn, look || null);

    panel = el("div", "kgtu-panel");
    panel.hidden = true;
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-label", "Layout tuning");
    panel.appendChild(el("div", "kgtu-head", '<strong>Layout tuning</strong><span>release a slider to re-run</span>'));
    statusEl = el("div", "kgtu-status");
    panel.appendChild(statusEl);

    KNOBS.forEach(([title, rows]) => {
      panel.appendChild(el("div", "kgtu-group", title));
      rows.forEach(([stage, key, label, min, max, step, hint]) => {
        const base = d[stage][key];
        const row = el("label", "kgtu-row");
        row.title = hint;
        const name = el("span", "kgtu-name", label);
        const val = el("span", "kgtu-val");
        const range = el("input");
        range.type = "range";
        range.min = min; range.max = max; range.step = step;
        const reset = el("button", "kgtu-reset", "↺");
        reset.type = "button";
        reset.title = `Back to the default (${fmt(base, step)})`;
        const paint = () => {
          const v = key in over[stage] ? over[stage][key] : base;
          range.value = v;
          val.textContent = fmt(v, step);
          const changed = key in over[stage] && over[stage][key] !== base;
          row.classList.toggle("is-moved", changed);
          reset.hidden = !changed;
        };
        range.addEventListener("input", () => { val.textContent = fmt(range.value, step); });
        range.addEventListener("change", () => {
          const v = +range.value;
          if (v === base) delete over[stage][key]; else over[stage][key] = v;
          paint();
          relayout();
        });
        reset.addEventListener("click", (e) => { e.preventDefault(); delete over[stage][key]; paint(); relayout(); });
        paint();
        painters.push(paint);
        row.append(name, val, reset, range, el("span", "kgtu-hint", hint));
        panel.appendChild(row);
      });
    });

    const foot = el("div", "kgtu-foot");
    const copy = el("button", "kgt-btn", "Copy values");
    const resetAll = el("button", "kgt-btn", "Reset all");
    const out = el("textarea", "kgtu-out");
    out.readOnly = true;
    out.rows = 3;
    out.hidden = true;
    copy.type = resetAll.type = "button";
    copy.addEventListener("click", () => {
      const text = "KG layout values: " + JSON.stringify(moved());
      out.value = text;
      out.hidden = false;
      out.select();
      const flash = (t) => { copy.textContent = t; setTimeout(() => { copy.textContent = "Copy values"; }, 1800); };
      const manual = () => flash("Selected — press Ctrl+C");
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(() => flash("Copied ✓"), manual);
      else manual();
    });
    resetAll.addEventListener("click", () => {
      over = { place: {}, route: {} };
      painters.forEach((p) => p());
      out.hidden = true;
      relayout();
    });
    foot.append(copy, resetAll);
    panel.append(foot, out);
    graph.appendChild(panel);

    btn.addEventListener("click", () => {
      panel.hidden = !panel.hidden;
      btn.setAttribute("aria-expanded", panel.hidden ? "false" : "true");
      btn.classList.toggle("is-on", !panel.hidden);
      if (!panel.hidden) show(cyNow());
    });
    return true;
  };

  const show = (cy) => {
    if (!cy || !statusEl) return;
    const m = measure(cy);
    clearInterval(timer);
    say(`<b>${m.crossings}</b> crossings · <b>${m.closeCells == null ? "–" : m.closeCells}</b> cells beside another edge<br>` +
      (m.gapMed == null ? "nearest-node gap: – · " : `nearest-node gap: median <b>${m.gapMed}</b> px, min ${m.gapMin} px · `) + ` ${m.cached ? "cached" : (m.ms / 1000).toFixed(1) + " s"}`);
  };
  // The layout's answer is on the canvas; measure after the frame that draws it.
  window.addEventListener("delta:kg-layout-done", (ev) => {
    const cy = ev.detail && ev.detail.cy;
    if (!panel || !cy) return;
    requestAnimationFrame(() => setTimeout(() => show(cy), 0));
  });

  // Built by kg-toolbar.js once its bar exists; saved values from a
  // previous visit apply to the next layout, so lay out again now.
  const boot = () => {
    if (!build()) { setTimeout(boot, 300); return; }
    if (Object.keys(over.place).length || Object.keys(over.route).length) relayout();
  };
  boot();
})();
