/* concept-graph/kg-toolbar.js — ONE toolbar across the top of the Knowledge
 * Graph, instead of controls in three corners (Seth, 2026-09-25: "the filter
 * in the bottom left is separate from the top left … at the bottom left
 * there's too much information").
 *
 * It builds nothing the graph needs. It gathers controls the other files
 * already built and wired, and moves them into one bar:
 *
 *   [Adaptive|Condensed|Complete]  [Mastery|Sections|Categories|Math/code]
 *   [Filter ▾]  ……  [▭ ●]  [Reset] [Fit]
 *   hint line · cold-start notice
 *
 *   View segment, Reset, hint, Course select, Chapters list — graph-views.js
 *   Colour segment (#kg-colormode), .kg2-controls (Fit, Colab links),
 *   #kg-nodata — lesson-graph.js
 *   Node look (▭ labelled box / ● dot) — this file, over kg-look.js
 *
 * Moving an element keeps its listeners, and every owner looks its elements
 * up by id (graph-views.js marks the view segment by `#kg-view-seg`, not
 * under its old card), so the owners don't know the bar exists. Without this
 * script the old layout stands: nothing else depends on it.
 *
 * The Filter popover holds the Course select and the Chapters list; its badge
 * counts what is filtered out, so a filtered map never looks complete. */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  let bar = null, sub = null;

  const el = (tag, cls, html) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  };

  /* ---------------- filter popover ------------------------------------- */
  const filterCount = (pop) => {
    let n = pop.querySelectorAll('input[data-sid]:not(:checked)').length;
    const course = pop.querySelector("select");
    if (course && course.value) n += 1;
    return n;
  };
  const buildFilter = (panel) => {
    const wrap = el("div", "kgt-filter");
    const btn = el("button", "kgt-btn kgt-filter-btn",
      '<span>Filter</span><span class="kgt-badge" hidden></span><span class="kgt-caret" aria-hidden="true">▾</span>');
    btn.type = "button";
    btn.setAttribute("aria-haspopup", "true");
    btn.setAttribute("aria-expanded", "false");
    const pop = el("div", "kgt-pop");
    pop.hidden = true;
    pop.setAttribute("role", "dialog");
    pop.setAttribute("aria-label", "Filter the graph");
    const course = panel.querySelector(".kgv-course");
    if (course) pop.appendChild(course);
    const chapters = panel.querySelector("details.kgv-chapters");
    if (chapters) {
      // Inside the popover the list is the point; no second fold.
      chapters.open = true;
      pop.appendChild(chapters);
    }
    wrap.append(btn, pop);

    const badge = btn.querySelector(".kgt-badge");
    const refresh = () => {
      const n = filterCount(pop);
      badge.hidden = !n;
      badge.textContent = n ? String(n) : "";
      btn.classList.toggle("is-on", !!n);
      btn.title = n ? `${n} filter${n === 1 ? "" : "s"} on` : "Filter by course or chapter";
    };
    // graph-views.js rebuilds the chapter list's HTML; listen on the popover.
    pop.addEventListener("change", () => setTimeout(refresh, 0));
    new MutationObserver(refresh).observe(pop, { childList: true, subtree: true });
    refresh();

    const close = () => { pop.hidden = true; btn.setAttribute("aria-expanded", "false"); };
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      pop.hidden = !pop.hidden;
      btn.setAttribute("aria-expanded", pop.hidden ? "false" : "true");
    });
    document.addEventListener("click", (e) => { if (!pop.hidden && !wrap.contains(e.target)) close(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !pop.hidden) { close(); btn.focus(); } });
    return wrap;
  };

  /* ---------------- node look ------------------------------------------ */
  const buildLook = () => {
    const look = window.DeltaKgLook;
    if (!look) return null;
    const seg = el("div", "kgt-seg kgt-look");
    seg.setAttribute("role", "group");
    seg.setAttribute("aria-label", "Node style");
    seg.innerHTML =
      '<button type="button" data-look="label" title="Labelled boxes — the name inside, the colour is the box">' +
        '<svg viewBox="0 0 20 14" aria-hidden="true"><rect x="1.5" y="2.5" width="17" height="9" rx="3"/></svg>' +
        '<span class="kgt-sr">Labels</span></button>' +
      '<button type="button" data-look="dot" title="Dots — a small circle, the name underneath">' +
        '<svg viewBox="0 0 20 14" aria-hidden="true"><circle cx="10" cy="7" r="4.5"/></svg>' +
        '<span class="kgt-sr">Dots</span></button>';
    const paint = () => seg.querySelectorAll("[data-look]").forEach((b) => {
      const on = b.dataset.look === look.look();
      b.classList.toggle("active", on);
      b.setAttribute("aria-pressed", on ? "true" : "false");
    });
    seg.querySelectorAll("[data-look]").forEach((b) => b.addEventListener("click", () => look.setLook(b.dataset.look)));
    window.addEventListener("delta:kg-look-changed", paint);
    paint();
    return seg;
  };

  /* ---------------- assemble ------------------------------------------- */
  // #kg-nodata is created lazily by lesson-graph.js; adopt it when it shows up.
  const adoptNoData = () => {
    const nd = $("kg-nodata");
    if (nd && sub && nd.parentNode !== sub) sub.appendChild(nd);
  };

  const assemble = () => {
    if (bar) return true;
    const graph = document.querySelector("#page-knowledge-graph .kg2-graph") || document.querySelector(".kg2-graph");
    const panel = $("kg-view-panel");
    const colour = $("kg-colormode");
    const viewSeg = $("kg-view-seg");
    if (!graph || !panel || !colour || !viewSeg) return false;

    bar = el("div", "kgt-bar");
    bar.setAttribute("role", "toolbar");
    bar.setAttribute("aria-label", "Knowledge graph");
    viewSeg.classList.add("kgt-seg");
    viewSeg.setAttribute("aria-label", "How much of the graph");
    colour.classList.add("kgt-seg");
    colour.setAttribute("role", "group");
    colour.setAttribute("aria-label", "Colour by");
    bar.append(viewSeg, colour, buildFilter(panel), el("span", "kgt-spacer"));
    const look = buildLook();
    if (look) bar.appendChild(look);
    // Reset and Fit as icons, their words kept for screen readers and the
    // tooltip: in a full bar the words were what made it wrap.
    const iconize = (btn, svg, label) => {
      btn.classList.add("kgt-btn", "kgt-icon");
      btn.setAttribute("aria-label", label);
      btn.title = label;
      btn.innerHTML = svg + '<span class="kgt-sr">' + label + "</span>";
    };
    const reset = $("kg-view-reset");
    if (reset) {
      iconize(reset, '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8a4.5 4.5 0 1 0 1.4-3.3"/><path d="M4.6 1.9v3h3"/></svg>',
        "Reset — fold away what you opened");
      bar.appendChild(reset);
    }
    const controls = graph.querySelector(".kg2-controls");
    if (controls) {
      const fit = controls.querySelector("#kg-fit");
      if (fit) iconize(fit, '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2 5.5V2h3.5M10.5 2H14v3.5M14 10.5V14h-3.5M5.5 14H2v-3.5"/></svg>',
        "Fit the graph to the window");
      bar.appendChild(controls);
    }

    sub = el("div", "kgt-sub");
    const hint = $("kg-view-hint");
    if (hint) sub.appendChild(hint);

    // Header = the bar + a caption line (hint, cold-start notice). The
    // canvases start under it (`--kgt-top`, kg-toolbar.css), measured rather
    // than assumed: the bar wraps on a narrow pane and the hint changes length.
    const head = el("div", "kgt-head");
    head.append(bar, sub);
    graph.prepend(head);
    panel.hidden = true;
    graph.classList.add("has-kgt");
    let lastTop = -1, queued = false;
    const measure = () => {
      queued = false;
      const top = Math.ceil(head.getBoundingClientRect().height);
      if (top === lastTop) return;
      const first = lastTop < 0;
      lastTop = top;
      graph.style.setProperty("--kgt-top", top + "px");
      const cy = typeof window.deltaConceptGraphCy === "function" ? window.deltaConceptGraphCy() : null;
      if (cy) cy.resize();
      const ccy = window.deltaKgView && window.deltaKgView.condensed && window.deltaKgView.condensed();
      if (ccy) ccy.resize();
      // Once, so the owners refit to the smaller canvas; after that a caption
      // changing length only re-measures (a refit per hint would jump the map).
      if (first) window.dispatchEvent(new Event("resize"));
    };
    new ResizeObserver(() => { if (!queued) { queued = true; requestAnimationFrame(measure); } }).observe(head);
    adoptNoData();
    new MutationObserver(adoptNoData).observe(graph, { childList: true });
    return true;
  };

  // The graph builds lazily (first visit to its tab), so wait for its ready
  // event; the colour segment and the view card land a tick or two after it.
  const poll = () => {
    let tries = 0;
    const tick = () => { if (!assemble() && tries++ < 80) setTimeout(tick, 150); };
    tick();
  };
  const boot = () => {
    if (assemble()) return;
    window.addEventListener("delta:practice-target-graph-ready", poll);
    poll();
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
