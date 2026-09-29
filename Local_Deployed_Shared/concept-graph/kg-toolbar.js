/* concept-graph/kg-toolbar.js — ONE toolbar across the top of the Knowledge
 * Graph, instead of controls in three corners (Seth, 2026-09-25: "the filter
 * in the bottom left is separate from the top left … at the bottom left
 * there's too much information"), cut to four controls on 2026-09-29 ("keep
 * condensed view. remove adaptive view. remove everything else"):
 *
 *   [Complete|Condensed]  [Mastery|Sections]  [Filter ▾]  ……  [Fit]
 *   hint line · cold-start notice
 *
 *   View segment, hint — graph-views.js
 *   Colour segment (#kg-colormode), .kg2-controls (Fit, Colab links),
 *   #kg-nodata — lesson-graph.js
 *   Filter — this file
 *
 * Moving an element keeps its listeners, and every owner looks its elements
 * up by id (graph-views.js marks the view segment by `#kg-view-seg`, not
 * under its old card), so the owners don't know the bar exists. Without this
 * script the old layout stands: nothing else depends on it.
 *
 * The Filter is the PRACTICE switch per area (it replaced the Knowledge Graph
 * page's "Practice focus" select, 2026-09-29: "make it more like the existing
 * filter ui where you check off multiple areas to enable or disable nodes").
 * Unticking an area turns its concepts off in kc-prefs (one bulk PUT
 * /api/practice/kc-prefs) — the same `enabled: false` a concept's own switch
 * in the panel's cog writes — so the practice queue skips them and the graph
 * greys them with the machinery it already has for a disabled concept
 * (lesson-graph.js `kc-disabled`). A partly-off area shows a dash. The badge
 * counts areas with anything off. A save re-reads the lattice and repaints
 * (DeltaKgCore.refresh), then `delta:kc-prefs-changed` tells the panel,
 * the headers and practice/practice-target.js. An account still on the old
 * Ray Tracing 0.1 focus sees it as a row of its own at the top, with a Turn
 * off button (practice-target.js explains why it isn't converted). */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  let bar = null, sub = null;
  const SELF = document.currentScript && document.currentScript.src;

  // `?kgtune=1` (remembered in this browser; `?kgtune=0` forgets it): load
  // kg-tune.js, the layout optimizer's sliders. Nobody else downloads it.
  const loadTune = () => {
    let on = false;
    try {
      const v = new URLSearchParams(location.search).get("kgtune");
      if (v === "0" || v === "1") localStorage.setItem("dd_kg_tune_on", v);
      on = localStorage.getItem("dd_kg_tune_on") === "1";
    } catch (_) {}
    if (!on || !SELF || document.querySelector("script[data-kg-tune]")) return;
    const sc = document.createElement("script");
    sc.src = new URL("kg-tune.js?v=1", SELF).href;
    sc.dataset.kgTune = "1";
    document.head.appendChild(sc);
  };

  const el = (tag, cls, html) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  };

  /* ---------------- filter popover: practise these areas --------------- */
  const lattice = () => (typeof window.getKcLattice === "function" ? window.getKcLattice() : null);
  const sections = () => (window.deltaKgView && window.deltaKgView.sections ? window.deltaKgView.sections() : []);
  const shortLabel = (s) => (window.DeltaKgSections && window.DeltaKgSections.label ? window.DeltaKgSections.label(s) : s.label);
  // Off = the learner's own switch (row.pref.enabled === false). A concept
  // whose whole COURSE is off in the Courses tab is a different switch
  // (lesson-graph.js `_courseOff`) and is left to that tab.
  const userOff = (row) => !!(row && row.pref && row.pref.enabled === false);
  const courseOff = (row) => !!(row && row.state === "disabled" && !userOff(row));
  const areaState = (sec) => {
    const L = lattice();
    const rows = sec.kcs.map((kc) => (L && L.kcs ? L.kcs[kc] : null));
    const own = rows.filter((r) => !courseOff(r));
    const off = own.filter(userOff).length;
    return { total: own.length, off, courseOff: !own.length && rows.some(courseOff) };
  };
  const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  let saving = Promise.resolve();
  const saveArea = (sec, enabled) => {
    const fn = typeof window.apiFetch === "function" ? window.apiFetch : null;
    if (!fn) return Promise.resolve(false);
    const L = lattice();
    const kcs = sec.kcs.filter((kc) => !(L && L.kcs && courseOff(L.kcs[kc])));
    if (!kcs.length) return Promise.resolve(true);
    // Serialized: two quick clicks must land in click order.
    const run = () => fn("/api/practice/kc-prefs", {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kcs, enabled }),
    }).then(async (res) => {
      if (!res || !res.ok) return false;
      const body = await res.json().catch(() => null);
      // Patch the cached lattice first, so the list and the badge are right
      // even if the re-read below fails; then re-read the server's report
      // (the real gate states) and repaint the bubbles before announcing.
      const cur = lattice();
      if (body && body.prefs && cur && cur.kcs) Object.keys(body.prefs).forEach((kc) => {
        const row = cur.kcs[kc];
        if (!row) return;
        row.pref = { enabled: body.prefs[kc].enabled, weight: body.prefs[kc].weight };
        // Off is "disabled" at once; back on, the real gate (locked,
        // frontier, …) is the server's to say — unknown until the re-read.
        row.state = enabled ? null : "disabled";
      });
      try { await window.DeltaKgCore?.refresh?.(); } catch (_) {}
      window.dispatchEvent(new CustomEvent("delta:kc-prefs-changed", { detail: { kcs, enabled } }));
      return true;
    }).catch(() => false);
    saving = saving.then(run, run);
    return saving;
  };

  const buildFilter = () => {
    const wrap = el("div", "kgt-filter");
    const btn = el("button", "kgt-btn kgt-filter-btn",
      '<span>Filter</span><span class="kgt-badge" hidden></span><span class="kgt-caret" aria-hidden="true">▾</span>');
    btn.type = "button";
    btn.setAttribute("aria-haspopup", "true");
    btn.setAttribute("aria-expanded", "false");
    const pop = el("div", "kgt-pop");
    pop.hidden = true;
    pop.setAttribute("role", "dialog");
    pop.setAttribute("aria-label", "Areas to practise");
    wrap.append(btn, pop);
    const badge = btn.querySelector(".kgt-badge");

    const focus = () => {
      const f = window.PracticeTarget && window.PracticeTarget.focus ? window.PracticeTarget.focus() : null;
      return f && f.target && f.target !== "all" ? f : null;
    };
    const refreshBadge = () => {
      const n = sections().filter((s) => areaState(s).off > 0).length + (focus() ? 1 : 0);
      badge.hidden = !n;
      badge.textContent = n ? String(n) : "";
      btn.classList.toggle("is-on", !!n);
      btn.title = n ? `${n} area${n === 1 ? "" : "s"} switched off` : "Choose which areas you practise";
    };
    const render = () => {
      const list = sections();
      const f = focus();
      pop.innerHTML = '<div class="kgt-pop-title">Practise these areas</div>' +
        (f ? `<div class="kgt-focus"><span>Focus: <strong>${esc(f.label || f.target)}</strong> and what it builds on — ` +
          `nothing else is served.</span><button type="button" class="kgt-btn kgt-focus-off">Turn off</button></div>` : "") +
        '<div class="kgt-area-list">' + list.map((s) => {
          const st = areaState(s);
          const on = st.total - st.off;
          const note = st.courseOff ? "course off" : st.off ? `${on}/${st.total}` : String(st.total);
          return `<label class="kgt-area${st.courseOff ? " is-course-off" : ""}">` +
            `<input type="checkbox" data-sid="${esc(s.id)}"${st.off < st.total ? " checked" : ""}${st.courseOff ? " disabled" : ""}>` +
            `<span class="kgt-swatch" style="background:${esc(s.color)}"></span>` +
            `<span class="kgt-area-label">${esc(shortLabel(s))}</span><span class="kgt-area-n">${note}</span></label>`;
        }).join("") + "</div>" +
        '<div class="kgt-pop-foot" aria-live="polite">Unticked areas are skipped in practice and greyed on the graph.</div>';
      list.forEach((s) => {
        const cb = pop.querySelector(`input[data-sid="${CSS.escape(s.id)}"]`);
        const st = areaState(s);
        if (cb) cb.indeterminate = st.off > 0 && st.off < st.total;
      });
      refreshBadge();
    };
    pop.addEventListener("click", async (e) => {
      const off = e.target.closest(".kgt-focus-off");
      if (!off || !window.PracticeTarget) return;
      off.disabled = true;
      off.textContent = "Turning off…";
      try { await window.PracticeTarget.clearFocus(); } catch (_) { /* row stays; re-rendered below */ }
      render();
    });
    pop.addEventListener("change", async (e) => {
      const cb = e.target.closest("input[data-sid]");
      if (!cb) return;
      const sec = sections().find((s) => s.id === cb.dataset.sid);
      if (!sec) return;
      const foot = pop.querySelector(".kgt-pop-foot");
      cb.disabled = true;
      if (foot) foot.textContent = "Saving…";
      const ok = await saveArea(sec, cb.checked);
      render();
      const f2 = pop.querySelector(".kgt-pop-foot");
      if (!ok && f2) f2.textContent = window.apiFetch ? "Not saved — try again." : "Sign in to save.";
    });

    const close = () => { pop.hidden = true; btn.setAttribute("aria-expanded", "false"); };
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (pop.hidden) render();
      pop.hidden = !pop.hidden;
      btn.setAttribute("aria-expanded", pop.hidden ? "false" : "true");
    });
    document.addEventListener("click", (e) => { if (!pop.hidden && !wrap.contains(e.target)) close(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !pop.hidden) { close(); btn.focus(); } });
    // The lattice lands after the bar, and moves with every graded attempt
    // and every switch in the panel's cog. A course scope resolves AFTER
    // kg-course-changed (its concept set is a fetch); kg-view-changed follows.
    ["delta:kc-readiness-changed", "delta:kc-prefs-changed", "delta:kg-course-changed", "delta:kg-view-changed", "delta:practice-focus-changed"].forEach((ev) =>
      window.addEventListener(ev, () => (pop.hidden ? refreshBadge() : render())));
    refreshBadge();
    return wrap;
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
    bar.append(viewSeg, colour, buildFilter(), el("span", "kgt-spacer"));
    // Fit as an icon, its words kept for screen readers and the tooltip.
    const iconize = (btn, svg, label) => {
      btn.classList.add("kgt-btn", "kgt-icon");
      btn.setAttribute("aria-label", label);
      btn.title = label;
      btn.innerHTML = svg + '<span class="kgt-sr">' + label + "</span>";
    };
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
    loadTune();
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
