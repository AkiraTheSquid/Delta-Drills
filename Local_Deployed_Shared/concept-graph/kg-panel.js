/* concept-graph/kg-panel.js — the Knowledge Graph's side panel (#kg-info).
 *
 * Seth, 2026-09-29: "It should just have the title of the lesson, along with
 * your progress on it, and … the bar to represent how strong you are for that
 * concept … a cog icon in the top right for that panel to … do all the other
 * settings that are currently in tabs … whenever you click on the cog icon, it
 * just displays everything on the right with no tabs."
 *
 * Four views, one at a time, in #kg-info-body:
 *   empty    — nothing selected: one line saying what a click does.
 *   concept  — title, area · lesson, the reading (% + band word), the mastery
 *              bar, one progress line (rung of the ladder + graded record).
 *              "Read lesson" unfolds the lesson itself underneath, with its
 *              prerequisites and what it unlocks.
 *   section  — an AREA (its title on the graph was clicked, kg-sections.js):
 *              the average reading and bar, how its concepts spread over the
 *              bands, and every concept weakest-first as a clickable row.
 *   settings — the cog: this concept's practice switch and priority weight,
 *              its details (folded), and the graph's display switches.
 *              Stacked, no tabs. The cog again (or Esc) goes back.
 *
 * It replaces lesson-graph.js's Lesson | Metadata | Settings tabs and the
 * readout docked under the canvas (hover preview included) — the numbers that
 * readout carried (evidence source, CI wording, difficulty …) are not shown
 * any more; the bar still draws the interval.
 *
 * Everything it reads comes through `window.DeltaKgCore` (lesson-graph.js):
 * the one reader of the learner model, so this panel, the bubbles and the
 * headers can never disagree about a number. It writes only kc-prefs (the
 * learner's own switch and weight, PUT /api/practice/kc-prefs/<kc>) and asks
 * the core to re-read and repaint afterwards. */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const core = () => window.DeltaKgCore || null;
  const sectionsApi = () => window.DeltaKgSections || null;

  let view = "empty";            // "empty" | "concept" | "section" | "settings"
  let kc = null;                 // the selected concept, if any
  let sid = null;                // the selected area, if any
  let lessonOpen = false;        // "Read lesson" unfolded (kept across repaints)
  let under = "empty";           // the view the cog returns to

  const esc = (v) => (core() ? core().esc(v) : String(v ?? ""));
  const pct = (v) => (Number.isFinite(v) ? Math.round(v * 100) + "%" : "—");
  const shortLabel = (s) => (sectionsApi() ? sectionsApi().label(s) : (s && s.label) || "");

  /* ---------------- head: area chip + cog ------------------------------ */
  const ensureCog = () => {
    let cog = $("kg-panel-cog");
    if (cog) return cog;
    const head = document.querySelector(".kg2-info .kg2-info-head");
    if (!head) return null;
    cog = document.createElement("button");
    cog.type = "button";
    cog.id = "kg-panel-cog";
    cog.className = "kg2-cog";
    cog.title = "Settings";
    cog.setAttribute("aria-label", "Settings");
    cog.setAttribute("aria-pressed", "false");
    cog.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z"/>' +
      '<path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>';
    head.appendChild(cog);
    cog.addEventListener("click", () => toggleSettings());
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && view === "settings" && !document.body.classList.contains("kg-maxi-open")) toggleSettings(false);
    });
    return cog;
  };

  const paintHead = () => {
    const c = core();
    const meta = $("kg-info-meta");
    const max = $("kg-maximize");
    const cog = ensureCog();
    if (cog) {
      cog.classList.toggle("active", view === "settings");
      cog.setAttribute("aria-pressed", view === "settings" ? "true" : "false");
    }
    if (max) {
      max.hidden = !kc;
      if (kc) max.dataset.kc = kc;
    }
    if (!meta) return;
    let s = null;
    if (kc && c) s = c.section(kc);
    else if (sid && window.deltaKgView) s = window.deltaKgView.sections().find((x) => x.id === sid) || null;
    if (!s || !s.id) { meta.innerHTML = ""; return; }
    // The area chip: from a concept, a way up to its area.
    meta.innerHTML = `<button type="button" class="kg2-area-chip" data-sid="${esc(s.id)}" title="How you're doing across this area">` +
      `<span class="kg2-chip-dot" style="background:${esc(s.color || "#999")}"></span>${esc(shortLabel(s))}</button>`;
    const chip = meta.querySelector("[data-sid]");
    if (chip) chip.addEventListener("click", () => {
      if (typeof window.deltaSelectKgSection === "function") window.deltaSelectKgSection(chip.dataset.sid);
    });
  };

  /* ---------------- views ---------------------------------------------- */
  const emptyHtml = () =>
    `<div class="kg2-placeholder"><strong>Click a concept</strong> to see how strong you are in it, ` +
    `or, with <strong>Area boxes</strong> on, an area's title to see the whole area.` +
    `<span class="kg2-placeholder-more">Use <strong>Practice ⤢</strong> on a concept to jump into practising it.</span></div>`;

  // One line: where the ladder has this learner, and their record on it.
  const progressLine = (id) => {
    const c = core();
    const row = c.row(id);
    if (!row) return c.signedIn() ? "No practice here yet." : "";
    const parts = [];
    if (row.state === "learned") parts.push("<strong>Learned</strong>");
    const sl = window.StageLadder;
    const stages = (sl && sl.STAGES) || [];
    const sidStage = sl && typeof sl.normalizeStage === "function" ? sl.normalizeStage(row.ladder_stage) : null;
    const i = sidStage ? stages.findIndex((s) => s.id === sidStage) : -1;
    if (i >= 0 && row.state !== "learned") parts.push(`Stage ${i + 1} of ${stages.length} · ${esc(stages[i].label)}`);
    const est = row.ladder_estimate;
    if (est && est.n > 0) parts.push(`${est.correct}/${est.n} correct`);
    else if (est && est.worked_seen) parts.push("lesson read, nothing graded yet");
    else parts.push("not attempted yet");
    return parts.join(" · ");
  };

  const conceptHtml = (id) => {
    const c = core();
    const k = c.kc(id) || {};
    const kp = c.kp(id) || {};
    const lm = c.lesson(k.lesson) || {};
    const x = c.readout(id);
    const row = c.row(id);
    const userOff = !!(row && row.pref && row.pref.enabled === false);
    const courseOff = !!(row && row.state === "disabled" && !userOff);
    const note = x.source === "extrapolated" ? "projected" : x.source === "topic" ? "topic-level"
      : x.source === "subtopic" ? "lesson-level" : !x.measured && Number.isFinite(x.r) ? "not measured yet" : "";
    let html = `<h2 class="kg2-title">${esc(kp.title || k.title || id)}</h2>`;
    if (lm.title && lm.title !== (kp.title || k.title)) html += `<div class="kgp-sub">${esc(lm.title)}</div>`;
    html += `<div class="kgp-score"><strong style="color:${c.color(x.r)}">${pct(x.r)}</strong>` +
      `<span>${esc(c.band(x.r))}${note ? ` <span class="kgp-dim">· ${esc(note)}</span>` : ""}</span></div>`;
    html += c.bar(x.r, x.ci, x.measured);
    const prog = progressLine(id);
    if (prog) html += `<div class="kgp-progress">${prog}</div>`;
    if (id === c.nextUp()) html += `<div class="kgp-flag is-next">Next up in practice</div>`;
    if (userOff) html += `<div class="kgp-flag is-off">Switched off — practice skips it. Turn it back on under the cog.</div>`;
    else if (courseOff) html += `<div class="kgp-flag is-off">Its course is off in the Courses tab.</div>`;
    const hasLesson = !!(kp.segments && kp.segments.length) || !!kp.concept_markdown;
    if (hasLesson) {
      html += `<button type="button" class="kgp-read" aria-expanded="${lessonOpen}">` +
        `${lessonOpen ? "Hide lesson" : "Read lesson"}<span aria-hidden="true">${lessonOpen ? "▴" : "▾"}</span></button>`;
      html += `<div class="kgp-lesson"${lessonOpen ? "" : " hidden"}>${lessonOpen ? lessonBody(id) : ""}</div>`;
    }
    return html;
  };

  const chip = (id) => {
    const c = core();
    const k = c.kc(id);
    if (!k) return "";
    const s = c.section(id);
    return `<button class="kg2-chip" data-goto="${esc(id)}" title="${esc(k.title)}">` +
      `<span class="kg2-chip-dot" style="background:${esc(s.color)}"></span>${esc(k.title)}</button>`;
  };
  const lessonBody = (id) => {
    const c = core();
    const ps = c.parents(id), ks = c.children(id);
    return c.lessonHtml(id) +
      `<div class="kg2-nav">` +
        `<div class="kg2-nav-col"><h4>Prerequisites (${ps.length})</h4>` +
          (ps.length ? ps.map(chip).join("") : `<span class="kg2-nav-empty">Foundation skill — none.</span>`) + `</div>` +
        `<div class="kg2-nav-col"><h4>Unlocks (${ks.length})</h4>` +
          (ks.length ? ks.map(chip).join("") : `<span class="kg2-nav-empty">Nothing downstream yet.</span>`) + `</div>` +
      `</div>`;
  };

  const sectionHtml = (id) => {
    const c = core();
    const st = sectionsApi() ? sectionsApi().stats(id) : null;
    if (!st) return emptyHtml();
    const b = st.bands;
    const seg = (n, cls, title) => (n ? `<span class="kgp-strip-${cls}" style="flex:${n}" title="${esc(title)}"></span>` : "");
    let html = `<h2 class="kg2-title">${esc(st.label)}</h2>`;
    html += `<div class="kgp-sub">${esc(st.fullLabel)} · ${st.total} concept${st.total === 1 ? "" : "s"}</div>`;
    html += `<div class="kgp-score"><strong style="color:${c.color(st.mean)}">${pct(st.mean)}</strong>` +
      `<span>${esc(c.band(st.mean))} <span class="kgp-dim">· average</span></span></div>`;
    html += window.DeltaMasteryBar
      ? window.DeltaMasteryBar.render({ value: st.mean, ci: null, measured: st.measured > 0, gates: false })
      : "";
    html += `<div class="kgp-strip" aria-hidden="true">` +
      seg(b.strong, "strong", "Strong") + seg(b.learning, "learning", "Learning") +
      seg(b.starting, "starting", "Just starting") + seg(b.none, "none", "No estimate") + `</div>`;
    html += `<ul class="kgp-counts">` +
      `<li><span class="kgp-key kgp-strip-strong"></span>${b.strong} strong</li>` +
      `<li><span class="kgp-key kgp-strip-learning"></span>${b.learning} learning</li>` +
      `<li><span class="kgp-key kgp-strip-starting"></span>${b.starting} just starting</li>` +
      (b.none ? `<li><span class="kgp-key kgp-strip-none"></span>${b.none} no estimate</li>` : "") + `</ul>`;
    html += `<div class="kgp-progress">${st.mastered} of ${st.active} mastered · ${st.measured} measured` +
      (st.off ? ` · <span class="kgp-dim">${st.off} switched off</span>` : "") + `</div>`;
    // Weakest first: what to work on next in this area is at the top. Off
    // concepts last; no estimate counts as weakest.
    const rows = st.rows.slice().sort((a, z) =>
      (a.off - z.off) || ((Number.isFinite(a.r) ? a.r : -1) - (Number.isFinite(z.r) ? z.r : -1)));
    html += `<h4 class="kgp-h">Concepts, weakest first</h4><ul class="kgp-kcs">` + rows.map((r) => {
      const k = c.kc(r.kc) || {};
      const w = Number.isFinite(r.r) ? Math.round(Math.max(0, Math.min(1, r.r)) * 100) : 0;
      return `<li><button type="button" data-goto="${esc(r.kc)}" class="${r.off ? "is-off" : ""}${r.measured ? "" : " is-inferred"}">` +
        `<span class="kgp-kc-name">${esc(k.title || r.kc)}</span>` +
        `<span class="kgp-kc-bar"><span style="width:${w}%;background:${c.color(r.r)}"></span></span>` +
        `<span class="kgp-kc-pct">${r.off ? "off" : pct(r.r)}</span></button></li>`;
    }).join("") + `</ul>`;
    return html;
  };

  /* ---------------- settings (the cog) --------------------------------- */
  const WEIGHT_PRESETS = [0.5, 0.75, 1, 1.5, 2];
  const prefFor = (id) => {
    const row = core().row(id);
    const p = row && row.pref ? row.pref : { enabled: true, weight: 1 };
    return { enabled: p.enabled !== false, weight: typeof p.weight === "number" ? p.weight : 1 };
  };
  const pctLabel = (w) => {
    const d = Math.round((w - 1) * 100);
    if (d === 0) return "normal priority";
    return d > 0 ? `${d}% more likely to come up` : `${-d}% less likely to come up`;
  };

  // What the registry and the server say about the concept. Read-only on
  // purpose: structure edits are proposals through instructor mode.
  const metadataHtml = (id) => {
    const c = core();
    const k = c.kc(id) || {};
    const kp = c.kp(id) || {};
    const row = c.row(id);
    const lm = c.lesson(k.lesson) || {};
    const cell = (a, v) => `<div class="kg2-md-k">${esc(a)}</div><div class="kg2-md-v">${v}</div>`;
    const num = (v, d = 3) => (typeof v === "number" ? String(parseFloat(v.toFixed(d))) : "—");
    const codes = (ids) => ids.map((p) => `<code>${esc(p)}</code>`).join(" ") || "none";
    let html = `<div class="kg2-md-grid">`;
    html += cell("Concept id", `<code>${esc(id)}</code>`);
    html += cell("Lesson", `${esc(lm.title || k.lesson || "—")} <span class="kg2-md-dim">(${esc(k.lesson || "—")})</span>`);
    html += cell("Topic", esc(c.topic(id) || "—"));
    html += cell("Prerequisites", codes(c.parents(id)));
    const enc = k.encompassing || {};
    html += cell("Encompasses", Object.keys(enc).map((p) => `<code>${esc(p)}</code> <span class="kg2-md-dim">×${esc(String(enc[p]))}</span>`).join(" ") || "none");
    html += cell("Integration index", `${c.integration(id)} <span class="kg2-md-dim">concepts reached by encompassing edges</span>`);
    html += cell("Unlocks", codes(c.children(id)));
    if (kp.segments && kp.segments.length) html += cell("Segments", String(kp.segments.length));
    if (row) {
      html += cell("Gate state", esc(row.state || "—"));
      html += cell("Mastery", `${num(row.mastery)} <span class="kg2-md-dim">${esc(row.tier || "")}${row.evidenced ? "" : " · not evidenced"}</span>`);
      html += cell("Coreness", `${row.coreness} <span class="kg2-md-dim">descendants</span>`);
      html += cell("Depth", String(row.depth));
      html += cell("Drills in bank", String(row.n_questions));
      html += cell("Frontier rank", row.frontier_rank == null ? "—" : String(row.frontier_rank + 1));
      html += cell("Ladder rung", esc(row.ladder_stage || "—"));
    }
    html += `</div>`;
    if (kp.notes_markdown) html += `<div class="kg2-md-notes"><h3>Author notes</h3>${c.md(kp.notes_markdown)}</div>`;
    return html;
  };

  const settingsHtml = () => {
    const c = core();
    let html = `<h2 class="kg2-title">Settings</h2>`;
    if (kc && c.kc(kc)) {
      const k = c.kc(kc);
      html += `<section class="kgp-block"><h3 class="kgp-h">This concept · ${esc(k.title)}</h3>`;
      if (!c.signedIn()) {
        html += `<div class="kg2-set-guest"><strong>Sign in</strong> to switch concepts off or change how often they come up. Settings are saved to your account.</div>`;
      } else {
        const pref = prefFor(kc);
        html += `<div class="kg2-set" data-kc="${esc(kc)}">` +
          `<label class="kg2-set-row kg2-set-toggle"><input type="checkbox" id="kg-set-enabled" ${pref.enabled ? "checked" : ""}>` +
            `<span><strong>Practice this concept</strong>` +
            `<small>Off = the queue skips it, and anything it unlocks is treated as if it were already learned.</small></span></label>` +
          `<div class="kg2-set-row kg2-set-weight ${pref.enabled ? "" : "is-off"}">` +
            `<div class="kg2-set-head"><strong>Priority weight</strong>` +
              `<output id="kg-set-out">${pref.weight.toFixed(2)} × — ${esc(pctLabel(pref.weight))}</output></div>` +
            `<input type="range" id="kg-set-range" min="0.25" max="4" step="0.25" value="${pref.weight}">` +
            `<div class="kg2-set-presets">${WEIGHT_PRESETS.map((w) =>
              `<button type="button" data-w="${w}" class="${Math.abs(w - pref.weight) < 1e-6 ? "active" : ""}">${w}×</button>`).join("")}</div>` +
            `<small>1 is normal. 1.5 sorts this concept as if it had 50% more dependents, so it reaches the front of the queue sooner; 0.75 as if it had 25% fewer. Only your queue changes — nobody else's.</small>` +
          `</div>` +
          `<div class="kg2-set-status" id="kg-set-status" aria-live="polite"></div></div>`;
      }
      html += `<details class="kgp-details"><summary>Concept details</summary>${metadataHtml(kc)}` +
        `<p class="kg2-md-foot">Read-only. Structure edits are proposals: open instructor mode and drag on the graph.</p></details>`;
      html += `</section>`;
    } else {
      html += `<p class="kgp-dim kgp-note">Pick a concept on the graph to change how often it comes up. ` +
        `To switch whole areas on or off, use <strong>Filter</strong> above the graph.</p>`;
    }
    const look = window.DeltaKgLook;
    html += `<section class="kgp-block"><h3 class="kgp-h">Graph</h3>` +
      `<label class="kg2-set-row kg2-set-toggle"><input type="checkbox" id="kg-set-dim-disabled" ${c.dimDisabled() ? "checked" : ""}>` +
        `<span><strong>Fade switched-off concepts</strong>` +
        `<small>Shows every concept you switched off in grey at low opacity. Stays in this browser.</small></span></label>` +
      (look ? `<label class="kg2-set-row kg2-set-toggle"><input type="checkbox" id="kg-set-shortcuts" ${look.shortcutsShown() ? "checked" : ""}>` +
        `<span><strong>Show shortcut links</strong>` +
        `<small>A link from A to C that a longer chain (A → B → C) already implies. Hidden by default: it adds lines, not information.</small></span></label>` : "") +
      `</section>`;
    return html;
  };

  // Writes are SERIALIZED: a slider fires several changes in a second, and two
  // PUTs in flight can land in either order — the server would keep whichever
  // arrived last while the controls show the last one clicked. One chain, in
  // click order, and the lattice is re-read once after the write that changed
  // it.
  let chain = Promise.resolve();
  const savePref = (id, patch) => {
    const run = () => savePrefNow(id, patch);
    chain = chain.then(run, run);
    return chain;
  };
  const savePrefNow = async (id, patch) => {
    const status = $("kg-set-status");
    const fn = typeof window.apiFetch === "function" ? window.apiFetch : null;
    if (!fn) { if (status) status.textContent = "Sign in to save."; return null; }
    if (status) status.textContent = "Saving…";
    try {
      const res = await fn(`/api/practice/kc-prefs/${encodeURIComponent(id)}`, {
        method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(patch),
      });
      if (!res || !res.ok) { if (status) status.textContent = `Not saved (${res ? res.status : "offline"}).`; return null; }
      const row = await res.json();
      const cur = core().row(id);
      if (cur) cur.pref = { enabled: row.enabled, weight: row.weight };
      if ($("kg-set-status")) $("kg-set-status").textContent = row.enabled ? "Saved." : "Saved — this concept is off for you.";
      // The frontier is the server's; re-read it so next-up and the gate
      // colours move with the change. Awaited, so the chain's next write sees
      // a current lattice.
      await core().refresh();
      window.dispatchEvent(new CustomEvent("delta:kc-prefs-changed", { detail: { kcs: [id], enabled: row.enabled } }));
      return row;
    } catch (_) {
      if ($("kg-set-status")) $("kg-set-status").textContent = "Not saved (network).";
      return null;
    }
  };

  const wireSettings = (body) => {
    const c = core();
    const dim = body.querySelector("#kg-set-dim-disabled");
    if (dim) dim.addEventListener("change", () => c.setDimDisabled(dim.checked));
    const sc = body.querySelector("#kg-set-shortcuts");
    if (sc) sc.addEventListener("change", () => { if (window.DeltaKgLook) window.DeltaKgLook.setShortcuts(sc.checked); });
    const box = body.querySelector(".kg2-set");
    if (!box) return;
    const id = box.dataset.kc;
    const enabled = box.querySelector("#kg-set-enabled");
    const range = box.querySelector("#kg-set-range");
    const out = box.querySelector("#kg-set-out");
    const weightRow = box.querySelector(".kg2-set-weight");
    const presets = box.querySelectorAll(".kg2-set-presets button");
    const paint = (w) => {
      if (range) range.value = String(w);
      if (out) out.textContent = `${Number(w).toFixed(2)} × — ${pctLabel(Number(w))}`;
      presets.forEach((b) => b.classList.toggle("active", Math.abs(Number(b.dataset.w) - Number(w)) < 1e-6));
    };
    const paintEnabled = (on) => {
      if (enabled) enabled.checked = on;
      if (weightRow) weightRow.classList.toggle("is-off", !on);
    };
    // A failed write puts every control back to what the server last said.
    const rollback = () => { const p = prefFor(id); paintEnabled(p.enabled); paint(p.weight); };
    if (enabled) enabled.addEventListener("change", async () => {
      paintEnabled(enabled.checked);
      if (!(await savePref(id, { enabled: enabled.checked }))) rollback();
    });
    const saveWeight = async (w) => { paint(w); if (!(await savePref(id, { weight: w }))) rollback(); };
    if (range) {
      range.addEventListener("input", () => paint(range.value));
      range.addEventListener("change", () => saveWeight(Number(range.value)));
    }
    presets.forEach((b) => b.addEventListener("click", () => saveWeight(Number(b.dataset.w))));
  };

  /* ---------------- render --------------------------------------------- */
  const render = () => {
    const body = $("kg-info-body");
    const c = core();
    if (!body || !c) return;
    paintHead();
    const pane = body.closest(".kg2-info");
    if (pane) pane.dataset.view = view;
    body.innerHTML = view === "concept" && kc ? conceptHtml(kc)
      : view === "section" && sid ? sectionHtml(sid)
      : view === "settings" ? settingsHtml()
      : emptyHtml();
    c.typeset(body);
    body.querySelectorAll("[data-goto]").forEach((b) =>
      b.addEventListener("click", () => c.select(b.getAttribute("data-goto"))));
    const read = body.querySelector(".kgp-read");
    if (read) read.addEventListener("click", () => {
      lessonOpen = !lessonOpen;
      const keep = body.scrollTop;
      render();
      body.scrollTop = keep;
    });
    if (view === "settings") wireSettings(body);
  };

  const toggleSettings = (force) => {
    const on = force == null ? view !== "settings" : !!force;
    if (on === (view === "settings")) return;
    if (on) { under = view; view = "settings"; }
    else view = under === "settings" ? "empty" : under;
    const body = $("kg-info-body");
    render();
    if (body) body.scrollTop = 0;
  };

  // lesson-graph.js calls these on a tap; each resets the scroll.
  const go = (v, id) => {
    if (v === "concept") { if (kc !== id) lessonOpen = false; kc = id; sid = null; }
    else if (v === "section") { sid = id; kc = null; }
    else { kc = null; sid = null; }
    view = v;
    under = v;
    render();
    const body = $("kg-info-body");
    if (body) body.scrollTop = 0;
  };

  // The numbers moved (a graded attempt, a lattice re-read, a pref switch):
  // repaint the reading in place. Not the settings view — a repaint would
  // drop a slider the learner is holding.
  const refresh = () => { if (view === "concept" || view === "section") {
    const body = $("kg-info-body");
    const keep = body ? body.scrollTop : 0;
    render();
    if (body) body.scrollTop = keep;
  } };
  window.addEventListener("delta:kc-readiness-changed", refresh);
  window.addEventListener("delta:kc-prefs-changed", refresh);

  window.DeltaKgPanel = {
    showEmpty: () => go("empty"),
    showConcept: (id) => go("concept", id),
    showSection: (id) => go("section", id),
    settings: toggleSettings,
    view: () => view,
  };
})();
