/* ================================================================
   LESSON KNOWLEDGE GRAPH — interactive Cytoscape.js embed for the
   Knowledge Graph tab.

   Built from scratch over the EXISTING lesson content:
     - lessons/kc_registry.json    → 64 knowledge components (KCs) + their
                                      `prereqs` (the edges) + lesson/topic.
     - lessons/lessons_structured.json → per-KC teaching content
                                      (concept / worked example / misconceptions).

   Each bubble is one KC, coloured by mastery or by area, laid out
   bottom-up so prerequisites sit beneath what they unlock. Click a bubble
   to light up its full prerequisite chain and show its reading in the side
   panel; click an AREA's header (kg-sections.js) to light up the whole area.
   The side panel itself — concept, area, the cog's settings — is
   concept-graph/kg-panel.js, which reads everything through
   `window.DeltaKgCore` below. Its "Practice" (maximize) button hands off to
   the real Delta-Drills practice screen in an overlay (openMaximize).

   Superseded concept-graph/graph-viz.js (the old ARENA 205-atom graph),
   which is no longer wired into index.html.

   Built on demand via window.deltaInitConceptGraph() — called by app.js
   switchTab() when the tab opens (Cytoscape can't size a display:none
   container, so we defer until the tab is visible).
   ================================================================ */
(() => {
  "use strict";

  /* ---- the AREA axis: which section of the course a concept is in ------
     Two colourings, switched above the map: Mastery, and Sections. (Categories
     and Math / code were removed with the legend on 2026-09-29 — the area
     headers on the map, kg-sections.js, now say what each colour is, and a
     reading with no header to name it would be a colour with no key.)

       SECTION - which ARENA section a concept belongs to, and the only
                 question it answers is: does ARENA's OWN curriculum test this
                 yet? A concept is 0.0 or 0.1 ONLY if an exercise the ARENA
                 authors wrote covers it. Everything else is OUR preparation
                 for those sections and sits below them: -1.0 for the Python
                 floor, -1.1 for the arrays / einops / tensor work.
                 So the prep tiers are muted and the two ARENA sections are
                 vivid: what pops out of the map is the part ARENA grades.

     Section membership is DATA, read from `lessons/arena_exercise_kcs.json` —
     the same file `practice/exercise-session.js` uses to decide which ARENA
     notebook headings get a Practice button. It is not inferred from lesson
     or KC ids, because "ARENA wrote a problem for this" is not a fact any id
     encodes, and it changes every time an exercise is mapped. Reading the map
     means the colour stays true with no edit here. It is loaded as soon as
     this file runs (loadArenaMap), not when the graph is first built, because
     the Courses tab's previews (course-graph.js) colour by it too.

     Node labels are painted #15151f, so every colour here is light enough to
     keep dark text legible on it. */

  // Notebook slug (the key in arena_exercise_kcs.json) -> the section it is.
  // A slug with no entry here still counts as ARENA-tested; it just lands in
  // ARENA_LATER, so mapping notebook 0-2 colours its concepts correctly on the
  // day it appears rather than silently demoting them to prep.
  const ARENA_SECTIONS = {
    "0-0": { id: "s00", label: "Section 0.0 — ARENA's own problems", color: "#4f9fe0" },
    "0-2": { id: "s02", label: "Section 0.2 — CNNs & ResNets", color: "#e8a765" },
    "0-1": { id: "s01", label: "Section 0.1 — ARENA's ray tracing", color: "#bb7de8" },
  };
  const ARENA_LATER = { id: "sNN", label: "Later ARENA sections", color: "#e0709a" };
  // Prep: ours, not ARENA's. Muted on purpose — see the SECTION note above.
  // Amber and NEUTRAL grey, not two tints of a hue ARENA also uses: -1.1 was a
  // steel blue and sat close enough to 0.0's blue that the one fact this mode
  // exists to show — which bubbles ARENA itself tests — came down to reading a
  // saturation difference across 44 nodes. Hue now carries it: grey and amber
  // are ours, blue and violet are theirs.
  const PREP_PYTHON = { id: "sm10", label: "Section −1.0 — Python", color: "#dfae74" };
  const PREP_ARRAYS = { id: "sm11", label: "Section −1.1 — arrays, einops, tensors (our prep)", color: "#b0b4c0" };
  // Maths is OURS too, and it is not array work: `math.*` used to fall through
  // to −1.1 and read as another einops tier. Its own tier, in the rose the
  // Mathematics family and the Math/Code mode both use, so the one colour means
  // "this is maths" on every reading of the map.
  const PREP_MATH = { id: "sm12", label: "Section −1.2 — the maths underneath (our prep)", color: "#f0a3a3" };
  // The standalone courses (course-registry.js) are not ARENA and not prep for
  // it: `leetcode.*` used to fall through to −1.1 and read as array prep. Each
  // gets its own tier, in a hue no ARENA or prep tier uses (Seth, 2026-09-25).
  // LeetCode is split into its own areas (leetcode-areas.js, Seth 2026-10-01:
  // one lime for the whole course showed no differences). This object is only
  // the fallback when that file did not load.
  const COURSE_LEETCODE = { id: "clc", label: "LeetCode Patterns — standalone course", color: "#b8dc6e" };
  const LC_AREAS = window.DeltaLeetcodeAreas || null;
  const COURSE_DELTA = { id: "cdd", label: "Delta Drills — how the app works (standalone course)", color: "#c4a8f0" };
  // Area order (headers, Filter rows): the prep tiers, ARENA's sections in
  // notebook order, then the standalone courses.
  const SECTION_ORDER = [PREP_PYTHON, PREP_ARRAYS, PREP_MATH]
    .concat(Object.keys(ARENA_SECTIONS).sort().map((k) => ARENA_SECTIONS[k]))
    .concat([ARENA_LATER])
    .concat(LC_AREAS ? LC_AREAS.list : [COURSE_LEETCODE])
    .concat([COURSE_DELTA]);

  // kc id -> section, built from the exercise map once it has loaded. Empty
  // until then, and empty forever if the fetch failed — which is why
  // `arenaMapLoaded` is tracked separately: with no map EVERY concept looks
  // like prep, and a map that says so without saying why would be a quiet
  // lie about what ARENA covers.
  let arenaSectionByKc = {};
  let arenaMapLoaded = false;
  const _buildArenaSections = (map) => {
    arenaSectionByKc = {};
    if (!map || typeof map !== "object") return;
    // Sorted so a concept exercised in two notebooks is filed under the
    // EARLIEST one — where the learner first meets it under ARENA's own name.
    Object.keys(map).sort().forEach((slug) => {
      if (slug.charAt(0) === "_") return;          // "_comment"
      const exercises = map[slug];
      if (!exercises || typeof exercises !== "object") return;
      const section = ARENA_SECTIONS[slug] || ARENA_LATER;
      Object.keys(exercises).forEach((fn) => {
        const kc = exercises[fn] && exercises[fn].kc;
        if (kc && !arenaSectionByKc[kc]) arenaSectionByKc[kc] = section;
      });
    });
    arenaMapLoaded = true;
  };
  // Once per page; a failure is not cached, so the graph's build retries it.
  let arenaMapLoad = null;
  const loadArenaMap = () => {
    arenaMapLoad ??= fetch("lessons/arena_exercise_kcs.json", { cache: "no-cache" })
      .then((r) => (r.ok ? r.json() : null))
      .then((map) => {
        if (!map) throw new Error("arena_exercise_kcs.json unavailable");
        _buildArenaSections(map);
        window.dispatchEvent(new CustomEvent("delta:kg-sections-ready"));
      })
      .catch(() => { arenaMapLoad = null; arenaMapLoaded = false; });
    return arenaMapLoad;
  };

  const FALLBACK = "#dddddd";
  const ACCENT = "#ffd23f"; // prerequisite-path highlight
  // "Next up" reuses the same yellow on purpose — it is the map's one attention
  // colour. The two never compete for meaning: the path highlight only exists
  // while a node is selected, and next-up is the standing marker, drawn with an
  // outer glow the path highlight does not have.
  const NEXT_UP = ACCENT;

  const $ = (id) => document.getElementById(id);
  /* Which ARENA section a concept sits in. The exercise map is the ONLY thing
     that can promote a concept out of prep — an id tells you the subject, not
     who wrote a problem for it. Which is also why Python is separated by id
     here and nothing else is: -1.0 vs -1.1 is a split of OUR material, and
     `python.*` is the whole of it. */
  const _sectionOf = (kc) => {
    const arena = arenaSectionByKc[kc];
    if (arena) return arena;
    const id = typeof kc === "string" ? kc : "";
    const lid = (kcById[kc] || {}).lesson || "";
    if (id.startsWith("leetcode.") || /^lc-/.test(lid)) return (LC_AREAS && LC_AREAS.of(id, lid)) || COURSE_LEETCODE;
    if (id.startsWith("deltadrills.") || /^dd-/.test(lid)) return COURSE_DELTA;
    if (id.startsWith("python.") || /^py-/.test(lid)) return PREP_PYTHON;
    if (id.startsWith("math.") || /^ma-/.test(lid)) return PREP_MATH;
    return PREP_ARRAYS;
  };
  const sectionColor = (kc) => _sectionOf(kc).color;

  /* The concept's topic ("Numpy", "Einops", "Einsum").
     `lessons/kc_registry.json` does NOT carry one — 0 of its 63 entries have a
     `topic` field, which is why the dock's meta line has always rendered a
     leading orphan separator. The server's report does carry it for all 63
     (kc_graph reads the fuller registry), so ask that first and treat a missing
     one as unknown rather than printing an empty string into a sentence. */
  const _kcTopic = (kc) => {
    const row = lattice && lattice.kcs ? lattice.kcs[kc] : null;
    const fromRow = row && typeof row.topic === "string" ? row.topic.trim() : "";
    if (fromRow) return fromRow;
    const local = ((kcById[kc] || {}).topic || "").trim();
    return local || null;
  };

  /* ---- mastery colouring (BKT posterior → red↔blue, gray = no estimate) ---- */
  const BKT_P_INIT = 0.10, BKT_HALF_LIFE_DAYS = 14.0;
  const UNKNOWN_COLOR = "#5b5b70";       // no estimate yet
  const DISABLED_COLOR = "#94949d";      // deliberately neutral in every colour mode
  /* Two reasons the server reports `state: "disabled"` (kc_prefs.is_disabled):
     the learner switched the concept off (row.pref.enabled === false), or it
     belongs to a standalone course — LeetCode Patterns, Delta Drills — that
     is off in the Courses tab (course_registry.course_off; the default for
     both). The first stays the neutral grey below. The second keeps its
     colour, faded: a whole course painted #94949d at 0.18 hid what the map
     is for, and "Off — your Settings" named a switch the learner never
     touched (Seth, 2026-09-25). */
  const _userOff = (row) => !!(row && row.pref && row.pref.enabled === false);
  const _courseOff = (row) => !!(row && row.state === "disabled" && !_userOff(row));
  const DIM_DISABLED_KEY = "dd_kg_dim_disabled";
  let colorMode = "mastery";             // "mastery" | "section"
  let dimDisabled = true;
  try { dimDisabled = localStorage.getItem(DIM_DISABLED_KEY) !== "false"; } catch (_) {}

  // Persisted engine state (guest: adaptive_state_guest) so the graph shows
  // mastery even before Practice has been opened this session.
  const _persistedState = () => {
    try {
      const email = (typeof authEmail === "string" && authEmail.trim()) ? authEmail.trim() : "guest";
      const raw = localStorage.getItem(`adaptive_state_${email}`) || localStorage.getItem("adaptive_state_guest");
      return raw ? JSON.parse(raw) : null;
    } catch (_) { return null; }
  };
  const _decay = (L, ts) => {
    if (!Number.isFinite(L)) return NaN;
    if (!ts) return L;
    const prev = Date.parse(ts);
    if (!Number.isFinite(prev)) return L;
    const days = Math.max(0, (Date.now() - prev) / 86400000);
    return BKT_P_INIT + (L - BKT_P_INIT) * Math.pow(0.5, days / BKT_HALF_LIFE_DAYS);
  };
  // Estimate for a KC in [0,1] plus WHERE it came from, or NaN when there is
  // none. Also returns `coveredW` — the share of THIS KC's own evidence that
  // actually exists — because the source alone does not size the confidence
  // band; see kc_interval.js. A "subtopic" or "extrapolated" reading carries
  // coveredW 0 by construction: neither rests on an observation of this concept.
  // Three sources, in order of precision:
  //   "atom"         — this KC's own decayed BKT posterior.
  //   "subtopic"     — the lesson subtopic's BKT mastery, shared by every KC
  //                    in that lesson.
  //   "topic"        — the server's number for this concept at TOPIC grain
  //                    (`kcTopicReadiness`): real belief the practice queue
  //                    acts on, shared with the concept's topic-mates. This is
  //                    what a finished placement test leaves behind — it seeds
  //                    per-atom posteriors from a per-AREA ability estimate
  //                    without creating a single attempt, so both
  //                    attempt-counting fallbacks above and below it come back
  //                    empty and the map went grey on a learner the queue had
  //                    already placed and locked.
  //   "extrapolated" — no evidence on this concept at all: projected from the
  //                    learner's overall demonstrated level, adjusted for how
  //                    hard this concept is (see `_extrapolated`).
  // The subtopic fallback exists because the graph's KC ids (`torch.dtype-astype`,
  // from kc_registry.json) and the backend's BKT atom ids (`argmax-prediction`,
  // from question_atom_tags.jsonl) are disjoint id spaces — zero overlap — so a
  // signed-in learner with real practice history had every bubble read
  // "Not yet estimated". Callers must surface the source: a subtopic number is
  // NOT a per-concept measurement, and an extrapolated one is not a measurement
  // at all. Presenting either as one overclaims.
  const kcReadinessInfo = (kc) => {
    if (typeof window.computeAtomReadiness === "function") {
      // NB: computeAtomReadiness coerces a non-finite fallback to 0, so we use
      // -1 as the "no posterior" sentinel (valid readiness is [0,1]). r >= 0 is
      // a real in-memory estimate; -1 falls through to the persisted read.
      const r = window.computeAtomReadiness(kc, -1);
      if (r >= 0) return { r, source: "atom", coveredW: 1 };
    }
    const s = _learnerState();
    const raw = s && s.atom_mastery ? s.atom_mastery[kc] : undefined;
    if (Number.isFinite(raw)) {
      return { r: _decay(raw, s.atom_last_ts ? s.atom_last_ts[kc] : null), source: "atom", coveredW: 1 };
    }
    // This KC's own evidence, under atom ids the graph does not share: the
    // server's reading first (the same code that gates practice), the browser's
    // crosswalk underneath for guests. Only the `measured` tier qualifies — a
    // topic proxy falls through to the labelled subtopic estimate below.
    if (typeof window.kcLatticeReadiness === "function") {
      const x = window.kcLatticeReadiness(kc, lattice, s, _decay);
      if (x) return x;
    }
    // Subtopic fallback. `p` (correctness rate) is the only field on the same
    // [0,1] scale as a posterior — `baseline` is a difficulty-weighted score in
    // [0,100]. Requires n > 0: p defaults to 0.5, and a never-practised subtopic
    // must stay grey rather than claim a coin-flip's worth of knowledge. No
    // decay is applied — the server already decayed it on write, and re-decaying
    // an EWMA with BKT's half-life would mix two different models.
    const sub = _subtopicState(kc);
    if (sub && Number.isFinite(sub.n) && sub.n > 0 && Number.isFinite(sub.p)) {
      // coveredW 0: the number is the LESSON's, borrowed. Whether this concept
      // has any evidence of its own is a separate question, answered per KC by
      // the lattice's covered_w in `_evidence` below.
      return { r: Math.max(0, Math.min(1, sub.p)), source: "subtopic", coveredW: 0 };
    }
    // Below the lesson average, above a projection: see the ordering note in
    // kc_lattice_read.js::kcTopicReadiness. Gated on the server's own
    // `evidenced` flag, so a learner who has answered nothing still gets an
    // honest grey map rather than 63 bubbles at the starting prior.
    if (typeof window.kcTopicReadiness === "function") {
      const t = window.kcTopicReadiness(kc, lattice);
      if (t) return t;
    }
    const ex = _extrapolated(kc);
    if (ex) return { r: ex.r, source: "extrapolated", coveredW: 0 };
    return { r: NaN, source: "none", coveredW: 0 };
  };
  const kcReadiness = (kc) => kcReadinessInfo(kc).r;
  // red (low) → amber → green (high); gray for no estimate.
  // #d64848 → #e2a92e → #34a862, restated in mastery-bar.js, why-graph.js,
  // arena-notebook-focus.js and how-it-works.css's .kg2-scale-bar.
  const masteryColor = (r) => {
    if (!Number.isFinite(r)) return UNKNOWN_COLOR;
    const t = Math.max(0, Math.min(1, r));
    // red (not known) → amber → green (known); Seth, 2026-09-25
    const lo = t < 0.5 ? [214, 72, 72] : [226, 169, 46], hi = t < 0.5 ? [226, 169, 46] : [52, 168, 98];
    const u = t < 0.5 ? t / 0.5 : (t - 0.5) / 0.5;
    const c = lo.map((v, i) => Math.round(v + (hi[i] - v) * u));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  };
  // Node labels sit under the circle, on the pane: its own text colour,
  // re-read on a theme switch (a mapper, so .kc-disabled's grey still wins).
  let nodeInk = "#c8cdd8";
  const readInk = () => {
    const el = $("kg-cy");
    if (el) nodeInk = getComputedStyle(el).color || nodeInk;
  };
  window.addEventListener("delta:theme-changed", () => { readInk(); if (cy) cy.style().update(); });
  const nodeColor = (kc) => {
    if (colorMode === "section") return sectionColor(kc);
    return masteryColor(kcReadiness(kc));
  };
  const masteryBand = (r) => {
    if (!Number.isFinite(r)) return "Not yet estimated";
    if (r < 0.30) return "Just starting";
    if (r < 0.60) return "Learning";
    if (r < 0.85) return "Proficient";
    return "Strong";
  };
  /* ---- learner-model readout (kg-panel.js draws it, via DeltaKgCore) ----
     Two layers of evidence, kept visually distinct because they are NOT the
     same measurement:
       - concept level: the per-KC BKT posterior (`atom_mastery`). Only the
         backend writes it today, so it is often absent offline.
       - subtopic level: the staircase/EWMA state the practice queue actually
         runs on (`subtopic_states`). Always present once the learner has
         answered anything in that subtopic — but SHARED by every KC in the
         lesson, which is why it is labelled as such rather than shown as this
         concept's number.
     Thresholds mirror the engine: 0.85 unlocks dependents, 0.95 = mastered. */
  const UNLOCK_T = 0.85, MASTERY_T = 0.95;

  let cy = null;
  let building = false;
  let kcById = {};        // id -> {id,lesson,topic,title,prereqs}
  let contentByKc = {};   // id -> kp block from lessons_structured
  let lessonMeta = {};    // lesson id -> {topic,title,subtopic_key}
  let parentsOf = {};     // id -> [prereq ids]
  let childrenOf = {};    // id -> [dependent ids]
  let selectedKc = null;

  /* How integrated a concept is = how many concepts its ENCOMPASSING edges
     reach transitively (a flashcard node reaches none; ResNet34 reaches the
     whole 0.2 stack). Only encompassing edges are followed — a plain
     prerequisite gates but is not exercised, so it does not count. */
  const integrationIndex = (id) => {
    const seen = new Set();
    const stack = [id];
    while (stack.length) {
      const cur = stack.pop();
      const enc = (kcById[cur] || {}).encompassing || {};
      Object.keys(enc).forEach((p) => {
        if (!seen.has(p) && kcById[p]) { seen.add(p); stack.push(p); }
      });
    }
    return seen.size;
  };

  /* ---------------- tiny markdown renderer ----------------------------- */
  const esc = (v) =>
    String(v ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  /* A `$…$` / `$$…$$` span is LaTeX, not prose: only escaped here, so `*`
     and backticks inside it stay maths. typesetMath() below runs KaTeX over
     the painted pane. Same split as practice/lessons.js MATH_SPAN. */
  const MATH_SPAN = /(\$\$[\s\S]+?\$\$|\$[^$\n]+?\$)/g;
  const inlineProse = (v) =>
    esc(v)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>");
  const inline = (v) =>
    String(v ?? "")
      .split(MATH_SPAN)
      .map((part, i) => (i % 2 ? esc(part) : inlineProse(part)))
      .join("");
  // KaTeX lives in practice/math-drill.js (DeltaMath.render); absent → raw text.
  const typesetMath = (root) => { if (window.DeltaMath) window.DeltaMath.render(root); };

  const md = (text, { renderCode = true } = {}) => {
    if (!text) return "";
    const lines = String(text).split("\n");
    const out = [];
    let i = 0, list = null, para = [];
    const flushPara = () => { if (para.length) { out.push("<p>" + inline(para.join(" ")) + "</p>"); para = []; } };
    const flushList = () => { if (list) { out.push("</" + list + ">"); list = null; } };
    while (i < lines.length) {
      const line = lines[i];
      const fence = line.match(/^```(.*)$/);
      if (fence) {
        flushPara(); flushList();
        const buf = []; i++;
        while (i < lines.length && !/^```/.test(lines[i])) { buf.push(lines[i]); i++; }
        i++;
        // A `figure <name>` fence keeps its info string so lesson-figures.js can
        // mount the figure here too. Only that one: carrying every fence's
        // name would hand this pane's code to the notebook painters.
        const figure = /^figure\s/.test(fence[1].trim()) ? ' data-fence="' + esc(fence[1].trim()) + '"' : "";
        if (renderCode) out.push("<pre" + figure + "><code>" + esc(buf.join("\n")) + "</code></pre>");
        continue;
      }
      const heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) { flushPara(); flushList(); out.push("<h4>" + inline(heading[2]) + "</h4>"); i++; continue; }
      const item = line.match(/^(\s*)([-*]|\d+\.)\s+(.*)$/);
      if (item) {
        flushPara();
        const kind = /\d+\./.test(item[2]) ? "ol" : "ul";
        if (list !== kind) { flushList(); out.push("<" + kind + ">"); list = kind; }
        let t = item[3]; i++;
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*]|\d+\.)\s/.test(lines[i])) { t += " " + lines[i].trim(); i++; }
        out.push("<li>" + inline(t) + "</li>");
        continue;
      }
      if (!line.trim()) { flushPara(); flushList(); i++; continue; }
      para.push(line.trim()); i++;
    }
    flushPara(); flushList();
    return out.join("\n");
  };

  /* ---------------- graph helpers -------------------------------------- */
  const ancestors = (id) => {
    const seen = new Set(), out = [];
    const q = [...(parentsOf[id] || [])];
    q.forEach((p) => seen.add(p));
    while (q.length) {
      const cur = q.shift();
      out.push(cur);
      for (const p of (parentsOf[cur] || [])) if (!seen.has(p)) { seen.add(p); q.push(p); }
    }
    return out;
  };

  /* ---------------- learner model card --------------------------------- */
  // Freshest engine state: the in-memory copy the practice scripts mutate,
  // falling back to what was persisted (the graph tab can be opened before
  // Practice has run this session).
  const _learnerState = () => {
    if (typeof adaptiveStateJson === "string" && adaptiveStateJson) {
      try { return JSON.parse(adaptiveStateJson); } catch (_) { /* fall through */ }
    }
    return _persistedState();
  };

  // The bank stores a subtopic under two names: the bare one ("Core array
  // literacy") offline, and the topic-prefixed composite ("Numpy: Core array
  // literacy") in backend mode. Match on either — see practice/README.md.
  const _subtopicKeys = (kc) => {
    const key = (lessonMeta[(kcById[kc] || {}).lesson] || {}).subtopic_key;
    if (!key) return [];
    const bare = key.includes(": ") ? key.slice(key.indexOf(": ") + 2) : key;
    return bare === key ? [key] : [key, bare];
  };

  // Bare subtopic names are not unique: "Core array literacy" and "Applied
  // patterns and advanced" each exist under both Numpy and Einsum. The backend
  // strips the topic prefix before sending state (subtopic_router
  // `_unprefix_subtopic`), so those two pairs arrive collided under one key —
  // the numbers a KC gets are then a merge of both topics. Name the merge
  // rather than pass it off as this lesson's own evidence.
  const _bareCollisions = () => {
    const byBare = {};
    Object.values(lessonMeta).forEach((m) => {
      const key = m && m.subtopic_key;
      if (!key) return;
      const bare = key.includes(": ") ? key.slice(key.indexOf(": ") + 2) : key;
      (byBare[bare] = byBare[bare] || new Set()).add(key);
    });
    return byBare;
  };

  const _subtopicState = (kc) => {
    const state = _learnerState();
    const states = state && state.subtopic_states;
    if (!states) return null;
    const keys = _subtopicKeys(kc);
    for (const k of keys) {
      if (!states[k]) continue;
      // Only the bare-key match can be a merge; the composite is unambiguous.
      const collided = k === keys[0] ? null : _bareCollisions()[k];
      const topics = collided && collided.size > 1
        ? [...collided].map((c) => (c.includes(": ") ? c.slice(0, c.indexOf(": ")) : c)).sort()
        : null;
      return { key: k, mergedTopics: topics, ...states[k] };
    }
    return null;
  };

  // How many KCs share this concept's subtopic — the honest denominator for
  // treating subtopic evidence as if it were about this one concept.
  const _siblingCount = (kc) => {
    const lid = (kcById[kc] || {}).lesson;
    const key = (lessonMeta[lid] || {}).subtopic_key;
    if (!key) return 0;
    return Object.values(kcById).filter((k) => (lessonMeta[k.lesson] || {}).subtopic_key === key).length;
  };

  /* ---------------- overall level → estimate for untouched concepts -------
     BKT is per-concept and independent: a concept with no attempts on it has
     no posterior, so every untouched bubble read "no estimate" no matter how
     much the learner had demonstrated elsewhere. That is the wrong default —
     if someone has missed most of what they've tried, the honest prior for the
     next thing is low, not blank.

     Same shape as the placement diagnostic's item model (diagnostic.py: 1PL on
     the bank's 0-100 difficulty scale), applied to ordinary practice instead of
     probes: centre the estimate on the learner's evidence-weighted mean mastery,
     then shift it in logit space by how much harder (or easier) this concept is
     than the ones they have actually been answering. Same level ⇒ same estimate;
     a much harder concept ⇒ lower. The slope is per SD of concept difficulty and
     capped — see EXTRAP_LOGIT_PER_SD.

     What this is NOT: a measurement. It carries no attempts of its own, so its
     interval is the spread of the learner's own results (floored — projecting
     across concepts is never tighter than that), never a Wilson interval, and
     both the bubble and the panel mark it as projected. With no graded evidence
     anywhere (a guest, or a fresh account) there is nothing to project from and
     the bubble stays grey. */
  // One logit of shift per standard deviation of concept difficulty, capped at
  // 1.5 SD. The cap matters: KC difficulties run 12→75 on a scale whose
  // item-level slope is 10 (diagnostic.py LOGISTIC_SCALE), so the raw 1PL shift
  // would drive a mid-range learner to 1% on the hardest concepts and 80% on the
  // easiest — a projection asserting more than the measurements it came from.
  // ±1.5 logits (odds ×/÷ 4.5) is as far as an inference with no attempts
  // behind it gets to move.
  const EXTRAP_LOGIT_PER_SD = 1.0;
  const EXTRAP_MAX_SHIFT = 1.5;
  const EXTRAP_MIN_SD = 0.15;            // floor on the learner's own result spread
  // A projection must never clear the unlock gate: "prerequisites ready" counts
  // concepts at/above UNLOCK_T, and nothing with zero attempts should count.
  const EXTRAP_CAP = 0.84;
  let kcDifficulty = null;               // concept-graph/kc_difficulty.json

  const _logit = (p) => { const q = Math.max(0.02, Math.min(0.98, p)); return Math.log(q / (1 - q)); };
  const _expit = (x) => 1 / (1 + Math.exp(-x));

  const _kcDifficulty = (kc) => {
    const e = kcDifficulty && kcDifficulty.kcs ? kcDifficulty.kcs[kc] : null;
    return e && Number.isFinite(e.d) ? e.d : NaN;
  };

  // Spread of concept difficulty across the whole map — the unit the projection
  // shifts in, so it stays meaningful if the bank's difficulty range changes.
  let _diffSd;
  const _difficultySd = () => {
    if (_diffSd !== undefined) return _diffSd;
    const v = Object.values((kcDifficulty && kcDifficulty.kcs) || {})
      .map((e) => e && e.d).filter(Number.isFinite);
    if (v.length < 2) return (_diffSd = NaN);
    const mean = v.reduce((a, b) => a + b, 0) / v.length;
    return (_diffSd = Math.sqrt(v.reduce((s, x) => s + (x - mean) * (x - mean), 0) / v.length));
  };

  // Mean difficulty of the concepts a subtopic covers — what "the difficulty of
  // what this learner has been practising" means. Static data, so memoised.
  const _subDiffMemo = {};
  const _bareName = (s) => (s.includes(": ") ? s.slice(s.indexOf(": ") + 2) : s);
  const _subtopicDifficulty = (key) => {
    if (!kcDifficulty || !key) return NaN;
    if (key in _subDiffMemo) return _subDiffMemo[key];
    const want = _bareName(key);
    const ds = [];
    Object.values(kcById).forEach((k) => {
      const sk = (lessonMeta[k.lesson] || {}).subtopic_key;
      if (!sk || _bareName(sk) !== want) return;
      const d = _kcDifficulty(k.id);
      if (Number.isFinite(d)) ds.push(d);
    });
    return (_subDiffMemo[key] = ds.length ? ds.reduce((a, b) => a + b, 0) / ds.length : NaN);
  };

  // The learner's overall level: evidence-weighted mean mastery, the difficulty
  // that mean was earned at, and how spread out their subtopics are.
  const learnerAbility = () => {
    const state = _learnerState();
    const states = state && state.subtopic_states;
    if (!states) return null;
    const parts = [];
    let wSum = 0, mSum = 0, bSum = 0, bW = 0;
    Object.keys(states).forEach((key) => {
      const s = states[key];
      if (!s || !Number.isFinite(s.n) || s.n <= 0 || !Number.isFinite(s.p)) return;
      const w = s.n, m = Math.max(0, Math.min(1, s.p));
      wSum += w; mSum += w * m;
      const b = _subtopicDifficulty(key);
      if (Number.isFinite(b)) { bSum += w * b; bW += w; }
      parts.push({ w, m });
    });
    if (!wSum) return null;
    const mBar = mSum / wSum;
    let varSum = 0;
    parts.forEach((p) => { varSum += p.w * (p.m - mBar) * (p.m - mBar); });
    return {
      mBar,
      bBar: bW ? bSum / bW : (kcDifficulty && Number.isFinite(kcDifficulty.mean) ? kcDifficulty.mean : NaN),
      sd: Math.max(EXTRAP_MIN_SD, Math.sqrt(varSum / wSum)),
      attempts: wSum,
      subtopics: parts.length,
    };
  };

  // Projected estimate + its range for a concept with no evidence of its own.
  // Null when there is nothing to project from, or no difficulty for this KC
  // (then the shift would be a guess dressed as arithmetic).
  const _extrapolated = (kc) => {
    const a = learnerAbility();
    if (!a) return null;
    const d = _kcDifficulty(kc);
    const sdD = _difficultySd();
    let delta = 0;
    if (Number.isFinite(d) && Number.isFinite(a.bBar) && Number.isFinite(sdD) && sdD > 0) {
      delta = ((d - a.bBar) / sdD) * EXTRAP_LOGIT_PER_SD;
      delta = Math.max(-EXTRAP_MAX_SHIFT, Math.min(EXTRAP_MAX_SHIFT, delta));
    }
    const shift = (m) => _expit(_logit(m) - delta);
    const r = Math.min(EXTRAP_CAP, shift(a.mBar));
    // The learner-spread band alone is NOT an uncertainty about this concept —
    // it is how varied their other results were, and a consistent learner
    // collapses it to the EXTRAP_MIN_SD floor. So it is handed to the interval
    // code as a WIDENING term only; the width itself comes from the effective
    // sample size of a structural prior (kc_interval.js), which is a fortieth of
    // one graded attempt. This function no longer computes its own band — one
    // place owns interval width for every source, or the sources drift apart
    // again, which is the bug this replaced.
    const spreadHalf = (shift(a.mBar + a.sd) - shift(a.mBar - a.sd)) / 2;
    return { r, spreadHalf, d, ...a };
  };

  const _pct = (v) => (Number.isFinite(v) ? Math.round(v * 100) + "%" : "—");
  /* ---- how wide the band should be (see concept-graph/kc_interval.js) ------
     The band used to be a Wilson interval over `subtopic_states[key].n` — the
     lesson's attempt count — for every KC in that lesson. A concept nobody had
     ever been tested on therefore drew the same band as one with real evidence,
     which is exactly backwards. Width is now driven by per-KC evidence:
     `covered_w` from the server's own KC report, discounted by how specific
     that evidence is to this concept, plus the (tiny) effective sample size of
     a structural prior. */
  const _ivl = () => window.DeltaKcInterval || null;

  // Per-KC evidence, assembled from every source that measures THIS concept
  // rather than its lesson.
  //   covered_w   — server truth (/api/practice/kc-lattice → kc_graph.kc_mastery).
  //                 Offline/guest it is not sent, so fall back to what the read
  //                 itself knew: a crosswalk hit reports its own covered weight,
  //                 a subtopic/extrapolated read has none by definition.
  //   reliability — the crosswalk's specificity of that evidence to this KC
  //                 (`shared_with` = 1/reliability − 1 sibling KCs share it).
  //   siblings    — the lesson-level denominator, used only when the crosswalk
  //                 has nothing to say about this KC.
  const _evidence = (kc, info) => {
    const sub = _subtopicState(kc);
    const nSub = Number.isFinite(sub && sub.n) ? sub.n : 0;
    const row = lattice && lattice.kcs ? lattice.kcs[kc] : null;
    // The server's covered_w is the truth for every reading EXCEPT a topic-grain
    // one, where it is the wrong question. `kc_graph.kc_mastery` counts an atom
    // as covered when it has a POSTERIOR, and `diagnostic.finish()` gives every
    // atom a posterior without a single attempt behind it — so after placement
    // covered_w is 1.0 across all 63 concepts while nothing has been observed
    // about any of them. A `topic` reading is exactly the case where no evidence
    // is specific to this concept, so take its own 0 and let `directEvidenceN`
    // short-circuit. Without this, nDirect = 1 × 1.0 × specificity, and
    // specificity falls back to 1/siblings whenever the browser's crosswalk
    // failed to load: one single-KC lesson in kc_registry.json away from a
    // placement seed being drawn as a measured concept with a tight band.
    const coveredW = info && (info.source === "topic" || info.source === "model")
      ? 0
      : (row && Number.isFinite(row.covered_w)
          ? row.covered_w
          : (info && Number.isFinite(info.coveredW) ? info.coveredW : 0));
    const cw = typeof window.kcCrosswalkInfo === "function" ? window.kcCrosswalkInfo(kc) : null;
    const reliability = cw && Number.isFinite(cw.reliability) ? cw.reliability : NaN;
    const siblings = _siblingCount(kc);
    const ivl = _ivl();
    const nDirect = ivl
      ? ivl.directEvidenceN({ nSub, coveredW, reliability, siblings })
      : 0;
    return {
      nSub, coveredW, reliability, siblings, nDirect,
      tier: (row && row.tier) || (cw && cw.tier) || null,
    };
  };

  // The one place a confidence band is computed, for every estimate source.
  // Returns {ci, half, nDirect, measured, ev, ex} — `ex` present only for a
  // projected estimate, whose learner-spread can widen (never narrow) the band.
  const _bandFor = (kc, info) => {
    const ev = _evidence(kc, info);
    const ex = info.source === "extrapolated" ? _extrapolated(kc) : null;
    const ivl = _ivl();
    if (!Number.isFinite(info.r)) return { ci: null, half: NaN, nDirect: ev.nDirect, measured: false, ev, ex };
    // kc_interval.js missing (a load failure) degrades to "we don't know", never
    // to a confident-looking band — the whole point of this change.
    if (!ivl) return { ci: [0, 1], half: 0.5, nDirect: ev.nDirect, measured: false, ev, ex };
    const band = ivl.kcInterval({
      r: info.r,
      nDirect: ev.nDirect,
      spreadHalf: ex ? ex.spreadHalf : NaN,
    });
    return { ...band, ev, ex };
  };

  // "Is this concept actually measured?" — one attempt's worth of direct
  // evidence is the threshold, matching the invariant kc_interval.js enforces.
  const _isMeasured = (kc) => {
    const ivl = _ivl();
    if (!ivl) return false;
    return _evidence(kc, kcReadinessInfo(kc)).nDirect >= 1;
  };

  // The mastery bar: fill = P(known), shaded band = the confidence interval,
  // ticks = the two thresholds the engine actually acts on.
  // `measured` false ⇒ the band is an inference, not a measurement; it is drawn
  // in a distinct hatch so a nearly-full-width stripe cannot be mistaken for a
  // very confident wide measurement.
  // The markup lives in `mastery-bar.js` so the Course content tab can draw the
  // same bar without a second copy. `gates` on, because the panel shows ONE
  // concept and 85%/95% are that concept's real thresholds; `unknownColor` is
  // passed so a no-estimate bar keeps this file's grey rather than the
  // module's default.
  const _masteryBar = (r, ci, measured) =>
    window.DeltaMasteryBar.render({
      value: r, ci, measured, gates: true, unknownColor: UNKNOWN_COLOR,
    });

  // The pane's CSS height assumes a fixed amount of chrome above it, but the
  // guest banner isn't always there — so the pane could end below the fold.
  // Size the pane to what's left.
  const fitWrap = () => {
    const wrap = document.querySelector(".kg2 .kg2-wrap");
    if (!wrap) return;
    const top = wrap.getBoundingClientRect().top;
    // Narrow AND actually stacked — the Colab edition hides the lesson pane, so
    // there is nothing under the graph to scroll to and the pane should fit.
    const info = document.querySelector(".kg2-info");
    const stacked = window.innerWidth <= 820 &&
      (!info || getComputedStyle(info).display !== "none");
    if (stacked) { wrap.style.height = ""; return; }
    wrap.style.height = Math.max(480, window.innerHeight - top - 14) + "px";
    if (cy) cy.resize();
  };

  /* ---------------- lesson content (drawn in kg-panel.js) --------------- */
  const panel = () => window.DeltaKgPanel || null;
  const announceSelection = (kind, id) =>
    window.dispatchEvent(new CustomEvent("delta:kg-selection-changed", { detail: { kind, id } }));

  /* The KP's teaching content, segment by segment — the same units, in the same
     order, that the practice screen pages through.

     `concept_markdown` and `worked_example_markdown` at the KP level are the
     segments CONCATENATED (compile_lessons.py builds them that way for the
     back-compat single-page renderers). Printing those two fields gave the pane
     one wall of every concept in the KP followed by one wall of every worked
     example — which is not a view the learner meets anywhere else, and reads as
     a different lesson from the one the practice page is showing. Walk the
     segments instead, so clicking a bubble shows the concept the lesson
     actually teaches, in the shape it teaches it. */
  const renderSegments = (kp) => {
    const segments = (kp.segments && kp.segments.length) ? kp.segments : null;
    if (!segments) {
      // Single-segment KPs never split, so the KP-level fields ARE the segment.
      return `<div class="kg2-concept">${md(kp.concept_markdown)}</div>` +
        (kp.worked_example_markdown
          ? `<div class="kg2-worked"><h3>Worked example</h3>${md(kp.worked_example_markdown)}</div>`
          : "");
    }
    return segments.map((seg, i) => {
      let out = `<div class="kg2-seg">`;
      out += `<div class="kg2-seg-num">Lesson ${i + 1} of ${segments.length}</div>`;
      if (seg.title) out += `<h3 class="kg2-seg-title">${esc(seg.title)}</h3>`;
      out += `<div class="kg2-concept">${md(seg.concept_markdown)}</div>`;
      if (seg.worked_example_markdown)
        out += `<div class="kg2-worked"><h3>Worked example</h3>${md(seg.worked_example_markdown)}</div>`;
      if (seg.watch_out_markdown)
        out += `<div class="kg2-watch"><h3>Watch out</h3>${md(seg.watch_out_markdown)}</div>`;
      return out + `</div>`;
    }).join("");
  };

  // The lesson under "Read lesson" in the panel: every segment, then the
  // KP's misconceptions. No title — the panel has already printed it.
  const lessonHtml = (id) => {
    const kp = contentByKc[id];
    if (!kp) return "";
    let html = renderSegments(kp);
    if (kp.misconceptions_markdown)
      html += `<div class="kg2-watch"><h3>Watch out</h3>${md(kp.misconceptions_markdown)}</div>`;
    return html;
  };

  /* ---------------- selection + highlight ------------------------------ */
  const selectNode = (id) => {
    if (!cy) return;
    const node = cy.getElementById(id);
    if (!node || node.empty()) return;
    const path = new Set([id, ...ancestors(id)]);
    cy.batch(() => {
      cy.elements().removeClass("hl hl-strong").addClass("faded");
      cy.nodes().forEach((el) => {
        if (path.has(el.id())) el.removeClass("faded").addClass(el.id() === id ? "hl-strong" : "hl");
      });
      cy.edges().forEach((e) => {
        if (path.has(e.source().id()) && path.has(e.target().id())) e.removeClass("faded").addClass("hl");
      });
    });
    selectedKc = id;
    if (panel()) panel().showConcept(id);
    // On the Colab edition the lesson the learner reads is in the notebook, so
    // choosing a concept sends the tab beside this one to the section that
    // teaches it. Inert on the normal deploy — concept-graph/kc-colab-route.js.
    if (window.DDGraphColab) window.DDGraphColab.onSelect(id);
    announceSelection("kc", id);
  };

  /* An AREA (a header on the map, kg-sections.js): every concept in it lights
     up, the edges between them too, everything else fades, and the panel shows
     the area's diagnostics. Same classes as a concept's chain, so the two
     selections look like one vocabulary. */
  const selectSection = (sid) => {
    if (!cy || !sid) return;
    const nodes = cy.nodes().filter((n) => _sectionOf(n.id()).id === sid);
    if (!nodes.length) return;
    const ids = new Set(nodes.map((n) => n.id()));
    cy.batch(() => {
      cy.elements().removeClass("hl hl-strong").addClass("faded");
      nodes.removeClass("faded").addClass("hl");
      cy.edges().forEach((e) => {
        if (ids.has(e.source().id()) && ids.has(e.target().id())) e.removeClass("faded").addClass("hl");
      });
    });
    // A header is outside the canvas, so no background tap unselected the
    // concept picked before; do it here.
    cy.nodes(":selected").unselect();
    selectedKc = null;
    if (panel()) panel().showSection(sid);
    if (window.DDGraphColab) window.DDGraphColab.onDeselect();
    announceSelection("section", sid);
  };

  const resetView = () => {
    if (cy) { cy.elements().removeClass("faded hl hl-strong"); cy.nodes(":selected").unselect(); }
    selectedKc = null;
    if (panel()) panel().showEmpty();
    if (window.DDGraphColab) window.DDGraphColab.onDeselect();
    announceSelection(null, null);
  };

  /* ---------------- maximize: focused practice page (own iframe) -------- */
  // Maximize opens the practice view for the KC as its OWN separate page — a
  // full-screen overlay hosting index.html?lesson=<kc>&embed=1 in an iframe
  // (embed=1 hides the app chrome so no tabs show). This is deliberately a
  // duplicate instance, NOT a tab switch — the graph stays live underneath, so
  // Minimize drops the learner right back onto the same node with its lesson.
  let overlay = null;

  const closeMaximize = () => {
    if (!overlay) return;
    overlay.classList.add("hidden");
    document.body.classList.remove("kg-maxi-open");
    const frame = overlay.querySelector("#kg-maxi-frame");
    if (frame) frame.src = "about:blank"; // tear down the embedded app (pyodide/audio)
    // The iframe is a separate app instance: anything it graded landed in
    // storage, not in this window's in-memory state. Re-read it so the card
    // and the node colours reflect the practice that just happened.
    Promise.all([_refreshLearnerState(), refreshLattice()])
      .then(() => recolor());
    // Back to the workflow: the node is still selected and its lesson is still
    // on the left — just re-centre it so focus returns cleanly.
    if (selectedKc && cy) {
      const n = cy.getElementById(selectedKc);
      if (n && !n.empty()) cy.animate({ center: { eles: n } }, { duration: 220 });
    }
  };

  const ensureOverlay = () => {
    if (overlay) return;
    overlay = document.createElement("div");
    overlay.id = "kg-maxi";
    overlay.className = "kg-maxi hidden";
    overlay.innerHTML =
      '<div class="kg-maxi-bar">' +
        '<span class="kg-maxi-title" id="kg-maxi-title"></span>' +
        '<button type="button" class="kg-maxi-min" id="kg-maxi-min" title="Back to the graph">⤡ Minimize</button>' +
      "</div>" +
      '<iframe class="kg-maxi-frame" id="kg-maxi-frame" title="Practice"></iframe>';
    document.body.appendChild(overlay);
    overlay.querySelector("#kg-maxi-min").addEventListener("click", closeMaximize);
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && overlay && !overlay.classList.contains("hidden")) closeMaximize();
    });
  };

  const openMaximize = (kc) => {
    if (!kc) return;
    ensureOverlay();
    const kcObj = kcById[kc];
    const title = overlay.querySelector("#kg-maxi-title");
    if (title) title.textContent = kcObj ? kcObj.title : "Practice";
    const frame = overlay.querySelector("#kg-maxi-frame");
    frame.src = "index.html?lesson=" + encodeURIComponent(kc) + "&embed=1";
    overlay.classList.remove("hidden");
    document.body.classList.add("kg-maxi-open");
  };

  /* ---------------- mastery handoff: iframe → graph -------------------- */
  // The embedded practice page posts `delta:kc-mastered` when the competency
  // bar crosses 0.95. Sequence: refresh the learner state the iframe just
  // wrote → drop back to the map → animate the node red→green → offer the next
  // concept. The iframe stays open for ~900ms so the bar visibly reaches the
  // gate before the overlay closes.
  const MASTERED_HOLD_MS = 900;
  const NODE_ANIM_MS = 1100;

  // The iframe is a second app instance: it writes mastery to localStorage (or
  // the backend), but THIS window's in-memory adaptiveStateJson — what
  // computeAtomReadiness reads — is stale until we re-read it. Without this the
  // node recolours to its old value and the animation lands on the wrong blue.
  const _refreshLearnerState = async () => {
    if (typeof practiceMode !== "undefined" && practiceMode === "backend" &&
        typeof loadBackendAdaptiveState === "function") {
      try { await loadBackendAdaptiveState(); return; } catch (_) { /* fall through */ }
    }
    try {
      const email = (typeof authEmail === "string" && authEmail.trim()) ? authEmail.trim() : "guest";
      const raw = localStorage.getItem(`adaptive_state_${email}`) || localStorage.getItem("adaptive_state_guest");
      // Bare assignment on purpose — adaptiveStateJson is a module-scope `let`
      // shared across the practice scripts; window.x = would shadow it.
      if (raw) adaptiveStateJson = raw;
    } catch (_) { /* keep the stale value rather than blanking it */ }
  };

  const _parseRgb = (css) => {
    const m = String(css || "").match(/rgba?\((\d+)[,\s]+(\d+)[,\s]+(\d+)/);
    return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : null;
  };

  const _animateNodeColor = (kc, fromCss, toCss) => {
    if (!cy) return;
    const node = cy.getElementById(kc);
    if (!node || node.empty()) return;
    const from = _parseRgb(fromCss);
    const to = _parseRgb(toCss);
    if (!from || !to) { node.style("background-color", toCss); return; }
    const started = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - started) / NODE_ANIM_MS);
      const eased = t * t * (3 - 2 * t);
      const c = from.map((v, i) => Math.round(v + (to[i] - v) * eased));
      node.style("background-color", `rgb(${c[0]},${c[1]},${c[2]})`);
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };

  // Next concept worth practising: a direct dependent of the mastered KC whose
  // OTHER prerequisites are already cleared (0.85 = the engine's unlock
  // threshold), least-mastered first. Returns null when nothing downstream is
  // ready — the learner picks their own next node instead.
  const _nextConcept = (kc) => {
    const ready = (id) => {
      const r = kcReadiness(id);
      return Number.isFinite(r) && r >= 0.85;
    };
    const candidates = (childrenOf[kc] || []).filter((id) => {
      const r = kcReadiness(id);
      if (Number.isFinite(r) && r >= 0.95) return false; // already mastered
      return (parentsOf[id] || []).every((p) => p === kc || ready(p));
    });
    if (!candidates.length) return null;
    return candidates.sort((a, b) => {
      const ra = kcReadiness(a), rb = kcReadiness(b);
      return (Number.isFinite(ra) ? ra : 0) - (Number.isFinite(rb) ? rb : 0);
    })[0];
  };

  let masteredToast = null;

  const _dismissToast = () => {
    if (masteredToast) { masteredToast.remove(); masteredToast = null; }
  };

  const _showMasteredToast = (kc, nextKc) => {
    _dismissToast();
    const kcObj = kcById[kc] || {};
    const nextObj = nextKc ? (kcById[nextKc] || {}) : null;
    masteredToast = document.createElement("div");
    masteredToast.className = "kg2-mastered-toast";
    masteredToast.innerHTML =
      `<div class="kg2-mastered-title">Mastered — ${esc(kcObj.title || kc)}</div>` +
      `<div class="kg2-mastered-body">${nextObj
        ? `This unlocks <strong>${esc(nextObj.title || nextKc)}</strong>.`
        : "Nothing downstream is waiting on it — pick any bubble to keep going."}</div>` +
      '<div class="kg2-mastered-actions">' +
        (nextObj ? '<button type="button" class="kg2-mastered-go">Practice it next</button>' : "") +
        '<button type="button" class="kg2-mastered-stay">Stay on the map</button>' +
      "</div>";
    document.body.appendChild(masteredToast);
    const go = masteredToast.querySelector(".kg2-mastered-go");
    if (go) go.addEventListener("click", () => {
      _dismissToast();
      selectNode(nextKc);
      openMaximize(nextKc);
    });
    masteredToast.querySelector(".kg2-mastered-stay").addEventListener("click", _dismissToast);
  };

  const _onKcMastered = async (kc) => {
    if (!kc || !kcById[kc]) return;
    const beforeCss = cy ? cy.getElementById(kc).style("background-color") : null;
    await _refreshLearnerState();
    setTimeout(() => {
      // Select BEFORE closing: closeMaximize re-centres on the selected node,
      // and the point of this moment is watching THIS node change colour.
      selectNode(kc);
      closeMaximize();
      if (cy) {
        const n = cy.getElementById(kc);
        if (n && !n.empty()) {
          cy.animate({ center: { eles: n }, zoom: Math.max(cy.zoom(), 1.1) }, { duration: 320 });
        }
      }
      const afterCss = nodeColor(kc);
      if (colorMode === "mastery") _animateNodeColor(kc, beforeCss, afterCss);
      // Every OTHER node also moved (FIRe credit reaches prerequisites), so
      // repaint the rest once the focused animation has finished.
      setTimeout(recolor, NODE_ANIM_MS + 60);
      _showMasteredToast(kc, _nextConcept(kc));
    }, MASTERED_HOLD_MS);
  };

  window.addEventListener("message", (e) => {
    // Same-origin only: the practice iframe is served from this app.
    if (e.origin !== window.location.origin) return;
    if (!e.data || e.data.type !== "delta:kc-mastered") return;
    _onKcMastered(e.data.kc);
  });

  /* ---------------- recolour ------------------------------------------ */
  // Unmeasured bubbles are washed out and dashed: they carry a colour because a
  // blank map was the wrong default, but they must never read as measured.
  // This used to key on `source === "extrapolated"` only, which let a concept
  // wearing its LESSON's number — the far commoner case — look solid. The test
  // is now the same one the band uses: less than one attempt's worth of
  // evidence bearing on this concept.
  // Border WIDTH is left to the stylesheet so the .hl prerequisite highlight
  // still wins; only the dash pattern is set per node.
  const recolor = () => {
    if (!cy) return;
    cy.batch(() => cy.nodes().forEach((n) => {
      const row = lattice && lattice.kcs ? lattice.kcs[n.id()] : null;
      const courseOff = dimDisabled && _courseOff(row);
      const disabled = dimDisabled && row && row.state === "disabled" && !courseOff;
      const inferred = !disabled && colorMode === "mastery" && !_isMeasured(n.id());
      n.style({
        "background-color": disabled ? DISABLED_COLOR : nodeColor(n.id()),
        "background-opacity": inferred ? 0.42 : 1,
        // Set here, not in the .kc-course-off rule: a per-element style
        // outranks any selector, so this is the only place it can win.
        "border-style": courseOff ? "dotted" : inferred ? "dashed" : "solid",
      });
    }));
    markNextUp();
    _refreshNoData();
    _announceReadiness();
  };

  /* Other surfaces read the learner model through the exports at the bottom of
     this file, and `delta:adaptive-state-changed` is NOT enough for them: this
     graph answers that event by re-fetching the lattice and only then
     recolouring, so anything repainting off the raw event reads the state this
     graph is about to replace. recolor() is the moment the numbers are settled,
     so that is what gets announced. Listener-only — nothing here reacts to it,
     so it cannot loop. */
  const _announceReadiness = () => {
    window.dispatchEvent(new CustomEvent("delta:kc-readiness-changed"));
  };

  /* ---- the cold-start notice --------------------------------------------
     At cold start every bubble is grey and dashed, which is correct and reads
     as a broken graph. The map cannot say why on its own, so it says it here:
     nothing has been answered, so there is nothing to colour. Top-LEFT, because
     .kg2-controls owns the top-right corner. It goes away the moment one
     concept has a measurement, and recolor() is the right place to decide that
     — it already runs on build, on every lattice refresh and on every graded
     attempt (`delta:adaptive-state-changed`). */
  const hasAnyMeasurement = () =>
    !!kcById && Object.keys(kcById).some((kc) => _isMeasured(kc));

  /* What the notice SAYS, which is not the same question as whether it shows.
     "No problems answered yet" was flatly false for anyone who had just
     finished the placement test: they answered six to fourteen probes, and the
     engine placed and locked the whole lattice off the result. It stayed false
     because placement produces no per-concept evidence at all, so
     `hasAnyMeasurement` can never go true from placement alone — the notice was
     both permanent and wrong for exactly the learner it was talking to.
     Exported for why-graph.js, so the landing map cannot drift from this. */
  const noDataText = () => {
    const pl = typeof window.kcPlacementStatus === "function" ? window.kcPlacementStatus() : null;
    if (pl && pl.completed) {
      return "Placed from your placement test — these are topic-level estimates, " +
             "not measurements of single concepts. Practise one to measure it.";
    }
    return "No problems answered yet — nothing on this map is measured.";
  };

  const _refreshNoData = () => {
    const graph = document.querySelector(".kg2-graph");
    if (!graph) return;
    let el = $("kg-nodata");
    if (!el) {
      el = document.createElement("div");
      el.id = "kg-nodata";
      el.className = "kg2-nodata";
      graph.appendChild(el);
    }
    // Set every time: the placement status arrives asynchronously, so the text
    // chosen at create time is the pre-placement one for the first paint.
    el.textContent = noDataText();
    el.hidden = hasAnyMeasurement();
  };

  /* ---------------- "next up": where the queue is pointing ----------------
     The practice queue picks weakest-first among what the learner can actually
     attempt, so this mirrors that rule on the graph: the lowest-readiness
     concept whose prerequisites all clear the unlock gate, ties broken by the
     easiest entry point. Concepts already at mastery drop out.

     It MIRRORS the queue, it is not the server's literal pick — the queue
     selects a subtopic and then a question inside it, and it may serve a
     placement probe instead. So the panel says "next up", not "you will be
     asked this". Getting that wrong would be a promise the app then breaks. */
  const _nextUpKc = () => {
    if (!kcById || !Object.keys(kcById).length) return null;
    let best = null;
    Object.keys(kcById).forEach((kc) => {
      const r = kcReadiness(kc);
      // No estimate at all still competes — an untouched concept is exactly
      // the kind of thing to practise next — at BKT's own prior.
      const score = Number.isFinite(r) ? r : BKT_P_INIT;
      if (score >= MASTERY_T) return;
      const parents = parentsOf[kc] || [];
      const locked = parents.some((p) => {
        const pr = kcReadiness(p);
        return !(Number.isFinite(pr) && pr >= UNLOCK_T);
      });
      if (locked) return;
      const d = _kcDifficulty(kc);
      const tie = Number.isFinite(d) ? d : 101;
      if (!best || score < best.score - 1e-9 ||
          (Math.abs(score - best.score) < 1e-9 && tie < best.tie)) {
        best = { kc, score, tie };
      }
    });
    return best ? best.kc : null;
  };

  let nextUpKc = null;

  /* ---------------- server truth ----------------------------------------
     `_nextUpKc` above is a MIRROR of the selection rule, recomputed in the
     browser. A mirror drifts: it was written against a weakest-first queue and
     the queue now gates on the KC lattice, so the two could highlight different
     nodes and the graph would be showing something the app was not doing.

     /api/practice/kc-lattice returns the backend's own `kc_graph.kc_report` —
     the same function that gates practice. When it answers, it WINS: the ring
     lands on the node the queue will actually serve from, and locked nodes are
     drawn locked because the server says they are locked, not because the
     browser guessed. The local mirror stays as the offline/guest fallback,
     where there is no server to ask. */
  let lattice = null;

  async function refreshLattice() {
    // The fetch + cache live in kc_lattice_read.js so why-graph.js, which
    // borrows this file's reader but never runs build(), reads the same body
    // instead of asking with a permanently null lattice. `force` because this
    // runs after a graded attempt, where a cached report is the stale one.
    if (typeof window.loadKcLattice === "function") {
      lattice = await window.loadKcLattice(true);
      return;
    }
    try {
      const fn = typeof window.apiFetch === "function" ? window.apiFetch : fetch;
      const res = await fn("/api/practice/kc-lattice");
      if (!res || !res.ok) { lattice = null; return; }
      const data = await res.json();
      lattice = window.kcLatticeNote ? window.kcLatticeNote(data) : (data && data.kcs ? data : null);
    } catch (_) {
      lattice = null;   // guest / offline — fall back to the local mirror
    }
  }

  /* Placement status decides COPY only — never a number — so it is fetched
     alongside the lattice and its failure costs wording, not the map. Once:
     a finished placement does not un-finish. */
  function refreshPlacement() {
    if (typeof window.loadPlacementStatus !== "function") return Promise.resolve(null);
    return window.loadPlacementStatus();
  }

  // Yellow ring on the concept, and the same yellow along the edges feeding it,
  // so the route INTO it is visible rather than just the destination.
  const markNextUp = () => {
    if (!cy) return;
    cy.elements(".next-up, .next-up-edge").removeClass("next-up next-up-edge");

    // Gate state, straight from the server's report. This is the difference
    // between a diagram of the curriculum and a picture of the tutor: a locked
    // node is one the learner genuinely cannot be served yet.
    cy.nodes().removeClass("kc-locked kc-frontier kc-disabled kc-course-off");
    if (lattice) {
      cy.nodes().forEach((n) => {
        const row = lattice.kcs[n.id()];
        if (!row) return;
        if (dimDisabled && _courseOff(row)) n.addClass("kc-course-off");
        else if (dimDisabled && row.state === "disabled") n.addClass("kc-disabled");
        else if (colorMode === "mastery" && row.state === "locked") n.addClass("kc-locked");
        else if (colorMode === "mastery" && row.state === "frontier") n.addClass("kc-frontier");
      });
    }

    nextUpKc = colorMode === "mastery"
      ? ((lattice && lattice.next_kc) || _nextUpKc())
      : null;
    if (!nextUpKc) return;
    const node = cy.getElementById(nextUpKc);
    if (!node || !node.length) return;
    node.addClass("next-up");
    node.incomers("edge").addClass("next-up-edge");
  };

  /* ---------------- stylesheet ----------------------------------------- */
  // The node's SHAPE and the edges come from concept-graph/kg-look.js (a
  // learner setting: labelled boxes or dots; soft curved edges; shortcuts
  // hidden). Everything below them is this file's own state styling — gate
  // classes, the highlighted chain, next-up — and is the same in either look.
  // Rebuilt whole on `delta:kg-look-changed`; per-node bypasses (recolor)
  // survive a stylesheet swap.
  // Read at call time: kg-look.js is a separate deferred script.
  const LOOK = () => window.DeltaKgLook || null;
  const sheet = () => [
    { selector: "node", style: Object.assign(
        LOOK() ? LOOK().node(1, () => nodeInk) : {
          "shape": "round-rectangle", "label": "data(label)",
          "width": "label", "height": "label", "padding": "13px",
          "text-wrap": "wrap", "text-max-width": "120px",
          "text-valign": "center", "text-halign": "center",
          "font-size": 13, "font-weight": 600, "color": "#15151f",
          "border-width": 1, "border-color": "rgba(0,0,0,.28)",
        },
        { "background-color": (n) => nodeColor(n.id()),
          "transition-property": "opacity, border-width, border-color", "transition-duration": "120ms" }) },
    ...(LOOK() ? LOOK().edgeRules() : [
      { selector: "edge", style: {
          "curve-style": "bezier", "width": 1.8, "line-color": "#e3212c",
          "target-arrow-shape": "triangle", "target-arrow-color": "#e3212c", "arrow-scale": 0.8, "opacity": 0.9,
      }},
      { selector: "edge[kind = 'prereq']", style: { "line-style": "dashed", "line-dash-pattern": [6, 4], "width": 1.2, "opacity": 0.55 } },
      { selector: "edge[kind = 'encompassing']", style: { "line-style": "solid", "width": (e) => 1.5 + 3 * (e.data("w") || 0), "opacity": 0.95 } },
    ]),
    { selector: ".faded", style: { "opacity": 0.1 } },
    { selector: "node.hl", style: { "opacity": 1, "border-width": 3, "border-color": ACCENT, "z-index": 50 } },
    { selector: "node.hl-strong", style: { "opacity": 1, "border-width": 5, "border-color": ACCENT, "font-size": 12, "z-index": 99 } },
    { selector: "edge.hl", style: { "opacity": 1, "width": 3, "line-color": ACCENT, "target-arrow-color": ACCENT, "z-index": 60 } },
    // Where the queue is pointing. The outline sits OUTSIDE the border, so
    // a dashed projected node keeps showing that it is projected instead of
    // having the marker overwrite that fact.
    { selector: "node.next-up", style: {
        "border-width": 4, "border-color": NEXT_UP, "border-style": "solid",
        "outline-width": 6, "outline-color": NEXT_UP, "outline-opacity": 0.35,
        "z-index": 80,
    }},
    // Gate state. Locked = the queue will not serve this yet, so it is
    // dimmed and desaturated; frontier = unlocked and unfinished, the set
    // the next-up ring is chosen from, so it keeps full presence.
    { selector: "node.kc-locked", style: {
        "opacity": 0.32, "border-style": "dotted", "z-index": 1,
    }},
    // Switched off by the learner (Settings tab). Not locked — nothing is
    // waiting on it — just out of the queue, so it fades harder than a
    // locked node and loses its border entirely.
    { selector: "node.kc-disabled", style: {
        "opacity": 0.18, "border-width": 0, "color": "#505058", "z-index": 0,
    }},
    // In a standalone course that is off in the Courses tab: not served,
    // but still the course it is, so it keeps its colour, faded.
    { selector: "node.kc-course-off", style: { "opacity": 0.45, "z-index": 1 } },
    { selector: "node.kc-frontier", style: {
        "opacity": 1, "border-width": 2.5, "border-color": NEXT_UP,
        "border-opacity": 0.55, "z-index": 40,
    }},
    { selector: "edge.next-up-edge", style: {
        "width": 3.5, "line-color": NEXT_UP, "target-arrow-color": NEXT_UP,
        "opacity": 1, "z-index": 70,
    }},
  ];
  window.addEventListener("delta:kg-look-changed", () => {
    if (!cy) return;
    cy.style(sheet());
    // A box and a dot are different sizes: the layout has to run again.
    if (window.deltaKgView && typeof window.deltaKgView.relayout === "function") window.deltaKgView.relayout();
  });

  /* ---------------- build ---------------------------------------------- */
  async function build() {
    if (cy || building) return;
    const container = $("kg-cy");
    if (!container || typeof cytoscape === "undefined") return;
    readInk();
    building = true;

    try { if (window.cytoscapeDagre) cytoscape.use(window.cytoscapeDagre); } catch (_) {}

    let registry, structured;
    try {
      [registry, structured] = await Promise.all([
        fetch("lessons/kc_registry.json", { cache: "no-cache" }).then((r) => r.json()),
        fetch("lessons/lessons_structured.json", { cache: "no-cache" }).then((r) => r.json()),
      ]);
    } catch (e) {
      if ($("kg-status")) $("kg-status").textContent = "Couldn't load the lesson graph data.";
      building = false;
      return;
    }

    // Per-concept difficulty for the projected estimates. Optional on purpose:
    // without it `_extrapolated` falls back to the learner's flat overall level
    // rather than blocking the graph.
    try {
      kcDifficulty = await fetch("concept-graph/kc_difficulty.json", { cache: "no-cache" }).then((r) =>
        r.ok ? r.json() : null
      );
    } catch (_) { kcDifficulty = null; }

    // Which concepts ARENA's OWN exercises cover — the whole of the section
    // axis. Optional like the difficulty table: a missing map costs the broad
    // colouring, not the graph.
    await loadArenaMap();
    // The eager load at boot may have failed; this build gets one more try.
    if (!arenaMapLoaded) await loadArenaMap();

    // The KC->atom join, so the 20 measurable concepts can read the mastery the
    // backend already holds instead of falling through to a lesson average.
    // Optional for the same reason as kc_difficulty: a missing file costs
    // precision, not the graph.
    try {
      if (typeof window.loadKcCrosswalk === "function") await window.loadKcCrosswalk();
    } catch (_) {}

    (registry.lessons || []).forEach((l) => { lessonMeta[l.id] = l; });
    (registry.kcs || []).forEach((k) => {
      kcById[k.id] = k;
      parentsOf[k.id] = [...(k.prereqs || [])];
      childrenOf[k.id] = childrenOf[k.id] || [];
    });
    Object.values(kcById).forEach((k) => {
      (k.prereqs || []).forEach((p) => {
        if (!childrenOf[p]) childrenOf[p] = [];
        childrenOf[p].push(k.id);
      });
    });
    (structured.lessons || []).forEach((l) => l.kps.forEach((kp) => { contentByKc[kp.kc] = kp; }));
    // Lesson-less courses (LeetCode) have no KP; their bubbles open a concept
    // page instead — concept-graph/lessonless-concepts.js. A real KP wins.
    try {
      const pages = window.DDLessonlessConcepts ? await window.DDLessonlessConcepts.load() : {};
      Object.entries(pages).forEach(([kc, kp]) => { if (kcById[kc] && !contentByKc[kc]) contentByKc[kc] = kp; });
    } catch (_) {}

    const elements = [];
    Object.values(kcById).forEach((k) => {
      elements.push({ data: { id: k.id, label: k.title, lesson: k.lesson } });
    });
    // Two kinds of edge share one arrow head. A prerequisite edge is the
    // gate: the parent must be known first. An ENCOMPASSING edge is a
    // prerequisite edge with a weight in (0,1]: working the child also works
    // that much of the parent (SPEC_CHAPTER0_GRAPH.md). Encompassing edges are
    // always a subset of the prereqs — the registry audit enforces it — so a
    // node's integration index is just the size of its transitive
    // encompassing closure, computed here rather than stored.
    let ei = 0;
    Object.values(kcById).forEach((k) => {
      const enc = k.encompassing || {};
      (k.prereqs || []).forEach((p) => {
        if (!kcById[p]) return;
        const w = typeof enc[p] === "number" ? enc[p] : 0;
        elements.push({ data: { id: "e" + (ei++), source: p, target: k.id, w, kind: w > 0 ? "encompassing" : "prereq" } });
      });
    });

    cy = cytoscape({
      container,
      elements,
      wheelSensitivity: 0.25,
      minZoom: 0.1, maxZoom: 3,
      style: sheet(),
      layout: { name: window.cytoscapeDagre ? "dagre" : "cose",
        rankDir: "BT", nodeSep: 26, rankSep: 150, edgeSep: 12, animate: false, fit: true, padding: 40,
        nodeDimensionsIncludeLabels: true },
    });
    // How it is drawn — node look, soft curved edges, hidden shortcuts — is
    // concept-graph/kg-look.js's; see sheet().
    if (LOOK()) { LOOK().markShortcuts(cy); LOOK().curve(cy); }

    if ($("kg-status")) $("kg-status").style.display = "none";

    cy.on("tap", "node", (evt) => selectNode(evt.target.id()));
    cy.on("tap", (evt) => { if (evt.target === cy) resetView(); });

    if ($("kg-fit")) $("kg-fit").onclick = () => cy.fit(undefined, 36);
    const maxBtn = $("kg-maximize");
    if (maxBtn) maxBtn.onclick = () => openMaximize(maxBtn.dataset.kc);

    /* ---- colour-mode toggle (Mastery ↔ Sections) ---- */
    const controls = document.querySelector(".kg2-controls");
    window.dispatchEvent(new CustomEvent("delta:practice-target-graph-ready"));
    if (controls && !$("kg-colormode")) {
      const seg = document.createElement("div");
      // Styled by ID. It carried `.kg2-seg` and that is the lesson-segment
      // class in the pane opposite — see how-it-works.css.
      seg.id = "kg-colormode";
      // Two readings (2026-09-29): how strong you are, and which area a
      // concept is in. The area headers (kg-sections.js) follow the switch —
      // in Mastery they wear the area's average reading.
      seg.innerHTML =
        '<button type="button" data-mode="mastery" class="active">Mastery</button>' +
        '<button type="button" data-mode="section">Sections</button>';
      controls.insertBefore(seg, controls.firstChild);
      seg.querySelectorAll("button").forEach((b) =>
        b.addEventListener("click", () => {
          colorMode = b.dataset.mode;
          seg.querySelectorAll("button").forEach((x) => x.classList.toggle("active", x === b));
          recolor();
          window.dispatchEvent(new CustomEvent("delta:kg-colormode-changed", { detail: { mode: colorMode } }));
        }));
    }

    fitWrap();
    window.addEventListener("resize", fitWrap);

    // Recolour when the learner model changes (a graded attempt updates BKT);
    // the panel repaints off recolor's `delta:kc-readiness-changed`.
    // A graded attempt can move a KC over the unlock threshold, which changes
    // the gate for everything downstream — so re-ask the server what the
    // lattice looks like now instead of repainting stale state.
    window.addEventListener("delta:adaptive-state-changed", () => {
      refreshLattice().then(() => recolor());
    });

    refreshLattice().then(() => recolor());
    refreshPlacement().then(() => { _refreshNoData(); _announceReadiness(); });
    recolor();
    resetView();
    building = false;
  }

  /* Jump straight to one concept from outside the graph — the Practice tab's
     "See in knowledge graph" button, so a question can be audited against the
     node the tutor thinks it is teaching. Builds the graph first if this is the
     first visit; an unknown id leaves the view untouched rather than clearing
     the selection, which would look like the concept does not exist. */
  window.deltaFocusConceptGraphKc = function (kc) {
    if (!kc) return false;
    let tries = 0;
    const focus = () => {
      if (!cy) {
        build();
        if (tries++ < 80) setTimeout(focus, 120);
        return;
      }
      if (!kcById[kc]) return;
      fitWrap();
      cy.resize();
      selectNode(kc);
      const node = cy.getElementById(kc);
      if (node && node.length) {
        cy.animate({ center: { eles: node }, zoom: 1.15 }, { duration: 260 });
      }
    };
    focus();
    return true;
  };

  /* ---- read-only exports for the landing page's map -----------------------
     concept-graph/why-graph.js draws the same 63 concepts on "Why this app
     exists" and offers a cold-start / your-mastery switch. Its "your mastery"
     side has to be THIS reading or the two surfaces disagree about the same
     learner — so it borrows the reader rather than reimplementing the
     atom → lattice → subtopic → extrapolation ladder. Read-only: neither
     function touches graph state, and both answer before build() has run
     (`lattice` is simply null, which is the offline path anyway). */
  window.deltaKcReadinessInfo = (kc) => kcReadinessInfo(kc);
  window.deltaKcMasteryColor = masteryColor;
  window.deltaKcMasteryBand = masteryBand;
  window.deltaKcIsMeasured = (kc) => _isMeasured(kc);
  window.deltaKcHasAnyMeasurement = () => hasAnyMeasurement();
  window.deltaKcNoDataText = () => noDataText();
  /* why-graph.js needs the server's report before it can show "your mastery",
     and it must come through HERE. Calling kc_lattice_read.js's loader directly
     would fill the shared cache while this file's own `lattice` stayed null —
     and `lattice` is what kcReadinessInfo passes to `kcLatticeReadiness`, so
     the 23 concepts with atom-level evidence would silently drop to their
     topic-level reading. One refresher, one owner. */
  window.deltaRefreshKcLattice = () => Promise.all([refreshLattice(), refreshPlacement()]);

  /* The live Cytoscape instance, for a surface that LAYERS on this graph
     rather than drawing its own. instructor-review.js hosts this very
     container full-bleed, and instructor-graph-edit.js then lets an instructor
     edit it: edge taps (which this file has no handler for — a learner taps
     bubbles, never edges), deletions, direction changes, proposed concepts.

     🔴 NOT read-only, and the contract is the way back. The borrower may add
     and remove elements, but every change is one `remove`/`add` pair whose
     removal collection it keeps, and it must `.restore()` all of them before
     handing the container back — the graph a learner is served is THIS one.
     It may not touch the stylesheet (proposals are styled inline) and it may
     not repaint gate state; the lesson pane and the gate ticks stay this
     file's alone. Null until build() has run, which is the same "not yet"
     every other export here answers. */
  window.deltaConceptGraphCy = () => cy;
  // Which ARENA section a concept belongs to ({id, label, color}); read by
  // graph-views.js (grouped layout, condensed view, area order), kg-sections.js
  // (the headers) and course-graph.js (the Courses previews).
  window.deltaKcSection = (kc) => _sectionOf(kc);
  window.deltaKcSectionOrder = () => SECTION_ORDER.slice();
  // The switch above the map; kg-sections.js colours its headers by it.
  window.deltaKgColorMode = () => colorMode;
  // A header was clicked (kg-sections.js), or the panel's area chip.
  window.deltaSelectKgSection = (sid) => selectSection(sid);

  /* The side panel's one door into this file (kg-panel.js). Everything the
     panel shows is read HERE, from the same reader the bubbles are painted
     with, so the two cannot disagree about a number; its only writes are the
     learner's own kc-prefs, after which it calls `refresh`. */
  window.DeltaKgCore = {
    kc: (id) => kcById[id] || null,
    kp: (id) => contentByKc[id] || null,
    lesson: (lid) => lessonMeta[lid] || null,
    topic: (id) => _kcTopic(id),
    section: (id) => _sectionOf(id),
    parents: (id) => (parentsOf[id] || []).slice(),
    children: (id) => (childrenOf[id] || []).slice(),
    integration: (id) => integrationIndex(id),
    row: (id) => (lattice && lattice.kcs ? lattice.kcs[id] || null : null),
    signedIn: () => !!(lattice && lattice.kcs),
    // {r, source, ci, measured} — the bar's inputs, from the one band function.
    readout: (id) => {
      const info = kcReadinessInfo(id);
      const band = _bandFor(id, info);
      return { r: info.r, source: info.source, ci: band.ci, measured: band.measured };
    },
    nextUp: () => nextUpKc,
    color: masteryColor,
    band: masteryBand,
    bar: _masteryBar,
    esc, md,
    typeset: typesetMath,
    lessonHtml,
    select: (id) => selectNode(id),
    // Recolour even when the re-read fails: a caller that patched the cached
    // rows (kg-toolbar.js's area switch) still needs the bubbles repainted.
    refresh: async () => { try { await refreshLattice(); } finally { recolor(); } },
    // Nothing selected: graph-views.js calls it when the course changes.
    deselect: () => resetView(),
    dimDisabled: () => dimDisabled,
    setDimDisabled: (on) => {
      dimDisabled = !!on;
      try { localStorage.setItem(DIM_DISABLED_KEY, String(dimDisabled)); } catch (_) {}
      recolor();
    },
  };

  // Load the area map now, not at build: the Courses previews colour by it
  // before the graph has ever been opened.
  loadArenaMap();
  // One concept's lesson as HTML — the segments, worked examples and "Watch
  // out" the pane renders — for a surface that shows a lesson without this
  // graph (course-builder/course-builder.js). Pure: `kp` is a
  // lessons_structured.json KP or a lesson-less concept page.
  window.deltaKcLessonHtml = (kp) =>
    `<h2 class="kg2-title">${esc(kp.title || kp.kc || "")}</h2>` + renderSegments(kp) +
    (kp.misconceptions_markdown ? `<div class="kg2-watch"><h3>Watch out</h3>${md(kp.misconceptions_markdown)}</div>` : "");

  window.deltaInitConceptGraph = function () {
    if (cy) { fitWrap(); cy.resize(); cy.fit(undefined, 36); return; }
    let tries = 0;
    const tick = () => { build(); if (!cy && tries++ < 80) setTimeout(tick, 120); };
    tick();
  };
})();
