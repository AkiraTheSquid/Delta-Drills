/* ================================================================
   LESSON KNOWLEDGE GRAPH — interactive Cytoscape.js embed for the
   Knowledge Graph tab.

   Built from scratch over the EXISTING lesson content:
     - lessons/kc_registry.json    → 64 knowledge components (KCs) + their
                                      `prereqs` (the edges) + lesson/topic.
     - lessons/lessons_structured.json → per-KC teaching content
                                      (concept / worked example / misconceptions).

   Each bubble is one KC, coloured by lesson, laid out bottom-up so
   prerequisites sit beneath what they unlock. Click a bubble to:
     - light up its full prerequisite chain, and
     - render that KC's learning content in the left pane.
   The left pane's "Practice" (maximize) button hands off to the real
   Delta-Drills practice screen via window.LessonGate.showLesson(kc).

   Superseded concept-graph/graph-viz.js (the old ARENA 205-atom graph),
   which is no longer wired into index.html.

   Built on demand via window.deltaInitConceptGraph() — called by app.js
   switchTab() when the tab opens (Cytoscape can't size a display:none
   container, so we defer until the tab is visible).
   ================================================================ */
(() => {
  "use strict";

  // One pastel per lesson (grouped by topic hue) so black labels stay legible.
  const LESSON_COLORS = {
    "np-1": "#cbe8f7", "np-2": "#aedaf0", "np-3": "#8fcbe9", "np-4": "#72bde1",
    "eo-1": "#cdf2d6", "eo-2": "#a9e6b8", "eo-3": "#88db9c",
    "es-1": "#e6d2ff", "es-2": "#d3b4ff",
  };
  const FALLBACK = "#dddddd";
  const ACCENT = "#ffd23f"; // prerequisite-path highlight
  // "Next up" reuses the same yellow on purpose — it is the map's one attention
  // colour. The two never compete for meaning: the path highlight only exists
  // while a node is selected, and next-up is the standing marker, drawn with an
  // outer glow the path highlight does not have.
  const NEXT_UP = ACCENT;

  const $ = (id) => document.getElementById(id);
  const lessonColor = (lid) => LESSON_COLORS[lid] || FALLBACK;

  /* ---- mastery colouring (BKT posterior → red↔blue, gray = no estimate) ---- */
  const BKT_P_INIT = 0.10, BKT_HALF_LIFE_DAYS = 14.0;
  const UNKNOWN_COLOR = "#5b5b70";       // no estimate yet
  let colorMode = "mastery";             // "mastery" | "lesson"

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
  // none. Three sources, in order of precision:
  //   "atom"         — this KC's own decayed BKT posterior.
  //   "subtopic"     — the lesson subtopic's BKT mastery, shared by every KC
  //                    in that lesson.
  //   "extrapolated" — no evidence on this concept at all: projected from the
  //                    learner's overall demonstrated level, adjusted for how
  //                    hard this concept is (see `_extrapolated`).
  // The subtopic fallback exists because the graph's KC ids (`numpy.dtype-astype`,
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
      if (r >= 0) return { r, source: "atom" };
    }
    const s = _persistedState();
    const raw = s && s.atom_mastery ? s.atom_mastery[kc] : undefined;
    if (Number.isFinite(raw)) {
      return { r: _decay(raw, s.atom_last_ts ? s.atom_last_ts[kc] : null), source: "atom" };
    }
    // Crosswalk read: this KC's own evidence, held under atom ids the graph
    // does not share. Only the `measured` tier qualifies — for a topic proxy
    // the atoms are coarser than the concept, so this returns null and the
    // subtopic fallback below handles it, which at least says what it is.
    if (s && s.atom_mastery && typeof window.kcCrosswalkReadiness === "function") {
      const x = window.kcCrosswalkReadiness(kc, s.atom_mastery, s.atom_last_ts, _decay);
      if (x) return { r: x.r, source: "atom", via: x.atoms, ts: x.ts };
    }
    // Subtopic fallback. `p` (correctness rate) is the only field on the same
    // [0,1] scale as a posterior — `baseline` is a difficulty-weighted score in
    // [0,100]. Requires n > 0: p defaults to 0.5, and a never-practised subtopic
    // must stay grey rather than claim a coin-flip's worth of knowledge. No
    // decay is applied — the server already decayed it on write, and re-decaying
    // an EWMA with BKT's half-life would mix two different models.
    const sub = _subtopicState(kc);
    if (sub && Number.isFinite(sub.n) && sub.n > 0 && Number.isFinite(sub.p)) {
      return { r: Math.max(0, Math.min(1, sub.p)), source: "subtopic" };
    }
    const ex = _extrapolated(kc);
    if (ex) return { r: ex.r, source: "extrapolated" };
    return { r: NaN, source: "none" };
  };
  const kcReadiness = (kc) => kcReadinessInfo(kc).r;
  const kcLastTs = (kc) => {
    const s = _persistedState();
    if (!s || !s.atom_last_ts) return null;
    if (s.atom_last_ts[kc]) return s.atom_last_ts[kc];
    // Same disjoint-id problem as the mastery read: for a measured KC the
    // timestamps live under the atom ids, so "last seen" was blank on exactly
    // the nodes with real evidence.
    if (typeof window.kcCrosswalkReadiness === "function") {
      const x = window.kcCrosswalkReadiness(kc, s.atom_mastery, s.atom_last_ts, _decay);
      if (x && x.ts) return x.ts;
    }
    return null;
  };
  // red (low) → muted purple → blue (high); gray for no estimate.
  const masteryColor = (r) => {
    if (!Number.isFinite(r)) return UNKNOWN_COLOR;
    const t = Math.max(0, Math.min(1, r));
    const lo = [214, 72, 72], hi = [59, 130, 246];  // #d64848 → #3b82f6
    const c = lo.map((v, i) => Math.round(v + (hi[i] - v) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  };
  const nodeColor = (kc) =>
    colorMode === "mastery" ? masteryColor(kcReadiness(kc)) : lessonColor((kcById[kc] || {}).lesson);
  const masteryBand = (r) => {
    if (!Number.isFinite(r)) return "Not yet estimated";
    if (r < 0.30) return "Just starting";
    if (r < 0.60) return "Learning";
    if (r < 0.85) return "Proficient";
    return "Strong";
  };
  const relTime = (ts) => {
    if (!ts) return null;
    const t = Date.parse(ts); if (!Number.isFinite(t)) return null;
    const s = Math.max(0, (Date.now() - t) / 1000);
    if (s < 90) return "just now";
    const m = s / 60; if (m < 90) return `${Math.round(m)} min ago`;
    const h = m / 60; if (h < 36) return `${Math.round(h)} h ago`;
    return `${Math.round(h / 24)} d ago`;
  };

  /* ---- learner-model readout (rendered into the left pane on select) ----
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

  /* ---------------- tiny markdown renderer ----------------------------- */
  const esc = (v) =>
    String(v ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  const inline = (v) =>
    esc(v)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>");

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
        if (renderCode) out.push("<pre><code>" + esc(buf.join("\n")) + "</code></pre>");
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

  const chipLink = (id) => {
    const kc = kcById[id];
    if (!kc) return "";
    return `<button class="kg2-chip" data-goto="${esc(id)}" title="${esc(kc.title)}">
      <span class="kg2-chip-dot" style="background:${lessonColor(kc.lesson)}"></span>${esc(kc.title)}</button>`;
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
  const EXTRAP_MIN_SD = 0.15;            // floor on the projected interval's half-width
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
    // collapses it to the EXTRAP_MIN_SD floor. That made untouched concepts
    // draw a TIGHTER band than concepts with two real attempts behind them,
    // which is backwards: zero evidence is the most uncertain state on the map,
    // not the least. Floor the half-width at what a single graded attempt would
    // earn, so the band can only narrow as evidence arrives.
    const spreadHalf = (shift(a.mBar + a.sd) - shift(a.mBar - a.sd)) / 2;
    const half = Math.max(spreadHalf, _noEvidenceHalf(r));
    return {
      r,
      ci: [Math.max(0, r - half), Math.min(1, r + half)],
      d, ...a,
    };
  };

  const _pct = (v) => (Number.isFinite(v) ? Math.round(v * 100) + "%" : "—");

  // 95% Wilson score interval — how much the estimate should be trusted, given
  // how little evidence backs it. Wilson (not the normal approximation) because
  // n is small and p sits near the edges, where the normal interval runs off
  // [0,1] and understates uncertainty. `n` is the count of graded attempts the
  // estimate rests on; with none, there is no interval to draw.
  const _wilson = (p, n, z = 1.96) => {
    if (!Number.isFinite(p) || !Number.isFinite(n) || n <= 0) return null;
    const d = 1 + (z * z) / n;
    const centre = (p + (z * z) / (2 * n)) / d;
    const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
    return [Math.max(0, centre - half), Math.min(1, centre + half)];
  };

  // The widest honest band: what one graded attempt would earn. A concept with
  // NO attempts must be at least this uncertain, or the bar tells the learner
  // the opposite of the truth. Wilson at n=1 is ~±0.45, i.e. "almost anything",
  // which is the correct thing to say about a concept nobody has tested.
  const _noEvidenceHalf = (p) => {
    const w = _wilson(Number.isFinite(p) ? p : 0.5, 1);
    return w ? (w[1] - w[0]) / 2 : 0.45;
  };

  // Evidence backing a KC's estimate. Per-attempt counts are only kept at
  // subtopic level, so that is the honest n — noted as such in the popup.
  const _evidenceN = (kc) => {
    const sub = _subtopicState(kc);
    return Number.isFinite(sub && sub.n) ? sub.n : 0;
  };

  // The mastery bar: fill = P(known), shaded band = the confidence interval,
  // ticks = the two thresholds the engine actually acts on.
  const _masteryBar = (r, ci) => {
    const w = Number.isFinite(r) ? Math.max(0, Math.min(1, r)) * 100 : 0;
    const band = ci
      ? `<span class="kg2-dock-ci" style="left:${ci[0] * 100}%;width:${Math.max(0.5, (ci[1] - ci[0]) * 100)}%"></span>`
      : "";
    return (
      `<div class="kg2-dock-track">` +
        `<div class="kg2-dock-fill" style="width:${w}%;background:${masteryColor(r)}"></div>` +
        band +
        `<span class="kg2-dock-gate" style="left:${UNLOCK_T * 100}%"></span>` +
        `<span class="kg2-dock-gate is-mastery" style="left:${MASTERY_T * 100}%"></span>` +
      `</div>` +
      `<div class="kg2-dock-scale"><span>0%</span><span>85% unlocks</span><span>95% mastered</span></div>`
    );
  };

  /* ---- the readout: a panel docked across the bottom of the graph pane ----
     Hovering a bubble previews that concept there; clicking one keeps it. The
     panel is always present and fixed-height, so the graph never reflows under
     the cursor — the canvas is inset by --kg2-dock-h rather than overlaid. */
  let dockEl = null;
  let dockedKc = null;   // node whose readout stays up (the gold-highlighted one)

  const _ensureDock = () => {
    if (dockEl) return dockEl;
    dockEl = $("kg-dock");
    if (!dockEl) {
      dockEl = document.createElement("div");
      dockEl.id = "kg-dock";
      dockEl.className = "kg2-dock";
      (document.querySelector(".kg2-graph") || document.body).appendChild(dockEl);
    }
    return dockEl;
  };

  const dockEmptyHtml = () =>
    `<div class="kg2-dock-empty">Hover a bubble to preview its learner model — click one to keep it here.</div>`;

  const dockHtml = (kc) => {
    const k = kcById[kc];
    if (!k) return dockEmptyHtml();
    const { r, source } = kcReadinessInfo(kc);
    const n = _evidenceN(kc);
    // A projected estimate has no attempts of its own, so its range is the
    // spread of the learner's own results — a Wilson interval would claim
    // evidence that doesn't exist.
    const ex = source === "extrapolated" ? _extrapolated(kc) : null;
    const ci = ex ? ex.ci : (Number.isFinite(r) ? _wilson(r, n) : null);
    const sub = _subtopicState(kc);
    const last = relTime(kcLastTs(kc)) || (sub ? relTime(sub.last_update_ts) : null);
    const parents = parentsOf[kc] || [];
    const ready = parents.filter((p) => { const pr = kcReadiness(p); return Number.isFinite(pr) && pr >= UNLOCK_T; }).length;
    const sibs = sub ? _siblingCount(kc) : 0;

    let rows = "";
    if (kc === nextUpKc) {
      rows += `<div class="kg2-dock-row is-next-up"><span>Next up</span>` +
              `<span>the queue's weakest unlocked concept</span></div>`;
    }
    rows += `<div class="kg2-dock-row"><span>Last practiced</span><span>${last ? esc(last) : "never"}</span></div>`;
    rows += parents.length
      ? `<div class="kg2-dock-row"><span>Prerequisites ready</span><span>${ready}/${parents.length}</span></div>`
      : `<div class="kg2-dock-row"><span>Prerequisites</span><span>none — foundation</span></div>`;
    if (sub) {
      rows += `<div class="kg2-dock-row"><span>Recent accuracy</span><span>${_pct(sub.p)}</span></div>`;
      rows += `<div class="kg2-dock-row"><span>Target difficulty</span><span>${Number.isFinite(sub.target_difficulty) ? Math.round(sub.target_difficulty) : "—"}/100</span></div>`;
    } else {
      const d = _kcDifficulty(kc);
      if (Number.isFinite(d)) {
        rows += `<div class="kg2-dock-row"><span>Concept difficulty</span><span>${Math.round(d)}/100</span></div>`;
      }
      if (ex && Number.isFinite(ex.bBar)) {
        rows += `<div class="kg2-dock-row"><span>Your practised level</span><span>${_pct(ex.mBar)} at ${Math.round(ex.bBar)}/100</span></div>`;
      }
    }

    return (
      `<div class="kg2-dock-col kg2-dock-id">` +
        `<div class="kg2-dock-title">${esc(k.title)}</div>` +
        `<div class="kg2-dock-meta">${esc(k.topic)} · ${esc((lessonMeta[k.lesson] || {}).title || k.lesson)}</div>` +
        // Named, because the numbers on the right are subtopic-wide: the count
        // and the accuracy are shared by every concept in the lesson.
        (sub
          ? `<div class="kg2-dock-evidence">Evidence: ${esc(sub.key)}` +
            (sibs > 1 ? ` · shared by ${sibs} concepts` : "") +
            (sub.mergedTopics ? ` · merged across ${esc(sub.mergedTopics.join(" + "))}` : "") +
            `</div>`
          : ex
            ? `<div class="kg2-dock-evidence">Evidence: your overall level — ${ex.attempts} attempt${ex.attempts === 1 ? "" : "s"} across ${ex.subtopics} lesson${ex.subtopics === 1 ? "" : "s"}, none on this concept</div>`
            : "") +
      `</div>` +
      `<div class="kg2-dock-col kg2-dock-mastery">` +
        `<div class="kg2-dock-headline">` +
          `<strong style="color:${masteryColor(r)}">${_pct(r)}</strong>` +
          `<span class="kg2-dock-band">${esc(masteryBand(r))}` +
            // The percentage is only about THIS concept when it came from a
            // per-atom posterior; say so when it didn't.
            (source === "subtopic" ? ` <span class="kg2-dock-dim">· lesson-level</span>` : "") +
            (source === "extrapolated" ? ` <span class="kg2-dock-dim">· projected</span>` : "") +
          `</span>` +
        `</div>` +
        _masteryBar(r, ci) +
        `<div class="kg2-dock-ci-label">` +
          (ex
            ? `Projected ${_pct(ci[0])}–${_pct(ci[1])} <span class="kg2-dock-dim">· nothing graded here yet. ` +
              (Number.isFinite(ex.d) && Number.isFinite(ex.bBar)
                ? `This concept rates ${Math.round(ex.d)}/100 against the ${Math.round(ex.bBar)}/100 you've been practising at`
                : `Carried straight across from your overall level`) + `.</span>`
          : ci
            ? `95% CI ${_pct(ci[0])}–${_pct(ci[1])} <span class="kg2-dock-dim">· from ${n} graded attempt${n === 1 ? "" : "s"}` +
              (source === "subtopic" ? ` across this lesson, not this concept alone` : "") + `</span>`
            : Number.isFinite(r)
              ? `<span class="kg2-dock-dim">No graded attempts behind this estimate yet — treat it as a prior.</span>`
              : `<span class="kg2-dock-dim">No estimate yet — nothing graded has landed on this concept.</span>`) +
        `</div>` +
      `</div>` +
      `<div class="kg2-dock-col kg2-dock-stats">${rows}</div>`
    );
  };

  const showDock = (kc) => {
    const el = _ensureDock();
    if (!kcById[kc]) return;
    el.innerHTML = dockHtml(kc);
    el.classList.toggle("is-pinned", kc === dockedKc);
  };

  // Docked readout: sticks to the selected node, survives hovering elsewhere,
  // and repaints when the learner model changes.
  const pinDock = (kc) => { dockedKc = kc; showDock(kc); };
  const unpinDock = () => {
    dockedKc = null;
    const el = _ensureDock();
    el.innerHTML = dockEmptyHtml();
    el.classList.remove("is-pinned");
  };
  const restoreDock = () => { if (dockedKc) showDock(dockedKc); else unpinDock(); };
  const refreshDock = () => { if (dockedKc) showDock(dockedKc); };

  // The pane's CSS height assumes a fixed amount of chrome above it, but the
  // guest banner isn't always there — so the pane could end below the fold.
  // That was survivable when the readout floated; with it docked at the pane's
  // bottom edge, an overhang hides the numbers. Size the pane to what's left.
  const fitWrap = () => {
    const wrap = document.querySelector(".kg2 .kg2-wrap");
    if (!wrap) return;
    const top = wrap.getBoundingClientRect().top;
    if (window.innerWidth <= 820) { wrap.style.height = ""; return; }  // stacked layout scrolls
    wrap.style.height = Math.max(480, window.innerHeight - top - 14) + "px";
    if (cy) cy.resize();
  };

  /* ---------------- left content pane ---------------------------------- */
  const setPlaceholder = () => {
    selectedKc = null;
    const btn = $("kg-maximize");
    if (btn) btn.hidden = true;
    if ($("kg-info-meta")) $("kg-info-meta").innerHTML = "";
    if ($("kg-info-body"))
      $("kg-info-body").innerHTML =
        `<div class="kg2-placeholder"><strong>Click a bubble</strong> to open its lesson here.<br><br>
         You'll see what the skill teaches and its worked example, and the whole
         prerequisite chain lights up on the graph. Use <strong>Practice ⤢</strong>
         to jump into the full practice screen for that skill.</div>`;
  };

  const renderContent = (id) => {
    const kc = kcById[id];
    const kp = contentByKc[id];
    if (!kc || !kp) return;
    selectedKc = id;
    const lm = lessonMeta[kc.lesson] || {};

    const meta = $("kg-info-meta");
    if (meta)
      meta.innerHTML =
        `<span class="kg2-meta-topic"><span class="kg2-chip-dot" style="background:${lessonColor(kc.lesson)}"></span>${esc(kc.topic)}</span>
         <span class="kg2-meta-lesson">${esc(lm.title || kc.lesson)}</span>`;

    const btn = $("kg-maximize");
    if (btn) { btn.hidden = false; btn.dataset.kc = id; }

    const parents = parentsOf[id] || [];
    const kids = childrenOf[id] || [];
    let html = `<h2 class="kg2-title">${esc(kp.title || kc.title)}</h2>`;
    html += `<div class="kg2-concept">${md(kp.concept_markdown)}</div>`;
    if (kp.worked_example_markdown)
      html += `<div class="kg2-worked"><h3>Worked example</h3>${md(kp.worked_example_markdown)}</div>`;
    if (kp.misconceptions_markdown)
      html += `<div class="kg2-watch"><h3>Watch out</h3>${md(kp.misconceptions_markdown)}</div>`;

    html += `<div class="kg2-nav">`;
    html += `<div class="kg2-nav-col"><h4>Prerequisites (${parents.length})</h4>` +
      (parents.length ? parents.map(chipLink).join("") : `<span class="kg2-nav-empty">Foundation skill — none.</span>`) + `</div>`;
    html += `<div class="kg2-nav-col"><h4>Unlocks (${kids.length})</h4>` +
      (kids.length ? kids.map(chipLink).join("") : `<span class="kg2-nav-empty">Nothing downstream yet.</span>`) + `</div>`;
    html += `</div>`;

    const body = $("kg-info-body");
    body.innerHTML = html;
    body.scrollTop = 0;
    body.querySelectorAll("[data-goto]").forEach((b) =>
      b.addEventListener("click", () => selectNode(b.getAttribute("data-goto"))));
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
    renderContent(id);
    pinDock(id);   // the bottom panel keeps the gold-highlighted node
  };

  const resetView = () => {
    if (cy) cy.elements().removeClass("faded hl hl-strong");
    unpinDock();
    setPlaceholder();
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
    _refreshLearnerState().then(() => { recolor(); refreshDock(); });
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
  // wrote → drop back to the map → animate the node red→blue → offer the next
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

  /* ---------------- legend (mode-aware) + recolour --------------------- */
  const buildLegend = () => {
    const el = $("kg-legend");
    if (!el) return;
    if (colorMode === "mastery") {
      el.classList.add("kg2-legend-mastery");
      el.innerHTML =
        '<span class="kg2-li"><span class="kg2-li-dot" style="background:' + UNKNOWN_COLOR + '"></span>No estimate</span>' +
        '<span class="kg2-li"><span class="kg2-li-dot kg2-li-projected"></span>Projected from your level</span>' +
        '<span class="kg2-li kg2-li-scale"><span>less</span><span class="kg2-scale-bar"></span><span>more mastered</span></span>';
    } else {
      el.classList.remove("kg2-legend-mastery");
      const seen = [];
      Object.values(kcById).forEach((k) => { if (!seen.includes(k.lesson)) seen.push(k.lesson); });
      seen.sort();
      el.innerHTML = seen.map((lid) => {
        const lm = lessonMeta[lid] || {};
        return `<span class="kg2-li"><span class="kg2-li-dot" style="background:${lessonColor(lid)}"></span>${esc(lm.title || lid)}</span>`;
      }).join("");
    }
  };

  // Projected bubbles are washed out and dashed: they carry a colour because a
  // blank map was the wrong default, but they must never read as measured.
  // Border WIDTH is left to the stylesheet so the .hl prerequisite highlight
  // still wins; only the dash pattern is set per node.
  const recolor = () => {
    if (!cy) return;
    cy.batch(() => cy.nodes().forEach((n) => {
      const projected = colorMode === "mastery" && kcReadinessInfo(n.id()).source === "extrapolated";
      n.style({
        "background-color": nodeColor(n.id()),
        "background-opacity": projected ? 0.42 : 1,
        "border-style": projected ? "dashed" : "solid",
      });
    }));
    markNextUp();
  };

  /* ---------------- "next up": where the queue is pointing ----------------
     The practice queue picks weakest-first among what the learner can actually
     attempt, so this mirrors that rule on the graph: the lowest-readiness
     concept whose prerequisites all clear the unlock gate, ties broken by the
     easiest entry point. Concepts already at mastery drop out.

     It MIRRORS the queue, it is not the server's literal pick — the queue
     selects a subtopic and then a question inside it, and it may serve a
     placement probe instead. So the dock says "next up", not "you will be
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

  // Yellow ring on the concept, and the same yellow along the edges feeding it,
  // so the route INTO it is visible rather than just the destination.
  const markNextUp = () => {
    if (!cy) return;
    cy.elements(".next-up, .next-up-edge").removeClass("next-up next-up-edge");
    nextUpKc = colorMode === "mastery" ? _nextUpKc() : null;
    if (!nextUpKc) return;
    const node = cy.getElementById(nextUpKc);
    if (!node || !node.length) return;
    node.addClass("next-up");
    node.incomers("edge").addClass("next-up-edge");
  };

  /* ---------------- build ---------------------------------------------- */
  async function build() {
    if (cy || building) return;
    const container = $("kg-cy");
    if (!container || typeof cytoscape === "undefined") return;
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

    buildLegend();

    const elements = [];
    Object.values(kcById).forEach((k) => {
      elements.push({ data: { id: k.id, label: k.title, lesson: k.lesson } });
    });
    let ei = 0;
    Object.values(kcById).forEach((k) => {
      (k.prereqs || []).forEach((p) => {
        if (kcById[p]) elements.push({ data: { id: "e" + (ei++), source: p, target: k.id } });
      });
    });

    cy = cytoscape({
      container,
      elements,
      wheelSensitivity: 0.25,
      minZoom: 0.1, maxZoom: 3,
      style: [
        { selector: "node", style: {
            "background-color": (n) => nodeColor(n.id()),
            "shape": "round-rectangle",
            "label": "data(label)",
            "width": "label", "height": "label", "padding": "13px",
            "text-wrap": "wrap", "text-max-width": "120px",
            "text-valign": "center", "text-halign": "center",
            "font-size": 13, "font-weight": 600, "color": "#15151f",
            "border-width": 1, "border-color": "rgba(0,0,0,.28)",
            "transition-property": "opacity, border-width, border-color", "transition-duration": "120ms",
        }},
        { selector: "edge", style: {
            "curve-style": "bezier", "width": 1.8, "line-color": "#e3212c",
            "target-arrow-shape": "triangle", "target-arrow-color": "#e3212c", "arrow-scale": 0.8, "opacity": 0.9,
        }},
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
        { selector: "edge.next-up-edge", style: {
            "width": 3.5, "line-color": NEXT_UP, "target-arrow-color": NEXT_UP,
            "opacity": 1, "z-index": 70,
        }},
      ],
      layout: { name: window.cytoscapeDagre ? "dagre" : "cose",
        rankDir: "BT", nodeSep: 26, rankSep: 150, edgeSep: 12, animate: false, fit: true, padding: 40 },
    });

    if ($("kg-status")) $("kg-status").style.display = "none";
    cy.on("tap", "node", (evt) => selectNode(evt.target.id()));
    cy.on("tap", (evt) => { if (evt.target === cy) resetView(); });

    if ($("kg-fit")) $("kg-fit").onclick = () => cy.fit(undefined, 36);
    const maxBtn = $("kg-maximize");
    if (maxBtn) maxBtn.onclick = () => openMaximize(maxBtn.dataset.kc);

    /* ---- colour-mode toggle (Mastery ↔ Lessons) ---- */
    const controls = document.querySelector(".kg2-controls");
    if (controls && !$("kg-colormode")) {
      const seg = document.createElement("div");
      seg.className = "kg2-seg";
      seg.id = "kg-colormode";
      seg.innerHTML =
        '<button type="button" data-mode="mastery" class="active">Mastery</button>' +
        '<button type="button" data-mode="lesson">Lessons</button>';
      controls.insertBefore(seg, controls.firstChild);
      seg.querySelectorAll("button").forEach((b) =>
        b.addEventListener("click", () => {
          colorMode = b.dataset.mode;
          seg.querySelectorAll("button").forEach((x) => x.classList.toggle("active", x === b));
          buildLegend();
          recolor();
        }));
    }

    /* ---- bottom panel: hover to preview any node, kept on the selected one ---- */
    unpinDock();   // creates the panel and seeds its empty state
    fitWrap();
    window.addEventListener("resize", fitWrap);
    cy.on("mouseover", "node", (evt) => showDock(evt.target.id()));
    // Hovering elsewhere borrows the panel; on mouse-out it goes back to the
    // selected node rather than leaving the selection without its readout.
    // The panel is docked, so pan/zoom/node-drag need no re-anchoring.
    cy.on("mouseout", "node", restoreDock);

    // Recolour when the learner model changes (a graded attempt updates BKT),
    // and repaint the panel with it.
    window.addEventListener("delta:adaptive-state-changed", () => { recolor(); refreshDock(); });

    buildLegend();
    recolor();
    setPlaceholder();
    building = false;
  }

  window.deltaInitConceptGraph = function () {
    if (cy) { fitWrap(); cy.resize(); cy.fit(undefined, 36); return; }
    let tries = 0;
    const tick = () => { build(); if (!cy && tries++ < 80) setTimeout(tick, 120); };
    tick();
  };
})();
