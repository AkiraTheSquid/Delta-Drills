// Atom-level readiness compute, parallel to arena/manifest.js#computeArenaReadiness.
// Computes how "ready" a learner is for a given iter-5 v2 atom, by bridging the
// atom to the existing per-subtopic EWMA state in adaptiveStateJson.
//
// Returns a number in [0, 1] — same convention as computeArenaReadiness.
//
// Bridge strategy (in order):
//   0. Drill-atom direct subtopic — atom appears in window.PREREQ_SUBTOPICS
//      (introduced by Colab procedural drills). Read baseline directly.
//   1. Direct subtopic hit — the atom's `subtopic` field matches a key in
//      adaptiveStateJson.subtopic_states. Read baseline.
//   2. Topic-alias bridge — map the atom's `topic` (ARENA part name like
//      "Ray Tracing") to one of the existing question-bank topics ("Numpy",
//      "Einops"...) and average baselines across that topic's subtopics.
//   3. Fallback — return the passed-in fallback (default 0).
//
// Assumes window.CONCEPT_GRAPH_V5_V2 is loaded (set by iter5_v2.js).
// Reads adaptiveStateJson the same way computeArenaReadiness does.

(function () {
  "use strict";

  // ARENA-part topic → existing question-bank topic. Conservative starter map.
  // NB: Phase 2a assigned `topic` by modal first_appearance in the ARENA chap-0
  // corpus, so atoms like `einops-rearrange` carry topic="Ray Tracing" rather
  // than topic="Einops". The atom-id-token bridge below catches that.
  const ATOM_TOPIC_TO_BANK_TOPIC = {
    "Ray Tracing": "Numpy",
    "CNNs": "CNN",
    "Optimization": null,
    "Backprop": null,
    "VAEs and GANs": null,
    "Prereqs (Einops/Einsum)": "Einops",
  };

  // Atom-id-token → bank topic. Token matching against the kebab-case atom id.
  // Order matters — first hit wins. Used as a 1.5th resolution pass (between
  // direct subtopic and the modal-part topic-alias bridge).
  // Tokens are substring matches; keep them specific to avoid spurious hits.
  const ATOM_ID_TOKEN_TO_BANK_TOPIC = [
    { tokens: ["einsum"], topic: "Einsum" },
    { tokens: ["einops", "rearrange", "reduce", "repeat"], topic: "Einops" },
    { tokens: ["broadcasting", "broadcast", "unbroadcast"], topic: "Numpy" },
    // NB: keep "as-strided" (Numpy/PyTorch reshape op) but NOT bare "stride" —
    // bare "stride" misroutes CNN concepts like `stride-kernel-element-step`
    // (conv kernel stride, not byte-stride) to Numpy.
    { tokens: ["as-strided"], topic: "Numpy" },
    { tokens: ["argmax", "softmax", "logsumexp", "log-sum-exp"], topic: "Numpy" },
    { tokens: ["boolean-mask", "integer-array-indexing", "isfinite-mask"], topic: "Numpy" },
    // NB: dropped "flatten" — the only atom matching it is `flatten-layer`
    // (CNN `nn.Flatten` module), which should route via CNN alias, not Numpy.
    { tokens: ["reshape", "view-vs-reshape", "permute", "transpose"], topic: "Numpy" },
    { tokens: ["torch-arange", "torch-where", "torch-stack", "linspace", "tensor-zeros", "tensor-unbind", "tensor-item"], topic: "Numpy" },
    { tokens: ["outer-product", "vector-normalisation", "rotation-matrix", "surface-normal"], topic: "Numpy" },
  ];

  const _resolveBankTopicFromAtomId = (atomId) => {
    if (typeof atomId !== "string") return null;
    for (const rule of ATOM_ID_TOKEN_TO_BANK_TOPIC) {
      if (rule.tokens.some((t) => atomId.includes(t))) return rule.topic;
    }
    return null;
  };

  const _atomIndex = (() => {
    const cache = { built: false, byId: new Map() };
    return () => {
      if (cache.built) return cache.byId;
      const g = window.CONCEPT_GRAPH_V5_V2;
      if (!g || !Array.isArray(g.concepts)) return cache.byId;
      g.concepts.forEach((c) => {
        if (c && c.id) cache.byId.set(c.id, c);
      });
      cache.built = true;
      return cache.byId;
    };
  })();

  const _topicSubtopicsCache = { built: false, byTopic: new Map() };

  const _buildTopicSubtopicIndex = () => {
    if (_topicSubtopicsCache.built) return _topicSubtopicsCache.byTopic;
    const bank = (typeof questionsBank !== "undefined" && Array.isArray(questionsBank)) ? questionsBank : null;
    if (!bank) return _topicSubtopicsCache.byTopic;
    bank.forEach((q) => {
      if (!q?.topic || !q?.subtopic) return;
      const key = String(q.topic).toLowerCase();
      if (!_topicSubtopicsCache.byTopic.has(key)) {
        _topicSubtopicsCache.byTopic.set(key, new Set());
      }
      _topicSubtopicsCache.byTopic.get(key).add(q.subtopic);
    });
    _topicSubtopicsCache.built = _topicSubtopicsCache.byTopic.size > 0;
    return _topicSubtopicsCache.byTopic;
  };

  const _readAdaptiveState = () => {
    if (typeof adaptiveStateJson !== "string" || !adaptiveStateJson) return null;
    try { return JSON.parse(adaptiveStateJson); } catch (_) { return null; }
  };

  // baseline ∈ [0, 100] in adaptive.py — normalize to [0, 1] at the boundary
  // so readiness matches the MasteryGatePolicy threshold convention.
  const _baselineForSubtopic = (state, subtopic) => {
    const b = Number(state?.subtopic_states?.[subtopic]?.baseline);
    if (!Number.isFinite(b)) return null;
    return Math.max(0, Math.min(1, b / 100));
  };

  const _avg = (xs) => xs.reduce((s, x) => s + x, 0) / xs.length;

  /**
   * computeAtomReadiness(atomId, fallback = 0)
   *
   * Returns a number in [0, 1] indicating how ready the learner is to attempt
   * this atom (i.e. their estimated mastery of its underlying skill).
   *
   * @param {string} atomId    — iter-5 v2 atom id (e.g. "einops-rearrange")
   * @param {number} [fallback] — score to return if no signal is available
   * @return {number}
   */
  // Drill-atom direct subtopic — set by Colab procedural drills via PREREQ_SUBTOPICS.
  // Resolves atoms that aren't in CONCEPT_GRAPH_V5_V2 but ARE in the drill catalog.
  const _drillSubtopic = (atomId) => {
    const reg = window.PREREQ_SUBTOPICS;
    if (!reg || !reg.atom_to_subtopic) return null;
    return reg.atom_to_subtopic[atomId] || null;
  };

  window.computeAtomReadiness = (atomId, fallback) => {
    const fb = Number.isFinite(Number(fallback)) ? Number(fallback) : 0;
    if (typeof atomId !== "string" || !atomId) return fb;
    const state = _readAdaptiveState();
    if (!state || !state.subtopic_states) return fb;

    // 0. Drill-atom direct subtopic — catches drill-introduced atoms that
    // may not appear in CONCEPT_GRAPH_V5_V2 yet but DO have an EWMA state
    // (because the drill beacon posted a rating against PREREQ_SUBTOPICS).
    const drillSub = _drillSubtopic(atomId);
    if (drillSub) {
      const b = _baselineForSubtopic(state, drillSub);
      if (b !== null) return b;
    }

    const atom = _atomIndex().get(atomId);
    if (!atom) return fb;

    // 1. Direct subtopic hit
    if (atom.subtopic) {
      const direct = _baselineForSubtopic(state, atom.subtopic);
      if (direct !== null) return direct;
    }

    const topicIndex = _buildTopicSubtopicIndex();

    // 1.5. Atom-id-token bridge — catches atoms whose semantic category
    // diverges from their modal-part topic (e.g. einops-rearrange tagged
    // topic="Ray Tracing" should still resolve to Einops).
    const tokenTopic = _resolveBankTopicFromAtomId(atomId);
    if (tokenTopic) {
      const subs = topicIndex.get(String(tokenTopic).toLowerCase());
      if (subs && subs.size) {
        const baselines = [];
        subs.forEach((sub) => {
          const b = _baselineForSubtopic(state, sub);
          if (b !== null) baselines.push(b);
        });
        if (baselines.length) return _avg(baselines);
      }
    }

    // 2. Topic-alias bridge — uses concept.topic (modal-part assignment)
    const bankTopic = ATOM_TOPIC_TO_BANK_TOPIC[atom.topic] || atom.topic;
    if (bankTopic) {
      const subs = topicIndex.get(String(bankTopic).toLowerCase());
      if (subs && subs.size) {
        const baselines = [];
        subs.forEach((sub) => {
          const b = _baselineForSubtopic(state, sub);
          if (b !== null) baselines.push(b);
        });
        if (baselines.length) return _avg(baselines);
      }
    }

    return fb;
  };

  /**
   * computeAtomReadinessBatch(atomIds, fallback = 0)
   *
   * Vectorized variant. Returns a map { atomId: readiness }.
   */
  window.computeAtomReadinessBatch = (atomIds, fallback) => {
    const out = {};
    if (!Array.isArray(atomIds)) return out;
    atomIds.forEach((id) => {
      out[id] = window.computeAtomReadiness(id, fallback);
    });
    return out;
  };

  // Helper: does this bank topic have any baseline-populated subtopic in state?
  const _topicHasSignal = (state, topicIndex, bankTopic) => {
    if (!bankTopic) return false;
    const subs = topicIndex.get(String(bankTopic).toLowerCase());
    if (!subs || !subs.size) return false;
    return [...subs].some((s) => _baselineForSubtopic(state, s) !== null);
  };

  /**
   * Diagnostic — count atoms by resolution path.
   * Returns { direct, token, alias, fallback, total } — mirrors the resolution
   * order in computeAtomReadiness (direct subtopic → token-bridge → topic-alias
   * → fallback). Useful to see which atoms still lack signal.
   */
  window._atomReadinessDiagnostic = () => {
    const g = window.CONCEPT_GRAPH_V5_V2;
    if (!g) return { error: "CONCEPT_GRAPH_V5_V2 not loaded" };
    const state = _readAdaptiveState();
    if (!state || !state.subtopic_states) {
      return { error: "adaptiveStateJson missing or empty", concept_count: g.concepts.length };
    }
    const topicIndex = _buildTopicSubtopicIndex();
    let drill = 0, direct = 0, token = 0, alias = 0, fb = 0;
    const fallbackIds = [];
    g.concepts.forEach((atom) => {
      const ds = _drillSubtopic(atom.id);
      if (ds && _baselineForSubtopic(state, ds) !== null) { drill += 1; return; }
      if (atom.subtopic && _baselineForSubtopic(state, atom.subtopic) !== null) {
        direct += 1; return;
      }
      const tokenTopic = _resolveBankTopicFromAtomId(atom.id);
      if (_topicHasSignal(state, topicIndex, tokenTopic)) { token += 1; return; }
      const aliasTopic = ATOM_TOPIC_TO_BANK_TOPIC[atom.topic] || atom.topic;
      if (_topicHasSignal(state, topicIndex, aliasTopic)) { alias += 1; return; }
      fb += 1;
      fallbackIds.push(atom.id);
    });
    return { drill, direct, token, alias, fallback: fb, total: g.concepts.length, fallback_ids: fallbackIds };
  };
})();
