/* ================================================================
   Delta Drills engine — browser port of the two models the figures run.

   Ported, not reimagined. Constants and update rules mirror the backend:
     BKT + FIRe   ← This-Directory-Only/backend/app/bkt_mastery.py
     Placement    ← This-Directory-Only/backend/app/placement_model.py
   If a constant changes there, change it here; the figures claim to show
   the real engine, so a drift here is a false claim on the page.
   ================================================================ */
(function (root) {
  "use strict";

  // ---- BKT (bkt_mastery.py) -------------------------------------------------
  var BKT = {
    P_INIT: 0.10,       // L0
    P_TRANSIT: 0.30,    // T, fires only on a correct attempt
    P_GUESS: 0.20,
    P_SLIP: 0.10,
    MASTERY: 0.95,
    UNLOCK: 0.85,
    HALF_LIFE_DAYS: 14.0,
  };

  function clamp01(x) { return Math.min(1, Math.max(0, x)); }

  // One graded attempt → posterior, then learn-transit on a correct answer.
  function observe(prior, correct) {
    var L = clamp01(prior), g = BKT.P_GUESS, s = BKT.P_SLIP, t = BKT.P_TRANSIT;
    var num, den;
    if (correct) { num = L * (1 - s); den = num + (1 - L) * g; }
    else { num = L * s; den = num + (1 - L) * (1 - g); }
    var post = den > 1e-12 ? num / den : L;
    return correct ? post + (1 - post) * t : post;
  }

  // FIRe: a learning step of size `gain` with no direct observation.
  function implicitTransit(prior, gain) {
    var L = clamp01(prior);
    return L + (1 - L) * clamp01(gain);
  }

  // Regress toward P_INIT by elapsed-time half-life.
  function decay(L, elapsedDays) {
    var f = Math.pow(0.5, Math.max(0, elapsedDays) / BKT.HALF_LIFE_DAYS);
    return BKT.P_INIT + (L - BKT.P_INIT) * f;
  }

  // Practise `kc`: direct update, then (correct only) FIRe credit to every KC
  // it encompasses, weighted by the encompassing edge. Returns {kc: newL}.
  function applyAttempt(mastery, encIndex, kc, correct) {
    var changed = {};
    var prior = mastery[kc] == null ? BKT.P_INIT : mastery[kc];
    mastery[kc] = changed[kc] = observe(prior, correct);
    if (correct) {
      (encIndex[kc] || []).forEach(function (e) {
        var p = mastery[e.kc] == null ? BKT.P_INIT : mastery[e.kc];
        mastery[e.kc] = changed[e.kc] = implicitTransit(p, BKT.P_TRANSIT * e.w);
      });
    }
    return changed;
  }

  // ---- Placement (placement_model.py) --------------------------------------
  var PL = {
    P_GUESS: 0.05, P_SLIP: 0.20, DK_GUESS: 0.02, DK_SLIP: 0.05,
    HOP_UP: 0.5, HOP_DOWN: 0.3, INCORRECT_DOWN_SCALE: 0.5, MAX_HOPS: 3,
    INDIRECT_CLIP: 1.2, IN_STATE: 0.80, OUT_OF_STATE: 0.20,
  };

  function sigmoid(x) { return x < -60 ? 0 : x > 60 ? 1 : 1 / (1 + Math.exp(-x)); }

  function logBayesFactor(result) {
    if (result === "correct") return Math.log((1 - PL.P_SLIP) / PL.P_GUESS);   // ≈ +2.77
    if (result === "dont_know") return Math.log(PL.DK_SLIP / (1 - PL.DK_GUESS)); // ≈ -2.98
    return Math.log(PL.P_SLIP / (1 - PL.P_GUESS));                               // ≈ -1.56
  }

  function pCorrectGivenBelief(p) { return p * (1 - PL.P_SLIP) + (1 - p) * PL.P_GUESS; }

  // BFS hop distances within MAX_HOPS, same as Graph._walk.
  function walk(start, adj) {
    var dist = {}, frontier = (adj[start] || []).map(function (n) { return [n, 1]; });
    while (frontier.length) {
      var cur = frontier.shift(), n = cur[0], d = cur[1];
      if (n === start || d > PL.MAX_HOPS || (dist[n] != null && dist[n] <= d)) continue;
      dist[n] = d;
      (adj[n] || []).forEach(function (m) { frontier.push([m, d + 1]); });
    }
    return dist;
  }

  function Graph(kcs) {
    var self = this;
    self.kcs = kcs.map(function (k) { return k.id; });
    var keep = {}; self.kcs.forEach(function (k) { keep[k] = true; });
    self.parents = {}; self.children = {};
    self.kcs.forEach(function (k) { self.children[k] = []; });
    kcs.forEach(function (k) {
      self.parents[k.id] = k.prereqs.filter(function (p) { return keep[p]; });
    });
    self.kcs.forEach(function (k) {
      self.parents[k].forEach(function (p) { self.children[p].push(k); });
    });
    self.ancestors = {}; self.descendants = {};
    self.kcs.forEach(function (k) {
      self.ancestors[k] = walk(k, self.parents);
      self.descendants[k] = walk(k, self.children);
    });
  }

  function Beliefs(kcs) {
    this.direct = {}; this.indirect = {};
    var self = this;
    kcs.forEach(function (k) { self.direct[k] = 0; self.indirect[k] = 0; });
  }
  Beliefs.prototype.copy = function () {
    var b = Object.create(Beliefs.prototype);
    b.direct = Object.assign({}, this.direct);
    b.indirect = Object.assign({}, this.indirect);
    return b;
  };
  Beliefs.prototype.p = function (k) {
    var ind = Math.max(-PL.INDIRECT_CLIP, Math.min(PL.INDIRECT_CLIP, this.indirect[k]));
    return sigmoid(this.direct[k] + ind);
  };
  Beliefs.prototype.probs = function () {
    var out = {}, self = this;
    Object.keys(this.direct).forEach(function (k) { out[k] = self.p(k); });
    return out;
  };

  // A pass flows UP to prerequisites; a miss flows DOWN to dependents.
  function applyProbe(B, G, kc, result) {
    var bf = logBayesFactor(result);
    B.direct[kc] += bf;
    if (bf > 0) {
      var anc = G.ancestors[kc];
      Object.keys(anc).forEach(function (o) { B.indirect[o] += bf * Math.pow(PL.HOP_UP, anc[o]); });
    } else {
      var scale = result === "incorrect" ? PL.INCORRECT_DOWN_SCALE : 1;
      var des = G.descendants[kc];
      Object.keys(des).forEach(function (o) { B.indirect[o] += bf * scale * Math.pow(PL.HOP_DOWN, des[o]); });
    }
  }

  function weightedVariance(P, value) {
    var s = 0;
    Object.keys(P).forEach(function (k) { s += (value[k] || 0) * P[k] * (1 - P[k]); });
    return s;
  }

  // Expected value-weighted variance removed by one probe of `kc`.
  function expectedVarianceReduction(B, G, kc, value) {
    var P = B.probs(), now = weightedVariance(P, value), pc = pCorrectGivenBelief(P[kc]), total = 0;
    [["correct", pc], ["incorrect", 1 - pc]].forEach(function (r) {
      var B2 = B.copy();
      applyProbe(B2, G, kc, r[0]);
      total += r[1] * (now - weightedVariance(B2.probs(), value));
    });
    return total;
  }

  // value = coreness = descendants + 1 (full transitive closure, not hop-capped).
  function coreness(G) {
    var memo = {};
    function desc(k) {
      if (memo[k]) return memo[k];
      var s = {};
      G.children[k].forEach(function (c) { s[c] = 1; Object.assign(s, desc(c)); });
      return (memo[k] = s);
    }
    var v = {};
    G.kcs.forEach(function (k) { v[k] = Object.keys(desc(k)).length + 1; });
    return v;
  }

  function rankProbes(B, G, value, candidates) {
    return candidates
      .map(function (k) { return [expectedVarianceReduction(B, G, k, value), k]; })
      .sort(function (a, b) { return b[0] - a[0] || (a[1] < b[1] ? -1 : 1); });
  }

  function classify(p) {
    return p >= PL.IN_STATE ? "known" : p <= PL.OUT_OF_STATE ? "unknown" : "uncertain";
  }

  root.DeltaEngine = {
    BKT: BKT, observe: observe, implicitTransit: implicitTransit, decay: decay,
    applyAttempt: applyAttempt,
    PL: PL, Graph: Graph, Beliefs: Beliefs, applyProbe: applyProbe,
    rankProbes: rankProbes, coreness: coreness, classify: classify,
    pCorrectGivenBelief: pCorrectGivenBelief,
  };
})(window);
