/* ================================================================
   EXERCISE-SESSION PLANNER — greedy within the learner's own budget

   Seth, 2026-09-06: "if you say that you want a maximum of five or six
   problems, then it maximizes the probability that you get that problem
   right within five or six tries … both prepping for the problem and
   answering the problem. … if you have a thousand problems, then it's not
   greedy whatsoever. … two or three problems, then it's insanely greedy."

   The learner sizes a block (N questions, the MAXIMUM). This file decides,
   one question at a time, whether the next slot is spent on

     ATTEMPT  — one of the exercise's VARIANTS (same concept, same difficulty,
                one thing moved so a memorised solution does not transfer;
                lessons/arena_exercise_kcs.json lists them), or
     PREP     — a drill that raises the chance the next attempt lands: the
                exercise's own scaffolded rungs (its faded / solo drills,
                including the original problem blanked) or a prerequisite
                concept's rung.

   It picks by a tiny dynamic programme over the REMAINING budget R:

     V(0, p) = 0
     V(R, p) = max( p + (1 - p) · V(R-1, p + gain_attempt),        attempt now
                    V(R-1, p + gain_prep) )                          prep first

   where p is the modelled chance of solving a variant right now. Greed falls
   out of R by itself: with R = 1 the only move is to attempt; with R = 2 a
   prep is worth it only if it lifts p more than an extra attempt would; with
   R = 40 both branches reach ~1 and the tie-break climbs the ordinary ladder
   (scaffold first while p is below the faded band). The block ends the moment
   a variant is solved — N was a maximum, never a target.

   🔴 ORDER ONLY. Nothing here grades, scores or writes mastery. `p` is a
   PLANNING estimate seeded from the server's per-concept interval
   (`/api/practice/kc-estimate`, Laplace-smoothed) and nudged by results
   inside this block so the next choice reflects what just happened; it is
   never shown as a competency and never sent anywhere. The bar the learner
   sees still comes from the submit response, as everywhere else.
   ================================================================ */

const ExercisePlanner = (() => {
  /* Learning credited to a concept for one drill served at that rung, applied
     as m += gain · (1 − m). A miss shrinks instead (MISS_KEEP). Chosen so one
     faded drill moves a cold concept a little and a solo drill more — the
     ORDER of the choices is what matters, not the numbers. A missed ATTEMPT
     teaches less than a drill aimed at the gap (its review is of the whole
     problem, not the move that failed) — were it credited as much, prep would
     never win and every block would be all attempts. */
  const GAIN = { faded: 0.22, guided: 0.26, independent: 0.32, attempt: 0.10 };
  const MISS_KEEP = 0.7;
  /* Below this p the tie-break prefers prep (climb the ladder); above it,
     attempt. Same band the ladder uses to decide "start scaffolded". */
  const READY = 0.75;
  const TIE = 1e-3;
  /* Each further prep is assumed to lift p less than the last (the pools run
     from the drill aimed squarely at the gap to the ones beside it). Without
     this the plan drifts: every step's prep wins by a hair and the block ends
     with one attempt at the very end. */
  const PREP_DECAY = 0.75;
  /* Prior for a concept with no attempts: Laplace (correct + 0.8) / (n + 2). */
  const PRIOR_HITS = 0.8;
  const PRIOR_N = 2;
  /* How much weak prerequisites drag the target down: p = own · (0.5 + 0.5·m̄). */
  const PREREQ_FLOOR = 0.5;
  const HIST_KEY = "dd_variant_hist:";

  const _clamp = (x) => Math.max(0.01, Math.min(0.99, x));
  const _mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 1);

  /** Laplace-smoothed mastery from a `kc_estimate` payload (or null → prior). */
  const masteryFromEstimate = (est) => {
    const n = Number(est && est.n) || 0;
    const k = Number(est && est.correct) || 0;
    return _clamp((k + PRIOR_HITS) / (n + PRIOR_N));
  };

  const _readHist = (kc) => {
    try {
      const v = JSON.parse(localStorage.getItem(HIST_KEY + kc) || "[]");
      return Array.isArray(v) ? v.filter(Number.isFinite) : [];
    } catch (_err) { return []; }
  };
  const _writeHist = (kc, ids) => {
    try { localStorage.setItem(HIST_KEY + kc, JSON.stringify(ids.slice(-50))); } catch (_err) { /* private mode */ }
  };

  class Planner {
    /**
     * @param {object} cfg
     *   quota    – the block's maximum question count
     *   target   – { kc, title, m, pools: {faded:[], guided:[], independent:[]}, variants: [items] }
     *   prereqs  – [{ kc, title, m, pools }]
     * Items are kc-practice.js ladder items ({kind, questionId, kc, kcTitle, …}).
     */
    constructor(cfg) {
      this.attemptFirst = cfg.attemptFirst === true;
      this.quota = Math.max(1, Number(cfg.quota) || 1);
      this.used = Number(cfg.used) || 0;
      this.solved = !!cfg.solved;
      this.target = cfg.target;
      this.prereqs = Array.isArray(cfg.prereqs) ? cfg.prereqs : [];
      this.attempted = Array.isArray(cfg.attempted) ? cfg.attempted.slice() : [];
      this.attempts = Number(cfg.attempts) || 0;
      this.preps = Number(cfg.preps) || 0;
      this.last = cfg.last || null; // { item, delta } of the previous choice
    }

    /* ── model ── */
    pTarget(own = this.target.m, pres = this.prereqs.map((p) => p.m)) {
      return _clamp(own * (PREREQ_FLOOR + (1 - PREREQ_FLOOR) * _mean(pres)));
    }

    _nextPrep(node) {
      const order = node.m < READY
        ? ["faded", "guided", "independent"]
        : ["independent", "guided", "faded"];
      for (const kind of order) {
        const pool = node.pools[kind] || [];
        if (pool.length) return { kind, item: pool[0] };
      }
      return null;
    }

    /* Every prep candidate with the lift it gives p. The target's own rungs
       raise `own`; a prerequisite's rung raises that prerequisite's m. */
    _candidates() {
      const now = this.pTarget();
      const out = [];
      const tp = this._nextPrep(this.target);
      if (tp) {
        const own = this.target.m + GAIN[tp.kind] * (1 - this.target.m);
        out.push({ node: this.target, ...tp, delta: this.pTarget(own) - now });
      }
      this.prereqs.forEach((pre, i) => {
        const pp = this._nextPrep(pre);
        if (!pp) return;
        const pres = this.prereqs.map((p) => p.m);
        pres[i] = pre.m + GAIN[pp.kind] * (1 - pre.m);
        out.push({ node: pre, ...pp, delta: this.pTarget(undefined, pres) - now });
      });
      // Biggest lift first; on a near-tie the weaker concept first.
      out.sort((a, b) => (b.delta - a.delta) || (a.node.m - b.node.m));
      return out;
    }

    _attemptGain() {
      const own = this.target.m + GAIN.attempt * (1 - this.target.m);
      return this.pTarget(own) - this.pTarget();
    }

    /* V(R, p) — see the header; the prep lift decays with each prep taken. */
    _value(R, p, gPrep, gAtt, memo) {
      if (R <= 0) return 0;
      const key = R + "|" + p.toFixed(2) + "|" + gPrep.toFixed(3);
      if (memo.has(key)) return memo.get(key);
      const attempt = p + (1 - p) * this._value(R - 1, _clamp(p + gAtt), gPrep, gAtt, memo);
      const prep = gPrep > 0.005 ? this._value(R - 1, _clamp(p + gPrep), gPrep * PREP_DECAY, gAtt, memo) : 0;
      const v = Math.max(attempt, prep);
      memo.set(key, v);
      return v;
    }

    /** The next item to serve, or null when the block is over. Two steps so
        the caller can HYDRATE before anything is spent: `peek()` decides,
        `commit(decision)` charges the slot; `drop(decision)` throws away an
        item the bank no longer has without charging (codex, 2026-09-06). */
    next() {
      const d = this.peek();
      return d ? this.commit(d) : null;
    }

    peek() {
      if (this.solved) return null;
      const R = this.quota - this.used;
      if (R <= 0) return null;
      const p = this.pTarget();
      if (this.attemptFirst && this.used === 0 && this.target.variants.length) {
        return { choice: "attempt", item: this._pickVariant(), p, R };
      }
      const cands = this._candidates();
      const best = cands[0] || null;
      // The same decay the look-ahead assumes, applied to the preps already
      // taken — otherwise every step sees an undecayed first prep.
      const gPrep = best ? best.delta * Math.pow(PREP_DECAY, this.preps) : 0;
      const gAtt = this._attemptGain();
      const memo = new Map();
      const vAttempt = p + (1 - p) * this._value(R - 1, _clamp(p + gAtt), gPrep, gAtt, memo);
      const vPrep = best ? this._value(R - 1, _clamp(p + gPrep), gPrep * PREP_DECAY, gAtt, memo) : -1;
      let choice;
      if (!best || !this.target.variants.length) choice = best ? "prep" : "attempt";
      else if (Math.abs(vAttempt - vPrep) < TIE) choice = p < READY ? "prep" : "attempt";
      else choice = vAttempt > vPrep ? "attempt" : "prep";

      if (choice === "attempt" && this.target.variants.length) {
        return { choice, item: this._pickVariant(), p, R };
      }
      return best ? { choice: "prep", item: best.item, best, p, R } : null;
    }

    commit(d) {
      if (!d || !d.item) return null;
      if (d.choice === "attempt") {
        this.attempted.push(d.item.questionId);
        this.attempts += 1;
        this._recordHist(d.item.questionId);
      } else {
        const pool = d.best.node.pools[d.best.kind];
        const i = pool.indexOf(d.item);
        pool.splice(i < 0 ? 0 : i, 1);
        this.preps += 1;
      }
      this.used += 1;
      this.last = { choice: d.choice, questionId: d.item.questionId, kc: d.item.kc, p: d.p, R: d.R };
      return d.item;
    }

    drop(d) {
      if (!d || !d.item) return;
      const id = d.item.questionId;
      if (d.choice === "attempt") {
        this.target.variants = this.target.variants.filter((v) => v.questionId !== id);
      } else {
        const pool = d.best.node.pools[d.best.kind];
        d.best.node.pools[d.best.kind] = pool.filter((it) => it.questionId !== id);
      }
    }

    /* A variant not yet attempted in this block, least recently attempted
       across blocks; once every variant has been tried here, cycle. */
    _pickVariant() {
      const vs = this.target.variants;
      const hist = _readHist(this.target.kc);
      const rank = (id) => { const i = hist.lastIndexOf(id); return i < 0 ? -1 : i; };
      const fresh = vs.filter((v) => !this.attempted.includes(v.questionId));
      const pool = fresh.length ? fresh : vs;
      return pool.slice().sort((a, b) => rank(a.questionId) - rank(b.questionId))[0];
    }

    _recordHist(id) {
      const hist = _readHist(this.target.kc).filter((x) => x !== id);
      hist.push(id);
      _writeHist(this.target.kc, hist);
    }

    /** Fold a graded result back into the plan. Returns a short note for the UI. */
    observe(item, correct) {
      if (!item) return "";
      const isVariant = this.target.variants.some((v) => v.questionId === item.questionId);
      const node = isVariant || item.kc === this.target.kc
        ? this.target
        : this.prereqs.find((p) => p.kc === item.kc);
      if (isVariant && correct) {
        this.solved = true;
        return `Solved on attempt ${this.attempts} — ${this.used} of ${this.quota} questions used.`;
      }
      if (node) {
        const g = isVariant ? GAIN.attempt : (GAIN[item.kind] || GAIN.independent);
        node.m = correct ? _clamp(node.m + g * (1 - node.m)) : _clamp(node.m * MISS_KEEP);
      }
      const R = this.quota - this.used;
      if (R <= 0) return isVariant ? "Out of questions — that was the last attempt." : "";
      if (!correct) {
        const weakest = [this.target, ...this.prereqs]
          .filter((n) => this._nextPrep(n))
          .sort((a, b) => a.m - b.m)[0];
        return weakest && weakest !== this.target
          ? `Miss — re-planning; ${weakest.title || weakest.kc} looks weakest.`
          : "Miss — re-planning within the remaining count.";
      }
      return "";
    }

    outcome() {
      return { solved: this.solved, attempts: this.attempts, used: this.used, quota: this.quota };
    }

    serialize() {
      const node = (n) => ({ kc: n.kc, title: n.title, m: n.m, pools: n.pools, variants: n.variants || [] });
      return {
        attemptFirst: this.attemptFirst, quota: this.quota, used: this.used, solved: this.solved, attempts: this.attempts, preps: this.preps,
        attempted: this.attempted.slice(), target: node(this.target), prereqs: this.prereqs.map(node),
      };
    }

    static restore(saved) {
      if (!saved || !saved.target || !saved.target.kc) return null;
      const fix = (n) => ({
        kc: String(n.kc), title: n.title || n.kc, m: _clamp(Number(n.m) || 0.4),
        pools: {
          faded: (n.pools && n.pools.faded) || [],
          guided: (n.pools && n.pools.guided) || [],
          independent: (n.pools && n.pools.independent) || [],
        },
        variants: Array.isArray(n.variants) ? n.variants : [],
      });
      return new Planner({
        ...saved, target: fix(saved.target), prereqs: (saved.prereqs || []).map(fix),
      });
    }
  }

  return { Planner, masteryFromEstimate, GAIN, READY };
})();

window.ExercisePlanner = ExercisePlanner;
