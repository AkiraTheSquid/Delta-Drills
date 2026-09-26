"""The tutor's beliefs under test. Every one answers the same questions, so the
policy in `sim.py` cannot tell them apart except by their numbers:

  K(t)          P(the learner knows c at t), guess aside, per concept
  K_row(c, t)   the same for one concept
  gate(c, t)    the number LEARNED is read off (P(learned); pure FSRS: E[R a day out])
  due_R(t)      the memory term the review queue reads (target 0.80)
  info(t)       expected drop in Var[K now] from one more answer on c
  answer / lesson   evidence

  BKTBelief  — "B": Bayesian P(learned) (guess/slip, lesson and answer
               transits) times a POINT FSRS-6 recall. The app's shape today
               (kc_explore posterior + memory_model), and learning_xp's.
  ParticleBelief(hybrid=False) — "A": Bayesian FSRS-6 and nothing else. Each
               concept is a cloud of (S, D, last review) particles; knowing
               = recall R. "Not known" can only be a tiny S. A lesson is a
               first Good review whose resulting S is uncertain (x e^N(0,1)).
  ParticleBelief(hybrid=True)  — "H": Bayesian FSRS-6 plus a never-learned
               state: particles carry L in {0,1}, K = L * R. This is B with
               the point FSRS replaced by a posterior over S.
  OracleBelief — reads the true learner. An upper bound, not a contender.

Shared by all (so only the belief differs): the area prior read off probe
answers (kc_explore.area_known), indirect credit from a correct answer to the
concepts it encompasses (tempered by the edge weight, capped at +1.2 logit),
FIRe on those concepts' memory, and the app's FSRS weights.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from sims import fsrs_vec as F
from sims.world import FIRE_SCALE, Graph, Truth

W = F.CONCEPT_PRIOR_WEIGHTS
T_LESSON, T_ANSWER, T_PROBE = 0.15, 0.10, 0.03   # learning_xp's transits
AREA_PRIOR_PSEUDO = 2.0
INDIRECT_CLIP = 1.2
N_PARTICLES = 64
LESSON_S_SIGMA = 1.0
SPREAD0 = 0.3              # BKTSpreadBelief: log-S sd right after an answer
SPREAD_GROW = 0.15         # ... plus this x log(1 + days since the last answer)
_GH_Z = np.array([-math.sqrt(3.0), 0.0, math.sqrt(3.0)])   # 3-pt Gauss-Hermite
_GH_W = np.array([1 / 6, 2 / 3, 1 / 6])


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))


class _Shared:
    """Area prior + indirect-credit bookkeeping, identical for every belief."""

    def __init__(self, g: Graph, rng: np.random.Generator):
        self.g, self.rng = g, rng
        C = len(g.ids)
        self.counts: Dict[str, List[int]] = {}
        self.ind = np.zeros(C)
        self.touched = np.zeros(C, bool)
        self.areas = sorted(set(g.area))
        self.area_ix = np.array([self.areas.index(a) for a in g.area])

    def prior(self) -> np.ndarray:
        """kc_explore.area_known for every concept, shrunk to a neutral 0.5."""
        g = self.g
        n = np.array([self.counts.get(a, [0, 0])[0] for a in self.areas], float)[self.area_ix]
        h = np.array([self.counts.get(a, [0, 0])[1] for a in self.areas], float)[self.area_ix]
        neutral = g.guess + (1 - g.guess - g.slip) * 0.5
        acc = (h + AREA_PRIOR_PSEUDO * neutral) / (n + AREA_PRIOR_PSEUDO)
        return np.clip((acc - g.guess) / (1 - g.guess - g.slip), 0.02, 0.98)

    def prior_eff(self) -> np.ndarray:
        return _sig(_logit(self.prior()) + self.ind)

    def record_probe(self, c, correct):
        a = self.g.area[c]
        n, h = self.counts.get(a, [0, 0])
        self.counts[a] = [n + 1, h + int(correct)]

    def indirect_budget(self, comp, wt) -> float:
        """Tempered logit credit for `comp` from a correct answer upstream."""
        g = self.g
        full = wt * math.log((1 - g.slip[comp]) / g.guess[comp])
        d = max(0.0, min(full, INDIRECT_CLIP - self.ind[comp]))
        self.ind[comp] += d
        return d / full if full > 0 else 0.0


class BKTBelief(_Shared):
    name = "bkt"

    def __init__(self, g, rng):
        super().__init__(g, rng)
        C = len(g.ids)
        self.pL = np.zeros(C)
        self.S = np.ones(C)
        self.D = np.full(C, 5.0)
        self.tl = np.zeros(C)
        self.has_mem = np.zeros(C, bool)

    def _p(self):
        return np.where(self.touched, self.pL, self.prior_eff())

    def _R(self, t):
        return np.where(self.has_mem, F.vR(self.S, self.tl, t, W), 1.0)

    def K(self, t):
        return self._p() * self._R(t)

    def K_row(self, c, t):
        return float(self.K(t)[c])

    def due_R(self, t):
        return self._R(t)

    def gate(self, c, t):
        """What LEARNED is read off: P(learned), as the app settles on."""
        return float(self._p()[c])

    def info(self, t):
        g = self.g
        p, R = self._p(), self._R(t)
        e = (1 - g.slip) * R + g.guess * (1 - R)
        q1 = p * e + (1 - p) * g.guess
        p1 = p * e / q1
        p0 = p * (1 - e) / np.maximum(1 - q1, 1e-12)
        var = R * R * p * (1 - p)
        return var - R * R * (q1 * p1 * (1 - p1) + (1 - q1) * p0 * (1 - p0))

    def p_after(self, cands, t):
        """P(learned) for every concept after a hypothetical wrong / right
        PROBE on each candidate: [k, 2, C], and P(right) [k]. Carries the
        probe's own update and the area prior it moves (indirect credit
        aside). Used by the decision-value explorer (voi.py)."""
        g = self.g
        p, R = self._p(), self._R(t)
        out = np.tile(p, (len(cands), 2, 1))
        q = np.empty(len(cands))
        for j, c in enumerate(cands):
            e = (1 - g.slip[c]) * R[c] + g.guess[c] * (1 - R[c])
            q[j] = p[c] * e + (1 - p[c]) * g.guess[c]
            same = (self.area_ix == self.area_ix[c]) & ~self.touched
            a = g.area[c]
            saved = self.counts.get(a)
            for y in (0, 1):
                ll, lu = (e, g.guess[c]) if y else (1 - e, 1 - g.guess[c])
                post = p[c] * ll / (p[c] * ll + (1 - p[c]) * lu)
                n, h = saved or [0, 0]
                self.counts[a] = [n + 1, h + y]
                out[j, y, same] = self.prior_eff()[same]
                out[j, y, c] = post + (1 - post) * T_PROBE
            if saved is None:
                del self.counts[a]
            else:
                self.counts[a] = saved
        return out, q

    def _touch(self, c):
        if not self.touched[c]:
            self.pL[c] = self.prior_eff()[c]
            self.touched[c] = True

    def lesson(self, c, t):
        self._touch(c)
        self.pL[c] += (1 - self.pL[c]) * T_LESSON

    def answer(self, c, t, correct, probe, lesson_read):
        g = self.g
        self._touch(c)
        R = self._R(t)[c]
        e = (1 - g.slip[c]) * R + g.guess[c] * (1 - R)
        ll, lu = (e, g.guess[c]) if correct else (1 - e, 1 - g.guess[c])
        p = self.pL[c]
        self.pL[c] = p * ll / (p * ll + (1 - p) * lu)
        grade = F.GOOD if correct else F.AGAIN
        mem = (self.S[c], self.D[c], self.tl[c]) if self.has_mem[c] else None
        self.S[c], self.D[c], self.tl[c] = F.review(mem, t, grade, W)
        self.has_mem[c] = True
        if correct:
            for comp, wt in g.enc[c]:
                if self.has_mem[comp]:
                    self.S[comp], self.D[comp], self.tl[comp] = F.implicit(
                        (self.S[comp], self.D[comp], self.tl[comp]), t, wt * FIRE_SCALE, W)
                frac = self.indirect_budget(comp, wt)
                if frac and self.touched[comp]:
                    full = wt * math.log((1 - g.slip[comp]) / g.guess[comp])
                    self.pL[comp] = float(_sig(_logit(self.pL[comp]) + frac * full))
        T = T_ANSWER if lesson_read else T_PROBE
        self.pL[c] += (1 - self.pL[c]) * T
        if probe:
            self.record_probe(c, correct)


class BKTSpreadBelief(BKTBelief):
    """B with uncertainty over stability, not just over learned-or-not.

    Point FSRS says "stability is exactly S". Here log S is Normal(log S,
    sigma) with sigma widening with time since the concept's last answer
    (an answer pins it; a long silence loosens it). Three-point Gauss-Hermite
    nodes carry the spread. What changes:
      - recall read everywhere (K, the answer update, review) is E[R] over
        the spread, not R at the point estimate;
      - probe value is the exact variance drop over the joint states
        (learned x stability node) + unlearned, so a probe on a known concept
        whose stability is unsure is worth something (B scores it ~0).
    Gating is unchanged (P(learned))."""
    name = "bkt_spread"

    def _nodes(self, t):
        el = np.maximum(t - self.tl, 0.0)
        sig = SPREAD0 + SPREAD_GROW * np.log1p(el)
        S = self.S[:, None] * np.exp(sig[:, None] * _GH_Z[None, :])
        R = F.vR(S, self.tl[:, None], t, W)
        return np.where(self.has_mem[:, None], R, 1.0)          # [C, 3]

    def _R(self, t):
        return self._nodes(t) @ _GH_W

    def info(self, t):
        g = self.g
        p = self._p()[:, None]
        Rn = self._nodes(t)
        # States: learned at node j (prob p w_j, K = R_j), unlearned (K = 0).
        pr = np.concatenate([p * _GH_W[None, :], 1 - p], 1)            # [C, 4]
        K = np.concatenate([Rn, np.zeros((len(g.ids), 1))], 1)
        lik = g.guess[:, None] + (1 - g.guess[:, None] - g.slip[:, None]) * K

        def var(wm):
            q = wm.sum(1, keepdims=True)
            wn = wm / np.maximum(q, 1e-12)
            m = (wn * K).sum(1, keepdims=True)
            return (wn * (K - m) ** 2).sum(1), q[:, 0]

        v, _ = var(pr)
        v1, q1 = var(pr * lik)
        v0, q0 = var(pr * (1 - lik))
        return v - q1 * v1 - q0 * v0


class ParticleBelief(_Shared):
    def __init__(self, g, rng, hybrid: bool):
        super().__init__(g, rng)
        self.hybrid = hybrid
        self.name = "hybrid" if hybrid else "fsrs"
        C, N = len(g.ids), N_PARTICLES
        h = N // 2
        self.S = np.empty((C, N))
        self.D = np.full((C, N), 5.0)
        self.tl = np.empty((C, N))
        self.L = np.ones((C, N), bool)
        # First half: known before we met them. Second half: never learned.
        self.S[:, :h] = np.exp(rng.normal(math.log(25), 0.8, (C, h)))
        self.tl[:, :h] = -rng.uniform(3, 90, (C, h))
        self.S[:, h:] = F.S_MIN
        self.tl[:, h:] = -365.0
        # Never reviewed: the first lesson/answer is FSRS's FIRST review
        # (init_s/init_d), as in the truth, not a 365-day-overdue one.
        self.new = np.zeros((C, N), bool)
        self.new[:, h:] = not hybrid
        if hybrid:
            self.L[:, h:] = False
            self.S[:, h:] = F.init_s(F.GOOD, W)
        self.logw = np.zeros((C, N))
        self.half = h

    def weights(self) -> np.ndarray:
        N, h = N_PARTICLES, self.half
        pi = self.prior_eff()[:, None]
        untouched = np.concatenate([np.repeat(pi / h, h, 1), np.repeat((1 - pi) / (N - h), N - h, 1)], 1)
        lw = self.logw - self.logw.max(1, keepdims=True)
        touched = np.exp(lw)
        touched /= touched.sum(1, keepdims=True)
        return np.where(self.touched[:, None], touched, untouched)

    def _Kmat(self, t):
        R = F.vR(self.S, self.tl, t, W)
        return np.where(self.L, R, 0.0)

    def K(self, t):
        return (self.weights() * self._Kmat(t)).sum(1)

    def K_row(self, c, t):
        wr = self.weights()[c]
        R = F.vR(self.S[c], self.tl[c], t, W)
        return float((wr * np.where(self.L[c], R, 0.0)).sum())

    def gate(self, c, t):
        """Hybrid: P(learned). Pure FSRS has no learned state, so known = its
        expected recall a day from now."""
        wr = self.weights()[c]
        if self.hybrid:
            return float((wr * self.L[c]).sum())
        return float((wr * F.vR(self.S[c], self.tl[c], t + 1.0, W)).sum())

    def due_R(self, t):
        Wt = self.weights()
        R = F.vR(self.S, self.tl, t, W)
        mass = (Wt * self.L).sum(1)
        return np.where(mass > 1e-9, (Wt * self.L * R).sum(1) / np.maximum(mass, 1e-12), 0.0)

    def info(self, t):
        g = self.g
        Wt, K = self.weights(), self._Kmat(t)
        gu, sl = g.guess[:, None], g.slip[:, None]
        lik1 = gu + (1 - gu - sl) * K

        def var(wm):
            s = wm.sum(1, keepdims=True)
            wn = wm / np.maximum(s, 1e-12)
            m = (wn * K).sum(1, keepdims=True)
            return (wn * (K - m) ** 2).sum(1), s[:, 0]

        v, _ = var(Wt)
        v1, q1 = var(Wt * lik1)
        v0, q0 = var(Wt * (1 - lik1))
        return v - q1 * v1 - q0 * v0

    def _touch(self, c):
        if not self.touched[c]:
            self.logw[c] = np.log(np.maximum(self.weights()[c], 1e-300))
            self.touched[c] = True

    def _fresh(self, c, mask, t):
        self.L[c, mask] = True
        self.S[c, mask] = F.init_s(F.GOOD, W)
        self.D[c, mask] = F.init_d(F.GOOD, W)
        self.tl[c, mask] = t

    def _review(self, c, t, grade):
        s, d = F.v_after(self.S[c], self.D[c], self.tl[c], t, grade, W)
        new = self.new[c]
        s = np.where(new, F.init_s(grade, W), s)
        d = np.where(new, F.init_d(grade, W), d)
        self.new[c] = False
        return s, d

    def lesson(self, c, t):
        self._touch(c)
        if self.hybrid:
            flip = ~self.L[c] & (self.rng.random(N_PARTICLES) < T_LESSON)
            self._fresh(c, flip, t)
        else:
            s, d = self._review(c, t, F.GOOD)
            s = s * np.exp(self.rng.normal(0, LESSON_S_SIGMA, N_PARTICLES))
            self.S[c] = np.clip(s, F.S_MIN, F.S_MAX)
            self.D[c], self.tl[c] = d, t

    def answer(self, c, t, correct, probe, lesson_read):
        g = self.g
        self._touch(c)
        R = F.vR(self.S[c], self.tl[c], t, W)
        K = np.where(self.L[c], R, 0.0)
        lik = g.guess[c] + (1 - g.guess[c] - g.slip[c]) * K
        self.logw[c] += np.log(np.maximum(lik if correct else 1 - lik, 1e-300))
        grade = F.GOOD if correct else F.AGAIN
        s, d = self._review(c, t, grade)
        on = self.L[c]
        self.S[c] = np.where(on, s, self.S[c])
        self.D[c] = np.where(on, d, self.D[c])
        self.tl[c] = np.where(on, t, self.tl[c])
        if self.hybrid:
            T = T_ANSWER if lesson_read else T_PROBE
            self._fresh(c, ~self.L[c] & (self.rng.random(N_PARTICLES) < T), t)
        if correct:
            for comp, wt in g.enc[c]:
                on = self.L[comp] & ~self.new[comp]
                s2, d2, tl2 = F.v_implicit(self.S[comp], self.D[comp], self.tl[comp], t,
                                           wt * FIRE_SCALE, W)
                self.S[comp] = np.where(on, s2, self.S[comp])
                self.D[comp] = np.where(on, d2, self.D[comp])
                self.tl[comp] = np.where(on, tl2, self.tl[comp])
                frac = self.indirect_budget(comp, wt)
                if frac and self.touched[comp]:
                    Kc = np.where(on, F.vR(self.S[comp], self.tl[comp], t, W), 0.0)
                    lc = g.guess[comp] + (1 - g.guess[comp] - g.slip[comp]) * Kc
                    self.logw[comp] += frac * wt * np.log(np.maximum(lc, 1e-300))
                    self._resample(comp)
        self._resample(c)
        if probe:
            self.record_probe(c, correct)

    def _resample(self, c):
        lw = self.logw[c] - self.logw[c].max()
        w = np.exp(lw)
        w /= w.sum()
        if 1.0 / (w * w).sum() >= N_PARTICLES / 2:
            return
        N = N_PARTICLES
        pos = (self.rng.random() + np.arange(N)) / N
        ix = np.minimum(np.searchsorted(np.cumsum(w), pos), N - 1)
        for arr in (self.S, self.D, self.tl, self.L, self.new):
            arr[c] = arr[c][ix]
        self.S[c] = np.clip(self.S[c] * np.exp(self.rng.normal(0, 0.05, N)), F.S_MIN, F.S_MAX)
        self.logw[c] = 0.0


class OracleBelief:
    """Reads the true learner: what a perfect model would say."""
    name = "oracle"

    def __init__(self, g: Graph, truth: Truth):
        self.g, self.truth = g, truth

    def K(self, t):
        return np.array([self.truth.know(c, t) for c in range(len(self.g.ids))])

    def K_row(self, c, t):
        return self.truth.know(c, t)

    def due_R(self, t):
        return np.array([self.truth._R(c, t) if self.truth.mem[c] else 1.0
                         for c in range(len(self.g.ids))])

    def gate(self, c, t):
        """The true learned state — what P(learned) estimates. (Gating on the
        durable crossing itself holds every concept back until reviews have
        spaced it, which no belief does; it made a poor bound.)"""
        tr = self.truth
        return float(tr.L[c]) if tr.world == "W1" else float(tr.k[c])

    def info(self, t):
        return np.zeros(len(self.g.ids))

    def lesson(self, c, t):
        pass

    def answer(self, c, t, correct, probe, lesson_read):
        pass
