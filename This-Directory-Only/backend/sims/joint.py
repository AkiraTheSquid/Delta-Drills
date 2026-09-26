"""One belief over the WHOLE learner (Seth 2026-09-26, "#1"): knowledge-space
style, the idea behind ALEKS placement.

Every other belief keeps each concept separately and passes credit between
them with a hand-tuned patch (area prior, capped indirect credit). Here the
belief is a cloud of N joint knowledge states: each particle is a full
learned / not-learned vector over the course. The prior carries the two
correlations real knowledge has:
  - AREA: each particle draws an ability per area, a ~ Beta(0.5, 0.5)
    (people tend to know an area or not), and each concept is learned with
    probability a;
  - PREREQUISITES: a concept whose prerequisites are not all learned in that
    particle is learned with probability x CLOSURE only (soft, not ALEKS's
    hard closure: people do know things out of order).
An answer on c reweights every particle by its likelihood, which uses the
registry's encompassing edges exactly (noisy-AND over the components' states
and recall), so a right answer on a deep concept raises its components and
prerequisites through the joint, and a wrong one on a basic concept lowers
what sits on top of it. Forgetting is the same point FSRS-6 as B.
"""
from __future__ import annotations

import numpy as np

from sims import fsrs_vec as F
from sims.beliefs import T_ANSWER, T_LESSON, T_PROBE, W, BKTBelief
from sims.world import FIRE_SCALE

N_JOINT = 1024
CLOSURE = 0.5
AREA_BETA = 0.5


class JointBelief(BKTBelief):
    name = "joint"

    def __init__(self, g, rng):
        super().__init__(g, rng)
        C, N = len(g.ids), N_JOINT
        ability = rng.beta(AREA_BETA, AREA_BETA, (N, len(self.areas)))
        L = np.zeros((N, C), bool)
        for c in range(C):                    # ids are depth-sorted: prereqs first
            p = ability[:, self.area_ix[c]]
            if g.prereqs[c]:
                p = p * np.where(L[:, g.prereqs[c]].all(1), 1.0, CLOSURE)
            L[:, c] = rng.random(N) < p
        self.L = L
        self.w = np.full(N, 1.0 / N)
        self._marg()

    def _marg(self):
        self.marg = self.w @ self.L

    def _p(self):
        return self.marg

    def _lik_right(self, c, t, R=None):
        """P(right | particle) for an answer on c at t: [N]."""
        g = self.g
        R = self._R(t) if R is None else R
        k = self.L[:, c] * R[c]
        for comp, wt in g.enc[c]:
            k = k * (1 - wt * (1 - self.L[:, comp] * R[comp]))
        return g.guess[c] + (1 - g.guess[c] - g.slip[c]) * k

    def _flip(self, c, T):
        off = ~self.L[:, c]
        self.L[off, c] = self.rng.random(off.sum()) < T

    def _resample(self):
        N = N_JOINT
        if 1.0 / (self.w * self.w).sum() >= N / 4:
            return
        pos = (self.rng.random() + np.arange(N)) / N
        ix = np.minimum(np.searchsorted(np.cumsum(self.w), pos), N - 1)
        self.L = self.L[ix]
        self.w = np.full(N, 1.0 / N)

    def lesson(self, c, t):
        self.touched[c] = True
        self._flip(c, T_LESSON)
        self._marg()

    def answer(self, c, t, correct, probe, lesson_read):
        g = self.g
        self.touched[c] = True
        lik = self._lik_right(c, t)
        self.w = self.w * (lik if correct else 1 - lik)
        self.w /= self.w.sum()
        grade = F.GOOD if correct else F.AGAIN
        mem = (self.S[c], self.D[c], self.tl[c]) if self.has_mem[c] else None
        self.S[c], self.D[c], self.tl[c] = F.review(mem, t, grade, W)
        self.has_mem[c] = True
        if correct:
            for comp, wt in g.enc[c]:
                if self.has_mem[comp]:
                    self.S[comp], self.D[comp], self.tl[comp] = F.implicit(
                        (self.S[comp], self.D[comp], self.tl[comp]), t, wt * FIRE_SCALE, W)
        self._resample()
        self._flip(c, T_ANSWER if lesson_read else T_PROBE)
        self._marg()

    def p_after(self, cands, t):
        """[k, 2, C] P(learned) after a hypothetical wrong / right probe on
        each candidate (the whole graph moves), and P(right) [k]."""
        R = self._R(t)
        lik = np.stack([self._lik_right(c, t, R) for c in cands], 1)       # [N, k]
        wr = self.w[:, None] * lik
        ww = self.w[:, None] * (1 - lik)
        q = wr.sum(0)
        out = np.empty((len(cands), 2, len(self.g.ids)))
        out[:, 1] = (wr.T @ self.L) / np.maximum(q, 1e-300)[:, None]
        out[:, 0] = (ww.T @ self.L) / np.maximum(1 - q, 1e-300)[:, None]
        for j, c in enumerate(cands):           # the probe's own transit
            out[j, :, c] += (1 - out[j, :, c]) * T_PROBE
        return out, q

    def probe_scores(self, t, cands):
        """Variance-drop explorer on the joint (arm K): for each candidate,
        the expected drop in sum_i value_i Var[K_i] from one probe on it
        (already goal-weighted, unlike `info`)."""
        g = self.g
        R = self._R(t)
        pa, q = self.p_after(cands, t)
        var = lambda p: (R * R * p * (1 - p) * g.value).sum(-1)       # noqa: E731
        return var(self.marg) - ((1 - q) * var(pa[:, 0]) + q * var(pa[:, 1]))
