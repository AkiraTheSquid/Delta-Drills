"""#3 (Seth 2026-09-26): a better PLANNER, not a better belief.

Arm P is B (same belief, same review -> explore -> exploit rule) plus a planner
that re-chooses B's two knobs (gate theta, review threshold) for THIS learner
by simulating the rest of the course before choosing:

  1. A particle filter over the LEARNER, per truth-world hypothesis (W1
     two-state, W2 gradual): for every concept a cloud of (learned flag or
     strength, FSRS memory, own learning speed) particles, driven by the
     simulator's own transition rules and updated with every lesson and
     observed answer. The two worlds' running marginal likelihoods give
     P(W1 | this learner's answers): "does knowledge jump or grow, for this
     learner?". (A first version kept one cloud of whole learners; with 71
     concepts it degenerated and misread every W1 learner as W2.)
  2. At a few session counts (PLAN_AT), draw N_ROLL learners from the
     posterior and, for every knob pair in KNOBS, fork the live episode onto
     each drawn learner and play it to the end of the course. The same drawn
     learners and random numbers serve every knob (common random numbers).
  3. Keep the knob pair with the lowest mean summed goal-crossing hours (the
     final metric itself) until the next plan.

What it may and may not see: never the true learner. It DOES know the model
family the true learner comes from (both worlds' transition rules, the speed
spread), which a deployed planner would have to fit from data. It does NOT fit
the learner's own forgetting weights (population FSRS weights; the truth's are
+-25%). So a win here is an upper-end estimate; a loss is a strong negative.
"""
from __future__ import annotations

import copy
import math
import os

import numpy as np

from sims import fsrs_vec as F
from sims import sim
from sims.beliefs import W
from sims.world import FIRE_SCALE, MAX_EVENTS, TAU, Truth

WORLDS = ("W1", "W2")
KNOBS = tuple((th, rv) for th in (0.9, 0.95, 0.99) for rv in (0.7, 0.8, 0.9))
N_CLOUD = 64                                # particles per (world, concept)
N_ROLL = 8
PLAN_AT = (2, 5, 10, 20, 35, 55, 80, 110)   # re-plan before these sessions (0-based)
FAST = os.environ.get("SIM_FAST", "1") != "0"   # rollouts on fast.py (numba; identical results)
if FAST:
    from sims import fast
SPEED_JITTER = 0.1                          # log-sd on a resampled duplicate's speeds
HYPO_SEED0 = 10 ** 6                        # hypothesized learners never reuse a real seed


class Hypo(Truth):
    """A hypothetical learner: the Truth model with no common-random-number
    arrays (its chance events draw from its own generator) until it is set
    up for a rollout."""
    crn_events = 0

    def __post_init__(self):
        super().__post_init__()
        self.rng2 = np.random.default_rng([self.seed, 99])

    def _draw_learn(self, c):
        if self.u_learn.shape[1] == 0:
            return self.rng2.random()
        return super()._draw_learn(c)

    def clone(self, rng=None) -> "Hypo":
        h = copy.copy(self)
        for k in ("speed", "L", "k", "lesson_read", "n_ans", "n_learn"):
            setattr(h, k, getattr(self, k).copy())
        h.mem = list(self.mem)
        if rng is not None:
            h.rng2 = np.random.default_rng(int(rng.integers(2 ** 63)))
        return h

    def for_rollout(self, u_ans, u_learn) -> "Hypo":
        """A copy that answers from the given (shared, read-only) arrays."""
        h = self.clone()
        h.u_ans, h.u_learn = u_ans, u_learn
        h.n_ans = np.zeros_like(self.n_ans)
        h.n_learn = np.zeros_like(self.n_learn)
        return h


class Cloud:
    """One concept's weighted particles under one world hypothesis: learned
    flag (W1) or strength (W2), FSRS memory, own learning speed."""

    def __init__(self, world, p_known, rng):
        M = N_CLOUD
        self.world = world
        known = rng.random(M) < p_known
        self.L = known.copy()
        self.k = np.where(known, rng.uniform(0.85, 1.0, M), 0.0)
        self.has = known.copy()
        self.S = np.where(known, np.exp(rng.normal(math.log(25), 0.7, M)), 1.0)
        self.D = np.full(M, 5.0)
        self.tl = np.where(known, -rng.uniform(3, 60, M), 0.0)
        self.speed = np.exp(rng.normal(0, 0.35, M))
        self.logw = np.full(M, -math.log(M))

    def own(self, t):
        R = np.where(self.has, F.vR(self.S, self.tl, t, W), 0.0)
        return (self.L if self.world == "W1" else self.k) * R

    def mean_own(self, t):
        return float(np.exp(self.logw) @ self.own(t))

    def review(self, t, grade, mask):
        """FSRS review with `grade` on the particles in `mask` (first review
        where they have no memory yet)."""
        new = mask & ~self.has
        old = mask & self.has
        if old.any():
            s, d = F.v_after(self.S[old], self.D[old], self.tl[old], t, grade, W)
            self.S[old], self.D[old], self.tl[old] = s, d, t
        if new.any():
            self.S[new], self.D[new], self.tl[new] = F.init_s(grade, W), F.init_d(grade, W), t
            self.has[new] = True

    def implicit(self, t, k):
        m = self.has
        if m.any():
            self.S[m], self.D[m], self.tl[m] = F.v_implicit(self.S[m], self.D[m], self.tl[m], t, k, W)

    def resample(self, rng):
        wt = np.exp(self.logw)
        M = len(wt)
        if 1.0 / (wt * wt).sum() >= M / 2:
            return
        pos = (rng.random() + np.arange(M)) / M
        ix = np.minimum(np.searchsorted(np.cumsum(wt), pos), M - 1)
        dup = np.r_[False, ix[1:] == ix[:-1]]
        for a in ("L", "k", "has", "S", "D", "tl", "speed"):
            setattr(self, a, getattr(self, a)[ix].copy())
        self.speed[dup] *= np.exp(rng.normal(0, SPEED_JITTER, dup.sum()))
        self.logw = np.full(M, -math.log(M))


def _lse(x):
    m = x.max()
    return m + math.log(np.exp(x - m).sum())


class LearnerFilter:
    """P(learner | lessons, answers), factored by concept: one particle cloud
    per (world, concept), coupled through the components' and prerequisites'
    posterior-mean recall (a joint cloud over 71 concepts degenerates, and
    that penalises the all-or-nothing W1 likelihood most, so world odds read
    off it were wrong). A concept's cloud starts at first use from B's
    P(learned) at that moment (area evidence included), the same for both
    worlds. Each world's running marginal likelihood gives P(W1)."""

    def __init__(self, g, bel, seed):
        self.g, self.bel = g, bel
        self.rng = np.random.default_rng([seed, 31])
        self.clouds = {w: {} for w in WORLDS}
        self.logZ = {w: 0.0 for w in WORLDS}
        self.lesson_read = np.zeros(len(g.ids), bool)

    def _cloud(self, w, c):
        cl = self.clouds[w].get(c)
        if cl is None:
            cl = self.clouds[w][c] = Cloud(w, float(self.bel._p()[c]), self.rng)
        return cl

    def _pre(self, w, c, t):
        out = 1.0
        for p in self.g.prereqs[c]:
            out *= 1 - TAU * (1 - self._cloud(w, p).mean_own(t))
        return out

    def _p_correct(self, w, c, t):
        g = self.g
        know = self._cloud(w, c).own(t)
        for comp, wt in g.enc[c]:
            know = know * (1 - wt * (1 - self._cloud(w, comp).mean_own(t)))
        return g.guess[c] + (1 - g.guess[c] - g.slip[c]) * know

    def lesson(self, c, t):
        self.lesson_read[c] = True
        for w in WORLDS:
            cl = self._cloud(w, c)
            rate = np.minimum(0.95, 0.6 * self._pre(w, c, t) * cl.speed)
            if w == "W1":
                learn = ~cl.L & (self.rng.random(N_CLOUD) < rate)
                if cl.L.any():
                    m = cl.L & cl.has
                    cl.S[m], cl.D[m], cl.tl[m] = F.v_implicit(cl.S[m], cl.D[m], cl.tl[m], t, 0.5, W)
                cl.L |= learn
                cl.review(t, F.GOOD, learn)
            else:
                cl.k += rate * (1 - cl.k)
                old = cl.has.copy()
                if old.any():
                    cl.S[old], cl.D[old], cl.tl[old] = F.v_implicit(
                        cl.S[old], cl.D[old], cl.tl[old], t, 0.5, W)
                cl.review(t, F.GOOD, ~old)

    def answer(self, c, t, correct):
        g = self.g
        grade = F.GOOD if correct else F.AGAIN
        base = 0.15 if self.lesson_read[c] else 0.04
        for w in WORLDS:
            cl = self._cloud(w, c)
            q = self._p_correct(w, c, t)
            lw = cl.logw + np.log(np.clip(q if correct else 1 - q, 1e-12, 1.0))
            step = _lse(lw)
            self.logZ[w] += step
            cl.logw = lw - step
            rate = base * self._pre(w, c, t) * cl.speed
            if w == "W1":
                was = cl.L.copy()
                cl.review(t, grade, was)
                learn = ~was & (self.rng.random(N_CLOUD) < np.minimum(0.95, rate))
                cl.L |= learn
                cl.review(t, F.GOOD, learn)
            else:
                cl.k += np.minimum(0.95, 2.7 * rate) * (1 - cl.k)
                cl.review(t, grade, np.ones(N_CLOUD, bool))
            if correct:
                for comp, wt in g.enc[c]:
                    self._cloud(w, comp).implicit(t, wt * FIRE_SCALE)
            cl.resample(self.rng)

    def p_w1(self) -> float:
        d = self.logZ["W2"] - self.logZ["W1"]
        return 1.0 / (1.0 + math.exp(min(d, 700.0)))

    def sample(self, n, rng, lesson_read):
        """n hypothetical learners drawn from the posterior, as Hypo."""
        p1 = self.p_w1()
        out = []
        for _ in range(n):
            w = "W1" if rng.random() < p1 else "W2"
            h = Hypo(self.g, w, "novice", HYPO_SEED0 + int(rng.integers(10 ** 9)))
            h.w = tuple(W)
            h.lesson_read = lesson_read.copy()
            for c in range(len(self.g.ids)):
                cl = self._cloud(w, c)
                wt = np.exp(cl.logw)
                i = int(rng.choice(N_CLOUD, p=wt / wt.sum()))
                h.L[c], h.k[c], h.speed[c] = cl.L[i], cl.k[i], cl.speed[i]
                h.mem[c] = (float(cl.S[i]), float(cl.D[i]), float(cl.tl[i])) if cl.has[i] else None
            out.append(h)
        return out


class Planner:
    def __init__(self, g, filt, seed):
        self.g, self.filt, self.seed = g, filt, seed
        self.sessions = 0
        self.log = []

    def __call__(self, ep):
        n = self.sessions
        self.sessions += 1
        if n not in PLAN_AT:
            return
        rng = np.random.default_rng([self.seed, 77, n])
        C = len(self.g.ids)
        score = np.zeros(len(KNOBS))
        for h in self.filt.sample(N_ROLL, rng, ep.lesson_read):
            u_ans, u_learn = rng.random((C, MAX_EVENTS)), rng.random((C, MAX_EVENTS))
            for j, (th, rv) in enumerate(KNOBS):
                if FAST:
                    score[j] += fast.run(ep, th, rv, truth=h.for_rollout(u_ans, u_learn)).sum_h
                    continue
                e = ep.fork(h.for_rollout(u_ans, u_learn))
                e.theta, e.review_at = th, rv
                score[j] += e.run().sum_h
        ep.theta, ep.review_at = KNOBS[int(np.argmin(score))]
        self.log.append(f"{n}:{ep.theta}/{ep.review_at}@{self.filt.p_w1():.2f}")


def run_planned(g, truth, seed, theta, break_days, review_at) -> sim.Result:
    """Arm P on one learner: starts at B's knobs (theta, review_at)."""
    bel = sim.make_belief("bkt", g, truth, seed)           # B's belief, B's random stream
    ep = sim.Episode(g, truth, bel, "P", theta, break_days, review_at)
    filt = LearnerFilter(g, bel, seed)

    def observe(what, c, t, correct):
        if what == "lesson":
            filt.lesson(c, t)
        else:
            filt.answer(c, t, correct)

    ep.observer = observe
    ep.planner = planner = Planner(g, filt, seed)
    res = ep.run()
    res.plan = " ".join(planner.log) + f" | P(W1)={filt.p_w1():.2f}"
    return res
