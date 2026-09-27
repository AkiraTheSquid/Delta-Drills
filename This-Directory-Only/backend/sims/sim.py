"""One simulated learner under one arm: the shared policy and the run loop.

The policy is the same for every arm; only the belief it reads changes
(`beliefs.py`). Its shape is today's app, simplified:

  1. REVIEW   — a concept marked learned whose belief recall has fallen below
                `review_at`, every other action at most. Tuned separately from
                the gate (Seth 09-26): unlocking the next concept and keeping
                this one up are different decisions.
  2. EXPLORE  — (explore arms only) a concept not yet taught, not learned,
                whose P(known) clears the cost gate (drill/lesson minutes) and
                whose prerequisites are learned or not refuted, ranked by
                goal value x expected variance drop. Served with no lesson.
                After a break of 14+ days, concepts taught before it are
                probed again for 7 days (kc_explore's return window).
  3. EXPLOIT  — the frontier (not learned, every prerequisite learned):
                lesson first, then drills, until the belief marks it learned;
                at most DAY_CAP answers on one concept per session (the app's
                cap), so practice spaces itself across days.

LEARNED (sticky) = the belief's `gate` >= theta after at least one of the
concept's own answers: P(learned) for the BKT and hybrid beliefs, expected
recall a day out for pure FSRS (it has no learned state). Each arm is run at
several theta and compared at its best. The oracle gates on the truth itself.

METRIC: the summed practice hours at which each ARENA goal TRULY reaches
P(correct a day later) >= 0.80, checked at the end of every session.
A goal never reached counts as 1.5 x the horizon.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from sims import beliefs as B
from sims import voi
from sims.joint import JointBelief
from sims.world import Graph, Truth, minutes

SESSION_MIN = 60.0          # practice minutes per day
HORIZON_H = 150.0
MAX_DAYS = 500.0            # a policy stuck with nothing to serve ends here (goals left = penalty)
BREAK_AT_H = 40.0
REFUTED = 0.20
COST_RATIO = {"math": 1.5 / 6.0, "code": 4.0 / 8.0}
DAY_CAP = 3                 # drills on one concept per session (the app's cap)
RETURN_GAP, RETURN_WINDOW = 14.0, 7.0

ARMS = {
    # name: (belief, explore, "var" | "voi" probe score)
    "C": ("bkt", False, None),         # control: today's belief, no explore
    "B": ("bkt", True, "var"),         # Bayesian BKT x point FSRS + explore
    "BS": ("bkt_spread", True, "var"),  # B, with a spread on FSRS stability
    "BV": ("bkt", True, "voi"),        # B, probing for minutes saved (voi.py)
    "K": ("joint", True, "var"),       # one joint belief over the graph (joint.py)
    "KV": ("joint", True, "voi"),      # joint belief + minutes-saved probing
    "A0": ("fsrs", False, None),       # Bayesian FSRS alone, no explore
    "A": ("fsrs", True, "var"),        # Bayesian FSRS alone + explore
    "H": ("hybrid", True, "var"),      # Bayesian FSRS + learned state + explore
    "B9": ("bkt", True, "var"),        # B at a fixed gate 0.9 (W1's best): a reference
    "P": ("bkt", True, "var"),         # B + rollout planner over its knobs (planner.py)
    "O": ("oracle", False, None),      # sees the truth
}


@dataclass
class Result:
    sum_h: float
    crossed: int
    goals: int
    false_decl: int
    decl: int
    probes: int
    lessons: int
    reviews: int
    drills: int
    plan: str = ""            # planner arm: its knob choices, P(W1) at the end


def make_belief(kind, g, truth, seed):
    rng = np.random.default_rng([seed, 7, ("bkt", "fsrs", "hybrid", "oracle", "bkt_spread", "joint").index(kind)])
    if kind == "bkt":
        return B.BKTBelief(g, rng)
    if kind == "joint":
        return JointBelief(g, rng)
    if kind == "bkt_spread":
        return B.BKTSpreadBelief(g, rng)
    if kind == "fsrs":
        return B.ParticleBelief(g, rng, hybrid=False)
    if kind == "hybrid":
        return B.ParticleBelief(g, rng, hybrid=True)
    return B.OracleBelief(g, truth)


class Episode:
    """One learner under one arm, as state that can be copied mid-course
    (the planner forks it to simulate ahead). `session()` plays one day."""

    def __init__(self, g: Graph, truth, bel, arm: str, theta: float,
                 break_days: float, review_at: float = 0.80):
        self.g, self.truth, self.bel = g, truth, bel
        self.arm = arm
        self.kind, self.explore, self.scorer = ARMS[arm]
        self.oracle = self.kind == "oracle"
        self.theta, self.review_at, self.break_days = theta, review_at, break_days
        C = len(g.ids)
        self.learned = np.zeros(C, bool)
        self.lesson_read = np.zeros(C, bool)
        self.own_answers = np.zeros(C, int)
        self.last_answer = np.full(C, -1e9)
        self.focus: Optional[int] = None
        self.today = np.zeros(C, int)                # drills per concept this session
        self.crossed_at = np.full(C, np.nan)
        self.stats = dict(false_decl=0, decl=0, probes=0, lessons=0, reviews=0, drills=0)
        self.cost = np.array([COST_RATIO[k] for k in g.kind])
        self.practice_min, self.day = 0.0, 0.0
        self.took_break = break_days <= 0
        self.last_was_review = False
        self.return_until, self.stale = -1.0, np.zeros(C, bool)
        self.last_session_day = None
        self.observer = None                         # f(what, c, t, correct) per event, before the belief
        self.planner = None                          # f(episode) before each session

    @staticmethod
    def _pre_mat(g):
        """[C, C] bool: row c marks c's prerequisites (cached on the graph)."""
        m = getattr(g, "_pre_mat", None)
        if m is None:
            m = np.zeros((len(g.ids), len(g.ids)), bool)
            for c, ps in enumerate(g.prereqs):
                m[c, ps] = True
            g._pre_mat = m
        return m

    def fork(self, truth):
        """A copy that shares nothing mutable with this one, on `truth`."""
        ep = copy.copy(self)
        ep.truth = truth
        ep.bel = copy.deepcopy(self.bel, {id(self.g): self.g})
        for k in ("learned", "lesson_read", "own_answers", "last_answer", "today",
                  "crossed_at", "stale"):
            setattr(ep, k, getattr(self, k).copy())
        ep.stats = dict(self.stats)
        ep.observer = ep.planner = None
        return ep

    def mark_learned(self, c, t):
        if self.learned[c]:
            return
        if self.bel.gate(c, t) >= self.theta and (self.oracle or self.own_answers[c] > 0):
            self.learned[c] = True
            if self.g.goal[c]:
                self.stats["decl"] += 1
                self.stats["false_decl"] += int(not self.truth.crossed(c, t))

    def done(self) -> bool:
        return self.practice_min >= HORIZON_H * 60 or self.day >= MAX_DAYS

    def run(self) -> Result:
        while not self.done():
            if self.planner is not None:
                self.planner(self)
            if self.session():
                break
        return self.result()

    def _choose(self, t):
        g, bel, theta = self.g, self.bel, self.theta
        learned, lesson_read, today = self.learned, self.lesson_read, self.today
        C = len(g.ids)
        K = bel.K(t)
        if self.oracle:
            for c in np.flatnonzero(~learned):
                self.mark_learned(c, t)
        dR = bel.due_R(t)
        due = learned & (dR < self.review_at)
        pm = self._pre_mat(g)
        if due.any() and not self.last_was_review:
            cands = np.flatnonzero(due)
            c = cands[np.lexsort((dR[cands], -g.value[cands]))[0]]
            return ("review", c)
        if self.explore:
            ok_pre = learned | (K >= REFUTED)
            pre_ok = ~(pm & ~ok_pre).any(1)
            fresh = ~learned & ~lesson_read & pre_ok & (today < DAY_CAP)
            back = (self.stale & (today < DAY_CAP) if t < self.return_until
                    else np.zeros(C, bool))
            if self.scorer == "voi" and back.any():
                # Return window: re-probe what the break may have erased
                # first, by the same rule every explore arm uses.
                score = np.where(back, g.value * bel.info(t), -1.0)
                c = int(np.argmax(score))
                if score[c] > 1e-6:
                    return ("probe", c)
            elif self.scorer == "voi":
                # Probe for minutes saved (voi.py); no cost gate: the
                # score already charges the probe's minutes.
                cands = np.flatnonzero(fresh)
                if len(cands):
                    pa, q = bel.p_after(cands, t)
                    sc = voi.scores(g, theta, cands, bel._p(), pa, q, learned, lesson_read)
                    j = int(np.argmax(sc))
                    if sc[j] > 0:
                        return ("probe", int(cands[j]))
            else:
                cand = (fresh & (K >= self.cost)) | back
                if cand.any():
                    if hasattr(bel, "probe_scores"):
                        score = np.full(C, -1.0)
                        ix = np.flatnonzero(cand)
                        score[ix] = bel.probe_scores(t, ix)
                    else:
                        score = np.where(cand, g.value * bel.info(t), -1.0)
                    c = int(np.argmax(score))
                    if score[c] > 1e-6:
                        return ("probe", c)
        focus = self.focus
        if focus is None or learned[focus] or today[focus] >= DAY_CAP:
            front = np.flatnonzero(~learned & (today < DAY_CAP) & ~(pm & ~learned).any(1))
            focus = self.focus = (int(max(front, key=lambda c: (g.value[c], -g.depth[c], -c)))
                                  if len(front) else None)
        if focus is not None:
            return ("lesson", focus) if not lesson_read[focus] else ("drill", focus)
        # Frontier used up for today (or done): review the weakest learned
        # concept not yet answered today, else end the session.
        spare = np.flatnonzero(learned & (today == 0))
        if not len(spare):
            return None
        return ("review", spare[np.argmin(dR[spare])])

    def session(self) -> bool:
        """Play one day. True when every goal has crossed (the run is over)."""
        g, truth, bel = self.g, self.truth, self.bel
        t = self.day + 0.4
        if (self.last_session_day is not None and self.explore
                and self.day - self.last_session_day >= RETURN_GAP):
            self.return_until = t + RETURN_WINDOW
            self.stale = ((self.lesson_read | self.learned)
                          & (self.last_answer < self.last_session_day + 1))
        self.last_session_day = self.day
        self.today[:] = 0
        session_end = min(self.practice_min + SESSION_MIN, HORIZON_H * 60)
        while self.practice_min < session_end:
            action = self._choose(t)
            if action is None:
                break
            what, c = action
            self.last_was_review = what == "review"
            if what == "lesson":
                truth.lesson(c, t)
                if self.observer is not None:
                    self.observer(what, c, t, None)
                bel.lesson(c, t)
                self.lesson_read[c] = True
                self.stats["lessons"] += 1
                dt = minutes(g, c, "lesson")
            else:
                probe = what == "probe"
                correct = truth.answer(c, t)
                if self.observer is not None:       # before the belief moves
                    self.observer(what, c, t, correct)
                bel.answer(c, t, correct, probe=probe and not self.lesson_read[c],
                           lesson_read=self.lesson_read[c])
                self.own_answers[c] += 1
                self.last_answer[c] = t
                if probe:
                    self.stale[c] = False
                    self.stats["probes"] += 1
                elif what == "review":
                    self.stats["reviews"] += 1
                else:
                    self.stats["drills"] += 1
                self.today[c] += 1
                self.mark_learned(c, t)
                dt = minutes(g, c, "drill")
            self.practice_min += dt
            t += dt / 1440.0
        for c in np.flatnonzero(g.goal & np.isnan(self.crossed_at)):
            if truth.crossed(c, t):
                self.crossed_at[c] = self.practice_min / 60.0
        if not np.isnan(self.crossed_at[g.goal]).any():
            return True
        self.day += 1.0
        if not self.took_break and self.practice_min >= BREAK_AT_H * 60:
            self.day += self.break_days
            self.took_break = True
        return False

    def result(self, plan: str = "") -> Result:
        goal_h = self.crossed_at[self.g.goal]
        n_cross = int((~np.isnan(goal_h)).sum())
        total = float(np.nansum(goal_h) + (len(goal_h) - n_cross) * 1.5 * HORIZON_H)
        return Result(total, n_cross, len(goal_h), **self.stats, plan=plan)


def run(g: Graph, world: str, ltype: str, seed: int, arm: str, theta: float,
        break_days: float, review_at: float = 0.80) -> Result:
    truth = Truth(g, world, ltype, seed)
    if arm == "P":
        from sims.planner import run_planned
        return run_planned(g, truth, seed, theta, break_days, review_at)
    bel = make_belief(ARMS[arm][0], g, truth, seed)
    return Episode(g, truth, bel, arm, theta, break_days, review_at).run()
