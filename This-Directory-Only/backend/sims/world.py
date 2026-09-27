"""The simulated learner (the hidden truth) and the concept graph it lives on.

Two truth worlds, so neither belief under test gets its own assumptions for
free (Seth, 2026-09-26):

  W1 "two-state": each concept is learned or not (the BKT/HMM world). Once
     learned, FSRS-6 forgetting runs on it.
  W2 "gradual":   no learned flag. Each concept has a strength k in [0, 1]
     that grows a fraction of what is left with every lesson and answer
     (the continuous-skill world FSRS-alone assumes). FSRS-6 forgetting
     scales it.

  W3 "misspecified" (Seth 2026-09-26: did the planner overfit the
     simulator?): a world NO belief or planner models. Each concept is
     randomly jump-style (as W1) or gradual with an S-shaped curve (slow
     start: growth x (0.3 + k), unlike W2's fraction-of-what-is-left);
     forgetting weights vary ~2x more per learner; prerequisites matter more
     (tau 0.8); and one day in five is a bad day (knowing x 0.75 on answers,
     not on the crossing metric, which measures ability).

All: the learner's own FSRS weights are the concept prior perturbed per
learner, each concept learns at its own speed, and an answer also needs the
concepts it encompasses (noisy-AND over the registry's `encompassing` edges).
Knowing the prerequisites speeds learning (multiplicative, tau 0.5 per
prerequisite, the 09-24 sims' T1 form).

COMMON RANDOM NUMBERS: every random draw the truth makes is indexed by
(concept, how many times that concept has been answered / taught), from
arrays drawn once per learner. Two arms that ask the same concept the same
number of times see the same answers; a comparison between arms is paired
per learner, not two independent samples.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from sims import fsrs_vec as F

REGISTRY = Path(__file__).resolve().parents[3] / "Local_Deployed_Shared/lessons/kc_registry.json"

# Guess / slip by concept kind: kc_explore's numbers (MC math, graded code).
GUESS_SLIP = {"math": (0.25, 0.10), "code": (0.10, 0.15)}
# Minutes: kc_explore's cost-gate calibration (math MC 1.5 of a 6 min page,
# code 4 of an 8 min page). Not measured; the same for every arm.
MINUTES = {"math": {"lesson": 6.0, "drill": 1.5}, "code": {"lesson": 8.0, "drill": 4.0}}
TAU = 0.5                  # prerequisite transfer (T1)
FIRE_SCALE = 0.5           # implicit repetition share, memory_model's
CROSS_P = 0.80             # a goal is mastered when P(correct) a day out >= this
LEARNER_TYPES = ("novice", "torch", "math", "strong")
WORLDS = ("W1", "W2", "W3")
MAX_EVENTS = 2000          # CRN array depth per concept (MAX_DAYS x cap, with room)


@dataclass
class Graph:
    ids: List[str]
    idx: Dict[str, int]
    kind: List[str]                        # "math" | "code"
    area: List[str]
    prereqs: List[List[int]]
    enc: List[List[Tuple[int, float]]]     # direct encompassing edges
    goal: np.ndarray                       # bool [C]
    value: np.ndarray                      # goals at or downstream of c
    depth: np.ndarray
    guess: np.ndarray
    slip: np.ndarray


def load_graph() -> Graph:
    """The ARENA goals (registry `arena_section`) and every prerequisite
    under them — the course the learner is actually on."""
    raw = json.loads(REGISTRY.read_text())
    kcs = {k["id"]: k for k in raw["kcs"]}
    goals = [k for k, v in kcs.items() if v.get("arena_section")]
    rel, stack = set(), list(goals)
    while stack:
        x = stack.pop()
        if x not in rel:
            rel.add(x)
            stack += [p for p in kcs[x].get("prereqs") or [] if p in kcs]
    depth: Dict[str, int] = {}

    def d(x):
        if x not in depth:
            ps = [p for p in kcs[x].get("prereqs") or [] if p in rel]
            depth[x] = 1 + max((d(p) for p in ps), default=-1)
        return depth[x]

    ids = sorted(rel, key=lambda x: (d(x), x))
    idx = {k: i for i, k in enumerate(ids)}
    prereqs = [[idx[p] for p in kcs[k].get("prereqs") or [] if p in idx] for k in ids]
    enc = [[(idx[c], float(w)) for c, w in (kcs[k].get("encompassing") or {}).items()
            if c in idx and c != k and float(w) > 0] for k in ids]
    kind = ["math" if k.startswith("math.") else "code" for k in ids]
    goal = np.array([bool(kcs[k].get("arena_section")) for k in ids])
    # value[c] = goals reachable downstream of c (itself included).
    children: List[List[int]] = [[] for _ in ids]
    for c, ps in enumerate(prereqs):
        for p in ps:
            children[p].append(c)
    value = np.zeros(len(ids))
    for c in range(len(ids)):
        seen, st = set(), [c]
        while st:
            x = st.pop()
            if x not in seen:
                seen.add(x)
                st += children[x]
        value[c] = sum(goal[x] for x in seen)
    gs = np.array([GUESS_SLIP[k] for k in kind])
    return Graph(ids, idx, kind, [k.split(".")[0] for k in ids], prereqs, enc, goal,
                 value, np.array([depth[k] for k in ids]), gs[:, 0], gs[:, 1])


def minutes(g: Graph, c: int, what: str) -> float:
    return MINUTES[g.kind[c]][what]


def _known_areas(ltype: str) -> Dict[str, float]:
    """P(a concept of each area is known at the start), by learner type."""
    if ltype == "torch":
        return {"torch": .85, "tensor": .85, "einops": .85, "python": .85}
    if ltype == "math":
        return {"math": .85, "python": .85}
    if ltype == "strong":
        return {a: .8 for a in ("torch", "tensor", "einops", "python", "math", "cnn", "raytracing")}
    return {}


@dataclass
class Truth:
    """One hidden learner. `world` is "W1", "W2" or "W3"."""
    g: Graph
    world: str
    ltype: str
    seed: int
    w: tuple = field(init=False)
    speed: np.ndarray = field(init=False)
    L: np.ndarray = field(init=False)          # W1 learned flag
    k: np.ndarray = field(init=False)          # W2 strength
    mem: List[Optional[tuple]] = field(init=False)
    lesson_read: np.ndarray = field(init=False)
    n_ans: np.ndarray = field(init=False)
    n_learn: np.ndarray = field(init=False)
    crn_events = MAX_EVENTS    # CRN array depth; the planner's hypothesized learners use 0
    tau = TAU

    def __post_init__(self):
        rng = np.random.default_rng([self.seed, WORLDS.index(self.world) + 1,
                                     LEARNER_TYPES.index(self.ltype)])
        w3 = self.world == "W3"
        C = len(self.g.ids)
        w = np.array(F.CONCEPT_PRIOR_WEIGHTS)
        # The learner's own forgetting: stability weights +-25%, decay +-15%.
        for i in (0, 1, 2, 3, 8, 11, 13):
            w[i] *= math.exp(rng.normal(0, 0.5 if w3 else 0.22))
        w[20] *= math.exp(rng.normal(0, 0.3 if w3 else 0.13))
        self.w = tuple(w)
        self.speed = np.exp(rng.normal(0, 0.35, C))
        self.u_ans = rng.random((C, self.crn_events))
        self.u_learn = rng.random((C, self.crn_events))
        self.L = np.zeros(C, bool)
        self.k = np.zeros(C)
        self.mem = [None] * C
        self.lesson_read = np.zeros(C, bool)
        self.n_ans = np.zeros(C, int)
        self.n_learn = np.zeros(C, int)
        known = _known_areas(self.ltype)
        start = rng.random(C)
        for c in range(C):
            p = known.get(self.g.area[c], 0.0)
            if self.ltype == "strong" and self.g.goal[c]:
                p = 0.3
            if start[c] < p:
                self.L[c] = True
                self.k[c] = rng.uniform(0.85, 1.0)
                self.mem[c] = (float(np.exp(rng.normal(math.log(25), 0.7))), 5.0,
                               -float(rng.uniform(3, 60)))
        # jump[c]: the concept is learned-or-not (W1) rather than a strength (W2).
        self.jump = np.full(C, self.world == "W1")
        self.bad_day = np.zeros(0, bool)
        if w3:                                 # drawn last: W1/W2 streams untouched
            self.jump = rng.random(C) < 0.5
            self.bad_day = rng.random(2000) < 0.2
            self.tau = 0.8

    # --- what the learner knows --------------------------------------------
    _fw = None                 # the w that _fac/_dec were computed for (F.R, hoisted: hot)

    def _R(self, c, t):
        m = self.mem[c]
        if m is None:
            return 0.0
        if self._fw is not self.w:
            self._fw, self._fac, self._dec = self.w, F.factor(self.w), F.decay(self.w)
        return (1 + self._fac * max(0.0, t - m[2]) / m[0]) ** self._dec

    def own(self, c, t) -> float:
        """This concept's own step, recalled at t (no components)."""
        base = float(self.L[c]) if self.jump[c] else self.k[c]
        return base * self._R(c, t)

    def know(self, c, t) -> float:
        """P(the learner can do a novel problem on c at t), guess aside."""
        p = self.own(c, t)
        for comp, wt in self.g.enc[c]:
            p *= 1 - wt * (1 - self.own(comp, t))
        return p

    def p_correct(self, c, t, luck=True) -> float:
        gu, sl = self.g.guess[c], self.g.slip[c]
        know = self.know(c, t)
        if luck and len(self.bad_day) and self.bad_day[int(t)]:
            know *= 0.75
        return gu + (1 - gu - sl) * know

    def crossed(self, c, t) -> bool:
        return self.p_correct(c, t + 1.0, luck=False) >= CROSS_P

    def _grow(self, c, r):
        """Gradual strength step: W2 takes r of what is left; W3 starts slow."""
        k = self.k[c]
        if self.world == "W3":
            return min(1.0, r * (0.3 + k)) * (1 - k)
        return r * (1 - k)

    def _pre(self, c, t) -> float:
        out = 1.0
        for p in self.g.prereqs[c]:
            out *= 1 - self.tau * (1 - self.own(p, t))
        return out

    def _draw_learn(self, c) -> float:
        if self.n_learn[c] >= MAX_EVENTS:
            raise RuntimeError(f"common random numbers exhausted on {self.g.ids[c]}")
        u = self.u_learn[c, self.n_learn[c]]
        self.n_learn[c] += 1
        return u

    # --- events ---------------------------------------------------------------
    def lesson(self, c, t):
        pre = self._pre(c, t) * self.speed[c]
        u = self._draw_learn(c)
        if self.jump[c]:
            if not self.L[c]:
                if u < min(0.95, 0.6 * pre):
                    self.L[c] = True
                    self.mem[c] = F.review(None, t, F.GOOD, self.w)
            else:
                self.mem[c] = F.implicit(self.mem[c], t, 0.5, self.w)
        else:
            self.k[c] += self._grow(c, min(0.95, 0.6 * pre))
            self.mem[c] = (F.review(None, t, F.GOOD, self.w) if self.mem[c] is None
                           else F.implicit(self.mem[c], t, 0.5, self.w))
        self.lesson_read[c] = True

    def answer(self, c, t) -> bool:
        if self.n_ans[c] >= MAX_EVENTS:
            raise RuntimeError(f"common random numbers exhausted on {self.g.ids[c]}")
        correct = bool(self.u_ans[c, self.n_ans[c]] < self.p_correct(c, t))
        self.n_ans[c] += 1
        self.update(c, t, correct)
        return correct

    def update(self, c, t, correct: bool):
        """The learner's state after answering c (right or wrong) at t."""
        grade = F.GOOD if correct else F.AGAIN
        pre = self._pre(c, t) * self.speed[c]
        u = self._draw_learn(c)
        rate = 0.15 if self.lesson_read[c] else 0.04
        if self.jump[c]:
            if self.L[c]:
                self.mem[c] = F.review(self.mem[c], t, grade, self.w)
            elif u < min(0.95, rate * pre):
                self.L[c] = True
                self.mem[c] = F.review(None, t, F.GOOD, self.w)
        else:
            self.k[c] += self._grow(c, min(0.95, 2.7 * rate * pre))
            self.mem[c] = F.review(self.mem[c], t, grade, self.w)
        if correct:
            for comp, wt in self.g.enc[c]:
                if self.mem[comp] is not None:
                    self.mem[comp] = F.implicit(self.mem[comp], t, wt * FIRE_SCALE, self.w)
