"""ability_model.py — what a learner can do on one concept, measured on the
problems' own difficulty scale. The knowledge model under XP (app/learning_xp.py)
and the planner (app/learning_pace.py).

WHY THIS EXISTS
---------------
Until 2026-10-02 the XP model had two states per concept, learned or not, and
every problem on a concept was the same problem to it. So ten easy solves were
as good as ten hard ones, and a concept's XP had to be priced by hand
(10 + 5·log2(1 + descendants), then × hardness²). Seth, 2026-10-02: "it needs to
be computed rather than hardcoded ... if you solve 10 problems correctly and
they are super easy for you, you are not learning anything. if you solve 10
problems incorrectly and they were super hard for you then you're also not
learning anything. if you solve 50% of problems incorrectly at a difficulty
that's medium, then later get up to 80% or 90%, THEN that counts as xp".

THE STATE
---------
Per concept, one hidden ABILITY θ, in the units of the question bank's
`difficulty_score` (15..100 on both LeetCode and ARENA): a learner at θ solves
a problem of difficulty θ half the time, skill alone. Held as a distribution
over GRID, so the model knows how sure it is.

  skill(θ, d)      = σ((θ − d) / TAU)
  P(correct | θ)   = R·(g + (1 − g − s)·skill(θ, d)) + (1 − R)·g

with the answer's guess/slip (`learning_xp._likelihood`) and FSRS recall R: a
learner who has the skill but cannot retrieve it answers like one who never
had it. KNOWLEDGE of the concept = P(solving a NOVEL problem on it) = the mean
of skill(θ, d) over the concept's question pool, in expectation over θ (the
2026-09-24 planner design's definition). READY is 80% of that pool.

LEARNING
--------
A problem teaches most where the learner is: each learning opportunity moves θ
up by STEP · Binomial(N_SUB, rate · zpd(θ, d)), where

  zpd(θ, d) = 4 · skill(θ, d) · (1 − skill(θ, d))

is 1 for a problem at the learner's level and falls to 0 for one far below
(solved without effort) or far above (out of reach) — Seth's rule, as the
model's transition. The lesson is such an opportunity at the pool's easiest
problem. The XP, the finish date and the problems-a-day all fall out of this.

A MISS TEACHES NOTHING (2026-10-02, the app's BKT rule too: no transit on a
miss). An answered problem is a learning opportunity only when it is solved
(or done behind a worked example, which is restudy). Otherwise a run of
at-level misses would read as practice that lifts θ about as much as the
misses lower it, and a learner failing every problem could never be told the
concept is out of reach. The planner, which cannot know the outcome, learns
in expectation (`paths`): the step's chance scaled by skill(θ, d), the
chance of the solve that carries it.

THE NUMBERS ARE PRIORS
----------------------
Fit them from pooled logs once there are logs to fit; on 2026-10-02 prod held
243 graded answers (mostly one learner) and no answer times (`latency_ms` is
never written; per-problem time stays in the browser, practice/answer-history.js).
  * TAU: 10 difficulty points per logit — a learner at 60 solves a 40 about
    85% of the time and an 80 about 15%.
  * The starting ability by self-reported level is the app's existing prior
    (`bkt_mastery.PRIOR_BY_LEVEL`, P_INIT) read as P(solving a mid-pool
    problem): beginner 0.02 → θ 11, no level 0.10 → θ 28, strong 0.45 → θ 48.
  * RATE_ANSWER: anchored on the ladder's own design, not on a target. The
    rung floors ask 6 Solo + 3 Integrated drills of a concept, so a learner
    with no level, on a concept with the median pool, served at their level,
    should cross READY in about 9 problems after the lesson
    (scripts/calibrate_ability.py prints the fit). Lesson / answer / aided
    ratios keep the v0 T_LESSON : T_ANSWER : T_AIDED = 1.5 : 1 : 1.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np

GRID = np.arange(-20.0, 140.0 + 1e-9, 2.5)
M = len(GRID)
TAU = 10.0
STEP = 2.5
N_SUB = 8
READY = 0.80
DIFFICULTY_MID = 50.0
PRIOR_SD = 20.0

RATE_ANSWER = 0.525
RATE_LESSON = min(1.0, 1.5 * RATE_ANSWER)
RATE_AIDED = RATE_ANSWER
# An explore probe has no lesson and no example in front of it: nearly pure
# measurement (the v0 T_PROBE 0.03 against T_ANSWER 0.10).
RATE_PROBE = 0.3 * 0.1 * RATE_ANSWER


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def skill(d: float) -> np.ndarray:
    """skill(θ, d) at every grid point."""
    return _sig((GRID - d) / TAU)


# --- a concept's question pool -----------------------------------------------

@lru_cache(maxsize=None)
def pool(kc: str) -> Tuple[float, ...]:
    """Difficulty scores of the concept's questions; a concept the bank does
    not know is one mid-difficulty problem."""
    from app import kc_graph, questions  # heavy; only on use
    questions.ensure_questions_loaded()
    ds = [float(q.difficulty_score) for qid in kc_graph.questions_for_kc(kc)
          if (q := questions.get_question_by_id(qid)) is not None and q.difficulty_score is not None]
    return tuple(sorted(ds)) or (DIFFICULTY_MID,)


def difficulty(qid: Optional[int], kc: str) -> float:
    """One question's difficulty; unknown → the pool's median."""
    if qid is not None:
        from app import questions
        questions.ensure_questions_loaded()
        q = questions.get_question_by_id(qid)
        if q is not None and q.difficulty_score is not None:
            return float(q.difficulty_score)
    ds = pool(kc)
    return ds[len(ds) // 2]


@lru_cache(maxsize=None)
def _curve(ds: Tuple[float, ...]) -> np.ndarray:
    return np.mean([skill(d) for d in ds], axis=0)


def curve(kc: str) -> np.ndarray:
    """P(solving a novel problem on `kc`) at every grid point."""
    return _curve(pool(kc))


def knowledge(post: np.ndarray, kc: str) -> float:
    return float(post @ curve(kc))


@lru_cache(maxsize=None)
def _ready_theta(ds: Tuple[float, ...]) -> float:
    lo, hi = -50.0, 250.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.mean([1.0 / (1.0 + math.exp(-(mid - d) / TAU)) for d in ds]) < READY:
            lo = mid
        else:
            hi = mid
    return hi


def ready_theta(kc: str) -> float:
    """The ability at which the concept is READY: 80% of its pool solved."""
    return _ready_theta(pool(kc))


# --- prior -------------------------------------------------------------------

def prior_mean(p_init: float) -> float:
    """The ability that solves a mid-pool problem with probability p_init."""
    p = min(max(p_init, 1e-6), 1 - 1e-6)
    return DIFFICULTY_MID + TAU * math.log(p / (1 - p))


@lru_cache(maxsize=16)
def _prior(mean: float) -> np.ndarray:
    w = np.exp(-0.5 * ((GRID - mean) / PRIOR_SD) ** 2)
    return w / w.sum()


def prior(p_init: float) -> np.ndarray:
    return _prior(round(prior_mean(p_init), 3))


# --- learning ----------------------------------------------------------------

def zpd(d: float) -> np.ndarray:
    s = skill(d)
    return 4.0 * s * (1.0 - s)


def _jumps(z: np.ndarray) -> np.ndarray:
    """M×M: row i moves up k grid steps w.p. Binomial(N_SUB, z_i)(k).
    Jumps past the top of the grid stay at the top."""
    A = np.zeros((M, M))
    for k in range(N_SUB + 1):
        pk = math.comb(N_SUB, k) * z ** k * (1 - z) ** (N_SUB - k)
        for i in range(M):
            A[i, min(M - 1, i + k)] += pk[i]
    return A


@lru_cache(maxsize=4096)
def transition(d: float, rate: float) -> np.ndarray:
    """M×M: row i = where θ_i goes after one opportunity on a problem of
    difficulty d."""
    if rate <= 0:
        return np.eye(M)
    return _jumps(np.clip(rate * zpd(d), 0.0, 1.0))


def lesson_difficulty(kc: str) -> float:
    return pool(kc)[0]


# --- evidence ----------------------------------------------------------------

def emission(d: float, correct: Optional[bool], R: float, g: float, s: float) -> np.ndarray:
    if correct is None:
        return np.ones(M)
    p = R * (g + (1.0 - g - s) * skill(d)) + (1.0 - R) * g
    return p if correct else 1.0 - p


# One observation: (pre, d, correct|None, R, g, s, post) where pre/post are
# (difficulty, rate) learning steps around the answer — the lesson before the
# first exploit answer, practice after each one — or None.
Obs = Tuple[Optional[Tuple[float, float]], float, Optional[bool], float, float, float,
            Optional[Tuple[float, float]]]


def _step(v: np.ndarray, st) -> np.ndarray:
    return v if st is None else v @ transition(*st)


def _back(v: np.ndarray, st) -> np.ndarray:
    return v if st is None else transition(*st) @ v


def smooth(start: np.ndarray, obs: Sequence[Obs]) -> List[np.ndarray]:
    """Posterior over θ at every slot 0..n given ALL of `obs` (forward-
    backward). Slot j = the state after observation j and its practice step."""
    fwd = [start]
    emits = []
    for pre, d, correct, R, g, s, post in obs:
        v = _step(fwd[-1], pre)
        e = emission(d, correct, R, g, s)
        emits.append(e)
        v = v * e
        z = v.sum()
        v = v / z if z > 0 else _step(fwd[-1], pre)
        fwd.append(_step(v, post))
    n = len(obs)
    bwd = [None] * (n + 1)
    bwd[n] = np.ones(M)
    for j in range(n, 0, -1):
        pre, _d, _c, _R, _g, _s, post = obs[j - 1]
        x = _back(bwd[j], post) * emits[j - 1]
        y = _back(x, pre)
        z = y.sum()
        bwd[j - 1] = y / z if z > 0 else np.ones(M)
    out = []
    for f, b in zip(fwd, bwd):
        v = f * b
        z = v.sum()
        out.append(v / z if z > 0 else f)
    return out


def at_level(post: np.ndarray, kc: str) -> float:
    """The pool problem a well-aimed picker serves next: the one nearest the
    learner's expected ability (the most learning per problem, see zpd)."""
    mean = float(post @ GRID)
    return min(pool(kc), key=lambda d: abs(d - mean))


CAP = 100  # problems; a learner not at READY by then is out of reach


@lru_cache(maxsize=1024)
def _paths(kc: str, lesson: bool, rate: float, lesson_rate: float):
    """Per starting ability θ_i: (problems to READY, M×CAP difficulties served
    — NaN once done —, reaches READY within CAP). Each θ_i is ONE learner
    walking its expected course: the lesson, then each step the pool problem
    nearest where it is (the app's evidence keeps finding it), θ up by the
    expected learning step on a solve, STEP · N_SUB · rate · zpd · skill.
    (A distribution per θ_i would strand its unlucky mass below the problems
    served to its mean, and read the learner as never getting there.)"""
    ds = np.asarray(pool(kc))
    D = np.full((M, CAP), np.nan)
    n = np.zeros(M)
    ok = np.zeros(M, dtype=bool)

    def known(th):
        return float(np.mean(_sig((th - ds) / TAU)))

    for i, th in enumerate(GRID):
        th = float(th)
        if lesson:
            sk = float(_sig((th - lesson_difficulty(kc)) / TAU))
            th += STEP * N_SUB * min(1.0, lesson_rate * 4.0 * sk * (1.0 - sk))
        for t in range(CAP):
            if known(th) >= READY:
                ok[i] = True
                break
            d = float(ds[np.abs(ds - th).argmin()])
            sk = float(_sig((th - d) / TAU))
            th += STEP * N_SUB * min(1.0, rate * 4.0 * sk * (1.0 - sk) * sk)
            D[i, t] = d
            n[i] += 1
        else:
            ok[i] = known(th) >= READY
    return n, D, ok


def paths(kc: str, lesson: bool):
    """`_paths` at the model's current rates."""
    return _paths(kc, lesson, RATE_ANSWER, RATE_LESSON)


def expected_problems(post: np.ndarray, kc: str, lesson: bool) -> Tuple[float, float]:
    """(P(the learner reaches READY within CAP), expected problems to READY
    for a learner who does) — the per-θ counts of `paths` averaged over
    `post`. Averaging COUNTS, not the distribution: a mean-field course over a
    wide prior waits for its slowest tail, and read a beginner's ARENA concept
    at 2× a simulated route (34.5 vs 14.9 problems, 2026-10-02). Checked
    against that simulation (true θ from the prior, served at the POSTERIOR's
    level, answers drawn, stop at the model's READY; 12 concepts a course):
    0–15% short at every level, about one problem — each θ here is served at
    its level from the first problem, the app takes a problem or two to find it."""
    n, _D, ok = paths(kc, lesson)
    p = float(post[ok].sum())
    return p, (float(post[ok] @ n[ok]) / p if p > 0 else float(CAP))
