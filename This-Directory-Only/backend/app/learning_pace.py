"""learning_pace.py — how many PROBLEMS, and roughly how many minutes, the rest
of a course takes, concept by concept.

WHY THIS EXISTS
---------------
The Learner Home's target is XP a day, and the concept list prices one
solved problem (`learning_xp.solve_xp`). A LeetCode learner, 2026-10-01, set a
finish date one month out, read "20 XP a day" next to "+15 XP / problem", and
concluded two problems a day would finish the course. XP per problem varies
with the learner's state, so no reader can turn XP into problems. The model
can. Seth, 2026-10-02: do the math "on a per node basis for leetcode, then
compare it to the ARENA, keeping in mind that the time per problem and
difficulty is different" — and check the gut AFTER, never solve back from it.

THE COUNT, PER CONCEPT
----------------------
From the learner's ability distribution on the concept now (replay's
"state_now"): for every ability θ it might be, the model's own course
(`ability_model.paths`) — the lesson if it was never read, then one problem a
step at that learner's level, learning only on a solve, until 80% of the
concept's pool is expected solved — and the counts averaged over the
distribution (`ability_model.expected_problems`). A concept known but faded
(skill at READY, recall below) takes one problem: the retrieval. A concept
whose pool has no problem near the learner never gets there (OUT OF REACH):
it is counted apart, not as a fuse's worth of problems. Slightly short: each
θ is served at its level from the first problem, while the app takes a
problem or two to find it (vs a simulated route: 0–15% short, every level).

MINUTES. Each counted problem is priced at the minutes a problem of its
course and difficulty takes (MINUTES_BY_TIER). These are ASSUMPTIONS, not
measurements: the server records no answer times (per-problem time stays in
the browser, practice/answer-history.js). LeetCode's tiers follow the usual
interview-practice clock (easy 15, medium 25, hard 40); an ARENA drill is a
short function (the explore simulation's 4 minutes at mid difficulty).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from app import ability_model as A

# (upper difficulty bound, minutes) per course, first match wins.
MINUTES_BY_TIER: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "leetcode": ((35, 15.0), (70, 25.0), (1e9, 40.0)),
    "arena": ((35, 3.0), (70, 4.0), (1e9, 6.0)),
}
MINUTES_DEFAULT = MINUTES_BY_TIER["arena"]


def _tiers(kc: str):
    from app import course_registry
    return MINUTES_BY_TIER.get(course_registry.course_of(kc) or "", MINUTES_DEFAULT)


def minutes(kc: str, d: float) -> float:
    return next(m for bound, m in _tiers(kc) if d <= bound)


def _row_minutes(kc: str, D: np.ndarray) -> np.ndarray:
    """Minutes of each row's served problems (NaN = none)."""
    out = np.zeros(D.shape)
    for bound, m in reversed(_tiers(kc)):
        out = np.where(D <= bound, m, out)
    return out.sum(axis=1)


def course(kc: str, state: dict) -> Tuple[float, float, bool]:
    """(expected problems, expected minutes, reaches READY) for `kc` from
    `state` — one entry of replay's "state_now" / "also_state". Not reaching
    it = more likely than not, the concept's pool has no problem near the
    learner (OUT OF REACH, see ability_model.CAP): a content gap, which no
    count of problems would close; counted as 0 problems."""
    post, R = state["post"], float(state["R"])
    skill = A.knowledge(post, kc)
    if skill * R >= A.READY:
        return 0.0, 0.0, True
    if skill >= A.READY:  # known, faded: one retrieval brings it back
        return 1.0, minutes(kc, A.at_level(post, kc)), True
    lesson = not state.get("lesson_seen")
    p, n = A.expected_problems(post, kc, lesson)
    if p < 0.5:
        return 0.0, 0.0, False
    _n, D, ok = A.paths(kc, lesson)
    w = post * ok / p
    return n, float(w @ _row_minutes(kc, D)), True


def problems_to_ready(kc: str, state: dict) -> float:
    return course(kc, state)[0]


def minutes_to_ready(kc: str, state: dict) -> float:
    return course(kc, state)[1]


def per_node(state_now: Dict[str, dict]) -> Dict[str, Tuple[float, float, bool]]:
    """{kc: (problems, minutes, reaches READY)}, every concept."""
    return {kc: course(kc, st) for kc, st in state_now.items()}


def remaining(state_now: Dict[str, dict]) -> Tuple[float, float, List[str]]:
    """(problems, minutes) the rest of the course takes over the concepts in
    reach, and the concepts out of reach (counted in neither)."""
    nodes = per_node(state_now)
    ok = [(n, m) for n, m, reached in nodes.values() if reached]
    away = [kc for kc, (_n, _m, reached) in nodes.items() if not reached]
    return float(sum(n for n, _m in ok)), float(sum(m for _n, m in ok)), away


def problems_remaining(state_now: Dict[str, dict]) -> float:
    """The course's count: every in-reach concept's problems to READY."""
    return remaining(state_now)[0]
