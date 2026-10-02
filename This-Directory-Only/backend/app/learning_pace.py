"""learning_pace.py — how many PROBLEMS the rest of a course takes.

WHY THIS EXISTS
---------------
The Learner Home's target is XP a day, and the concept list prices one
solved problem (`learning_xp.solve_xp`). A LeetCode learner, 2026-10-01, set
a finish date one month out, read "20 XP a day" next to "+15 XP / problem",
and concluded two problems a day would finish the course. They will not:
+15 is the FIRST solve on a 30-XP concept (solves pay 15.7 → 9.7 → 0.8 as
the concept nears READY) and a leaf pays 5, so XP per problem varies
threefold and no reader can turn XP into problems. The model can: this
module asks it.

THE COUNT
---------
For each concept, from where the learner's knowledge stands now
(`replay`'s "state_now"), simulate the model's own story of a learner:
the true state is learned with the model's belief, the lesson and each
answer teach with the concept's learning rates (`learning_xp.rates`, slower
on a hard concept), a learned concept is answered right unless slipped, an
unlearned one right only by the exploit guess. The belief the app keeps is
updated from those answers exactly as `solve_xp` does, and the run ends when
the belief reaches READY. Problems = answers to get there, missed ones
included. The mean over SIMS runs is the concept's count; the course's is
the sum. Same model that pays the XP, so the two can be converted: the home
divides problems by remaining XP and shows "≈ N problems a day" beside the
XP target.

Seeded by concept and state, so a read is deterministic.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache
from typing import Dict

from app import learning_xp as lx

SIMS = 500
# A run that never gets there (it should not; the learning rates are
# positive) stops here. Far above the ~10 a novice takes, so the cut-off
# hardly biases the mean (codex: at 40 it undercounted).
MAX_PROBLEMS = 100


@lru_cache(maxsize=4096)
def _count(kc: str, p: float, R: float, lesson_seen: bool) -> float:
    if p * R >= lx.READY:
        return 0.0
    g, s = lx._likelihood(kc, None, False)
    t_lesson, t_answer, _ = lx.rates(kc)
    rng = random.Random(f"{kc}|{p}|{R}|{lesson_seen}")
    total = 0
    for _ in range(SIMS):
        belief, recall = p, R
        learned = rng.random() < p
        if not lesson_seen:
            belief += (1.0 - belief) * t_lesson
            learned = learned or rng.random() < t_lesson
        n = 0
        while belief * recall < lx.READY and n < MAX_PROBLEMS:
            n += 1
            p_right_learned = (1.0 - s) * recall + g * (1.0 - recall)
            right = rng.random() < (p_right_learned if learned else g)
            if right:
                belief = belief * p_right_learned / (belief * p_right_learned + (1.0 - belief) * g)
            else:
                miss = 1.0 - p_right_learned
                belief = belief * miss / (belief * miss + (1.0 - belief) * (1.0 - g))
            # An answer is a retrieval: recall is back to 1 whatever it was.
            recall = 1.0
            belief += (1.0 - belief) * t_answer
            learned = learned or rng.random() < t_answer
        total += n
    return total / SIMS


def problems_to_ready(kc: str, state: dict) -> float:
    """Expected problems until `kc` reaches READY from `state` (one entry of
    replay's "state_now" / "also_state"). The state keys the cache to two
    decimals, rounded DOWN: rounding to nearest turned p 0.796 into 0.80 and
    an unfinished concept into zero problems (codex)."""
    key = lambda v: math.floor(float(v) * 100) / 100
    return _count(kc, key(state["p"]), key(state["R"]), bool(state.get("lesson_seen")))


def problems_remaining(state_now: Dict[str, dict]) -> float:
    """The course's count: every concept's problems to READY."""
    return sum(problems_to_ready(kc, st) for kc, st in state_now.items())
