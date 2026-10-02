#!/usr/bin/env python3
"""calibrate_ability.py — where app/ability_model.RATE_ANSWER comes from.

The anchor is the ladder's design, not a wanted answer: the rung floors ask
6 Solo + 3 Integrated drills of a concept, so a learner with no level on the
median concept, served at their level, should cross READY in about ANCHOR
problems after the lesson. For each candidate rate this prints the median
(and spread) over every LeetCode and ARENA concept of the model's expected
problems to READY; the rate whose median is nearest ANCHOR is the one to use.

Run: .venv/bin/python scripts/calibrate_ability.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import ability_model as A, bkt_mastery, course_registry, kc_graph  # noqa: E402

ANCHOR = 9


def problems(kc: str, rate: float) -> float:
    old = A.RATE_ANSWER, A.RATE_LESSON
    A.RATE_ANSWER, A.RATE_LESSON = rate, min(1.0, 1.5 * rate)
    try:
        return A.expected_problems(A.prior(bkt_mastery.P_INIT), kc, lesson=True)[1]
    finally:
        A.RATE_ANSWER, A.RATE_LESSON = old


def main():
    kcs = [k for k in kc_graph._registry() if course_registry.course_of(k) in ("leetcode", "arena")]
    best = None
    for rate in (0.3, 0.4, 0.45, 0.5, 0.525, 0.55, 0.6, 0.7, 0.85, 1.0):
        ns = sorted(problems(k, rate) for k in kcs)
        med = statistics.median(ns)
        print(f"rate {rate:.3f}: median {med:5.1f}  p10 {ns[len(ns) // 10]:5.1f}  p90 {ns[9 * len(ns) // 10]:5.1f}")
        if best is None or abs(med - ANCHOR) < abs(best[1] - ANCHOR):
            best = (rate, med)
    print(f"nearest the {ANCHOR}-problem anchor: rate {best[0]} (median {best[1]})")


if __name__ == "__main__":
    main()
