#!/usr/bin/env python
"""test_difficulty_calibration.py — tests for difficulty calibration on new KCs."""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from app.adaptive import UserPracticeState
from app.practice.grading import select_question_for_difficulty
from app.prioritization import (
    demonstrated_learner_baseline,
    target_difficulty,
)
from app.questions import Question

PASS, FAIL = [], []


def check(label, condition, detail=""):
    if condition:
        PASS.append(label)
        print(f"  PASS  {label}" + (f"  — {detail}" if detail else ""))
    else:
        FAIL.append((label, detail))
        print(f"  FAIL  {label}" + (f"  — {detail}" if detail else ""))


print("A. DEMONSTRATED LEARNER BASELINE")
fresh_user = UserPracticeState(user_id="fresh-learner", self_reported_level="beginner")
check("fresh learner has no demonstrated baseline",
      demonstrated_learner_baseline(fresh_user) is None)

struggling_user = UserPracticeState(
    user_id="struggling-learner",
    kc_ladder={
        "kc1": {
            "attempts": [
                {"correct": i < 3, "stage": "faded"} for i in range(10)
            ]
        }
    }
)
check("struggling learner (<70% accuracy) has no elevated baseline",
      demonstrated_learner_baseline(struggling_user) is None)

strong_user = UserPracticeState(
    user_id="strong-learner",
    self_reported_level="beginner",  # Originally seeded as beginner
    kc_ladder={
        "kc1": {
            "attempts": [
                {"correct": True, "stage": "solo"} for _ in range(15)
            ] + [
                {"correct": False, "stage": "solo"} for _ in range(2)
            ]
        }
    }
)
base = demonstrated_learner_baseline(strong_user)
check("strong learner gets elevated baseline",
      base is not None and 0.40 <= base <= 0.50,
      f"baseline={base:.3f}")

print("\nB. TARGET DIFFICULTY ELEVATION ON UNPRACTICED KC")
subtopic = "Numpy: Indexing and selection"
kc = "numpy.slicing-views"

aim_fresh = target_difficulty(fresh_user, subtopic, kc=kc)
check("fresh beginner aim lands near difficulty floor (~20-25)",
      aim_fresh < 30.0,
      f"aim_fresh={aim_fresh:.1f}")

aim_strong = target_difficulty(strong_user, subtopic, kc=kc)
check("experienced learner aim is elevated (>= 45) even with self_reported_level=beginner",
      aim_strong >= 45.0,
      f"aim_strong={aim_strong:.1f}")

print("\nC. PRIMITIVE DRILL FILTERING")
q_trivial = Question(
    id=506,
    topic="Numpy",
    subtopic=subtopic,
    question_text="Return x[:k]",
    answer_code="return x[:k]",
    difficulty_score=10,
    difficulty_label="easy",
    expected_output="",
    starter_code="",
    test_cases=[],
)
q_rich = Question(
    id=189,
    topic="Numpy",
    subtopic=subtopic,
    question_text="Solve 2D tensor indexing",
    answer_code="return z[:k]",
    difficulty_score=50,
    difficulty_label="medium",
    expected_output="",
    starter_code="",
    test_cases=[],
)

candidates = [q_trivial, q_rich]

# For experienced learner (target_diff >= 35), trivial drill (diff <= 15) is bypassed
picked_strong = select_question_for_difficulty(candidates, target_difficulty=55.0, served_ids=set())
check("experienced learner receives rich drill instead of primitive drill",
      picked_strong is not None and picked_strong.id == 189,
      f"picked id={picked_strong.id if picked_strong else None}")

# For novice learner (target_diff < 35), trivial drill remains eligible
picked_novice = select_question_for_difficulty(candidates, target_difficulty=15.0, served_ids=set())
check("novice learner receives primitive drill when target difficulty is low",
      picked_novice is not None and picked_novice.id == 506,
      f"picked id={picked_novice.id if picked_novice else None}")

# Pool with ONLY primitive drills still serves them rather than 404ing
only_primitive = [q_trivial]
picked_only = select_question_for_difficulty(only_primitive, target_difficulty=55.0, served_ids=set())
check("pool with only primitive drills still serves without starving",
      picked_only is not None and picked_only.id == 506,
      f"picked id={picked_only.id if picked_only else None}")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    sys.exit(1)
print("All checks passed.")
