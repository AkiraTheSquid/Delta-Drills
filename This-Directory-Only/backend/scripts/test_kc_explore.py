#!/usr/bin/env python3
"""Explore-then-exploit (app/kc_explore.py), 2026-09-22.

Seth: a learner with no math history was started at vector addition and built
up from the roots; the tutor should probe a hard concept first and let a
correct answer there carry the easier ones. Covers:

  * cold start: the first math concept served is a probe of a NON-root, at the
    Solo rung, with no lesson page and no example;
  * settling is the posterior against SETTLE_P — one correct answer settles a
    concept whose harder neighbour was answered, a neutral one needs two;
  * a settled concept counts as learned and unlocks its dependents;
  * a refuted concept falls back to the lesson-first ladder;
  * outside the explore area nothing changes (frontier slots, torch stage).

Run: .venv/bin/python scripts/test_kc_explore.py
"""
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ["USER_DATA_DIR"] = tempfile.mkdtemp(prefix="kc_explore_test_")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import kc_explore as X  # noqa: E402
from app import kc_graph, prioritization  # noqa: E402
from app.adaptive import UserPracticeState  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


def state():
    return UserPracticeState(user_id="kc-explore-test")


def answer(s, kc, ok, example=False):
    kc_graph.ladder_row(s, kc)["attempts"].append(
        # Stamped NOW: kc_evidence weighs each answer by its FSRS retention at
        # wall-clock time, so a fixed date made these checks decay as the
        # calendar moved (3 failed by 2026-09-23 23:00).
        {"correct": ok, "stage": "partial", "ts": datetime.now(timezone.utc).isoformat(),
         "question_id": 0, "example": example})


reg = kc_graph._registry()
area = [k for k in reg if X.in_area(k)]

print("cold start")
s = state()
fr = kc_graph.frontier(s, require_questions=False)
first = next(k for k in fr if X.in_area(k))
check("first math concept is not a root", bool(reg[first]["prereqs"]), first)
check("served as a probe at Solo", kc_graph.kc_stage(s, first) == kc_graph.DRILL_FLOOR)
check("probe shows no example", prioritization.example_plan(s, first).get("show") is False)
check("a root is probed too (no lesson first)",
      kc_graph.kc_stage(s, "math.vector-arithmetic") == kc_graph.DRILL_FLOOR)

s = state()
s.kc_exposure = {first: "2026-09-22T00:00:00+00:00"}
check("a concept whose lesson was read is not a probe", not X.probing(s, first))

print("settling is the posterior")
s = state()
answer(s, "math.linear-combinations", True)      # encompasses vector-arithmetic w=0.9
check("prereq not settled without its own answer", not X.settled(s, "math.vector-arithmetic"))
answer(s, "math.vector-arithmetic", True)
check("one correct settles a concept a harder answer vouched for", X.settled(s, "math.vector-arithmetic"))
s = state()
answer(s, "math.ray-distance", True)             # neutral-ish: ray-distance has no encompassing parent
check("one correct on a neutral concept does not settle", not X.settled(s, "math.ray-distance"))
answer(s, "math.ray-distance", True)
check("two do", X.settled(s, "math.ray-distance"))
s = state()
answer(s, "math.vector-arithmetic", True, example=True)
answer(s, "math.vector-arithmetic", True, example=True)
check("aided answers are not evidence", not X.settled(s, "math.vector-arithmetic"))

print("settled = learned")
s = state()
answer(s, "math.linear-combinations", True)
answer(s, "math.vector-arithmetic", True)
check("settled concept is learned", kc_graph.kc_is_learned(s, "math.vector-arithmetic"))
check("learned concept leaves the frontier",
      "math.vector-arithmetic" not in kc_graph.frontier(s, require_questions=False))

print("refuted → exploit")
s = state()
answer(s, "math.vector-arithmetic", False)
check("refuted concept is not a probe", not X.probing(s, "math.vector-arithmetic"))
check("refuted concept is lesson-first", kc_graph.kc_stage(s, "math.vector-arithmetic") == "worked")
check("dependent of a refuted concept is locked",
      not kc_graph.kc_is_unlocked(s, "math.linear-combinations"))

print("outside the explore area")
s = state()
fr = kc_graph.frontier(s, require_questions=False)
non = [k for k in fr if not X.in_area(k)]
check("non-math concepts keep their frontier positions",
      [i for i, k in enumerate(fr) if not X.in_area(k)]
      == [i for i, k in enumerate(X.reorder(s, fr)) if not X.in_area(k)] and bool(non))
check("torch concept never settles by belief", not X.settled(s, "torch.tensor-model"))

print(f"\n{len(fails)} failed" if fails else "\nall passed")
sys.exit(1 if fails else 0)
