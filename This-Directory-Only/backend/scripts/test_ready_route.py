#!/usr/bin/env python3
"""Practice until ready (app/ready_route.py + app/kc_evidence.py), 2026-09-23.

Seth: start on the exercise's own concept; if the learner is not ready, route
backward, probing for the prerequisite weakness, drill whichever weakness is
found, and stop when the target is ready — not after a set count. Covers:

  * the first question is an attempt at the exercise (one of its variants);
  * two clean attempts make the target ready and the route stops;
  * a missed attempt routes backward: the next question probes a DIRECT
    prerequisite, not the target again;
  * a simulated learner missing exactly one prerequisite is routed down to
    that concept, drills it, and ends ready — every other prerequisite is
    probed at most twice;
  * evidence weights: an ARENA variant outweighs a single-move drill, and an
    answer from two months ago weighs less than one from today.

Run: .venv/bin/python scripts/test_ready_route.py
"""
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ["USER_DATA_DIR"] = tempfile.mkdtemp(prefix="ready_route_test_")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import kc_evidence, kc_graph, ready_route  # noqa: E402
from app.adaptive import UserPracticeState  # noqa: E402

TARGET = "raytracing.make-rays-1d"
EXERCISE = [831, 842, 843, 844, 845, 846]

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


def state():
    return UserPracticeState(user_id="ready-route-test")


def answer(st, step, correct):
    kc_graph.record_kc_outcome(st, step["question_id"], correct)


def run(st, knows, limit=40):
    """Drive a route with a learner who answers `knows(kc)`; return the steps."""
    served, steps = [], []
    for _ in range(limit):
        step = ready_route.plan(st, TARGET, EXERCISE, served)
        if step["done"]:
            steps.append(step)
            return steps
        steps.append(step)
        served.append(step["question_id"])
        answer(st, step, knows(step["kc"]))
    return steps


print("first question")
st = state()
first = ready_route.plan(st, TARGET, EXERCISE, [])
check("opens on an attempt at the exercise", first.get("attempt") and first["question_id"] in EXERCISE,
      f"{first.get('mode')} q{first.get('question_id')} on {first.get('kc')}")

print("ready by doing the problem")
st = state()
steps = run(st, lambda kc: True)
check("two correct attempts, then ready", [s.get("mode") for s in steps] == ["attempt", "attempt", None]
      and steps[-1]["reason"] == "ready", str([(s.get("mode"), s.get("kc")) for s in steps]))

print("a miss routes backward")
st = state()
s1 = ready_route.plan(st, TARGET, EXERCISE, [])
answer(st, s1, False)
s2 = ready_route.plan(st, TARGET, EXERCISE, [s1["question_id"]])
direct = kc_graph.registry_node(TARGET)["prereqs"]
check("second question probes a direct prerequisite", s2.get("mode") == "probe" and s2.get("kc") in direct,
      f"{s2.get('mode')} {s2.get('kc')}")

print("finds the one weakness")
for weak in ("torch.slice-assignment", "torch.out-argument"):
    st = state()
    # Knows everything except `weak`, which two questions on it teach; the
    # exercise is solved once `weak` is.
    seen = {"n": 0}

    def knows(kc, w=weak, seen=seen):
        if kc == w:
            seen["n"] += 1
            return seen["n"] > 2
        return kc != TARGET or seen["n"] > 2

    steps = run(st, knows)
    modes = [(s.get("mode"), s.get("kc")) for s in steps]
    drilled = [kc for m, kc in modes if m == "drill"]
    probes = {}
    for m, kc in modes:
        if m == "probe":
            probes[kc] = probes.get(kc, 0) + 1
    check(f"{weak}: drilled, and only it", drilled and set(drilled) == {weak}, str(modes))
    check(f"{weak}: route ends ready", steps[-1].get("reason") == "ready", str(steps[-1].get("reason")))
    check(f"{weak}: no concept probed more than twice", all(n <= 2 for n in probes.values()), str(probes))

print("a concept with no drills does not hide the ones under it")
real_pools = ready_route.pools
ready_route.pools = lambda kc, ex, arena: ({"arena": [], "integrated": [], "drill": []}
                                          if kc == "torch.out-argument" else real_pools(kc, ex, arena))
try:
    st = state()
    s1 = ready_route.plan(st, TARGET, EXERCISE, [])
    answer(st, s1, False)
    served = [s1["question_id"]]
    reached = set()
    for _ in range(12):
        step = ready_route.plan(st, TARGET, EXERCISE, served)
        if step["done"]:
            break
        reached.add(step["kc"])
        served.append(step["question_id"])
        answer(st, step, step["kc"] != "torch.slicing-views")
    check("walks through the empty concept to its prerequisites",
          "torch.slicing-views" in reached and "torch.out-argument" not in reached, str(sorted(reached)))
finally:
    ready_route.pools = real_pools

print("evidence weights")
arena = kc_evidence.arena_ids()
check("ARENA variant weighs 1", kc_evidence.similarity(TARGET, 842, arena) == kc_evidence.W_ARENA)
drill_q = next(q for q in kc_graph.questions_for_kc("torch.ranges") if kc_graph.ladder_rank(q) == 2)
check("single-move drill weighs less", kc_evidence.similarity("torch.ranges", drill_q, arena) == kc_evidence.W_DRILL)
check("math MC unchanged", kc_evidence.similarity("math.vector-addition", 50000, arena) == 1.0)

st = state()
kc_graph.record_kc_outcome(st, drill_q, True)
row = kc_graph.ladder_view(st, "torch.ranges")["attempts"][-1]
w_now = kc_evidence.weigher(st)("torch.ranges", row)
old = dict(row, ts=(datetime.now(timezone.utc) - timedelta(days=60)).isoformat())
w_old = kc_evidence.weigher(st)("torch.ranges", old)
try:
    from app import memory_model  # noqa: F401
    check("an answer from 60 days ago weighs less than today's", w_old < w_now, f"{w_old:.3f} < {w_now:.3f}")
except Exception:
    print("  SKIP  decay (app/memory_model.py not present)")

print()
print("all passed" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
