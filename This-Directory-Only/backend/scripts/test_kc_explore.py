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
  * narrowed to an area (EXPLORE_PREFIXES), nothing outside it changes;
  * 2026-09-24, whole graph: the cost gate + area prior (a probe must be worth
    its minutes; one python miss does not block math), the `probe` flag on
    recorded attempts, and the RETURN WINDOW (taught concepts re-probed after
    a break, never below their rung).

Run: .venv/bin/python scripts/test_kc_explore.py
"""
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
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


def answer(s, kc, ok, example=False, days_ago=0.0, probe=False):
    kc_graph.ladder_row(s, kc)["attempts"].append(
        # Stamped NOW: kc_evidence weighs each answer by its FSRS retention at
        # wall-clock time, so a fixed date made these checks decay as the
        # calendar moved (3 failed by 2026-09-23 23:00).
        {"correct": ok, "stage": "partial",
         "ts": (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(),
         "question_id": 0, "example": example, "probe": probe})


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

print("narrowed to an area")
saved = X.EXPLORE_PREFIXES
X.EXPLORE_PREFIXES = ("math.",)
try:
    s = state()
    fr = kc_graph.frontier(s, require_questions=False)
    non = [k for k in fr if not X.in_area(k)]
    check("non-math concepts keep their frontier positions",
          [i for i, k in enumerate(fr) if not X.in_area(k)]
          == [i for i, k in enumerate(X.reorder(s, fr)) if not X.in_area(k)] and bool(non))
    answer(s, "torch.tensor-model", True)
    answer(s, "torch.tensor-model", True)
    check("torch concept never settles by belief", not X.settled(s, "torch.tensor-model"))
finally:
    X.EXPLORE_PREFIXES = saved

print("whole graph (2026-09-24)")
check("every concept is in the area", all(X.in_area(k) for k in reg))
s = state()
answer(s, "torch.tensor-model", True)
check("one correct code answer does not settle", not X.settled(s, "torch.tensor-model"))
before = X.beliefs(s)["torch.tensor-model"]
answer(s, "torch.tensor-model", True)
check("a torch answer moves its belief (the memo sees the new answer)",
      X.beliefs(s)["torch.tensor-model"] > before > 0.5)

print("cost gate + area prior")
check("no probes yet: area prior is neutral 0.5",
      abs(X.area_known("torch.x", {}) - 0.5) < 1e-9 and abs(X.area_known("math.x", {}) - 0.5) < 1e-9)
check("cold math concept is worth a probe", X.worth_probing("math.x", 0.0, {}))
check("cold code concept is worth a probe at the break-even", X.worth_probing("torch.x", 0.0, {}) == (X.PROBE_MARGIN <= 0))
check("three torch probe misses close torch probing",
      not X.worth_probing("torch.x", 0.0, {"torch": (3, 0)}))
check("... but not math probing (neutral shrink)", X.worth_probing("math.x", 0.0, {"torch": (3, 0)}))
check("one python miss does not close math", X.worth_probing("math.x", 0.0, {"python": (1, 0)}))
check("torch probe hits raise the area prior", X.area_known("torch.x", {"torch": (3, 3)}) > 0.8)
s = state()
answer(s, "torch.tensor-model", False, probe=True)
answer(s, "torch.constructors", False, probe=True)
answer(s, "torch.dtype-astype", False, probe=True)
answer(s, "torch.sorting", False, probe=False)
check("only probe-flagged answers feed the area prior", X._inputs(s).counts.get("torch") == (3, 0),
      str(X._inputs(s).counts))
check("a concept's own probes are left out of its own prior",
      X.leave_out(X._inputs(s).counts, "torch.tensor-model", X._inputs(s).own.get("torch.tensor-model"))["torch"] == (2, 0))
s = state()
row = {"correct": False, "stage": "partial", "ts": datetime.now(timezone.utc).isoformat(),
       "question_id": 7, "example": False, "probe": True}
kc_graph.ladder_row(s, "torch.tensor-model")["attempts"].append(dict(row))
kc_graph.ladder_row(s, "torch.constructors")["attempts"].append(dict(row))
check("one response tagged with two concepts counts once", X._inputs(s).counts.get("torch") == (1, 0),
      str(X._inputs(s).counts))
check("the memo hits for a learner with probe rows", X._inputs(s) is X._inputs(s))

print("probe flag on the record")
s = state()
qids = kc_graph.questions_for_kc("math.linear-combinations")
if qids:
    qid = qids[0]
    kc = kc_graph.question_kcs(qid)[0]
    was = X.probing(s, kc)
    kc_graph.record_kc_outcome(s, qid, True, stage=kc_graph.DRILL_FLOOR)
    check("attempt carries probe = probing() before the append",
          kc_graph.ladder_view(s, kc)["attempts"][-1].get("probe") is was and was, kc)
else:
    check("a math.linear-combinations question exists", False)

print("return window")
check("no answers: no return", X.return_start([], 100.0) is None)
check("steady practice: no return", not X.in_return_window([90.0, 95.0, 99.0], 100.0))
check("back from a 25-day break 5 days ago: in window", X.in_return_window([70.0, 95.0, 99.0], 100.0))
check("the window closes after RETURN_WINDOW_DAYS", not X.in_return_window([70.0, 90.0], 100.0))
check("away 20 days, not answered yet: in window", X.in_return_window([80.0], 100.0))
from app import memory_model  # noqa: E402
s = state()
real_log = memory_model._log_answers
memory_model._log_answers = lambda st: [("torch.x", datetime.now(timezone.utc).timestamp() / 86400.0 - d, 0, True, False)
                                        for d in (0.5, 10, 20, 30)] if st is s else real_log(st)
answer(s, "math.vector-arithmetic", True, days_ago=40)
check("the untrimmed attempt log fills a break the ladder rows fake", not X.returning(s))
memory_model._log_answers = real_log
s = state()
kc_graph.ladder_row(s, "math.vector-arithmetic")["worked_seen"] = 1
answer(s, "math.vector-arithmetic", True, days_ago=40)
check("taught concept after a break: re-probed", X.returning(s) and X.probing(s, "math.vector-arithmetic"))
check("re-probe shows no example", prioritization.example_plan(s, "math.vector-arithmetic").get("show") is False)
check("re-probe is never the lesson page",
      kc_graph.kc_stage(s, "math.vector-arithmetic") != "worked")
check("return_probes lists it", "math.vector-arithmetic" in X.return_probes(s), str(X.return_probes(s)))
s2 = state()
kc_graph.ladder_row(s2, "math.vector-arithmetic")["worked_seen"] = 1
answer(s2, "math.vector-arithmetic", True, days_ago=0.1)
check("no break: a taught concept is not a probe", not X.returning(s2) and not X.probing(s2, "math.vector-arithmetic"))
check("no break: return_probes is empty", X.return_probes(s2) == [])
s3 = state()
row = kc_graph.ladder_row(s3, "math.vector-arithmetic")
row["worked_seen"] = 1
for _ in range(4):
    answer(s3, "math.vector-arithmetic", True, days_ago=40)
row["attempts"][-1]["stage"] = "solo"
answer(s3, "math.ray-distance", False, days_ago=40)
stage_now = kc_graph.kc_stage(s3, "math.vector-arithmetic")
check("a re-probe never lowers the rung",
      kc_graph.LADDER_STAGES.index(stage_now) >= kc_graph.LADDER_STAGES.index(kc_graph.DRILL_FLOOR), stage_now)
s4 = state()
for k in ("math.vector-arithmetic", "math.linear-combinations"):
    kc_graph.ladder_row(s4, k)["worked_seen"] = 1
answer(s4, "math.vector-arithmetic", True, days_ago=40)
answer(s4, "math.linear-combinations", True, days_ago=40)
answer(s4, "math.vector-arithmetic", False, days_ago=0.1)      # re-probed since coming back
check("re-probed once per return: answered since the return is not stale",
      X.returning(s4) and not X.stale(s4, "math.vector-arithmetic") and X.stale(s4, "math.linear-combinations"))
kc_graph.ladder_row(s4, "math.ray-geometry")["worked_seen"] = 1
check("a concept taught after the return, never answered before it, is not stale",
      not X.stale(s4, "math.ray-geometry") and "math.ray-geometry" not in X.return_probes(s4))
from app import kc_prefs  # noqa: E402
kc_prefs.set_pref(s4, "math.linear-combinations", enabled=False)
check("a concept switched off in Graph Settings is not re-probed",
      "math.linear-combinations" not in X.return_probes(s4), str(X.return_probes(s4)))

print(f"\n{len(fails)} failed" if fails else "\nall passed")
sys.exit(1 if fails else 0)
