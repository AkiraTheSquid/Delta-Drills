#!/usr/bin/env python3
"""FSRS-6 + FIRe memory model (app/memory_model.py), 2026-09-23.

Covers:
  * the spacing effect: the same correct answers spread over days give more
    stability than crammed into an hour;
  * the early-review discount: a correct answer at R ~ 1 barely moves S;
  * implicit credit: weight 0 is a no-op, weight 1 equals a full review,
    anything between never beats a full review and never resets the clock;
  * blame on a miss: a component just practised keeps its stability, a
    faded one lapses;
  * replay: rows for one question tagged with two concepts are ONE answer;
    replay is deterministic; the scoring-path exclusion drops the newest
    answer;
  * due_reviews: nothing learned → nothing due; a learned concept whose recall
    fell below the target is due, and a fresh one is not;
  * remediation.targets puts a due review first after a frontier answer.

Run: .venv/bin/python scripts/test_memory_model.py
"""
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ["USER_DATA_DIR"] = tempfile.mkdtemp(prefix="memory_model_test_")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import memory_model as M  # noqa: E402
from app import kc_graph, remediation  # noqa: E402
from app.adaptive import UserPracticeState  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


T0 = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() / 86400.0


def run(times, g=M.GOOD):
    mem = None
    for t in times:
        mem = M.review(mem, T0 + t, g)
    return mem


print("spacing and the early-review discount")
massed = run([0, 1 / 24, 2 / 24, 3 / 24])
spaced = run([0, 1, 3, 7])
check("spaced reps give more stability than massed", spaced.S > 2 * massed.S,
      f"spaced S={spaced.S:.2f}  massed S={massed.S:.2f}")
m = run([0, 5])
again_now = M.review(m, m.t_last + 1.0 + 1e-9, M.GOOD)  # next day, R still high
later = M.review(m, m.t_last + 40.0, M.GOOD)
check("a review at high R gains far less than one at low R",
      (again_now.S / m.S) < (later.S / m.S),
      f"x{again_now.S / m.S:.2f} vs x{later.S / m.S:.2f}")
check("R is 0.9 at t = S", abs(M.retrievability(m, m.t_last + m.S) - 0.9) < 1e-9)

print("implicit credit")
m = run([0, 3])
t = m.t_last + 6
check("weight 0 is a no-op", M.implicit_review(m, t, 0.0) == m)
full = M.review(m, t, M.GOOD)
one = M.implicit_review(m, t, 1.0)
check("weight 1 equals a full Good review",
      abs(one.S - full.S) < 1e-9 and abs(M.retrievability(one, t) - 1.0) < 1e-9)
half = M.implicit_review(m, t, 0.5)
r0, rh = M.retrievability(m, t), M.retrievability(half, t)
check("weight 0.5 moves R halfway to 1", abs(rh - (r0 + 0.5 * (1 - r0))) < 1e-6,
      f"{r0:.3f} -> {rh:.3f}")
check("fractional credit never beats a full review", m.S <= half.S <= full.S)
check("no clock reset: virtual last review is in the past", half.t_last < t)
check("never practised stays never practised", M.implicit_review(None, t, 1.0) is None)

print("partial lapse keeps R")
lap = M.partial_lapse(m, t, 0.5)
check("S shrinks, R kept", lap.S < m.S and abs(M.retrievability(lap, t) - r0) < 1e-6)

print("replay: FIRe, blame, grouping")
reg = kc_graph._registry()
closure, parents = M._graph()
x, c, w = next((x, c, w) for x, comps in closure.items() for c, w in comps.items() if w >= 0.5)
print(f"  (edge {x} -> {c}, w={w:.2f})")


base = [(T0, {c: M.GOOD}), (T0, {x: M.GOOD})]
# X last seen 20 days ago both times; c either just practised or also stale.
fresh_c = M.replay_events(base + [(T0 + 20, {c: M.GOOD}), (T0 + 20 + 1 / 24, {x: M.AGAIN})])
faded_c = M.replay_events(base + [(T0 + 20, {x: M.AGAIN})])
pre_fresh = M.replay_events(base + [(T0 + 20, {c: M.GOOD})])[c]
pre_faded = M.replay_events(base)[c]
check("a miss on X spares a component just practised",
      fresh_c[c].S >= 0.9 * pre_fresh.S, f"{pre_fresh.S:.3f} -> {fresh_c[c].S:.3f}")
check("a miss on X lapses a faded component", faded_c[c].S < pre_faded.S,
      f"{pre_faded.S:.3f} -> {faded_c[c].S:.3f}")
credited = M.replay_events(base + [(T0 + 10, {x: M.GOOD})])[c]
check("a correct X credits c without a direct answer", credited.S > pre_faded.S)

# Two tags that both encompass one component: ONE implicit repetition, and the
# result does not depend on which tag is visited first.
pair = next(((a, b, k) for a in closure for b in closure if a < b
             for k in set(closure[a]) & set(closure[b]) if k not in (a, b)), None)
if pair:
    a, b, k = pair
    seed = [(T0, {k: M.GOOD}), (T0, {a: M.GOOD}), (T0, {b: M.GOOD})]
    both = M.replay_events(seed + [(T0 + 10, {a: M.GOOD, b: M.GOOD})])[k]
    heavier = a if closure[a][k] >= closure[b][k] else b
    one = M.replay_events(seed + [(T0 + 10, {heavier: M.GOOD})])[k]
    check("two encompassing tags credit a shared component once",
          abs(both.S - one.S) < 1e-9, f"{a}+{b} -> {k}: {both.S:.3f} vs {one.S:.3f}")
    missed = M.replay_events(seed + [(T0 + 10, {a: M.AGAIN, b: M.AGAIN})])[k]
    check("a double miss lapses a shared component at most fully",
          missed.S >= M.replay_events(seed + [(T0 + 10, {k: M.AGAIN})])[k].S - 1e-9)

# Upward lapse walks the whole chain, not only the direct parent.
_c, ancestors = M._graph()
chain = next(((x2, p) for x2, anc in ancestors.items() for p, wp in anc.items()
              if x2 not in reg.get(p, {}).get("encompassing", {})), None)
if chain:
    x2, p = chain
    seed = [(T0, {x2: M.GOOD}), (T0, {p: M.GOOD})]
    before = M.replay_events(seed)[p]
    after = M.replay_events(seed + [(T0 + 1, {x2: M.AGAIN})])[p]
    check("a miss lapses an indirect ancestor", after.S < before.S, f"{x2} -> {p}")

s = UserPracticeState(user_id="memory-model-test")
iso = lambda d: datetime.fromtimestamp((T0 + d) * 86400, timezone.utc).isoformat()
kc_graph.ladder_row(s, x)["attempts"].append({"correct": True, "ts": iso(0), "question_id": 7, "example": False})
kc_graph.ladder_row(s, c)["attempts"].append({"correct": True, "ts": iso(0.5 / 86400), "question_id": 7, "example": False})
evs = M._events(s)
check("two concepts tagged on one question = one answer", len(evs) == 1 and set(evs[0][1]) == {x, c})
check("replay is deterministic", M.replay_events(evs) == M.replay_events(M._events(s)))
kc_graph.ladder_row(s, x)["attempts"].append({"correct": False, "ts": iso(9), "question_id": 8, "example": False})
now = datetime.fromtimestamp((T0 + 9) * 86400, timezone.utc)
with_latest = M.kc_retrievability(s, x, now=now)
without = M.kc_retrievability(s, x, now=now, exclude_latest=True)
check("scoring path excludes the answer being scored",
      with_latest > 0.99 and without < 0.99, f"with={with_latest:.3f} without={without:.3f}")
# c was last touched by x's miss (implicitly): excluding drops THAT answer.
c_with = M.kc_retrievability(s, c, now=now)
c_without = M.kc_retrievability(s, c, now=now, exclude_latest=True)
c_direct_only = M.replay_events(M._events(s)[:-1])[c]
check("exclusion drops the newest answer that touched c implicitly",
      abs(c_without - M.retrievability(c_direct_only, M._now_days(now))) < 1e-9,
      f"with={c_with:.3f} without={c_without:.3f}")
fp0 = M._fingerprint(s)
s.kc_ladder[x]["attempts"][-1]["question_id"] = 9
check("fingerprint sees a changed question id", M._fingerprint(s) != fp0)
s.kc_ladder[x]["attempts"][-1]["question_id"] = 8

print("due_reviews and the picker")
s2 = UserPracticeState(user_id="memory-model-due")
check("no answers, nothing due", M.due_reviews(s2) == [])
learned_kc = kc_graph.frontier(s2)[0]
old = datetime.now(timezone.utc) - timedelta(days=60)
for i in range(3):
    kc_graph.ladder_row(s2, learned_kc)["attempts"].append(
        {"correct": True, "ts": (old + timedelta(minutes=i)).isoformat(), "question_id": 90000 + i, "example": False})
orig = kc_graph.kc_is_learned
try:
    kc_graph.kc_is_learned = lambda st, k: k == learned_kc
    due = M.due_reviews(s2)
    check("a learned concept 60 days stale is due", learned_kc in due, str(due))
    first = next(remediation.targets(s2), None)
    check("the picker serves the due review first", first == learned_kc, str(first))
    s3 = UserPracticeState(user_id="memory-model-fresh")
    kc_graph.ladder_row(s3, learned_kc)["attempts"].append(
        {"correct": True, "ts": datetime.now(timezone.utc).isoformat(), "question_id": 1, "example": False})
    check("a concept answered just now is not due", M.due_reviews(s3) == [])
finally:
    kc_graph.kc_is_learned = orig

print()
print("FAILED: " + ", ".join(fails) if fails else "ALL PASS")
sys.exit(1 if fails else 0)
