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
  * remediation.targets puts a due review first after a frontier answer;
  * the 2026-09-23 research priors: implicit credit × fire_scale, aided
    correct = a small step with D untouched, a miss scaled by p_skill (the
    engine's pass probability without its memory term), a lapse keeps a
    quarter-ish of S.

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
iso = lambda d: datetime.fromtimestamp((T0 + d) * 86400, timezone.utc).isoformat()


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

print("research-report priors (2026-09-23)")
cfg = M.DEFAULT_CONFIG
seed = [(T0, {c: M.GOOD}), (T0, {x: M.GOOD})]
pre = M.replay_events(seed)
t10 = T0 + 10
got = M.replay_events(seed + [(t10, {x: M.GOOD})])[c]
want = M.implicit_review(pre[c], t10, w * cfg.fire_scale)
check("implicit credit is the registry weight times fire_scale",
      abs(got.S - want.S) < 1e-9 and abs(got.t_last - want.t_last) < 1e-9)
aided = M.replay_events(seed + [(t10, {x: M.AIDED})])
full_x = M.review(pre[x], t10, M.GOOD)
check("aided correct grows S less than a Good", pre[x].S < aided[x].S < full_x.S,
      f"{pre[x].S:.2f} < {aided[x].S:.2f} < {full_x.S:.2f}")
check("aided correct leaves D alone", aided[x].D == pre[x].D)
check("aided correct credits no component", aided[c] == pre[c])
check("aided-only concept has no memory", M.replay_events([(T0, {x: M.AIDED})]) == {})
t20 = T0 + 20
sure = M.replay_events(seed + [(t20, {x: M.AGAIN}, {x: 0.95})])
unlikely = M.replay_events(seed + [(t20, {x: M.AGAIN}, {x: 0.3})])
unstamped = M.replay_events(seed + [(t20, {x: M.AGAIN})])
floor = M.replay_events(seed + [(t20, {x: M.AGAIN}, {x: 0.01})])
at_floor = M.replay_events(seed + [(t20, {x: M.AGAIN}, {x: cfg.lapse_floor})])
check("a miss on a problem expected to fail lapses less",
      unlikely[x].S > sure[x].S, f"p=.3 S={unlikely[x].S:.2f}  p=.95 S={sure[x].S:.2f}")
check("an unstamped miss counts in full", unstamped[x].S <= sure[x].S + 1e-9)
check("p_skill below the floor counts as the floor", abs(floor[x].S - at_floor[x].S) < 1e-9)
check("a discounted miss still lapses", unlikely[x].S < pre[x].S)
m20 = M.review(M.review(None, T0, M.GOOD), T0 + 3, M.GOOD)
while m20.S < 20:
    m20 = M.review(m20, M.due_at(m20), M.GOOD)
lap = M.review(m20, M.due_at(m20), M.AGAIN)
check("a lapse keeps ~20-40% of S (was ~11% on flashcard defaults)",
      0.15 < lap.S / m20.S < 0.5, f"{m20.S:.1f} -> {lap.S:.1f}")

first_hard = M.replay_events([(T0, {x: M.AGAIN}, {x: 0.2})])[x]
first_sure = M.replay_events([(T0, {x: M.AGAIN}, {x: 1.0})])[x]
check("a first miss on an unlikely problem marks D less hard",
      first_hard.D < first_sure.D and first_hard.S == first_sure.S,
      f"D {first_hard.D:.2f} vs {first_sure.D:.2f}")
check("an aided answer above c does not touch c",
      not M._touches({x: M.AIDED}, c, closure, ancestors)
      and M._touches({x: M.GOOD}, c, closure, ancestors)
      and not M._touches({c: M.GOOD}, x, closure, ancestors)
      and M._touches({c: M.AGAIN}, x, closure, ancestors))

print("p_skill stamp (engine_bridge)")
from app import engine_bridge, logistic_engine as E  # noqa: E402
st = UserPracticeState(user_id="memory-model-stamp")
kc_graph.ladder_row(st, x)["attempts"].append({"correct": False, "ts": iso(0), "question_id": 5, "example": False})
pred = E.Prediction(p=0.1, p_mean=0.1, logit_mean=-1.0, logit_var=0.4,
                    contributions={E.RECENCY.name: -1.5, E.ABILITY.name: 0.5})
engine_bridge._stamp_skill_p(st, x, 5, pred)
expect = E.sigmoid(0.5 * E.attenuation(0.4))
check("p_skill = the prediction with the recency term removed",
      abs(st.kc_ladder[x]["attempts"][-1]["p_skill"] - round(expect, 4)) < 1e-9,
      f"{st.kc_ladder[x]['attempts'][-1]['p_skill']} vs full p {pred.p}")
check("replay reads the stamp", M._events(st)[0][2] == {x: round(expect, 4)})
engine_bridge._stamp_skill_p(st, c, 5, pred)
check("stamping a concept with no row creates none", c not in st.kc_ladder)
kc_graph.ladder_row(st, x)["attempts"].append({"correct": False, "ts": iso(1), "question_id": 6, "example": False})
engine_bridge._stamp_skill_p(st, x, 5, pred)
check("only the newest row, and only for its own question",
      "p_skill" not in st.kc_ladder[x]["attempts"][-1])

s = UserPracticeState(user_id="memory-model-test")
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

print("attempt_log answers the ladder lost (2026-09-24)")
from app import attempt_log, bkt_mastery  # noqa: E402
uid = "memory-model-log"
st4 = UserPracticeState(user_id=uid)
row = kc_graph.ladder_row(st4, x)
row["attempts"].append({"correct": True, "ts": iso(10), "question_id": 7, "example": False})


def logrow(d, q, correct, note=None, kind=attempt_log.KIND_ATTEMPT, k=x):
    attempt_log.append(attempt_log.AttemptRow(
        ts=iso(d), kind=kind, user_id=uid, kc=k, question_id=q, correct=correct,
        stage="partial", features={"ability": 1.0}, note=note))


logrow(10 + 1 / 86400, 7, True)          # the same answer as the ladder row
check("a log row matching a ladder row is not a second answer",
      [set(e.grades) for e in M._events(st4)] == [{x}])
logrow(0, 3, True)                        # an answer the ladder no longer holds
logrow(5, 4, None, kind=attempt_log.KIND_TIMEOUT)
logrow(6, 5, True, note="backfill")
evs = M._events(st4)
check("the lost answer is replayed; timeout rows and backfill are not",
      [round(e.t - T0, 3) for e in evs] == [0.0, 10.0], str([round(e.t - T0, 3) for e in evs]))
with_log = M.memories(st4)[x]
check("the older answer adds stability", with_log.S > M.review(None, T0 + 10, M.GOOD).S)

print("kc_mastery forgets by FSRS recall, not a clock")
atoms = [a["a"] for a in (kc_graph._crosswalk().get(x) or {}).get("atoms", []) if a.get("a")]
if atoms:
    st5 = UserPracticeState(user_id="memory-model-kcm")
    for a in atoms:
        st5.atom_mastery[a] = 0.9
        st5.atom_last_ts[a] = iso(0)
    learned, _c, _t = kc_graph.kc_mastery(st5, x, decay=False)
    check("no answered concept: mastery is the learned value, whatever the date",
          kc_graph.kc_mastery(st5, x, now=datetime.now(timezone.utc))[0] == learned)
    kc_graph.ladder_row(st5, x)["attempts"].append(
        {"correct": True, "ts": iso(0), "question_id": 1, "example": False})
    later = datetime.fromtimestamp((T0 + 40) * 86400, timezone.utc)
    r = M.kc_retrievability(st5, x, now=later)
    p0 = bkt_mastery.params_for_level(None).p_init
    got = kc_graph.kc_mastery(st5, x, now=later)[0]
    check("answered concept: pulled toward p_init by 1 - R",
          abs(got - (p0 + (learned - p0) * r)) < 1e-12, f"R={r:.3f} got={got:.3f}")
    check("atom_recall is the best R of the concepts covering it",
          M.atom_recall(st5, atoms[0], now=later) >= r - 1e-12)

print("scoring-path exclusion is scoped to the scored concept (codex 2026-09-24)")
shared = next(((a, ks) for a, ks in M._atom_kcs_index().items() if len(ks) >= 2), None)
if shared:
    atom, (ka, kb) = shared[0], shared[1][:2]
    st6 = UserPracticeState(user_id="memory-model-shared")
    kc_graph.ladder_row(st6, ka)["attempts"].append(
        {"correct": True, "ts": iso(0), "question_id": 11, "example": False})
    kc_graph.ladder_row(st6, kb)["attempts"].append(
        {"correct": True, "ts": iso(5), "question_id": 12, "example": False})
    at = datetime.fromtimestamp((T0 + 9) * 86400, timezone.utc)
    before = M._memories_before_latest(st6, kb)
    want = max((M.retrievability(before[k], T0 + 9) for k in shared[1] if k in before), default=None)
    got = M.atom_recall(st6, atom, now=at, exclude_latest_of=kb)
    check("the other concept's newest answer is kept", want is not None and got == want,
          f"{ka} / {kb}: got={got} want={want}")
    check("the scored concept's own answer is not",
          M.kc_retrievability(st6, kb, now=at, exclude_latest_of=kb) is None
          or kb not in before)

print("log/ladder matching is one-to-one")
uid7 = "memory-model-retry"
st7 = UserPracticeState(user_id=uid7)
kc_graph.ladder_row(st7, x)["attempts"].append(
    {"correct": True, "ts": iso(20 + 30 / 86400), "question_id": 9, "example": False})
for d in (20, 20 + 30 / 86400):   # an answer the ladder dropped, then a quick retry
    attempt_log.append(attempt_log.AttemptRow(
        ts=iso(d), kind=attempt_log.KIND_ATTEMPT, user_id=uid7, kc=x, question_id=9,
        correct=True, stage="partial", features={"ability": 1.0}))
check("a quick retry does not hide the older logged answer", len(M._events(st7)) == 2,
      str(len(M._events(st7))))

print()
print("FAILED: " + ", ".join(fails) if fails else "ALL PASS")
sys.exit(1 if fails else 0)
