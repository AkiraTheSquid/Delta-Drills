#!/usr/bin/env python3
"""Validation suite for bkt_mastery.py — structural + behavioral + independent.

Run: .venv/bin/python scripts/test_bkt_mastery.py
Exits non-zero on any failed assertion. No pytest dependency.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import bkt_mastery as B  # noqa: E402

P = B.DEFAULT_PARAMS
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


print("A. STRUCTURAL — observe / transit / clamp")
c = B.observe(P.p_init, True)
w = B.observe(P.p_init, False)
check("correct raises posterior above incorrect", c > w, f"correct={c:.3f} incorrect={w:.3f}")
check("posterior monotonic in prior (correct)",
      B.observe(0.2, True) < B.observe(0.6, True) < B.observe(0.9, True))
check("observe stays in [0,1] across extremes",
      all(0.0 <= B.observe(p, b) <= 1.0 for p in (0.0, 0.5, 1.0) for b in (True, False)))
check("repeated correct attempts approach mastery",
      (lambda L: L >= B.MASTERY_THRESHOLD)(
          __import__("functools").reduce(lambda a, _: B.observe(a, True), range(15), P.p_init)))

print("\nB. BEHAVIORAL — FIRe propagation through encompassing graph")
idx = B._encompassing_index()
adv = max(idx, key=lambda k: len(idx[k]))
# higher weight -> more credit
m, t = {}, {}
B.apply_attempt(m, t, adv, True)
children = idx[adv]
hi = max(children, key=lambda x: x[1])
lo = min(children, key=lambda x: x[1])
check("higher-weight edge gets >= credit than lower-weight",
      m[hi[0]] >= m[lo[0]] if hi[1] > lo[1] else True,
      f"w={hi[1]:.2f}->{m[hi[0]]:.3f}  w={lo[1]:.2f}->{m[lo[0]]:.3f}")
# exact hand-computed credit for one child
b_id, w_edge = children[0]
expected = P.p_init + (1 - P.p_init) * (P.p_transit * w_edge)
check("FIRe credit matches hand calc  L0+(1-L0)*pT*w",
      abs(m[b_id] - expected) < 1e-9, f"got={m[b_id]:.6f} exp={expected:.6f}")
# wrong attempt credits no children
m2, t2 = {}, {}
ch2 = B.apply_attempt(m2, t2, adv, False)
check("wrong attempt credits zero encompassed atoms", len(ch2) == 1)
# single-hop only: a credited simpler atom does not itself propagate this turn
non_self = [k for k in m if k != adv]
check("propagation is single-hop (children only, no grandchildren)",
      set(non_self) == {b for b, _ in children})
check("all posteriors bounded in [0,1]", all(0.0 <= v <= 1.0 for v in m.values()))

print("\nC. DECAY / FORGETTING")
m3, t3 = {}, {}
B.apply_attempt(m3, t3, adv, True)
L0 = m3[adv]
future = datetime.now(timezone.utc) + timedelta(days=B.HALF_LIFE_DAYS)
half = B.current_mastery(m3, t3, adv, now=future)
check("one half-life halves the gap to p_init",
      abs(half - (P.p_init + (L0 - P.p_init) * 0.5)) < 1e-6, f"L0={L0:.3f} half={half:.3f}")
far = datetime.now(timezone.utc) + timedelta(days=B.HALF_LIFE_DAYS * 20)
check("decays toward p_init, never below it",
      P.p_init <= B.current_mastery(m3, t3, adv, now=far) < L0)

print("\nD. INDEPENDENT — replay vs from-scratch recompute of a sequence")
# Drive a deterministic sequence on an advanced atom; recompute the direct
# atom's posterior independently (ignoring FIRe, which only affects others)
# and confirm apply_attempt's direct-atom value matches.
seq = [True, False, True, True, False]
m4, t4 = {}, {}
now = datetime.now(timezone.utc)
for correct in seq:
    B.apply_attempt(m4, t4, adv, correct, now=now)
indep = P.p_init
for correct in seq:
    indep = B.observe(indep, correct)  # same now => no decay between steps
check("direct-atom posterior matches independent replay",
      abs(m4[adv] - indep) < 1e-9, f"engine={m4[adv]:.6f} indep={indep:.6f}")

print("\nE. AREA READOUT")
scores = B.area_scores(m4, t4)
check("area_scores returns per-topic means in [0,1]",
      scores and all(0.0 <= v <= 1.0 for v in scores.values()), str(scores))

print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
