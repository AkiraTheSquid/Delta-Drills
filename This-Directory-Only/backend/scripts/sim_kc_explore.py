#!/usr/bin/env python3
"""Calibration table + simulated diagnostics for app/kc_explore.py.

Two outputs:

  1. ANSWERS TO SETTLE: how many consecutive correct unaided answers a concept
     needs to clear SETTLE_P, by the indirect prior its neighbours built.
  2. SIMULATED LEARNERS through the real `kc_graph.frontier` / `kc_stage`
     (math concepts only): which concept is probed first, how many problems
     each concept costs, and the total before every concept is settled
     (known) or refuted (handed to the lesson-first ladder).

Learners answer deterministically (a known concept is answered right, an
unknown one wrong) unless --noisy, which draws answers from P_GUESS / P_SLIP
and averages over seeds.

Run: .venv/bin/python scripts/sim_kc_explore.py [--noisy]
"""
import os
import random
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("USER_DATA_DIR", tempfile.mkdtemp(prefix="kc_explore_sim_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import kc_explore as X  # noqa: E402
from app import kc_graph  # noqa: E402
from app.adaptive import UserPracticeState  # noqa: E402

NOW = "2026-09-22T00:00:00+00:00"


def settle_table():
    print("ANSWERS TO SETTLE (consecutive correct, unaided)")
    print(f"  bf(correct)={X.LOG_BF_CORRECT:+.2f}  bf(miss)={X.LOG_BF_INCORRECT:+.2f}  "
          f"settle at logodds {X.SETTLE_LOGODDS:+.2f} (P={X.SETTLE_P})")
    for label, prior in [("strong neighbours (clip)", X.INDIRECT_CLIP), ("one encompassing hop w=0.8", X.LOG_BF_CORRECT * 0.8),
                         ("neutral", 0.0), ("self-report beginner", -0.5),
                         ("prereq missed, 1 hop", X.LOG_BF_INCORRECT * X.INCORRECT_DOWN_SCALE * X.HOP_ATTENUATION_DOWN),
                         ("after one own miss", X.LOG_BF_INCORRECT)]:
        x, n = prior, 0
        while x < X.SETTLE_LOGODDS:
            x += X.LOG_BF_CORRECT
            n += 1
        print(f"  {label:<28} prior {prior:+.2f}  -> {n}")


def fresh_state():
    s = UserPracticeState(user_id="sim")
    s.kc_ladder = {}
    return s


def run(known, rng=None, cap=80):
    s = fresh_state()
    area = [k for k in kc_graph._registry() if X.in_area(k)]
    log = []
    for _ in range(cap):
        todo = [k for k in area if not X.settled(s, k)
                and X.beliefs(s)[k] > X.OUT_OF_STATE]
        if not todo:
            break
        fr = [k for k in kc_graph.frontier(s, require_questions=False) if X.in_area(k)]
        if not fr:
            break
        kc = fr[0]
        if kc_graph.kc_stage(s, kc) == "worked":
            # Refuted: the ladder teaches it. The diagnostic is over for this
            # concept; model the lesson being read, then keep counting only
            # probes of concepts still open.
            break
        if rng is None:
            ok = kc in known
        else:
            ok = rng.random() < ((1 - X.P_SLIP) if kc in known else X.P_GUESS)
        kc_graph.ladder_row(s, kc)["attempts"].append(
            {"correct": ok, "stage": "partial", "ts": NOW, "question_id": 0, "example": False})
        log.append((kc, ok))
    P = X.beliefs(s)
    return log, P


LEARNERS = {
    "knows all": None,
    "knows nothing": set(),
    "knows up to matrix-equations": {
        "math.vector-arithmetic", "math.dot-products-norms", "math.linear-combinations",
        "math.matrix-equations", "math.ray-geometry"},
    "knows all but determinants": "all-but",
}


def main():
    settle_table()
    area = [k for k in kc_graph._registry() if X.in_area(k)]
    noisy = "--noisy" in sys.argv
    for name, known in LEARNERS.items():
        if known is None:
            known = set(area)
        elif known == "all-but":
            known = set(area) - {"math.determinants-invertibility", "math.singular-systems"}
        print(f"\n{name}")
        if noisy:
            totals = []
            false_settle = false_refute = 0
            for seed in range(200):
                log, P = run(known, random.Random(seed))
                totals.append(len(log))
                false_settle += sum(1 for k in area if P[k] >= X.SETTLE_P and k not in known)
                false_refute += sum(1 for k in area if P[k] <= X.OUT_OF_STATE and k in known)
            totals.sort()
            print(f"  problems: median {totals[len(totals)//2]}, p90 {totals[int(len(totals)*.9)]}; "
                  f"per run: settled-but-unknown {false_settle/200:.2f}, "
                  f"refuted-but-known {false_refute/200:.2f}")
            continue
        log, P = run(known)
        print("  order:", " → ".join(f"{k[5:]}{'✓' if ok else '✗'}" for k, ok in log))
        print(f"  problems: {len(log)}")
        for k in area:
            st = "settled" if P[k] >= X.SETTLE_P else ("refuted" if P[k] <= X.OUT_OF_STATE else "open")
            n = sum(1 for kk, _ in log if kk == k)
            print(f"    {k:<34} P={P[k]:.2f} {st:<8} asked {n}")


if __name__ == "__main__":
    main()
