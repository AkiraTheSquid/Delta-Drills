#!/usr/bin/env python3
"""Grid runner: FSRS-Bayes vs BKT-Bayes explore/exploit.

  pilot  — seeds 0..n-1, no break, every arm at every (gate theta, review
           threshold) pair. Picks ONE pair per arm (lowest mean of the two
           worlds' medians) into out/pilot_best.json. Gate and review are
           separate knobs (Seth 09-26): unlock the next concept at one level,
           re-review this one when recall falls below another.
  final  — FRESH seeds 1000.., every arm at its pilot pair, breaks
           0 / 30 / 180 days. Tuning and scoring never share a learner, so
           picking each arm's best theta cannot flatter it.
  report — paired comparison from out/final.jsonl: per learner, the arm's
           summed crossing hours vs the reference arm's on the SAME learner;
           median % difference with a bootstrap 95% interval.

Run from This-Directory-Only/backend:
  .venv/bin/python -m sims.run pilot  --n 20
  .venv/bin/python -m sims.run final  --n 100
  .venv/bin/python -m sims.run report
CPU: 11 workers at nice 19 (Seth: keep the machine usable).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sims import sim, world  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
WORLDS = ("W1", "W2")
# Gate grids differ by what the gate reads: P(learned) for C/B/H (the last
# pilot's optimum sat at 0.95-0.98), E[R a day out] for pure FSRS (a first Good
# review alone gives 0.93, so the grid runs up to where it can still bind).
P_THETAS = (0.8, 0.9, 0.95, 0.98, 0.99)
R_THETAS = (0.5, 0.7, 0.8, 0.9, 0.95)
THETAS = {"C": P_THETAS, "B": P_THETAS, "H": P_THETAS, "A0": R_THETAS, "A": R_THETAS,
          "O": (0.9,)}
REVIEWS = (0.6, 0.7, 0.8, 0.9, 0.95)
CONTENDERS = ("C", "B", "A0", "A", "H")
WORKERS = 11
_G = None


def _init():
    global _G
    os.nice(19)
    _G = world.load_graph()


def _one(task):
    w, lt, seed, arm, theta, rev, brk = task
    r = sim.run(_G, w, lt, seed, arm, theta, brk, review_at=rev)
    return {"world": w, "ltype": lt, "seed": seed, "arm": arm, "theta": theta,
            "review": rev, "break": brk, **asdict(r)}


def _grid(tasks, path: Path):
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    with Pool(WORKERS, initializer=_init) as pool, path.open("w") as f:
        for i, row in enumerate(pool.imap_unordered(_one, tasks, chunksize=4)):
            f.write(json.dumps(row) + "\n")
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    print(f"{len(tasks)} runs in {time.time() - t0:.0f}s → {path}")


def pilot(n):
    tasks = [(w, lt, s, arm, th, rev, 0.0)
             for w, lt, s in itertools.product(WORLDS, world.LEARNER_TYPES, range(n))
             for arm in (*CONTENDERS, "O")
             for th, rev in itertools.product(THETAS[arm], REVIEWS)]
    _grid(tasks, OUT / "pilot.jsonl")
    pick(OUT / "pilot.jsonl")


def pick(path: Path):
    rows = [json.loads(l) for l in path.open()]
    best = {}
    print("\nmedian summed crossing hours (mean of W1, W2 medians); rows = gate, cols = review:")
    for arm in (*CONTENDERS, "O"):
        score = {}
        for th, rev in itertools.product(THETAS[arm], REVIEWS):
            meds = [np.median([r["sum_h"] for r in rows if r["arm"] == arm and r["theta"] == th
                               and r["review"] == rev and r["world"] == w])
                    for w in WORLDS]
            score[(th, rev)] = float(np.mean(meds))
        best[arm] = list(min(score, key=score.get))
        print(f"  {arm}   " + "".join(f"  r{rev:<5}" for rev in REVIEWS))
        for th in THETAS[arm]:
            print(f"   θ{th:<5}" + "".join(f"{score[(th, rev)]:8.0f}" for rev in REVIEWS))
        print(f"   → gate {best[arm][0]}, review {best[arm][1]}")
    (OUT / "pilot_best.json").write_text(json.dumps(best, indent=1))


def final(n):
    best = json.loads((OUT / "pilot_best.json").read_text())
    tasks = [(w, lt, s, arm, *best[arm], brk)
             for w, lt, s in itertools.product(WORLDS, world.LEARNER_TYPES, range(1000, 1000 + n))
             for brk in (0.0, 30.0, 180.0)
             for arm in (*CONTENDERS, "O")]
    _grid(tasks, OUT / "final.jsonl")
    report()


def _boot_median(x, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    meds = np.median(rng.choice(x, (reps, len(x))), axis=1)
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def _paired(rows, arm, ref, **where):
    key = lambda r: (r["world"], r["ltype"], r["seed"], r["break"])  # noqa: E731
    sel = [r for r in rows if all(r[k] == v for k, v in where.items())]
    a = {key(r): r["sum_h"] for r in sel if r["arm"] == arm}
    b = {key(r): r["sum_h"] for r in sel if r["arm"] == ref}
    ks = sorted(set(a) & set(b))
    return np.array([100.0 * (a[k] - b[k]) / b[k] for k in ks])


def _boot_learners(rows, arm, ref, w, lt, reps=2000):
    """Median % difference with every learner's three break runs kept
    together: resample LEARNERS, not rows (the runs share a learner)."""
    per = {}
    for brk in (0.0, 30.0, 180.0):
        sel = [r for r in rows if r["world"] == w and r["ltype"] == lt and r["break"] == brk]
        a = {r["seed"]: r["sum_h"] for r in sel if r["arm"] == arm}
        b = {r["seed"]: r["sum_h"] for r in sel if r["arm"] == ref}
        for k in set(a) & set(b):
            per.setdefault(k, []).append(100.0 * (a[k] - b[k]) / b[k])
    clusters = [np.array(v) for v in per.values()]
    rng = np.random.default_rng(0)
    pick = rng.integers(0, len(clusters), (reps, len(clusters)))
    meds = [np.median(np.concatenate([clusters[i] for i in row])) for row in pick]
    point = float(np.median(np.concatenate(clusters)))
    return point, float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def report(path: Path = OUT / "final.jsonl"):
    rows = [json.loads(l) for l in path.open()]
    best = json.loads((OUT / "pilot_best.json").read_text())
    print(f"\n{len(rows)} runs. (gate, review) per arm: {best}")
    print("negative % = FASTER than the reference on the same learner; [95% CI]\n")
    for ref in ("B", "C"):
        print(f"### vs {ref}")
        print("| world | break | " + " | ".join(a for a in (*CONTENDERS, "O") if a != ref) + " |")
        print("|---|---|" + "---|" * (len(CONTENDERS)) + "")
        for w in WORLDS:
            for brk in (0.0, 30.0, 180.0):
                cells = []
                for arm in (*CONTENDERS, "O"):
                    if arm == ref:
                        continue
                    d = _paired(rows, arm, ref, world=w, **{"break": brk})
                    if not len(d):
                        cells.append("–")
                        continue
                    m, lo, hi = _boot_median(d)
                    cells.append(f"{m:+.1f}% [{lo:+.1f}, {hi:+.1f}] ({(d < 0).mean():.0%} faster)")
                print(f"| {w} | {brk:.0f}d | " + " | ".join(cells) + " |")
        print()
    print("### A and H vs B by learner type, all breaks (cluster bootstrap over learners)")
    for w in WORLDS:
        for lt in world.LEARNER_TYPES:
            parts = []
            for arm in ("A", "H"):
                m, lo, hi = _boot_learners(rows, arm, "B", w, lt)
                parts.append(f"{arm} {m:+6.1f}% [{lo:+.1f}, {hi:+.1f}]")
            print(f"  {w} {lt:7s}  " + "   ".join(parts))
    print("\n### medians (summed crossing hours), goals crossed, false declarations")
    for w in WORLDS:
        for arm in (*CONTENDERS, "O"):
            sel = [r for r in rows if r["world"] == w and r["arm"] == arm]
            print(f"  {w} {arm:3s} Σh {np.median([r['sum_h'] for r in sel]):6.0f}  "
                  f"crossed {np.mean([r['crossed'] for r in sel]):4.1f}/27  "
                  f"false decl {np.mean([r['false_decl'] for r in sel]):4.1f}  "
                  f"probes {np.mean([r['probes'] for r in sel]):5.1f}  "
                  f"lessons {np.mean([r['lessons'] for r in sel]):4.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("pilot", "pick", "final", "report"))
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    {"pilot": lambda: pilot(a.n), "pick": lambda: pick(OUT / "pilot.jsonl"),
     "final": lambda: final(a.n), "report": report}[a.phase]()
