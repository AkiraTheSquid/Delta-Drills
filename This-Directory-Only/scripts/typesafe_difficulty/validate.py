#!/usr/bin/env python3
"""Sanity report on the new scores BEFORE the layer is registered.

Prints, per concept and overall: Spearman rank correlation between the new
and current `difficulty_score`, the number of bands and the smallest band,
and the drills that moved furthest. Nothing here decides; a human reads it.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bank import load_problems  # noqa: E402
from scale import SCORES  # noqa: E402


def spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return None if vx == 0 or vy == 0 else cov / (vx * vy)


def main() -> int:
    current = {p.id: p.current_score for p in load_problems()}
    by_concept: dict[str, list[tuple[int, int, float]]] = {}
    with SCORES.open() as fh:
        for r in csv.DictReader(fh):
            qid = int(r["question_id"])
            by_concept.setdefault(r["concept"], []).append((qid, int(r["difficulty_score"]), float(r["p_harder"])))

    all_new, all_old, movers = [], [], []
    print(f"{'concept':48} {'n':>3} {'rho':>6} {'bands':>5} {'min':>3} {'top-p-range':>12}")
    for concept, rows in sorted(by_concept.items()):
        new = [s for _, s, _ in rows]
        old = [current[q] for q, _, _ in rows]
        rho = spearman(old, new)
        bands: dict[int, int] = {}
        for s in new:
            bands[s] = bands.get(s, 0) + 1
        ps = sorted(p for _, _, p in rows)
        print(f"{concept[:48]:48} {len(rows):>3} {('  n/a' if rho is None else f'{rho:6.2f}'):>6} "
              f"{len(bands):>5} {min(bands.values()):>3} {ps[0]:.3f}..{ps[-2] if len(ps) > 1 else ps[-1]:.3f}")
        all_new += new
        all_old += old
        movers += [(abs(s - current[q]), q, current[q], s, concept) for q, s, _ in rows]
    overall = spearman(all_old, all_new)
    print(f"\noverall rho={'n/a' if overall is None else f'{overall:.3f}'} over {len(all_new)} drills")
    print("biggest movers (|Δ|, q, old→new, concept):")
    for d, q, o, s, c in sorted(movers, reverse=True)[:25]:
        print(f"  {d:>3} q{q:<5} {o:>3}→{s:<3} {c}")
    singles = [(c, len(r)) for c, r in by_concept.items()
               if any(v == 1 for v in __import__('collections').Counter(s for _, s, _ in r).values())]
    print(f"\nconcepts with a one-drill band: {len(singles)} of {len(by_concept)} (current bank: 91 of 92 — a band of one is tolerated by the last-served guard)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
