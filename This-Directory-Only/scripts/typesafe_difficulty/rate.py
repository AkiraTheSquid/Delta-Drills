#!/usr/bin/env python3
"""Find each concept's hardest drill and rate every drill against it.

King-of-the-hill per concept: the seed champion is the drill with the highest
current `difficulty_score`; every other drill is judged against it in one
batched pass; the strongest drill that beats it (combined p > 0.5) takes the
title and the pass repeats. Pairwise judgments need not be transitive, so the
title can cycle (A beats B, B beats A on the return match): when the next
champion is one we already held, or MAX_ROUNDS is spent, the champion is the
candidate the FEWEST drills beat, and the ratings are its pass. A stable
champion costs one pass and each upset one more — the legacy rater paid a
full tournament PLUS a full rating pass, two calls per pair.

    This-Directory-Only/backend/.venv/bin/python3 rate.py [--concept <id>] [--limit N]

Writes This-Directory-Only/typesafe_difficulty/ratings.csv:
    concept,question_id,champion_id,p_harder,rounds
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bank import DATA_DIR, Problem, group_by_concept, load_problems  # noqa: E402
from judge import Judge  # noqa: E402

MAX_ROUNDS = 6
RATINGS = DATA_DIR / "ratings.csv"


def rate_concept(judge: Judge, drills: list[Problem]) -> tuple[Problem, dict[int, float], int, bool]:
    """→ (champion, {id: P(harder than champion)}, rounds, unbeaten)."""
    by_id = {d.id: d for d in drills}
    champion = max(drills, key=lambda d: (d.current_score, -d.id))
    held: dict[int, tuple[int, dict[int, float]]] = {}  # champion id → (upset count, its pass)
    for rounds in range(1, MAX_ROUNDS + 1):
        probs = judge.p_harder(champion, drills)
        upsets = [(p, qid) for qid, p in probs.items() if p > 0.5]
        held[champion.id] = (len(upsets), probs)
        if not upsets:
            return champion, probs, rounds, True
        _, best = max(upsets)
        if best in held:
            break  # a cycle: the title would just go round again
        champion = by_id[best]
    # No unbeaten drill found: the best-supported title is the one fewest drills beat.
    best_id = min(held, key=lambda qid: (held[qid][0], qid))
    return by_id[best_id], held[best_id][1], len(held), False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", help="rate only this concept id")
    ap.add_argument("--limit", type=int, help="rate only the first N concepts (smoke test)")
    args = ap.parse_args()

    groups = group_by_concept(load_problems())
    if args.concept:
        groups = {args.concept: groups[args.concept]}
    names = sorted(groups)[: args.limit] if args.limit else sorted(groups)

    judge = Judge()
    rows = []
    for name in names:
        drills = groups[name]
        if len(drills) < 2:
            print(f"{name}: {len(drills)} drill, nothing to compare", file=sys.stderr)
            continue
        champion, probs, rounds, unbeaten = rate_concept(judge, drills)
        for d in drills:
            rows.append((name, d.id, champion.id, f"{probs[d.id]:.6f}", rounds))
        above = sum(p > 0.5 for p in probs.values())
        print(f"{name}: {len(drills)} drills, champion q{champion.id}, {rounds} round(s)"
              f"{'' if unbeaten else f', CYCLE — {above} drill(s) still rate above it'}, "
              f"{judge.calls} calls / {judge.input_tokens} tok so far", file=sys.stderr)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = {}
    if RATINGS.exists() and (args.concept or args.limit):
        with RATINGS.open() as fh:
            for r in csv.DictReader(fh):
                existing[(r["concept"], int(r["question_id"]))] = r
    for name, qid, champ, p, rounds in rows:
        existing[(name, qid)] = {"concept": name, "question_id": qid, "champion_id": champ,
                                 "p_harder": p, "rounds": rounds}
    with RATINGS.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["concept", "question_id", "champion_id", "p_harder", "rounds"])
        w.writeheader()
        for key in sorted(existing, key=lambda k: (k[0], k[1])):
            w.writerow(existing[key])
    print(f"wrote {RATINGS} ({len(existing)} rows); {judge.calls} API calls, "
          f"{judge.input_tokens} input tokens ≈ ${judge.input_tokens * 0.042 / 1e6:.3f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
