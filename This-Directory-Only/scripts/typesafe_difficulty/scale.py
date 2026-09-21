#!/usr/bin/env python3
"""Turn P(harder than the concept champion) into `difficulty_score` bands.

Within a concept: `score = FLOOR + (100 - FLOOR) * (logit(p) - lo) / (0 - lo)`
where `lo` is the concept's lowest logit. The champion (p = 0.5, logit 0)
lands on 100 and the easiest drill on FLOOR; everything between keeps the
log-odds spacing the legacy `score = odds^0.8` formula ranked on. Scores are
then rounded to the nearest BAND so `solo_progress.next_band` still sees
bands of several drills rather than one band per drill.

Reads ratings.csv, writes difficulty_scores.csv and the override layer
This-Directory-Only/chatgpt/typesafe_difficulty_overrides.jsonl
({"id": <qid>, "difficulty_score": <int>} per line).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bank import DATA_DIR, ROOT, group_by_concept, load_problems  # noqa: E402
from judge import logit  # noqa: E402

FLOOR = 15
BAND = 5
RATINGS = DATA_DIR / "ratings.csv"
SCORES = DATA_DIR / "difficulty_scores.csv"
OVERRIDES = ROOT / "This-Directory-Only" / "chatgpt" / "typesafe_difficulty_overrides.jsonl"


def scale_concept(p_by_id: dict[int, float]) -> dict[int, int] | None:
    """Affine in logit space, champion → 100, easiest → FLOOR, rounded to BAND.

    `None` when the concept is degenerate (nothing below the champion), so the
    caller keeps the current scores rather than shipping a flat 100.
    """
    logits = {qid: min(0.0, logit(p)) for qid, p in p_by_id.items()}
    lo = min(logits.values())
    if lo == 0:
        return None
    out = {}
    for qid, lg in logits.items():
        raw = FLOOR + (100 - FLOOR) * (lg - lo) / (0 - lo)
        out[qid] = int(BAND * round(raw / BAND))
    return out


def _write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def load_ratings() -> dict[str, dict[int, float]]:
    by_concept: dict[str, dict[int, float]] = {}
    with RATINGS.open() as fh:
        for r in csv.DictReader(fh):
            by_concept.setdefault(r["concept"], {})[int(r["question_id"])] = float(r["p_harder"])
    return by_concept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true",
                    help="write the override even though some rateable concept has no ratings")
    args = ap.parse_args()

    groups = group_by_concept(load_problems())
    ratings = load_ratings()
    rateable = {c for c, ds in groups.items() if len(ds) >= 2}
    missing = sorted(rateable - set(ratings))
    incomplete = sorted(c for c in ratings if set(ratings[c]) != {d.id for d in groups.get(c, [])})
    if incomplete:
        print(f"ratings do not match the bank for: {incomplete} — re-run rate.py on them", file=sys.stderr)
        return 1
    if missing:
        print(f"{len(missing)} rateable concept(s) unrated: {missing[:8]}{'…' if len(missing) > 8 else ''}",
              file=sys.stderr)
        if not args.allow_partial:
            print("refusing to write a partial override (pass --allow-partial to keep their current scores)",
                  file=sys.stderr)
            return 1

    rows, flat = [], []
    for concept, p_by_id in sorted(ratings.items()):
        champions = [qid for qid, p in p_by_id.items() if p == 0.5]
        assert len(champions) == 1, f"{concept}: expected exactly one champion at p=0.5, got {champions}"
        scores = scale_concept(p_by_id)
        if scores is None:
            flat.append(concept)
            continue
        for qid, score in scores.items():
            rows.append({"concept": concept, "question_id": qid,
                         "p_harder": f"{p_by_id[qid]:.6f}", "difficulty_score": score})
    rows.sort(key=lambda r: r["question_id"])
    if flat:
        print(f"{len(flat)} concept(s) with nothing below the champion keep their current scores: {flat}",
              file=sys.stderr)

    buf = []
    w = csv.DictWriter(_Lines(buf), fieldnames=["question_id", "concept", "p_harder", "difficulty_score"])
    w.writeheader()
    w.writerows(rows)
    _write_atomic(SCORES, "".join(buf))
    _write_atomic(OVERRIDES, "".join(
        json.dumps({"id": r["question_id"], "difficulty_score": r["difficulty_score"]}) + "\n" for r in rows))
    skipped = len(load_problems()) - len(rows)
    print(f"wrote {SCORES} and {OVERRIDES}: {len(rows)} drills rescored, {skipped} keep their current score "
          f"(one-drill concepts{', unrated' if missing else ''}{', flat' if flat else ''})", file=sys.stderr)
    return 0


class _Lines:
    """A write() sink for csv.writer that collects into a list."""

    def __init__(self, buf: list[str]) -> None:
        self.buf = buf

    def write(self, s: str) -> None:
        self.buf.append(s)


if __name__ == "__main__":
    raise SystemExit(main())
