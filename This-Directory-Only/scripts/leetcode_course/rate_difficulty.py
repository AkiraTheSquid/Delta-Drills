"""Per-concept difficulty for the LeetCode bank, with the same Jev king-of-the-hill
rater the main bank uses (../typesafe_difficulty/): champion = the concept's
hardest problem at 100, easiest at 15, bands of 5.

Reuses judge.Judge / rate.rate_concept / scale.scale_concept unchanged; only the
context paragraph and criteria (they describe PyTorch drills there) and the cache
file are swapped, before the Judge is built. Both are in the judge's cache
fingerprint, so these judgments never mix with the tensor bank's.

Seed champion = the source label (Hard 90 / Medium 60 / Easy 30, generated 50).
Writes the `difficulty_score` back into leetcode_course/bank.jsonl.
Needs TYPESAFE_API_KEY (This-Directory-Only/typesafe_difficulty/.env).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "typesafe_difficulty"))

import bank as ts_bank  # noqa: E402
import judge  # noqa: E402
from rate import rate_concept  # noqa: E402
from scale import scale_concept  # noqa: E402

from build_pool import RAW_DIR  # noqa: E402

COURSE_DIR = HERE.parents[1] / "leetcode_course"
SEED = {"Hard": 90, "Medium": 60, "Easy": 30}

judge.CACHE = RAW_DIR / "difficulty_cache.jsonl"
judge.CONTEXT = (
    "These are two coding-interview problems (LeetCode style) that practise the same "
    "algorithmic pattern. Each has a statement and a reference solution. A problem is "
    "HARDER when a learner who knows the pattern would need more insight to see how it "
    "applies, more cases or invariants to get right, a less obvious reduction, or more "
    "steps to implement correctly. Longer statement text alone does not make a problem harder."
)
judge.CRITERIA = {
    "true": ("The first-named problem needs more insight, trickier invariants or edge cases, "
             "or a longer correct implementation than the other"),
    "false": ("The first-named problem is about as hard as, or easier than, the other; "
              "a longer statement on its own is not harder"),
}


def main() -> int:
    rows = [json.loads(line) for line in open(COURSE_DIR / "bank.jsonl")]
    groups: dict[str, list[ts_bank.Problem]] = {}
    for i, r in enumerate(rows):
        seed = SEED.get(r["difficulty_label"], 50)
        groups.setdefault(r["concept"], []).append(
            ts_bank.Problem(id=i, concept=r["concept"], question=r["statement"][:3500],
                            answer=r["solution"][:2500], current_score=seed))
    j = judge.Judge()
    for concept, probs in groups.items():
        if len(probs) < 2:
            for p in probs:
                rows[p.id]["difficulty_score"] = 50
            continue
        champion, p_by, rounds, unbeaten = rate_concept(j, probs)
        scores = scale_concept(p_by) or {p.id: 50 for p in probs}
        for p in probs:
            rows[p.id]["difficulty_score"] = scores[p.id]
        print(f"{concept}: {len(probs)} problems, champion {rows[champion.id]['id']} "
              f"({rows[champion.id]['title']}), {rounds} round(s){'' if unbeaten else ', cycle'}", file=sys.stderr)
    with open(COURSE_DIR / "bank.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"rated {len(rows)} problems; {j.calls} API calls, {j.input_tokens} input tokens "
          f"≈ ${j.input_tokens * 0.042 / 1e6:.3f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
