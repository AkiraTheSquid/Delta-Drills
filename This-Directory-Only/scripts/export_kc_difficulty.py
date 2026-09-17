#!/usr/bin/env python3
"""Export a per-KC difficulty index for the lesson knowledge graph.

The graph needs to answer "how hard is this concept?" for concepts the learner
has never touched, so it can extrapolate a mastery estimate from their overall
demonstrated level instead of showing an empty grey bubble. The question bank
already carries a calibrated `difficulty_score` (0-100) per question, and
`lessons/qmatrix_tags.json` says which KCs each question targets — so a KC's
difficulty is the mean difficulty of the questions written to test it.

Only `target_kcs` count. `supporting_kcs` are prerequisites the question leans
on, not what it measures; folding them in would drag every foundational KC's
difficulty up toward the hardest question that happens to use it.

Output: Local_Deployed_Shared/concept-graph/kc_difficulty.json
    {
      "version": 1,
      "scale": "question difficulty_score, 0-100",
      "mean": 38.2,                       # bank-wide mean, the neutral default
      "kcs": { "<kc id>": {"d": 22.4, "n": 8}, ... }
    }

Regenerate after editing the question bank or the q-matrix:
    python3 This-Directory-Only/scripts/export_kc_difficulty.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "Local_Deployed_Shared"
REGISTRY = SHARED / "lessons" / "kc_registry.json"
QMATRIX = SHARED / "lessons" / "qmatrix_tags.json"
BANK = SHARED / "questions.json"
OUT = SHARED / "concept-graph" / "kc_difficulty.json"


def main() -> int:
    registry = json.loads(REGISTRY.read_text())
    qmatrix = json.loads(QMATRIX.read_text())
    bank = json.loads(BANK.read_text())

    kc_ids = {k["id"] for k in registry.get("kcs", [])}
    difficulty_by_qid = {}
    for q in bank:
        try:
            difficulty_by_qid[str(q["id"])] = float(q["difficulty_score"])
        except (KeyError, TypeError, ValueError):
            continue

    per_kc = defaultdict(list)
    for qid, tags in qmatrix.items():
        d = difficulty_by_qid.get(str(qid))
        if d is None:
            continue
        for kc in tags.get("target_kcs") or []:
            if kc in kc_ids:
                per_kc[kc].append(d)

    all_scores = [d for scores in per_kc.values() for d in scores]
    if not all_scores:
        raise SystemExit("no difficulty scores matched any KC — check the q-matrix")

    out = {
        "version": 1,
        "scale": "question difficulty_score, 0-100",
        "generated_from": ["lessons/qmatrix_tags.json target_kcs", "questions.json difficulty_score"],
        "mean": round(sum(all_scores) / len(all_scores), 1),
        "kcs": {
            kc: {"d": round(sum(v) / len(v), 1), "n": len(v)}
            for kc, v in sorted(per_kc.items())
        },
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")

    missing = sorted(kc_ids - set(per_kc))
    print(f"[kc-difficulty] {len(per_kc)}/{len(kc_ids)} KCs covered, mean {out['mean']} → {OUT.relative_to(ROOT)}")
    if missing:
        print(f"[kc-difficulty] no targeted questions for: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
