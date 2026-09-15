#!/usr/bin/env python3
"""Fail when a served question's declared API has no teachable KC first.

`qmatrix_tags.json:new_syntax` is the per-question API contract. Runtime
`LessonGate` blocks the question until every target KC is exposed, unless a
mapped atom is above 85% mastery. This watch checks the static half: each API
must have an owning lesson KC at or before every KC the question targets.

Run: python3 Local_Deployed_Shared/lessons/watch_problem_prerequisites.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent


def frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---", text, re.S)
    if not match:
        return {}
    result = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            result[key.strip()] = [x.strip().strip("'\"") for x in value[1:-1].split(",") if x.strip()]
        else:
            result[key.strip()] = value.strip("'\"")
    return result


def main() -> None:
    registry = json.loads((HERE / "kc_registry.json").read_text(encoding="utf-8"))
    qmatrix = json.loads((HERE / "qmatrix_tags.json").read_text(encoding="utf-8"))
    kcs = [row["id"] for row in registry["kcs"]]
    prereqs = {row["id"]: row.get("prereqs") or [] for row in registry["kcs"]}

    rank, visiting = {}, set()
    def visit(kc: str) -> int:
        if kc in rank:
            return rank[kc]
        if kc in visiting:
            raise AssertionError(f"prerequisite cycle through {kc}")
        visiting.add(kc)
        rank[kc] = 1 + max((visit(parent) for parent in prereqs[kc]), default=-1)
        visiting.remove(kc)
        return rank[kc]
    for kc in kcs:
        visit(kc)

    owners = {}
    for page in HERE.glob("**/kp-*.md"):
        fm = frontmatter(page)
        kc = fm.get("kc")
        if kc not in rank:
            continue
        for symbol in fm.get("new_syntax") or []:
            owners.setdefault(symbol, kc)
            if rank[kc] < rank[owners[symbol]]:
                owners[symbol] = kc

    problems = []
    for qid, tags in qmatrix.items():
        targets = tags.get("target_kcs") or []
        for symbol in tags.get("new_syntax") or []:
            owner = owners.get(symbol) or owners.get(symbol.split("#")[0])
            if owner is None:
                problems.append(f"q{qid}: {symbol} has no teaching KC")
            elif any(rank.get(owner, 10**9) > rank.get(kc, -1) for kc in targets):
                problems.append(f"q{qid}: {symbol} first taught by {owner}, after target {targets}")
    assert not problems, "Untaught/late question APIs:\n  " + "\n  ".join(problems[:30])
    print(f"OK: {len(qmatrix)} tagged questions; every declared API has a lesson at/before its target KC")


if __name__ == "__main__":
    main()
