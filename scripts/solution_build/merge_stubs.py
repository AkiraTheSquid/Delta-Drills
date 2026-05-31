#!/usr/bin/env python3
"""Merge authored starter_code stubs back into the round-3 override file.

Reads scripts/solution_build/stubs/*.jsonl ({id, starter_code}) and updates the
matching rows' `starter_code` in
This-Directory-Only/chatgpt/function_mode_overrides_round3.jsonl in place,
preserving every other field and row order. Rows with no authored stub are left
untouched.
"""
from __future__ import annotations
import json, glob
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STUBS_DIR = Path(__file__).resolve().parent / "stubs"
OVERRIDES = REPO / "This-Directory-Only" / "chatgpt" / "function_mode_overrides_round3.jsonl"


def main() -> None:
    stubs: dict[int, str] = {}
    for f in sorted(STUBS_DIR.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                if o.get("starter_code"):
                    stubs[int(o["id"])] = o["starter_code"]

    rows = [json.loads(l) for l in OVERRIDES.read_text().splitlines() if l.strip()]
    updated = 0
    for r in rows:
        qid = r.get("id", r.get("question_id"))
        if qid in stubs:
            r["starter_code"] = stubs[qid]
            updated += 1

    OVERRIDES.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"stubs authored: {len(stubs)}  override rows updated: {updated}  total rows: {len(rows)}")
    missing = sorted(set(stubs) - {r.get("id", r.get("question_id")) for r in rows})
    if missing:
        print("WARN authored ids not present in override file:", missing)


if __name__ == "__main__":
    main()
