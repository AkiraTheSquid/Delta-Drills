"""Build the LeetCode Patterns problem pool from the raw sources.

Two tiers:
- core: NeetCode 250 + Striver A2Z LeetCode problems, minus every problem
  either sheet flags as "In Top 100 Liked".
- fallback: the rest of the big LeetCode sheet, plus the Codewars subset.

Statements, starter code, reference solutions and test asserts for LeetCode
problems come from newfacade/LeetCodeDataset (Apache-2.0), joined on the
LeetCode number; a problem with no dataset row is dropped (the sheets carry
no tests). Codewars rows carry their own solutions and input/output pairs.

Raw inputs live outside the repo in RAW_DIR (100+ MB). Writes RAW_DIR/pool.jsonl.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

RAW_DIR = Path(os.environ.get("LEETCODE_RAW_DIR", Path.home() / ".cache/delta-drills/leetcode"))


def _rows(name: str) -> list[dict]:
    with open(RAW_DIR / name, newline="") as fh:
        return list(csv.DictReader(fh))


def load_dataset() -> dict[str, dict]:
    ds = {}
    for split in ("train", "test"):
        for line in open(RAW_DIR / f"{split}.jsonl"):
            row = json.loads(line)
            ds[str(row["question_id"])] = row
    return ds


def sheet_tiers() -> tuple[dict[str, str], set[str]]:
    """{lc number: sheet topic} for core problems, and the Top-100 set."""
    core, top = {}, set()
    for name, topic_col in (("neetcode_250.csv", "Topic"), ("striver_a2z_dsa_sheet.csv", "Subtopic")):
        for r in _rows(name):
            lc = r["LC #"].strip()
            if not lc:
                continue
            if r["In Top 100 Liked"] == "Yes":
                top.add(lc)
            else:
                core.setdefault(lc, r[topic_col])
    for lc in top:
        core.pop(lc, None)
    return core, top


def leetcode_rows(ds, core, top) -> list[dict]:
    big = {r["Question ID"]: r for r in _rows("leetcode_big.csv")}
    out = []
    for lc in sorted(set(core) | set(big), key=int):
        if lc in top or lc not in ds:
            continue
        d = ds[lc]
        tier = "core" if lc in core else "fallback"
        out.append({
            "id": f"lc{lc}",
            "source": "leetcode",
            "tier": tier,
            "title": d["task_id"].replace("-", " ").title(),
            "slug": d["task_id"],
            "url": f"https://leetcode.com/problems/{d['task_id']}/",
            "difficulty_label": d["difficulty"],
            "source_tags": list(d["tags"]),
            "sheet_topic": core.get(lc, ""),
            "statement": d["problem_description"],
            "starter_code": d["starter_code"],
            "entry_point": d["entry_point"],
            # The dataset's preamble imports sortedcontainers for every problem; only
            # solutions that name Sorted* need it, and the app's kernel lacks it.
            "preamble": d["prompt"] if "Sorted" in d["completion"]
            else d["prompt"].replace("from sortedcontainers import SortedList\n", ""),
            "solution": d["completion"],
            "tests": {"kind": "assert", "check": d["test"]},
        })
    return out


def codewars_rows() -> list[dict]:
    out = []
    for r in _rows("codewars.csv"):
        try:
            io = json.loads(r["input_output"])
            sols = json.loads(r["solutions"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not io.get("fn_name") or not sols or not io.get("inputs"):
            continue  # stdin-style or unsolved kata: no function to call
        if len(io["inputs"]) != len(io.get("outputs") or []):
            continue  # zip() would silently skip the unmatched cases
        out.append({
            "id": f"cw{r['problem_id']}",
            "source": "codewars",
            "tier": "fallback",
            "title": r["question"].strip().splitlines()[0].lstrip("# ").strip()[:80],
            "slug": r["Link"].rsplit("/", 1)[-1],
            "url": r["Link"],
            "difficulty_label": r["interview"],
            "source_tags": [],
            "sheet_topic": "",
            "statement": r["question"],
            "starter_code": r["starter_code"],
            "entry_point": io["fn_name"],
            "preamble": "",
            "solution": sols[0],
            "alt_solutions": sols[1:3],
            "tests": {"kind": "io", "inputs": io["inputs"], "outputs": io["outputs"]},
        })
    return out


def main() -> None:
    ds = load_dataset()
    core, top = sheet_tiers()
    rows = leetcode_rows(ds, core, top) + codewars_rows()
    with open(RAW_DIR / "pool.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    by = {}
    for r in rows:
        by[(r["source"], r["tier"])] = by.get((r["source"], r["tier"]), 0) + 1
    print(f"{len(rows)} problems -> {RAW_DIR / 'pool.jsonl'}", by)
    missing = sorted(set(core) - set(ds), key=int)
    print(f"core problems with no dataset row (dropped): {len(missing)} {missing}")


if __name__ == "__main__":
    main()
