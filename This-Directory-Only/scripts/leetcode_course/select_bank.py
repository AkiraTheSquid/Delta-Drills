"""Assemble the course bank from tagged problems.

- Every core problem (NeetCode 250 / Striver, verified or generated) goes to its
  primary pattern, unless TypeSafe judged its statement not self-contained or
  dependent on a missing figure.
- A concept with fewer than FLOOR core problems is topped up to FLOOR from the
  verified LeetCode fallback tier: problems whose probability for THAT concept is
  at least TOPUP_P (primary or not), Medium/Hard before Easy, strongest first.
  No fallback problem is added to a concept that already meets the floor.

Writes This-Directory-Only/leetcode_course/bank.jsonl: one problem per line with
its concept, tier, statement, starter code, reference solution and tests.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from build_pool import RAW_DIR

FLOOR = 8
TOPUP_P = 0.8
COURSE_DIR = Path(__file__).resolve().parents[2] / "leetcode_course"
LEVEL = {"Hard": 0, "Medium": 1, "Easy": 2}


def answers_by_id() -> dict[str, dict]:
    """Tag answers whose fingerprint matches each problem AS IT IS NOW (the same
    state tag.py builds), so an edited problem or question set reads as untagged
    instead of reusing stale answers."""
    import tag
    tag.FALLBACK = True
    by_fp = {}
    for line in open(RAW_DIR / "tags_cache.jsonl"):
        r = json.loads(line)
        by_fp[r["fp"]] = r["answers"]
    kcs_rows = json.loads((COURSE_DIR / "patterns.json").read_text())["kcs"]
    qs = tag.questions(kcs_rows)
    out = {}
    for p in tag.load_problems():
        fp = tag.fingerprint(tag.state_of(p), qs)
        if fp in by_fp:
            out[p["id"]] = by_fp[fp]
    return out


def problems() -> dict[str, dict]:
    ok = {json.loads(l)["id"] for l in open(RAW_DIR / "verified.jsonl") if json.loads(l)["ok"]}
    out = {}
    for line in open(RAW_DIR / "pool.jsonl"):
        p = json.loads(line)
        if p["source"] == "leetcode" and p["id"] in ok:
            out[p["id"]] = {k: p[k] for k in ("id", "tier", "title", "url", "difficulty_label", "statement",
                                                "starter_code", "entry_point", "preamble", "solution", "tests")}
    for line in open(RAW_DIR / "generated.jsonl"):
        g = json.loads(line)
        if g["ok"]:
            out[g["key"]] = {"id": g["key"], "tier": "core", "title": g["title"], "url": g["meta"]["url"],
                             "difficulty_label": "", "statement": g["statement"], "starter_code": g["starter_code"],
                             "entry_point": g["entry_point"], "preamble": "", "solution": g["solution"],
                             "tests": {"kind": "generated", "shape": g["shape"], "arg_types": g.get("arg_types"),
                                       "return_type": g.get("return_type"), "compare": g.get("compare", "exact"),
                                       "inputs": g["inputs"], "outputs": g["outputs"]}}
    return out


def main() -> int:
    kcs = [k["id"] for k in json.loads((COURSE_DIR / "patterns.json").read_text())["kcs"]]
    ans, probs = answers_by_id(), problems()
    untagged = [pid for pid, p in probs.items() if p["tier"] == "core" and pid not in ans]
    if untagged:
        print(f"{len(untagged)} core problems not tagged yet (run tag.py): {untagged[:8]}", file=sys.stderr)
        return 1
    clean = lambda a: a["self_contained"] >= 0.5 and a["needs_figure"] < 0.5
    bank, dropped = {}, []
    for pid, p in probs.items():
        if p["tier"] != "core":
            continue
        a = ans[pid]
        if not clean(a):
            dropped.append(pid)
            continue
        concept = max(kcs, key=lambda k: a[k])
        bank[pid] = {**p, "concept": concept, "p_concept": round(a[concept], 3)}
    count = {k: sum(b["concept"] == k for b in bank.values()) for k in kcs}
    for k in kcs:
        need = FLOOR - count[k]
        if need <= 0:
            continue
        cands = [pid for pid, p in probs.items() if p["tier"] == "fallback" and pid in ans and pid not in bank
                 and ans[pid][k] >= TOPUP_P and clean(ans[pid])]
        cands.sort(key=lambda pid: (LEVEL.get(probs[pid]["difficulty_label"], 3), -ans[pid][k]))
        for pid in cands[:need]:
            bank[pid] = {**probs[pid], "concept": k, "p_concept": round(ans[pid][k], 3)}
    with open(COURSE_DIR / "bank.jsonl", "w") as fh:
        for pid in sorted(bank, key=lambda i: (kcs.index(bank[i]["concept"]), i)):
            fh.write(json.dumps(bank[pid]) + "\n")
    print(f"{len(bank)} problems -> {COURSE_DIR / 'bank.jsonl'}; dropped {len(dropped)} core as not self-contained: {dropped}")
    for k in kcs:
        core = sum(b["concept"] == k and b["tier"] == "core" for b in bank.values())
        fb = sum(b["concept"] == k and b["tier"] == "fallback" for b in bank.values())
        print(f"  {core:3d} core + {fb:2d} fallback  {k}{'  <- BELOW FLOOR' if core + fb < FLOOR else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
