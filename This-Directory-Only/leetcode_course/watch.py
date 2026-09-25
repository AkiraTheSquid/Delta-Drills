"""watch.py — health checks for leetcode_course (data)

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import json
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "leetcode_course"))


def check_imports():
    import check_graph  # noqa: F401


def check_public_api():
    kcs = json.load(open(os.path.join(HERE, "patterns.json")))["kcs"]
    assert all(k["id"].startswith("leetcode.") and k.get("title") for k in kcs)


def check_invariants():
    import check_graph
    kcs = json.load(open(os.path.join(HERE, "patterns.json")))["kcs"]
    assert not check_graph.findings(kcs), check_graph.findings(kcs)
    bank = os.path.join(HERE, "bank.jsonl")
    if os.path.exists(bank):
        ids, known = set(), {k["id"] for k in kcs}
        for line in open(bank):
            row = json.loads(line)
            assert row["id"] not in ids, f"duplicate bank id {row['id']}"
            ids.add(row["id"])
            assert row["concept"] in known, f"{row['id']}: unknown concept {row['concept']}"
            assert row.get("solution") and row.get("tests"), f"{row['id']}: no solution or tests"


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
