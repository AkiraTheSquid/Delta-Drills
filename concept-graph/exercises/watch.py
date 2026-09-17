"""watch.py — health checks for exercises/

Verifies every 0_*.json file is valid JSON, has the required schema fields,
and that every atom reference resolves to a known vocab atom.
"""
import sys
import os
import json
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB = os.path.join(HERE, "..", "vocab", "atoms.json")

EX_REQUIRED = {"exercise_id", "title", "arena_chapter", "arena_part", "arena_index", "atoms"}
ATOM_TAG_REQUIRED = {"id", "role", "evidence"}
VALID_ROLES = {"core", "incidental"}


def _files():
    return sorted(glob.glob(os.path.join(HERE, "0_*.json")))


# ── Import checks ──────────────────────────────
def check_imports():
    assert os.path.exists(VOCAB), f"missing vocab/atoms.json"
    files = _files()
    assert files, "no exercise files found"
    for p in files:
        with open(p) as f:
            json.load(f)


# ── Public API checks ─────────────────────────
def check_public_api():
    for p in _files():
        with open(p) as f:
            ex = json.load(f)
        missing = EX_REQUIRED - set(ex.keys())
        assert not missing, f"{os.path.basename(p)} missing fields: {missing}"
        assert isinstance(ex["atoms"], list), f"{os.path.basename(p)} atoms is not a list"


# ── Invariant checks ──────────────────────────
def check_invariants():
    with open(VOCAB) as f:
        known = {a["id"] for a in json.load(f)["atoms"]}
    for p in _files():
        with open(p) as f:
            ex = json.load(f)
        seen_ids = set()
        for a in ex["atoms"]:
            missing = ATOM_TAG_REQUIRED - set(a.keys())
            assert not missing, f"{os.path.basename(p)} atom {a.get('id', '?')} missing: {missing}"
            assert a["role"] in VALID_ROLES, f"{os.path.basename(p)} atom {a['id']} bad role: {a['role']}"
            assert a["id"] in known, f"{os.path.basename(p)} unknown atom id: {a['id']}"
            assert a["evidence"], f"{os.path.basename(p)} empty evidence for {a['id']}"
            assert a["id"] not in seen_ids, f"{os.path.basename(p)} duplicate atom tag: {a['id']}"
            seen_ids.add(a["id"])


# ── Run all checks ────────────────────────────
if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
