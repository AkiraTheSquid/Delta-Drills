"""watch.py — health checks for aisc/data (figure data snapshots)

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


def check_imports():
    _load("kc_graph.json")
    _load("seth_progress.json")


def check_public_api():
    g = _load("kc_graph.json")
    for key in ("kcs", "lessons", "n_questions"):
        assert key in g, f"kc_graph.json lost {key!r}, which fig-graphs.js reads"


def check_invariants():
    ignore = os.path.join(HERE, "..", "..", ".vercelignore")
    if os.path.exists(ignore):
        with open(ignore, encoding="utf-8") as f:
            rules = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        for r in rules:
            assert not r.startswith("aisc"), (
                f".vercelignore rule {r!r} would drop the figures' runtime data"
            )


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
