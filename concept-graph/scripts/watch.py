"""watch.py — health checks for scripts/

Sanity-checks that each script is importable and exposes its main() entry
point. The deep validation already happens in concept-graph/watch.py
(which subprocess-runs validate.py + validate_graph.py).
"""
import sys
import os
import importlib

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ("validate", "validate_graph", "validate_concept", "derive_prereqs", "export_graphml")


# ── Import checks ──────────────────────────────
def check_imports():
    sys.path.insert(0, HERE)
    try:
        for name in SCRIPTS:
            importlib.import_module(name)
    finally:
        sys.path.pop(0)


# ── Public API checks ─────────────────────────
def check_public_api():
    sys.path.insert(0, HERE)
    try:
        for name in SCRIPTS:
            m = importlib.import_module(name)
            assert callable(getattr(m, "main", None)), f"{name}.main is missing or not callable"
    finally:
        sys.path.pop(0)


# ── Invariant checks ──────────────────────────
def check_invariants():
    for name in SCRIPTS:
        p = os.path.join(HERE, name + ".py")
        assert os.path.exists(p), f"missing scripts/{name}.py"


# ── Run all checks ────────────────────────────
if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
