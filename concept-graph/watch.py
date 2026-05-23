"""watch.py — top-level health check for concept-graph/

Runs all four validators / regenerators as subprocesses:
  1. derive_prereqs.py — regenerate vocab/prereqs.json from the two source files
  2. validate.py — atom-ref check across exercises/
  3. validate_concept.py — schema + per-kind DAG check on concept_edges + prereq_manual
  4. validate_graph.py — DAG check on the generated prereqs.json
"""
import sys
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "scripts")
SOURCE_FILES = ("atoms.json", "concept_edges.json", "prereq_manual.json")


def _run(script):
    r = subprocess.run([sys.executable, os.path.join(SCRIPTS, script)],
                       capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


# ── Import checks ──────────────────────────────
def check_imports():
    for f in SOURCE_FILES:
        assert os.path.exists(os.path.join(HERE, "vocab", f)), f"missing vocab/{f}"
    sys.path.insert(0, SCRIPTS)
    try:
        import export_graphml  # noqa: F401
        import validate  # noqa: F401
        import validate_concept  # noqa: F401
        import validate_graph  # noqa: F401
        import derive_prereqs  # noqa: F401
    finally:
        sys.path.pop(0)


# ── Public API checks ─────────────────────────
def check_public_api():
    code, out, err = _run("validate.py")
    assert code == 0, f"validate.py failed:\n{out}\n{err}"
    code, out, err = _run("validate_concept.py")
    assert code == 0, f"validate_concept.py failed:\n{out}\n{err}"


# ── Invariant checks ──────────────────────────
def check_invariants():
    code, out, err = _run("derive_prereqs.py")
    assert code == 0, f"derive_prereqs.py failed (cycle in derived prereqs?):\n{out}\n{err}"
    code, out, err = _run("validate_graph.py")
    assert code == 0, f"validate_graph.py failed:\n{out}\n{err}"


# ── Run all checks ────────────────────────────
if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
