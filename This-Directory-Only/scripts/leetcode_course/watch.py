"""watch.py — health checks for scripts/leetcode_course

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL. Offline only: no
codex / TypeSafe calls, no raw-data dependency (RAW_DIR may be absent).
"""
import json
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def check_imports():
    import build_pool, check_graph, harness, verify  # noqa: F401


def check_public_api():
    import harness
    # A design problem and a tree-argument function round-trip through the runner.
    design = {"shape": "design", "entry_point": "C", "inputs": [{"ops": ["C", "inc", "get"], "args": [[], [2], []]}]}
    code = "class C:\n    def __init__(self): self.n = 0\n    def inc(self, k): self.n += k\n    def get(self): return self.n\n"
    out, err = harness.run(design, code)
    assert out == [[None, None, 2]], (out, err)
    tree = {"shape": "function", "entry_point": "depth", "arg_types": ["tree_node"], "inputs": [[[3, 9, 20, None, None, 15, 7]]]}
    code = "def depth(r):\n    return 0 if r is None else 1 + max(depth(r.left), depth(r.right))\n"
    out, err = harness.run(tree, code)
    assert out == [3], (out, err)
    # "unordered" ignores the outer order only; "unordered_deep" also the inner lists'.
    assert harness.same([[1, 2], [3]], [[3], [1, 2]], "unordered")
    assert not harness.same([[1, 2]], [[2, 1]], "unordered")
    assert harness.same([[1, 2], [3]], [[3], [2, 1]], "unordered_deep")


def check_invariants():
    import check_graph
    kcs = json.loads(check_graph.GRAPH.read_text())["kcs"]
    bad = check_graph.findings(kcs)
    assert not bad, bad


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
