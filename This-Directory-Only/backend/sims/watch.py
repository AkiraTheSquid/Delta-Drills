"""watch.py — health checks for sims

The simulator's numbers are only worth reading while its FSRS copy is the
app's FSRS. These checks pin that (weights + one update chain) and that the
graph still loads with its 27 ARENA goals. The full parity/CRN/sanity suite
is `test_sim.py`.
Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault("USER_DATA_DIR", tempfile.mkdtemp(prefix="sims_watch_"))


def check_imports():
    from sims import beliefs, fsrs_vec, run, sim, world  # noqa: F401


def check_public_api():
    from sims import sim, world
    assert callable(sim.run) and callable(world.load_graph)
    assert set(sim.ARMS) >= {"C", "B", "A0", "A", "H", "O"}, sim.ARMS


def check_invariants():
    from app import memory_model as M
    from sims import fsrs_vec as F, world
    assert F.CONCEPT_PRIOR_WEIGHTS == M.CONCEPT_PRIOR_WEIGHTS, "FSRS weights drifted from memory_model"
    m, s = None, None
    for t, g in ((0.0, F.GOOD), (2.0, F.GOOD), (9.0, F.AGAIN), (30.0, F.GOOD)):
        m = M.review(m, t, g, M.DEFAULT_CONFIG)
        s = F.review(s, t, g, F.CONCEPT_PRIOR_WEIGHTS)
    assert abs(m.S - s[0]) < 1e-9 and abs(m.D - s[1]) < 1e-9, "FSRS update rules drifted"
    g = world.load_graph()
    assert int(g.goal.sum()) == 27, f"{int(g.goal.sum())} ARENA goals (expected 27)"


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
