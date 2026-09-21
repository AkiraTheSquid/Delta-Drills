"""watch.py — health checks for typesafe_difficulty

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL. Pure structure and
offline invariants; nothing here calls the TypeSafe API.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def check_imports():
    import bank  # noqa: F401
    import judge  # noqa: F401


def check_public_api():
    import bank, judge
    assert callable(bank.load_problems) and callable(bank.group_by_concept)
    assert hasattr(judge.Judge, "p_harder")
    assert set(judge.FRAMINGS) == {"fwd", "rev"}, "both framings are what cancels the bias"


def check_invariants():
    import bank, judge
    problems = bank.load_problems()
    groups = bank.group_by_concept(problems)
    assert sum(len(g) for g in groups.values()) == len(problems), "every drill in exactly one concept"
    assert all(p.question and p.answer for p in problems), "a drill with no prompt or solution cannot be judged"
    # The two framings must cancel: asking the opposite and getting the opposite
    # answer lands on the same combined probability.
    c = judge.Judge._combine
    assert abs(c(0.8, 0.2) - 0.8) < 1e-9
    assert abs(c(0.3, 0.7) - 0.3) < 1e-9
    assert abs(c(0.9, 0.9) - 0.5) < 1e-9, "contradictory framings average to a coin flip"


if __name__ == "__main__":
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    print("PASS typesafe_difficulty")
