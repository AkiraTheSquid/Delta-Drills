"""watch.py — health checks for the typesafe_difficulty data directory.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def check_imports():
    pass  # data directory; nothing to import


def check_public_api():
    pass  # data directory; no API


def check_invariants():
    ratings = os.path.join(HERE, "ratings.csv")
    cache = os.path.join(HERE, "cache.jsonl")
    if os.path.exists(ratings):
        assert os.path.exists(cache), "ratings.csv without cache.jsonl = a dry-run artefact, not real judgments"
        with open(ratings) as fh:
            for r in csv.DictReader(fh):
                p = float(r["p_harder"])
                assert 0.0 <= p <= 1.0, f"p_harder out of range for q{r['question_id']}"


if __name__ == "__main__":
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    print("PASS typesafe_difficulty data")
