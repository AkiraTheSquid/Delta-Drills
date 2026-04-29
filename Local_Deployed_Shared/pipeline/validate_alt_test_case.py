#!/usr/bin/env python3
"""validate_alt_test_case.py — validate one hand-crafted alt test_case.

Loads the question by ID from questions_full.json, synthesizes the canonical
solution from starter_code + the question's primary test_case, then runs the
canonical against the supplied alt setup_code/call/expected_expr.

Usage:
    python3 validate_alt_test_case.py <id> --setup-code "<code>"
    python3 validate_alt_test_case.py <id> --setup-code "<code>" \\
        --call "solve()" --expected-expr "<expr>"

If --call or --expected-expr are omitted, they default to the question's
primary test_case values (most alts only need to vary setup_code).

Exit 0: canonical passed on the alt fixture; the JSONL record is printed to
stdout in the format used by function_mode_test_cases_extra.jsonl.
Exit 1: alt rejected; reason is printed to stderr.
Exit 2: invalid input (id not found, missing test_cases, etc.).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from delta_paths import THIS_DIR_ONLY, get_backend_python  # noqa: E402
from validate_function_bank import synthesize_solution_code  # noqa: E402

QUESTIONS_PATH = THIS_DIR_ONLY / "questions_full.json"


def load_code_runner():
    path = THIS_DIR_ONLY / "backend" / "app" / "code_runner.py"
    spec = importlib.util.spec_from_file_location("delta_code_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    backend_python = get_backend_python()
    if backend_python.exists():
        module.sys.executable = str(backend_python)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question_id", type=int)
    ap.add_argument("--setup-code", required=True)
    ap.add_argument("--call", default=None)
    ap.add_argument("--expected-expr", default=None)
    args = ap.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    q = next((x for x in questions if x.get("id") == args.question_id), None)
    if q is None:
        print(f"id={args.question_id} not found", file=sys.stderr)
        return 2
    tcs = q.get("test_cases") or []
    if not tcs:
        print(f"id={args.question_id} has no test_cases", file=sys.stderr)
        return 2
    primary = tcs[0]
    starter = q.get("starter_code", "") or ""
    alt_tc = {
        "setup_code": args.setup_code,
        "call": args.call if args.call is not None else primary.get("call", "solve()"),
        "expected_expr": args.expected_expr if args.expected_expr is not None else primary.get("expected_expr", ""),
    }

    code_runner = load_code_runner()
    solution = synthesize_solution_code(starter, [primary])
    try:
        results, execution = code_runner.run_function_tests(solution, [alt_tc])
    except Exception as exc:
        print(f"harness error: {exc}", file=sys.stderr)
        return 1
    if not results:
        print("no results returned", file=sys.stderr)
        return 1
    r = results[0]
    if not r.passed:
        msg = getattr(r, "error", None) or getattr(r, "diff_summary", None) or "canonical rejected alt"
        print(f"alt rejected: {msg}", file=sys.stderr)
        return 1

    record = {"id": args.question_id, "test_cases_extra": [alt_tc]}
    print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
