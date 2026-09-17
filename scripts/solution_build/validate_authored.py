#!/usr/bin/env python3
"""Validate agent-authored solution_code against each question's grader.

Reads an authored JSONL (one obj per line: {id, solution_code, explanation,
hint}) and checks every solution_code with the SAME oracle the live grader
uses:

  * function-mode (has test_cases): run_function_tests — seeds 0, runs the
    `call`, compares to expected_expr via _delta_equal. This is the
    self-consistent correctness anchor (independent of the stale literal
    expected_output string, which for RNG questions came from a different
    seed).
  * stdout-mode: the code must run cleanly; if a deterministic expected_output
    is present AND contains no float/array noise we also require an exact
    match, otherwise clean-exit is the bar.

Prints a JSON summary and writes <authored>.report.json next to the input.
Exit code 0 iff all PASS — so an agent can loop until green.

Usage (from backend dir, venv):
    .venv/bin/python ../../scripts/solution_build/validate_authored.py <authored.jsonl>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2] / "This-Directory-Only" / "backend"
sys.path.insert(0, str(BACKEND))

from app.code_runner import run_code, run_function_tests  # noqa: E402

HERE = Path(__file__).resolve().parent
QUESTIONS = {q["id"]: q for q in json.loads((HERE / "dd_questions.json").read_text())}


def validate(qid: int, solution_code: str) -> dict:
    q = QUESTIONS.get(qid)
    if q is None:
        return {"id": qid, "ok": False, "detail": "unknown id"}
    if not solution_code or not solution_code.strip():
        return {"id": qid, "ok": False, "detail": "empty solution_code"}

    if q.get("submission_mode") == "function" and q.get("test_cases"):
        results, execution = run_function_tests(solution_code, q["test_cases"])
        ok = bool(results) and all(r.passed for r in results)
        if ok:
            return {"id": qid, "ok": True, "detail": ""}
        fails = [{"actual": r.actual, "expected": r.expected, "error": r.error}
                 for r in results if not r.passed]
        det = json.dumps(fails)[:600]
        if execution.stderr.strip():
            det += " | stderr=" + execution.stderr.strip()[:300]
        return {"id": qid, "ok": False, "detail": det}

    # stdout / other: must run cleanly
    res = run_code(solution_code, timeout=8)
    if not res.success:
        return {"id": qid, "ok": False, "detail": "RUNTIME: " + res.stderr.strip()[:400]}
    expected = (q.get("expected_output") or "").strip()
    actual = res.stdout.strip()
    # only enforce exact match when output looks deterministic (no float/array noise)
    deterministic = expected and not any(c in expected for c in ".[]()") and expected.replace("-", "").replace(" ", "").isalnum()
    if deterministic and actual != expected:
        return {"id": qid, "ok": False, "detail": f"stdout mismatch actual={actual!r} expected={expected!r}"}
    return {"id": qid, "ok": True, "detail": ""}


def main() -> None:
    src = Path(sys.argv[1])
    items = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    out = [validate(it["id"], it.get("solution_code", "")) for it in items]
    failed = [r for r in out if not r["ok"]]
    (src.parent / (src.stem + ".report.json")).write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "total": len(out),
        "pass": len(out) - len(failed),
        "fail": len(failed),
        "fail_ids": [r["id"] for r in failed],
        "fail_detail": {r["id"]: r["detail"][:300] for r in failed},
    }, indent=2))
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
