#!/usr/bin/env python3
"""Validate every canonical bank answer against its own grader.

Reuses the backend's run_function_tests / run_code so the verdict matches what
the live grader would say. Writes a report JSON and prints a summary.

Usage (from backend dir, with venv):
    cd This-Directory-Only/backend
    .venv/bin/python ../../scripts/solution_build/validate_solutions.py
Optional arg: a JSON list of ids to restrict to (for repair re-runs).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2] / "This-Directory-Only" / "backend"
sys.path.insert(0, str(BACKEND))

from app.code_runner import run_code, run_function_tests  # noqa: E402

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "dd_questions.json"
REPORT = HERE / "validation_report.json"


def validate_one(q: dict) -> dict:
    """Return {id, ok, mode, detail}. ok=True means the canonical answer is
    correct under the same checks the live grader applies."""
    code = q["full_solution"] or q["answer_code"]
    task_type = q.get("task_type")
    expected = (q.get("expected_output") or "").strip()
    # stdout-prediction path: exact stdout match
    if task_type == "stdout_prediction" and expected and not q.get("supports_visual_output"):
        res = run_code(code, timeout=8)
        actual = res.stdout.strip()
        ok = actual == expected
        return {"id": q["id"], "ok": ok, "mode": "stdout",
                "detail": "" if ok else f"actual={actual!r} expected={expected!r} err={res.stderr.strip()[:300]}"}
    # function path: run the stored test_cases
    if q.get("submission_mode") == "function" and q.get("test_cases"):
        results, execution = run_function_tests(code, q["test_cases"])
        ok = bool(results) and all(r.passed for r in results)
        if ok:
            return {"id": q["id"], "ok": True, "mode": "function", "detail": ""}
        fails = [{"actual": r.actual, "expected": r.expected, "error": r.error}
                 for r in results if not r.passed]
        return {"id": q["id"], "ok": False, "mode": "function",
                "detail": json.dumps(fails)[:500] + (f" | stderr={execution.stderr.strip()[:200]}" if execution.stderr.strip() else "")}
    # no deterministic check available (AI-judge questions) — run for errors only
    res = run_code(code, timeout=8)
    ok = res.success
    return {"id": q["id"], "ok": ok, "mode": "run-only",
            "detail": "" if ok else res.stderr.strip()[:300]}


def main() -> None:
    rows = json.loads(QUESTIONS.read_text())
    only = set(json.loads(sys.argv[1])) if len(sys.argv) > 1 else None
    if only:
        rows = [r for r in rows if r["id"] in only]
    out = [validate_one(q) for q in rows]
    passed = [r for r in out if r["ok"]]
    failed = [r for r in out if not r["ok"]]
    REPORT.write_text(json.dumps(out, indent=2))
    print(f"validated {len(out)}: PASS {len(passed)}  FAIL {len(failed)}")
    from collections import Counter
    print("by mode:", dict(Counter(r["mode"] for r in out)))
    if failed:
        print("FAIL ids:", [r["id"] for r in failed])
        for r in failed[:15]:
            print(f"  #{r['id']} [{r['mode']}] {r['detail'][:200]}")


if __name__ == "__main__":
    main()
