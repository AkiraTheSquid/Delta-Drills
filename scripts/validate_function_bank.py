#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = REPO_DIR / "questions.json"
FAILURES_PATH = REPO_DIR / "chatgpt" / "function_mode_validation_failures.jsonl"


def load_code_runner():
    path = REPO_DIR / "backend" / "app" / "code_runner.py"
    spec = importlib.util.spec_from_file_location("delta_code_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthesize_solution_code(starter_code: str, answer_code: str) -> str:
    steps = [line.strip() for line in answer_code.replace(";", "\n").splitlines() if line.strip()]
    if not steps:
        body = ["return None"]
    else:
        last = steps[-1]
        if "\n" not in answer_code and not any(tok in answer_code for tok in ("print(", "assert ", "def ", "return ", ";")):
            body = [f"return {answer_code.strip()}"]
        elif last.startswith("print(") and last.endswith(")"):
            body = steps[:-1] + [f"return {last[len('print('):-1].strip()}"]
        elif "=" in last and not last.startswith(("if ", "for ", "while ")):
            lhs = last.split("=", 1)[0].strip()
            body = steps + [f"return {lhs}"]
        else:
            body = steps + ["return None"]

    placeholder = "    # Write your solution here\n    return None"
    replacement = "\n".join(f"    {line}" for line in body)
    if placeholder in starter_code:
        return starter_code.replace(placeholder, replacement)

    lines = starter_code.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "# Write your solution here":
            indent = re.match(r"^\s*", line).group(0)
            repl = [indent + l for l in body]
            lines[i : i + 1] = repl
            return "\n".join(lines)
    return starter_code


def main() -> None:
    code_runner = load_code_runner()
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    failures = []

    for question in questions:
        if question.get("submission_mode") != "function":
            continue
        starter_code = question.get("starter_code") or ""
        test_cases = question.get("test_cases") or []
        answer_code = question.get("answer_code") or ""

        failure_reasons = []
        if "def solve" not in starter_code:
            failure_reasons.append("starter_code_missing_solve")
        if not test_cases:
            failure_reasons.append("missing_test_cases")

        if failure_reasons:
            failures.append(
                {
                    "id": question["id"],
                    "question_text": question["question_text"],
                    "answer_code": answer_code,
                    "starter_code": starter_code,
                    "test_cases": test_cases,
                    "reasons": failure_reasons,
                    "validation_results": [],
                }
            )
            continue

        solution_code = synthesize_solution_code(starter_code, answer_code)
        results, execution = code_runner.run_function_tests(solution_code, test_cases)
        bad = [r for r in results if not r.passed]
        if bad:
            failures.append(
                {
                    "id": question["id"],
                    "question_text": question["question_text"],
                    "answer_code": answer_code,
                    "starter_code": starter_code,
                    "test_cases": test_cases,
                    "reasons": ["canonical_solution_failed_tests"],
                    "validation_results": [
                        {
                            "actual": r.actual,
                            "expected": r.expected,
                            "error": r.error,
                        }
                        for r in bad
                    ],
                    "stdout": execution.stdout,
                    "stderr": execution.stderr,
                }
            )

    FAILURES_PATH.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    print(f"Validated {len(questions)} questions; failures: {len(failures)}")
    print(f"Wrote failures to {FAILURES_PATH}")


if __name__ == "__main__":
    main()
