#!/usr/bin/env python3

from __future__ import annotations

import ast
import builtins
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
THIS_DIR_ONLY = REPO_DIR / "This-Directory-Only"
QUESTIONS_PATH = THIS_DIR_ONLY / "questions_full.json"
FAILURES_PATH = THIS_DIR_ONLY / "chatgpt" / "function_mode_validation_failures.jsonl"
BROKEN_IDS_PATH = THIS_DIR_ONLY / "chatgpt" / "function_mode_broken_ids.json"
BACKEND_PYTHON = THIS_DIR_ONLY / "backend" / ".venv" / "bin" / "python3"
ALLOWED_GLOBAL_NAMES = {
    "np",
    "einops",
    "einsum",
    "rearrange",
    "reduce",
    "repeat",
    "display_array_as_img",
    "solve",
    "True",
    "False",
    "None",
    *dir(builtins),
}


def load_code_runner():
    path = THIS_DIR_ONLY / "backend" / "app" / "code_runner.py"
    spec = importlib.util.spec_from_file_location("delta_code_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if BACKEND_PYTHON.exists():
        module.sys.executable = str(BACKEND_PYTHON)
    return module


def synthesize_solution_code(starter_code: str, test_cases: list[dict]) -> str:
    if not test_cases:
        body = ["return None"]
    else:
        primary_case = test_cases[0]
        body: list[str] = []
        expected_setup = (primary_case.get("expected_setup_code") or "").strip()
        setup_code = (primary_case.get("setup_code") or "").strip()
        if expected_setup and expected_setup != setup_code:
            body.extend(line for line in expected_setup.splitlines() if line.strip())
        expected_expr = (primary_case.get("expected_expr") or "").strip()
        body.append(f"return {expected_expr}" if expected_expr else "return None")

    replacement = "\n".join(f"    {line}" for line in body)
    placeholder_pattern = re.compile(
        r"(?m)^([ \t]*)# Write your solution here[^\n]*\n\1return None$"
    )
    if placeholder_pattern.search(starter_code):
        return placeholder_pattern.sub(replacement, starter_code, count=1)

    lines = starter_code.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("# Write your solution here"):
            indent = re.match(r"^\s*", line).group(0)
            repl = [indent + l for l in body]
            lines[i : i + 1] = repl
            return "\n".join(lines)
    return starter_code


def _extract_defined_names(code: str) -> set[str]:
    if not code.strip():
        return set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(re.findall(r"\b([A-Za-z_]\w*)\s*=", code))

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_extract_target_names(target))
        elif isinstance(node, ast.AnnAssign):
            names.update(_extract_target_names(node.target))
        elif isinstance(node, ast.For):
            names.update(_extract_target_names(node.target))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _extract_target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for elt in node.elts:
            names.update(_extract_target_names(elt))
        return names
    return set()


def _extract_referenced_names(expr: str) -> set[str]:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return set(re.findall(r"\b([A-Za-z_]\w*)\b", expr))
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _validate_test_case(code_runner, starter_code: str, case: dict) -> tuple[list[str], list[dict]]:
    reasons: list[str] = []
    details: list[dict] = []

    setup_code = case.get("setup_code", "") or ""
    expected_setup_code = case.get("expected_setup_code", "") or ""
    expected_expr = case.get("expected_expr", "") or ""

    for label, code in (
        ("setup_code", setup_code),
        ("expected_setup_code", expected_setup_code),
    ):
        if code:
            try:
                ast.parse(code)
            except SyntaxError as exc:
                reasons.append(f"{label}_syntax_error")
                details.append({"error": f"{label} syntax error: {exc}"})

    if expected_expr:
        try:
            ast.parse(expected_expr, mode="eval")
        except SyntaxError as exc:
            reasons.append("expected_expr_syntax_error")
            details.append({"error": f"expected_expr syntax error: {exc}"})

    available_names = (
        ALLOWED_GLOBAL_NAMES
        | _extract_defined_names(starter_code)
        | _extract_defined_names(setup_code)
        | _extract_defined_names(expected_setup_code)
    )
    referenced_names = _extract_referenced_names(expected_expr)
    undefined = sorted(name for name in referenced_names if name not in available_names)
    if undefined:
        reasons.append("expected_expr_undefined_names")
        details.append(
            {
                "error": f"expected_expr uses undefined names: {undefined}",
                "expected": expected_expr,
            }
        )

    if reasons:
        return reasons, details

    harness = (
        f"{starter_code}\n"
        f"{setup_code}\n"
        f"{expected_setup_code}\n"
        "_delta_expected_value = None\n"
        f"_delta_expected_value = eval({expected_expr!r}, globals())\n"
        "print('__DELTA_EXPECTED_OK__')\n"
    )
    execution = code_runner.run_code(harness)
    if "__DELTA_EXPECTED_OK__" not in execution.stdout:
        return (
            ["expected_expr_execution_failed"],
            [
                {
                    "error": execution.stderr.strip() or execution.stdout.strip() or "expected_expr execution failed",
                    "expected": expected_expr,
                }
            ],
        )

    return [], []


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

        for case in test_cases:
            case_reasons, case_details = _validate_test_case(code_runner, starter_code, case)
            failure_reasons.extend(case_reasons)
            if case_details:
                failures.append(
                    {
                        "id": question["id"],
                        "question_text": question["question_text"],
                        "answer_code": answer_code,
                        "starter_code": starter_code,
                        "test_cases": test_cases,
                        "reasons": list(dict.fromkeys(failure_reasons)),
                        "validation_results": case_details,
                    }
                )
                break
        else:
            pass

        if failures and failures[-1]["id"] == question["id"]:
            continue

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

        solution_code = synthesize_solution_code(starter_code, test_cases)
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
    BROKEN_IDS_PATH.write_text(
        json.dumps(sorted({int(item["id"]) for item in failures}), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Validated {len(questions)} questions; failures: {len(failures)}")
    print(f"Wrote failures to {FAILURES_PATH}")
    print(f"Wrote broken IDs to {BROKEN_IDS_PATH}")


if __name__ == "__main__":
    main()
