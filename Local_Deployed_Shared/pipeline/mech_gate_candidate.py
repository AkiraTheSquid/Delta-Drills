#!/usr/bin/env python3
"""Deterministic mechanical gate for GENERATED question candidates.

This is the innermost gate of the Fable exercise-generation harness
(docs/fable-exercise-generation-harness.md). It runs BEFORE the LLM
validator and uses the REAL grading harness (run_function_tests), so a
candidate that passes here is known to grade correctly for the honest
solution and to reject the cheat battery.

Candidate JSON contract (one question):
  {
    "id": 24,                      # bank id being regenerated
    "question_text": "...",
    "function_name": "solve",
    "starter_code": "...",         # def solve(<params>) + example print call
    "test_cases": [{"setup_code", "call", "expected_expr",
                    "expected_setup_code"?}, ...],   # 3..6 cases
    "answer_code": "..."           # full canonical submission (passes all cases)
  }

Checks (all must pass):
  schema             required fields present, 3..6 test cases
  starter_syntax     starter_code and answer_code ast.parse
  starter_contract   solve takes >=1 parameter; starter contains an example
                     `print(solve(...))` invocation with >=1 argument
  case_contract      every case: nonempty setup_code + expected_expr, call
                     invokes solve with >=1 argument, expected_expr does not
                     reference solve
  case_self_suff     setup/expected of every case run in a fresh namespace
                     with ONLY numpy preloaded (imports live in setup_code)
  canonical_passes   answer_code passes ALL cases on the real harness
  starter_fails      starter_code passes NO case
  cheat_battery      None-return, identity-return (per arg position), and
                     every setup-variable bare return each FAIL >=1 case
  non_degenerate     expected values differ across >=2 cases

Usage:
  <backend venv python> mech_gate_candidate.py candidate.json
  <backend venv python> mech_gate_candidate.py --batch candidates.jsonl
Exit 0 = PASS (all candidates), 1 = FAIL. Failures print as one line each:
  FAIL <check>: <detail>
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

_sys_path_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_sys_path_root))

from delta_paths import THIS_DIR_ONLY  # noqa: E402

MIN_CASES, MAX_CASES = 3, 6


def load_code_runner():
    path = THIS_DIR_ONLY / "backend" / "app" / "code_runner.py"
    spec = importlib.util.spec_from_file_location("delta_code_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_call_args(call: str, fn: str) -> int | None:
    """Return positional-arg count of the solve(...) invocation, or None."""
    try:
        tree = ast.parse(call.strip(), mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name) and target.id == fn:
                return len(node.args) + len(node.keywords)
    return None


def _setup_assigned_names(setup_codes: list[str]) -> set[str]:
    names: set[str] = set()
    for code in setup_codes:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                if isinstance(node.target, ast.Name):
                    names.add(node.target.id)
    return names - {"_", "np"}


def _values_equal(a, b) -> bool:
    import numpy as np

    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(a), np.asarray(b)))
        except Exception:
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:
        return False


def gate_candidate(candidate: dict, code_runner) -> list[str]:
    import numpy as np

    failures: list[str] = []
    fn = candidate.get("function_name") or "solve"
    cases = candidate.get("test_cases") or []
    starter = candidate.get("starter_code") or ""
    answer = candidate.get("answer_code") or ""

    # --- schema ---
    for field in ("question_text", "starter_code", "test_cases", "answer_code"):
        if not candidate.get(field):
            failures.append(f"schema: missing/empty field {field!r}")
    if not MIN_CASES <= len(cases) <= MAX_CASES:
        failures.append(f"schema: {len(cases)} test cases (need {MIN_CASES}..{MAX_CASES})")
    if failures:
        return failures

    # --- syntax ---
    for label, code in (("starter_code", starter), ("answer_code", answer)):
        try:
            ast.parse(code)
        except SyntaxError as exc:
            failures.append(f"starter_syntax: {label} does not parse: {exc}")
    if failures:
        return failures

    # --- starter contract: >=1-param solve + example invocation ---
    starter_tree = ast.parse(starter)
    fn_def = next(
        (n for n in ast.walk(starter_tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn),
        None,
    )
    if fn_def is None:
        failures.append(f"starter_contract: starter does not define `def {fn}(...)`")
    else:
        n_params = (len(fn_def.args.args) + len(fn_def.args.posonlyargs)
                    + len(fn_def.args.kwonlyargs))
        if n_params < 1 and not fn_def.args.vararg:
            failures.append(f"starter_contract: `{fn}` takes no parameters — "
                            "the whole point is a parameterized stub")
    # The bank audit exec's starters directly (setup_exec_error is blocking),
    # so the starter must run clean top-to-bottom with its stub in place —
    # e.g. no attribute access on the stub's None return (bit id 113).
    try:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            exec(starter, {"np": np})
    except Exception as exc:
        failures.append(f"starter_contract: starter does not execute cleanly "
                        f"({type(exc).__name__}: {exc})")
    example_args = None
    for node in ast.walk(starter_tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == fn:
            example_args = len(node.args) + len(node.keywords)
    if example_args is None:
        failures.append(f"starter_contract: no example `{fn}(...)` invocation in starter")
    elif example_args < 1:
        failures.append(f"starter_contract: example invocation calls `{fn}()` with no arguments")

    # --- case contract ---
    max_args = 0
    for i, case in enumerate(cases):
        setup = (case.get("setup_code") or "").strip()
        call = (case.get("call") or "").strip()
        expected = (case.get("expected_expr") or "").strip()
        if not setup:
            failures.append(f"case_contract: case {i} has empty setup_code")
        if not expected:
            failures.append(f"case_contract: case {i} has empty expected_expr")
        n_args = _parse_call_args(call, fn)
        if n_args is None:
            failures.append(f"case_contract: case {i} call does not invoke `{fn}`: {call!r}")
        elif n_args < 1:
            failures.append(f"case_contract: case {i} calls `{fn}()` with no arguments")
        else:
            max_args = max(max_args, n_args)
        if expected:
            try:
                for node in ast.walk(ast.parse(expected, mode="eval")):
                    if isinstance(node, ast.Name) and node.id == fn:
                        failures.append(
                            f"case_contract: case {i} expected_expr references `{fn}`")
                        break
            except SyntaxError as exc:
                failures.append(f"case_contract: case {i} expected_expr does not parse: {exc}")
    if failures:
        return failures

    # --- case self-sufficiency + collect expected values (numpy-only ns) ---
    expected_values = []
    for i, case in enumerate(cases):
        ns: dict = {"np": np}
        try:
            np.random.seed(0)
            exec(case["setup_code"], ns)
            exp_setup = case.get("expected_setup_code")
            if exp_setup:
                np.random.seed(0)
                exec(exp_setup, ns)
            value = eval(case["expected_expr"], ns)
        except Exception as exc:
            failures.append(
                f"case_self_suff: case {i} does not run with only numpy preloaded "
                f"({type(exc).__name__}: {exc}) — put ALL other imports inside setup_code")
            continue
        if value is None:
            failures.append(f"case_self_suff: case {i} expected value is None")
        expected_values.append(value)
    if failures:
        return failures

    # --- non-degenerate: expected values must differ across cases ---
    if all(_values_equal(expected_values[0], v) for v in expected_values[1:]):
        failures.append("non_degenerate: every case expects the SAME value — "
                        "a constant-return cheat passes; vary the inputs")

    # --- real-harness runs ---
    results, execution = code_runner.run_function_tests(answer, cases)
    if not results or not all(r.passed for r in results):
        bad = [f"case {i}: {r.error or f'got {r.actual}, expected {r.expected}'}"
               for i, r in enumerate(results) if not r.passed] or [execution.stderr[:300]]
        failures.append("canonical_passes: answer_code fails the harness — " + "; ".join(bad))

    starter_results, _ = code_runner.run_function_tests(starter, cases)
    if starter_results and any(r.passed for r in starter_results):
        idx = [i for i, r in enumerate(starter_results) if r.passed]
        failures.append(f"starter_fails: the UNMODIFIED starter passes case(s) {idx}")

    cheats: list[tuple[str, str]] = [
        ("none_return", f"def {fn}(*_a, **_k):\n    return None\n")]
    for pos in range(max_args):
        cheats.append((f"identity_arg{pos}",
                       f"def {fn}(*_a, **_k):\n    return _a[{pos}]\n"))
    for var in sorted(_setup_assigned_names([c.get("setup_code") or "" for c in cases])):
        cheats.append((f"setup_var_{var}", f"def {fn}(*_a, **_k):\n    return {var}\n"))
    for name, cheat_code in cheats:
        cheat_results, _ = code_runner.run_function_tests(cheat_code, cases)
        if cheat_results and all(r.passed for r in cheat_results):
            failures.append(f"cheat_battery: cheat `{name}` passes EVERY case — "
                            "at least one case must reject it")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="candidate .json, or .jsonl with --batch")
    parser.add_argument("--batch", action="store_true", help="gate a JSONL of candidates")
    args = parser.parse_args()

    code_runner = load_code_runner()
    path = Path(args.path)
    if args.batch:
        candidates = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        candidates = [json.loads(path.read_text())]

    any_fail = False
    for cand in candidates:
        cid = cand.get("id", "?")
        failures = gate_candidate(cand, code_runner)
        if failures:
            any_fail = True
            for f in failures:
                print(f"FAIL [{cid}] {f}")
        else:
            print(f"PASS [{cid}]")
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
