#!/usr/bin/env python3
"""Hardened question-bank auditor for Delta Drills.

Catches defect classes the shipping validator (validate_function_bank.py)
does not:

  starter_syntax        starter_code does not ast.parse (any submission mode)
  precompute_leak       a variable computed by test setup_code already equals
                        the expected value before solve() runs
  grading_gameable      `def solve(): return <leaked_var>` passes the REAL
                        grading harness (run_function_tests) — confirmed cheat
  degenerate_expected   expected value is None / empty, so a placeholder
                        starter passes
  todo_answer_leak      a TODO comment in starter_code spells out the graded
                        expression

Unlike validate_function_bank.py this script never writes
function_mode_broken_ids.json (that file silently EXCLUDES ids from the
exported bank — see docs/alex-feedback notes). Reports go to
chatgpt/bank_audit_report.json only.

Usage:
  python3 audit_question_bank.py           # fast in-process scan
  python3 audit_question_bank.py --confirm # + run_function_tests cheat confirm
  python3 audit_question_bank.py --gate    # --confirm + exit 1 on confirmed defects
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import importlib.util
import io
import json
import re
import sys
from pathlib import Path

_sys_path_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_sys_path_root))

from delta_paths import THIS_DIR_ONLY, get_backend_python, get_chatgpt_runtime_dir

QUESTIONS_PATH = THIS_DIR_ONLY / "questions_full.json"
REPORT_PATH = get_chatgpt_runtime_dir() / "bank_audit_report.json"

_TORCH_RE = re.compile(r"(?m)^\s*(?:import\s+torch\b|from\s+torch[\s.])")


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
    # Enable the fork runner so torch questions confirm against the real
    # grading path (same as prod, where torch preloads at app startup).
    module.preload_torch()
    return module


def values_equal(a, b) -> bool:
    """Mirror of the grading harness's _delta_equal (+ torch, which the
    harness never sees but the in-process scan does)."""
    import numpy as np

    torch = sys.modules.get("torch")
    if torch is not None and (isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor)):
        try:
            return bool(torch.equal(torch.as_tensor(a), torch.as_tensor(b)))
        except Exception:
            return False
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(a), np.asarray(b)))
        except Exception:
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(values_equal(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:
        return False


_CODE_PREAMBLE: str | None = None


def base_namespace() -> dict:
    """Namespace seeded with the grading harness's own CODE_PREAMBLE
    (numpy, einops/einsum shims, /delta_numbers.npy redirect, arr preload)."""
    global _CODE_PREAMBLE
    if _CODE_PREAMBLE is None:
        _CODE_PREAMBLE = load_code_runner().CODE_PREAMBLE
    ns: dict = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(_CODE_PREAMBLE, ns)
    return ns


def exec_seeded(code: str, ns: dict) -> None:
    """Exec code the way the grading harness does: seeded, stdout swallowed."""
    import numpy as np

    np.random.seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, ns)


def snapshot_value(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.copy()
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def check_starter_syntax(question: dict) -> list[dict]:
    starter = question.get("starter_code") or ""
    if not starter.strip():
        return []
    try:
        ast.parse(starter)
        return []
    except SyntaxError as exc:
        return [{"check": "starter_syntax", "detail": f"{exc.msg} (line {exc.lineno})"}]


def check_todo_answer_leak(question: dict) -> list[dict]:
    """TODO comments that hand over the graded expression."""
    starter = question.get("starter_code") or ""
    answer_blob = " ".join(
        filter(None, [question.get("answer_code") or ""] +
               [c.get("expected_expr") or "" for c in question.get("test_cases") or []])
    )
    # calls to a function the learner is asked to implement are spec, not leak
    own_defs = set(re.findall(r"(?m)^\s*def\s+(\w+)", starter))
    findings = []
    for line in starter.splitlines():
        if "TODO" not in line:
            continue
        comment = line.split("TODO", 1)[1]
        # answer-bearing = the comment contains a >=10-char chunk of the
        # graded expression (call chains, kwargs), not just a variable name
        for token in re.findall(r"[\w.]+\([^)]*\)|[\w.]+\s*=\s*[^#]+", comment):
            token = token.strip()
            called = token.split("(", 1)[0].split(".")[-1]
            if called in own_defs:
                continue
            if len(token) >= 10 and token in answer_blob:
                findings.append({
                    "check": "todo_answer_leak",
                    "detail": f"TODO comment contains graded expression chunk: {token!r}",
                })
                break
    return findings


def scan_function_question(question: dict) -> list[dict]:
    """In-process precompute-leak + degenerate-expected scan for one question.

    Mirrors the grading harness's exact execution order:
    starter (user_code) -> seed+setup_code -> [cheat would return a var here]
    -> seed+expected_setup_code -> eval expected_expr, all in ONE namespace.
    """
    import numpy as np

    findings: list[dict] = []
    cases = question.get("test_cases") or []
    if not cases:
        return findings
    case = cases[0]
    starter = question.get("starter_code") or ""
    setup = case.get("setup_code") or ""
    expected_setup = case.get("expected_setup_code") or setup
    expected_expr = case.get("expected_expr") or ""
    if not expected_expr:
        return findings
    blob = "\n".join([starter, setup, expected_setup, expected_expr])
    uses_torch = bool(_TORCH_RE.search(blob)) or "torch." in blob
    if uses_torch:
        try:
            import torch  # noqa: F401 — available locally; only the prod sandbox refuses it
        except ImportError:
            findings.append({"check": "torch_unavailable",
                             "detail": "torch not importable here — question unscanned"})
            return findings

    ns = base_namespace()
    preamble_ids = {name: id(value) for name, value in ns.items()}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(starter, ns)
        if setup:
            exec_seeded(setup, ns)
    except Exception as exc:
        findings.append({"check": "setup_exec_error",
                         "detail": f"{type(exc).__name__}: {exc}"})
        return findings

    # what a bare `return <var>` cheat would see (deep-copied: expected_setup
    # may mutate the same arrays in place). Preamble names are skipped unless
    # the starter/setup reassigned them (e.g. `arr = np.load(...)` fixtures).
    cheat_values = {
        name: snapshot_value(value)
        for name, value in ns.items()
        if name not in ("solve", "__builtins__") and not name.startswith("_")
        and preamble_ids.get(name) != id(value)
        and not (callable(value) and not hasattr(value, "shape"))
    }

    try:
        if expected_setup:
            exec_seeded(expected_setup, ns)
        with contextlib.redirect_stdout(io.StringIO()):
            expected_value = eval(expected_expr, ns)
    except Exception as exc:
        findings.append({"check": "expected_eval_error",
                         "detail": f"{type(exc).__name__}: {exc}"})
        return findings

    if expected_value is None:
        findings.append({"check": "degenerate_expected",
                         "detail": "expected value is None — placeholder starter passes"})
        return findings

    expr_names = set(re.findall(r"\b([A-Za-z_]\w*)\b", expected_expr))
    for name, value in cheat_values.items():
        if values_equal(value, expected_value):
            scalar = np.isscalar(expected_value) or isinstance(
                expected_value, (bool, int, float, complex, str, np.generic))
            findings.append({
                "check": "precompute_leak_scalar" if scalar else "precompute_leak",
                "var": name,
                "detail": f"post-setup var {name!r} already equals the expected value",
            })
            if name in expr_names and expected_expr.strip() != name:
                # expected_expr transforms this var yet the result is
                # unchanged — the exercise's operation is a no-op (e.g. an
                # identity rearrange pattern). Needs re-authoring, not a
                # fixture/derivation split.
                findings.append({
                    "check": "identity_expected", "var": name,
                    "detail": f"expected_expr is a no-op on {name!r}: {expected_expr[:80]!r}",
                })
    return findings


def confirm_gameable(code_runner, question: dict, leaked_vars: list[str]) -> list[dict]:
    """Run the REAL grading harness with a bare-return cheat per leaked var."""
    findings = []
    cases = question.get("test_cases") or []
    for var in leaked_vars:
        cheat = f"def solve():\n    return {var}\n"
        try:
            results, _execution = code_runner.run_function_tests(cheat, cases)
        except Exception as exc:
            findings.append({"check": "confirm_error", "var": var,
                             "detail": f"{type(exc).__name__}: {exc}"})
            continue
        if results and all(r.passed for r in results):
            findings.append({"check": "grading_gameable", "var": var,
                             "detail": f"`def solve(): return {var}` passes the grading harness"})
    return findings


def check_stdout_expected(code_runner, question: dict) -> list[dict]:
    """For questions that ACTUALLY grade by stdout compare (stdout_prediction,
    non-visual, and no function test_cases to take precedence), the stored
    expected_output must equal what answer_code prints under the real harness
    (preamble + seed). The CSV-era 'Output' column was captured from unseeded
    runs — 85/274 stored strings were unreachable by any honest solution
    (found 2026-07-06, tester's id-24). The exporter now recomputes them;
    this check keeps the contract from regressing."""
    if question.get("task_type") != "stdout_prediction":
        return []
    if question.get("supports_visual_output"):
        return []
    if question.get("submission_mode") == "function" and question.get("test_cases"):
        return []  # function tests grade this question; stored string is display-only
    answer = (question.get("answer_code") or "").strip()
    if not answer:
        return []
    if code_runner.code_uses_torch(answer):
        # Skip torch answers here UNCONDITIONALLY. Prod's stdout branch
        # recomputes expected live (fork runner), so the stored string is
        # never graded against — and raw run_code-with-torch inside the
        # audit process hung the fork runner during deploy 2026-07-06
        # (4×20s timeouts → spurious GATE FAIL; the venv python HAS torch,
        # so a torch_available() guard doesn't help). Non-blocking finding
        # mirrors torch_unavailable elsewhere.
        return [{"check": "torch_unavailable",
                 "detail": "stdout-graded torch answer — prod recomputes expected live; not audited here"}]
    result = code_runner.run_code(answer, timeout=20)
    actual = result.stdout.strip()
    if not actual:
        return [{"check": "answer_exec_error",
                 "detail": (result.stderr.strip().splitlines() or ["no stdout"])[-1][:120]}]
    stored = (question.get("expected_output") or "").strip()
    if actual != stored:
        return [{"check": "stdout_expected_stale",
                 "detail": f"stored expected differs from harness run of answer_code (stored {stored[:40]!r} vs run {actual[:40]!r})"}]
    return []


def audit(confirm: bool) -> dict:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    code_runner = load_code_runner() if confirm else None
    report: dict = {"total": len(questions), "questions": {}}

    for question in questions:
        qid = question["id"]
        findings = check_starter_syntax(question)
        findings += check_todo_answer_leak(question)
        if confirm:
            findings += check_stdout_expected(code_runner, question)
        if question.get("submission_mode") == "function":
            findings += scan_function_question(question)
            if confirm:
                leaked = [f["var"] for f in findings
                          if f["check"] in ("precompute_leak", "precompute_leak_scalar")]
                if leaked:
                    # torch confirms too: load_code_runner preloads torch, so
                    # the harness grades torch via the fork runner like prod.
                    findings += confirm_gameable(code_runner, question, leaked)
        if findings:
            report["questions"][str(qid)] = findings

    counts: dict[str, int] = {}
    for flist in report["questions"].values():
        for f in flist:
            counts[f["check"]] = counts.get(f["check"], 0) + 1
    report["counts"] = counts
    return report


BLOCKING_CHECKS = {"starter_syntax", "grading_gameable", "degenerate_expected",
                   "expected_eval_error", "setup_exec_error", "identity_expected",
                   "torch_unconfirmable", "stdout_expected_stale", "answer_exec_error"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true",
                        help="confirm precompute leaks against the real grading harness")
    parser.add_argument("--gate", action="store_true",
                        help="--confirm + exit 1 if any blocking defect is confirmed")
    args = parser.parse_args()
    confirm = args.confirm or args.gate

    report = audit(confirm=confirm)
    REPORT_PATH.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print(f"Audited {report['total']} questions -> {REPORT_PATH}")
    for check, count in sorted(report["counts"].items()):
        print(f"  {check}: {count}")

    if args.gate:
        blocking = sorted(
            int(qid) for qid, flist in report["questions"].items()
            if any(f["check"] in BLOCKING_CHECKS for f in flist)
        )
        if blocking:
            print(f"GATE FAIL — {len(blocking)} questions with blocking defects: {blocking}")
            sys.exit(1)
        print("GATE PASS")


if __name__ == "__main__":
    main()
