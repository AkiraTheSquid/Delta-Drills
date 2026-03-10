import argparse
import asyncio
import ast
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI

from ChatGPT_batch import get_configured_model, load_api_key


CODE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = Path(os.environ.get("DELTA_CHATGPT_RUNTIME_DIR", str(CODE_DIR))).resolve()
REPO_DIR = CODE_DIR.parent
BACKEND_PYTHON = REPO_DIR / "backend" / ".venv" / "bin" / "python3"
DEFAULT_REQUESTS_PATH = RUNTIME_DIR / "function_mode_requests.jsonl"
DEFAULT_OUTPUTS_PATH = RUNTIME_DIR / "function_mode_overrides.jsonl"
DEFAULT_REJECTS_PATH = RUNTIME_DIR / "function_mode_rejected.jsonl"
DEFAULT_DELETED_IDS_PATH = RUNTIME_DIR / "function_mode_deleted_ids.json"
DEFAULT_QUALITY_CHECK_PROMPT_PATH = CODE_DIR / "function_mode_quality_check_system.txt"
DEFAULT_QUALITY_FIX_PROMPT_PATH = CODE_DIR / "function_mode_quality_fix_system.txt"
DEFAULT_SYSTEM_PROMPT = """You convert Python coding drills into function-mode exercises for an automated test harness.

Return JSON only with this schema:
{
  "id": <int>,
  "question_text": <string>,
  "function_name": "solve",
  "starter_code": <string>,
  "submission_mode": "function",
  "test_cases": [
    {
      "setup_code": <string>,
      "call": "solve()",
      "expected_expr": <string>
    }
  ]
}

Rules:
- The student must implement solve().
- starter_code must be runnable Python and include all imports plus any fixture setup needed inside solve().
- Do not put the final answer in starter_code.
- expected_expr must evaluate to the canonical answer under the same setup_code.
- solve() must return the value under test. Do not use print() as the final behavior.
- If setup_code creates fixture variables, solve() must use those fixtures instead of recreating different random data.
- expected_expr must be valid executable Python in the harness namespace, e.g. use np.array(...) not bare array(...).
- Prefer deterministic tests.
- Preserve the original question text unless a tiny cleanup is necessary.
- No markdown fences or commentary.
"""

QUALITY_REQUIREMENTS = [
    {
        "id": "question_text_function_mode",
        "requirement": "The question text must describe the function-mode task clearly and must not require the student to print the result.",
        "good_example": "Count the number of unique colors in the provided image array and return that count.",
    },
    {
        "id": "starter_code_scaffold",
        "requirement": "starter_code must define solve() exactly once, include needed imports, and leave real work for the student instead of including the final answer.",
        "good_example": "import numpy as np\\n\\ndef solve():\\n    img = np.zeros((2, 2, 3), dtype=np.uint8)\\n    # Write your solution here\\n    return None\\n\\nprint(solve())",
    },
    {
        "id": "fixture_consistency",
        "requirement": "Any fixture variables used by the tests must be consistent between starter_code and test_cases. If the tests expect named fixtures, the scaffold should clearly use the same fixture names rather than invent unrelated ones.",
        "good_example": "setup_code defines img once, and solve() uses img instead of sampling a new random array.",
    },
    {
        "id": "test_case_validity",
        "requirement": "test_cases must be executable, deterministic, and compatible with the harness. expected_expr must be valid Python in the harness namespace.",
        "good_example": "{'setup_code': 'import numpy as np\\nimg = np.zeros((2,2,3), dtype=np.uint8)', 'call': 'solve()', 'expected_expr': 'len(np.unique(img.reshape(-1, 3), axis=0))'}",
    },
    {
        "id": "no_reference_solution_leak",
        "requirement": "starter_code must not dump full reference solutions, multiple alternative solutions, or solution commentary into the student scaffold.",
        "good_example": "starter_code contains only setup plus a placeholder, not author notes or a solved implementation.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", default=str(DEFAULT_REQUESTS_PATH))
    parser.add_argument("--outputs", default=str(DEFAULT_OUTPUTS_PATH))
    parser.add_argument("--rejects", default=str(DEFAULT_REJECTS_PATH))
    parser.add_argument("--drop-failed-ids-file", default="")
    parser.add_argument("--system-prompt-file", default="")
    parser.add_argument("--quality-check-prompt-file", default=str(DEFAULT_QUALITY_CHECK_PROMPT_PATH))
    parser.add_argument("--quality-fix-prompt-file", default=str(DEFAULT_QUALITY_FIX_PROMPT_PATH))
    parser.add_argument("--max-attempts", type=int, default=1)
    return parser.parse_args()


def load_requests(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_completed_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(int(json.loads(line)["id"]))
        except Exception:
            continue
    return done


def load_code_runner():
    path = REPO_DIR / "backend" / "app" / "code_runner.py"
    spec = importlib.util.spec_from_file_location("delta_code_runner_batch", path)
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
            repl = [indent + line for line in body]
            lines[i : i + 1] = repl
            return "\n".join(lines)
    return starter_code


def validate_candidate(code_runner, source_item: dict, candidate: dict) -> tuple[bool, dict]:
    reasons: list[str] = []
    starter_code = candidate.get("starter_code") or ""
    test_cases = candidate.get("test_cases") or []

    if candidate.get("submission_mode") != "function":
        reasons.append("submission_mode_not_function")
    if candidate.get("function_name") != "solve":
        reasons.append("function_name_not_solve")
    if "def solve" not in starter_code:
        reasons.append("starter_code_missing_solve")
    if not test_cases:
        reasons.append("missing_test_cases")
    for case in test_cases:
        case_reasons, case_results = validate_test_case_fixture(code_runner, starter_code, case)
        reasons.extend(case_reasons)
        if case_results:
            return False, {
                "reasons": list(dict.fromkeys(reasons)),
                "validation_results": case_results,
                "stdout": "",
                "stderr": "",
            }

    if reasons:
        return False, {
            "reasons": reasons,
            "validation_results": [],
            "stdout": "",
            "stderr": "",
        }

    solution_code = synthesize_solution_code(starter_code, test_cases)
    results, execution = code_runner.run_function_tests(solution_code, test_cases)
    bad = [r for r in results if not r.passed]
    if bad:
        return False, {
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

    return True, {
        "reasons": [],
        "validation_results": [],
        "stdout": execution.stdout,
        "stderr": execution.stderr,
    }


ALLOWED_VALIDATION_NAMES = {
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
    *dir(__builtins__),
}


def extract_referenced_names(expr: str) -> set[str]:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return set(re.findall(r"\b([A-Za-z_]\w*)\b", expr))
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def validate_test_case_fixture(code_runner, starter_code: str, case: dict) -> tuple[list[str], list[dict]]:
    reasons: list[str] = []
    validation_results: list[dict] = []
    setup_code = case.get("setup_code", "") or ""
    expected_setup_code = case.get("expected_setup_code", "") or ""
    expected_expr = case.get("expected_expr", "") or ""

    for label, code in (("setup_code", setup_code), ("expected_setup_code", expected_setup_code)):
        if code:
            try:
                ast.parse(code)
            except SyntaxError as exc:
                reasons.append(f"{label}_syntax_error")
                validation_results.append({"error": f"{label} syntax error: {exc}"})

    if expected_expr:
        try:
            ast.parse(expected_expr, mode="eval")
        except SyntaxError as exc:
            reasons.append("expected_expr_syntax_error")
            validation_results.append({"error": f"expected_expr syntax error: {exc}"})

    available_names = (
        ALLOWED_VALIDATION_NAMES
        | _extract_defined_names(starter_code)
        | _extract_defined_names(setup_code)
        | _extract_defined_names(expected_setup_code)
    )
    undefined = sorted(name for name in extract_referenced_names(expected_expr) if name not in available_names)
    if undefined:
        reasons.append("expected_expr_undefined_names")
        validation_results.append(
            {
                "error": f"expected_expr uses undefined names: {undefined}",
                "expected": expected_expr,
            }
        )
        return reasons, validation_results

    if reasons:
        return reasons, validation_results

    execution = code_runner.run_code(
        f"{starter_code}\n"
        f"{setup_code}\n"
        f"{expected_setup_code}\n"
        f"_delta_expected_value = eval({expected_expr!r}, globals())\n"
        "print('__DELTA_EXPECTED_OK__')\n"
    )
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


def _compact_candidate(candidate: dict) -> dict:
    compact = dict(candidate)
    if "starter_code" in compact and isinstance(compact["starter_code"], str):
        compact["starter_code"] = _truncate_text(compact["starter_code"], 1800)
    if "question_text" in compact and isinstance(compact["question_text"], str):
        compact["question_text"] = _truncate_text(compact["question_text"], 1200)
    if "test_cases" in compact and isinstance(compact["test_cases"], list):
        compact["test_cases"] = compact["test_cases"][:3]
    return compact


def _truncate_text(value: str, limit: int = 1200) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...<truncated>..."


def _compact_question(item: dict) -> dict:
    compact = dict(item)
    if "answer_code" in compact and isinstance(compact["answer_code"], str):
        compact["answer_code"] = _truncate_text(compact["answer_code"], 2000)
    if "question_text" in compact and isinstance(compact["question_text"], str):
        compact["question_text"] = _truncate_text(compact["question_text"], 1200)
    return compact


def _compact_feedback(feedback: dict) -> dict:
    candidate = feedback.get("candidate") or {}
    validation = feedback.get("validation") or {}
    return {
        "attempt": feedback.get("attempt"),
        "candidate_summary": {
            "question_text": candidate.get("question_text", ""),
            "function_name": candidate.get("function_name"),
            "submission_mode": candidate.get("submission_mode"),
            "starter_code": _truncate_text(candidate.get("starter_code", ""), 1200),
            "test_cases": (candidate.get("test_cases") or [])[:2],
        },
        "validation": {
            "reasons": validation.get("reasons", []),
            "validation_results": (validation.get("validation_results") or [])[:3],
            "stdout": _truncate_text(validation.get("stdout", ""), 600),
            "stderr": _truncate_text(validation.get("stderr", ""), 600),
        },
    }


def _compact_quality_feedback(review_results: list[dict]) -> list[dict]:
    compact: list[dict] = []
    for result in review_results:
        compact.append(
            {
                "requirement_id": result.get("requirement_id"),
                "score": result.get("score"),
                "issue": _truncate_text(result.get("issue", ""), 500),
            }
        )
    return compact


def build_user_prompt(item: dict, attempt: int, feedback: dict | None) -> str:
    payload = {
        "task": "repair_or_convert_function_mode_question",
        "attempt": attempt,
        "question": _compact_question(item),
    }
    if feedback:
        payload["previous_attempt_feedback"] = _compact_feedback(feedback)
    return json.dumps(payload, ensure_ascii=False)


def build_quality_check_prompt(item: dict, candidate: dict, requirement: dict) -> str:
    payload = {
        "task": "binary_requirement_check",
        "source_question": _compact_question(item),
        "candidate_output": _compact_candidate(candidate),
        "requirement": requirement["requirement"],
        "requirement_id": requirement["id"],
        "good_example": requirement["good_example"],
    }
    return json.dumps(payload, ensure_ascii=False)


def build_quality_fix_prompt(
    item: dict,
    candidate: dict,
    failed_check: dict,
    validation: dict | None,
    fix_attempt: int,
) -> str:
    payload = {
        "task": "repair_one_failed_quality_requirement",
        "fix_attempt": fix_attempt,
        "source_question": _compact_question(item),
        "current_candidate": _compact_candidate(candidate),
        "failed_quality_check": _compact_quality_feedback([failed_check])[0],
        "programmatic_validation": _compact_feedback(
            {"candidate": candidate, "validation": validation or {}, "attempt": 0}
        )["validation"],
    }
    return json.dumps(payload, ensure_ascii=False)


def run_programmatic_quality_check(requirement_id: str, item: dict, candidate: dict) -> dict | None:
    question_text = candidate.get("question_text", "") or ""
    starter_code = candidate.get("starter_code", "") or ""
    test_cases = candidate.get("test_cases") or []

    if requirement_id == "question_text_function_mode":
        if re.search(r"\bprint the result\b", question_text, flags=re.IGNORECASE):
            return {
                "requirement_id": requirement_id,
                "score": 0,
                "issue": "The question text still tells the student to print the result instead of returning a value from solve().",
            }
        return {
            "requirement_id": requirement_id,
            "score": 1,
            "issue": "",
        }

    if requirement_id == "starter_code_scaffold":
        if starter_code.count("def solve") != 1:
            return {
                "requirement_id": requirement_id,
                "score": 0,
                "issue": "starter_code must define solve() exactly once.",
            }
        if "# Write your solution here" not in starter_code or "return None" not in starter_code:
            return {
                "requirement_id": requirement_id,
                "score": 0,
                "issue": "starter_code is missing the placeholder scaffold with '# Write your solution here' and 'return None'.",
            }
        return {
            "requirement_id": requirement_id,
            "score": 1,
            "issue": "",
        }

    if requirement_id == "fixture_consistency":
        if not test_cases:
            return {
                "requirement_id": requirement_id,
                "score": 0,
                "issue": "No test cases were provided.",
            }
        setup_names: set[str] = set()
        for case in test_cases:
            setup_names.update(_extract_defined_names(case.get("setup_code", "") or ""))
        if not setup_names:
            return {
                "requirement_id": requirement_id,
                "score": 1,
                "issue": "",
            }
        starter_names = _extract_defined_names(starter_code)
        answer_names = _extract_defined_names(item.get("answer_code", "") or "")
        if setup_names & (starter_names | answer_names):
            return {
                "requirement_id": requirement_id,
                "score": 1,
                "issue": "",
            }
        return {
            "requirement_id": requirement_id,
            "score": 0,
            "issue": f"Fixture names from setup_code are not reflected in the scaffold or canonical answer: {sorted(setup_names)}",
        }

    if requirement_id == "test_case_validity":
        if not test_cases:
            return {
                "requirement_id": requirement_id,
                "score": 0,
                "issue": "No test cases were provided.",
            }
        for case in test_cases:
            if case.get("call") != "solve()":
                return {
                    "requirement_id": requirement_id,
                    "score": 0,
                    "issue": "Each test case must call solve().",
                }
            expected_expr = case.get("expected_expr", "") or ""
            if "array(" in expected_expr and "np.array(" not in expected_expr:
                return {
                    "requirement_id": requirement_id,
                    "score": 0,
                    "issue": "expected_expr uses bare array(...) instead of valid harness Python like np.array(...).",
                }
        return {
            "requirement_id": requirement_id,
            "score": 1,
            "issue": "",
        }

    if requirement_id == "no_reference_solution_leak":
        leakage = _detect_solution_leak(item.get("answer_code", "") or "", starter_code)
        if leakage:
            return {
                "requirement_id": requirement_id,
                "score": 0,
                "issue": leakage,
            }
        return {
            "requirement_id": requirement_id,
            "score": 1,
            "issue": "",
        }

    return None


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


def _normalize_solution_lines(code: str) -> list[str]:
    lines: list[str] = []
    for raw in code.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped in {"# Write your solution here", "return None", "print(solve())"}:
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        lines.append(stripped)
    return lines


def _detect_solution_leak(answer_code: str, starter_code: str) -> str:
    starter_lines = _normalize_solution_lines(_extract_solve_body(starter_code))
    answer_lines = _normalize_solution_lines(answer_code)
    if not starter_lines or not answer_lines:
        for marker in ("# Author:", "Faster version", "canonical solution", "multiple solutions"):
            if marker.lower() in starter_code.lower():
                return f"starter_code includes solution/commentary marker: {marker}"
        return ""

    shared = [line for line in starter_lines if line in answer_lines]
    if shared:
        return f"starter_code includes canonical solution lines: {shared[:3]}"

    leak_markers = ("# Author:", "Faster version", "canonical solution", "multiple solutions")
    for marker in leak_markers:
        if marker.lower() in starter_code.lower():
            return f"starter_code includes solution/commentary marker: {marker}"
    return ""


def _extract_solve_body(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    lines = code.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "solve":
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            return "\n".join(lines[start:end])
    return code


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_deleted_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {int(x) for x in data}
    except Exception:
        pass
    return set()


def write_deleted_ids(path: Path, ids: set[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(ids), indent=2) + "\n", encoding="utf-8")


async def process_item_with_config(
    client: AsyncOpenAI,
    model: str,
    item: dict,
    lock: asyncio.Lock,
    system_prompt: str,
    quality_check_prompt: str,
    quality_fix_prompt: str,
    output_path: Path,
    rejects_path: Path,
    max_attempts: int,
    code_runner,
    deleted_ids_path: Path | None,
) -> None:
    feedback = None
    last_record = None
    last_validation = None

    for attempt in range(1, max_attempts + 1):
        response = await client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_prompt(item, attempt, feedback)},
            ],
            temperature=0,
        )
        text = getattr(response, "output_text", "").strip()
        try:
            record = json.loads(text)
        except Exception as exc:
            last_record = {"raw_output": text}
            last_validation = {
                "reasons": ["invalid_json"],
                "validation_results": [{"actual": "", "expected": "", "error": str(exc)}],
                "stdout": "",
                "stderr": "",
            }
            feedback = {
                "attempt": attempt,
                "candidate": last_record,
                "validation": last_validation,
            }
            continue

        record["id"] = int(item["id"])
        ok, validation = validate_candidate(code_runner, item, record)
        if not ok:
            last_record = record
            last_validation = validation
            feedback = {
                "attempt": attempt,
                "candidate": record,
                "validation": validation,
            }
            continue

        sequential_ok, record, validation = await repair_requirements_sequentially(
            client=client,
            model=model,
            quality_check_prompt=quality_check_prompt,
            quality_fix_prompt=quality_fix_prompt,
            item=item,
            candidate=record,
            code_runner=code_runner,
            max_attempts=max_attempts,
        )

        if sequential_ok:
            async with lock:
                append_jsonl(output_path, record)
            return

        last_record = record
        last_validation = validation
        feedback = {
            "attempt": attempt,
            "candidate": record,
            "validation": validation,
        }

    reject_record = {
        "id": int(item["id"]),
        "source_question": item,
        "last_candidate": last_record,
        "last_validation": last_validation,
        "attempts": max_attempts,
    }
    async with lock:
        append_jsonl(rejects_path, reject_record)
        if deleted_ids_path is not None:
            deleted_ids = load_deleted_ids(deleted_ids_path)
            deleted_ids.add(int(item["id"]))
            write_deleted_ids(deleted_ids_path, deleted_ids)


async def main() -> None:
    args = parse_args()
    requests_path = Path(args.requests)
    outputs_path = Path(args.outputs)
    rejects_path = Path(args.rejects)
    deleted_ids_path = Path(args.drop_failed_ids_file) if args.drop_failed_ids_file else None
    system_prompt = (
        Path(args.system_prompt_file).read_text(encoding="utf-8")
        if args.system_prompt_file
        else DEFAULT_SYSTEM_PROMPT
    )
    quality_check_prompt = Path(args.quality_check_prompt_file).read_text(encoding="utf-8")
    quality_fix_prompt = Path(args.quality_fix_prompt_file).read_text(encoding="utf-8")

    requests = load_requests(requests_path)
    completed = load_completed_ids(outputs_path)
    pending = [item for item in requests if int(item["id"]) not in completed]
    if not pending:
        print("No pending function-mode requests.")
        return

    outputs_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_path.parent.mkdir(parents=True, exist_ok=True)
    if deleted_ids_path is not None:
        deleted_ids_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = load_api_key(str(CODE_DIR))
    if not api_key:
        raise RuntimeError("No OpenAI API key configured in chatgpt/.")

    client = AsyncOpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_MODEL", get_configured_model(str(CODE_DIR))) or "gpt-4o-mini"
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(8)
    code_runner = load_code_runner()

    async def bounded(item: dict) -> None:
        async with semaphore:
            await process_item_with_config(
                client,
                model,
                item,
                lock,
                system_prompt,
                quality_check_prompt,
                quality_fix_prompt,
                outputs_path,
                rejects_path,
                args.max_attempts,
                code_runner,
                deleted_ids_path,
            )

    await asyncio.gather(*(bounded(item) for item in pending))
    successful = len(load_completed_ids(outputs_path) & {int(item["id"]) for item in pending})
    failed = len(pending) - successful
    print(f"Processed {len(pending)} function-mode requests: {successful} validated, {failed} rejected")
    print(f"Validated overrides: {outputs_path}")
    print(f"Rejected requests: {rejects_path}")


async def run_quality_checks(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    item: dict,
    candidate: dict,
) -> list[dict]:
    return [await run_single_quality_check(client, model, system_prompt, item, candidate, requirement) for requirement in QUALITY_REQUIREMENTS]


async def run_single_quality_check(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    item: dict,
    candidate: dict,
    requirement: dict,
) -> dict:
    programmatic = run_programmatic_quality_check(requirement["id"], item, candidate)
    if programmatic is not None:
        return programmatic

    response = await client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_quality_check_prompt(item, candidate, requirement)},
        ],
        temperature=0,
    )
    text = getattr(response, "output_text", "").strip()
    try:
        review = json.loads(text)
    except Exception as exc:
        review = {
            "requirement_id": requirement["id"],
            "score": 0,
            "issue": f"Invalid quality-check JSON: {exc}",
        }
    review["requirement_id"] = requirement["id"]
    review["score"] = int(review.get("score", 0))
    review["issue"] = review.get("issue", "")
    return review


async def run_quality_fix(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    item: dict,
    candidate: dict,
    failed_check: dict,
    validation: dict | None,
    fix_attempt: int,
) -> dict | None:
    response = await client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_quality_fix_prompt(item, candidate, failed_check, validation, fix_attempt),
            },
        ],
        temperature=0,
    )
    text = getattr(response, "output_text", "").strip()
    try:
        return json.loads(text)
    except Exception:
        return None


async def repair_requirements_sequentially(
    client: AsyncOpenAI,
    model: str,
    quality_check_prompt: str,
    quality_fix_prompt: str,
    item: dict,
    candidate: dict,
    code_runner,
    max_attempts: int,
) -> tuple[bool, dict, dict]:
    current = candidate
    baseline_validation = {"reasons": [], "validation_results": [], "stdout": "", "stderr": ""}

    for requirement in QUALITY_REQUIREMENTS:
        check = await run_single_quality_check(
            client,
            model,
            quality_check_prompt,
            item,
            current,
            requirement,
        )
        if int(check.get("score", 0)) == 1:
            continue

        fixed = False
        last_check = check
        for fix_attempt in range(1, max_attempts + 1):
            proposed = await run_quality_fix(
                client,
                model,
                quality_fix_prompt,
                item,
                current,
                last_check,
                baseline_validation,
                fix_attempt,
            )
            if proposed is None:
                continue
            proposed["id"] = int(item["id"])

            ok, validation = validate_candidate(code_runner, item, proposed)
            if not ok:
                baseline_validation = validation
                continue

            baseline_validation = validation
            recheck = await run_single_quality_check(
                client,
                model,
                quality_check_prompt,
                item,
                proposed,
                requirement,
            )
            if int(recheck.get("score", 0)) == 1:
                current = proposed
                fixed = True
                break
            last_check = recheck

        if not fixed:
            return False, current, {
                "reasons": ["quality_check_failed"],
                "validation_results": [last_check],
                "stdout": baseline_validation.get("stdout", ""),
                "stderr": baseline_validation.get("stderr", ""),
            }

    ok, final_validation = validate_candidate(code_runner, item, current)
    if not ok:
        return False, current, final_validation
    return True, current, final_validation


if __name__ == "__main__":
    asyncio.run(main())
