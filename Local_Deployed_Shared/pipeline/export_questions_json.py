#!/usr/bin/env python3
"""
Export the question bank to JSON.

Outputs:
  - questions.json: frontend-compatible flat bank (plus extra metadata fields)
  - questions_structured.json: richer export for extraction / curation workflows

This script now exports all supported sources, not just NumPy:
  - Export of numpy problems with outputs.csv
  - einsum_problems.csv
  - einops_problems_with_outputs.csv / einops_problems.csv
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable
# ── pipeline bootstrap ──
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from delta_paths import CSV_DIR, REPO_DIR, SHARED_DIR, THIS_DIR_ONLY, get_chatgpt_runtime_dir

CHATGPT_RUNTIME_DIR = get_chatgpt_runtime_dir()
OUT_PATH = SHARED_DIR / "questions.json"
FULL_OUT_PATH = THIS_DIR_ONLY / "questions_full.json"
STRUCTURED_OUT_PATH = SHARED_DIR / "questions_structured.json"
FUNCTION_OVERRIDES_PATH = CHATGPT_RUNTIME_DIR / "function_mode_overrides.jsonl"
DELETED_IDS_PATH = CHATGPT_RUNTIME_DIR / "function_mode_deleted_ids.json"
BROKEN_IDS_PATH = CHATGPT_RUNTIME_DIR / "function_mode_broken_ids.json"

CSV_SOURCES = [
    {
        "path": CSV_DIR / "Export of numpy problems with outputs.csv",
        "skip_rows": 2,
    },
    {
        "path": CSV_DIR / "einsum_problems.csv",
        "skip_rows": 0,
    },
    {
        "path": (
            CSV_DIR / "einops_problems_with_outputs.csv"
            if (CSV_DIR / "einops_problems_with_outputs.csv").exists()
            else CSV_DIR / "einops_problems.csv"
        ),
        "skip_rows": 0,
    },
    {
        "path": CSV_DIR / "cnn_problems.csv",
        "skip_rows": 0,
    },
]

CURATED_EXCLUDED_IDS = {9, 20, 21, 33, 39, 44, 45, 57, 88, 161, 188, 203, 221, 222, 223, 226}


def classify_difficulty(question_text: str, numeric_score: int) -> str:
    if "★★★" in question_text:
        return "hard"
    if "★★☆" in question_text:
        return "medium"
    if "★☆☆" in question_text:
        return "easy"
    if numeric_score <= 35:
        return "easy"
    if numeric_score <= 65:
        return "medium"
    return "hard"


def infer_primary_library(topic: str, question_text: str, answer_code: str) -> str:
    topic_lower = topic.lower()
    blob = f"{question_text}\n{answer_code}".lower()
    # Topic wins for numpy problems — answers may mention np.einsum without
    # being einops.einsum problems (np.einsum and einops.einsum are different).
    if topic_lower == "numpy":
        return "numpy"
    if "einops" in topic_lower or "einops" in blob:
        return "einops"
    if "einsum" in topic_lower or "einsum" in blob:
        return "einops.einsum"
    if "numpy" in topic_lower or "np." in blob:
        return "numpy"
    if "torch" in blob or "t." in blob:
        return "torch"
    return "python"


def infer_task_type(topic: str, question_text: str, answer_code: str, expected_output: str) -> str:
    topic_lower = topic.lower()
    blob = f"{question_text}\n{answer_code}".lower()
    if "display_array_as_img" in blob:
        return "image_transform"
    if topic_lower == "einops":
        if is_visual_einops_prompt(question_text, answer_code):
            return "image_transform"
        return "tensor_transform"
    if "einsum" in blob or topic_lower == "einsum":
        return "tensor_expression"
    if re.search(r"\bdef\s+\w+\s*\(", answer_code):
        return "function_impl"
    if expected_output:
        return "stdout_prediction"
    return "code_completion"


def is_visual_einops_prompt(question_text: str, answer_code: str) -> bool:
    blob = f"{question_text}\n{answer_code}".lower()
    if "display_array_as_img" in blob:
        return True
    output_shape_question = question_text.rsplit("->", 1)[-1].lower() if "->" in question_text else ""
    output_shape_answer = answer_code.rsplit("->", 1)[-1].lower() if "->" in answer_code else ""
    output_shape_blob = f"{output_shape_question}\n{output_shape_answer}"
    shape_is_image_like = (
        ("h" in output_shape_blob and "w" in output_shape_blob)
        or ("img" in blob and any(axis in output_shape_blob for axis in ("h", "w", "c")))
    )
    visual_markers = (
        "image",
        "img",
        "digit",
        "digits",
        "pixel",
        "pixels",
        "c h w",
        "h w c",
        "row",
        "column",
    )
    return shape_is_image_like and any(marker in blob for marker in visual_markers)


def infer_function_name(answer_code: str) -> str | None:
    match = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", answer_code)
    if match:
        return match.group(1)
    return None


def requires_float_fixture(answer_code: str) -> bool:
    lowered = answer_code.lower()
    return "'mean'" in lowered or '"mean"' in lowered


def extract_prompt_setup_code(question_text: str) -> str:
    markers = [
        "Use the exact inputs below and print the result.",
        "Use the exact inputs below.",
        "Given the exact inputs below and print the result.",
    ]
    tail = ""
    for marker in markers:
        idx = question_text.find(marker)
        if idx >= 0:
            tail = question_text[idx + len(marker) :].strip()
            break
    if not tail:
        return ""

    assignments = re.findall(r"([A-Za-z_]\w*\s*=\s*.*?)(?=(?:\s+[A-Za-z_]\w*\s*=)|$)", tail, flags=re.DOTALL)
    cleaned = [a.strip() for a in assignments if "=" in a]
    return "\n".join(cleaned)


def looks_like_expression(answer_code: str) -> bool:
    stripped = answer_code.strip()
    if not stripped:
        return False
    if "\n" in stripped:
        return False
    forbidden = ("print(", "assert ", "def ", "return ", ";")
    return not any(token in stripped for token in forbidden)


def split_answer_steps(answer_code: str) -> list[str]:
    normalized = answer_code.replace(";", "\n")
    return [line.strip() for line in normalized.splitlines() if line.strip()]


def is_assignment_statement(code_line: str) -> bool:
    return bool(re.match(r"^[A-Za-z_]\w*\s*=", code_line))


def extract_display_variable(answer_code: str) -> str | None:
    match = re.search(r"display_array_as_img\(([^)]+)\)", answer_code)
    if match:
        return match.group(1).strip()
    assign_match = re.search(r"\b([A-Za-z_]\w*)\s*=", answer_code)
    return assign_match.group(1) if assign_match else None


def strip_display_calls(answer_code: str) -> str:
    return "\n".join(
        line for line in answer_code.replace(";", "\n").splitlines()
        if "display_array_as_img(" not in line
    ).strip()


def infer_default_fixture_setup(question_text: str, answer_code: str, primary_library: str) -> str:
    lines: list[str] = []
    blob = f"{question_text}\n{answer_code}"
    needs_float = requires_float_fixture(answer_code)
    if primary_library in {"einops", "einops.einsum"}:
        # delta_numbers.npy is stored channels-first (N, C, H, W). Only convert
        # to (N, H, W, C) when the einops pattern's INPUT side names axes as
        # HWC-like (e.g. 'h w c', 'h w ch', 'b h w c', '(r nc) h w ch').
        # Do NOT use question_text or RHS-only matches — conversion problems
        # like "BCHW → NHWC" have HWC on the output side but need raw BCHW.
        def _input_is_hwc(code: str) -> bool:
            # HWC = h axis appears before w which appears before c (or ch),
            # even when other tokens like (two h) or (w w2) sit between them.
            for m in re.finditer(r"['\"]([^'\"]*?)\s*->\s*([^'\"]*?)['\"]", code):
                tokens = re.sub(r"[()]", " ", m.group(1)).split()
                h_idx = next((i for i, t in enumerate(tokens) if t == "h"), None)
                w_idx = next((i for i, t in enumerate(tokens) if t == "w"), None)
                c_idx = next((i for i, t in enumerate(tokens) if t in ("c", "ch")), None)
                if None in (h_idx, w_idx, c_idx):
                    continue
                if h_idx < w_idx < c_idx:
                    return True
            return False
        uses_nhwc = _input_is_hwc(answer_code)
        required_arr_batch = infer_required_arr_batch(question_text, answer_code)
        if "arr" in blob or "img" in blob:
            if required_arr_batch and required_arr_batch > 0:
                lines.append("_base_arr = np.load('/delta_numbers.npy')")
                if needs_float:
                    lines.append("_base_arr = _base_arr.astype(np.float32)")
                lines.append(f"_arr_repeats = -(-{required_arr_batch} // _base_arr.shape[0])")
                lines.append("arr = np.concatenate([_base_arr] * _arr_repeats, axis=0)")
                lines.append(f"arr = arr[:{required_arr_batch}]")
                if uses_nhwc:
                    lines.append("arr = np.moveaxis(arr, 1, -1)")
            else:
                if uses_nhwc:
                    load_expr = "np.load('/delta_numbers.npy').astype(np.float32)" if needs_float else "np.load('/delta_numbers.npy')"
                    lines.append(f"arr = np.moveaxis({load_expr}, 1, -1)")
                else:
                    load_expr = "np.load('/delta_numbers.npy').astype(np.float32)" if needs_float else "np.load('/delta_numbers.npy')"
                    lines.append(f"arr = {load_expr}")
        if "img" in blob:
            lines.append("img = arr[0]")
        if "seq" in blob:
            lines.append("seq = np.arange(2 * 12 * 3).reshape(2, 12, 3)")
        if "cls" in blob:
            if not re.search(r"\bb\b", "\n".join(lines)):
                lines.append("b = 2")
            if "t =" not in "\n".join(lines):
                lines.append("t = 4")
            lines.append("cls = np.arange(b * 3).reshape(b, 3)")
        if "weights" in blob:
            if "b =" not in "\n".join(lines):
                lines.append("b = 2")
            if "t =" not in "\n".join(lines):
                lines.append("t = 4")
            if "d =" not in "\n".join(lines):
                lines.append("d = 3")
            lines.append("weights = np.arange(b * t).reshape(b, t)")
        if "patches" in blob:
            lines.append("h = 2")
            lines.append("w = 3")
            # Pick fixture layout based on where the channel axis sits in the
            # einops pattern: '(h w) c p1 p2' → channels-first; '(h w) p1 p2 c'
            # → channels-last. Mismatched fixtures pass tests numerically but
            # are visually meaningless.
            patches_pattern_lhs = ""
            for m in re.finditer(r"['\"]([^'\"]*?)\s*->\s*[^'\"]*?['\"]", answer_code):
                lhs_clean = re.sub(r"[()]", " ", m.group(1)).split()
                if "p1" in lhs_clean and "p2" in lhs_clean and "c" in lhs_clean:
                    patches_pattern_lhs = lhs_clean
                    break
            if patches_pattern_lhs:
                c_pos = patches_pattern_lhs.index("c")
                p1_pos = patches_pattern_lhs.index("p1")
                if c_pos < p1_pos:
                    lines.append("patches = np.arange(h * w * 3 * 2 * 2).reshape(h * w, 3, 2, 2)")
                else:
                    lines.append("patches = np.arange(h * w * 2 * 2 * 3).reshape(h * w, 2, 2, 3)")
            else:
                lines.append("patches = np.arange(h * w * 2 * 2 * 3).reshape(h * w, 2, 2, 3)")
        if re.search(r"\bx\b", blob):
            lines.append("x = np.arange(2 * 12 * 8 * 8).reshape(2, 12, 8, 8)")
        if "hs" in blob:
            lines.append("hs = 2")
        if "ws" in blob:
            lines.append("ws = 2")
        if re.search(r"\bb\b", blob) and "b =" not in "\n".join(lines):
            lines.append("b = 2")
    return "\n".join(dict.fromkeys(lines))


def infer_required_arr_batch(question_text: str, answer_code: str) -> int | None:
    slice_match = re.search(r"\barr\[\s*:\s*(\d+)\s*\]", answer_code)
    if slice_match:
        return int(slice_match.group(1))
    prompt_match = re.search(r"\bfirst\s+(\d+)\s+images?\b", question_text, flags=re.IGNORECASE)
    if prompt_match:
        return int(prompt_match.group(1))
    return None


def derive_test_case(answer_code: str, question_text: str, primary_library: str) -> tuple[str, str]:
    steps = split_answer_steps(answer_code)
    if looks_like_expression(answer_code):
        setup_code = extract_prompt_setup_code(question_text) or infer_default_fixture_setup(question_text, answer_code, primary_library)
        return setup_code, answer_code.strip()

    if not steps:
        return "", "None"

    last = steps[-1]
    if last.startswith("print(") and last.endswith(")"):
        return "\n".join(steps[:-1]), last[len("print(") : -1].strip()
    if last.startswith("display_array_as_img(") and last.endswith(")"):
        return "\n".join(steps[:-1]), last[len("display_array_as_img(") : -1].strip()
    if is_assignment_statement(last) and not last.startswith(("if ", "for ", "while ")):
        lhs = last.split("=", 1)[0].strip()
        return "\n".join(steps[:-1]), lhs
    return "\n".join(steps[:-1]), last


def derive_function_payload(
    question_text: str,
    answer_code: str,
    primary_library: str,
    task_type: str | None = None,
) -> tuple[str, str, list[dict], str]:
    import_line = "import numpy as np"
    if primary_library in {"einops", "einops.einsum"}:
        import_line += "\nimport einops\nfrom einops import einsum, rearrange, reduce, repeat"

    task_type = task_type or infer_task_type("", question_text, answer_code, "")
    if task_type == "image_transform":
        fixture_setup = extract_prompt_setup_code(question_text) or infer_default_fixture_setup(
            question_text, answer_code, primary_library
        )
        visual_setup, visual_expr = derive_test_case(answer_code, question_text, primary_library)
        variable_name = extract_display_variable(answer_code)
        placeholder = (
            f"    # Write your solution here - define {variable_name}\n"
            if variable_name
            else "    # Write your solution here\n"
        )
        top_level_setup = f"{fixture_setup}\n\n" if fixture_setup else ""
        starter_code = (
            f"{import_line}\n\n"
            f"{top_level_setup}"
            "def solve():\n"
            f"{placeholder}"
            "    return None\n\n"
            "print(solve())\n"
        )
        test_case = {
            "setup_code": fixture_setup,
            "call": "solve()",
            "expected_expr": visual_expr or "None",
        }
        if visual_setup and visual_setup != fixture_setup:
            test_case["expected_setup_code"] = visual_setup
        test_cases = [test_case]
        return "solve", starter_code, test_cases, "function"

    setup_code, expected_expr = derive_test_case(answer_code, question_text, primary_library)
    if not setup_code:
        setup_code = extract_prompt_setup_code(question_text) or infer_default_fixture_setup(question_text, answer_code, primary_library)

    top_level_setup = f"{setup_code}\n\n" if setup_code else ""
    starter_code = (
        f"{import_line}\n\n"
        f"{top_level_setup}"
        "def solve():\n"
        "    # Write your solution here\n"
        "    return None\n\n"
        "print(solve())\n"
    )
    test_cases = [{"setup_code": setup_code, "call": "solve()", "expected_expr": expected_expr or "None"}]
    return "solve", starter_code, test_cases, "function"


def normalize_subtopic(topic: str, subtopic: str) -> str:
    if topic and subtopic:
        return f"{topic}: {subtopic}"
    return subtopic


def _read_jsonl_overrides(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    overrides: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        overrides[int(record["id"])] = record
    return overrides


def load_function_overrides() -> dict[int, dict]:
    """Merge all override layers in the same order as the backend.

    See ``backend/app/questions.py::_load_function_overrides`` — round-1 is
    the base, round-2/round-3 are manual repair layers, and the einops
    prompt rewrite file is layered last so its question_text rewrites win
    while preserving starter/test fields from the earlier rounds.
    """
    base = _read_jsonl_overrides(FUNCTION_OVERRIDES_PATH)
    for layer_name in (
        "function_mode_overrides_round2.jsonl",
        "function_mode_overrides_round3.jsonl",
        "einops_prompt_rewrite_overrides.jsonl",
        "numpy_einsum_prompt_rewrite_overrides.jsonl",
        "prompt_expansion_overrides.jsonl",
        "difficulty_overrides.jsonl",
        # question_text-only rewrites (de-giveaway + readability) — keep in
        # sync with backend/app/questions.py::_load_function_overrides.
        "question_text_quality_overrides.jsonl",
        # Starter-code fixes: SyntaxErrors + answer-leaking comments +
        # precomputed-answer-before-solve() bugs. Keep in sync with backend.
        "starter_leak_and_syntax_fixes.jsonl",
        # Canonical answer_code repairs: numpy-2.0 removals (ptp / np.int),
        # missing imports, truncated CSV code, wrong lookup-table bound
        # (2026-07-06 stdout-expected sweep). Layered last so these win.
        "answer_code_repairs.jsonl",
        # Parameterized regeneration (2026-07-07): full question rewrites to
        # `def solve(<real params>)` + 3-6 test cases incl. edge cases, from
        # the Fable generation pass (mech-gate verified + hand-reviewed).
        # Layered last — replaces question_text/starter/test_cases/answer_code
        # wholesale for its ids. Keep in sync with backend.
        "parameterized_regen_overrides.jsonl",
        # Einops visual triage (2026-07-07): 41 demotions to non-visual +
        # 7 duplicate-image fixture swaps. Keep in sync with backend.
        "einops_visual_fixes.jsonl",
    ):
        layer = _read_jsonl_overrides(CHATGPT_RUNTIME_DIR / layer_name)
        for qid, record in layer.items():
            merged = dict(base.get(qid, {}))
            merged.update(record)
            base[qid] = merged
    return base


def load_deleted_ids() -> set[int]:
    if not DELETED_IDS_PATH.exists():
        return set()
    try:
        data = json.loads(DELETED_IDS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if not isinstance(data, list):
        return set()
    return {int(item) for item in data}


def load_broken_ids() -> set[int]:
    if not BROKEN_IDS_PATH.exists():
        return set()
    try:
        data = json.loads(BROKEN_IDS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if not isinstance(data, list):
        return set()
    return {int(item) for item in data}


def iter_csv_rows(path: Path, skip_rows: int) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for _ in range(skip_rows):
            next(f, None)
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def load_questions() -> list[dict]:
    questions: list[dict] = []
    next_id = 1
    function_overrides = load_function_overrides()
    deleted_ids = load_deleted_ids()

    for source in CSV_SOURCES:
        path = source["path"]
        skip_rows = source["skip_rows"]

        if not path.exists():
            print(f"WARNING: CSV not found at {path} — skipping", file=sys.stderr)
            continue

        for row in iter_csv_rows(path, skip_rows):
            qid = next_id
            next_id += 1

            topic = (row.get("Topic") or "").strip()
            subtopic = (row.get("Subtopic") or "").strip()
            question_text = (row.get("Question") or "").strip()
            answer_code = (row.get("Answer") or "").strip()
            raw_difficulty = (row.get("Problem difficulty") or "0").strip()
            expected_output = (row.get("Output") or "").strip()

            if not question_text or not subtopic:
                continue
            if qid in CURATED_EXCLUDED_IDS:
                continue
            if qid in deleted_ids:
                continue

            try:
                difficulty_score = int(float(raw_difficulty))
            except ValueError:
                difficulty_score = 50

            difficulty_label = classify_difficulty(question_text, difficulty_score)
            primary_library = infer_primary_library(topic, question_text, answer_code)
            task_type = infer_task_type(topic, question_text, answer_code, expected_output)
            function_name = infer_function_name(answer_code)
            if not function_name:
                function_name, starter_code, test_cases, submission_mode = derive_function_payload(
                    question_text, answer_code, primary_library, task_type
                )
            else:
                starter_code, test_cases, submission_mode = None, [], "function"

            override = function_overrides.get(qid)
            if override:
                function_name = override.get("function_name", function_name)
                starter_code = override.get("starter_code", starter_code)
                test_cases = override.get("test_cases", test_cases)
                submission_mode = override.get("submission_mode", submission_mode)
                question_text = override.get("question_text", question_text)
                answer_code = override.get("answer_code", answer_code)
                if "difficulty_score" in override:
                    try:
                        difficulty_score = int(override["difficulty_score"])
                        difficulty_label = classify_difficulty(question_text, difficulty_score)
                    except (TypeError, ValueError):
                        pass
            expected_artifact_type = "image" if task_type == "image_transform" else "stdout"
            supports_visual_output = expected_artifact_type == "image"
            # Visual-flag overrides (mirror backend/app/questions.py): used by the
            # 2026-07-07 einops demotions — layout-only/garbage-render questions
            # flipped to ordinary function-graded drills.
            if override:
                task_type = override.get("task_type", task_type)
                if "expected_artifact_type" in override:
                    expected_artifact_type = override["expected_artifact_type"]
                    supports_visual_output = expected_artifact_type == "image"
                if "supports_visual_output" in override:
                    supports_visual_output = bool(override["supports_visual_output"])

            questions.append(
                {
                    "id": qid,
                    "topic": topic,
                    "subtopic": subtopic,
                    "subtopic_key": normalize_subtopic(topic, subtopic),
                    "question_text": question_text,
                    "answer_code": answer_code,
                    "difficulty_score": difficulty_score,
                    "difficulty_label": difficulty_label,
                    "expected_output": expected_output,
                    "language": "python",
                    "primary_library": primary_library,
                    "task_type": task_type,
                    "function_name": function_name,
                    "starter_code": starter_code,
                    "test_cases": test_cases,
                    "submission_mode": submission_mode,
                    "expected_artifact_type": expected_artifact_type,
                    "supports_visual_output": supports_visual_output,
                    "source_type": "csv",
                    "source_path": str(path.relative_to(REPO_DIR)),
                }
            )

    return questions


def build_structured_questions(flat_questions: list[dict]) -> list[dict]:
    structured: list[dict] = []
    for question in flat_questions:
        structured.append(
            {
                "id": question["id"],
                "source": {
                    "type": question["source_type"],
                    "path": question["source_path"],
                },
                "curriculum": {
                    "topic": question["topic"],
                    "subtopic": question["subtopic"],
                    "subtopic_key": question["subtopic_key"],
                    "difficulty_score": question["difficulty_score"],
                    "difficulty_label": question["difficulty_label"],
                },
                "exercise": {
                    "question_text": question["question_text"],
                    "task_type": question["task_type"],
                    "language": question["language"],
                    "primary_library": question["primary_library"],
                    "function_name": question["function_name"],
                    "starter_code": question["starter_code"],
                    "test_cases": question["test_cases"],
                    "submission_mode": question["submission_mode"],
                    "canonical_solution": question["answer_code"],
                    "expected_output": question["expected_output"],
                    "expected_artifact_type": question["expected_artifact_type"],
                    "supports_visual_output": question["supports_visual_output"],
                },
            }
        )
    return structured


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_harness():
    """Load the backend grading harness (code_runner.py) standalone — same
    trick as audit_question_bank.py; the `app` package can't import outside
    the backend venv (pydantic_settings etc.)."""
    import importlib.util

    path = THIS_DIR_ONLY / "backend" / "app" / "code_runner.py"
    spec = importlib.util.spec_from_file_location("delta_code_runner_export", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        module.preload_torch()
    except Exception:
        pass  # torch unavailable — torch questions keep their stored value
    return module


def recompute_expected_outputs(questions: list) -> None:
    """Replace each stdout-graded question's expected_output with the stdout
    its canonical answer_code ACTUALLY produces under the grading harness
    (preamble + np.random.seed(0)).

    The CSV 'Output' column was captured from unseeded, out-of-harness runs —
    85 of 274 stored strings were unreachable by ANY honest solution (found
    2026-07-06 via the tester's id-24 argmax drill). The backend now
    recomputes live at grade time; this keeps the exported bank (offline
    Pyodide grading + display strings) consistent with it. Questions whose
    answer errors under the harness keep the stored value and are reported —
    the audit gate flags them separately."""
    runner = _load_harness()
    corrected, consistent, errored = 0, 0, []
    for q in questions:
        if q.get("task_type") != "stdout_prediction" or q.get("supports_visual_output"):
            continue
        answer = (q.get("answer_code") or "").strip()
        if not answer:
            continue
        result = runner.run_code(answer, timeout=20)
        actual = result.stdout.strip()
        if not actual:
            errored.append(q["id"])
            continue
        if actual != (q.get("expected_output") or "").strip():
            corrected += 1
        else:
            consistent += 1
        q["expected_output"] = actual
    print(
        f"expected_output recompute: {corrected} corrected, {consistent} already-consistent, "
        f"{len(errored)} answer-errors kept stored value: {errored}"
    )


def main() -> None:
    questions = load_questions()
    if not questions:
        print("ERROR: no questions were exported", file=sys.stderr)
        sys.exit(1)

    recompute_expected_outputs(questions)

    broken_ids = load_broken_ids()
    student_questions = [question for question in questions if int(question["id"]) not in broken_ids]
    structured = build_structured_questions(questions)

    write_json(FULL_OUT_PATH, questions)
    write_json(OUT_PATH, student_questions)
    write_json(STRUCTURED_OUT_PATH, structured)

    print(f"Exported {len(questions)} full questions to {FULL_OUT_PATH}")
    print(f"Exported {len(student_questions)} student-facing questions to {OUT_PATH}")
    print(f"Excluded {len(questions) - len(student_questions)} validator-broken questions from {OUT_PATH}")
    print(f"Exported structured question bank to {STRUCTURED_OUT_PATH}")


if __name__ == "__main__":
    main()
