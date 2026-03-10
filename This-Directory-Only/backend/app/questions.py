"""
CSV question parser and in-memory question store.

Reads numpy, einsum, and einops problem CSVs on startup and provides
lookup by ID, subtopic, and difficulty.

CSV layouts:
  Numpy CSV (Export of numpy problems with outputs.csv):
    - Rows 1-2 are empty
    - Row 3 is the header: Topic, Subtopic, Question, Answer, Problem difficulty, Output
    - Data starts at row 4

  Einsum CSV (einsum_problems.csv):
    - Row 1 is the header: Topic, Subtopic, Question, Problem difficulty, Output, Answer
    - Data starts at row 2 (no empty rows)

  Einops CSV (einops_problems_with_outputs.csv, or einops_problems.csv):
    - Row 1 is the header: Topic, Subtopic, Question, Answer, Problem difficulty, Output
    - Data starts at row 2 (no empty rows)
    - Subtopics: Rearrange, Reduce, Repeat, Deep Learning

  All CSVs use DictReader so column order doesn't matter.
  Subtopics are stored as "{Topic}: {Subtopic}" to keep topics distinct
  (e.g. "Numpy: Core array literacy" vs "Einsum: Core array literacy"
   vs "Einops: Rearrange" vs "Einops: Deep Learning").

  Difficulty emoji markers in Question text: ★☆☆ = easy, ★★☆ = medium, ★★★ = hard
  "Problem difficulty" column is a numeric score (roughly 10-100)
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_CSV_DIR = Path(__file__).resolve().parents[3] / "This-Directory-Only" / "csv files of problems"
NUMPY_CSV_PATH = _CSV_DIR / "Export of numpy problems with outputs.csv"
EINSUM_CSV_PATH = _CSV_DIR / "einsum_problems.csv"
# Prefer the pre-computed-outputs version; fall back to the base CSV
EINOPS_CSV_PATH = (
    _CSV_DIR / "einops_problems_with_outputs.csv"
    if (_CSV_DIR / "einops_problems_with_outputs.csv").exists()
    else _CSV_DIR / "einops_problems.csv"
)


@dataclass
class Question:
    id: int
    topic: str
    subtopic: str
    question_text: str
    answer_code: str
    difficulty_score: int  # numeric 10-100
    difficulty_label: str  # "easy", "medium", "hard"
    expected_output: str  # stdout from running answer_code
    primary_library: str = "python"
    task_type: str = "stdout_prediction"
    expected_artifact_type: str = "stdout"
    supports_visual_output: bool = False
    function_name: str | None = None
    starter_code: str | None = None
    test_cases: List[dict] = field(default_factory=list)
    submission_mode: str = "stdout"


# ---------------------------------------------------------------------------
# Module-level store – populated once by load_questions()
# ---------------------------------------------------------------------------
_questions: List[Question] = []
_questions_by_id: Dict[int, Question] = {}
_questions_by_subtopic: Dict[str, List[Question]] = {}
_subtopics: List[str] = []
_subtopic_to_topic: Dict[str, str] = {}  # "Numpy: Core array literacy" -> "Numpy"

# Manual curation: remove questions that are effectively copy/paste of the prompt.
_CURATED_EXCLUDED_IDS = {9, 20, 21, 33, 39, 44, 45, 57, 88, 161, 188, 203, 221, 222, 223, 226}


def _classify_difficulty(question_text: str, numeric_score: int) -> str:
    """Derive easy/medium/hard from the star emoji in the question text."""
    if "★★★" in question_text:
        return "hard"
    elif "★★☆" in question_text:
        return "medium"
    elif "★☆☆" in question_text:
        return "easy"
    # Fallback based on numeric score
    if numeric_score <= 35:
        return "easy"
    elif numeric_score <= 65:
        return "medium"
    return "hard"


def _infer_primary_library(topic: str, question_text: str, answer_code: str) -> str:
    blob = f"{question_text}\n{answer_code}".lower()
    topic_lower = topic.lower()
    if "einops" in topic_lower or "einops" in blob:
      return "einops"
    if "einsum" in topic_lower or "einsum" in blob:
      return "einops.einsum"
    if "numpy" in topic_lower or "np." in blob:
      return "numpy"
    if "torch" in blob or "t." in blob:
      return "torch"
    return "python"


def _infer_task_type(topic: str, question_text: str, answer_code: str, expected_output: str) -> str:
    blob = f"{question_text}\n{answer_code}".lower()
    topic_lower = topic.lower()
    if "display_array_as_img" in blob:
        return "image_transform"
    if topic_lower == "einops":
        if _is_visual_einops_prompt(question_text, answer_code):
            return "image_transform"
        return "tensor_transform"
    if "einsum" in blob or topic_lower == "einsum":
        return "tensor_expression"
    if re.search(r"\bdef\s+\w+\s*\(", answer_code):
        return "function_impl"
    if expected_output:
        return "stdout_prediction"
    return "code_completion"


def _is_visual_einops_prompt(question_text: str, answer_code: str) -> bool:
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


def _extract_prompt_setup_code(question_text: str) -> str:
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


def _requires_float_fixture(answer_code: str) -> bool:
    lowered = answer_code.lower()
    return "'mean'" in lowered or '"mean"' in lowered


def _looks_like_expression(answer_code: str) -> bool:
    stripped = answer_code.strip()
    if not stripped or "\n" in stripped:
        return False
    forbidden = ("print(", "assert ", "def ", "return ", ";")
    return not any(token in stripped for token in forbidden)


def _split_answer_steps(answer_code: str) -> List[str]:
    normalized = answer_code.replace(";", "\n")
    return [line.strip() for line in normalized.splitlines() if line.strip()]


def _is_assignment_statement(code_line: str) -> bool:
    return bool(re.match(r"^[A-Za-z_]\w*\s*=", code_line))


def _extract_display_variable(answer_code: str) -> str | None:
    match = re.search(r"display_array_as_img\(([^)]+)\)", answer_code)
    if match:
        return match.group(1).strip()
    assign_match = re.search(r"\b([A-Za-z_]\w*)\s*=", answer_code)
    return assign_match.group(1) if assign_match else None


def _strip_display_calls(answer_code: str) -> str:
    return "\n".join(
        line for line in answer_code.replace(";", "\n").splitlines()
        if "display_array_as_img(" not in line
    ).strip()


def _infer_default_fixture_setup(question_text: str, answer_code: str, primary_library: str) -> str:
    lines: List[str] = []
    blob = f"{question_text}\n{answer_code}"
    needs_float = _requires_float_fixture(answer_code)
    if primary_library in {"einops", "einops.einsum"}:
        uses_nhwc = "b h w c" in blob or "NHWC" in question_text
        required_arr_batch = _infer_required_arr_batch(question_text, answer_code)
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
            if "b =" not in "\n".join(lines):
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


def _infer_required_arr_batch(question_text: str, answer_code: str) -> int | None:
    slice_match = re.search(r"\barr\[\s*:\s*(\d+)\s*\]", answer_code)
    if slice_match:
        return int(slice_match.group(1))
    prompt_match = re.search(r"\bfirst\s+(\d+)\s+images?\b", question_text, flags=re.IGNORECASE)
    if prompt_match:
        return int(prompt_match.group(1))
    return None


def _derive_test_case(answer_code: str, question_text: str, primary_library: str) -> tuple[str, str]:
    steps = _split_answer_steps(answer_code)
    if _looks_like_expression(answer_code):
        setup_code = _extract_prompt_setup_code(question_text) or _infer_default_fixture_setup(question_text, answer_code, primary_library)
        return setup_code, answer_code.strip()
    if not steps:
        return "", "None"
    last = steps[-1]
    if last.startswith("print(") and last.endswith(")"):
        return "\n".join(steps[:-1]), last[len("print(") : -1].strip()
    if last.startswith("display_array_as_img(") and last.endswith(")"):
        return "\n".join(steps[:-1]), last[len("display_array_as_img(") : -1].strip()
    if _is_assignment_statement(last) and not last.startswith(("if ", "for ", "while ")):
        lhs = last.split("=", 1)[0].strip()
        return "\n".join(steps[:-1]), lhs
    return "\n".join(steps[:-1]), last


def _derive_function_payload(
    question_text: str,
    answer_code: str,
    primary_library: str,
    task_type: str | None = None,
) -> tuple[str | None, str | None, List[dict], str]:
    import_line = "import numpy as np"
    if primary_library in {"einops", "einops.einsum"}:
        import_line += "\nimport einops\nfrom einops import einsum, rearrange, reduce, repeat"

    task_type = task_type or _infer_task_type(
        question_text.split(":")[0] if ":" in question_text else "",
        question_text,
        answer_code,
        "",
    )

    if task_type == "image_transform":
        fixture_setup = _extract_prompt_setup_code(question_text) or _infer_default_fixture_setup(
            question_text, answer_code, primary_library
        )
        visual_setup, visual_expr = _derive_test_case(answer_code, question_text, primary_library)
        variable_name = _extract_display_variable(answer_code)
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

    setup_code, expected_expr = _derive_test_case(answer_code, question_text, primary_library)
    if not setup_code:
        setup_code = _extract_prompt_setup_code(question_text) or _infer_default_fixture_setup(question_text, answer_code, primary_library)
    top_level_setup = f"{setup_code}\n\n" if setup_code else ""
    starter_code = (
        f"{import_line}\n\n"
        f"{top_level_setup}"
        "def solve():\n"
        + "\n    # Write your solution here\n"
        + "    return None\n\n"
        + "print(solve())\n"
    )
    test_cases = [{"setup_code": setup_code, "call": "solve()", "expected_expr": expected_expr or "None"}]
    return "solve", starter_code, test_cases, "function"


def _load_csv_into(
    path: Path,
    questions: List[Question],
    start_id: int,
    skip_rows: int = 0,
) -> int:
    """
    Load questions from a single CSV file into the provided list.
    Subtopics are stored as "{Topic}: {Subtopic}" for uniqueness.
    Returns the next available ID after loading.
    """
    if not path.exists():
        logger.warning("Questions CSV not found at %s — skipping", path)
        return start_id

    idx = start_id
    with path.open("r", encoding="utf-8") as f:
        for _ in range(skip_rows):
            next(f, None)
        reader = csv.DictReader(f)
        for row in reader:
            qid = idx
            topic = (row.get("Topic") or "").strip()
            subtopic_raw = (row.get("Subtopic") or "").strip()
            # Prefix subtopic with topic to keep Numpy/Einsum subtopics distinct
            subtopic = f"{topic}: {subtopic_raw}" if topic and subtopic_raw else subtopic_raw
            question_text = (row.get("Question") or "").strip()
            answer_code = (row.get("Answer") or "").strip()
            raw_difficulty = (row.get("Problem difficulty") or "0").strip()
            expected_output = (row.get("Output") or "").strip()

            if not question_text or not subtopic_raw:
                idx += 1
                continue
            if qid in _CURATED_EXCLUDED_IDS:
                idx += 1
                continue

            try:
                difficulty_score = int(float(raw_difficulty))
            except ValueError:
                difficulty_score = 50

            difficulty_label = _classify_difficulty(question_text, difficulty_score)
            primary_library = _infer_primary_library(topic, question_text, answer_code)
            task_type = _infer_task_type(topic, question_text, answer_code, expected_output)
            function_name, starter_code, test_cases, submission_mode = _derive_function_payload(
                question_text, answer_code, primary_library, task_type
            )

            questions.append(
                Question(
                    id=qid,
                    topic=topic,
                    subtopic=subtopic,
                    question_text=question_text,
                    answer_code=answer_code,
                    difficulty_score=difficulty_score,
                    difficulty_label=difficulty_label,
                    expected_output=expected_output,
                    primary_library=primary_library,
                    task_type=task_type,
                    expected_artifact_type="image" if task_type == "image_transform" else "stdout",
                    supports_visual_output=task_type == "image_transform",
                    function_name=function_name,
                    starter_code=starter_code,
                    test_cases=test_cases,
                    submission_mode=submission_mode,
                )
            )
            idx += 1

    loaded = idx - start_id
    logger.info("Loaded %d questions from %s", loaded, path)
    return idx


def load_questions(csv_path: Optional[Path] = None) -> None:
    """Parse all CSV files and populate the in-memory store.

    Loading order (to preserve existing question IDs):
      1. Numpy CSV  — IDs 1..N
      2. Einsum CSV — IDs N+1..N+70
      3. Einops CSV — IDs N+71..N+70+92

    If csv_path is given (e.g. in tests), only that file is loaded using
    the numpy CSV layout (2 empty header rows).
    """
    global _questions, _questions_by_id, _questions_by_subtopic, _subtopics, _subtopic_to_topic

    questions: List[Question] = []

    if csv_path is not None:
        # Legacy / test path — single CSV, numpy layout
        _load_csv_into(csv_path, questions, start_id=1, skip_rows=2)
    else:
        next_id = _load_csv_into(NUMPY_CSV_PATH, questions, start_id=1, skip_rows=2)
        next_id = _load_csv_into(EINSUM_CSV_PATH, questions, start_id=next_id, skip_rows=0)
        _load_csv_into(EINOPS_CSV_PATH, questions, start_id=next_id, skip_rows=0)

    _questions = questions
    _questions_by_id = {q.id: q for q in questions}

    by_sub: Dict[str, List[Question]] = {}
    sub_to_topic: Dict[str, str] = {}
    for q in questions:
        by_sub.setdefault(q.subtopic, []).append(q)
        sub_to_topic[q.subtopic] = q.topic
    _questions_by_subtopic = by_sub
    _subtopics = sorted(by_sub.keys())
    _subtopic_to_topic = sub_to_topic

    logger.info(
        "Loaded %d questions across %d subtopics",
        len(questions),
        len(_subtopics),
    )


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------

def get_all_questions() -> List[Question]:
    return _questions


def get_question_by_id(qid: int) -> Optional[Question]:
    return _questions_by_id.get(qid)


def get_questions_by_subtopic(subtopic: str) -> List[Question]:
    return _questions_by_subtopic.get(subtopic, [])


def get_subtopics() -> List[str]:
    return list(_subtopics)


def get_questions_by_subtopic_and_difficulty(subtopic: str, difficulty_label: str) -> List[Question]:
    return [q for q in _questions_by_subtopic.get(subtopic, []) if q.difficulty_label == difficulty_label]


def get_topic_for_subtopic(subtopic: str) -> str:
    """Return the topic name for a given subtopic key (e.g. 'Numpy' for 'Numpy: Core array literacy')."""
    return _subtopic_to_topic.get(subtopic, subtopic.split(":")[0] if ":" in subtopic else "")
