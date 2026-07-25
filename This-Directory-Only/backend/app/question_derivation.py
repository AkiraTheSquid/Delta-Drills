"""
Question derivation helpers — pure text/code inference, no module state.

Split out of app/questions.py (which was over the 700-LOC ceiling). Every
function here is a pure function of a question's text/answer_code: difficulty
classification, library and task-type inference, fixture/test-case derivation.
Nothing here touches the question store, the CSVs, or any global.

The public solution-display API (compose_full_solution, wrap_answer_as_function)
deliberately STAYS in app/questions.py — backend/watch.py asserts those two defs
live in that file, and the practice frontend rides on them.

Imported back into app/questions.py, so `from app.questions import ...` keeps
working unchanged for all ten importers.
"""

from __future__ import annotations

import re
from typing import List


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

