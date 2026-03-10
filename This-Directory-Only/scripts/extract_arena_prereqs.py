#!/usr/bin/env python3
"""
Extract structured exercises from the ARENA prerequisites notebook.

This preserves metadata that is lost in the current CSV-only pipeline:
  - starter code / function stubs
  - solution code
  - visual/image task markers
  - source notebook cell references
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
SHARED_DIR = REPO_DIR / "Local_Deployed_Shared"
NOTEBOOK_PATH = (
    SHARED_DIR
    / "ARENA_3.0-main"
    / "chapter0_fundamentals"
    / "exercises"
    / "part0_prereqs"
    / "0.0_Prerequisites_exercises.ipynb"
)
OUT_PATH = SHARED_DIR / "arena_prereqs_structured.json"


def cell_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def markdown_title(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#### "):
            return stripped.removeprefix("#### ").strip()
    return None


def clean_markdown(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("<details>") or stripped.startswith("</details>"):
            continue
        if stripped.startswith("<summary>") or stripped.startswith("```"):
            continue
        lines.append(stripped)
    return "\n".join(lines).strip()


def code_block_from_markdown(text: str) -> str | None:
    match = re.search(r"```python\s+(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def infer_topic(section_name: str) -> str:
    normalized = section_name.lower()
    if "einsum" in normalized:
        return "Einsum"
    if "einops" in normalized:
        return "Einops"
    if "broadcast" in normalized:
        return "Broadcasting"
    return "Prerequisites"


def infer_task_type(code: str) -> str:
    if "display_array_as_img" in code:
        return "image_transform"
    if "raise NotImplementedError()" in code:
        return "function_impl"
    return "code_cell"


def extract_function_names(code: str) -> list[str]:
    return re.findall(r"\bdef\s+([A-Za-z_]\w*)\s*\(", code)


def extract_variable_name(code: str) -> str | None:
    match = re.search(r"display_array_as_img\((arr\d+)\)", code)
    if match:
        return match.group(1)
    return None


def extract_records() -> list[dict]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])

    records = []
    current_section = "Prerequisites"
    last_prompt_markdown = ""

    for idx, cell in enumerate(cells):
        source = cell_text(cell)
        if cell.get("cell_type") == "markdown":
            if source.lstrip().startswith("## "):
                current_section = source.lstrip()[3:].splitlines()[0].strip()
            if source.lstrip().startswith("### ") or source.lstrip().startswith("#### "):
                last_prompt_markdown = source
            continue

        if cell.get("cell_type") != "code":
            continue

        is_function_exercise = "raise NotImplementedError()" in source
        is_image_exercise = "display_array_as_img(" in source and "# Your code here" in source
        if not is_function_exercise and not is_image_exercise:
            continue

        next_markdown = ""
        if idx + 1 < len(cells) and cells[idx + 1].get("cell_type") == "markdown":
            next_markdown = cell_text(cells[idx + 1])

        topic = infer_topic(current_section)
        task_type = infer_task_type(source)
        function_names = extract_function_names(source)
        variable_name = extract_variable_name(source)
        title = markdown_title(last_prompt_markdown) or (function_names[0] if function_names else variable_name or f"cell_{idx}")

        records.append(
            {
                "id": len(records) + 1,
                "source": {
                    "type": "ipynb",
                    "path": str(NOTEBOOK_PATH.relative_to(REPO_DIR)),
                    "cell_index": idx,
                },
                "curriculum": {
                    "topic": topic,
                    "section": current_section,
                    "title": title,
                },
                "exercise": {
                    "task_type": task_type,
                    "language": "python",
                    "primary_library": "einops" if topic == "Einops" else ("einops.einsum" if topic == "Einsum" else "torch"),
                    "function_names": function_names,
                    "variable_name": variable_name,
                    "prompt_markdown": clean_markdown(last_prompt_markdown),
                    "starter_code": source.strip(),
                    "canonical_solution": code_block_from_markdown(next_markdown),
                    "expected_artifact_type": "image" if task_type == "image_transform" else "function",
                    "supports_visual_output": task_type == "image_transform",
                    "render_helper": "display_array_as_img" if task_type == "image_transform" else None,
                },
            }
        )

    return records


def main() -> None:
    records = extract_records()
    OUT_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Extracted {len(records)} ARENA prerequisite exercises to {OUT_PATH}")


if __name__ == "__main__":
    main()
