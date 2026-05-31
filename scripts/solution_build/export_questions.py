#!/usr/bin/env python3
"""Export the full Delta Drills question bank to a single JSON file the
solution-Colab authoring workflow consumes.

Persisted to the repo (scripts/solution_build/dd_questions.json) so a session
compaction can't wipe it the way /tmp did.

Run from the backend dir with its venv:
    cd This-Directory-Only/backend
    .venv/bin/python ../../scripts/solution_build/export_questions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# backend package importable
BACKEND = Path(__file__).resolve().parents[2] / "This-Directory-Only" / "backend"
sys.path.insert(0, str(BACKEND))

from app import questions as Q  # noqa: E402

OUT = Path(__file__).resolve().parent / "dd_questions.json"


def main() -> None:
    Q.load_questions()
    rows = []
    for q in Q.get_all_questions():
        rows.append(
            {
                "id": q.id,
                "topic": q.topic,
                "subtopic": q.subtopic,
                "question_text": q.question_text,
                "answer_code": q.answer_code,
                "full_solution": Q.compose_full_solution(q.starter_code, q.answer_code),
                "starter_code": q.starter_code,
                "expected_output": q.expected_output,
                "primary_library": q.primary_library,
                "task_type": q.task_type,
                "submission_mode": q.submission_mode,
                "function_name": q.function_name,
                "test_cases": q.test_cases,
                "supports_visual_output": q.supports_visual_output,
                "difficulty_score": q.difficulty_score,
                "difficulty_label": q.difficulty_label,
            }
        )
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} questions -> {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
