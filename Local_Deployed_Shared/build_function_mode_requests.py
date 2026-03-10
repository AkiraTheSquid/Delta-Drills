#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
from delta_paths import THIS_DIR_ONLY, get_chatgpt_runtime_dir

QUESTIONS_PATH = THIS_DIR_ONLY / "questions_full.json"
CHATGPT_RUNTIME_DIR = get_chatgpt_runtime_dir()
OUT_PATH = CHATGPT_RUNTIME_DIR / "function_mode_requests.jsonl"


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    pending = [q for q in questions if q.get("submission_mode") != "function"]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for q in pending:
            f.write(
                json.dumps(
                    {
                        "id": q["id"],
                        "topic": q["topic"],
                        "subtopic": q["subtopic"],
                        "question_text": q["question_text"],
                        "answer_code": q["answer_code"],
                        "expected_output": q.get("expected_output", ""),
                        "primary_library": q.get("primary_library", "python"),
                        "task_type": q.get("task_type", "stdout_prediction"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote {len(pending)} function-mode requests to {OUT_PATH}")


if __name__ == "__main__":
    main()
