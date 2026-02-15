#!/usr/bin/env python3
"""
Export the CSV question bank to frontend/questions.json.

Reads the same CSV that the backend uses (skipping 2 header rows),
and produces a JSON array ready for the frontend / Pyodide practice engine.

Run manually or automatically via deploy_delta_drills.sh.
"""

import csv
import json
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_DIR / "csv files of problems" / "Export of numpy problems with outputs.csv"
OUT_PATH = REPO_DIR / "frontend" / "questions.json"


def classify_difficulty(question_text: str, numeric_score: int) -> str:
    if "\u2605\u2605\u2605" in question_text:
        return "hard"
    elif "\u2605\u2605\u2606" in question_text:
        return "medium"
    elif "\u2605\u2606\u2606" in question_text:
        return "easy"
    if numeric_score <= 35:
        return "easy"
    elif numeric_score <= 65:
        return "medium"
    return "hard"


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    questions = []
    with CSV_PATH.open("r", encoding="utf-8") as f:
        # Skip two empty header rows (matches backend/app/questions.py)
        next(f, None)
        next(f, None)
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            topic = (row.get("Topic") or "").strip()
            subtopic = (row.get("Subtopic") or "").strip()
            question_text = (row.get("Question") or "").strip()
            answer_code = (row.get("Answer") or "").strip()
            raw_difficulty = (row.get("Problem difficulty") or "0").strip()
            expected_output = (row.get("Output") or "").strip()

            if not question_text or not subtopic:
                continue

            try:
                difficulty_score = int(float(raw_difficulty))
            except ValueError:
                difficulty_score = 50

            difficulty_label = classify_difficulty(question_text, difficulty_score)

            questions.append({
                "id": idx,
                "topic": topic,
                "subtopic": subtopic,
                "question_text": question_text,
                "answer_code": answer_code,
                "difficulty_score": difficulty_score,
                "difficulty_label": difficulty_label,
                "expected_output": expected_output,
            })

    OUT_PATH.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {len(questions)} questions to {OUT_PATH}")


if __name__ == "__main__":
    main()
