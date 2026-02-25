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

_CSV_DIR = Path(__file__).resolve().parent.parent.parent / "csv files of problems"
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
