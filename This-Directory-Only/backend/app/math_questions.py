"""Load the math (multiple-choice) bank into the in-memory question store.

Split out of app/questions.py (at its LOC ceiling). `question_cls` is passed
in rather than imported so this module never imports questions.py back.
See docs/spec-math-mc-backbone.md and lessons/math_bank.py for the format.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MATH_BANK_PATH = Path(__file__).resolve().parents[3] / "Local_Deployed_Shared" / "lessons" / "math_bank.py"


def load_math_into(questions: list, question_cls) -> None:
    """Append the math (multiple-choice) bank after every CSV source.

    Same loader the exporter and validator use (lessons/math_bank.py, loaded
    by path because that folder is data the image copies, not a package). Ids
    are explicit and >= MATH_ID_FLOOR, so nothing here is positional; a file
    that collides with a CSV id is dropped and logged rather than allowed to
    shadow the drill in `_questions_by_id`. A missing or unparsable file is a
    boot-time warning, never a refusal: the bank must serve without math.
    """
    if not MATH_BANK_PATH.exists():
        return
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("delta_math_bank", MATH_BANK_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        rows = module.load_math_rows(MATH_BANK_PATH.parent)
    except Exception as exc:  # noqa: BLE001 — content error must not stop boot
        logger.warning("Math bank not loaded: %s", exc)
        return
    taken = {q.id for q in questions}
    for row in rows:
        if row["id"] < module.MATH_ID_FLOOR or row["id"] in taken:
            logger.error("Math problem id %s collides with the drill bank — skipped", row["id"])
            continue
        taken.add(row["id"])
        questions.append(question_cls(
            id=row["id"],
            topic=row["topic"],
            subtopic=row["subtopic_key"],
            question_text=row["question_text"],
            answer_code="",
            difficulty_score=row["difficulty_score"],
            difficulty_label=row["difficulty_label"],
            expected_output=row["expected_output"],
            primary_library=row["primary_library"],
            task_type=row["task_type"],
            expected_artifact_type=row["expected_artifact_type"],
            supports_visual_output=False,
            function_name=None,
            starter_code="",
            test_cases=[],
            submission_mode=row["submission_mode"],
            hint=row.get("hint"),
            choices=row["choices"],
            correct_choice=row["correct_choice"],
            math_kind=row["math_kind"],
            solution_md=row["solution_md"],
        ))


