"""Load the LeetCode Patterns drills into the in-memory question store.

Split out of app/questions.py (at its LOC ceiling), same pattern as
math_questions.py. The rows are Local_Deployed_Shared/lessons/leetcode/
problems.json, written whole by This-Directory-Only/scripts/leetcode_course/
export_app.py in questions.json's flat shape; every row there was graded by
code_runner before it was written (reference passes, bare starter fails).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROBLEMS_PATH = Path(__file__).resolve().parents[3] / "Local_Deployed_Shared" / "lessons" / "leetcode" / "problems.json"
ID_FLOOR = 60000
# A drill's own answer clock, question id -> seconds (problems.json
# `secs_allowed`, sized by LeetCode difficulty and pattern in
# scripts/leetcode_course/drill_format.py). question_pick.secs_allowed_for
# serves it in place of the concept table's 5:00-clamped clock.
CLOCK_MAX_SECS = 60 * 60
_CLOCKS: dict[int, int] = {}


def clock_secs(question_id: int) -> int | None:
    """The LeetCode drill's own clock, or None for any other question."""
    return _CLOCKS.get(question_id)


def load_leetcode_into(questions: list, question_cls) -> None:
    """Append the LeetCode drills after every other source. A missing or
    unreadable file is a boot-time warning, never a refusal; an id below
    ID_FLOOR or already taken is dropped and logged so it can never shadow
    another drill in `_questions_by_id`."""
    if not PROBLEMS_PATH.exists():
        return
    try:
        rows = json.loads(PROBLEMS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — content error must not stop boot
        logger.warning("LeetCode bank not loaded: %s", exc)
        return
    # Rebuilt whole on every load: a row dropped from problems.json must not
    # keep overriding its id's concept clock. Codex, 2026-09-25.
    _CLOCKS.clear()
    taken = {q.id for q in questions}
    for row in rows:
        if row["id"] < ID_FLOOR or row["id"] in taken:
            logger.error("LeetCode problem id %s collides with the drill bank — skipped", row["id"])
            continue
        taken.add(row["id"])
        secs = row.get("secs_allowed")
        if isinstance(secs, int) and not isinstance(secs, bool) and secs > 0:
            _CLOCKS[row["id"]] = min(secs, CLOCK_MAX_SECS)
        questions.append(question_cls(
            id=row["id"],
            topic=row["topic"],
            subtopic=row["subtopic_key"],
            question_text=row["question_text"],
            answer_code=row["answer_code"],
            difficulty_score=row["difficulty_score"],
            difficulty_label=row["difficulty_label"],
            expected_output="",
            primary_library=row["primary_library"],
            task_type=row["task_type"],
            expected_artifact_type=row["expected_artifact_type"],
            supports_visual_output=False,
            function_name=row["function_name"],
            starter_code=row["starter_code"],
            test_cases=row["test_cases"],
            submission_mode=row["submission_mode"],
            provenance=row.get("provenance"),
        ))
