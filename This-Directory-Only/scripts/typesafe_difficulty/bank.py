"""Load the question bank and group it into concepts for difficulty rating.

A "concept" is the group a difficulty score is compared WITHIN: the KC the
q-matrix files a question under (`target_kcs`, always exactly one), because
that is the pool `prioritization` narrows to before it walks difficulty bands.
Questions the q-matrix does not place fall back to their `subtopic_key`, the
pool the legacy subtopic picker draws from, so every bank question lands in
exactly one group and gets a score on that group's scale.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BANK = ROOT / "This-Directory-Only" / "questions_full.json"
QMATRIX = ROOT / "Local_Deployed_Shared" / "lessons" / "qmatrix_tags.json"
DATA_DIR = ROOT / "This-Directory-Only" / "typesafe_difficulty"


@dataclass(frozen=True)
class Problem:
    id: int
    concept: str
    question: str
    answer: str
    current_score: int

    def as_state(self) -> dict:
        """The problem as the model sees it: prompt + reference solution.

        The reference solution is what makes "how hard" answerable — the
        legacy champion rater sent question AND answer for the same reason.
        """
        return {"question": self.question, "solution": self.answer}


def load_problems() -> list[Problem]:
    bank = json.loads(BANK.read_text())  # a plain list (questions.json wraps it, this file does not)
    qmatrix = json.loads(QMATRIX.read_text())
    problems = []
    for q in bank:
        tags = qmatrix.get(str(q["id"])) or {}
        kcs = tags.get("target_kcs") or []
        concept = kcs[0] if kcs else f"subtopic:{q.get('subtopic_key') or q.get('subtopic')}"
        problems.append(Problem(
            id=int(q["id"]),
            concept=concept,
            question=(q.get("question_text") or "").strip(),
            answer=(q.get("answer_code") or "").strip(),
            current_score=int(q.get("difficulty_score") or 0),
        ))
    return problems


def group_by_concept(problems: list[Problem]) -> dict[str, list[Problem]]:
    groups: dict[str, list[Problem]] = {}
    for p in problems:
        groups.setdefault(p.concept, []).append(p)
    return groups
