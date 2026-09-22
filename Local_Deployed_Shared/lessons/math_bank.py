"""The math (multiple-choice, non-code) problem bank — one loader, three readers.

Spec: docs/spec-math-mc-backbone.md. A math KP is a lesson markdown with
`kind: math` in its frontmatter and a colocated ``kp-<slug>.problems.json``
beside it:

    {
      "kc": "math.dot-product",
      "problems": [
        {
          "id": 50001,                      # >= MATH_ID_FLOOR, unique bank-wide
          "kind": "compute",                # compute | derivation-step | statement
          "difficulty": 30,                 # 10..100, same scale as the bank
          "prompt": "Compute $\\vec a \\cdot \\vec b$ for ...",   # markdown + LaTeX
          "choices": [                      # 2..6, keys A.. in authored order
            {"key": "A", "text": "$32$", "value": "32"},
            {"key": "B", "text": "$-32$", "value": "-32"}
          ],
          "answer": "A",
          "solution": "markdown + LaTeX shown after the verdict",
          "hint": "optional",
          "verify": {"sympy": {"symbols": "", "truth": "Matrix([1,2,3]).dot(Matrix([4,5,6]))"}}
        }
      ]
    }

`verify.sympy` is REQUIRED for `compute` and `derivation-step` and FORBIDDEN
for `statement`: `truth` is the author's own derivation of the answer, every
choice carries a `value` in SymPy syntax, and scripts/validate_math.py asserts
`truth == value[answer]` and `truth != value[k]` for every distractor. A
`verify.lean` key is reserved (never read) for a later pass.

This module is stdlib-only on purpose: the backend (app/questions.py), the
exporter (pipeline/export_questions_json.py) and the validator all read the
bank through `load_math_rows`, and the backend image carries this folder
already (Dockerfile copies Local_Deployed_Shared/lessons). One reader shape,
so the three cannot disagree about what a problem is.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

LESSONS_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = LESSONS_DIR / "kc_registry.json"

# Explicit ids in the JSON, above anything the positional CSV loaders could
# ever reach, so appending a CSV row can never renumber a math problem and a
# math problem can never shadow a drill.
MATH_ID_FLOOR = 50000
KINDS = ("compute", "derivation-step", "statement")
SYMPY_KINDS = ("compute", "derivation-step")
MIN_CHOICES, MAX_CHOICES = 2, 6
KEY_RE = re.compile(r"^[A-F]$")
# The bank's own difficulty scale (questions.py: "numeric 10-100").
DIFFICULTY_RANGE = (10, 100)
# What a math row is to every reader that switches on these two fields.
SUBMISSION_MODE = "mc"
TASK_TYPE = "mc"
PRIMARY_LIBRARY = "math"


def problem_files(lessons_dir: Path = LESSONS_DIR) -> list[Path]:
    return sorted(lessons_dir.glob("*/kp-*.problems.json"))


def kp_markdown_for(problems_path: Path) -> Path:
    """`kp-foo.problems.json` -> `kp-foo.md` beside it."""
    return problems_path.with_name(problems_path.name[: -len(".problems.json")] + ".md")


def _difficulty_label(score: int) -> str:
    # Mirrors export_questions_json.classify_difficulty's numeric bands.
    if score <= 39:
        return "easy"
    if score <= 69:
        return "medium"
    return "hard"


def _kc_curriculum(registry: dict) -> dict[str, dict]:
    """kc id -> {topic, subtopic, subtopic_key} via the KC's lesson."""
    lessons = {l["id"]: l for l in registry.get("lessons", [])}
    out = {}
    for kc in registry.get("kcs", []):
        lesson = lessons.get(kc.get("lesson"))
        if not lesson:
            continue
        key = str(lesson.get("subtopic_key") or "")
        topic = str(lesson.get("topic") or "")
        prefix = f"{topic}: "
        out[kc["id"]] = {
            "topic": topic,
            "subtopic": key[len(prefix):] if key.startswith(prefix) else key,
            "subtopic_key": key,
        }
    return out


def read_problem_file(path: Path) -> dict:
    """The parsed file, shape-checked only as far as the loader needs.

    Deeper rules (id range, choice keys, SymPy) belong to validate_math.py;
    this raises only on what would make the rows themselves unreadable.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("problems"), list):
        raise ValueError(f"{path}: expected {{'kc': ..., 'problems': [...]}}")
    if not str(data.get("kc") or "").strip():
        raise ValueError(f"{path}: missing top-level 'kc'")
    return data


def flat_row(problem: dict, kc: str, curriculum: dict, source_path: str) -> dict:
    """One problem as a flat bank row (the questions.json shape).

    Coding-only fields are present and empty so every reader that indexes
    them (`starter_code`, `test_cases`, `answer_code`) finds the key. The
    keyed answer doubles as `expected_output`: the router's
    `expected_output or run(answer_code)` fallback then never runs anything.
    """
    score = int(problem.get("difficulty", 0) or 0)
    choices = [
        {
            "key": str(c.get("key", "")),
            "text": str(c.get("text", "")),
            "value": str(c["value"]) if c.get("value") is not None else None,
        }
        for c in (problem.get("choices") or [])
        if isinstance(c, dict)
    ]
    return {
        "id": int(problem["id"]),
        "topic": curriculum["topic"],
        "subtopic": curriculum["subtopic"],
        "subtopic_key": curriculum["subtopic_key"],
        "question_text": str(problem.get("prompt") or ""),
        "answer_code": "",
        "difficulty_score": score,
        "difficulty_label": _difficulty_label(score),
        "expected_output": str(problem.get("answer") or ""),
        "language": "math",
        "primary_library": PRIMARY_LIBRARY,
        "task_type": TASK_TYPE,
        "function_name": None,
        "starter_code": "",
        "test_cases": [],
        "submission_mode": SUBMISSION_MODE,
        "wrong_examples": [],
        "provenance": None,
        "expected_artifact_type": "choice",
        "supports_visual_output": False,
        "hint": str(problem.get("hint") or "") or None,
        "source_type": "math_json",
        "source_path": source_path,
        # The MC-only fields. Readers that do not know them ignore them.
        "math_kind": str(problem.get("kind") or ""),
        "math_kc": kc,
        "choices": choices,
        "correct_choice": str(problem.get("answer") or ""),
        "solution_md": str(problem.get("solution") or ""),
        "verify": problem.get("verify") if isinstance(problem.get("verify"), dict) else None,
    }


def load_math_rows(lessons_dir: Path = LESSONS_DIR, registry: dict | None = None) -> list[dict]:
    """Every math problem in the tree as flat bank rows, sorted by id.

    A KC the registry does not know gets topic/subtopic "" rather than an
    exception here: the validator reports it with the file named, and the
    backend must still boot on a half-authored tree.
    """
    if registry is None:
        registry = json.loads((lessons_dir / "kc_registry.json").read_text(encoding="utf-8"))
    curriculum = _kc_curriculum(registry)
    repo = lessons_dir.parent.parent
    rows: list[dict] = []
    # Explicit ids are only safe if they are unique ACROSS files: a static
    # by-id map would keep one row and the backend the other. Refused here,
    # so the exporter, the backend and the validator cannot disagree.
    owner: dict = {}
    for path in problem_files(lessons_dir):
        data = read_problem_file(path)
        kc = str(data["kc"])
        cur = curriculum.get(kc) or {"topic": "", "subtopic": "", "subtopic_key": ""}
        try:
            rel = str(path.relative_to(repo))
        except ValueError:
            rel = str(path)
        for problem in data["problems"]:
            if not isinstance(problem, dict) or "id" not in problem:
                raise ValueError(f"{path}: every problem needs an integer 'id'")
            pid = problem["id"]
            if pid in owner:
                raise ValueError(f"{rel}: problem id {pid} is already used in {owner[pid]}")
            owner[pid] = rel
            rows.append(flat_row(problem, kc, cur, rel))
    rows.sort(key=lambda r: r["id"])
    return rows


def is_math_row(row: dict) -> bool:
    return row.get("submission_mode") == SUBMISSION_MODE
