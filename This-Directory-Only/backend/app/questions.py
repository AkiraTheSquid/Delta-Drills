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
import json
import logging
import os
import re
from threading import Lock
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from app import lessons

logger = logging.getLogger(__name__)

_THIS_DIR_ONLY = Path(__file__).resolve().parents[3] / "This-Directory-Only"
_CSV_DIR = _THIS_DIR_ONLY / "csv files of problems"
NUMPY_CSV_PATH = _CSV_DIR / "Export of numpy problems with outputs.csv"
EINSUM_CSV_PATH = _CSV_DIR / "einsum_problems.csv"
# Prefer the pre-computed-outputs version; fall back to the base CSV
EINOPS_CSV_PATH = (
    _CSV_DIR / "einops_problems_with_outputs.csv"
    if (_CSV_DIR / "einops_problems_with_outputs.csv").exists()
    else _CSV_DIR / "einops_problems.csv"
)
CNN_CSV_PATH = _CSV_DIR / "cnn_problems.csv"
# Course-authored questions, loaded LAST so their positional ids sit above the
# whole imported bank and appending one can never renumber an existing question.
# Keep in sync with pipeline/export_questions_json.py::CSV_SOURCES.
CURATED_ADDITIONS_CSV_PATH = _CSV_DIR / "curated_additions.csv"


def _chatgpt_runtime_dir() -> Path:
    configured = os.environ.get("DELTA_CHATGPT_RUNTIME_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (_THIS_DIR_ONLY / "chatgpt").resolve()


def _load_jsonl_overrides(filename: str) -> Dict[int, dict]:
    path = _chatgpt_runtime_dir() / filename
    if not path.exists():
        return {}
    overrides: Dict[int, dict] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            overrides[int(record["id"])] = record
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Failed to load %s: %s", filename, exc)
        return {}
    return overrides


def _load_function_overrides() -> Dict[int, dict]:
    """Per-id override records produced by the AI quality-fix pipeline.

    Files are merged in order, with later files winning for any id collision.
    ``function_mode_overrides.jsonl`` is the round-1 quality-fix output,
    ``function_mode_overrides_round2.jsonl`` and ``_round3.jsonl`` are
    follow-up manual repairs of validator-flagged failures.
    ``einops_prompt_rewrite_overrides.jsonl`` is layered last so its
    question_text rewrites win, while preserving starter/test fields from
    the earlier rounds.
    """
    base = _load_jsonl_overrides("function_mode_overrides.jsonl")
    for layer_name in (
        "function_mode_overrides_round2.jsonl",
        "function_mode_overrides_round3.jsonl",
        "einops_prompt_rewrite_overrides.jsonl",
        "numpy_einsum_prompt_rewrite_overrides.jsonl",
        "prompt_expansion_overrides.jsonl",
        "difficulty_overrides.jsonl",
        # question_text-only rewrites of the authored torch/autograd/optimizer
        # batch (ids 405-479): de-giveaway + readability (2026-07 tester pass).
        # Layered last so these texts win; starter/test fields untouched.
        "question_text_quality_overrides.jsonl",
        # Starter-code fixes: 3 SyntaxErrors (bad indent/broken string literal)
        # and answer-leaking comments/precomputed-answer-before-solve() bugs
        # found during the 2026-07-04 tester pass. Layered last so these win.
        "starter_leak_and_syntax_fixes.jsonl",
        # Canonical answer_code repairs: numpy-2.0 removals (ptp / np.int),
        # missing imports, truncated CSV code, wrong lookup-table bound
        # (2026-07-06 stdout-expected sweep). Layered last so these win.
        "answer_code_repairs.jsonl",
        # Parameterized regeneration (2026-07-07): full question rewrites to
        # `def solve(<real params>)` + 3-6 test cases incl. edge cases, from
        # the Fable generation pass (mech-gate verified + hand-reviewed).
        # Layered last — replaces question_text/starter/test_cases/answer_code
        # wholesale for its ids. Keep in sync with the exporter.
        "parameterized_regen_overrides.jsonl",
        # Einops visual triage (2026-07-07): 41 layout-only/garbage-render
        # questions demoted to ordinary function grading (visual flags off);
        # 7 duplicate-image keepers get distinct fixtures (digit swaps).
        "einops_visual_fixes.jsonl",
        # PyTorch dialect conversion (2026-07-27): re-expresses drills in the
        # dialect ARENA actually uses (`import torch as t`), lesson by lesson.
        # Layered last so a conversion wins over every earlier numpy-era
        # repair. The torch import it introduces is itself what unparks the
        # question — see lessons.is_torch_dialect / torch_only_serving.
        # Keep in sync with pipeline/export_questions_json.py.
        "torch_dialect_overrides.jsonl",
        # Second conversion pass (2026-07-27): the einops + einsum drills, which
        # are ARENA 0.0's own exercise material. Kept in its own layer so the
        # np-1 pass stays reviewable on its own.
        "torch_dialect_overrides_einops_einsum.jsonl",
        # Third conversion pass (2026-07-28): np-2 "Indexing and selection" and
        # np-3 "Vectorization and broadcasting". These two carry the numpy
        # functions with no torch spelling, so ~30 of them are hand-translated
        # in This-Directory-Only/scripts/torchify_np23_manual.py. Keep in sync
        # with pipeline/export_questions_json.py.
        "torch_dialect_overrides_np23.jsonl",
        # Fourth conversion pass (2026-07-28): np-4 "Applied patterns", 45 of
        # its 51 drills. The other 6 are the whole numpy.structured-dtypes KC —
        # record dtypes, datetime64 and genfromtxt have no torch form at all,
        # so they stay NumPy and are listed in
        # This-Directory-Only/scripts/torchify_np4_manual.py::EXCLUDE. Keep in
        # sync with pipeline/export_questions_json.py.
        "torch_dialect_overrides_np4.jsonl",
        # Fifth conversion pass (2026-07-28): the 17 parked CNN/backprop drills
        # that no lesson tags yet. Converted so the bank holds ONE dialect —
        # eleven of them only ever had a vestigial `import numpy as np`. Keep in
        # sync with pipeline/export_questions_json.py.
        "torch_dialect_overrides_parked.jsonl",
        # Hand-authored payloads for course-written questions (curated_additions.csv)
        # plus targeted fixes the generated layers cannot express. Layered LAST so
        # a deliberate hand edit is never re-overwritten by a bulk pass. Keep in
        # sync with pipeline/export_questions_json.py.
        "curated_overrides.jsonl",
    ):
        layer = _load_jsonl_overrides(layer_name)
        for qid, record in layer.items():
            merged = dict(base.get(qid, {}))
            merged.update(record)
            base[qid] = merged
    return base


_ATOM_TAGS_PATH = Path(__file__).resolve().parent / "data" / "question_atom_tags.jsonl"
_HINTS_PATH = Path(__file__).resolve().parent / "data" / "question_hints.jsonl"
_SOLUTION_NB_PATH = Path(__file__).resolve().parent / "data" / "question_solution_notebooks.jsonl"
# Per-question PROBLEM notebooks (starter, no answer) — only torch questions get
# one, since those route to Colab instead of the in-app runner.
_PROBLEM_NB_PATH = Path(__file__).resolve().parent / "data" / "question_problem_notebooks.jsonl"


def _apply_solution_aids(questions: List["Question"]) -> None:
    """Attach the per-question Show-Hint text and Show-Answer Colab path from
    data/question_hints.jsonl and data/question_solution_notebooks.jsonl.

    Both are produced by scripts/solution_build (one solution notebook per bank
    question). Missing files / malformed lines are skipped silently — a question
    with no hint or notebook simply renders neither button.
    """
    hints: Dict[int, str] = {}
    if _HINTS_PATH.exists():
        try:
            for line in _HINTS_PATH.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                h = (rec.get("hint") or "").strip()
                if h:
                    hints[int(rec["question_id"] if "question_id" in rec else rec["id"])] = h
        except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            logger.warning("Failed to load question_hints.jsonl: %s", exc)

    nbs: Dict[int, str] = {}
    if _SOLUTION_NB_PATH.exists():
        try:
            for line in _SOLUTION_NB_PATH.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                p = (rec.get("path") or "").strip()
                if p:
                    nbs[int(rec["question_id"] if "question_id" in rec else rec["id"])] = p
        except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            logger.warning("Failed to load question_solution_notebooks.jsonl: %s", exc)

    problem_nbs: Dict[int, str] = {}
    if _PROBLEM_NB_PATH.exists():
        try:
            for line in _PROBLEM_NB_PATH.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                p = (rec.get("path") or "").strip()
                if p:
                    problem_nbs[int(rec["question_id"] if "question_id" in rec else rec["id"])] = p
        except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            logger.warning("Failed to load question_problem_notebooks.jsonl: %s", exc)

    for q in questions:
        if q.id in hints:
            q.hint = hints[q.id]
        if q.id in nbs:
            q.solution_notebook_path = nbs[q.id]
        if q.id in problem_nbs:
            q.problem_notebook_path = problem_nbs[q.id]
    logger.info("Solution aids: %d hints, %d notebooks, %d problem notebooks",
                len(hints), len(nbs), len(problem_nbs))


def _apply_atom_tags(questions: List["Question"]) -> None:
    """Attach concept-graph atom tags (+confidence) to each question from
    data/question_atom_tags.jsonl. Missing file / malformed lines are skipped
    silently — questions without tags simply produce no BKT update on submit.
    """
    if not _ATOM_TAGS_PATH.exists():
        logger.info("No question_atom_tags.jsonl — questions will have no atom tags")
        return
    by_id: Dict[int, List[dict]] = {}
    try:
        for line in _ATOM_TAGS_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tags = []
            for a in rec.get("atoms", []):
                aid = a.get("atom_id")
                conf = float(a.get("confidence", 0.0))
                if aid and 0.0 <= conf <= 1.0:
                    tags.append({"atom_id": aid, "confidence": conf})
            if tags:
                by_id[int(rec["question_id"])] = tags
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("Failed to load question_atom_tags.jsonl: %s", exc)
        return
    tagged = 0
    for q in questions:
        t = by_id.get(q.id)
        if t:
            q.atom_tags = t
            tagged += 1
    logger.info("Applied atom tags to %d/%d questions", tagged, len(questions))


def _load_id_set(filename: str) -> set[int]:
    path = _chatgpt_runtime_dir() / filename
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load %s: %s", filename, exc)
        return set()
    if not isinstance(data, list):
        return set()
    try:
        return {int(item) for item in data}
    except (TypeError, ValueError):
        return set()


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
    primary_library: str = "python"
    task_type: str = "stdout_prediction"
    expected_artifact_type: str = "stdout"
    supports_visual_output: bool = False
    function_name: str | None = None
    starter_code: str | None = None
    test_cases: List[dict] = field(default_factory=list)
    submission_mode: str = "stdout"
    # Concept-graph atoms this question's solution exercises, each with a
    # confidence ∈ [0,1] (see data/question_atom_tags.jsonl). Drives the
    # per-atom BKT mastery update on submit. Empty until tags are loaded.
    atom_tags: List[dict] = field(default_factory=list)
    hint: str | None = None
    solution_notebook_path: str | None = None
    # Repo-relative PROBLEM Colab (starter, no answer) for torch questions that
    # route to Colab instead of the in-app runner. None for in-app questions.
    problem_notebook_path: str | None = None


# ---------------------------------------------------------------------------
# Module-level store – populated once by load_questions()
# ---------------------------------------------------------------------------
_questions: List[Question] = []
_questions_by_id: Dict[int, Question] = {}
_questions_by_subtopic: Dict[str, List[Question]] = {}
_subtopics: List[str] = []
_subtopic_to_topic: Dict[str, str] = {}  # "Numpy: Core array literacy" -> "Numpy"
_questions_loaded = False
_load_lock = Lock()

# Manual curation: remove questions that are effectively copy/paste of the prompt.
_CURATED_EXCLUDED_IDS = {9, 20, 21, 33, 39, 44, 45, 57, 88, 161, 188, 203, 221, 222, 223, 226}


# Pure derivation helpers (difficulty classification, library/task-type
# inference, fixture + test-case derivation) live in app/question_derivation.py
# — they are stateless text transforms and were pushing this file past the
# 700-LOC ceiling. Re-imported here so `from app.questions import ...` keeps
# working for every existing caller.
from app.question_derivation import (  # noqa: F401
    _classify_difficulty,
    _derive_test_case,
    _extract_display_variable,
    _extract_prompt_setup_code,
    _infer_default_fixture_setup,
    _infer_primary_library,
    _infer_required_arr_batch,
    _infer_task_type,
    _is_assignment_statement,
    _is_visual_einops_prompt,
    _looks_like_expression,
    _requires_float_fixture,
    _split_answer_steps,
    _strip_display_calls,
)


def compose_full_solution(starter_code: str | None, answer_code: str) -> str:
    """Render a paste-ready full solution: starter imports + fixtures, with
    the ``def solve(): …`` stub replaced by the function-form answer.

    Users want to select-all and paste a working file, not stitch a snippet
    into the existing editor body. Returning the complete script lets them
    do exactly that and submit it as-is.
    """
    text = (answer_code or "").strip()
    # Parameterized-regen answers are already COMPLETE paste-ready scripts
    # (imports + fixture + def solve(<params>) + example invocation). Splicing
    # them into the starter would nest/mangle them (bit the visual regen:
    # solution_code came back as a def-inside-def with broken indentation).
    if re.search(r"(?m)^def\s+solve\s*\(", text):
        return text
    answer_func = wrap_answer_as_function(answer_code)
    if not answer_func:
        return starter_code or ""
    if not starter_code:
        return f"{answer_func}\n\nprint(solve())\n"
    # Match the `def solve(...):` line plus its indented body. The body ends
    # at the next blank-or-non-indented top-level statement (typically
    # `print(solve())`).
    pattern = re.compile(
        r"^def\s+solve\s*\([^)]*\)\s*:\s*\n(?:[ \t]+[^\n]*\n|[ \t]*\n)*",
        re.MULTILINE,
    )
    match = pattern.search(starter_code)
    if not match:
        return f"{starter_code.rstrip()}\n\n{answer_func}\n\nprint(solve())\n"
    return starter_code[: match.start()] + answer_func + "\n\n" + starter_code[match.end():]


def wrap_answer_as_function(answer_code: str) -> str:
    """Render answer_code as a paste-ready ``def solve(): …`` block.

    The CSV / override answer_code is often a bare top-level expression or
    assignment (e.g. ``einops.rearrange(img, 'h w c -> c h w')`` or
    ``arr4 = einops.repeat(...)``). The starter is a function stub, so a raw
    paste mismatches indentation and produces no return value. Wrapping at
    display time lets the user select-all + paste the solution into the
    editor and submit it as-is — without changing the stored ``answer_code``
    used by the AI judge or the grader.
    """
    text = (answer_code or "").strip()
    if not text:
        return text
    # A full-script answer (imports/fixture lines before its own def solve)
    # is already function-form — wrapping would nest def-inside-def.
    if re.search(r"(?m)^def\s+solve\s*\(", text):
        return text
    # Drop any display_array_as_img(...) side-effect calls.
    cleaned = re.sub(r";?\s*display_array_as_img\([^)]*\)", "", text).strip()
    if not cleaned:
        return text
    # Split on `;` and newlines while preserving statement order.
    stmts = [s.strip() for s in re.split(r";|\n", cleaned) if s.strip()]
    if not stmts:
        return text
    *body, last = stmts
    # `print(EXPR)` as the trailing statement: the starter wraps `print(solve())`
    # already, so return EXPR so stdout matches the canonical.
    print_match = re.match(r"^\s*print\s*\((.*)\)\s*$", last, flags=re.DOTALL)
    if print_match:
        last = print_match.group(1).strip()
    last_assign = re.match(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+)$", last)
    if last_assign and not last.startswith(("if ", "for ", "while ")):
        return_expr = last_assign.group(2).strip()
        body_block = "".join(f"    {s}\n" for s in body)
        return f"def solve():\n{body_block}    return {return_expr}"
    body_block = "".join(f"    {s}\n" for s in body)
    return f"def solve():\n{body_block}    return {last}"


def _derive_starter_docstring(question_text: str, task_type: str | None) -> str:
    if not question_text:
        base = "Implement the requested computation and return the result."
    else:
        cleaned = re.sub(r"\s*\(★[★☆]+\)", "", question_text).strip()
        cleaned = re.sub(r"\n?\s*Print the result\.?\s*$", "", cleaned, flags=re.IGNORECASE).strip()
        first_line = cleaned.splitlines()[0].rstrip(".")
        base = f"{first_line}."
    if task_type == "image_transform":
        return f"{base}\n\n    Return the resulting numpy array."
    return f"{base}\n\n    Return the value the prompt asks for."


def _derive_function_payload(
    question_text: str,
    answer_code: str,
    primary_library: str,
    task_type: str | None = None,
) -> tuple[str | None, str | None, List[dict], str]:
    import_line = "import numpy as np"
    if primary_library in {"einops", "einops.einsum"}:
        import_line += "\nimport einops\nfrom einops import einsum, rearrange, reduce, repeat"

    task_type = task_type or _infer_task_type(
        question_text.split(":")[0] if ":" in question_text else "",
        question_text,
        answer_code,
        "",
    )
    docstring = _derive_starter_docstring(question_text, task_type)

    if task_type == "image_transform":
        fixture_setup = _extract_prompt_setup_code(question_text) or _infer_default_fixture_setup(
            question_text, answer_code, primary_library
        )
        visual_setup, visual_expr = _derive_test_case(answer_code, question_text, primary_library)
        top_level_setup = f"{fixture_setup}\n\n" if fixture_setup else ""
        starter_code = (
            f"{import_line}\n\n"
            f"{top_level_setup}"
            "def solve():\n"
            f'    """{docstring}"""\n'
            "    raise NotImplementedError()\n\n\n"
            "print(solve())\n"
        )
        test_case = {
            "setup_code": fixture_setup,
            "call": "solve()",
            "expected_expr": visual_expr or "None",
        }
        if visual_setup and visual_setup != fixture_setup:
            test_case["expected_setup_code"] = visual_setup
        test_cases = [test_case]
        return "solve", starter_code, test_cases, "function"

    setup_code, expected_expr = _derive_test_case(answer_code, question_text, primary_library)
    # Legitimate fixtures only — never leak the answer's setup into the starter.
    fixture_setup = (
        _extract_prompt_setup_code(question_text)
        or _infer_default_fixture_setup(question_text, answer_code, primary_library)
    )
    top_level_setup = f"{fixture_setup}\n\n" if fixture_setup else ""
    starter_code = (
        f"{import_line}\n\n"
        f"{top_level_setup}"
        "def solve():\n"
        f'    """{docstring}"""\n'
        "    raise NotImplementedError()\n\n\n"
        "print(solve())\n"
    )
    # Grader still uses the derived setup so expected_expr resolves to the canonical value.
    test_cases = [{"setup_code": setup_code or fixture_setup, "call": "solve()", "expected_expr": expected_expr or "None"}]
    return "solve", starter_code, test_cases, "function"


def _load_csv_into(
    path: Path,
    questions: List[Question],
    start_id: int,
    skip_rows: int = 0,
    overrides: Optional[Dict[int, dict]] = None,
    deleted_ids: Optional[set[int]] = None,
    broken_ids: Optional[set[int]] = None,
) -> int:
    """
    Load questions from a single CSV file into the provided list.
    Subtopics are stored as "{Topic}: {Subtopic}" for uniqueness.
    Returns the next available ID after loading.

    `overrides` / `deleted_ids` / `broken_ids` come from the AI quality-fix
    pipeline (see _load_function_overrides / _load_id_set). Override fields
    (function_name, starter_code, test_cases, submission_mode) replace the
    CSV-derived values when present. IDs in the deleted/broken sets are
    skipped entirely.
    """
    if not path.exists():
        logger.warning("Questions CSV not found at %s — skipping", path)
        return start_id

    overrides = overrides or {}
    deleted_ids = deleted_ids or set()
    broken_ids = broken_ids or set()

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
            if qid in _CURATED_EXCLUDED_IDS or qid in deleted_ids or qid in broken_ids:
                idx += 1
                continue

            try:
                difficulty_score = int(float(raw_difficulty))
            except ValueError:
                difficulty_score = 50

            difficulty_label = _classify_difficulty(question_text, difficulty_score)
            primary_library = _infer_primary_library(topic, question_text, answer_code)
            task_type = _infer_task_type(topic, question_text, answer_code, expected_output)
            function_name, starter_code, test_cases, submission_mode = _derive_function_payload(
                question_text, answer_code, primary_library, task_type
            )

            override = overrides.get(qid)
            # Default visual-output flags from inferred task_type; an override
            # can flip them per-id without changing the rest of the pipeline.
            expected_artifact_type = "image" if task_type == "image_transform" else "stdout"
            supports_visual_output = task_type == "image_transform"
            if override:
                function_name = override.get("function_name", function_name)
                starter_code = override.get("starter_code", starter_code)
                test_cases = override.get("test_cases", test_cases)
                submission_mode = override.get("submission_mode", submission_mode)
                task_type = override.get("task_type", task_type)
                question_text = override.get("question_text", question_text)
                answer_code = override.get("answer_code", answer_code)
                # Reference stdout shown beside the drill. Override-able because
                # a rewrite that changes answer_code invalidates the CSV's
                # Output column — a torch conversion changes the repr itself
                # (`tensor([1., 1.])`, not `[1. 1.]`). Keep in sync with the
                # exporter.
                expected_output = override.get("expected_output", expected_output)
                if "expected_artifact_type" in override:
                    expected_artifact_type = override["expected_artifact_type"]
                if "supports_visual_output" in override:
                    supports_visual_output = bool(override["supports_visual_output"])
                if "difficulty_score" in override:
                    try:
                        difficulty_score = int(override["difficulty_score"])
                        difficulty_label = _classify_difficulty(question_text, difficulty_score)
                    except (TypeError, ValueError):
                        pass

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
                    primary_library=primary_library,
                    task_type=task_type,
                    expected_artifact_type=expected_artifact_type,
                    supports_visual_output=supports_visual_output,
                    function_name=function_name,
                    starter_code=starter_code,
                    test_cases=test_cases,
                    submission_mode=submission_mode,
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
      4. CNN CSV    — IDs after einops; bridges to atoms with topic="CNNs"

    If csv_path is given (e.g. in tests), only that file is loaded using
    the numpy CSV layout (2 empty header rows).
    """
    global _questions, _questions_by_id, _questions_by_subtopic, _subtopics, _subtopic_to_topic, _questions_loaded

    questions: List[Question] = []

    overrides = _load_function_overrides()
    deleted_ids = _load_id_set("function_mode_deleted_ids.json")
    broken_ids = _load_id_set("function_mode_broken_ids.json")
    if overrides or deleted_ids or broken_ids:
        logger.info(
            "AI overrides applied: %d records, %d deleted, %d broken",
            len(overrides), len(deleted_ids), len(broken_ids),
        )

    if csv_path is not None:
        # Legacy / test path — single CSV, numpy layout
        _load_csv_into(
            csv_path, questions, start_id=1, skip_rows=2,
            overrides=overrides, deleted_ids=deleted_ids, broken_ids=broken_ids,
        )
    else:
        next_id = _load_csv_into(
            NUMPY_CSV_PATH, questions, start_id=1, skip_rows=2,
            overrides=overrides, deleted_ids=deleted_ids, broken_ids=broken_ids,
        )
        next_id = _load_csv_into(
            EINSUM_CSV_PATH, questions, start_id=next_id, skip_rows=0,
            overrides=overrides, deleted_ids=deleted_ids, broken_ids=broken_ids,
        )
        next_id = _load_csv_into(
            EINOPS_CSV_PATH, questions, start_id=next_id, skip_rows=0,
            overrides=overrides, deleted_ids=deleted_ids, broken_ids=broken_ids,
        )
        next_id = _load_csv_into(
            CNN_CSV_PATH, questions, start_id=next_id, skip_rows=0,
            overrides=overrides, deleted_ids=deleted_ids, broken_ids=broken_ids,
        )
        _load_csv_into(
            CURATED_ADDITIONS_CSV_PATH, questions, start_id=next_id, skip_rows=0,
            overrides=overrides, deleted_ids=deleted_ids, broken_ids=broken_ids,
        )

    _apply_atom_tags(questions)
    _apply_solution_aids(questions)

    # By-id stays COMPLETE even when serving is restricted: past attempts,
    # served_question_ids in stored state, and an in-flight question a client
    # already holds must all still resolve. Only the SELECTION pools below are
    # narrowed — see lessons.kc_only_serving().
    _questions_by_id = {q.id: q for q in questions}

    if lessons.kc_only_serving():
        servable = [q for q in questions if lessons.has_target_kcs(q.id)]
    else:
        servable = questions
    # Dialect gate: a torch lesson followed by a numpy drill teaches the wrong
    # muscle memory, so un-converted questions park themselves rather than
    # contradict the lesson that just ran — see lessons.torch_only_serving().
    if lessons.torch_only_serving():
        servable = [
            q for q in servable
            if lessons.is_torch_dialect(q.answer_code, q.starter_code)
        ]
    parked = len(questions) - len(servable)
    _questions = servable

    by_sub: Dict[str, List[Question]] = {}
    sub_to_topic: Dict[str, str] = {}
    for q in servable:
        by_sub.setdefault(q.subtopic, []).append(q)
        sub_to_topic[q.subtopic] = q.topic
    _questions_by_subtopic = by_sub
    _subtopics = sorted(by_sub.keys())
    _subtopic_to_topic = sub_to_topic

    logger.info(
        "Loaded %d questions across %d subtopics (%d parked; kc_only=%s torch_only=%s)",
        len(servable),
        len(_subtopics),
        parked,
        lessons.kc_only_serving(),
        lessons.torch_only_serving(),
    )
    _questions_loaded = True


def ensure_questions_loaded() -> None:
    global _questions_loaded

    if _questions_loaded:
        return

    with _load_lock:
        if _questions_loaded:
            return
        load_questions()


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------

def get_all_questions() -> List[Question]:
    ensure_questions_loaded()
    return _questions


def get_question_by_id(qid: int) -> Optional[Question]:
    ensure_questions_loaded()
    return _questions_by_id.get(qid)


def get_questions_by_subtopic(subtopic: str) -> List[Question]:
    ensure_questions_loaded()
    return _questions_by_subtopic.get(subtopic, [])


def get_subtopics() -> List[str]:
    ensure_questions_loaded()
    return list(_subtopics)


def get_questions_by_subtopic_and_difficulty(subtopic: str, difficulty_label: str) -> List[Question]:
    ensure_questions_loaded()
    return [q for q in _questions_by_subtopic.get(subtopic, []) if q.difficulty_label == difficulty_label]


def get_topic_for_subtopic(subtopic: str) -> str:
    """Return the topic name for a given subtopic key (e.g. 'Numpy' for 'Numpy: Core array literacy')."""
    ensure_questions_loaded()
    return _subtopic_to_topic.get(subtopic, subtopic.split(":")[0] if ":" in subtopic else "")


def get_atoms_for_subtopic(subtopic: str) -> List[str]:
    """Return the distinct concept-graph atom ids exercised by a subtopic's
    questions (union of their atom tags). This is how BKT prioritization +
    difficulty map a per-subtopic surface onto per-atom mastery, replacing the
    old per-subtopic EWMA. Empty if no question in the subtopic carries tags.
    """
    ensure_questions_loaded()
    seen: Dict[str, None] = {}
    for q in _questions_by_subtopic.get(subtopic, []):
        for tag in q.atom_tags or []:
            aid = tag.get("atom_id")
            if aid:
                seen.setdefault(aid, None)
    return list(seen)
