"""
First-encounter lesson metadata — KC registry + Q-matrix loader.

Backs the exposure guard (Pass 2 of the first-encounter course): before a
question whose target KC the learner has never been taught is served,
next_question attaches a lesson_gate entry pointing at the KP that
introduces the KC. Lesson CONTENT stays on the static frontend
(lessons/lessons_structured.json); the backend only needs the mappings
  question_id -> target KCs        (qmatrix_tags.json)
  kc          -> introducing KP    (lessons_structured.json)

Files resolve from the repo checkout first (local dev), then the Docker
image copy (Local_Deployed_Shared/lessons is COPY'd into the build — see
This-Directory-Only/Dockerfile). Missing files disable the guard rather
than break question serving.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# repo_root/This-Directory-Only/backend/app/lessons.py -> repo_root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LESSONS_DIR = _REPO_ROOT / "Local_Deployed_Shared" / "lessons"

_loaded = False
_question_target_kcs: Dict[int, List[str]] = {}
# kc -> {"kc", "kc_title", "lesson_id", "lesson_title", "topic", "kp_title"}
_kc_gate_info: Dict[str, dict] = {}
# question_id -> the KP author's own faded starter (the `_____` blanks).
# Hand-cut for the concept, so it beats a mechanical fade wherever it exists.
_authored_faded: Dict[int, str] = {}
# Independent-rung question ids the KP gave a worked example to. See
# `has_worked_example` — this is the ladder's third rung, not a content detail.
_applied_with_example: set[int] = set()


def _read_json(name: str) -> Optional[dict]:
    path = _LESSONS_DIR / name
    if not path.exists():
        logger.warning("Lesson metadata missing: %s — exposure guard disabled for it", path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Failed to parse %s: %s", path, e)
        return None


def _load() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True

    qmatrix = _read_json("qmatrix_tags.json") or {}
    for qid_str, tags in qmatrix.items():
        try:
            qid = int(qid_str)
        except (TypeError, ValueError):
            continue
        targets = tags.get("target_kcs") or []
        if isinstance(targets, list) and targets:
            _question_target_kcs[qid] = [str(kc) for kc in targets]

    registry = _read_json("kc_registry.json") or {}
    kc_titles = {kc["id"]: kc.get("title", kc["id"]) for kc in registry.get("kcs", [])}
    lesson_meta = {l["id"]: l for l in registry.get("lessons", [])}

    compiled = _read_json("lessons_structured.json") or {}
    for lesson in compiled.get("lessons", []):
        meta = lesson_meta.get(lesson.get("id"), {})
        for kp in lesson.get("kps", []):
            kc = kp.get("kc")
            if not kc:
                continue
            _kc_gate_info[kc] = {
                "kc": kc,
                "kc_title": kc_titles.get(kc, kp.get("title", kc)),
                "kp_title": kp.get("title", kc),
                "lesson_id": lesson.get("id", ""),
                "lesson_title": lesson.get("title", meta.get("title", "")),
                "topic": lesson.get("topic", meta.get("topic", "")),
            }
            # Segment-level lists shadow the KP-level one (same items, grouped),
            # so read both and let the later write win — they agree by
            # construction, and taking either alone misses single-segment KPs.
            item_lists = [kp.get("faded_items") or []]
            item_lists += [seg.get("faded_items") or [] for seg in kp.get("segments") or []]
            # Guided drills sit on the SAME rung as faded ones
            # (kc_graph._STAGE_TO_RANKS gives `faded` ranks 0 and 1) and had no
            # scaffold at all: hints only, and the mechanical backward fade
            # gives up on a one-statement body. So the rung promised "most of
            # the solution is written — supply the rest" and served a bare
            # `def solve(x)`. `compile_lessons._derived_faded` now writes one,
            # by blanking the calls in the canonical answer's LAST step, and it
            # is only present when something was actually blanked — an
            # unblanked "starter" would be the answer.
            item_lists += [kp.get("guided_items") or []]
            for items in item_lists:
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    starter = item.get("starter_code")
                    try:
                        qid = int(item.get("question_id"))
                    except (TypeError, ValueError):
                        continue
                    if isinstance(starter, str) and starter.strip():
                        _authored_faded[qid] = starter

            # Applied practice: independent-rung drills the KP wrote an example
            # for. Having one is what puts a drill on the ladder's third rung
            # (`partial` — read an example, then write the whole thing) instead
            # of its fourth (`solo` — write it with nothing to read first).
            for item in kp.get("applied_items") or []:
                if not isinstance(item, dict):
                    continue
                try:
                    qid = int(item.get("question_id"))
                except (TypeError, ValueError):
                    continue
                if str(item.get("worked_example_code") or "").strip():
                    _applied_with_example.add(qid)

    logger.info(
        "Lesson metadata: %d tagged questions, %d KCs with introducing KPs, "
        "%d authored faded starters",
        len(_question_target_kcs), len(_kc_gate_info), len(_authored_faded),
    )


def has_worked_example(question_id: int) -> bool:
    """Does an independent-rung drill come with an example above it?

    True only for questions listed under a KP's `## Applied practice`, which is
    the ladder's third rung: the learner writes the whole function, but has just
    read the same move worked through. Everything else on that rung is `solo`.
    """
    _load()
    return int(question_id) in _applied_with_example


def authored_faded_starter(question_id: int) -> Optional[str]:
    """The KP author's faded version of this question, if one was written.

    These are the `_____` blank-filling starters in the lesson frontmatter.
    They are cut by hand for the specific idea the KP teaches, so where one
    exists it is a better `faded` rung than anything derived mechanically from
    the answer — and for a one-statement solution it is the ONLY faded form
    available, since there is no tail to remove.
    """
    _load()
    return _authored_faded.get(question_id)


def unexposed_target_kcs(question_id: int, kc_exposure: Dict[str, str]) -> List[dict]:
    """Gate entries for this question's target KCs the learner has not been
    exposed to. Empty list = no gate (untagged question, all KCs exposed, or
    metadata unavailable)."""
    _load()
    gates = []
    seen = set()
    for kc in _question_target_kcs.get(question_id, []):
        if kc in seen or kc in kc_exposure:
            continue
        seen.add(kc)
        info = _kc_gate_info.get(kc)
        if info:  # a KC with no introducing KP can't be taught — never gate on it
            gates.append(info)
    return gates


def kc_exists(kc: str) -> bool:
    _load()
    return kc in _kc_gate_info


def has_target_kcs(question_id: int) -> bool:
    """True when this question is tagged to at least one KC of the lesson graph
    (lessons/kc_registry.json via qmatrix_tags.json)."""
    _load()
    return bool(_question_target_kcs.get(question_id))


def kc_only_serving() -> bool:
    """Whether the ITS may serve ONLY questions tagged to a lesson-graph KC.

    The lesson graph is being validated chapter by chapter, by the learner
    working through it. Until a chapter's KCs exist and have been walked, its
    questions have no validated structure to be scheduled against — no
    prerequisites, no difficulty ordering, no mastery target — so serving them
    would be guessing dressed as a curriculum. They stay in the bank and stay
    resolvable by id (history and in-flight attempts keep working); they are
    simply not selectable.

    Today that parks the 75 CNN / PyTorch Fundamentals / Autograd / Optimizer
    questions and leaves the 380 Numpy / Einops / Einsum ones servable.

    Set DELTA_KC_ONLY=0 to lift the restriction (e.g. after authoring and
    validating the next chapter's KCs). Default is ON — parking is the safe
    state, so a missing env var must not silently reopen the whole bank."""
    return os.environ.get("DELTA_KC_ONLY", "1").strip().lower() not in {"0", "false", "no"}


_TORCH_IMPORT_RE = re.compile(r"(?m)^\s*(?:import\s+torch\b|from\s+torch[\s.])")


def is_torch_dialect(answer_code: Optional[str], starter_code: Optional[str] = "") -> bool:
    """True when a question drills the PyTorch dialect (torch tensors).

    Derived from the question's own code rather than a hand-set field: a
    converted question imports torch and an unconverted one does not, so the
    marker cannot drift out of sync with what the question actually asks, and
    conversion needs no schema change. Mirrors code_runner.code_uses_torch —
    kept local so two high-fan-in modules don't gain a dependency edge.
    """
    return bool(
        _TORCH_IMPORT_RE.search(answer_code or "")
        or _TORCH_IMPORT_RE.search(starter_code or "")
    )


def torch_only_serving() -> bool:
    """Whether the ITS may serve ONLY PyTorch-dialect questions.

    ARENA is written in torch (`import torch as t`; the 0.0 exercises say
    "using only t.arange and einops.rearrange"), and the lessons are being
    converted to match. A torch lesson followed by a numpy drill teaches the
    wrong muscle memory, so questions the conversion has not reached yet hide
    themselves instead of contradicting the lesson that just ran.

    That makes "add more problems as needed" mechanically safe: converting a
    question unparks it, and nothing has to be remembered or maintained.

    Default is ON as of 2026-07-28: every served lesson (np-1..np-4, einsum,
    einops) now teaches torch, so a numpy drill can only contradict the lesson
    that just ran. What this parks is the residue that has no torch form at
    all — q65's `ndarray.flags.writeable` and the retired structured-dtypes
    drills (record dtypes, datetime64, genfromtxt: a tensor is homogeneous) —
    plus any un-tagged numpy question outside the course. By-id lookup stays
    complete, so history and in-flight attempts still resolve; only the
    SELECTION pools narrow.

    Set DELTA_TORCH_ONLY=0 to serve the numpy residue again."""
    return os.environ.get("DELTA_TORCH_ONLY", "1").strip().lower() not in {"0", "false", "no"}
