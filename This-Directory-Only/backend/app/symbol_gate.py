"""Syntax gate: a drill is served only when every symbol it uses has been taught.

The rule Seth stated (2026-09-17), after meeting `a.T` on his second day and
`torch.arange` on a boolean-masking drill: a problem may use a piece of
syntax only if the learner has (a) mastered a concept whose lesson teaches
it, or (b) already read the lesson, in a concept not yet mastered, that
introduces it. Neither → don't show the problem.

`kc_graph.question_kc_gate` covers the drill's own concept and its
prerequisite chain. It says nothing about a symbol owned by a page on a
branch the learner has not walked — the build-time audit
(`scripts/audit_solution_prereqs.py`) used to pass those on registry RANK
alone, and 532 symbol uses on 374 drills sat in that hole. This module reads
`lessons/question_symbol_kcs.json` (built by `scripts/build_symbol_prereqs.py`:
question -> {owner KC: [symbols]}, target KCs excluded) and asks, per owner
KC, learned-or-exposed.

Two owners are deliberately NOT gated on:
- a KC the learner switched off (kc_prefs) — "off" must never lock content;
- a KC DOWNSTREAM of the drill's own concept. That is the audit's `late`
  debt (the page teaching the symbol sits after the drill). Gating on it
  would deadlock: the drill waits for a concept whose prerequisite is the
  drill's own concept. It is served as before and stays on the audit's
  report; the fix is content, not a lock.

Missing index → gate disabled, logged once, same policy as lessons.py.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, List, Set

from app import kc_prefs, lessons

logger = logging.getLogger(__name__)

_INDEX_NAME = "question_symbol_kcs.json"


@lru_cache(maxsize=1)
def _index() -> Dict[int, Dict[str, List[str]]]:
    raw = lessons._read_json(_INDEX_NAME) or {}
    out: Dict[int, Dict[str, List[str]]] = {}
    for qid, owners in (raw.get("questions") or {}).items():
        try:
            out[int(qid)] = {k: list(v) for k, v in owners.items()}
        except (TypeError, ValueError, AttributeError):
            continue
    if not out:
        logger.warning("%s empty or missing — syntax gate disabled", _INDEX_NAME)
    return out


@lru_cache(maxsize=1)
def _ancestors() -> Dict[str, Set[str]]:
    """kc -> every KC transitively before it. Cycle-safe (finite on a bad registry)."""
    from app import kc_graph
    reg = kc_graph._registry()
    memo: Dict[str, Set[str]] = {}

    def walk(kc: str, trail: Set[str]) -> Set[str]:
        if kc in memo:
            return memo[kc]
        out: Set[str] = set()
        for p in (reg.get(kc) or {}).get("prereqs") or []:
            if p in trail:
                continue
            out.add(p)
            out |= walk(p, trail | {kc})
        memo[kc] = out
        return out

    return {kc: walk(kc, set()) for kc in reg}


def blocking_kcs(user_state, qid: int) -> Dict[str, List[str]]:
    """{owner KC: [symbols]} the learner has neither learned nor read, for this drill.

    Empty → the drill passes the syntax gate."""
    owners = _index().get(int(qid))
    if not owners:
        return {}
    from app import kc_graph, practice_targets
    targets = set(kc_graph.question_kcs(qid))
    anc = _ancestors()
    exposure = practice_targets.effective_exposure(user_state)
    out: Dict[str, List[str]] = {}
    for kc, syms in owners.items():
        if kc in targets or kc_prefs.is_disabled(user_state, kc):
            continue
        # Downstream of the drill's own concept: `late` content debt, not a gate.
        if targets & anc.get(kc, set()):
            continue
        if kc_graph.kc_is_learned(user_state, kc):
            continue
        if lessons._kc_is_fully_exposed(kc, exposure):
            continue
        out[kc] = list(syms)
    return out


def question_passes(user_state, qid: int) -> bool:
    return not blocking_kcs(user_state, qid)
