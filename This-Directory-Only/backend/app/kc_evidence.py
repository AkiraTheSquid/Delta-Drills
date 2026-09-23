"""How much one graded answer says about a concept NOW — the weight kc_explore
multiplies its Bayes factor by.

Seth, 2026-09-23: "the ultimate source of knowledge is whether the learner was
able to do the arena problems themselves. And so the more similar the problems
are to the original arena problems, that's what matters most." And: "maybe
after a long time the uncertainty increases and so it is more likely to do a
probe."

Two factors, multiplied:

  SIMILARITY — how close the item is to the ARENA problem the concept exists
    for. The ARENA exercise itself and its variants (lessons/
    arena_exercise_kcs.json) are the thing being predicted, weight 1; a
    whole-KP integrated problem is most of it; a single-move independent drill
    is a piece of it. Formally the answer's likelihood ratio is TEMPERED,
    BF**w — the standard way to count evidence from a related-but-not-identical
    task at a fraction of its face value (a power prior). Math MC items keep
    weight 1: kc_explore's MC likelihoods were calibrated on them directly.

  RETENTION — how much of that answer still describes the learner today. The
    FSRS-6 curve of the concept's own memory (app/memory_model.py): an answer
    made `t` days ago is weighted R(t) = (1 + F·t/S)^-d with the concept's
    CURRENT stability S. A fresh answer counts in full; an answer from before a
    long gap counts less, so a settled concept drifts back toward its prior,
    its variance p(1−p) rises, and the probe ranking picks it again. A concept
    practised on a spaced schedule has a large S and keeps its evidence far
    longer than one crammed in an afternoon — the spacing effect, which a
    single global half-life cannot express. Tempering (not forgetting toward
    "unknown") is deliberate: a long gap makes the old answer less
    INFORMATIVE in either direction — an old miss fades too — which is what
    "uncertainty increases" means.

memory_model is read lazily and optionally: without it (or for a concept it
has no memory of) the weight is similarity alone.

🔴 ORDER AND BELIEF ONLY. Nothing here writes state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Dict, Optional

# Tempering exponent by how close the item is to the ARENA problem.
W_ARENA = 1.0        # the exercise itself, or one of its variants
W_INTEGRATED = 0.75  # a whole-KP problem (`kp-integrated`)
W_DRILL = 0.5        # a single-move drill (`kp-independent`, unranked)

_INTEGRATED_RANK = 3


def rung(qid: int) -> str:
    """"integrated" (a whole-KP problem) or "drill" — the one reading of the
    ladder rank both the evidence weight and the ready route's pools use."""
    from app import kc_graph
    return "integrated" if kc_graph.ladder_rank(int(qid)) == _INTEGRATED_RANK else "drill"


def arena_ids() -> Dict[str, frozenset]:
    """kc -> the ARENA exercises' own question ids (original + variants)."""
    from app import diagnostic
    return {k: frozenset(v.get("question_ids") or ()) for k, v in diagnostic._arena_links().items()}


def similarity(kc: str, qid: Optional[int], arena: Optional[Dict[str, frozenset]] = None) -> float:
    """Tempering weight for an answer to `qid` as evidence on `kc`."""
    if kc.startswith("math."):
        return 1.0
    if qid is None:
        return W_DRILL
    arena = arena if arena is not None else arena_ids()
    if int(qid) in arena.get(kc, ()):
        return W_ARENA
    return W_INTEGRATED if rung(qid) == "integrated" else W_DRILL


def _days(ts) -> Optional[float]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() / 86400.0


def retention_fn(user_state, now: Optional[datetime] = None) -> Callable[[str, object], float]:
    """(kc, ts) -> R of an answer on `kc` made at `ts`, as of `now`.

    1.0 whenever the memory model, the concept's memory, or the timestamp is
    missing — an answer is never discounted on a guess."""
    try:
        from app import memory_model
        mems = memory_model.memories(user_state)
        cfg = memory_model.DEFAULT_CONFIG
    except Exception:
        return lambda kc, ts: 1.0
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    t_now = now.timestamp() / 86400.0

    def weigh(kc: str, ts) -> float:
        mem = mems.get(kc)
        t = _days(ts)
        if mem is None or t is None or not mem.S > 0:
            return 1.0
        elapsed = max(0.0, t_now - t)
        return max(0.0, min(1.0, (1.0 + cfg.factor * elapsed / mem.S) ** cfg.decay))

    return weigh


def weigher(user_state, now: Optional[datetime] = None) -> Callable[[str, dict], float]:
    """(kc, attempt row) -> similarity × retention for one ladder attempt."""
    arena = arena_ids()
    retain = retention_fn(user_state, now)

    def weigh(kc: str, attempt: dict) -> float:
        qid = attempt.get("question_id")
        try:
            qid = int(qid) if qid is not None else None
        except (TypeError, ValueError):
            qid = None
        return similarity(kc, qid, arena) * retain(kc, attempt.get("ts"))

    return weigh
