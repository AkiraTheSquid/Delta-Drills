"""ARENA share: a fixed fraction of practice served from the course's OWN
exercise concepts, the rest from the graph the way it is served today.

Seth, 2026-09-20: "I wasn't doing any integrated problems that challenge
multiple parts of the knowledge that I already have ... it's attempting to
make it such that I have 85% ability in all of the different coding areas,
but I could still be practicing subsets of those areas by doing coding
problems that are related to the curriculum itself. ... as an experiment, so
that we can collect more data ... 50% of the problems that I'm doing are from
the arena course itself, and the other 50% are drilling the prerequisites."

Two pools, decided by the learner's scope (`practice_target`):

  * ARENA — the concepts ARENA's own exercises are tagged to
    (`lessons/arena_exercise_kcs.json`, read through `diagnostic._arena_links`).
    Scope `raytracing-0.1` narrows it to that section's exercises; `all` is
    every exercise concept in the registry. A drill on one of these IS an
    ARENA problem or an alternative on the same concept.
  * GRAPH — everything else in scope: the prerequisites, served by the
    frontier / struggle / cap rules exactly as before.

`arena_share` on the learner's state (0 = off, the default; 0.5 = half) is
the fraction of ANSWERED drills that should come from the ARENA pool. Each
request compares the ARENA fraction of the last WINDOW answers to that
number: under it, this is an ARENA turn, otherwise a GRAPH turn. Stateless —
the same history gives the same turn, so a replay agrees with the live app
and the graph's "next up" ring (`question_pick.queue_next_kc`) agrees with
the drill served.

What an ARENA turn changes:

  * the ARENA concepts count as UNLOCKED regardless of prerequisite mastery
    (`kc_graph.kc_is_unlocked` asks `unlocks`) — that is the whole point:
    the learner meets the course's problem before the 0.85 gate says they
    are ready, and the ladder still starts them at the lesson;
  * the picker serves ONLY that pool (`rounds`), falling back to the other
    pool when it is dry — a dry half never 409s;
  * the pool is ordered by READINESS — mean prerequisite mastery, highest
    first — so the exercise just past the gate comes before the far ones
    (`order`); the frontier's coreness order is kept for the rest;
  * a struggling ARENA concept is NOT redirected to its prerequisites on its
    own turn (`holds`) — "makes it such that I continue with the problem".
    The GRAPH turn does the remediating: the ARENA concepts are walked
    FIRST there, so a course exercise the learner is struggling on sends
    its weakest prerequisite to the top of the graph turn
    (`remediation.targets`).

A drill counts as ARENA when ANY concept it targets is in the pool, on both
the turn's accounting and the picker's exclusion (a drill is excluded only
when EVERY concept it targets is excluded). The two agree because the bank
holds no drill tagged across the two pools — 0 of 1367 under either scope
on 2026-09-20; a mixed drill would be servable on a graph turn and counted
as ARENA (codex). Re-measure if such drills get authored.

The experiment's data is in the attempt log as it is: every row carries its
concept, and `arena_fraction` is re-derivable from the history.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Set

from app import attempt_history, kc_graph, kc_prefs, practice_targets

# Answered drills the turn decision looks back over.
WINDOW = 10


def share(user_state) -> float:
    try:
        s = float(getattr(user_state, "arena_share", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, s))


def milestones(user_state) -> List[str]:
    """The ARENA exercise concepts inside the learner's scope, course order."""
    from app import diagnostic
    reg = kc_graph.registry()
    if getattr(user_state, "practice_target", "all") == practice_targets.RAY:
        return [k for k in practice_targets.ray_kcs() if k in reg]
    return [k for k in diagnostic._arena_links() if k in reg]


def _pool(user_state) -> Set[str]:
    return set(milestones(user_state))


def arena_fraction(user_state) -> Optional[float]:
    """Share of the last WINDOW answered drills that touched an ARENA
    concept; None with no answers yet."""
    recent = attempt_history.recent_question_sequence(user_state, WINDOW)
    if not recent:
        return None
    pool = _pool(user_state)
    hits = sum(1 for qid in recent if pool & set(kc_graph.question_kcs(qid)))
    return hits / len(recent)


def _turn(s: float, f: Optional[float]) -> bool:
    """ARENA turn? Under the share, or no history yet; a share of 1 is
    always ARENA (a full window of ARENA answers reads 1.0, never < 1.0).
    The window sets the resolution: 5% with ten answers is one ARENA drill
    in about eleven (codex, 2026-09-20)."""
    if s <= 0.0:
        return False
    return s >= 1.0 or f is None or f < s


def arena_turn(user_state) -> bool:
    s = share(user_state)
    return s > 0.0 and _turn(s, arena_fraction(user_state))


def unlocks(user_state, kc: str) -> bool:
    """With a share set, an ARENA concept in scope is servable whatever its
    prerequisites' mastery says. Both turns see it unlocked: the GRAPH turn
    skips it (`rounds`) unless it is struggling, when its prerequisite is
    what gets served."""
    return share(user_state) > 0.0 and kc in _pool(user_state)


def holds(user_state, kc: str) -> bool:
    """Serve this concept itself rather than its prerequisite: an ARENA
    concept on an ARENA turn."""
    return kc in _pool(user_state) and arena_turn(user_state)


def rounds(user_state) -> List[Set[str]]:
    """Skip-sets for `question_pick.run_queue`: the half NOT wanted this
    turn first, then the other, so a dry half falls back instead of 409ing.
    One empty round when the share is off."""
    if share(user_state) <= 0.0:
        return [set()]
    pool = _pool(user_state)
    rest = set(kc_graph.registry()) - pool
    return [rest, pool] if arena_turn(user_state) else [pool, rest]


def _readiness(user_state, kc: str) -> float:
    pre = [p for p in (kc_graph.registry_node(kc) or {}).get("prereqs") or []
           if not kc_prefs.is_disabled(user_state, p)]
    if not pre:
        return 1.0
    return sum(kc_graph.kc_mastery(user_state, p, decay=False)[0] for p in pre) / len(pre)


def order(user_state, kcs: Iterable[str]) -> List[str]:
    """The frontier with the ARENA concepts first, most-ready first (mean
    prerequisite mastery, then course order); the rest keep their order.
    Identity when the share is off."""
    kcs = list(kcs)
    if share(user_state) <= 0.0:
        return kcs
    pool = _pool(user_state)
    course = {k: i for i, k in enumerate(milestones(user_state))}
    arena = sorted((k for k in kcs if k in pool),
                   key=lambda k: (-_readiness(user_state, k), course.get(k, 0), k))
    return arena + [k for k in kcs if k not in pool]


def status(user_state) -> dict:
    """What the account page shows."""
    f = arena_fraction(user_state)
    s = share(user_state)
    return {
        "share": s,
        "scope": getattr(user_state, "practice_target", "all"),
        "arena_kcs": len(milestones(user_state)),
        "window": WINDOW,
        "recent_fraction": f,
        "next_turn": None if s <= 0.0 else ("arena" if _turn(s, f) else "graph"),
    }
