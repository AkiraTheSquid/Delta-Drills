"""What the picker does when a learner is struggling on a concept: go DOWN
the graph, not down the rung.

Until 2026-09-19 every response to a miss stayed inside the concept — drop a
rung (kc_graph._stage_from), show a worked example on the next drill
(example_schedule.after_miss), hand the missed drill back (prioritization's
retry). Seth, 2026-09-19: "the response to a learner who's struggling in that
rung needs to be to transition them to the prerequisites with higher
variation rather than the current concept with easier problems." The lesson
and its worked example are the whole of the novice support; the drills are
high-variance, every pick fresh; and struggle is answered by the
prerequisites. That is what the compass research doc proposed in the first
place ("demotion → drill the weakest prerequisite atoms, with
diversification", Rohrer 2012 / Math Academy ch. 32) — the ladder rungs were
implemented in its place.

Three rules live here, all read by `prioritization` on every pick:

  * `redirect` — a concept the learner is STRUGGLING on (STRUGGLE_MISSES
    misses since their last unaided success on it) is served through its
    weakest prerequisite that still has fresh drills, for a DOSE of fresh
    drills per prerequisite per trigger. One hop first, then the
    prerequisites' prerequisites. When no prerequisite has anything fresh
    left, the concept's own fresh drills are served — struggle never withholds
    work. The dose is what keeps this from being gameable: failing on purpose
    buys at most DOSE easier drills per prerequisite before the concept is
    back (The Math Academy Way ch. "peeling back the knowledge profile").

  * `capped` — no concept is served more than CONSECUTIVE_CAP times in a row
    while another frontier concept has work. Interleaving over blocking; the
    number is the starting heuristic the research doc suggested, not a
    finding.

  * `targets` interleaves DUE REVIEWS with the frontier (2026-09-23): learned
    concepts whose FSRS+FIRe predicted recall has fallen below the target
    (`memory_model.due_reviews`). Before this nothing scheduled review; a
    learned concept came back only as filler once the frontier ran dry.

Stateless: everything is re-derived from the ladder rows and the attempt
history, so a replay of a learner's state reaches the same decisions.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Set

from app import arena_mix, attempt_history, kc_graph, kc_prefs, memory_model

# Misses on a concept, with no unaided correct answer among them, before the
# picker turns to its prerequisites.
STRUGGLE_MISSES = 2
# Fresh drills served on ONE prerequisite per trigger before the picker moves
# to the next prerequisite, or back to the concept.
DOSE = 3
# Consecutive answered drills on one concept before the frontier's next
# concept gets the turn.
CONSECUTIVE_CAP = 3


def _trailing_misses(attempts: List[dict]) -> List[dict]:
    """The concept's misses since its last unaided correct answer, newest
    first. An answer made behind an example neither ends the run nor counts
    as a miss: it says nothing about what the learner can do alone."""
    out: List[dict] = []
    for a in reversed(attempts or []):
        if a.get("correct"):
            if not a.get("example"):
                break
            continue
        out.append(a)
    return out


def struggling(user_state, kc: str) -> bool:
    return len(_trailing_misses(kc_graph.ladder_view(user_state, kc).get("attempts") or [])) >= STRUGGLE_MISSES


def _fresh(user_state, kc: str, answered: Set[int]) -> bool:
    """Does `kc` own a servable drill the learner has not answered and can be
    given right now?"""
    return any(
        qid not in answered
        and kc_graph.ladder_rank(qid) in kc_graph._SERVABLE_RANKS
        and kc_graph.question_kc_gate(user_state, qid)
        for qid in kc_graph.questions_for_kc(kc)
    )


def _dosed(user_state, prereq: str, since: str) -> bool:
    """Has the prerequisite already had its DOSE of drills since the miss that
    triggered this remediation? Read off its own ladder row — the attempts
    are timestamped, and the trigger is the concept's latest miss."""
    rows = kc_graph.ladder_view(user_state, prereq).get("attempts") or []
    return sum(1 for a in rows if (a.get("ts") or "") > since) >= DOSE


def redirect(user_state, kc: Optional[str]) -> Optional[str]:
    """The concept to SERVE for frontier concept `kc`: `kc` itself, or the
    prerequisite the learner should be drilling instead."""
    if not kc or not struggling(user_state, kc):
        return kc
    misses = _trailing_misses(kc_graph.ladder_view(user_state, kc).get("attempts") or [])
    since = misses[0].get("ts") or ""
    answered = attempt_history.answered_question_ids(user_state)
    seen: Set[str] = {kc}
    layer = list((kc_graph.registry_node(kc) or {}).get("prereqs") or [])
    while layer:
        eligible = [
            p for p in layer
            if p not in seen
            and not kc_prefs.is_disabled(user_state, p)
            and _fresh(user_state, p, answered)
            and not _dosed(user_state, p, since)
        ]
        if eligible:
            # Weakest first — the prerequisite the learner most needs.
            return min(eligible, key=lambda p: (kc_graph.kc_mastery(user_state, p, decay=False)[0], p))
        seen.update(layer)
        layer = [
            g for p in layer
            for g in ((kc_graph.registry_node(p) or {}).get("prereqs") or [])
            if g not in seen
        ]
    return kc


def capped(user_state) -> Set[str]:
    """Concepts that have had the last CONSECUTIVE_CAP answers in a row."""
    recent = attempt_history.recent_question_sequence(user_state, CONSECUTIVE_CAP)
    if len(recent) < CONSECUTIVE_CAP:
        return set()
    sets = [set(kc_graph.question_kcs(qid)) for qid in recent]
    return set.intersection(*sets) if sets else set()


def _last_was_review(user_state) -> bool:
    """Was the newest answered drill on concepts the learner had already
    learned (review), rather than on the frontier?

    Read off the learned state NOW, not when the drill was served: the
    frontier answer that completes a concept reads as a review, so the
    frontier gets one extra turn before the reviews. Bounded — the next
    frontier answer flips it back. Exact needs the pick's kind persisted
    with the served id (not done; critic 2026-09-23)."""
    recent = attempt_history.recent_question_sequence(user_state, 1)
    if not recent:
        return False
    kcs = kc_graph.question_kcs(recent[-1])
    return bool(kcs) and all(kc_graph.kc_is_learned(user_state, k) for k in kcs)


def targets(user_state, skip: Optional[Set[str]] = None, cap: bool = True):
    """The frontier, in serving order, with both rules applied: each concept
    redirected (a struggling one to its prerequisite), and with `cap` a
    concept that has just had its run of consecutive answers left out.
    Yields the concept to SERVE. Callers make a second pass with `cap=False`
    so the cap orders work and never withholds it.

    Due REVIEWS (memory_model.due_reviews — learned concepts whose predicted
    recall has fallen below the target, compression order) are interleaved
    with the frontier: after a frontier answer the reviews go first, after a
    review answer the frontier does, so neither starves the other while both
    have work. With no frontier work left the reviews simply run."""
    blocked = capped(user_state) if cap else set()
    # `skip` is tested on the concept SERVED, not the frontier concept it
    # stands for: a concept the caller has excluded (dry, or the other half
    # of the ARENA mix) still sends its struggling learner to a prerequisite
    # that is not excluded. An ARENA concept on its own turn is served as
    # itself (arena_mix.holds), and the ARENA concepts are walked first.
    def new_work():
        for kc in arena_mix.order(user_state, kc_graph.frontier(user_state)):
            yield kc if arena_mix.holds(user_state, kc) else redirect(user_state, kc)

    reviews = memory_model.due_reviews(user_state)
    if not reviews:
        streams = (new_work(),)
    elif _last_was_review(user_state):
        streams = (new_work(), iter(reviews))
    else:
        streams = (iter(reviews), new_work())
    seen: Set[str] = set()
    for stream in streams:
        for target in stream:
            if target in seen or target in blocked or (skip and target in skip):
                continue
            seen.add(target)
            yield target


def select_next_kc(
    user_state, eligible: Optional[Callable[[int], bool]] = None,
    skip: Optional[Set[str]] = None, cap: bool = True,
) -> Optional[str]:
    """`kc_graph.select_next_kc` over `targets`: the first concept with a
    drill `eligible` can serve."""
    for target in targets(user_state, skip=skip, cap=cap):
        qs = kc_graph.questions_for_kc(target)
        if eligible is None or any(eligible(q) for q in qs):
            return target
    return None
