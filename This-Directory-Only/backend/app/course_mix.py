"""Course share: a fixed fraction of practice served from an ENABLED course's
own exercise concepts, the rest from the graph the way it is served today.

Generalized from the single-course `arena_mix.py` (2026-09-20) to N courses
(`docs/spec-multi-course-catalog.md`, 2026-09-25) when the Courses tab grew a
second course (Delta Drills) beside ARENA. The mechanics are unchanged; every
public function that used to be ARENA-only now loops the learner's ENABLED
courses (`course_registry.COURSE_IDS`, in registry order) instead of checking
one flag. A single enabled course reproduces the old arena_mix behaviour
exactly — this file is not a new design, just its N-course shape.

One pool per enabled course, decided by the learner's scope (`practice_target`,
still ARENA-only — Delta Drills has no scoped sub-target):

  * a COURSE pool — the concepts that course's own exercises are tagged to
    (`course_registry.milestones`). A drill on one of these IS that course's
    own problem or an alternative on the same concept.
  * the GRAPH — everything else: the prerequisites, served by the frontier /
    struggle / cap rules exactly as before.

`user_state.course_shares[course_id]` (0 = off/disabled, the default; 0.4 the
Courses tab's fixed "enable" value) is the fraction of ANSWERED drills that
should come from that course's pool. Each request compares the course's
fraction of the last WINDOW answers to its share: under it, this could be
that course's turn. At most ONE course wins a given turn — `_wanted_course`
picks the first enabled course (registry order) whose fraction is under its
share — so two enabled courses never fight over the same drill; the loser
falls back to being served as ordinary graph work on its off turns, same as
before, and both stay force-unlocked regardless (`unlocks`).

What a course's own turn changes (identical to the old ARENA-only story, now
per the WINNING course):

  * that course's concepts count as UNLOCKED regardless of prerequisite
    mastery (`kc_graph.kc_is_unlocked` asks `unlocks`) — the learner meets
    the course's problem before the 0.85 gate says they are ready, for EVERY
    enabled course at once, not only the one whose turn it is;
  * the picker serves ONLY the winning pool (`rounds`), falling back to
    everything else when it is dry — a dry pool never 409s;
  * the pool is ordered by READINESS — mean prerequisite mastery, highest
    first — so the exercise just past the gate comes before the far ones
    (`order`); enabled courses are grouped in registry order, the frontier's
    coreness order kept for the rest;
  * a struggling concept in the winning pool is NOT redirected to its
    prerequisites on its own turn (`holds`). Every OTHER turn (graph, or
    another course's turn) does the remediating: a course's concepts are
    walked FIRST in `remediation.targets`, so a course exercise the learner
    is struggling on sends its weakest prerequisite to the top.

A drill counts as a course's when ANY concept it targets is in that course's
pool, on both the turn's accounting and the picker's exclusion. See
arena_mix's original note on 2026-09-20: the bank held no drill tagged across
two pools at the time this was ARENA-only; unchanged assumption, re-measure if
a Delta-Drills MC item and an ARENA concept are ever tagged on the same row
(they should never be — the MC lane's `kc` is always its own course's).
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Set

from app import attempt_history, course_registry, kc_graph, kc_prefs, practice_targets

# Answered drills the turn decision looks back over.
WINDOW = 10

# The share the Courses tab's binary enable toggle sets (no fine-tune slider
# this pass — docs/spec-multi-course-catalog.md).
DEFAULT_ENABLE_SHARE = 0.4


def share(user_state, course_id: str) -> float:
    shares = getattr(user_state, "course_shares", None) or {}
    try:
        s = float(shares.get(course_id, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, s))


def enabled_courses(user_state) -> List[str]:
    """Registry order, share > 0 only."""
    return [c for c in course_registry.COURSE_IDS if share(user_state, c) > 0.0]


def milestones(user_state, course_id: str) -> List[str]:
    """The concepts inside the learner's scope for this course, course order."""
    reg = kc_graph.registry()
    if course_id == "arena" and getattr(user_state, "practice_target", "all") == practice_targets.RAY:
        return [k for k in practice_targets.ray_kcs() if k in reg]
    return [k for k in course_registry.milestones(course_id) if k in reg]


def _pool(user_state, course_id: str) -> Set[str]:
    return set(milestones(user_state, course_id))


def course_fraction(user_state, course_id: str) -> Optional[float]:
    """Share of the last WINDOW answered drills that touched this course's
    concepts; None with no answers yet."""
    recent = attempt_history.recent_question_sequence(user_state, WINDOW)
    if not recent:
        return None
    pool = _pool(user_state, course_id)
    hits = sum(1 for qid in recent if pool & set(kc_graph.question_kcs(qid)))
    return hits / len(recent)


def _turn(s: float, f: Optional[float]) -> bool:
    """This course's turn? Under its share, or no history yet; a share of 1
    is always its turn (a full window of its own answers reads 1.0, never
    < 1.0). The window sets the resolution: 5% with ten answers is one drill
    in about eleven (codex, 2026-09-20, carried over unchanged)."""
    if s <= 0.0:
        return False
    return s >= 1.0 or f is None or f < s


def serving_courses(user_state) -> List[str]:
    """The courses practice serves, registry order: the studied ones
    (course_registry.studied), a standalone course only while its toggle is
    on — the same two rules course_registry.course_off applies."""
    return [c for c in course_registry.studied(user_state)
            if c == "arena" or share(user_state, c) > 0.0]


def _courses_of(qid) -> Set[str]:
    return {course_registry.course_of(k) for k in kc_graph.question_kcs(qid)}


def course_turns(user_state) -> List[str]:
    """🔑 TWO OR MORE COURSES = AN EVEN SPLIT (Seth, 2026-10-01: "if you
    have both checked it does both of them 50/50"). The order this request
    tries the served courses in: ROUND ROBIN — the one answered LONGEST ago
    first (never, within the last WINDOW answers, before anything; then
    registry order) — so two courses alternate drill by drill, three take
    turns. Not "fewest of the last WINDOW": that balances the window, so a
    learner who ticks a second course after ten ARENA answers got five of
    the new course in a row and then four ARENA in a row paying it back.
    A course that has nothing to serve falls through to the next (`rounds`),
    so a dry course never 409s the request.

    Until then a second course was only an exercise SHARE beside ARENA's
    graph, and the split came out wherever the pools happened to be dry. A
    replay with ARENA and LeetCode both ticked: a fresh learner got 30 of 30
    LeetCode (ARENA's exercise concepts were not servable yet, so every
    ARENA turn fell back to LeetCode, and LeetCode's own turns took the
    rest); Seth's state got 30 of 30 ARENA (see practice_targets.includes).

    One course: [that course], and the old share logic decides everything."""
    courses = serving_courses(user_state)
    if len(courses) < 2:
        return courses
    last = dict.fromkeys(courses, -1)
    for i, qid in enumerate(attempt_history.recent_question_sequence(user_state, WINDOW)):
        for c in _courses_of(qid) & set(courses):
            last[c] = i
    return sorted(courses, key=lambda c: (last[c], courses.index(c)))


def _exercise_turn(user_state, course_id: str) -> bool:
    """Inside one course's turn of an even split: that course's own share,
    counted over ITS recent answers only — the other course's answers would
    halve the fraction and serve ARENA's exercises twice as often as set.
    A standalone course's pool is the whole course, so after its first
    answer this is False, the same as with that course alone: its concepts
    are remediated to their prerequisites (`holds`), not held."""
    s = share(user_state, course_id)
    if s <= 0.0:
        return False
    n = WINDOW * len(serving_courses(user_state))
    recent = [q for q in attempt_history.recent_question_sequence(user_state, n)
              if course_id in _courses_of(q)][-WINDOW:]
    if not recent:
        return _turn(s, None)
    pool = _pool(user_state, course_id)
    return _turn(s, sum(1 for q in recent if pool & set(kc_graph.question_kcs(q))) / len(recent))


def _wanted_course(user_state) -> Optional[str]:
    """The one enabled course whose turn this is, or None (a graph turn).
    First enabled course (registry order) under its own share wins — the
    same tie-break every other stateless-turn decision in this file uses.
    With an even split (`course_turns`), the course whose turn it is, on its
    own exercise share's turn (`_exercise_turn`); a graph turn otherwise."""
    turns = course_turns(user_state)
    if len(turns) > 1:
        return turns[0] if _exercise_turn(user_state, turns[0]) else None
    for c in enabled_courses(user_state):
        if _turn(share(user_state, c), course_fraction(user_state, c)):
            return c
    return None


def unlocks(user_state, kc: str) -> bool:
    """With any course enabled, its concepts in scope are servable whatever
    their prerequisites' mastery says — every enabled course, not only the
    one whose turn this is."""
    return any(kc in _pool(user_state, c) for c in enabled_courses(user_state))


def holds(user_state, kc: str) -> bool:
    """Serve this concept itself rather than its prerequisite: it's in the
    pool of the course whose turn this is."""
    wanted = _wanted_course(user_state)
    return wanted is not None and kc in _pool(user_state, wanted)


def rounds(user_state) -> List[Set[str]]:
    """Skip-sets for `question_pick.run_queue`: the half NOT wanted this turn
    first, then the other, so a dry half falls back instead of 409ing. One
    empty round with nothing enabled.

    A course's own turn: try its pool alone (round 0 excludes everything
    else), then everything else as fallback (round 1 excludes its pool) —
    "everything else" includes other enabled courses' concepts, which is
    correct: a dry winning pool should still prefer them over cold graph
    concepts no more than it prefers anything else in round 1.

    A graph turn (no course wins, but ≥1 is enabled): try plain graph first
    (round 0 excludes every enabled course's pool), then every enabled
    course's pool as fallback (round 1 excludes the graph rest) — identical
    to the old two-pool shape when exactly one course is enabled."""
    turns = course_turns(user_state)
    if len(turns) > 1:
        return _split_rounds(user_state, turns)
    enabled = enabled_courses(user_state)
    if not enabled:
        return [set()]
    wanted = _wanted_course(user_state)
    if wanted is not None:
        pool = _pool(user_state, wanted)
        return [set(kc_graph.registry()) - pool, pool]
    everyone = set().union(*(_pool(user_state, c) for c in enabled))
    rest = set(kc_graph.registry()) - everyone
    return [everyone, rest]


def _split_rounds(user_state, turns: List[str]) -> List[Set[str]]:
    """`rounds` for an even split: each course in turn order, everything
    outside it skipped. ARENA's turn keeps ARENA's own two halves (its
    exercise pool vs the rest of its graph, `_exercise_turn` first), so its
    share still means what it meant with ARENA alone; a standalone course's
    pool IS the course, one round. Every course is a round, so a dry course
    falls through to the next, and a course down to review repeats yields to
    the next one's fresh work (question_pick)."""
    reg = set(kc_graph.registry())
    out: List[Set[str]] = []
    for c in turns:
        own = {k for k in reg if course_registry.course_of(k) == c}
        others = reg - own
        pool = _pool(user_state, c) & own
        if share(user_state, c) > 0.0 and pool and own - pool:
            halves = [own - pool, pool]
            if not _exercise_turn(user_state, c):
                halves.reverse()
            out.extend(others | h for h in halves)
        else:
            out.append(others)
    return out


def _readiness(user_state, kc: str) -> float:
    pre = [p for p in (kc_graph.registry_node(kc) or {}).get("prereqs") or []
           if not kc_prefs.is_disabled(user_state, p)]
    if not pre:
        return 1.0
    return sum(kc_graph.kc_mastery(user_state, p, decay=False)[0] for p in pre) / len(pre)


def order(user_state, kcs: Iterable[str]) -> List[str]:
    """The frontier with every enabled course's concepts first (registry
    order, then most-ready first within a course), the rest keep their
    order. Identity with nothing enabled."""
    kcs = list(kcs)
    enabled = enabled_courses(user_state)
    if not enabled:
        return kcs
    course_rank = {c: i for i, c in enumerate(enabled)}
    in_a_pool: dict = {}
    for c in enabled:
        milestone_rank = {k: i for i, k in enumerate(milestones(user_state, c))}
        for k in kcs:
            if k in in_a_pool:
                continue
            if k in milestone_rank:
                in_a_pool[k] = (course_rank[c], -_readiness(user_state, k), milestone_rank[k], k)
    prioritized = sorted(in_a_pool, key=lambda k: in_a_pool[k])
    return prioritized + [k for k in kcs if k not in in_a_pool]


# --- the two writers ---------------------------------------------------------
# `course_shares` and `study_courses` are coupled: a standalone course is on
# exactly while it is studied, and a course left out has no share. Both
# endpoints (practice/settings_router.py) go through these two, so the pair
# cannot drift apart.


def toggle(user_state, course_id: str, enabled: bool) -> None:
    """The Courses tab toggle: the tick IS "practice serves this course",
    ARENA included (Seth, 2026-09-29 — see course_registry.studied).
    Enabling sets the fixed share and studies the course; disabling zeroes
    the share and stops studying it. The first toggle of a learner never
    asked "which courses?" writes down the set they were being served, so
    ticking LeetCode beside a default ARENA keeps ARENA.

    Raises ValueError when the toggle would leave no course at all."""
    study = set(course_registry.studied(user_state))
    if enabled:
        study.add(course_id)
    else:
        study.discard(course_id)
    study = course_registry.normalize_study(study)
    if not study:
        raise ValueError("Keep at least one course in practice — tick another course first.")
    shares = dict(user_state.course_shares or {})
    shares[course_id] = DEFAULT_ENABLE_SHARE if enabled else 0.0
    user_state.course_shares = shares
    user_state.study_courses = study


def set_study(user_state, picked: List[str]) -> None:
    """The onboarding answer (already normalized, non-empty). A course left
    out gets share 0, ARENA's mix included; a standalone course picked is
    switched on at the fixed share, unchanged if already on. ARENA picked
    keeps whatever mix it has — re-picking it does not switch its exercise
    mix on, which the learner never asked for."""
    shares = dict(user_state.course_shares or {})
    for c in course_registry.COURSE_IDS:
        if c not in picked:
            shares[c] = 0.0
        elif c != "arena" and share(user_state, c) <= 0.0:
            shares[c] = DEFAULT_ENABLE_SHARE
    user_state.course_shares = shares
    user_state.study_courses = list(picked)


def status(user_state, course_id: str) -> dict:
    """What the Courses tab card shows for one course."""
    f = course_fraction(user_state, course_id)
    s = share(user_state, course_id)
    # In an even split the course whose turn it is, whatever half of it the
    # turn serves (codex, 2026-10-01: a LeetCode turn read "other").
    turns = course_turns(user_state)
    wanted = turns[0] if len(turns) > 1 else _wanted_course(user_state)
    return {
        "course": course_id,
        "share": s,
        "enabled": s > 0.0,
        # Practice serves it (course_registry.course_off) — what the Courses
        # tab's tick shows. With no onboarding answer, the ticked courses, or
        # ARENA alone when nothing is ticked.
        "studied": not course_registry.course_off_by_study(user_state, course_id)
        and (course_id == "arena" or s > 0.0),
        "scope": getattr(user_state, "practice_target", "all") if course_id == "arena" else "all",
        "course_kcs": len(milestones(user_state, course_id)),
        "window": WINDOW,
        "recent_fraction": f,
        "next_turn": None if s <= 0.0 else ("this" if wanted == course_id else "other"),
    }
