"""Per-learner curriculum targets, and the readiness a stored ray placement
still vouches for.

Readiness inferred from a passed composite permits targeted practice; it is
not a mastery measurement or a fabricated ladder attempt. A later miss
revokes that inference. All-course practice retains its original gates. The
ray placement itself (its probe picker) was retired with the placement test
on 2026-09-26; a run stored before then keeps its 14 days.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

RAY = "raytracing-0.1"


def ray_kcs():
    from app import diagnostic, kc_graph
    return list(dict.fromkeys(k for k, row in diagnostic._arena_links().items()
                              if "0-1" in row["notebooks"] and k in kc_graph._registry()))


def ancestors(kcs):
    from app import kc_graph
    reg = kc_graph._registry()
    found = set(kcs)
    todo = list(kcs)
    while todo:
        for p in reg.get(todo.pop(), {}).get("prereqs", []):
            if p not in found:
                found.add(p)
                todo.append(p)
    return found


def scope_kcs(target):
    return ancestors(ray_kcs()) if target == RAY else None


def _arena(kc):
    from app import course_registry
    return course_registry.course_of(kc) == "arena"


# 🔴 The target narrows ARENA, never another course (2026-10-01). It used to
# read every concept outside the 0.1 scope as out, so a learner on the 0.1
# target who ticked LeetCode too was served NO LeetCode at all: all 383
# drills failed allows_question, and the even course split (course_mix.
# course_turns) fell through to ARENA every turn — 30 of 30 in a replay of
# Seth's state.
def includes(user_state, kc):
    scope = scope_kcs(getattr(user_state, "practice_target", "all"))
    return scope is None or kc in scope or not _arena(kc)


def allows_question(user_state, qid):
    from app import kc_graph
    scope = scope_kcs(getattr(user_state, "practice_target", "all"))
    targets = kc_graph.question_kcs(qid)
    return scope is None or bool(targets) and {k for k in targets if _arena(k)} <= scope


def placement(user_state):
    if getattr(user_state, "practice_target", "all") != RAY:
        return {}
    row = (getattr(user_state, "practice_placements", {}) or {}).get(RAY) or {}
    try:
        if datetime.fromisoformat(row["completed_at"]) < datetime.now(timezone.utc) - timedelta(days=14):
            return {}
    except (KeyError, ValueError, TypeError):
        return {}
    return row


def readiness(user_state):
    row = placement(user_state)
    latest = {p["kc"]: p for p in row.get("probes", []) if p.get("kc")}
    passed = {k for k, p in latest.items() if p["result"] == "correct"}
    ready = ancestors(passed)
    ready -= {k for k, p in latest.items() if p["result"] != "correct"}
    for kc in list(ready):
        attempts = (getattr(user_state, "kc_ladder", {}).get(kc) or {}).get("attempts", [])
        if any(not a.get("correct") and a.get("ts", "") > row["completed_at"] for a in attempts):
            ready.discard(kc)
    return ready


def solo_entry(user_state, kc):
    """An unassisted pass places this KC at Solo, until ordinary practice resumes."""
    row = placement(user_state)
    probes = [p for p in row.get("probes", []) if p.get("kc") == kc]
    if not probes or probes[-1]["result"] != "correct":
        return False
    attempts = (getattr(user_state, "kc_ladder", {}).get(kc) or {}).get("attempts", [])
    return not any(a.get("ts", "") > row["completed_at"] for a in attempts)


def effective_exposure(user_state):
    """Skip introductory gates justified by placement without writing lesson completion."""
    from app import lessons
    lessons._load()
    placed_at = placement(user_state).get("completed_at", "")
    placed_map = {}
    for k in readiness(user_state):
        placed_map[k] = placed_at
        for seg in lessons._kc_segments.get(k, []):
            placed_map[f"{k}#{seg['concept_id']}"] = placed_at
    return {**placed_map, **user_state.kc_exposure}
