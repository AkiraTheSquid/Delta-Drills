"""Per-learner curriculum targets and fast, section-specific placement.

Readiness inferred from a passed composite permits targeted practice; it is
not a mastery measurement or a fabricated ladder attempt. A later miss
revokes that inference. All-course practice retains its original gates.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

RAY = "raytracing-0.1"
MAX_PROBES = 8
BUDGET_SECS = 25 * 60


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


def includes(user_state, kc):
    scope = scope_kcs(getattr(user_state, "practice_target", "all"))
    return scope is None or kc in scope


def allows_question(user_state, qid):
    from app import kc_graph
    scope = scope_kcs(getattr(user_state, "practice_target", "all"))
    targets = kc_graph.question_kcs(qid)
    return scope is None or bool(targets) and set(targets) <= scope


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
    return {**{k: placement(user_state)["completed_at"] for k in readiness(user_state)},
            **user_state.kc_exposure}


def pick_probe(user_state, candidates):
    """Jump two exercise milestones after a pass; bisect after a miss.

    Following a miss, spend one probe on a direct prerequisite. Diagnostic
    probes are always unscaffolded Solo/Integrated items, including retakes.
    """
    from app import diagnostic, kc_graph
    from app.questions import get_question_by_id
    d = diagnostic.get_diag(user_state)
    probes = d["probes"]
    used = {p["question_id"] for p in probes}
    # A fresh diagnostic may reuse old practice items if its bank is spent,
    # but never repeats a probe within this run.
    pools = {}
    for kc in diagnostic.assessed_kcs(user_state):
        ids = [i for i in kc_graph.questions_for_kc(kc) if i not in used
               and kc_graph._qmatrix().get(i, {}).get("source") in ("kp-independent", "kp-integrated")
               and set(kc_graph.question_kcs(i)) <= set(diagnostic.assessed_kcs(user_state))]
        qs = [q for i in ids if (q := get_question_by_id(i)) is not None]
        if qs:
            pools[kc] = qs
    if not pools:
        return None, None
    milestones = ray_kcs()
    tested = {p["kc"] for p in probes}
    last = probes[-1] if probes else None
    chosen = None
    if last and last["result"] != "correct" and last["kc"] in milestones:
        parents = kc_graph._registry()[last["kc"]]["prereqs"]
        eligible = [k for k in parents if k in pools and k not in tested]
        if eligible:
            beliefs = diagnostic.beliefs(user_state)
            chosen = min(eligible, key=lambda k: (beliefs.get(k, .5), k))
    passes = [milestones.index(p["kc"]) for p in probes
              if p["kc"] in milestones and p["result"] == "correct"]
    misses = [milestones.index(p["kc"]) for p in probes
              if p["kc"] in milestones and p["result"] != "correct"]
    lo = max(passes, default=-1)
    hi = min((i for i in misses if i > lo), default=len(milestones))
    aim = (lo + hi) // 2 if misses else min(len(milestones) - 1, lo + 2)
    remaining = [k for k in milestones if k in pools and k not in tested
                 and (not misses or lo < milestones.index(k) < hi)]
    if chosen is None and remaining:
        chosen = min(remaining, key=lambda k: (abs(milestones.index(k) - aim), milestones.index(k)))
    if chosen is None:
        # Resolve the prerequisite chain of the failed exercise, not arbitrary
        # roots elsewhere in the curriculum.
        gaps = [p["kc"] for p in probes if p["result"] != "correct"]
        relevant = ancestors(gaps) if gaps else set(milestones)
        remaining = [k for k in pools if k in relevant and k not in tested]
        if not remaining:
            return None, None
        chosen = max(remaining, key=lambda k: len(ancestors([k])))
    fresh = set(sum((list(a) + list(g) for a, g in candidates.values()), []))
    qs = pools[chosen]
    unseen = [q for q in qs if q.id in fresh]
    qs = unseen or qs
    target = 60 if not last else min(90, float(last.get("difficulty") or 50) + 15) if last["result"] == "correct" else 45
    q = min(qs, key=lambda q: (abs(float(q.difficulty_score or 50) - target), q.id))
    return chosen, q
