"""Practice until ready — the notebook exercise dialog's route (2026-09-23).

Seth: "starting with the concept, and practicing that, and then if you are not
ready for it, it essentially routes backward to try to probe what the
prerequisite weakness is using the probe. And for whichever weakness you have
it essentially does that for the more greedy portion." And: "this is also a
probe ... it should be basically the same thing" as the math explorer.

So there is ONE belief — app/kc_explore.py's posterior, run over the target's
prerequisite closure instead of the math area, with code likelihoods and each
answer tempered by app/kc_evidence.py (similarity to the ARENA problem ×
retention since it was made). This module only decides what to ask next with
it; it stores nothing. Every call replays the learner's ladder record, so the
route survives a reload, a pause, or a second device for free.

THE ROUTE, one question per call:

  1. READY when P(target known now) >= READY_P (kc_explore.SETTLE_P): stop.
  2. ATTEMPT the target — a variant of the exercise — when this is the
     session's first question, when the target is not weak, or when a
     weakness has been fixed since the last attempt. The harder problem
     first, and the attempt is itself the test of whether more prerequisites
     are needed.
  3. Otherwise walk backward from the target. At each node:
       * a direct prerequisite that is a WEAKNESS → descend into it (the
         least-held first). A weakness is a concept missed in this session
         and not yet at READY_P, or one the record already refutes
         (P <= OUT_OF_STATE). Once found, it stays one until it is fixed —
         that is the greedy portion.
       * else a direct prerequisite still UNCLEAR (OUT_OF_STATE < P < CLEAR_P)
         → PROBE it: the one with the largest expected variance reduction,
         valued toward this node (kc_explore.scored_probes). A probe is served
         as the prerequisite's integrated problem where it has one — the
         closest thing to the ARENA problem is the strongest evidence. A miss
         makes it a weakness, and the walk goes down through it next time.
       * else → DRILL this node until it reaches READY_P: single-move drills
         while P < 0.5, then its integrated problem (for the target, the
         exercise's variants).

So a miss at any level sends the route one level down, and a fixed weakness
sends it straight back up to the problem. No count is set in advance;
MAX_ITEMS is a fuse, not a budget.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from app import kc_evidence, kc_explore, kc_graph, kc_prefs

READY_P = kc_explore.SETTLE_P
OUT_P = kc_explore.OUT_OF_STATE
# "Probably holds" for DIAGNOSIS only: a prerequisite this likely is not the
# weakness the miss above it points at, so it is not probed again. One correct
# answer on any item from the prior clears it (drill: 0.5 → 0.74). Settling it
# still takes READY_P.
CLEAR_P = 0.70
# A fuse against a pathological loop, far above any honest route.
MAX_ITEMS = 60

def closure(target: str) -> List[str]:
    """The target and every concept it transitively rests on, target first.
    A concept the learner switched off is not in it (and counts as held)."""
    reg = kc_graph._registry()
    out, todo = [], [target]
    while todo:
        kc = todo.pop(0)
        if kc in out or kc not in reg:
            continue
        out.append(kc)
        todo.extend(reg[kc]["prereqs"])
    return out


def _direct(reg: Dict[str, dict], kc: str, area: Sequence[str]) -> List[str]:
    return [p for p in reg[kc]["prereqs"] if p in area]


def _hops_from(reg: Dict[str, dict], root: str, area: Sequence[str]) -> Dict[str, int]:
    dist, todo = {root: 0}, [root]
    while todo:
        kc = todo.pop(0)
        for p in _direct(reg, kc, area):
            if p not in dist:
                dist[p] = dist[kc] + 1
                todo.append(p)
    return dist


# --- items ------------------------------------------------------------------------


def _servable(qid: int) -> bool:
    from app.questions import get_question_by_id
    try:
        return get_question_by_id(int(qid)) is not None
    except Exception:
        return False


def pools(kc: str, exercise_ids: Iterable[int], arena: Dict[str, frozenset]) -> Dict[str, List[int]]:
    """{"arena", "integrated", "drill"} question ids for `kc`. `arena` is the
    exercise's own original + variants for the target, and the ARENA problems
    of the concept for anything else."""
    ex = [int(q) for q in exercise_ids or ()]
    arena_ids = ex if ex else sorted(arena.get(kc, ()))
    taken = set(arena_ids)
    integrated, drill = [], []
    for q in kc_graph.questions_for_kc(kc):
        if q in taken or kc_graph.ladder_rank(q) in (0, 1):  # retired faded/guided
            continue
        (integrated if kc_evidence.rung(q) == "integrated" else drill).append(q)
    return {"arena": arena_ids, "integrated": integrated, "drill": drill}


def _last_seen(user_state, kc: str) -> Dict[int, str]:
    seen = {}
    for a in kc_graph.ladder_view(user_state, kc).get("attempts") or []:
        if a.get("question_id") is not None:
            seen[int(a["question_id"])] = a.get("ts") or ""
    return seen


def _pick(user_state, kc: str, order: Sequence[str], pool: Dict[str, List[int]],
          skip: set) -> Optional[tuple]:
    """(rung, qid): the first rung in `order` with an unserved, servable item,
    least recently answered first."""
    seen = _last_seen(user_state, kc)
    for rung in order:
        fresh = [q for q in pool.get(rung) or () if q not in skip and _servable(q)]
        if fresh:
            fresh.sort(key=lambda q: (seen.get(q, ""), q))
            return rung, fresh[0]
    return None


def _has_items(pool: Dict[str, List[int]], skip: set) -> bool:
    return any(q not in skip for rung in pool.values() for q in rung)


PROBE_ORDER = ("integrated", "arena", "drill")
ATTEMPT_ORDER = ("arena", "integrated", "drill")


def _drill_order(p: float, is_target: bool) -> tuple:
    if p < 0.5:
        return ("drill", "integrated", "arena")
    return ("arena", "integrated", "drill") if is_target else ("integrated", "drill", "arena")


# --- the route --------------------------------------------------------------------


def plan(user_state, target: str, exercise_ids: Iterable[int] = (),
         served: Iterable[int] = (), skip: Iterable[int] = ()) -> dict:
    """The next question of a ready-route session on `target`, or done.

    `served` = question ids already asked in this session (never re-asked);
    `skip` = ids the client could not hydrate. Read-only."""
    reg = kc_graph._registry()
    if target not in reg:
        return {"done": True, "reason": "unknown", "target": target}
    served = [int(q) for q in served or ()]
    skip_set = set(served) | {int(q) for q in skip or ()}
    area = [k for k in closure(target) if k == target or not kc_prefs.is_disabled(user_state, k)]
    attempts = {k: kc_graph.ladder_view(user_state, k).get("attempts") or [] for k in area}
    ev = kc_explore.evidence(attempts, kc_evidence.weigher(user_state))
    level = getattr(user_state, "self_reported_level", None)
    graph = kc_explore._area_graph(reg, area)
    L = kc_explore.logodds(reg, ev, level, graph)
    P = {k: kc_explore._sigmoid(x) for k, x in L.items()}
    arena = kc_evidence.arena_ids()
    pool = {k: pools(k, exercise_ids if k == target else (), arena) for k in area}

    # This session's answers, in order: [(kc, correct)] per served question,
    # read back from the ladder rows the submit path wrote. A served id is
    # never re-served, so its newest row is this session's answer.
    session = []
    for qid in served:
        for kc in kc_graph.question_kcs(qid):
            if kc not in pool:
                continue
            rows = [a for a in kc_graph.ladder_view(user_state, kc).get("attempts") or []
                    if a.get("question_id") == qid]
            if rows:
                session.append((kc, bool(rows[-1].get("correct"))))
    missed = {kc for kc, ok in session if not ok}

    def weakness(kc):
        return P[kc] <= OUT_P or (kc in missed and P[kc] < READY_P)

    last_attempt = max((i for i, (kc, _ok) in enumerate(session) if kc == target), default=-1)
    fixed_since = any(kc != target and kc in missed and P[kc] >= READY_P
                      for kc, _ok in session[last_attempt + 1:])

    def out(**kw):
        base = {"done": False, "reason": None, "target": target, "p_target": round(P[target], 4),
                "ready_at": READY_P,
                "beliefs": {k: round(P[k], 4) for k in area},
                "path": [target]}
        base.update(kw)
        return base

    if P[target] >= READY_P:
        return out(done=True, reason="ready")
    if len(served) >= MAX_ITEMS:
        return out(done=True, reason="fuse")

    def serve(kc, mode, order, path):
        got = _pick(user_state, kc, order, pool[kc], skip_set)
        if not got:
            return None
        rung, qid = got
        return out(question_id=qid, kc=kc, kc_title=reg[kc].get("title") or kc,
                   rung=rung, mode=mode, attempt=(kc == target and rung == "arena"),
                   p_kc=round(P[kc], 4), path=path)

    # The harder problem first: every session opens on an attempt, and every
    # fixed weakness sends the route back to one.
    # The target's own "weakness" is only a refutation: an earlier miss on it is
    # what the walk below answers, not a reason to stop attempting it.
    if not served or P[target] > OUT_P or fixed_since:
        got = serve(target, "attempt", ATTEMPT_ORDER, [target])
        if got:
            return got

    def servable_prereqs(kc, seen):
        """Direct prerequisites of `kc` that have something left to ask; one
        with nothing is walked THROUGH to its own prerequisites, so a concept
        with no drills never hides the ones under it (codex, 2026-09-23)."""
        out = []
        for p in _direct(reg, kc, area):
            if p in seen:
                continue
            seen.add(p)
            if _has_items(pool[p], skip_set):
                out.append(p)
            else:
                out.extend(servable_prereqs(p, seen))
        return out

    node, path = target, [target]
    visited = {target}
    while True:
        prereqs = [p for p in servable_prereqs(node, set(visited)) if p not in visited]
        weak = sorted((p for p in prereqs if weakness(p)), key=lambda k: (P[k], k))
        if weak:
            node = weak[0]
            visited.add(node)
            path = path + [node]
            continue
        unclear = [p for p in prereqs if OUT_P < P[p] < CLEAR_P]
        if unclear:
            dist = _hops_from(reg, node, area)
            value = {k: 0.5 ** d for k, d in dist.items()}
            weights = {k: kc_evidence.W_INTEGRATED if pool[k]["integrated"] else kc_evidence.W_DRILL
                       for k in unclear}
            for _gain, kc in kc_explore.scored_probes(reg, ev, unclear, value, level, area, weights):
                got = serve(kc, "probe", PROBE_ORDER, path + [kc])
                if got:
                    return got
        got = serve(node, "drill", _drill_order(P[node], node == target), path)
        if got:
            return got
        # This weakness has nothing left to serve. Drill what it rests on
        # instead (the least-held first); if that is nothing, the route is spent.
        rest = sorted((p for p in prereqs if P[p] < READY_P), key=lambda k: (P[k], k))
        for kc in rest:
            got = serve(kc, "drill", _drill_order(P[kc], False), path + [kc])
            if got:
                return got
        return out(done=True, reason="exhausted", path=path)
