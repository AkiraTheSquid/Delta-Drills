"""Choose your own concept — the Learner Home's candidate list and the route a
chosen concept runs (Seth, 2026-09-28).

Seth: "for the learner home, there is a list of candidate concepts to work on,
along with displaying your ability for each area ... the candidate concepts
are the concepts that it thinks you would benefit from learning according to
the algorithm. currently the algorithm determines everything, but this would
give the learner some choice". And: "there should be a button that says
'practice with AI' as well as the candidate list (and one of the items is the
one that the ai would have recommended). after they learn the concept, it
brings them back to the original page or if they fail too many times, it
shows a screen saying that they were not ready for that concept".

BOUNDED CHOICE. The list is the algorithm's own: the concept the queue would
serve next (`question_pick.queue_next_kc`, marked as the AI's pick) and then
the knowledge frontier in its serving order (`kc_graph.frontier`) — every
entry is unlocked, so no choice skips a prerequisite. N_CANDIDATES of them.

ABILITY = K, the XP model's knowledge of the concept (app/learning_xp.py):
P(learned) from the per-concept HMM × FSRS retrievability. Seth chose it over
the topbar pill ("it seemed not to really display the actual ability") and
over FSRS alone (memory freshness, not whether it was ever learned). It is
drawn as the meter's fill toward READY. A concept already at the READY line
is left off the list — the XP model counts it done even where the frontier's
own gate (`kc_graph.kc_is_learned`) has not closed yet — except the AI's pick,
which is shown as it is.

XP PER PROBLEM (Seth, 2026-09-29: "not all concepts give the same amount of
exp ... on the far right, make it such that it displays the amount of exp you
would get from solving that sort of problem"): `learning_xp.solve_xp` — what
the model expects one solved problem to add, priced at the concept's worth —
shown to the nearest XP_STEP, never below it while there is anything left to
learn ("on the order of magnitude of like 10XP or 15 xp or 5xp").

THE ROUTE for a chosen concept, one question per call, same response shape as
app/ready_route.py so the client's Route drives both:
  * READY (learned) when K >= READY → the client goes back to the Learner Home;
  * NOT READY when K <= NOT_READY_K after at least MIN_ANSWERS answers in this
    session — the model decides, no count of misses (Seth picked this);
  * else a drill on the concept itself, never a prerequisite: the learner
    chose this concept, and "not ready" is the answer when it is out of reach.
Stateless like the ready route: every call re-reads the learner's record.
"""

from __future__ import annotations

from datetime import timezone
from typing import Dict, Iterable, List, Optional

from app import kc_evidence, kc_graph, learning_xp, ready_route

N_CANDIDATES = 5
READY = learning_xp.READY
NOT_READY_K = 0.20
MIN_ANSWERS = 3
MAX_ITEMS = 40  # a fuse, far above any honest session
XP_STEP = 5


def _read(user_state, kcs: Iterable[str]) -> dict:
    """The replay's per-concept readout now. Day boundaries are irrelevant
    to it, so UTC."""
    return learning_xp.replay(user_state, timezone.utc, also=tuple(dict.fromkeys(kcs)))


def knowledge(user_state, kcs: Iterable[str]) -> Dict[str, float]:
    """K per concept now."""
    kcs = tuple(dict.fromkeys(kcs))
    r = _read(user_state, kcs)
    return {kc: float(r["also_now"].get(kc, 0.0)) for kc in kcs}


def per_problem(xp: float) -> int:
    """XP to show for one problem: the nearest XP_STEP, at least one step
    while the model expects any gain at all — a concept just under READY
    still pays for the problem that takes it there (codex, 2026-09-29: it
    read "+0")."""
    if xp <= 1e-9:
        return 0
    return max(XP_STEP, XP_STEP * int(round(xp / XP_STEP)))


def _pool(kc: str) -> Dict[str, List[int]]:
    return ready_route.pools(kc, (), kc_evidence.arena_ids())


def _servable(kc: str) -> bool:
    return any(ready_route._servable(q) for rung in _pool(kc).values() for q in rung)


def _answers(user_state, kc: str) -> int:
    return len(kc_graph.ladder_view(user_state, kc).get("attempts") or [])


def candidates(user_state, n: int = N_CANDIDATES, keep: Optional[str] = None) -> dict:
    """{"items": [...], "ready_at"} — the AI's pick first.
    `keep`: a paused chosen block's concept, appended when the list would
    leave it out (it left the frontier or the top five mid-block) — the
    list is the only way back into that block."""
    from app.practice.question_pick import queue_next_kc  # app.practice imports this package's routers

    reg = kc_graph._registry()
    frontier = [kc for kc in kc_graph.frontier(user_state) if kc in reg]
    try:
        ai = queue_next_kc(user_state)
    except Exception:  # a dry queue raises deep in the picker; the list still stands
        ai = None
    # The queue's turn can belong to another course (course_mix), whose
    # concepts have no route here: then the graph's own next concept is the pick.
    if ai not in reg or not _servable(ai):
        ai = next((kc for kc in frontier if _servable(kc)), None)
    order = ([ai] if ai else []) + [kc for kc in frontier if kc != ai]
    keep = keep if keep in reg else None
    r = _read(user_state, order + ([keep] if keep else []))
    K, state = r["also_now"], r["also_state"]

    def item(kc):
        node = reg[kc]
        return {
            "kc": kc,
            "title": node.get("title") or kc,
            "lesson_title": node.get("lesson_title"),
            "k": round(K[kc], 4),
            "xp_per_problem": per_problem(learning_xp.solve_xp(kc, state[kc])),
            "worth": learning_xp.worth(kc),
            "answers": _answers(user_state, kc),
            "recommended": kc == ai,
        }

    items = []
    for kc in order:
        if kc != ai and (K[kc] >= READY or not _servable(kc)):
            continue
        items.append(item(kc))
        if len(items) >= n:
            break
    if keep and all(i["kc"] != keep for i in items):
        items.append(item(keep))
    return {"items": items, "ready_at": READY}


def plan(user_state, kc: str, served: Iterable[int] = (), skip: Iterable[int] = ()) -> dict:
    """The next question of a chosen-concept session, or done (see header)."""
    reg = kc_graph._registry()
    if kc not in reg:
        return {"done": True, "reason": "unknown", "target": kc}
    served = [int(q) for q in served or ()]
    skip_set = set(served) | {int(q) for q in skip or ()}
    k = knowledge(user_state, [kc])[kc]
    rows = kc_graph.ladder_view(user_state, kc).get("attempts") or []
    answered = len({q for q in served if any(a.get("question_id") == q for a in rows)})

    def out(**kw):
        base = {"done": False, "reason": None, "target": kc, "p_target": round(k, 4),
                "ready_at": READY, "beliefs": {}, "path": [kc]}
        base.update(kw)
        return base

    if k >= READY:
        return out(done=True, reason="ready")
    if answered >= MIN_ANSWERS and k <= NOT_READY_K:
        return out(done=True, reason="not_ready")
    if len(served) >= MAX_ITEMS:
        return out(done=True, reason="fuse")
    got = ready_route._pick(user_state, kc, ready_route._drill_order(k, False), _pool(kc), skip_set)
    if not got:
        return out(done=True, reason="exhausted")
    rung, qid = got
    return out(question_id=qid, kc=kc, kc_title=reg[kc].get("title") or kc, rung=rung,
               mode="drill", attempt=False, p_kc=round(k, 4))
