"""
Subtopic prioritization module — BKT-driven (EWMA fully removed).

Decides which subtopic to pull the next question from, and at what difficulty,
using ONLY the per-atom Bayesian Knowledge Tracing posteriors (bkt_mastery.py).
A subtopic's mastery is the mean BKT posterior over the atoms its questions
exercise (see questions.get_atoms_for_subtopic — populated from the per-question
atom tags). The old per-subtopic EWMA gradient/learning-rate is gone.

Selection policy: WEAKEST-FIRST. priority = effective_weight * (1 - mastery).
Un-practiced atoms sit at the BKT prior (~0.10), so fresh subtopics surface
first naturally; decay regresses mastery over time, resurfacing stale ones
without a separate staleness rule.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from app import bkt_mastery
from app import kc_graph
from app import ladder_fade
from app import lessons
from app.adaptive import UserPracticeState
from app.questions import (
    get_atoms_for_subtopic,
    get_subtopics,
    get_questions_by_subtopic,
    get_topic_for_subtopic,
)

logger = logging.getLogger(__name__)

# Difficulty target maps mastery∈[0,1] → numeric difficulty. Low mastery serves
# easy items; near-mastery serves the hardest. v0 affine map, tune empirically.
_DIFF_FLOOR = 20.0
_DIFF_SPAN = 80.0


def _get_weight(user_state: UserPracticeState, st_name: str, uniform_weight: float) -> float:
    """Return the effective weight for a subtopic, using custom weights if set."""
    if user_state.custom_weights:
        return user_state.custom_weights.get(st_name, uniform_weight)
    return uniform_weight


def question_is_unlocked(user_state: UserPracticeState, question) -> bool:
    """PREREQUISITE gate for a bank question. ONE lattice decides, never two.

    The **KC lattice** (`kc_graph`) is the authority whenever the question is
    tagged into it: the knowledge graph is the structure the course was authored
    against and the structure the learner is shown, so it is the structure that
    gets to say what is servable.

    The **atom lattice** (`bkt_mastery`) applies only to questions the q-matrix
    does not place — where the KC graph has nothing to say, the finer BKT
    prerequisite index is better than no gate at all.

    Intersecting the two was tried and is wrong. They disagree: the KC registry
    makes `numpy.ndarray-model` a root with no prerequisites, while the atom
    graph gives its atom `tensor-wraps-ndarray` unmet prerequisites. Requiring
    both left a fresh learner exactly ONE servable question in a 373-question
    bank. Two prerequisite structures voting produces the stricter of two
    disagreements rather than an answer, so the graph decides alone and the
    disagreement becomes a data bug to fix in the atom graph, not a lock.

    Untagged questions pass. Locking out content the q-matrix never placed
    would delete it from the course rather than order it, which The Math Academy
    Way ch. 32 is explicit about not doing ("no topics are skipped").
    """
    qid = getattr(question, "id", None)
    if kc_graph.question_kcs(qid):
        return kc_graph.question_kc_gate(user_state, qid)
    tags = getattr(question, "atom_tags", None) or []
    if not tags:
        return True
    return all(
        bkt_mastery.atom_is_ready(
            t["atom_id"], user_state.atom_mastery, user_state.atom_last_ts
        )
        for t in tags
    )


def narrow_to_next_kc(
    user_state: UserPracticeState, candidates: List, served: Optional[set] = None
) -> List:
    """Restrict a subtopic's servable questions to the frontier KC the tutor
    actually intends to teach next, when the subtopic carries any.

    A subtopic can hold questions for several frontier KCs at once, so passing
    the gate is not the same as being the next thing to learn. Without this the
    graph could truthfully highlight one node while the queue served a sibling —
    the exact "visualisation that does not represent the system" problem. If the
    next KC has nothing left unserved here, the unnarrowed list is returned so
    the learner is never stalled by the preference.
    """
    here = {q.id for q in candidates}
    served = served or set()
    # Eligibility must match what the caller can actually serve, or the
    # narrowing targets a KC whose questions are all spent and hands back a
    # list the difficulty picker then rejects — a 404 with fresh sibling work
    # sitting right there. `select_next_subtopic` chose this subtopic because
    # SOME frontier KC has unserved work in it; find that same KC.
    next_kc = kc_graph.select_next_kc(
        user_state, eligible=lambda qid: qid in here and qid not in served
    )
    if not next_kc:
        return candidates
    narrowed = [q for q in candidates if q.id in set(kc_graph.questions_for_kc(next_kc))]
    if not narrowed:
        return candidates

    # Within the KC, serve the rung the learner's own attempt record says they
    # are on — faded (fill in the blank) or independent (write it unaided).
    # `kc_stage` owns that decision; here we only honour it, and fall back to
    # the least-scaffolded rung still available when the wanted rung is spent,
    # so an exhausted rung can never stall the queue.
    unserved = [q for q in narrowed if q.id not in served]
    if not unserved:
        return narrowed
    stage = kc_graph.kc_stage(user_state, next_kc)
    wanted = kc_graph.questions_at_stage([q.id for q in unserved], stage)
    if wanted:
        return [q for q in unserved if q.id in set(wanted)]

    # The supported rungs are SMALL — `numpy.ndarray-model` authors three faded
    # drills and one guided against five independent ones — so "spent" arrives
    # after a handful of questions and long before the ladder has the twenty
    # attempts it wants to promote on. Falling through to the next rung there is
    # the ladder promoting on exhaustion instead of on evidence: the strip still
    # reads `Worked` (that is what `kc_stage` says, and it is right), the learner
    # is told an example is coming, and what arrives is an unaided problem with
    # nothing above it. Repeating a supported drill is the honest answer — the
    # rung has not been earned yet, and a second pass at a faded drill is
    # ordinary spaced practice. Only a KC with NO supported drill at all may fall
    # through, which is what the rung fallback below still handles.
    if kc_graph.stage_requires_support(stage):
        repeat = kc_graph.questions_at_stage([q.id for q in narrowed], stage)
        if repeat:
            return [q for q in narrowed if q.id in set(repeat)]

    rung = [q for q in unserved if q.id in set(kc_graph.lowest_rung([q.id for q in unserved]))]
    if stage == "worked" and rung:
        # The `worked` rung is the concept's first contact, and the question
        # attached to it is what the learner meets the moment they finish
        # reading the example. There is no evidence yet, so `target_difficulty`
        # is only reporting the BKT prior — letting it choose here picks at
        # random inside its band, which on a rung with several drills can open a
        # brand-new concept on its hardest one. Serve the easiest instead; the
        # difficulty ladder starts moving on the next question, from evidence.
        return [min(rung, key=lambda q: (q.difficulty_score, q.id))]
    return rung


def ladder_starter(question, stage: str) -> Optional[str]:
    """The starter to hand the learner at this rung, or None to keep the
    question's own.

    `worked` and `solo` never override — one has no question yet, the other is
    the rung defined by having no support.

    At `faded`, an authored starter from the KP wins when one exists: it was cut
    by hand for the idea the lesson just taught, and for a one-statement
    solution it is the only faded form there is (nothing to remove but the whole
    answer). Otherwise, and always at `partial`, the starter is backward-faded
    from the canonical answer (see ladder_fade). A body too short to fade
    returns None and is served unmodified, which is correct — there is no honest
    middle between "one line shown" and "one line hidden".
    """
    if stage not in ("faded", "partial"):
        return None
    if stage == "faded":
        authored = lessons.authored_faded_starter(getattr(question, "id", -1))
        if authored:
            return authored
    return ladder_fade.fade(
        getattr(question, "answer_code", "") or "",
        getattr(question, "function_name", "") or "solve",
        reveal="most" if stage == "faded" else "half",
    )


def ladder_fields(user_state: UserPracticeState, qid: int) -> dict:
    """Ladder stage for the question being served, keyed to its primary KC.

    Lives here rather than in the router because `questions_router` is already
    at ORANGE on the structural score and this needs `kc_graph`, which that
    module does not otherwise import.

    Returns {} for an untagged question: with no KC there is no per-concept
    attempt record to place the learner on, and inventing a stage would put a
    scaffold in front of a drill the map does not claim to teach.
    """
    kcs = kc_graph.question_kcs(qid)
    if not kcs:
        return {}
    kc = kcs[0]
    node = kc_graph.registry_node(kc) or {}
    return {
        "ladder_stage": kc_graph.kc_stage(user_state, kc),
        "ladder_kc": kc,
        "ladder_kc_title": node.get("title"),
        "ladder_estimate": kc_graph.kc_estimate(user_state, kc),
    }


def record_ladder_outcome(user_state: UserPracticeState, qid: int, correct: bool) -> None:
    """Log a graded attempt onto the ladder, at the stage it was actually served.

    The stage is recomputed here rather than trusted from the client, and it is
    read BEFORE the attempt is appended — `kc_stage` reflects the record so far,
    which is precisely the rung the learner was just sitting on.
    """
    for kc in kc_graph.question_kcs(qid):
        kc_graph.record_kc_outcome(
            user_state, qid, correct, stage=kc_graph.kc_stage(user_state, kc)
        )
        break  # record_kc_outcome already fans out to every KC the question tags


def subtopic_mastery(user_state: UserPracticeState, subtopic: str) -> float:
    """Mean decay-adjusted BKT posterior over the atoms this subtopic exercises.
    Falls back to the learner's prior (self-reported level, else the BKT
    default) when the subtopic has no tagged atoms / no practice yet — so a
    self-reported beginner starts at the floor and a self-reported strong
    learner starts high, and evidence takes over from the first attempt."""
    params = bkt_mastery.params_for_level(user_state.self_reported_level)
    atoms = get_atoms_for_subtopic(subtopic)
    if not atoms:
        return params.p_init
    vals = [
        bkt_mastery.current_mastery(
            user_state.atom_mastery, user_state.atom_last_ts, a, params=params
        )
        for a in atoms
    ]
    return sum(vals) / len(vals) if vals else params.p_init


def target_difficulty(user_state: UserPracticeState, subtopic: str) -> float:
    """BKT-derived target difficulty for the next question in a subtopic.
    Scales with the learner's mastery of the subtopic's atoms."""
    m = subtopic_mastery(user_state, subtopic)
    raw = _DIFF_FLOOR + _DIFF_SPAN * m
    return max(10.0, min(100.0, raw))


def _difficulty_reach_factor(easiest_available: float, target: float) -> float:
    """Down-weight subtopics whose EASIEST unserved question sits far above the
    learner's target difficulty. Weakest-first alone is perverse for a
    self-reported beginner: every untouched subtopic ties at the prior, so the
    queue happily opens with 'the easiest matmul-backward drill' — which is
    still an advanced problem (tester hit exactly this). A subtopic is only
    servable-at-level if it actually HAS a question near the target; the
    factor fades as mastery (and thus target) rises, so advanced subtopics
    re-enter naturally. 1.0 when reachable, ~0.5 one 8-point gap out."""
    gap = max(0.0, easiest_available - target)
    return 1.0 / (1.0 + gap / 8.0)


def select_next_subtopic(user_state: UserPracticeState) -> Optional[str]:
    """Select the subtopic to pull the next question from — weakest-first by
    BKT mastery, weighted by effective (custom) weight and by whether the
    subtopic has questions reachable at the learner's target difficulty.
    Skips subtopics whose questions are all served; resets served sets if
    everything is exhausted.
    """
    subtopics = get_subtopics()
    if not subtopics:
        return None
    uniform_weight = 1.0 / len(subtopics)

    def _candidates(skip_served: bool) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        for st_name in subtopics:
            available = [
                q for q in get_questions_by_subtopic(st_name)
                if question_is_unlocked(user_state, q)
            ]
            if not available:
                continue
            if skip_served:
                served = set(user_state.get_subtopic_state(st_name).served_question_ids)
                available = [q for q in available if q.id not in served]
                if not available:
                    continue
            weight = _get_weight(user_state, st_name, uniform_weight)
            if weight <= 0:
                continue
            easiest = min(q.difficulty_score for q in available)
            reach = _difficulty_reach_factor(easiest, target_difficulty(user_state, st_name))
            priority = weight * (1.0 - subtopic_mastery(user_state, st_name)) * reach
            out.append((st_name, priority, easiest))
        return out

    # KC LATTICE FIRST. The knowledge graph decides what comes next; the
    # weakest-first machinery below is the fallback for when the lattice has
    # nothing to say (frontier exhausted, or a KC with no servable questions
    # left). Ordering the frontier is `kc_graph`'s job — coreness then depth,
    # per The Math Academy Way ch. 32 — so this only has to translate the KC it
    # picks into a subtopic that actually has an unserved question for it.
    for kc in kc_graph.frontier(user_state):
        wanted = set(kc_graph.questions_for_kc(kc))
        if not wanted:
            continue
        for st_name in subtopics:
            if _get_weight(user_state, st_name, uniform_weight) <= 0:
                continue
            served = set(user_state.get_subtopic_state(st_name).served_question_ids)
            if any(
                q.id in wanted and q.id not in served and question_is_unlocked(user_state, q)
                for q in get_questions_by_subtopic(st_name)
            ):
                return st_name

    cands = _candidates(skip_served=True)
    if not cands:
        # Nothing UNLOCKED and unserved is left, so the next question has to be
        # a repeat. It used to be a repeat with amnesia: this branch cleared
        # `served_question_ids` for every subtopic in the course, which is the
        # only record of what the learner has already solved. A beginner hits
        # this on their second question — the lattice opens with one root KC, so
        # "everything unlocked is served" is the normal state early on, not an
        # end-of-bank condition — and from then on the app genuinely could not
        # tell a solved question from a fresh one.
        #
        # So: no wipe. Fall through to the full set and let
        # select_question_for_difficulty pick the least-recently-served repeat,
        # with the service log intact.
        cands = _candidates(skip_served=False)
    if not cands:
        return None

    # Highest priority (weakest, weighted, reachable) first. Ties — common on
    # fresh accounts where every subtopic sits at the same prior — resolve to
    # the subtopic with the genuinely easiest entry question, THEN alpha (the
    # old alpha-first tiebreak opened beginners on "CNN: Conv2d mechanics").
    cands.sort(key=lambda item: (-item[1], item[2], item[0]))
    return cands[0][0]


def get_subtopic_weights(user_state: UserPracticeState) -> List[Dict]:
    """All subtopics with current prioritization info, sorted weakest-first.

    `baseline`/`p` now carry the BKT subtopic mastery (0-100 and 0-1) so the
    existing frontend score readers (getArenaPrereqSubtopicScore) and the area
    readout reflect BKT, not EWMA. `gradient` is the selection priority.
    """
    subtopics = get_subtopics()
    uniform_weight = 1.0 / len(subtopics) if subtopics else 1.0

    result = []
    for st_name in subtopics:
        sub_state = user_state.get_subtopic_state(st_name)
        weight = _get_weight(user_state, st_name, uniform_weight)
        mastery = subtopic_mastery(user_state, st_name)
        priority = weight * (1.0 - mastery)
        result.append({
            "subtopic": st_name,
            "topic": get_topic_for_subtopic(st_name),
            "weight": weight,
            "learning_rate": priority,          # repurposed: BKT selection priority
            "gradient": priority,
            "questions_answered": sub_state.n,
            "current_difficulty": target_difficulty(user_state, st_name),
            "baseline": mastery * 100.0,        # BKT mastery on the 0-100 scale
            "p": mastery,                       # BKT mastery 0-1 (frontend score)
        })

    return sorted(result, key=lambda r: r["gradient"], reverse=True)
