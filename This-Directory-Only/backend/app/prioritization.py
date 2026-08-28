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
from app import engine_bridge
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
) -> Tuple[List, Optional[str], Optional[dict]]:
    """Restrict a subtopic's servable questions to the frontier KC the tutor
    actually intends to teach next, and to the RUNG that concept is on.

    A subtopic can hold questions for several frontier KCs at once, so passing
    the gate is not the same as being the next thing to learn. Without this the
    graph could truthfully highlight one node while the queue served a sibling —
    the exact "visualisation that does not represent the system" problem.

    Returns `(candidates, kc, gap)`.

    `gap` is None in the ordinary case. It is a dict — `{"kc", "kc_title",
    "stage", "seen", "total"}` — when the concept's current rung holds nothing
    the learner has not already answered. THAT IS NOT A REASON TO SERVE A
    REPEAT, and it used to be:

        if kc_graph.stage_requires_support(stage):
            supported = set(kc_graph.questions_at_stage(...))
            if supported:
                fresh = [q for q in narrowed if ... q.id not in served]
                return fresh or [q for q in narrowed if q.id in supported], next_kc
                       #        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                       #        every drill on the rung, served or not

    With three faded drills authored for `numpy.ndarray-model`, that `or`
    branch was the steady state within about ten minutes of practice, and the
    learner then went round the same three problems indefinitely. Seth,
    2026-08-28: "I basically memorized all the problems for the first part ...
    the problem is that, like, they're currently repeating ... it should notify
    the user that they need to make the AI create more problems since they ran
    out of problems to practice rather than serving up the old problems they
    have already done."

    So a spent rung reports itself and serves nothing. The caller decides what
    to tell the learner; falling through to the next rung would promote on
    exhaustion rather than on evidence, and repeating is the behaviour being
    removed.
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
        # Nothing here is unserved. Re-ask on membership alone so the concept
        # the learner is actually on can still report its own exhaustion —
        # dropping the narrowing here is what used to hand the difficulty
        # picker the whole subtopic and let a `solo` problem reach somebody
        # sitting on `Worked`.
        next_kc = kc_graph.select_next_kc(user_state, eligible=lambda qid: qid in here)
    if not next_kc:
        return candidates, None, None
    narrowed = [q for q in candidates if q.id in set(kc_graph.questions_for_kc(next_kc))]
    if not narrowed:
        return candidates, None, None

    # Serve the rung the learner's own attempt record says they are on —
    # `kc_stage` owns that decision; here we only honour it.
    stage = kc_graph.kc_stage(user_state, next_kc)
    at_stage = set(kc_graph.questions_at_stage([q.id for q in narrowed], stage))
    rung = [q for q in narrowed if q.id in at_stage]
    if not rung:
        # This concept authored nothing at this rung at all. Not a content gap
        # the learner can do anything about — fall back to the least-scaffolded
        # drill available, which is what they would have been served anyway.
        floor = set(kc_graph.lowest_rung([q.id for q in narrowed]))
        rung = [q for q in narrowed if q.id in floor]

    fresh = [q for q in rung if q.id not in served]
    if not fresh:
        # The learner's own rung is spent. Two different situations, and
        # collapsing them was the first version's mistake:
        #
        #   a) lower rungs still hold problems they have not seen. Serve those.
        #      They are NOT repeats, the learner keeps practising, and the gap
        #      is still reported so the strip can say where the problem came
        #      from and `/drill-gaps` still gets the work item. Stopping here
        #      instead would deadlock the queue: the top rung is the smallest,
        #      and a learner who exhausts it can no longer answer anything, so
        #      no evidence can ever move them off it.
        #   b) nothing anywhere on this concept is unseen. Now there genuinely
        #      is nothing to serve, and the caller 409s.
        #
        # Walk DOWN from the learner's rung rather than up. Serving a rung they
        # have not earned is the promotion-on-exhaustion this whole change
        # removes; serving one they have already left is just review.
        node = kc_graph.registry_node(next_kc) or {}
        gap = {
            "kc": next_kc,
            "kc_title": node.get("title") or next_kc,
            "stage": stage,
            "seen": len(rung),
            "total": len(narrowed),
            "served_from": None,
        }
        order = list(kc_graph.LADDER_STAGES)
        below = order[: order.index(stage)] if stage in order else []
        for lower in reversed(below):
            at_lower = set(kc_graph.questions_at_stage([q.id for q in narrowed], lower))
            spare = [q for q in narrowed if q.id in at_lower and q.id not in served]
            if spare:
                gap["served_from"] = lower
                return spare, next_kc, gap
        # Last resort: drills this KC owns that carry NO rung tag at all.
        # 🔴 This used to be every unseen question in `narrowed`, which quietly
        # included the rungs ABOVE the learner — a spent `faded` rung with an
        # unseen integrated problem behind it handed over that problem, labelled
        # "unranked", and promoted on exhaustion instead of on evidence. That is
        # the one thing this whole change exists to stop. (codex, 2026-08-28.)
        spare = [
            q for q in narrowed
            if q.id not in served and kc_graph.ladder_rank(q.id) == kc_graph.LADDER_UNRANKED
        ]
        if spare:
            gap["served_from"] = "unranked"
            return spare, next_kc, gap
        return [], next_kc, gap

    if stage == "partial":
        # Examples first, then the unaided remainder — see
        # kc_graph.with_example_first. This is the whole of the third-to-fourth
        # rung fade: no schedule, no counter, just "the ones with something to
        # read come first and then there are none left".
        preferred = set(kc_graph.with_example_first([q.id for q in fresh]))
        fresh = [q for q in fresh if q.id in preferred]

    if stage == "worked":
        # The `worked` rung is the concept's first contact, and the question
        # attached to it is what the learner meets the moment they finish
        # reading the example. There is no evidence yet, so `target_difficulty`
        # is only reporting the BKT prior — letting it choose here picks at
        # random inside its band, which on a rung with several drills can open a
        # brand-new concept on its hardest one. Serve the easiest instead; the
        # difficulty ladder starts moving on the next question, from evidence.
        return [min(fresh, key=lambda q: (q.difficulty_score, q.id))], next_kc, None
    return fresh, next_kc, None


def ladder_starter(question, stage: str) -> Optional[str]:
    """The starter to hand the learner at this rung, or None to keep the
    question's own.

    `faded` is now the ONLY rung that overrides. `worked` has no question yet,
    and `partial` and `solo` are both rungs where the learner writes the whole
    function — the difference between them is how many concepts the problem
    needs, not how much of it is pre-written.

    `partial` used to get a half-faded starter as well, which meant the rung
    labelled "write this one yourself" handed over half the answer. Renkl's
    completion problems fade to nothing; they do not fade to half forever.

    At `faded`, an authored starter from the KP wins when one exists: it was cut
    by hand for the idea the lesson just taught, and for a one-statement
    solution it is the only faded form there is (nothing to remove but the whole
    answer). Otherwise the starter is backward-faded from the canonical answer
    (see ladder_fade). A body too short to fade returns None and is served
    unmodified, which is correct — there is no honest middle between "one line
    shown" and "one line hidden".
    """
    if stage != "faded":
        return None
    authored = lessons.authored_faded_starter(getattr(question, "id", -1))
    if authored:
        return authored
    return ladder_fade.fade(
        getattr(question, "answer_code", "") or "",
        getattr(question, "function_name", "") or "solve",
        reveal="most",
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
    stage = kc_graph.kc_stage(user_state, kc)
    return {
        "ladder_stage": stage,
        "ladder_kc": kc,
        "ladder_kc_title": node.get("title"),
        "ladder_estimate": kc_graph.kc_estimate(user_state, kc),
        # Reported, never stored — see lessons.is_integrated for why the record
        # keeps saying `solo` while the strip says something further along.
        "ladder_integrated": stage == "solo"
        and lessons.is_integrated(qid, user_state.kc_exposure),
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


def _aim_mastery(
    user_state: UserPracticeState, subtopic: str, kc: Optional[str]
) -> float:
    """How strong the learner is at the thing they are about to be ASKED.

    Scoped to the concept whenever one is in play, and to the subtopic only
    when none is. A subtopic is not a concept: "Numpy: Core array literacy"
    exercises thirty atoms, so `subtopic_mastery` averages a learner's command
    of the concept in front of them with twenty-five atoms they have never met,
    all sitting at the beginner prior. Measured on a real account, that pinned
    the aim at 24.5/100 across a session in which the concept's own mastery
    reached 0.92 — and the aim finished the session 0.15 points LOWER than it
    started, because BKT decay on the untouched atoms outran the evidence on
    the practised ones. The learner is shown a per-concept estimate and served
    a per-subtopic aim, and the two numbers had no reason ever to agree.

    Which per-concept number: the concept's own attempt record, through the
    Wilson LOWER bound that `kc_estimate` already computes and the strip
    already draws as the left edge of the estimate range. Aiming at the bound
    rather than the raw rate is what makes a thin record behave — at n=0 it is
    0 and the prior below takes over, at 3/3 it is 0.44, at 13/20 it is 0.43,
    at 20/20 it is 0.84 — so the aim rises with demonstrated success and falls
    for a learner who is genuinely struggling, without a lucky pair of answers
    launching them.

    With no attempts on the concept, the atom crosswalk is the better answer
    where it is `measured` (23 of 63 KCs): a fresh concept then opens at the
    learner's own prior rather than at the floor. `topic-proxy` rows are the
    topic's atoms wearing the concept's name, so they fall through to the
    subtopic, which is at least honest about what it averaged.
    """
    if kc:
        # The logistic engine first, once the concept has real evidence behind
        # it. It answers the aim's question directly — P(correct) on a
        # median-difficulty item at the solo rung — and it answers it from a
        # model that knows what the ladder's raw success rate cannot: how hard
        # the items actually were, how much scaffold was on screen, how strong
        # the prerequisites are, and how long ago. A learner going 13/20 on
        # scaffolded easy items and one going 13/20 unaided on hard ones have
        # the same Wilson bound and are not the same learner.
        engine = engine_bridge.mastery(user_state, kc)
        if engine is not None:
            return engine
        est = kc_graph.kc_estimate(user_state, kc)
        if est["n"]:
            return float(est["ci"][0])
        mastery, _covered, tier = kc_graph.kc_mastery(user_state, kc)
        if tier == "measured":
            return float(mastery)
    return subtopic_mastery(user_state, subtopic)


def target_difficulty(
    user_state: UserPracticeState, subtopic: str, kc: Optional[str] = None
) -> float:
    """Target difficulty for the next question, on the concept being served.
    Scales with how strong the learner is at it (see `_aim_mastery`), then adds
    their own correction from the felt-difficulty rating (see
    `adaptive.nudge_difficulty_offset`) — the estimate says how hard they can
    go, the rating says how far off that aim has been landing in practice.

    `kc` omitted means "no concept in play", which is the honest state for the
    subtopic scorer: it ranks every subtopic there is, and there is no single
    concept to measure across a whole subtopic's worth of them.

    `.get`, not `get_subtopic_state`: that one CREATES the row it cannot find,
    and merely ASKING where to aim must not stamp an empty subtopic into stored
    state — this is called from that same scorer, over subtopics the learner
    has never touched."""
    m = _aim_mastery(user_state, subtopic, kc)
    raw = _DIFF_FLOOR + _DIFF_SPAN * m
    sub_state = user_state.subtopic_states.get(subtopic)
    if sub_state is not None:
        raw += sub_state.difficulty_offset
    return max(10.0, min(100.0, raw))


def question_target_difficulty(
    user_state: UserPracticeState, subtopic: str, qid: int
) -> float:
    """The aim, scoped to the concept an ANSWERED question belongs to.

    `question_kcs(qid)[0]` is the same primary KC `ladder_fields` reports, so
    the target the strip redraws after an answer is measured on the concept the
    strip is naming. Serving reads the concept it narrowed to and finalizing
    reads the concept it just graded; both are the concept on screen, and if
    they ever stop agreeing the bar moves on submit and moves back on load.
    """
    kcs = kc_graph.question_kcs(qid)
    return target_difficulty(user_state, subtopic, kc=kcs[0] if kcs else None)


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
