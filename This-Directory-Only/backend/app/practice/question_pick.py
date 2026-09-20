"""The per-subtopic pick behind /next-question.

Lifted out of questions_router.py on 2026-09-09 when the router grew a
fallback loop: one subtopic with nothing servable is a content gap to record,
not a reason to serve nothing (see `next_question` for the account of the
brick this fixes). Everything here used to be inline in the router; the rules
are unchanged and the comments travelled with them.
"""

from __future__ import annotations

from app import arena_mix, content_gaps, diagnostic, kc_graph
from app.practice.grading import select_question_for_difficulty
from app.attempt_history import owed_question_ids
from app.prioritization import (
    answered_question_ids,
    narrow_to_next_kc,
    question_is_unlocked,
    rung_gap,
    select_next_subtopic,
    target_difficulty,
)
from app.questions import get_questions_by_subtopic
# The most any PRACTICE drill gets on the clock. Seth, 2026-09-19: "why don't
# we set a limit of 5 minutes" — his own log runs 3–5 minutes per answer, and
# the table below charges 6–20. The table still orders concepts relative to
# each other under this ceiling (a 5:00 python drill stays 5:00); the
# placement is NOT on this path — a probe keeps the table's full number so
# its evidence stays comparable across learners (diagnostic.kc_cap_secs).
PRACTICE_MAX_SECS = 5 * 60


def secs_allowed_for(question_id: int, ladder_kc: str | None) -> int:
    """The answer clock this practice question gets: its concept's cap from
    lessons/placement_time_caps.json — the same number a placement probe on
    that concept gets (diagnostic.kc_cap_secs), so the time is the PROBLEM's,
    never a choice made before the block (Seth, 2026-09-09) — clamped to
    PRACTICE_MAX_SECS. The concept is the one the ladder narrowed to; a
    question served un-narrowed is timed by the LONGEST of the concepts it
    targets (a problem that integrates two needs the longer clock); one with
    no concept at all gets the table's default."""
    kcs = [ladder_kc] if ladder_kc else kc_graph.question_kcs(question_id)
    caps = [diagnostic.kc_cap_secs(kc) for kc in kcs if kc]
    cap = max(caps) if caps else diagnostic.kc_cap_secs(None)
    return min(cap, PRACTICE_MAX_SECS)


class SubtopicDry(Exception):
    """This subtopic has nothing servable right now. `gap` is the dict
    content_gaps.record already stored (None when the pool was simply empty
    for this learner — no KC to name)."""

    def __init__(self, gap: dict | None):
        super().__init__("subtopic dry")
        self.gap = gap


def pick_for_subtopic(
    user_id: str,
    user_state,
    subtopic: str,
    focus_subtopic: str | None,
    exclude_kcs: set | None = None,
    record: bool = True,
    cooldown: bool = True,
):
    """Choose the question to serve from ONE subtopic.

    Returns `(sub_state, question, next_kc, gap)`; raises `SubtopicDry` when
    the subtopic holds nothing this learner can be given — every drill on the
    concept answered, or the only unanswered one is the drill they were just
    shown. The caller decides whether that ends the request or moves the pick
    to another subtopic. Every gap found here is recorded before it is raised,
    on both paths: the rung really did run dry, whatever gets served instead.
    `exclude_kcs` are the concepts already found dry on this request; the
    narrowing skips them so a sibling concept in this subtopic gets its turn.
    `record=False` is the lattice asking what WOULD be served: the pick is
    made in full, but nothing is written down — the gap is recorded when the
    learner actually reaches it.
    """
    sub_state = user_state.get_subtopic_state(subtopic)
    candidates = [
        q for q in get_questions_by_subtopic(subtopic)
        if question_is_unlocked(user_state, q)
    ]
    # A focused request is the learner explicitly opening one concept, so honour
    # that over the queue's own idea of what comes next; on the normal path,
    # keep the served question on the same KC the graph is highlighting. The
    # concept is resolved either way and only the NARROWING is conditional: the
    # aim has to be measured on a concept on both paths, or focused practice
    # keeps the subtopic-wide average this change exists to remove.
    served = set(sub_state.served_question_ids)
    # Two different questions, and collapsing them is what bricked Seth's
    # account on 2026-08-31: `served` is "don't hand back the drill they just
    # skipped" and only the difficulty picker should read it; `answered` is
    # "this learner has given evidence on this drill" and it is the only thing
    # allowed to decide that the course has run out. See
    # prioritization.answered_question_ids.
    #
    # 🔴 Re-collapsed on 2026-09-15 (Antigravity, "prevent repeat drills"):
    # the pool handed to the picker was filtered to `q.id not in served` and
    # an empty result raised SubtopicDry. That is the 08-31 brick again with a
    # different spelling — a skip, a reload or a double-fetch spent the drill,
    # and the Solo rung's missed-retry (prioritization.narrow_to_next_kc,
    # attempt_history.missed_question_ids) became unreachable because every
    # retry is by definition already served. Restored 2026-09-16. The one
    # served-based stop that is legitimate is the last-served guard below.
    answered = answered_question_ids(user_state)
    narrowed, next_kc, gap = narrow_to_next_kc(
        user_state, candidates, served, answered, exclude_kcs=exclude_kcs,
        last_served=user_state.last_served_question_id, cooldown=cooldown,
    )
    # 🔴 Applied BEFORE the exhaustion check, not after. A focused request keeps
    # the whole subtopic pool on purpose, so `narrowed` being empty says nothing
    # about whether there is anything to serve — 409ing on it would tell a
    # focused learner the course had run out while unseen questions from a
    # sibling concept sat in their own pool. Gate on the pool actually used.
    # (codex, 2026-08-28.)
    if focus_subtopic is None:
        candidates = narrowed
    if gap:
        # This concept's current rung holds nothing the learner has not already
        # answered. Serving a repeat is what this replaces — see
        # prioritization.narrow_to_next_kc — so the gap is written down where
        # the /drill-gaps skill will find it, every time it is hit.
        if record:
            content_gaps.record(user_id, gap)
        if not [q for q in candidates if q.id not in answered]:
            # Nothing unseen anywhere on the concept.
            raise SubtopicDry(gap)
    target_diff = target_difficulty(user_state, subtopic, kc=next_kc)
    question = select_question_for_difficulty(
        candidates, target_diff, served, sub_state.served_question_ids
    )
    if question is None:
        raise SubtopicDry(gap)

    # The gate in front of this question teaches whichever concept of the KP
    # the learner has not read yet; the drill stays the ladder's pick. Until
    # 2026-09-18 `lessons.segment_drill` swapped in the concept's FADED item
    # here — the one rung retired on 2026-09-11 — so a segment whose id changed
    # under recompile (a re-titled heading) put a Solo-rung learner back on
    # fill-in-the-blank drills: Seth on `torch.ranges`, five faded items in a
    # row, "trivially easy. a bit ridiculous". A re-teach of the prose is the
    # accepted cost of a re-titled concept; a demotion beneath DRILL_FLOOR is
    # not.

    # 🔴 "NEXT" MAY NOT HAND BACK THE PROBLEM THE LEARNER IS LOOKING AT.
    #
    # Measured on Seth's account, 2026-09-06: `Einops: Rearrange`, concept
    # `einops.pattern-language`, rung `worked`. The concept owns twelve drills
    # in the subtopic and NONE is tagged at the worked rung, so the "authored
    # nothing at this rung" fallback in prioritization.narrow_to_next_kc pinned
    # the pool to `lowest_rung`, which is the single rank-0 drill q345. He had
    # been served it and skipped it, so `fresh` held it (unanswered ≠ spent) and
    # `unshown` was empty — the pool handed to the picker was exactly [345],
    # every time. His served log reads `... 345, 345, 345, ...`: pressing Skip
    # re-rendered the identical problem with no message, indefinitely. "It
    # wouldn't let me go."
    #
    # The existing exhaustion machinery does not fire there, because it asks
    # whether anything is UNANSWERED and 345 is. But from the learner's side
    # this is the same event, and Seth already said what it should do (quoted in
    # narrow_to_next_kc, 2026-08-28): "it should notify the user that they need
    # to make the AI create more problems since they ran out of problems to
    # practice rather than serving up the old problems they have already done."
    #
    # NARROW ON PURPOSE — the last served id, not "is a repeat". Recycling a
    # spent pool is deliberate behaviour with a whole ranking behind it
    # (grading.select_question_for_difficulty), and it is fine as long as it
    # ROTATES: seen-least, longest-ago. What is not fine is the degenerate case
    # where the pool is so small that the rotation returns the question the
    # learner was just shown. That, and only that, is what this catches.
    #
    # `answered` and not `served`: a learner who answered it and asked for
    # another may legitimately get it back on review.
    #
    # 🔴 "On screen" is the last drill served ANYWHERE, not this subtopic's
    # served tail. The tail never expires: a drill skipped weeks ago is still
    # the tail today, and when it is the concept's only unlocked drill — the
    # segment-0 carrier of a KP the learner has not read, q239 on
    # `torch.linalg-basics` — this guard called the concept dry on every
    # request, the walk excluded it, and every rung above stayed locked.
    # Seth, 2026-09-18: four of the six chapter-0.1 concepts sat behind it.
    last_served = user_state.last_served_question_id
    if question.id == last_served and question.id not in answered:
        stuck = gap or rung_gap(user_state, next_kc, question)
        if record:
            content_gaps.record(user_id, stuck)
        raise SubtopicDry(stuck)
    return sub_state, question, next_kc, gap


def run_queue(
    user_id: str,
    user_state,
    subtopic: str | None,
    focus_subtopic: str | None,
    record: bool = True,
) -> tuple[tuple | None, dict | None]:
    """The selection behind /next-question, minus the serving: which subtopic,
    which concept, which drill — retrying past dry concepts. Returns
    `(picked, first_gap)`; `picked` is `pick_for_subtopic`'s tuple or None when
    the course is out of material, and `first_gap` the first exhaustion met on
    the way (the router reports it on the drill it serves instead).

    `subtopic` is a focused pool (single-KC practice from the graph) or None
    for the queue's own choice. Nothing here writes: the caller appends the
    served drill; with `record=False` even the content gaps stay unrecorded.
    """
    # ONE DRY SUBTOPIC DOES NOT END THE REQUEST. Both exhaustion checks in
    # `pick_for_subtopic` used to raise the 409 straight out of this
    # function, and on Seth's own account that was a permanent brick: he
    # finished his placement on 2026-09-09, the lattice put
    # `einops.pattern-language` at the head of his frontier, that concept
    # owns ONE rank-0 drill and no `worked` rung, he had skipped that drill —
    # and from then on every /next-question was a 409 (seven hits in
    # content-gaps.json in 25 seconds, the practice page "went in for half a
    # second and then exited"). Nothing else on the course was ever asked.
    #
    # So a subtopic that has nothing to serve is RECORDED as the gap it is —
    # the /drill-gaps skill still gets its work item — and the selector is
    # asked again with that subtopic excluded. The 409 is kept for the one
    # case it is true in: nothing anywhere on the course can be served. The
    # FIRST gap is the one reported then, and the one attached to a question
    # served from elsewhere, because it names the concept the lattice actually
    # wanted to teach — that is the drill somebody needs to write.
    #
    # A focused request (`?focus_subtopic=`) is the learner opening one concept
    # on purpose, and "this concept has run out" is the honest answer there;
    # it is not retried on a sibling. (Seth, 2026-08-28, quoted in
    # prioritization.narrow_to_next_kc: notify, don't repeat — and a learner
    # who did not pick the concept should not be stopped by it either.)
    #
    # 🔴 EXCLUDE THE CONCEPT, NOT THE SUBTOPIC. A dry pick names one concept
    # (`gap["kc"]`), and the subtopic it sits in usually holds siblings with
    # fresh drills — writing the whole subtopic off would hide those and, when
    # every frontier concept lives in one subtopic, 409 with work still on the
    # shelf (codex, 2026-09-09). So the concept is excluded and the SAME
    # selection is asked again; the subtopic is excluded only when the pick
    # came back dry with no concept to name, or named one already excluded
    # (the narrowing's last resort can hand back a concept off the frontier).
    # Every iteration adds a new concept or a new subtopic to a finite set,
    # which is what terminates the loop.
    # TWO ROUNDS (2026-09-19). The first serves nothing a learner missed
    # inside its cooldown (attempt_history.RETAKE_COOLDOWN): fresh work,
    # then fresh drills anywhere, then review repeats fill the gap and the
    # retake comes back on its own once three other answers have passed.
    # Only when that round finds NOTHING on the whole course is the cooldown
    # relaxed — the retake beats a 409, and a learner whose one remaining
    # drill is a miss would otherwise be bricked (no answer can ever expire
    # a cooldown nothing is served into). A focused request takes both
    # rounds too (its pool alone, the selector never asked): a focused
    # concept whose only work is an in-cooldown retake serves the retake,
    # not a 409 (codex, 2026-09-19).
    # THE ARENA MIX (2026-09-20, app/arena_mix.py). With a share set, each
    # request is an ARENA turn or a GRAPH turn, and the round is run with
    # the OTHER half's concepts excluded from the start — then again with
    # the wanted half excluded, so a dry half falls back to the other rather
    # than 409. A focused request is one concept by the learner's own hand
    # and takes no side.
    # A half whose only work is a REVIEW REPEAT — a drill already answered
    # and not owed a retake — does not win its turn on that: within a
    # cooldown round the other half's fresh work is served first, and the
    # repeat only when both halves are down to repeats. A repeat still ends
    # the round the way any pick did before: the cooldown is relaxed only when
    # a round finds NOTHING. Replay, 2026-09-20: with the 0.1 target the
    # ARENA half spent raytracing.make-rays-2d and recycled its three
    # Integrated drills forty times while the graph half sat on unseen
    # prerequisite work. Without the mix (one round, `[set()]`) nothing
    # changes: the repeat is the only candidate and is served as before.
    first_gap: dict | None = None
    halves = [set()] if focus_subtopic is not None else arena_mix.rounds(user_state)
    for cooldown in (True, False):
        repeat = None
        for half in halves:
            picked, gap = _pick_in_half(user_id, user_state, subtopic, focus_subtopic, half, record, cooldown)
            if first_gap is None:
                first_gap = gap
            if not picked:
                continue
            if len(halves) > 1 and _is_review_repeat(user_state, picked[1].id):
                repeat = repeat or picked
                continue
            return picked, first_gap
        if repeat:
            return repeat, first_gap
    return None, first_gap


def _pick_in_half(user_id, user_state, subtopic, focus_subtopic, half: set, record: bool, cooldown: bool):
    """One pass of the selection with `half`'s concepts excluded from the
    start. Returns `(picked, first_gap)` for this pass — `picked` None when
    nothing in the rest of the course can be served in this cooldown mode."""
    first_gap: dict | None = None
    tried: set = set()
    tried_kcs: set = set(half)
    sub = subtopic
    while True:
        if sub is None:
            sub = select_next_subtopic(user_state, exclude=tried, exclude_kcs=tried_kcs, cooldown=cooldown)
        if sub is None:
            return None, first_gap
        try:
            return pick_for_subtopic(
                user_id, user_state, sub, focus_subtopic, exclude_kcs=tried_kcs,
                record=record, cooldown=cooldown,
            ), first_gap
        except SubtopicDry as dry:
            if first_gap is None and dry.gap:
                first_gap = dry.gap
            if focus_subtopic is not None:
                return None, first_gap
            dry_kc = (dry.gap or {}).get("kc")
            if dry_kc and dry_kc not in tried_kcs:
                tried_kcs.add(dry_kc)
            else:
                tried.add(sub)
            sub = None


def _is_review_repeat(user_state, qid: int) -> bool:
    """Answered before and not owed a retake. The picker recycles such a drill
    only once its concept has nothing unseen or owed left (prioritization.
    narrow_to_next_kc), so this IS "the concept is spent" — serving it again
    is review, not work, and must not pre-empt fresh work on the other half."""
    return qid in answered_question_ids(user_state) and qid not in owed_question_ids(user_state)


def queue_next_kc(user_state) -> str | None:
    """The concept /next-question would serve from RIGHT NOW — the one the
    knowledge graph rings as "next up".

    Not `kc_graph.select_next_kc`: that names the frontier head that still has
    an unserved drill, and the queue does not always serve it. A head whose
    current RUNG is spent is a content gap the router steps past (`tried_kcs`
    above), so practice hands over the next concept along while the graph kept
    ringing the head — Seth, 2026-09-13, on `torch.tensor-model`: "it
    highlights the lower concept that is not currently being tested". Asking
    the real selection, dry, is the only answer that cannot drift from it.
    None when the queue would find nothing (a 409/404 request).
    """
    picked, _first_gap = run_queue("", user_state, None, None, record=False)
    return picked[2] if picked else None
