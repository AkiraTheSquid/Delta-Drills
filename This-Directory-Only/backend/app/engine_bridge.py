"""engine_bridge.py — the wire between the serving path and `logistic_engine`.

The engine was written, tested and left unwired: nothing imported it except its
own test, and `attempt_log` was never written to. This module is the whole of
the connection, and it lives in one file on purpose — the engine must stay a
pure model (no state loading, no question bank, no graph), and the serving path
must not learn the engine's vocabulary. Everything that has to know both is
here.

WHAT IT DOES

  * `feature_values` materialises one row of the design matrix for a specific
    (learner, concept, question, rung). This is the only place that decides how
    Delta Drills' data maps onto the engine's features, so a feature added to
    the engine shows up as one line here and nowhere else.
  * `record` folds a graded outcome into the learner's posterior for that
    concept AND appends the attempt log row — the prediction made *before* the
    outcome, beside the outcome. That pairing is the point: it is what makes
    the model falsifiable rather than merely opinionated, and it is why the
    engine returns its prediction from `update` instead of letting the caller
    recompute one afterwards from state that has already moved.
  * `mastery` answers the one question the serving path actually asks: how
    strong is this learner on this concept, in [0,1].

ONE POSTERIOR PER (LEARNER, CONCEPT)

`ability` is declared per-KC by the engine's own docstring, and that is how it
is stored: `user_state.kc_posteriors[kc][feature] = {...}`. Not per-atom, and
not global. Per-atom would need a design matrix over atoms and is what BKT is
already doing; global would make the model unable to say that someone is strong
at indexing and weak at broadcasting, which is the entire job.

WHAT IT DELIBERATELY DOES NOT DO

It does not replace BKT. Both run: BKT keeps feeding the atom posteriors that
`kc_mastery`, the unlock lattice and the Statistics panel read, and the engine
supplies the ability estimate the difficulty aim prefers when it has evidence.
Two reasons to keep both rather than cut over. The engine's weights are v0
defaults — sane, but not fitted and not literature-derived — so until the
attempt log has enough rows to score a Brier/reliability curve against, the
honest position is that the engine is a better *estimator* whose *calibration*
is unproven. And the unlock lattice gates content: a miscalibrated engine
deciding what is servable could lock a learner out of the course, whereas a
miscalibrated aim serves a question that is somewhat too hard.

A lesson view is NOT recorded here. A lesson is read, not answered — see
`logistic_engine.GRADED_STAGES` — and an ungraded rung returns None from
`normalize_stage`, which this module treats as "do not update the model from
this attempt" rather than guessing at a rung.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, Mapping, Optional

from app import attempt_log, bkt_mastery, kc_graph
from app import logistic_engine as E

# Attempts a concept's posterior needs before the difficulty aim will prefer it
# over the ladder's own Wilson bound. One graded answer moves a wide prior a
# long way, and the aim is a visible number the learner watches — letting it
# swing on a single attempt would read as noise even when the estimate is
# sound. Three is the same evidence bar `kc_graph._PROMOTE_STREAK` uses to move
# a rung on a streak alone.
MIN_ATTEMPTS_TO_SERVE = 3


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _store(user_state) -> Dict[str, Dict[str, dict]]:
    """The learner's posterior store, created on demand.

    Mirrors `kc_graph.ladder_row`'s shape deliberately: a plain dict of plain
    dicts, so `_save_user_state` can write it without knowing what a Posterior
    is and an older build can read a newer one's file.
    """
    store = getattr(user_state, "kc_posteriors", None)
    if not isinstance(store, dict):
        store = {}
        try:
            user_state.kc_posteriors = store
        except Exception:
            pass
    return store


def _seeded_ability(user_state, kc: str, *, exclude_latest: bool = False) -> E.Posterior:
    """The `ability` prior for a concept the engine has never scored.

    `attempt_log.backfill_from_state` is explicit that old records cannot
    reconstruct estimator state — there are no stored feature values and no
    stored predictions, so backfilled rows are excluded from replay. That is
    correct and this does not argue with it. But refusing to look at the ladder
    at all means every existing learner meets the engine at the cold prior
    (mean -1.0, P ~= 0.27), and the aim would VISIBLY regress the moment the
    engine took over: someone sitting at 20/20 on a concept, whose ladder bound
    puts the aim near 87, would be dropped to about 60 for the crime of the
    system having learned a new way to think about them.

    So the ladder's own Wilson lower bound locates the MEAN, and the prior's
    full width is kept for the VARIANCE. That is what a prior is: a belief and
    an admission of how loosely it is held. The bound is measured across mixed
    rungs and mixed difficulties, which is exactly the imprecision the engine
    exists to remove — so it is worth using as a starting point and not worth
    being confident about. The wide variance means the engine's own evidence
    dominates within a handful of attempts either way.

    A concept with no ladder record gets the untouched prior, which is right:
    nothing has been observed, and the engine is built to run on zero data.

    🔴 `exclude_latest` is not a tuning knob. `record_ladder_outcome` runs
    BEFORE the scoring tail, so by the time an attempt reaches this module the
    ladder already contains it — and seeding from a bound that includes the
    answer being scored would both leak the outcome into the `predicted_p`
    logged as preceding it, and fold the same answer in twice (once through the
    seed, once through `E.step`). The scoring path therefore excludes the last
    row; every read-only caller uses the whole record, which is correct for
    them because they are not about to add it again.
    """
    prior = E.initial_posterior(E.ABILITY)
    attempts = kc_graph.ladder_view(user_state, kc).get("attempts") or []
    if exclude_latest:
        attempts = attempts[:-1]
    recent = attempts[-kc_graph._LADDER_WINDOW:]
    if not recent:
        return prior
    lo, _hi = kc_graph._wilson(sum(1 for a in recent if a.get("correct")), len(recent))
    # Clamped off the ends: logit(0) is -inf, and a learner who has missed
    # everything so far is not infinitely unable.
    p = min(max(float(lo), 0.02), 0.98)
    return E.Posterior(mean=math.log(p / (1.0 - p)), var=prior.var, n=0, last_seen=None)


def posteriors_for(
    user_state, kc: str, *, exclude_latest_attempt: bool = False
) -> Dict[str, E.Posterior]:
    """This learner's LEARNED-feature posteriors for one concept.

    Missing features fall back to their priors rather than being absent, so a
    config that grows a learned feature starts every existing learner at that
    feature's prior instead of at zero — which would be a confident claim, and
    a wrong one. `ability` gets the seeded prior (see `_seeded_ability`); any
    other learned feature a future config adds gets the plain one, because
    there is no existing record that speaks to it.

    `exclude_latest_attempt` is passed by the scoring path only — see
    `_seeded_ability` for why the answer being scored must not seed the prior
    it is about to be folded into.
    """
    raw = _store(user_state).get(kc) or {}
    out: Dict[str, E.Posterior] = {}
    for feature in E.DEFAULT_CONFIG.learned_features:
        post = E.Posterior.from_dict(raw.get(feature.name)) if isinstance(raw, dict) else None
        if post is None:
            post = (
                _seeded_ability(user_state, kc, exclude_latest=exclude_latest_attempt)
                if feature.name == E.ABILITY.name
                else E.initial_posterior(feature)
            )
        out[feature.name] = post
    return out


def _save_posteriors(user_state, kc: str, posteriors: Mapping[str, E.Posterior]) -> None:
    _store(user_state)[kc] = {
        name: post.to_dict() for name, post in posteriors.items()
    }


def _days_since(last_seen: Optional[str]) -> float:
    """Days between `last_seen` and now, or 0.0 when that cannot be answered.

    0.0 rather than a guess: elapsed time only ever widens a posterior and adds
    a forgetting penalty, so inventing one would quietly make a learner look
    worse on the strength of an unparseable timestamp.
    """
    when = attempt_log.parse_ts(last_seen)
    if when is None:
        return 0.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - when).total_seconds() / 86400.0)


def _prereq_mastery(user_state, kc: str) -> float:
    node = kc_graph.registry_node(kc) or {}
    prereqs = node.get("prereqs") or ()
    return E.centred_mastery(kc_graph.kc_mastery(user_state, p)[0] for p in prereqs)


def _encompassing_mastery(user_state, kc: str) -> float:
    """Centred mean BKT posterior over the atoms this concept exercises.

    Borrowed strength, and the engine discounts it accordingly (the
    `encompassing` weight is deliberately below `prereq`). It is read from BKT
    rather than from the engine because atoms are exactly what BKT tracks well
    — this is the one place the two models are genuinely complementary rather
    than redundant.
    """
    row = kc_graph.crosswalk_row(kc) or {}
    atoms = [a.get("a") for a in (row.get("atoms") or []) if a.get("a")]
    if not atoms:
        return 0.0
    params = bkt_mastery.params_for_level(getattr(user_state, "self_reported_level", None))
    return E.centred_mastery(
        bkt_mastery.current_mastery(
            user_state.atom_mastery, user_state.atom_last_ts, atom, params=params
        )
        for atom in atoms
    )


def feature_values(
    user_state,
    kc: str,
    *,
    difficulty_score: Optional[float],
    stage: Optional[str],
    example: bool = False,
    posteriors: Optional[Mapping[str, E.Posterior]] = None,
) -> Dict[str, float]:
    """One row of the design matrix, for this learner on this item.

    `ability` is 1.0 — the design-matrix entry saying "this learner's ability
    applies to this item"; the posterior supplies the coefficient. Every other
    feature is FIXED, so its value here is the quantity and the engine's config
    supplies the weight.

    `posteriors` may be passed by a caller that already holds them — which the
    scoring path must do, because it holds the ones seeded WITHOUT the attempt
    being scored and re-deriving here would quietly reintroduce it.
    """
    if posteriors is None:
        posteriors = posteriors_for(user_state, kc)
    ability = posteriors.get(E.ABILITY.name)
    return {
        E.ABILITY.name: 1.0,
        E.DIFFICULTY.name: E.difficulty_to_logits(difficulty_score),
        E.STAGE.name: E.stage_offset(stage),
        E.EXAMPLE.name: E.example_offset(example),
        E.PREREQ.name: _prereq_mastery(user_state, kc),
        E.ENCOMPASSING.name: _encompassing_mastery(user_state, kc),
        E.RECENCY.name: E.recency_value(_days_since(ability.last_seen if ability else None)),
    }


def predict(
    user_state,
    kc: str,
    *,
    difficulty_score: Optional[float],
    stage: Optional[str],
    example: bool = False,
) -> E.Prediction:
    """P(correct) for an item this learner has not answered yet."""
    return E.predict(
        feature_values(
            user_state, kc, difficulty_score=difficulty_score, stage=stage, example=example
        ),
        posteriors_for(user_state, kc),
    )


def mastery(user_state, kc: str) -> Optional[float]:
    """How strong this learner is on one concept, in [0,1], or None.

    Defined as the engine's P(correct) on a MEDIAN-difficulty item at the SOLO
    rung — the two reference points the engine's own scale is built on
    (`difficulty_to_logits` centres at 50, `solo` is the 0.0 stage offset). So
    the number means "how likely are they to get an average problem right,
    unaided" — literally so since 2026-08-31, when `example` became a feature:
    this asks for the no-example case, which is the question the difficulty aim
    is asking, and it is on
    the same 0–1 scale the aim already consumes.

    None until the concept has `MIN_ATTEMPTS_TO_SERVE` graded attempts behind
    it. Before that the posterior is the prior with a nudge, and reporting it
    would dress a default up as a measurement.
    """
    ability = posteriors_for(user_state, kc).get(E.ABILITY.name)
    if ability is None or ability.n < MIN_ATTEMPTS_TO_SERVE:
        return None
    return predict(user_state, kc, difficulty_score=50.0, stage=E.STAGE_SOLO).p


def record(
    user_state,
    user_id: str,
    *,
    kc: str,
    question_id: Optional[int],
    subtopic: Optional[str],
    difficulty_score: Optional[float],
    stage: Optional[str],
    correct: bool,
    example: bool = False,
    grade: Optional[float] = None,
    atoms: Optional[list] = None,
) -> Optional[E.Prediction]:
    """Fold one graded attempt into the concept's posterior, and log it.

    Returns the prediction the model made BEFORE seeing the outcome, or None
    when the attempt was not scoreable. Two things make it unscoreable, and
    both are deliberate refusals rather than fallbacks: a concept the q-matrix
    does not name (there is no posterior to move, and inventing a key would
    file evidence under a concept nobody teaches), and a rung
    `normalize_stage` does not recognise (mis-attributing an attempt to the
    wrong rung corrupts the stage offsets for every learner, which is worse
    than dropping one row).

    The log write is best-effort. A full disk must not cost the learner their
    answer — the posterior has already moved in memory and will be persisted
    with the rest of their state, and a missing log row costs a future refit
    one observation.
    """
    normalized = E.normalize_stage(stage)
    if not kc or kc_graph.registry_node(kc) is None or normalized not in E.GRADED_STAGES:
        return None

    posteriors = posteriors_for(user_state, kc, exclude_latest_attempt=True)
    values = feature_values(
        user_state, kc, difficulty_score=difficulty_score, stage=normalized,
        example=example, posteriors=posteriors,
    )
    ability = posteriors.get(E.ABILITY.name)
    now = _now_iso()
    updated, prediction = E.step(
        values,
        posteriors,
        correct,
        days_elapsed=_days_since(ability.last_seen if ability else None),
        timestamp=now,
    )
    _save_posteriors(user_state, kc, updated)

    try:
        attempt_log.record_attempt(
            user_id,
            kc,
            question_id,
            normalized,
            values,
            prediction,
            correct,
            subtopic=subtopic,
            atoms=atoms or [],
            difficulty_score=int(difficulty_score) if difficulty_score is not None else None,
            grade=grade,
            ts=now,
        )
    except Exception:  # pragma: no cover — logging must never break scoring
        pass
    return prediction


def served_stage(user_state, kc: str, question_id: int) -> Optional[str]:
    """The rung this answer was SERVED at, for one concept.

    Read from the ladder row rather than recomputed, and that distinction is
    the whole function. `record_ladder_outcome` runs before the scoring tail
    and appends `{"correct", "stage", "ts"}` using the rung the learner was
    sitting on — after which `kc_stage` may well have moved, because answering
    correctly is exactly what promotes it. Asking `kc_stage` here would
    therefore charge a scaffolded answer at the unscaffolded rung it just
    earned, which is the one error the engine's stage offsets cannot survive:
    the model would learn that assistance does not help.

    Matched on `question_id`, not merely taken as "the newest row". A placement
    probe deliberately never reaches the ladder — it measures prior knowledge on
    material nobody taught — so on that route the newest row belongs to some
    earlier question, and reading it would score today's answer at a rung it was
    never served at. None when the newest row is not this question's, and
    `record` then declines rather than guessing.
    """
    attempts = kc_graph.ladder_view(user_state, kc).get("attempts") or []
    if not attempts:
        return None
    latest = attempts[-1]
    if latest.get("question_id") != question_id:
        return None
    return latest.get("stage")


def served_example(user_state, kc: str, question_id: int) -> bool:
    """Was this answer given behind a worked example, for one concept?

    Read from the SAME ladder row as `served_stage`, under the same
    question-id match and for the same reason: `record_ladder_outcome` has
    already written what the learner actually saw (the client's report wins
    over the schedule's plan, because a popup the client could not draw is not
    assistance), and recomputing the schedule here would ask what the NEXT
    drill gets rather than what this one had.

    False when the newest row is not this question's — `served_stage` returns
    None there and `record` declines the attempt entirely, so the value is
    never used, but defaulting to "unaided" keeps this readable on its own: it
    is the value that adds nothing to the prediction.
    """
    attempts = kc_graph.ladder_view(user_state, kc).get("attempts") or []
    if not attempts:
        return False
    latest = attempts[-1]
    if latest.get("question_id") != question_id:
        return False
    return bool(latest.get("example"))


def answer_was_aided(user_state, question_id: int) -> bool:
    """Did ANY concept's ladder row record this answer as given behind an example?

    One answer, one screen: the popup is drawn once in front of the drill, not
    once per concept the drill targets. The per-KC rows are all written from
    that single fact, so any of them can report it — but a multi-KC question
    can reach a concept whose row was written by an earlier question, so this
    asks all of them rather than trusting the first.

    For the atom-BKT path, which is keyed by atom rather than by concept and so
    has no row of its own to read.
    """
    return any(
        served_example(user_state, kc, question_id)
        for kc in kc_graph.question_kcs(question_id)
    )


def record_attempt_across_kcs(
    user_state,
    user_id: str,
    *,
    question_id: int,
    subtopic: Optional[str],
    difficulty_score: Optional[float],
    correct: bool,
    grade: Optional[float] = None,
    atoms: Optional[list] = None,
) -> Dict[str, E.Prediction]:
    """Score one answer against EVERY concept the question targets.

    A problem is not evidence about one concept. `question_kcs` returns all of
    them, `record_kc_outcome` has always logged the ladder attempt against all
    of them, and the engine is built for exactly this — an attempt is one row
    per concept, each with its own posterior, its own design-matrix entry and
    its own rung. Attributing only to the primary KC would let a question that
    exercises three ideas teach the model about one.

    The naive objection is double-counting, and the engine's own arithmetic
    answers it: each concept's posterior moves by its own residual against its
    own prediction, so a concept the learner is already strong at barely moves
    while a weak one moves a lot. That is attribution, not duplication.
    """
    out: Dict[str, E.Prediction] = {}
    for kc in kc_graph.question_kcs(question_id):
        prediction = record(
            user_state,
            user_id,
            kc=kc,
            question_id=question_id,
            subtopic=subtopic,
            difficulty_score=difficulty_score,
            stage=served_stage(user_state, kc, question_id),
            example=served_example(user_state, kc, question_id),
            correct=correct,
            grade=grade,
            atoms=atoms,
        )
        if prediction is not None:
            out[kc] = prediction
    return out
