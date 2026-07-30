"""logistic_engine.py — the additive-logistic mastery engine (Layers 1 and 2).

WHAT THIS REPLACES AND WHY
--------------------------
`bkt_mastery.py` gives a per-atom P(skill known). That is a *latent state*
probability, not a predicted response probability, and it carries no
uncertainty at all. Three things we want are therefore unavailable from it:

  * P(correct) for the item about to be served — needed to target difficulty
    honestly instead of by a hand-tuned ladder;
  * an interval on that estimate — needed so a 2-for-2 streak stops reading as
    100%, and needed by the explore-mode prerequisite prober, which has to know
    *what it does not know* in order to probe it;
  * a place to put prerequisite and encompassing evidence as first-class model
    inputs rather than as a bespoke propagation rule.

The model here is the additive logistic (Logistic Knowledge Tracing, Pavlik et
al.), which generalises AFM, PFA, R-PFA, IRT **and Elo/Glicko**:

    logit P(correct) = SUM_f  w_f * x_f

Elo is exactly this with the feature set {learner ability, -item difficulty} and
online SGD as the fitter. Glicko is that plus a Gaussian posterior on ability.
So nothing about the Elo/Glicko family is being rejected — it is the two-feature
configuration of this engine, and everything else is a feature we are free to
add later. That is the whole point: **a new edge type in the concept graph
becomes a new feature, not a new derivation.** Bolting prerequisite terms onto
Glicko's closed-form update would mean re-deriving its Laplace approximation
every time the graph grows a relation; here the graph grows and the config file
grows with it.

LAYER 1 — the link function (`predict`)
---------------------------------------
Features are declared in an `EngineConfig`, not hard-coded. Each is either

  * FIXED    — a known quantity with a configured weight (item difficulty, the
               scaffold-stage offset, prerequisite mastery, recency), or
  * LEARNED  — a quantity we are estimating and therefore carry a posterior for
               (the learner's ability on the concept).

LAYER 2 — the estimator (`update`)
----------------------------------
Each LEARNED feature carries a Gaussian posterior N(mu, var). Observations are
folded in by assumed-density filtering (a one-step Laplace/Kalman update for
logistic regression). Two consequences worth stating because they are the
reason to prefer this over plain Elo:

  * **Uncertainty enters the prediction.** We do not report sigma(mean); we
    integrate the logistic over the posterior, using MacKay's approximation

        P = sigma( m / sqrt(1 + pi*s^2/8) )

    so a wide posterior pulls the prediction toward 0.5. This is the same
    behaviour as Glicko's g(RD) factor — a different approximation to the same
    integral, generalised to any number of features.

  * **Uncertainty grows when unobserved.** `inflate` widens the posterior with
    elapsed time, which is Glicko's inactivity behaviour. Note this is decay of
    *confidence*, not decay of *mastery* — they are different claims and the
    engine deliberately only makes the first one. Mastery decay, if wanted,
    belongs in a recency FEATURE (see `RECENCY`), where it is visible and
    fittable rather than buried in the state update.

The diagonal posterior (no cross-feature covariance) is a deliberate v0
simplification: it keeps the update O(features), and in the single-feature case
it reduces exactly to Glicko's shape. Correlated features would need a full
covariance matrix; nothing in the current feature set is correlated enough to
pay for that.

CALIBRATION HONESTY
-------------------
Every weight and variance below is a v0 default chosen for sane behaviour at
N=16 attempts, **not** fitted and **not** literature-derived. The engine is
prior-dominated at present and that is expected — see
`ITS-procedural-AI-SYNC/glicko-vs-lkt-mastery-engine.md` section 8. The point of
shipping it now is that it runs correctly on zero data, and that
`attempt_log.py` records `predicted_p` beside every outcome so these numbers
become *checkable* (Brier score, reliability curve) rather than asserted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, Mapping, Optional, Tuple

# ---------------------------------------------------------------------------
# Model version. Stamped onto every attempt-log row. Bump whenever the FEATURES
# tuple or the default weights change, so a later refit can tell which rows were
# produced by which model instead of silently mixing them.
# ---------------------------------------------------------------------------
MODEL_VERSION = "logistic-v0.1"


# ---------------------------------------------------------------------------
# Stage vocabulary
#
# The learner-facing ladder is: lesson -> worked -> faded -> solo.
#
# `kc_graph.LADDER_STAGES` is the LEGACY vocabulary (worked, faded, partial,
# solo) and differs in two ways: it has no `lesson` rung, and it splits the
# scaffolded middle into `faded`/`partial`. Both are mapped here rather than in
# the caller, because the attempt log must carry exactly ONE stage vocabulary
# forever — a log holding two is a log that cannot be replayed.
# ---------------------------------------------------------------------------
STAGE_LESSON = "lesson"
STAGE_WORKED = "worked"
STAGE_FADED = "faded"
STAGE_SOLO = "solo"

STAGES: Tuple[str, ...] = (STAGE_LESSON, STAGE_WORKED, STAGE_FADED, STAGE_SOLO)

# Rungs that produce a GRADED attempt. A lesson is read, not answered; it
# generates an exposure event, never evidence of ability. Feeding lesson views
# to the estimator would let a learner "prove" mastery by clicking Next.
GRADED_STAGES: Tuple[str, ...] = (STAGE_WORKED, STAGE_FADED, STAGE_SOLO)

# Legacy rung -> current rung. `partial` and `faded` collapse: both are
# completion problems with part of the solution visible, and splitting them
# doubled the data needed to move through the middle of the ladder for a
# distinction the learner could not reliably feel.
LEGACY_STAGE_MAP: Dict[str, str] = {
    "worked": STAGE_WORKED,
    "faded": STAGE_FADED,
    "partial": STAGE_FADED,
    "solo": STAGE_SOLO,
    "independent": STAGE_SOLO,  # `record_kc_outcome`'s default argument
}


def normalize_stage(stage: Optional[str]) -> Optional[str]:
    """Map any stage name — current or legacy — onto the current vocabulary.

    Returns None for an unrecognised value rather than guessing. Callers should
    treat None as "do not update the model from this attempt": mis-attributing
    an attempt to the wrong rung corrupts the stage offsets for every learner,
    which is worse than dropping one row.
    """
    if not stage:
        return None
    key = str(stage).strip().lower()
    if key in STAGES:
        return key
    return LEGACY_STAGE_MAP.get(key)


# ---------------------------------------------------------------------------
# Feature declaration
# ---------------------------------------------------------------------------

FIXED = "fixed"
LEARNED = "learned"


@dataclass(frozen=True)
class Feature:
    """One additive term in the logit.

    `weight` multiplies the feature value for FIXED features. For LEARNED
    features the weight is the thing being estimated, so `weight` is ignored and
    `prior_mean`/`prior_var` seed the posterior instead.

    `drift_per_day` is how fast the posterior widens while unobserved, in
    variance units per day. Only meaningful for LEARNED features. Zero means the
    engine never becomes less sure on its own.
    """

    name: str
    kind: str = FIXED
    weight: float = 1.0
    prior_mean: float = 0.0
    prior_var: float = 0.0
    drift_per_day: float = 0.0
    min_var: float = 1e-4
    max_var: Optional[float] = None
    description: str = ""

    @property
    def is_learned(self) -> bool:
        return self.kind == LEARNED


# --- the v0 feature set ----------------------------------------------------
#
# ORDER IS NOT SIGNIFICANT — this is a sum. Features may be added, removed, or
# reweighted without touching any other module; that is the property the whole
# design exists to protect. What DOES matter is that `name` is stable, because
# it is the key under which the posterior is persisted and the key the attempt
# log records.

ABILITY = Feature(
    name="ability",
    kind=LEARNED,
    prior_mean=-1.0,          # sigma(-1) ~= 0.27 before any evidence: a fresh
                              # learner is assumed to fail a mid-difficulty item
                              # more often than not. Deliberately pessimistic —
                              # the cost of under-estimating is a slightly easy
                              # first item; the cost of over-estimating is
                              # serving someone a solo problem on day one.
    prior_var=1.2,            # wide. sqrt(1.2) ~= 1.1 logits ~ a genuinely
                              # uninformative start, and the attenuation term
                              # will pull early predictions hard toward 0.5.
    drift_per_day=0.010,      # ~0.3 variance regained per month idle. Glicko's
                              # inactivity inflation, in logit units.
    max_var=1.2,              # never widen past the prior: absence of practice
                              # is not evidence of anything WORSE than "unknown".
    description="Learner ability on this knowledge component, in logits.",
)

DIFFICULTY = Feature(
    name="difficulty",
    kind=FIXED,
    weight=-1.0,              # harder item -> lower P(correct). The sign is the
                              # whole content of this feature; the magnitude is
                              # carried by the caller's scaling (see
                              # `difficulty_to_logits`).
    description="Item difficulty in logits, centred on the mid-difficulty item.",
)

STAGE = Feature(
    name="stage",
    kind=FIXED,
    weight=1.0,
    description="Scaffold assistance for the rung served, in logits.",
)

PREREQ = Feature(
    name="prereq",
    kind=FIXED,
    weight=0.9,               # a fully-mastered prereq set is worth ~+0.45
                              # logits over a neutral one; a fully-unknown set
                              # about -0.45. Enough to matter for attribution,
                              # not enough to let graph structure overrule the
                              # learner's own record.
    description="Centred mean mastery of the KC's prerequisites (-0.5..+0.5).",
)

ENCOMPASSING = Feature(
    name="encompassing",
    kind=FIXED,
    weight=0.6,               # weaker than `prereq` on purpose. Encompassing
                              # credit is an inference about a relation, not an
                              # observation of the learner; the architecture
                              # report's variance-tax argument says borrowed
                              # strength should always be discounted relative to
                              # own evidence.
    description="Centred mean mastery of atoms this KC encompasses (-0.5..+0.5).",
)

RECENCY = Feature(
    name="recency",
    kind=FIXED,
    weight=-0.5,              # forgetting, as an EXPLICIT predictor rather than
                              # a hidden decay applied to stored mastery. Made
                              # visible here so it can be fitted (Settles &
                              # Meeder's half-life regression is the model to
                              # fit it against) instead of asserted.
    description="Elapsed-time forgetting term in [0,1]; 0 = just practised.",
)

DEFAULT_FEATURES: Tuple[Feature, ...] = (
    ABILITY,
    DIFFICULTY,
    STAGE,
    PREREQ,
    ENCOMPASSING,
    RECENCY,
)


# --- stage offsets ---------------------------------------------------------
#
# `solo` is the reference level at 0.0 by construction: the solo rung is the one
# whose difficulty the item's own difficulty score already describes, so its
# offset must be zero or the difficulty scale means two different things.
#
# The ordering constraint worked > faded > solo == 0 is asserted by
# `EngineConfig.validate()`, because a violation is not a bad calibration, it is
# a sign error — it would mean the engine believes a scaffold makes a problem
# HARDER, and the ladder would then promote learners for failing.
#
# v0 magnitudes: a worked-stage item is ~+1.4 logits easier than the same item
# solo (sigma(0) 0.50 -> sigma(1.4) 0.80). Not fitted. The number to watch once
# data exists is whether this shrinks with ability — see the expertise-reversal
# note in `set_stage_offsets`.
DEFAULT_STAGE_OFFSETS: Dict[str, float] = {
    STAGE_WORKED: 1.4,
    STAGE_FADED: 0.7,
    STAGE_SOLO: 0.0,
}


@dataclass(frozen=True)
class EngineConfig:
    """A complete, versioned specification of the model.

    Everything the engine does is determined by this object, which is why it is
    frozen and version-stamped: two attempts scored under the same
    `model_version` are comparable, and two scored under different versions are
    not. `attempt_log` records the version per row so a refit can tell them
    apart.
    """

    version: str = MODEL_VERSION
    features: Tuple[Feature, ...] = DEFAULT_FEATURES
    stage_offsets: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_STAGE_OFFSETS)
    )
    # Item difficulty arrives on the question bank's 1..100 scale. This divisor
    # converts it to logits: 25 puts a difficulty-100 item ~2 logits above a
    # difficulty-50 one, i.e. roughly 0.5 -> 0.12 P(correct) for a median
    # learner. Sets the *slope* of the whole difficulty scale.
    difficulty_scale: float = 25.0
    # Half-life for the recency feature. Matches bkt_mastery.HALF_LIFE_DAYS so
    # the two modules do not disagree about forgetting while both are live.
    recency_half_life_days: float = 14.0

    def feature(self, name: str) -> Optional[Feature]:
        for f in self.features:
            if f.name == name:
                return f
        return None

    @property
    def learned_features(self) -> Tuple[Feature, ...]:
        return tuple(f for f in self.features if f.is_learned)

    def validate(self) -> None:
        """Raise on a config that is internally inconsistent.

        Called by the tests and cheap enough to call at import in a caller. The
        checks here are for SIGN and ORDER errors, not for calibration: a badly
        calibrated weight makes bad predictions, but a sign error makes the
        ladder actively punish success, which is unrecoverable from the
        learner's point of view.
        """
        names = [f.name for f in self.features]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate feature names in config: {names}")
        if not self.learned_features:
            raise ValueError("config has no LEARNED feature; nothing to estimate")
        for f in self.learned_features:
            if f.prior_var <= 0:
                raise ValueError(f"learned feature {f.name!r} needs prior_var > 0")
            if f.max_var is not None and f.max_var < f.min_var:
                raise ValueError(f"feature {f.name!r} has max_var below min_var")
        if self.difficulty_scale <= 0:
            raise ValueError("difficulty_scale must be positive")
        if self.recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be positive")

        offs = self.stage_offsets
        missing = [s for s in GRADED_STAGES if s not in offs]
        if missing:
            raise ValueError(f"stage_offsets missing graded stages: {missing}")
        if abs(offs[STAGE_SOLO]) > 1e-9:
            raise ValueError(
                "solo must be the reference stage at offset 0.0 — otherwise the "
                "item difficulty scale is ambiguous"
            )
        if not offs[STAGE_WORKED] > offs[STAGE_FADED] > offs[STAGE_SOLO]:
            raise ValueError(
                "stage offsets must satisfy worked > faded > solo == 0; got "
                f"{offs}. A violation means the model believes a scaffold makes "
                "a problem harder, which would promote learners for failing."
            )


DEFAULT_CONFIG = EngineConfig()


def set_stage_offsets(config: EngineConfig, **offsets: float) -> EngineConfig:
    """Return a copy of `config` with stage offsets overridden.

    The intended use is calibration once data exists. The first question to ask
    of real data is NOT "are these numbers right" but "is a single number right"
    — expertise reversal predicts the scaffold's benefit shrinks as ability
    grows, i.e. the offset is really an interaction term (stage x ability). That
    is deliberately not modelled at v0; this function is where the answer would
    land if a held-out log-loss comparison says it should be.
    """
    merged = dict(config.stage_offsets)
    merged.update(offsets)
    out = replace(config, stage_offsets=merged)
    out.validate()
    return out


# ---------------------------------------------------------------------------
# Feature-value helpers
#
# These convert app-native quantities to the engine's logit-ish scale. They live
# here (not in the caller) so the scale is defined in exactly one place, but they
# take plain numbers so the engine stays free of app imports and stays testable
# without the concept graph on disk.
# ---------------------------------------------------------------------------


def difficulty_to_logits(difficulty_score: Optional[float], config: EngineConfig = DEFAULT_CONFIG) -> float:
    """Map a 1..100 question-bank difficulty onto centred logits.

    Difficulty 50 is the origin, so the `ability` posterior is interpretable as
    "logit P(correct) on a median-difficulty solo item" rather than as an
    arbitrary rating. A missing score is treated as median: an unscored question
    should not be silently assumed easy, and it should not be assumed brutal
    either.
    """
    if difficulty_score is None:
        return 0.0
    return (float(difficulty_score) - 50.0) / config.difficulty_scale


def centred_mastery(values: Iterable[float]) -> float:
    """Mean of mastery values in [0,1], recentred to [-0.5, +0.5].

    Centring matters: an UNCENTRED mean would make an all-prereqs-unknown state
    contribute 0 and an all-known state contribute +w, so the feature could only
    ever help. Prerequisite weakness has to be able to *lower* the prediction,
    or the model can never attribute a failure upward to a missing parent —
    which is the entire reason the feature exists.

    An empty set returns 0.0: a KC with no prerequisites is neutral, not
    penalised.
    """
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return sum(vals) / len(vals) - 0.5


def recency_value(days_since_last: Optional[float], config: EngineConfig = DEFAULT_CONFIG) -> float:
    """Forgetting term in [0,1]: 0 just after practice, -> 1 as time passes.

    `None` (never practised) returns 0.0 rather than 1.0. Never-practised is
    already expressed by the wide `ability` posterior; charging it a forgetting
    penalty as well would double-count the same ignorance and make cold-start
    predictions implausibly bleak.
    """
    if days_since_last is None or days_since_last <= 0:
        return 0.0
    return 1.0 - 0.5 ** (float(days_since_last) / config.recency_half_life_days)


def stage_offset(stage: Optional[str], config: EngineConfig = DEFAULT_CONFIG) -> float:
    """Assistance value of a rung, in logits. Unknown/ungraded rungs give 0.0."""
    key = normalize_stage(stage)
    if key is None:
        return 0.0
    return float(config.stage_offsets.get(key, 0.0))


# ---------------------------------------------------------------------------
# Posterior state (Layer 2)
# ---------------------------------------------------------------------------


@dataclass
class Posterior:
    """Gaussian belief about one LEARNED feature, for one (learner, KC).

    `mean` is in logits. `var` is the squared uncertainty — Glicko's RD is
    sqrt(var), in the same units as `mean`.

    `n` counts graded attempts folded in. It is NOT used by the update (the
    variance already encodes evidence strength); it is carried for display and
    for sanity-checking a replay against the log.
    """

    mean: float
    var: float
    n: int = 0
    last_seen: Optional[str] = None  # ISO-8601 UTC of the most recent update

    @property
    def sd(self) -> float:
        return math.sqrt(max(self.var, 0.0))

    def to_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "n": self.n, "last_seen": self.last_seen}

    @classmethod
    def from_dict(cls, raw: Optional[Mapping]) -> Optional["Posterior"]:
        if not isinstance(raw, Mapping):
            return None
        try:
            return cls(
                mean=float(raw["mean"]),
                var=float(raw["var"]),
                n=int(raw.get("n") or 0),
                last_seen=raw.get("last_seen"),
            )
        except (KeyError, TypeError, ValueError):
            return None


def initial_posterior(feature: Feature) -> Posterior:
    """A never-updated belief, straight from the feature's prior."""
    return Posterior(mean=feature.prior_mean, var=feature.prior_var, n=0, last_seen=None)


def inflate(post: Posterior, feature: Feature, days_elapsed: float) -> Posterior:
    """Widen a posterior for time passed without observation.

    This is Glicko's inactivity behaviour and it is the ONLY thing the engine
    does to state between attempts. Note carefully what it does not do: the mean
    is untouched. Not practising something is evidence that we have become less
    sure, not evidence that the learner has become worse. Believing otherwise
    lets the system silently demote a learner who simply went on holiday.

    Actual forgetting is modelled separately and visibly by the RECENCY feature.
    """
    if days_elapsed <= 0 or feature.drift_per_day <= 0:
        return post
    ceiling = feature.max_var if feature.max_var is not None else float("inf")
    widened = min(post.var + feature.drift_per_day * float(days_elapsed), ceiling)
    return Posterior(mean=post.mean, var=max(widened, feature.min_var), n=post.n, last_seen=post.last_seen)


# ---------------------------------------------------------------------------
# Prediction (Layer 1 + the Layer 2 attenuation)
# ---------------------------------------------------------------------------


def sigmoid(x: float) -> float:
    """Numerically stable logistic."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def attenuation(var: float) -> float:
    """MacKay's factor for integrating a logistic over a Gaussian.

        P(correct) ~= sigma( mean / sqrt(1 + pi*var/8) )

    This is what makes the prediction honest at low data: as `var` grows the
    factor -> 0 and the prediction -> 0.5. Glicko's g(RD) does the same job with
    a slightly different constant (3q^2/pi^2 rather than pi/8) — both are
    approximations to the same intractable integral, and the choice between them
    is numerically immaterial at our scale.
    """
    return 1.0 / math.sqrt(1.0 + math.pi * max(var, 0.0) / 8.0)


@dataclass(frozen=True)
class Prediction:
    """What the engine believed, before seeing the outcome.

    Every field here is written to the attempt log. `p` without `logit_mean` and
    `logit_var` would be unreplayable, and `contributions` is what makes a
    prediction explainable to the learner ("this looks hard because the
    prerequisite is weak") rather than an oracle.
    """

    p: float                      # marginal P(correct), uncertainty folded in
    p_mean: float                 # sigma(logit_mean) — the point estimate
    logit_mean: float
    logit_var: float
    contributions: Dict[str, float]

    @property
    def sd(self) -> float:
        return math.sqrt(max(self.logit_var, 0.0))

    def interval(self, z: float = 1.96) -> Tuple[float, float]:
        """Credible interval on P(correct), by pushing the logit interval
        through the link. Asymmetric in probability space, which is correct —
        a symmetric probability interval near 0 or 1 would leave the unit
        interval."""
        sd = self.sd
        return sigmoid(self.logit_mean - z * sd), sigmoid(self.logit_mean + z * sd)


def predict(
    values: Mapping[str, float],
    posteriors: Mapping[str, Posterior],
    config: EngineConfig = DEFAULT_CONFIG,
) -> Prediction:
    """P(correct) for one item, given materialised feature values.

    `values` maps feature name -> feature value. For LEARNED features the value
    is the design-matrix entry (normally 1.0 — "this learner's ability applies
    to this item"); the posterior supplies the coefficient. A feature absent
    from `values` contributes nothing, which is how a config can carry features
    the caller has not wired up yet without breaking.
    """
    total = 0.0
    var = 0.0
    contributions: Dict[str, float] = {}

    for f in config.features:
        x = float(values.get(f.name, 0.0))
        if x == 0.0 and not f.is_learned:
            continue
        if f.is_learned:
            post = posteriors.get(f.name) or initial_posterior(f)
            term = post.mean * x
            var += (x ** 2) * max(post.var, 0.0)
        else:
            term = f.weight * x
        total += term
        contributions[f.name] = term

    return Prediction(
        p=sigmoid(total * attenuation(var)),
        p_mean=sigmoid(total),
        logit_mean=total,
        logit_var=var,
        contributions=contributions,
    )


# ---------------------------------------------------------------------------
# Update (Layer 2)
# ---------------------------------------------------------------------------


def update(
    values: Mapping[str, float],
    posteriors: Mapping[str, Posterior],
    correct: bool,
    config: EngineConfig = DEFAULT_CONFIG,
    timestamp: Optional[str] = None,
) -> Tuple[Dict[str, Posterior], Prediction]:
    """Fold one graded outcome into the LEARNED posteriors.

    Returns `(new_posteriors, prediction_made_before_seeing_the_outcome)`. The
    prediction comes back so the caller can log it: storing what the model said
    BEFORE the outcome is the only thing that makes the model falsifiable, and
    recomputing it afterwards from updated state would quietly cheat.

    The update is assumed-density filtering for Bayesian logistic regression —
    match the posterior mean and variance after one observation, keeping it
    Gaussian. With a diagonal covariance and feature value x_g:

        v      = p(1-p)
        denom  = 1 + v * s^2
        mu_g  += var_g * x_g * (y - p) / denom
        var_g -= (var_g * x_g)^2 * v / denom

    Sanity check on the shape: with a single feature at x=1 this is
    `mu += var*(y-p)/(1+var*p*(1-p))`, which is Glicko's rating update with the
    same self-damping behaviour — a confident posterior (small var) moves little,
    an uncertain one moves a lot. The engine did not have to be told that; it
    falls out of the arithmetic.

    Only LEARNED features move. FIXED weights are model parameters, not learner
    state: fitting them is a batch job over the attempt log, not something to do
    online from one learner's evidence.
    """
    pred = predict(values, posteriors, config)

    p = pred.p
    v = p * (1.0 - p)
    denom = 1.0 + v * pred.logit_var
    residual = (1.0 if correct else 0.0) - p

    out: Dict[str, Posterior] = {name: post for name, post in posteriors.items()}

    for f in config.learned_features:
        x = float(values.get(f.name, 0.0))
        post = out.get(f.name) or initial_posterior(f)
        if x == 0.0:
            # Feature did not apply to this item — no information about it.
            out[f.name] = post
            continue

        new_mean = post.mean + (post.var * x * residual) / denom
        new_var = post.var - ((post.var * x) ** 2 * v) / denom

        # A variance floor keeps the engine able to learn again after a long
        # correct streak. Without it the posterior collapses, `attenuation`
        # approaches 1, and a genuine regression (a learner who has forgotten,
        # or a mis-tagged item) can no longer move the estimate.
        new_var = max(new_var, f.min_var)
        if f.max_var is not None:
            new_var = min(new_var, f.max_var)

        out[f.name] = Posterior(
            mean=new_mean,
            var=new_var,
            n=post.n + 1,
            last_seen=timestamp or post.last_seen,
        )

    return out, pred


def step(
    values: Mapping[str, float],
    posteriors: Mapping[str, Posterior],
    correct: bool,
    config: EngineConfig = DEFAULT_CONFIG,
    *,
    days_elapsed: float = 0.0,
    timestamp: Optional[str] = None,
) -> Tuple[Dict[str, Posterior], Prediction]:
    """Advance state by one graded attempt: inflate for elapsed time, then update.

    **Every caller must use this, live and replay alike.** `update` on its own is
    only half a transition, and a live path that skipped the inflation while
    replay applied it would produce two different posteriors from the same
    history — which would quietly falsify the one property the attempt log
    exists to provide (`replay(log) == live state`). Having one function makes
    that divergence impossible rather than merely discouraged; the equivalence
    test in `scripts/test_logistic_engine.py` section G is what caught it.

    `days_elapsed` is the gap since the previous graded attempt for this KC, not
    since the last time anything happened.
    """
    state: Dict[str, Posterior] = dict(posteriors)
    if days_elapsed > 0:
        for f in config.learned_features:
            post = state.get(f.name) or initial_posterior(f)
            state[f.name] = inflate(post, f, days_elapsed)
    return update(values, state, correct, config, timestamp=timestamp)


# ---------------------------------------------------------------------------
# Ladder decisions
#
# What replaces the Wilson-bound promotion in kc_graph. The old rule looked only
# at the last 20 attempts AT ONE RUNG, which threw away every attempt made at the
# other rungs. Here every graded attempt at every rung has already been folded
# into one shared ability posterior, so the decision uses all of it.
# ---------------------------------------------------------------------------

# Promote when we are CONFIDENT the learner clears the bar at the next rung;
# demote when we are confident they do not clear it at the current one. The
# asymmetry (lower bound to promote, upper bound to demote) is deliberate and
# is inherited from the existing ladder: both directions are conservative, so
# uncertainty alone never moves anyone. It costs some extra practice at a rung
# already mastered, which is the cheaper error.
PROMOTE_P = 0.55   # required lower credible bound at the NEXT rung
DEMOTE_P = 0.40    # required upper credible bound at the CURRENT rung
LADDER_Z = 1.0     # ~68% credible bound

# Calibrated against the real question bank rather than by taste. Sweeping an
# all-correct learner from a cold start (see scripts/test_logistic_engine.py,
# section F) gives attempts-to-promotion:
#
#     PROMOTE_P    worked->faded    faded->solo
#       0.50             5              10
#       0.55             6              12      <- chosen
#       0.70            15              30
#
# and at z=1.96 rather than 1.0 the 0.70 row becomes 29 / 53.
#
# The binding constraint is pool size: KC question pools run 3..11 with a median
# of 7, and 13 of 63 KCs own 4 or fewer questions. A ladder needing 30 attempts
# to reach `solo` would exhaust those pools several times over before promoting,
# so the learner would grind repeats on a concept they had demonstrably cleared.
# 0.55 reaches `solo` in ~12 attempts, which the median pool sustains because
# the SAME question is legitimately re-served at different rungs (that is what
# the rungs are — worked shows the example, faded hides part of it, solo hides
# all of it), so 12 attempts does not mean 12 distinct questions.
#
# The discrimination check that matters: a learner answering at 50% never
# promotes off `worked` at ANY of these thresholds. Loosening the bar buys speed
# for a learner who is actually correct without letting a coin-flip through.


def _p_at_stage(
    values: Mapping[str, float],
    posteriors: Mapping[str, Posterior],
    stage: str,
    config: EngineConfig,
) -> Prediction:
    """Predict at a counterfactual rung, holding everything else fixed."""
    probe = dict(values)
    probe["stage"] = stage_offset(stage, config)
    return predict(probe, posteriors, config)


def next_stage(
    values: Mapping[str, float],
    posteriors: Mapping[str, Posterior],
    current: str,
    config: EngineConfig = DEFAULT_CONFIG,
) -> str:
    """Which rung to serve next for this concept.

    Note this asks a *counterfactual*: "what would P(correct) be if we served
    the next rung?" — which is only answerable because the stage offset is part
    of the model rather than a separate rating per rung. That is the concrete
    payoff of the shared-ability design.
    """
    stage = normalize_stage(current) or STAGE_WORKED
    if stage == STAGE_LESSON:
        # A lesson is never graded, so there is nothing to decide from; the
        # learner leaves the lesson by reading it, which the caller records as
        # an exposure event.
        return STAGE_WORKED

    order = list(GRADED_STAGES)
    idx = order.index(stage)

    if idx + 1 < len(order):
        up = order[idx + 1]
        lo, _ = _p_at_stage(values, posteriors, up, config).interval(LADDER_Z)
        if lo >= PROMOTE_P:
            return up

    if idx > 0:
        _, hi = _p_at_stage(values, posteriors, stage, config).interval(LADDER_Z)
        if hi < DEMOTE_P:
            return order[idx - 1]

    return stage


def mastered(
    values: Mapping[str, float],
    posteriors: Mapping[str, Posterior],
    config: EngineConfig = DEFAULT_CONFIG,
    threshold: float = PROMOTE_P,
) -> bool:
    """True when the learner clears the bar UNAIDED, with confidence.

    Deliberately evaluated at `solo` regardless of the rung currently served: a
    concept is not mastered because someone can finish a faded problem, and the
    counterfactual makes that testable without waiting for them to reach the
    rung.
    """
    lo, _ = _p_at_stage(values, posteriors, STAGE_SOLO, config).interval(LADDER_Z)
    return lo >= threshold
