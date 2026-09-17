#!/usr/bin/env python3
"""Validation suite for logistic_engine.py + attempt_log.py.

Run: .venv/bin/python scripts/test_logistic_engine.py
Exits non-zero on any failed assertion. No pytest dependency.

Structure mirrors scripts/test_bkt_mastery.py: structural checks first (does the
arithmetic hold), then behavioural (does it do the pedagogically right thing),
then the independent-replay check that the whole design rests on — the stored
log must reproduce the live posteriors exactly, or the log is not a substitute
for the state and the "posteriors are disposable" claim is false.
"""
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import attempt_log as L  # noqa: E402
from app import logistic_engine as E  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


CFG = E.DEFAULT_CONFIG


def fresh():
    return {f.name: E.initial_posterior(f) for f in CFG.learned_features}


def row(ability=1.0, difficulty=0.0, stage=0.0, prereq=0.0, encompassing=0.0, recency=0.0):
    return {
        "ability": ability,
        "difficulty": difficulty,
        "stage": stage,
        "prereq": prereq,
        "encompassing": encompassing,
        "recency": recency,
    }


# ---------------------------------------------------------------------------
print("A. CONFIG — sign and order guards")

CFG.validate()
check("default config validates", True)


def raises(fn):
    try:
        fn()
    except ValueError:
        return True
    except Exception:
        return False
    return False


check(
    "rejects solo offset != 0 (difficulty scale would be ambiguous)",
    raises(lambda: E.set_stage_offsets(CFG, solo=0.3)),
)
check(
    "rejects inverted stage order (scaffold making problems harder)",
    raises(lambda: E.set_stage_offsets(CFG, worked=0.1, faded=0.9)),
)
check(
    "accepts a valid recalibration",
    E.set_stage_offsets(CFG, worked=2.0, faded=1.0).stage_offsets["worked"] == 2.0,
)
check(
    "config has exactly one learned feature at v0",
    [f.name for f in CFG.learned_features] == ["ability"],
    str([f.name for f in CFG.learned_features]),
)

# ---------------------------------------------------------------------------
print("\nB. STAGE VOCABULARY — legacy mapping")

check("legacy 'partial' collapses into 'faded'", E.normalize_stage("partial") == "faded")
check("legacy 'independent' maps to 'solo'", E.normalize_stage("independent") == "solo")
check("current names pass through", E.normalize_stage("worked") == "worked")
check("unknown stage returns None, not a guess", E.normalize_stage("bogus") is None)
check("lesson is not a graded stage", "lesson" not in E.GRADED_STAGES)
check(
    "every graded stage has an offset",
    all(s in CFG.stage_offsets for s in E.GRADED_STAGES),
)

# ---------------------------------------------------------------------------
print("\nC. PREDICTION — monotonicity and attenuation")

p = fresh()
easy = E.predict(row(difficulty=E.difficulty_to_logits(10)), p, CFG).p
mid = E.predict(row(difficulty=E.difficulty_to_logits(50)), p, CFG).p
hard = E.predict(row(difficulty=E.difficulty_to_logits(95)), p, CFG).p
check("P(correct) falls as difficulty rises", easy > mid > hard, f"{easy:.3f} > {mid:.3f} > {hard:.3f}")

worked = E.predict(row(stage=E.stage_offset("worked")), p, CFG).p
faded = E.predict(row(stage=E.stage_offset("faded")), p, CFG).p
solo = E.predict(row(stage=E.stage_offset("solo")), p, CFG).p
check("scaffolded rungs predict higher than solo", worked > faded > solo, f"{worked:.3f} > {faded:.3f} > {solo:.3f}")

strong = E.predict(row(prereq=0.5), p, CFG).p
weak = E.predict(row(prereq=-0.5), p, CFG).p
check("weak prerequisites lower the prediction", strong > weak, f"{strong:.3f} vs {weak:.3f}")
check(
    "prereq feature is centred — it can hurt, not only help",
    weak < E.predict(row(prereq=0.0), p, CFG).p < strong,
)

enc_hi = E.predict(row(encompassing=0.5), p, CFG).p
enc_lo = E.predict(row(encompassing=-0.5), p, CFG).p
check("encompassed sub-skills raise the prediction", enc_hi > enc_lo)
check(
    "encompassing is discounted relative to prereq (borrowed strength)",
    abs(CFG.feature("encompassing").weight) < abs(CFG.feature("prereq").weight),
)

fresh_p = E.predict(row(recency=0.0), p, CFG).p
stale_p = E.predict(row(recency=1.0), p, CFG).p
check("elapsed time lowers the prediction", fresh_p > stale_p, f"{fresh_p:.3f} vs {stale_p:.3f}")
check("never-practised gets no forgetting penalty", E.recency_value(None) == 0.0)

# attenuation
wide = {"ability": E.Posterior(mean=2.0, var=4.0)}
narrow = {"ability": E.Posterior(mean=2.0, var=0.01)}
pw = E.predict(row(), wide, CFG)
pn = E.predict(row(), narrow, CFG)
check(
    "uncertainty pulls the prediction toward 0.5 (Glicko's g(RD))",
    abs(pw.p - 0.5) < abs(pn.p - 0.5),
    f"wide={pw.p:.3f} narrow={pn.p:.3f}",
)
check("point estimate is unaffected by variance", abs(pw.p_mean - pn.p_mean) < 1e-9)
check("attenuation is monotone decreasing in variance", E.attenuation(0.0) > E.attenuation(1.0) > E.attenuation(9.0))
check("attenuation(0) == 1 (no uncertainty, no shrinkage)", abs(E.attenuation(0.0) - 1.0) < 1e-12)

lo, hi = pw.interval()
check("credible interval brackets the point estimate", lo < pw.p_mean < hi, f"[{lo:.3f}, {hi:.3f}]")
check("interval stays inside the unit interval", 0.0 < lo and hi < 1.0)
check(
    "wider posterior gives a wider interval",
    (pw.interval()[1] - pw.interval()[0]) > (pn.interval()[1] - pn.interval()[0]),
)

# ---------------------------------------------------------------------------
print("\nD. UPDATE — direction, damping, floors")

post, pred = E.update(row(), fresh(), True, CFG)
check("a correct answer raises the ability mean", post["ability"].mean > fresh()["ability"].mean)
post_w, _ = E.update(row(), fresh(), False, CFG)
check("an incorrect answer lowers the ability mean", post_w["ability"].mean < fresh()["ability"].mean)
check("evidence shrinks the variance", post["ability"].var < fresh()["ability"].var)
check("attempt counter advances", post["ability"].n == 1)
check(
    "prediction returned is the PRE-outcome one",
    abs(pred.p - E.predict(row(), fresh(), CFG).p) < 1e-12,
)

# Glicko's self-damping: a confident posterior moves less than an uncertain one.
conf = {"ability": E.Posterior(mean=0.0, var=0.02)}
unconf = {"ability": E.Posterior(mean=0.0, var=2.0)}
d_conf = abs(E.update(row(), conf, True, CFG)[0]["ability"].mean - 0.0)
d_unconf = abs(E.update(row(), unconf, True, CFG)[0]["ability"].mean - 0.0)
check(
    "confident posteriors move less than uncertain ones (Glicko damping)",
    d_conf < d_unconf,
    f"{d_conf:.4f} < {d_unconf:.4f}",
)

# Variance floor keeps the model able to learn again.
long_streak = fresh()
for _ in range(60):
    long_streak, _ = E.update(row(), long_streak, True, CFG)
floor = CFG.feature("ability").min_var
check("variance never collapses below its floor", long_streak["ability"].var >= floor, f"var={long_streak['ability'].var:.6f}")
before = long_streak["ability"].mean
after, _ = E.update(row(), long_streak, False, CFG)
check(
    "a miss after a long streak still moves the estimate",
    after["ability"].mean < before,
    f"{before:.4f} -> {after['ability'].mean:.4f}",
)

# A feature absent from the design row must not move.
untouched, _ = E.update({"ability": 0.0}, fresh(), True, CFG)
check("a non-applying feature is left alone", untouched["ability"].mean == fresh()["ability"].mean)

# Fixed weights are model parameters, not learner state.
check(
    "update touches only learned features",
    set(post.keys()) == {"ability"},
    str(sorted(post.keys())),
)

# ---------------------------------------------------------------------------
print("\nE. INFLATE — uncertainty decay, not mastery decay")

start = E.Posterior(mean=0.9, var=0.2)
grown = E.inflate(start, E.ABILITY, days_elapsed=30.0)
check("idle time widens the variance", grown.var > start.var, f"{start.var:.3f} -> {grown.var:.3f}")
check("idle time leaves the mean untouched", grown.mean == start.mean)
check(
    "variance never exceeds the prior (idleness is not evidence of being worse)",
    E.inflate(start, E.ABILITY, days_elapsed=100000.0).var <= E.ABILITY.max_var,
)
check("zero elapsed time is a no-op", E.inflate(start, E.ABILITY, 0.0) is start)

# ---------------------------------------------------------------------------
print("\nF. LADDER — counterfactual promotion on a shared ability")

beginner = fresh()
check("a fresh learner starts at the worked rung", E.next_stage(row(), beginner, "worked", CFG) == "worked")

skilled = {"ability": E.Posterior(mean=3.0, var=0.05, n=30)}
check("a strong learner is promoted off worked", E.next_stage(row(), skilled, "worked", CFG) == "faded")
check("and off faded", E.next_stage(row(), skilled, "faded", CFG) == "solo")
check("solo is terminal", E.next_stage(row(), skilled, "solo", CFG) == "solo")

struggling = {"ability": E.Posterior(mean=-3.0, var=0.05, n=30)}
check("a struggling learner is demoted from solo", E.next_stage(row(), struggling, "solo", CFG) == "faded")
check("and from faded", E.next_stage(row(), struggling, "faded", CFG) == "worked")
check("worked is the floor", E.next_stage(row(), struggling, "worked", CFG) == "worked")

check("lesson always hands off to worked", E.next_stage(row(), beginner, "lesson", CFG) == "worked")
check("legacy 'partial' is accepted as a current rung", E.next_stage(row(), skilled, "partial", CFG) in E.GRADED_STAGES)

# Thin evidence must not promote — the conservatism that stops a 2-for-2 streak
# reading as mastery. Driven through the real update path rather than by hand:
# a hand-written posterior can express states the estimator cannot reach (the
# variance only ever shrinks with evidence), and asserting against those tests
# nothing about the shipped behaviour.
two_for_two = fresh()
for _ in range(2):
    two_for_two, _ = E.update(row(stage=E.stage_offset("worked")), two_for_two, True, CFG)
check(
    "two correct answers do not promote off worked",
    E.next_stage(row(), two_for_two, "worked", CFG) == "worked",
    f"P(faded)={E.predict(row(stage=E.stage_offset('faded')), two_for_two, CFG).p:.3f}",
)

# Promotion must arrive within the smallest real question pool's reach. Pools
# run 3..11 with a median of 7; a ladder needing 30 attempts would exhaust the
# small ones several times before moving anyone.
climb, stage_now, promoted_at = fresh(), "worked", None
for i in range(1, 41):
    climb, _ = E.update(row(stage=E.stage_offset(stage_now)), climb, True, CFG)
    nxt = E.next_stage(row(), climb, stage_now, CFG)
    if nxt != stage_now:
        promoted_at = promoted_at or i
        stage_now = nxt
    if stage_now == "solo":
        break
check(
    "an all-correct learner leaves worked within ~8 attempts",
    promoted_at is not None and promoted_at <= 8,
    f"promoted at attempt {promoted_at}",
)
check(
    "and reaches solo within ~16",
    stage_now == "solo" and i <= 16,
    f"reached {stage_now} at attempt {i}",
)

# Discrimination: a coin-flip learner must never climb.
coin, stage_c = fresh(), "worked"
for i in range(60):
    coin, _ = E.update(row(stage=E.stage_offset(stage_c)), coin, i % 2 == 0, CFG)
    stage_c = E.next_stage(row(), coin, stage_c, CFG)
check("a 50%-correct learner never promotes off worked", stage_c == "worked", f"ended at {stage_c}")

# The whole point of the shared-ability design: evidence at one rung moves the
# estimate at another.
scaffolded = fresh()
for _ in range(12):
    scaffolded, _ = E.update(row(stage=E.stage_offset("worked")), scaffolded, True, CFG)
solo_pred_after = E.predict(row(stage=E.stage_offset("solo")), scaffolded, CFG).p
solo_pred_before = E.predict(row(stage=E.stage_offset("solo")), fresh(), CFG).p
check(
    "worked-rung successes raise the SOLO prediction (cross-stage transfer)",
    solo_pred_after > solo_pred_before,
    f"{solo_pred_before:.3f} -> {solo_pred_after:.3f}",
)
check(
    "mastery is judged unaided regardless of the rung served",
    E.mastered(row(), skilled, CFG) and not E.mastered(row(), beginner, CFG),
)

# ---------------------------------------------------------------------------
print("\nG. ATTEMPT LOG — round-trip, replay equivalence, calibration")

tmp = Path(tempfile.mkdtemp(prefix="dd-attempt-log-"))
try:
    USER, KC = "test-user", "numpy.ndarray-model"

    live = fresh()
    outcomes = [True, False, True, True, True, False, True, True]
    stages = ["worked"] * 3 + ["faded"] * 3 + ["solo"] * 2

    for i, (ok, st) in enumerate(zip(outcomes, stages)):
        vals = row(difficulty=E.difficulty_to_logits(40 + i), stage=E.stage_offset(st))
        stamp = f"2026-07-{10+i:02d}T12:00:00Z"
        # Driven through `step`, exactly as the live serving path must be —
        # attempts are one day apart, so the elapsed-time inflation is live here
        # too, and the SAME timestamp goes to the estimator and to the log.
        # Using `update` directly, or letting the two take different clock
        # readings, would make this test pass while the equivalence it asserts
        # was false.
        new_live, prediction = E.step(
            vals, live, ok, CFG, days_elapsed=1.0 if i else 0.0, timestamp=stamp
        )
        L.record_attempt(
            USER, KC, 400 + i, st, vals, prediction, ok,
            difficulty_score=40 + i, base_dir=tmp, ts=stamp,
        )
        live = new_live

    rows = list(L.iter_rows(USER, base_dir=tmp))
    check("every attempt round-trips through JSONL", len(rows) == len(outcomes), f"{len(rows)} rows")
    check("all rows count as graded evidence", all(r.is_graded for r in rows))
    check("stage is stored in the current vocabulary", {r.stage for r in rows} <= set(E.GRADED_STAGES))
    check("model_version is stamped on every row", all(r.model_version == CFG.version for r in rows))
    check("features are stored, not just the outcome", all(r.features.get("ability") == 1.0 for r in rows))
    check("predicted_p is stored beside the outcome", all(r.predicted_p is not None for r in rows))

    # THE property. If this fails, the posteriors are not disposable and the
    # log is not the source of truth.
    replayed, consumed = L.replay(USER, KC, CFG, base_dir=tmp)
    check("replay consumes every graded row", consumed == len(outcomes))
    check(
        "replay(log) reproduces the live posterior mean",
        abs(replayed["ability"].mean - live["ability"].mean) < 1e-9,
        f"replay={replayed['ability'].mean:.9f} live={live['ability'].mean:.9f}",
    )
    check(
        "replay(log) reproduces the live posterior variance",
        abs(replayed["ability"].var - live["ability"].var) < 1e-9,
    )
    # Compare the SERIALIZED state, not just mean and variance. `last_seen`
    # diverged for a while behind a mean/variance-only assertion: the live path
    # was not passing its timestamp to the estimator while replay took one from
    # each logged row, so the two states were unequal in a field the narrower
    # check could not see.
    check(
        "replay(log) reproduces the full serialized posterior",
        replayed["ability"].to_dict() == live["ability"].to_dict(),
        f"replay={replayed['ability'].to_dict()} live={live['ability'].to_dict()}",
    )

    # Lesson views: recorded, never scored.
    L.record_lesson_view(USER, KC, base_dir=tmp, ts="2026-07-09T09:00:00Z")
    check("lesson view is recorded", L.has_seen_lesson(USER, KC, base_dir=tmp))
    check("a lesson view is not graded evidence", L.replay(USER, KC, CFG, base_dir=tmp)[1] == len(outcomes))
    check("unseen concepts report false", not L.has_seen_lesson(USER, "numpy.einsum", base_dir=tmp))

    cal = L.calibration(USER, base_dir=tmp)
    check("calibration reports over stored predictions", cal["n"] == len(outcomes))
    check("brier score is computed", cal["brier"] is not None and 0.0 <= cal["brier"] <= 1.0, f"brier={cal['brier']:.4f}")
    check("baseline brier is reported for comparison", cal["brier_baseline"] is not None)
    check("reliability bins are populated", len(cal["bins"]) > 0)

    # Corruption tolerance: a torn final line must not destroy the history.
    with L.log_path(USER, tmp).open("a", encoding="utf-8") as fh:
        fh.write('{"ts":"2026-07-30T00:00:00Z","kind":"att')
    check(
        "a truncated tail line is skipped, not fatal",
        len(list(L.iter_rows(USER, base_dir=tmp))) == len(outcomes) + 1,
    )

    # Unknown fields from a newer build must not break an older reader.
    forward = L.AttemptRow.from_dict(
        {"ts": "2026-07-30T00:00:00Z", "kind": "attempt", "user_id": "u", "future_field": 1}
    )
    check("unknown fields are dropped, not fatal", forward is not None and forward.kind == "attempt")
    check("a row with no kind is rejected", L.AttemptRow.from_dict({"ts": "x"}) is None)

    # An attempt at an unrecognised rung carries a wrong offset — refuse it.
    bad = L.AttemptRow(ts="2026-07-30T00:00:00Z", kind="attempt", user_id="u", stage="bogus", correct=True)
    check("an attempt at an unknown rung is not graded evidence", not bad.is_graded)
    pending = L.AttemptRow(ts="2026-07-30T00:00:00Z", kind="attempt", user_id="u", stage="solo")
    check("an ungraded (in-flight) attempt is not evidence", not pending.is_graded)

    # --- regressions for defects found by cross-model review (codex) --------

    # A torn tail must not swallow the NEXT good row. The original code appended
    # straight onto an unterminated fragment, fusing them into one bad line and
    # losing both.
    _, pr = E.step(row(), fresh(), True, CFG)
    L.record_attempt("torn", "kc", 1, "solo", row(), pr, True, base_dir=tmp, ts="2026-07-01T00:00:00Z")
    with L.log_path("torn", tmp).open("a", encoding="utf-8") as fh:
        fh.write('{"ts":"2026-07-01T12:00:00Z","kind":"att')  # no newline
    L.record_attempt("torn", "kc", 2, "solo", row(), pr, True, base_dir=tmp, ts="2026-07-02T00:00:00Z")
    torn_rows = list(L.iter_rows("torn", base_dir=tmp))
    check(
        "a good row appended after a torn tail survives",
        [r.question_id for r in torn_rows] == [1, 2],
        f"got {[r.question_id for r in torn_rows]}",
    )

    class _State:
        kc_ladder = {
            "kc": {"attempts": [
                {"ts": "2026-06-01T00:00:00Z", "stage": "faded", "correct": True, "question_id": 9},
                {"ts": "2026-06-02T00:00:00", "stage": "partial", "correct": False, "question_id": 10},
            ]}
        }
        kc_exposure = {"kc": "2026-05-30T00:00:00Z"}

    res = L.backfill_from_state("bf", _State(), base_dir=tmp, dry_run=False)
    check("backfill writes attempts and lesson views", res["attempts"] == 2 and res["lesson_views"] == 1)
    bf_attempts = [r for r in L.iter_rows("bf", base_dir=tmp) if r.kind == L.KIND_ATTEMPT]
    check(
        "backfilled rows are NOT graded evidence (they carry no features)",
        not any(r.is_graded for r in bf_attempts),
    )
    check(
        "so replay consumes none of them",
        L.replay("bf", "kc", CFG, base_dir=tmp)[1] == 0,
    )
    check(
        "backfill still records the lesson view",
        L.has_seen_lesson("bf", "kc", base_dir=tmp),
    )
    # Mixed naive/aware timestamps (legacy state writes naive) must not abort a
    # learner's whole reconstruction with a TypeError.
    L.record_attempt("bf", "kc", 11, "solo", row(), pr, True, base_dir=tmp, ts="2026-06-03T00:00:00")
    L.record_attempt("bf", "kc", 12, "solo", row(), pr, True, base_dir=tmp, ts="2026-06-04T00:00:00Z")
    try:
        _, n_mixed = L.replay("bf", "kc", CFG, base_dir=tmp)
        check("replay survives mixed naive/aware timestamps", n_mixed == 2, f"consumed {n_mixed}")
    except TypeError as exc:
        check("replay survives mixed naive/aware timestamps", False, str(exc))
    check(
        "naive timestamps are normalised to aware UTC",
        L.parse_ts("2026-06-03T00:00:00").tzinfo is not None,
    )

    # Re-running a migration must not double a learner's history.
    again = L.backfill_from_state("bf", _State(), base_dir=tmp, dry_run=False)
    check("a second backfill is refused, not duplicated", again.get("skipped") is True, str(again))

    # Injected rows must respect the requested KC.
    foreign = [L.AttemptRow(
        ts="2026-07-01T00:00:00Z", kind="attempt", user_id="u", kc="OTHER",
        stage="solo", correct=True, features=row(),
    )]
    check(
        "injected rows for another KC are filtered out",
        L.replay(USER, KC, CFG, base_dir=tmp, rows=foreign)[1] == 0,
    )

finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------------------
print("\nI. GRAPH BRIDGE — features off the real concept graph")

from app import adaptive, engine_features as F  # noqa: E402

state = adaptive.UserPracticeState(user_id="engine-test")
# 🔴 The root is DERIVED from kc_registry.json, not named. It used to be
# hard-coded to "numpy.ndarray-model", which stopped being a root the moment
# the python course went in front of it (it now declares python.indexing,
# python.defining-functions and python.dots-and-imports), so both root checks
# had been failing while testing nothing: a KC with parents obviously has a
# non-zero prereq term and something to probe. Ask the graph which KC has no
# parents, and the assertions follow the curriculum instead of a memory of it.
_reg = json.loads(
    (Path(F.__file__).resolve().parents[3]
     / "Local_Deployed_Shared" / "lessons" / "kc_registry.json").read_text()
)
_roots = [k["id"] for k in _reg.get("kcs", []) if not (k.get("prereqs") or [])]
check("the concept graph has a root to start a learner on", bool(_roots), str(_roots))
KC_ROOT = _roots[0]
KC_CHILD = "numpy.broadcasting-rules"

vec = F.build(state, KC_CHILD, "worked", difficulty_score=55)
check("build returns every configured feature", set(vec) == {f.name for f in CFG.features}, str(sorted(vec)))
check("ability is the unit design entry", vec["ability"] == 1.0)
check("provenance names the prerequisites used", isinstance(vec.sources.get("prereqs"), dict))
check(
    "a real KC with parents produces a non-neutral prereq term",
    vec["prereq"] != 0.0 and len(vec.sources["prereqs"]) > 0,
    f"prereq={vec['prereq']:+.3f} from {list(vec.sources['prereqs'])}",
)
root_vec = F.build(state, KC_ROOT, "worked", difficulty_score=55)
check("a lattice root has no prerequisites and scores neutral", root_vec["prereq"] == 0.0)

# The duplicate-weighting defect: an item tag naming a concept the graph already
# lists as a parent must not count that concept twice.
parents = list(vec.sources["prereqs"])
dup_vec = F.build(state, KC_CHILD, "worked", difficulty_score=55,
                  question={"prereq_tags": [parents[0]]})
check(
    "an item tag duplicating a graph prereq does not double its weight",
    abs(dup_vec["prereq"] - vec["prereq"]) < 1e-12,
    f"{vec['prereq']:+.6f} -> {dup_vec['prereq']:+.6f}",
)
check(
    "prereq provenance is deduplicated too",
    len(dup_vec.sources["prereqs"]) == len(vec.sources["prereqs"]),
)

off_vec = F.build(state, KC_CHILD, "worked", difficulty_score=55,
                  question={"prereq_tags": ["for-loops", "tuple-unpacking"]})
check(
    "off-graph prereq tags are recorded, not scored",
    off_vec.sources.get("off_graph_prereqs") == ["for-loops", "tuple-unpacking"]
    and abs(off_vec["prereq"] - vec["prereq"]) < 1e-12,
)
check("build tolerates a null KC", F.build(state, None, "solo")["prereq"] == 0.0)

enc = F.encompassed_mastery(state, "numpy.reshape-flatten")
check("encompassing edges resolve to atoms", isinstance(enc, dict) and len(enc) > 0, f"{len(enc)} atoms")

check("pooling weight falls as own evidence accumulates",
      F.pooling_weight(0) > F.pooling_weight(5) > F.pooling_weight(50))
check("probe priority falls with own evidence",
      F.probe_priority(0) > F.probe_priority(5) > F.probe_priority(50))
check("a perfectly trusted edge is never worth probing", F.probe_priority(0, kappa=1.0) == 0.0)

probe = F.select_probe_kc(state, KC_CHILD)
check("a struggling concept yields a prerequisite probe", probe is not None and probe["kc"] in parents, str(probe and probe["kc"]))
check("a lattice root has nothing to probe", F.select_probe_kc(state, KC_ROOT) is None)
check(
    "exclusion is honoured",
    (lambda p: p is None or p["kc"] != parents[0])(
        F.select_probe_kc(state, KC_CHILD, exclude=[parents[0]])
    ),
)
pred_f, vals_f = F.predict_for(state, KC_CHILD, "worked", fresh(), difficulty_score=55)
check("predict_for returns a usable probability and its inputs",
      0.0 < pred_f.p < 1.0 and vals_f["ability"] == 1.0, f"P={pred_f.p:.3f}")

# ---------------------------------------------------------------------------
print("\nH. HELPERS — scale definitions")

check("difficulty 50 is the origin", E.difficulty_to_logits(50) == 0.0)
check("missing difficulty is treated as median", E.difficulty_to_logits(None) == 0.0)
check("harder scores map above the origin", E.difficulty_to_logits(100) > 0 > E.difficulty_to_logits(1))
check("empty prerequisite set is neutral, not penalised", E.centred_mastery([]) == 0.0)
check("all-known prereqs give +0.5", abs(E.centred_mastery([1.0, 1.0]) - 0.5) < 1e-12)
check("all-unknown prereqs give -0.5", abs(E.centred_mastery([0.0, 0.0]) + 0.5) < 1e-12)
check(
    "recency rises with elapsed time and saturates below 1",
    0 < E.recency_value(7) < E.recency_value(28) < 1.0,
)
check(
    "recency at one half-life is 0.5",
    abs(E.recency_value(CFG.recency_half_life_days) - 0.5) < 1e-12,
)
check("sigmoid is stable at extremes", E.sigmoid(-800) == 0.0 and E.sigmoid(800) == 1.0)
check("posterior sd is sqrt(var)", abs(E.Posterior(0.0, 0.25).sd - 0.5) < 1e-12)
check(
    "posterior survives a dict round-trip",
    E.Posterior.from_dict(E.Posterior(0.3, 0.4, 5, "t").to_dict()).mean == 0.3,
)
check("malformed posterior dict returns None", E.Posterior.from_dict({"mean": "x"}) is None)

print("\n" + ("ALL PASS" if not fails else f"FAILURES ({len(fails)}): {fails}"))
sys.exit(1 if fails else 0)
