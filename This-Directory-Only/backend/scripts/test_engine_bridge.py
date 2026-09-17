#!/usr/bin/env python
"""test_engine_bridge.py — the logistic engine is actually wired in.

`scripts/test_logistic_engine.py` covers the model: given feature values and a
posterior, does the arithmetic do what the paper says. This file covers the
WIRE, which is the part that was missing for months — the engine was written,
tested and imported by nothing, so every one of its own tests passed while it
scored no learner and `attempt_log` recorded nothing.

Everything here therefore drives the REAL scoring tail (`finalize_attempt`)
rather than calling the bridge directly, because "the model updates when you
call the updater" is not the property in doubt.

Run: .venv/bin/python scripts/test_engine_bridge.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

_TMP = tempfile.mkdtemp(prefix="engine-bridge-test-")
os.environ["USER_DATA_DIR"] = _TMP

from app import attempt_log, engine_bridge, kc_graph  # noqa: E402
from app import logistic_engine as E  # noqa: E402
from app.adaptive import UNRATED, UserPracticeState, record_attempt  # noqa: E402
from app.practice.attempt_scoring import finalize_attempt  # noqa: E402
from app.prioritization import record_ladder_outcome, target_difficulty  # noqa: E402
from app.questions import get_question_by_id  # noqa: E402

PASS, FAIL = [], []


def check(label, condition, detail=""):
    (PASS if condition else FAIL).append(label)
    print(f"  {'PASS' if condition else 'FAIL'}  {label}" + (f"  — {detail}" if detail else ""))


def answer(state, qid, correct):
    """One graded attempt, through the same calls in the same order as the
    router. `record_ladder_outcome` BEFORE `finalize_attempt` is not incidental:
    the ladder stores the rung the learner was sitting on, and the bridge reads
    the rung from there precisely because `kc_stage` has moved by now."""
    q = get_question_by_id(qid)
    record_attempt(
        user_state=state, question_id=qid, subtopic=q.subtopic,
        difficulty_score=q.difficulty_score, correct=correct,
    )
    record_ladder_outcome(state, qid, correct)
    return finalize_attempt(state, UNRATED)


def _kc_with_questions():
    """A KC the q-matrix actually places questions on, so the test does not
    silently pass by exercising nothing."""
    for kc in kc_graph._registry():
        qids = kc_graph.questions_for_kc(kc)
        if len(qids) >= 4:
            return kc, list(qids)
    raise SystemExit("no KC carries four questions — the fixture is wrong, not the code")


print("A. THE ENGINE SEES GRADED ANSWERS")
KC, QIDS = _kc_with_questions()
state = UserPracticeState(user_id="engine-bridge-test")
before = engine_bridge.posteriors_for(state, KC)["ability"]
check("a fresh concept starts at the feature prior", before.n == 0 and before.var == E.ABILITY.prior_var,
      f"mean={before.mean:.3f} var={before.var:.3f}")
check("mastery is withheld before there is evidence", engine_bridge.mastery(state, KC) is None)

for qid in QIDS[:3]:
    answer(state, qid, True)
after = engine_bridge.posteriors_for(state, KC)["ability"]
check("three answers moved the posterior", after.n == 3, f"n={after.n}")
check("correct answers raised ability", after.mean > before.mean,
      f"{before.mean:.3f} -> {after.mean:.3f}")
check("evidence narrowed the posterior", after.var < before.var,
      f"{before.var:.3f} -> {after.var:.3f}")
check("mastery is reported once the bar is met", engine_bridge.mastery(state, KC) is not None,
      f"{engine_bridge.mastery(state, KC):.3f}")

print("\nB. A MISS MOVES IT BACK")
peak = engine_bridge.mastery(state, KC)
answer(state, QIDS[3], False)
check("a miss lowered mastery", engine_bridge.mastery(state, KC) < peak,
      f"{peak:.3f} -> {engine_bridge.mastery(state, KC):.3f}")

print("\nC. THE RUNG IS THE ONE IT WAS SERVED AT")
# `kc_stage` is read AFTER the ladder has recorded this attempt, so it can have
# promoted. The rung the engine scores against must be the one on the ladder row.
recorded = (kc_graph.ladder_view(state, KC)["attempts"] or [{}])[-1].get("stage")
check("served_stage reads the ladder, not today's stage",
      engine_bridge.served_stage(state, KC, QIDS[3]) == recorded,
      f"served={recorded!r} vs kc_stage={kc_graph.kc_stage(state, KC)!r}")
check("an unrecognised rung is refused rather than guessed",
      engine_bridge.record(state, "engine-bridge-test", kc=KC, question_id=1,
                           subtopic="x", difficulty_score=50, stage="not-a-rung",
                           correct=True) is None)
check("an empty concept is refused",
      engine_bridge.record(state, "engine-bridge-test", kc="", question_id=1,
                           subtopic="x", difficulty_score=50, stage="solo",
                           correct=True) is None)
check("an unknown but non-empty concept is refused too",
      engine_bridge.record(state, "engine-bridge-test", kc="numpy.not-a-real-kc",
                           question_id=1, subtopic="x", difficulty_score=50,
                           stage="solo", correct=True) is None)
check("a refused concept left no posterior behind",
      "numpy.not-a-real-kc" not in (state.kc_posteriors or {}))
# A placement probe never appends to the ladder, so the newest row belongs to an
# earlier question. Reading it would score today's answer at yesterday's rung.
check("a rung from a DIFFERENT question is not borrowed",
      engine_bridge.served_stage(state, KC, question_id=-999) is None)

print("\nC2. THE PREDICTION DOES NOT KNOW THE ANSWER")
# The central temporal invariant, and the one the first cut of this wiring got
# wrong in two places at once. `predicted_p` is logged as the belief that
# PRECEDED the outcome, so two learners in identical states must be predicted
# identically regardless of how they then answer. It failed before because the
# engine scored after the BKT block (whose atom updates feed `encompassing`)
# and because the ladder seed read a bound that already contained the answer.
# Nothing surfaced it: a leaked prediction is not an error, it is a flattering
# Brier score.
def _twin(uid, correct):
    st = UserPracticeState(user_id=uid)
    for qid in QIDS[:2]:
        answer(st, qid, True)
    answer(st, QIDS[2], correct)
    row = [r for r in attempt_log.iter_rows(uid) if r.is_graded][-1]
    return row.predicted_p, row.features


p_right, f_right = _twin("leak-probe-correct", True)
p_wrong, f_wrong = _twin("leak-probe-wrong", False)
# Not exact equality. The twins are built microseconds apart in wall-clock, and
# two features are time-dependent by design — `recency` reads elapsed days, and
# `encompassing` reads BKT posteriors that decay continuously — so they differ
# in the twelfth decimal for reasons that have nothing to do with the label.
# LEAKAGE is not a rounding-scale effect: when this was broken, one atom update
# moved `encompassing` by ~0.02 and `predicted_p` by ~0.005, thousands of times
# the tolerance below. The gap between the two scales is what makes this a test
# rather than a coin flip.
TOL = 1e-6
check("the same state predicts the same p whichever way it is answered",
      abs(p_right - p_wrong) < TOL, f"{p_right:.9f} vs {p_wrong:.9f}")
drifted = [k for k in f_right if abs(f_right[k] - f_wrong.get(k, 0.0)) > TOL]
check("no feature carries the outcome into the prediction",
      not drifted,
      "; ".join(f"{k}: {f_right[k]:.6f} vs {f_wrong[k]:.6f}" for k in drifted) or "identical")

print("\nD. THE ATTEMPT LOG IS WRITTEN")
rows = [r for r in attempt_log.iter_rows("engine-bridge-test") if r.is_graded]
check("one row per graded attempt per concept", len(rows) >= 4, f"{len(rows)} rows")
check("every row carries the prediction made BEFORE the outcome",
      all(r.predicted_p is not None for r in rows))
check("every row carries the design matrix it was scored on",
      all(r.features for r in rows))
check("calibration is computable from the log",
      attempt_log.calibration("engine-bridge-test").get("n") == len(rows))

print("\nE. THE AIM PREFERS THE ENGINE ONCE IT HAS EVIDENCE")
sub = get_question_by_id(QIDS[0]).subtopic
aim = target_difficulty(state, sub, kc=KC)
engine_m = engine_bridge.mastery(state, KC)
check("the aim is the engine's mastery on the 20-100 scale",
      abs(aim - max(10.0, min(100.0, 20.0 + 80.0 * engine_m))) < 1e-6,
      f"aim={aim:.2f} from mastery={engine_m:.3f}")
fresh = UserPracticeState(user_id="engine-bridge-test-2")
check("a concept with no engine evidence still falls back",
      target_difficulty(fresh, sub, kc=KC) > 0)

print("\nF. AN EXISTING LEARNER IS NOT DEMOTED BY TURNING THE ENGINE ON")
# The engine's cold prior is deliberately pessimistic (mean -1.0). A learner
# with a strong ladder record must not be dropped to it the moment the engine
# starts being consulted — the seed carries their record across.
strong = UserPracticeState(user_id="engine-bridge-test-3")
row = kc_graph.ladder_row(strong, KC)
row["worked_seen"] = 1
row["attempts"] = [{"correct": True, "stage": "solo", "ts": "2026-08-01T00:00:00+00:00"}] * 20
seeded = engine_bridge.posteriors_for(strong, KC)["ability"]
check("a strong record seeds a higher prior than the cold one",
      seeded.mean > E.ABILITY.prior_mean, f"{E.ABILITY.prior_mean} -> {seeded.mean:.3f}")
check("the seed keeps the prior's width — it locates, it does not confirm",
      abs(seeded.var - E.ABILITY.prior_var) < 1e-9)
weak = UserPracticeState(user_id="engine-bridge-test-4")
wrow = kc_graph.ladder_row(weak, KC)
wrow["worked_seen"] = 1
wrow["attempts"] = [{"correct": False, "stage": "solo", "ts": "2026-08-01T00:00:00+00:00"}] * 20
check("a weak record seeds lower than a strong one",
      engine_bridge.posteriors_for(weak, KC)["ability"].mean < seeded.mean)

print("\nG. POSTERIORS SURVIVE A SAVE")
# `_save_user_state` builds its dict BY HAND, so a new dataclass field does not
# round-trip by itself — the difficulty offset was lost exactly this way. Driven
# through the public API only (get_user_state / save_user_state) and read back
# off disk, because the file is what a restart actually sees.
import json  # noqa: E402

from app.adaptive import get_user_state, save_user_state  # noqa: E402

live = get_user_state("engine-bridge-persist")
for qid in QIDS[:3]:
    answer(live, qid, True)
expected = engine_bridge.mastery(live, KC)
save_user_state("engine-bridge-persist")

with open(os.path.join(_TMP, "engine-bridge-persist.json"), encoding="utf-8") as fh:
    on_disk = json.load(fh)
check("kc_posteriors reached the save file at all",
      KC in (on_disk.get("kc_posteriors") or {}),
      "the writer builds its dict by hand — a new field is dropped silently")
restored = UserPracticeState(user_id="engine-bridge-restored")
# Everything `mastery` reads, not just the posteriors: the prediction also uses
# `encompassing`, a mean over BKT atom posteriors, so restoring the posterior
# alone would compare two different learners and call the difference a
# persistence bug.
for field_name in ("kc_posteriors", "atom_mastery", "atom_last_ts", "kc_ladder"):
    setattr(restored, field_name, on_disk.get(field_name) or {})
restored.self_reported_level = on_disk.get("self_reported_level")
check("a restart reads back the same mastery",
      expected is not None
      and abs((engine_bridge.mastery(restored, KC) or -1) - expected) < 1e-9,
      f"{expected:.6f}" if expected is not None else "no mastery")
check("the ladder rows carry the question they were about",
      all("question_id" in a for a in kc_graph.ladder_view(live, KC)["attempts"]),
      "served_stage cannot match a rung to a question without it")

shutil.rmtree(_TMP, ignore_errors=True)
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
    sys.exit(1)
print("All checks passed.")
