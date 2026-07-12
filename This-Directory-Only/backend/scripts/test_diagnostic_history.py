#!/usr/bin/env python3
"""Validation suite for the history-informed placement diagnostic (2026-07-12).

Covers: history pseudo-probes in the area posterior (with recency decay),
history credit against the probe budget (with per-area coverage floor), the
relaxed min-probes path, the deploy-migration guard (active diagnostic whose
probe count already meets a newly-shrunken budget), and post-completion
estimate stability.

Run: .venv/bin/python scripts/test_diagnostic_history.py
Exits non-zero on any failed assertion. No pytest dependency.
"""
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import diagnostic as D  # noqa: E402
from app.adaptive import AttemptRecord, SubtopicState, UserPracticeState  # noqa: E402
from app.questions import get_subtopics, get_topic_for_subtopic  # noqa: E402

random.seed(42)
fails = []
NOW = datetime.now(timezone.utc)


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


def add_history(state, subtopic, theta_true, n, age_days=0.0):
    st = state.subtopic_states.setdefault(subtopic, SubtopicState(subtopic=subtopic))
    ts = (NOW - timedelta(days=age_days)).isoformat()
    for _ in range(n):
        diff = random.uniform(max(5.0, theta_true - 15), theta_true + 15)
        correct = random.random() < D.p_correct(theta_true, diff)
        st.history.append(AttemptRecord(
            question_id=0, subtopic=subtopic, difficulty_score=int(diff),
            grade=100.0 if correct else 0.0, correct=correct, timestamp=ts,
        ))
        st.n += 1


subtopics = get_subtopics()
numpy_subs = [s for s in subtopics if (get_topic_for_subtopic(s) or "") == "Numpy"]
n_areas = len({get_topic_for_subtopic(s) or "Other" for s in subtopics})
assert numpy_subs, "bank must have Numpy subtopics"

print("A. HISTORY EVIDENCE — posterior uses past attempts")
blank = UserPracticeState(user_id="t-blank")
rich = UserPracticeState(user_id="t-rich")
for s in numpy_subs[:3]:
    add_history(rich, s, 65, 5)  # fresh: full weight

mean_b, sd_b = D.posterior_summary(D.area_posterior(blank, "Numpy"))
mean_r, sd_r = D.posterior_summary(D.area_posterior(rich, "Numpy"))
check("history shifts area mean toward demonstrated ability",
      mean_r > mean_b + 10, f"rich={mean_r:.0f} blank={mean_b:.0f}")
check("history tightens area posterior", sd_r < sd_b - 5,
      f"rich sd={sd_r:.1f} blank sd={sd_b:.1f}")
hist = D._history_evidence(rich)
check("history evidence collected", len(hist) == 15, f"n={len(hist)}")
check("fresh attempts get ~full recency weight",
      all(h["w"] > 0.95 for h in hist))

print("B. RECENCY DECAY — stale evidence fades (BKT half-life)")
old = UserPracticeState(user_id="t-old")
for s in numpy_subs[:3]:
    add_history(old, s, 65, 5, age_days=3 * 365)
check("years-old attempts contribute no evidence",
      len(D._history_evidence(old)) == 0)
check("stale account keeps the full budget",
      D.effective_budget(old) == D.MAX_PROBES)
mean_o, sd_o = D.posterior_summary(D.area_posterior(old, "Numpy"))
check("stale posterior ≈ blank posterior",
      abs(mean_o - mean_b) < 1 and abs(sd_o - sd_b) < 1,
      f"old={mean_o:.0f}±{sd_o:.1f} blank={mean_b:.0f}±{sd_b:.1f}")
mid = UserPracticeState(user_id="t-mid")
add_history(mid, numpy_subs[0], 65, 5, age_days=14)
check("half-life-old attempts weigh ~0.5",
      all(0.45 < h["w"] < 0.55 for h in D._history_evidence(mid)))

print("C. BUDGET — history credit shortens the placement")
check("blank budget = MAX_PROBES", D.effective_budget(blank) == D.MAX_PROBES)
total_w = sum(h["w"] for h in D._history_evidence(rich))
uncovered = n_areas - 1  # rich has (fresh) evidence only in Numpy
expect = max(D.MIN_PROBES, uncovered, D.MAX_PROBES - int(D.HISTORY_WEIGHT * total_w))
check("budget = max(floor, MAX - credit) with decayed credit",
      D.effective_budget(rich) == expect,
      f"budget={D.effective_budget(rich)} expected={expect}")
heavy = UserPracticeState(user_id="t-heavy")
for s in numpy_subs[:3]:
    add_history(heavy, s, 65, 30)
check("per-subtopic history capped", len(D._history_evidence(heavy))
      == 3 * D.HISTORY_PER_SUBTOPIC)

print("D. COVERAGE FLOOR — one-area history can't starve unknown areas")
check("heavy one-area history floors budget at #uncovered areas",
      D.effective_budget(heavy) >= n_areas - 1,
      f"budget={D.effective_budget(heavy)} areas={n_areas}")
covered = UserPracticeState(user_id="t-covered")
by_area = {}
for s in subtopics:
    by_area.setdefault(get_topic_for_subtopic(s) or "Other", s)
for s in by_area.values():
    add_history(covered, s, 50, 5)
check("full-coverage history floors at MIN_PROBES",
      D.effective_budget(covered) == D.MIN_PROBES,
      f"budget={D.effective_budget(covered)}")

print("E. MIN PROBES — relaxed only with meaningful history")
check("blank min = MIN_PROBES", D.effective_min_probes(blank) == D.MIN_PROBES)
check("rich min = 2", D.effective_min_probes(rich) == 2)
check("stale-history min = MIN_PROBES", D.effective_min_probes(old) == D.MIN_PROBES)

print("F. MIGRATION GUARD — probes >= shrunken budget stops BEFORE serving")
mig = UserPracticeState(user_id="t-mig")
for s in numpy_subs[:3]:
    add_history(mig, s, 40, 7)
D.start(mig)
diag = D.get_diag(mig)
for i in range(D.effective_budget(mig)):  # fill exactly to the new budget
    diag["probes"].append({"question_id": 90000 + i, "subtopic": "x",
                           "topic": "Numpy", "difficulty": 40.0,
                           "result": "correct", "ts": NOW.isoformat()})
check("should_finish fires at budget without recording another probe",
      D.should_finish(mig) is True,
      f"probes={len(diag['probes'])} budget={D.effective_budget(mig)}")
D.start(rich)
check("fresh diagnostic does not insta-finish", D.should_finish(rich) is False)

print("G. STOP RULE — probe recording still finishes at budget")
sim = UserPracticeState(user_id="t-sim")
for s in numpy_subs[:2]:
    add_history(sim, s, 60, 10)
D.start(sim)
served = 0
while True:
    q = D.select_probe(sim)
    if q is None:
        D.finish(sim)
        break
    served += 1
    res = "correct" if random.random() < D.p_correct(45, q.difficulty_score) else "incorrect"
    d = D.record_probe(sim, q, res)
    if not d["active"]:
        break
check("placement finishes within effective budget",
      served <= D.effective_budget(sim), f"served={served}")
check("finish seeded atoms", D.get_diag(sim).get("atoms_seeded", 0) > 0)

print("H. COMPLETION — estimates frozen against later practice")
before = D.area_estimates(sim)
add_history(sim, numpy_subs[2], 90, 10)  # post-placement practice
after = D.area_estimates(sim)
check("completed estimates don't drift with new history", before == after)

print("I. OVERRIDE AFTER FINISH — re-finish re-seeds from corrected posterior")
ov = UserPracticeState(user_id="t-override")
D.start(ov)
diag = D.get_diag(ov)
for i in range(D.MIN_PROBES - 1):
    diag["probes"].append({"question_id": 91000 + i, "subtopic": "x",
                           "topic": "Numpy", "difficulty": 40.0,
                           "result": "incorrect", "ts": NOW.isoformat()})


class _FakeQ:
    id = 91999
    subtopic = "Numpy: Core array literacy"
    topic = "Numpy"
    difficulty_score = 50.0


while len(diag["probes"]) < D.MAX_PROBES - 1:  # leave room for ONE finishing probe
    diag["probes"].append({"question_id": 92000 + len(diag["probes"]), "subtopic": "x",
                           "topic": "Einops", "difficulty": 40.0,
                           "result": "incorrect", "ts": NOW.isoformat()})
D.record_probe(ov, _FakeQ(), "incorrect")
check("finishing probe closed the diagnostic", not D.get_diag(ov)["active"])
theta_before = {e["topic"]: e["theta"] for e in D.area_estimates(ov)}["Numpy"]
assert D.override_probe(ov, _FakeQ.id, True)
D.finish(ov)  # router re-finishes after overriding the finishing probe
theta_after = {e["topic"]: e["theta"] for e in D.area_estimates(ov)}["Numpy"]
check("override of finishing probe moves the frozen estimate",
      theta_after > theta_before, f"{theta_before} -> {theta_after}")

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("ALL PASS")
