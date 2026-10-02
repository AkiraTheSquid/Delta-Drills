#!/usr/bin/env python3
"""Choose your own concept (app/concept_choice.py), 2026-09-28.

Seth: a list of candidate concepts on the Learner Home — the AI's pick among
them — each with the learner's ability; a chosen concept is practised until it
is learned (back to the Learner Home) or the model says the learner is not
ready for it. Covers:

  * the list: at most five, the AI's pick first and flagged, every entry on
    the knowledge frontier, ability as K, and the XP one solved problem would
    earn on the order of 5-15, a multiple of 5 (2026-09-29);
  * the route serves the chosen concept itself, never a prerequisite, and never
    asks a question twice in one session;
  * right answers → done, "ready" (K crossed the 80% line). Since the
    2026-10-02 ability model that takes climbing the pool — about 6 solves for
    a learner with no level, served at their level — so the run allows 10;
  * three misses → done, "not_ready" (K at or under 20% after 3 answers) —
    and NOT after two: the model needs MIN_ANSWERS first;
  * an unknown concept ends at once;
  * `keep` (a paused chosen block's concept) stays on the list.

Run: .venv/bin/python scripts/test_concept_choice.py
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ["USER_DATA_DIR"] = tempfile.mkdtemp(prefix="concept_choice_test_")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import concept_choice, kc_graph  # noqa: E402
from app.adaptive import UserPracticeState  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


def state():
    return UserPracticeState(user_id="concept-choice-test")


def run(st, kc, answers):
    """Answer the route's questions in order from `answers`; return the steps."""
    served, steps = [], []
    for ok in answers:
        step = concept_choice.plan(st, kc, served)
        steps.append(step)
        if step["done"]:
            return steps
        served.append(step["question_id"])
        kc_graph.record_kc_outcome(st, step["question_id"], ok)
    steps.append(concept_choice.plan(st, kc, served))
    return steps


print("the list")
st = state()
got = concept_choice.candidates(st)
items = got["items"]
front = set(kc_graph.frontier(st))
check("at most five", 0 < len(items) <= 5, str(len(items)))
check("the AI's pick is first and the only one flagged",
      items[0]["recommended"] and not any(i["recommended"] for i in items[1:]))
check("every entry is on the frontier", all(i["kc"] in front for i in items),
      str([i["kc"] for i in items if i["kc"] not in front]))
check("ability is K in [0, 1]", all(0 <= i["k"] <= 1 for i in items))
check("XP per problem: a multiple of 5, a problem's worth not a concept's",
      all(i["xp_per_problem"] % 5 == 0 and 0 < i["xp_per_problem"] <= i["worth"] for i in items),
      str([(i["kc"], i["xp_per_problem"], i["worth"]) for i in items]))
KC = items[0]["kc"]
off = next(kc for kc in kc_graph._registry() if kc not in {i["kc"] for i in items})
kept = concept_choice.candidates(st, keep=off)["items"]
check("a paused concept off the list is kept at the end", kept[-1]["kc"] == off and len(kept) == len(items) + 1)
check("keep of a listed concept adds nothing", len(concept_choice.candidates(st, keep=KC)["items"]) == len(items))

print("the route")
st = state()
steps = run(st, KC, [True] * 10)
asked = [s["question_id"] for s in steps if not s["done"]]
check("every question is on the chosen concept",
      all(KC in kc_graph.question_kcs(q) for q in asked), str(asked))
check("no question twice in a session", len(asked) == len(set(asked)), str(asked))
check("right answers end it as learned", steps[-1]["done"] and steps[-1]["reason"] == "ready",
      f'{len(asked)} answers, K {steps[-1]["p_target"]}')
check("learned means K at the 80% line", steps[-1]["p_target"] >= concept_choice.READY)
ds = [concept_choice.ability_model.difficulty(q, KC) for q in asked]
check("served at level: each solve is met with a problem no easier",
      all(a <= b for a, b in zip(ds, ds[1:])), str(ds))

st = state()
steps = run(st, KC, [False, False])
check("two misses are not enough to call it", not steps[-1]["done"], str(steps[-1].get("reason")))
st = state()
steps = run(st, KC, [False, False, False])
check("three misses end it as not ready", steps[-1]["done"] and steps[-1]["reason"] == "not_ready",
      f'K {steps[-1]["p_target"]}')

st = state()
check("an unknown concept ends at once", concept_choice.plan(st, "no.such-concept")["reason"] == "unknown")

print(f"\n{'FAIL' if fails else 'OK'} — {len(fails)} failed")
sys.exit(1 if fails else 0)
