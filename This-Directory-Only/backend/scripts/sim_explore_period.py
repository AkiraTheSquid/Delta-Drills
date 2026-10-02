#!/usr/bin/env python3
"""sim_explore_period.py — simulated learners through the REAL explore/exploit.

Seth, 2026-10-01: a LeetCode learner who has done a lot of LeetCode saw her
whole graph red after one answer, and the app had stopped probing ("if she
answers one or two questions and it instantly doesn't explore you must of done
something terribly wrong with the hyperparameters ... that's why you need to
simulate agents. simulate it both for leetcode and for the arena").

Unlike `sims/` (a research copy of the model) this drives the app itself:
`kc_graph.frontier` (greedy order + `kc_explore.reorder`), `kc_explore.probing`,
`kc_graph.kc_stage`, and `kc_graph.record_kc_outcome` with real question ids.
Each simulated learner has a HIDDEN true knowledge set and answers from it:
a known concept right unless slipped, an unknown one right only by a guess.
Taught concepts are learned at a fixed rate, so the run can finish.

Per learner it reports:
  probes     answers served as diagnostic probes (the explore period)
  first_x    the step at which the first lesson / exploit item was served
  waste      lessons served on concepts the learner already knew
  false_ok   concepts settled (counted learned) the learner did not know
  minutes    to finish the course (or the cap), at the minutes below
  done       every course concept learned before the cap

Minutes are assumptions, not measurements: a LeetCode problem ~15 min, an
ARENA drill ~4 min, a lesson ~8 min.

Run: .venv/bin/python scripts/sim_explore_period.py [--course leetcode|arena]
     [--seeds N] [--set NAME=VALUE ...]   (VALUE is a Python literal; NAME is
     an attribute of app.kc_explore or app.kc_evidence, e.g.
     --set kc_explore.PROBE_COST_RATIO_CODE=0.3)
"""
from __future__ import annotations

import argparse
import ast
import atexit
import os
import random
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

if "USER_DATA_DIR" not in os.environ:
    _tmp = tempfile.mkdtemp(prefix="explore_period_sim_")
    os.environ["USER_DATA_DIR"] = _tmp
    atexit.register(shutil.rmtree, _tmp, True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import course_registry, kc_evidence, kc_explore, kc_graph  # noqa: E402
from app.adaptive import UserPracticeState  # noqa: E402

MINUTES = {"leetcode": {"answer": 15.0, "lesson": 8.0},
           "arena": {"answer": 4.0, "lesson": 8.0}}
GUESS, SLIP = kc_explore.P_GUESS_CODE, kc_explore.P_SLIP_CODE  # the app's own
LEARN_LESSON, LEARN_ANSWER = 0.35, 0.20
CAP = {"leetcode": 900, "arena": 1500}


def course_kcs(course):
    return [k for k in kc_graph._registry() if course_registry.course_of(k) == course]


def closed(known, reg):
    """Knowledge closed under prerequisites: nobody knows DP without arrays."""
    out, changed = set(known), True
    while changed:
        changed = False
        for k in list(out):
            if any(p not in out for p in reg[k]["prereqs"] if p in reg):
                out.discard(k)
                changed = True
    return out


def learner_types(course):
    reg = kc_graph._registry()
    kcs = course_kcs(course)
    _desc, depth = kc_graph._closure()
    deepest = max(depth.get(k, 0) for k in kcs)
    by_frac = lambda f: {k for k in kcs if depth.get(k, 0) <= f * deepest}
    return {
        "novice": lambda rng: set(),
        "knows 1/3": lambda rng: closed(by_frac(1 / 3), reg),
        "knows 2/3": lambda rng: closed(by_frac(2 / 3), reg),
        "expert": lambda rng: set(kcs),
        "spotty 70%": lambda rng: closed({k for k in kcs if rng.random() < 0.85}, reg),
    }


def run(course, known, level, rng, first_miss=False):
    s = UserPracticeState(user_id="sim")
    s.kc_ladder = {}
    s.self_reported_level = level
    s.study_courses = [course]
    s.course_shares = {course: 1.0}  # a standalone course at share 0 is off
    kcs = set(course_kcs(course))
    known = set(known)
    used = {}
    mins = MINUTES[course]
    out = dict(probes=0, first_x=None, waste=0, minutes=0.0, steps=0, done=False)
    settled_unknown = set()
    for step in range(CAP[course]):
        fr = [k for k in kc_graph.frontier(s) if k in kcs]
        if not fr:
            out["done"] = all(kc_graph.kc_is_learned(s, k) for k in kcs)
            break
        kc = fr[0]
        probe = kc_explore.probing(s, kc)
        stage = kc_graph.kc_stage(s, kc)
        if stage == "worked":
            # The lesson page: not graded, and it teaches.
            if out["first_x"] is None:
                out["first_x"] = step
            if kc in known:
                out["waste"] += 1
            elif rng.random() < LEARN_LESSON:
                known.add(kc)
            kc_graph.note_worked_seen(s, kc)
            out["minutes"] += mins["lesson"]
            continue
        pool = kc_graph.questions_at_stage(kc_graph.questions_for_kc(kc), stage) \
            or list(kc_graph.questions_for_kc(kc))
        i = used.get(kc, 0)
        used[kc] = i + 1
        qid = pool[i % len(pool)]
        if probe:
            out["probes"] += 1
        elif out["first_x"] is None:
            out["first_x"] = step
        ok = rng.random() < ((1 - SLIP) if kc in known else GUESS)
        if first_miss and step == 0:
            ok = False  # ran out of time on the first problem
        kc_graph.record_kc_outcome(s, qid, ok, stage=stage, example=False)
        out["minutes"] += mins["answer"]
        if kc not in known and not probe and rng.random() < LEARN_ANSWER:
            known.add(kc)
        for k in kcs:
            if k not in known and k not in settled_unknown and kc_explore.settled(s, k):
                settled_unknown.add(k)
        out["steps"] = step + 1
    else:  # the cap: the last action may have finished the course
        out["done"] = all(kc_graph.kc_is_learned(s, k) for k in kcs)
    out["false_ok"] = len(settled_unknown)
    if out["first_x"] is None:
        out["first_x"] = out["steps"]
    return out


def apply_overrides(pairs):
    mods = {"kc_explore": kc_explore, "kc_evidence": kc_evidence}
    for pair in pairs or []:
        name, value = pair.split("=", 1)
        mod, attr = name.split(".", 1)
        setattr(mods[mod], attr, ast.literal_eval(value))
    # Derived constants kc_explore computes at import.
    X = kc_explore
    X.SETTLE_LOGODDS = X._logit(X.SETTLE_P)
    X.OUT_LOGODDS = X._logit(X.OUT_OF_STATE)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", choices=["leetcode", "arena"], default="leetcode")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--levels", default="beginner,None,strong")
    ap.add_argument("--types", default="")
    ap.add_argument("--set", action="append")
    a = ap.parse_args(argv)
    apply_overrides(a.set)
    types = learner_types(a.course)
    if a.types:
        types = {k: v for k, v in types.items() if k in a.types.split(",")}
    levels = [None if x == "None" else x for x in a.levels.split(",")]
    print(f"course={a.course} seeds={a.seeds} overrides={a.set or []}")
    print(f"{'level':<9}{'learner':<14}{'probes':>7}{'first_x':>8}{'waste':>6}"
          f"{'false_ok':>9}{'hours':>7}{'done':>5}")
    rows = []
    for level in levels:
        cases = [(n, f, False) for n, f in types.items()]
        if "knows 2/3" in types:
            cases.append(("2/3 +timeout", types["knows 2/3"], True))
        for name, draw, first_miss in cases:
            res = []
            for seed in range(a.seeds):
                rng = random.Random(f"{a.course}|{name}|{seed}")
                res.append(run(a.course, draw(rng), level, rng, first_miss))
            med = lambda key: statistics.median(r[key] for r in res)
            row = (str(level), name, med("probes"), med("first_x"), med("waste"),
                   med("false_ok"), med("minutes") / 60, sum(r["done"] for r in res))
            rows.append(row)
            print(f"{row[0]:<9}{row[1]:<14}{row[2]:>7.0f}{row[3]:>8.0f}{row[4]:>6.0f}"
                  f"{row[5]:>9.0f}{row[6]:>7.1f}{row[7]:>3}/{a.seeds}", flush=True)
    return rows


if __name__ == "__main__":
    main()
