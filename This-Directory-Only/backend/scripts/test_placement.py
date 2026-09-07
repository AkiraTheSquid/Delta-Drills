#!/usr/bin/env python3
"""Validation suite for the graph-wide placement diagnostic (2026-09-07).

Covers what the time-plan rewrite made true and what it broke:

  A. LEGACY MIGRATION. A run started before the rewrite has no `plan`, so it has
     no clock and no budget, and the placement page hides the 1h/3h/6h picker
     for as long as a run is `active`. Nine of sixty production learners sat on
     "In progress · 0 of at most 15" with no picker and no way to finish. An
     active run with no plan is an abandoned old one: it reopens as "not
     started" and the probe log is kept.
  B. RESUME. An unanswered probe is the problem on screen. A reload has to get
     THAT problem back — re-rolling burnt the question (it is already in
     `served_question_ids`) and left its serve-to-answer time uncharged, which
     made the plan's hard cap dodgeable by reloading.
  C. THE CLOCK. Time is the server's serve-to-answer reading, capped at the
     per-problem allowance; the client's own number can never buy free time.
  D. THE STOPPING RULE. A plan ends when the clock runs out as well as when the
     evidence settles, and a 6-hour run really does reach every concept.

The old suite for this module, scripts/test_diagnostic_history.py, tests the
retired topic-θ model (`p_correct`, `_DIFF_FLOOR`) and has been failing at
import since that model was removed. This is its replacement; nothing here
depends on the network or on a running server.

Run: .venv/bin/python scripts/test_placement.py
Exits non-zero on any failed assertion. No pytest dependency.
"""
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

_RUNTIME = tempfile.mkdtemp(prefix="dd-placement-test-")
os.environ.setdefault("USER_DATA_DIR", _RUNTIME + "/user_data")
os.environ.setdefault("STORAGE_DIR", _RUNTIME + "/storage")
os.environ.setdefault("DELTA_FEEDBACK_AI_DIR", _RUNTIME + "/ai_feedback")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import diagnostic as D  # noqa: E402
from app.adaptive import UserPracticeState  # noqa: E402

fails = []


def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  [{extra}]" if extra and not cond else ""))
    if not cond:
        fails.append(name)


# --- A. legacy migration ------------------------------------------------------

print("A. LEGACY MIGRATION — a plan-less active run reopens as not started")

# Exactly the shape found on production for user c813fa78 on 2026-09-07.
seth = UserPracticeState(user_id="legacy-zero")
seth.diagnostic = {"active": True, "completed_at": None, "declined": False, "probes": []}
d = D.get_diag(seth)
check("active is cleared, so the picker is reachable", d["active"] is False, d)
check("the run is not passed off as completed", d["completed_at"] is None)
check("the learner is not marked as having declined", d["declined"] is False)
check("nothing is pending", d["pending"] is None)

# The four production accounts that had answered something first. Old probes
# carry no `kc`, which is why they were never evidence for the new estimator.
OLD_PROBES = [{
    "question_id": 314, "subtopic": "Einops: Rearrange", "topic": "Einops",
    "difficulty": 48, "result": "dont_know", "ts": "2026-08-23T19:55:06.720375+00:00",
}]
partial = UserPracticeState(user_id="legacy-answered")
partial.diagnostic = {"active": True, "completed_at": None, "declined": False,
                      "probes": [dict(p) for p in OLD_PROBES],
                      "atoms_seeded": 3, "estimates": [{"topic": "Einops"}]}
d = D.get_diag(partial)
check("an answered legacy run also reopens", d["active"] is False)
check("its probe log is preserved, not deleted", d["probes"] == OLD_PROBES, d["probes"])
check("kc-less legacy probes stay out of the estimator", D._model_probes(d) == [], D._model_probes(d))

live = UserPracticeState(user_id="planned")
D.start(live, hours=6)
check("a run WITH a plan is left alone", D.get_diag(live)["active"] is True)
check("its budget is the 6 hours that were asked for", D.get_diag(live)["plan"]["budget_secs"] == 21600)
check("get_diag stays idempotent", D.get_diag(live)["active"] is True)
check("an unknown length falls back to the default, never to a 0-second plan",
      D.start(UserPracticeState(user_id="odd"), hours=4)["plan"]["hours"] == D.DEFAULT_PLAN_HOURS)


# --- B. resume ----------------------------------------------------------------

print("B. RESUME — a reload gets the problem it was already on")

u = UserPracticeState(user_id="resume")
D.start(u, hours=1)
first = D.select_probe(u)
check("a run serves a probe at all", first is not None)
served_at = D.get_diag(u)["pending"]["served_at"]
again = D.select_probe(u)
check("the reload returns the SAME question", again is not None and again.id == first.id,
      (getattr(first, "id", None), getattr(again, "id", None)))
check("the reload does not restart the problem clock",
      D.get_diag(u)["pending"]["served_at"] == served_at)
D.record_probe(u, first, "dont_know", elapsed_secs=0)
check("answering clears the pending probe", D.get_diag(u)["pending"] is None)
nxt = D.select_probe(u)
check("only then does a new question come out", nxt is not None and nxt.id != first.id)


# --- C. the clock -------------------------------------------------------------

print("C. THE CLOCK — the server's reading wins and the per-problem cap holds")


def _age_pending(state, secs):
    D.get_diag(state)["pending"]["served_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=secs)).isoformat()


c = UserPracticeState(user_id="clock")
D.start(c, hours=1)
q = D.select_probe(c)
_age_pending(c, 400)
D.record_probe(c, q, "dont_know", elapsed_secs=0)          # client claims nothing elapsed
spent = D.get_diag(c)["spent_secs"]
check("a client reporting 0 cannot buy free time", 395 <= spent <= 405, spent)
check("remaining_secs is the budget minus what was spent",
      D.remaining_secs(c) == 3600 - spent, (D.remaining_secs(c), spent))

q = D.select_probe(c)
_age_pending(c, 99_999)
D.record_probe(c, q, "dont_know", elapsed_secs=0)
check("no single problem can spend more than its 20:00 cap",
      D.get_diag(c)["spent_secs"] - spent == D.PER_PROBLEM_SECS,
      D.get_diag(c)["spent_secs"] - spent)

# The last problem of a run gets whatever is left, not a full 20:00 — this is
# what stops a 1-hour plan from finishing at 1h19m.
tail = UserPracticeState(user_id="tail")
D.start(tail, hours=1)
D.get_diag(tail)["spent_secs"] = 3600 - 300
check("the final problem is only given the time that is left",
      D.problem_secs_allowed(tail) == 300, D.problem_secs_allowed(tail))


# --- D. the stopping rule -----------------------------------------------------

print("D. STOPPING — on the clock as well as on the evidence")

out_of_time = UserPracticeState(user_id="expired")
D.start(out_of_time, hours=1)
D.get_diag(out_of_time)["spent_secs"] = 3600
check("a spent budget stops the run", D.should_finish(out_of_time) is True)
check("and leaves no time for another problem", D.problem_secs_allowed(out_of_time) == 0)

fresh = UserPracticeState(user_id="fresh")
D.start(fresh, hours=6)
check("a fresh 6-hour run does not stop immediately", D.should_finish(fresh) is False)

# A whole 6-hour run, answered instantly so it can only end on evidence. The
# point of the 6-hour plan is that it reaches EVERY assessed concept, and every
# concept ARENA has a problem for is probed with ARENA's own problem.
run = UserPracticeState(user_id="full-6h")
D.start(run, hours=6)
for _ in range(500):
    probe = D.select_probe(run)
    if probe is None or not D.get_diag(run)["active"]:
        break
    D.record_probe(run, probe, "dont_know", elapsed_secs=0)
diag = D.get_diag(run)
rows = D.kc_estimates(run)
assessed = D.assessed_kcs(run)
links = D._arena_links()
arena_kcs = {k for k in assessed if links.get(k)}
check("the run ends on its own", diag["active"] is False and diag["completed_at"] is not None,
      (diag["active"], diag["completed_at"]))
check("every assessed concept was probed",
      all(r["probes"] for r in rows), [r["kc"] for r in rows if not r["probes"]])
check("every ARENA-linked concept got an ARENA problem",
      arena_kcs and all(r["arena_probes"] for r in rows if r["kc"] in arena_kcs),
      sorted(k for k in arena_kcs if not next(r["arena_probes"] for r in rows if r["kc"] == k)))
check("no concept was probed twice with the same question",
      len({p["question_id"] for p in diag["probes"]}) == len(diag["probes"]))
check("a placement never certifies mastery",
      all(v <= D.SEED_MASTERY_CAP for v in run.atom_mastery.values()),
      max(run.atom_mastery.values(), default=0))

print()
if fails:
    print(f"{len(fails)} FAILED: " + ", ".join(fails))
    sys.exit(1)
print("all placement checks passed")
