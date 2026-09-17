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
import json
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
_age_pending(c, 200)                                        # under every concept's cap (shortest is 300)
D.record_probe(c, q, "dont_know", elapsed_secs=0)          # client claims nothing elapsed
spent = D.get_diag(c)["spent_secs"]
check("a client reporting 0 cannot buy free time", 195 <= spent <= 205, spent)
check("remaining_secs is the budget minus what was spent",
      D.remaining_secs(c) == 3600 - spent, (D.remaining_secs(c), spent))

q = D.select_probe(c)
ckc = D.get_diag(c)["pending"]["kc"]
_age_pending(c, 99_999)
D.record_probe(c, q, "dont_know", elapsed_secs=0)
check("no single problem can spend more than its concept's cap",
      D.get_diag(c)["spent_secs"] - spent == D.kc_cap_secs(ckc) <= D.PER_PROBLEM_SECS,
      (D.get_diag(c)["spent_secs"] - spent, ckc))

# The last problem of a run gets whatever is left, not a full 20:00 — this is
# what stops a 1-hour plan from finishing at 1h19m.
tail = UserPracticeState(user_id="tail")
D.start(tail, hours=1)
D.get_diag(tail)["spent_secs"] = 3600 - 300
check("the final problem is only given the time that is left",
      D.problem_secs_allowed(tail) == 300, D.problem_secs_allowed(tail))

# The clock is PER CONCEPT (lessons/placement_time_caps.json): a one-call
# Python drill gets 5:00, einops gets ARENA's 10:00, nothing more than the
# 20:00 ceiling. The pending probe's concept decides both what the question
# payload quotes and what the server may charge for it.
check("every concept has a clock and none exceeds the ceiling",
      all(0 < D.kc_cap_secs(k) <= D.PER_PROBLEM_SECS for k in D.kc_graph._registry()),
      sorted(set(D.kc_cap_secs(k) for k in D.kc_graph._registry())))
check("a one-call python drill gets five minutes, einops gets ARENA's ten",
      (D.kc_cap_secs("python.indexing"), D.kc_cap_secs("einops.merge-axes"),
       D.kc_cap_secs("numpy.broadcasting-rules"), D.kc_cap_secs("raytracing.make-rays-1d"))
      == (300, 600, 600, 900))
check("an unknown concept falls back to the default, never to nothing",
      D.kc_cap_secs("no.such-kc") == D.PER_PROBLEM_SECS and D.kc_cap_secs(None) == D.PER_PROBLEM_SECS)
per = UserPracticeState(user_id="perkc")
D.start(per, hours=6)
q = D.select_probe(per)
pkc = D.get_diag(per)["pending"]["kc"]
check("the next problem's clock is its concept's cap",
      D.problem_secs_allowed(per) == D.kc_cap_secs(pkc), (D.problem_secs_allowed(per), pkc))
_age_pending(per, 99_999)
D.record_probe(per, q, "dont_know", elapsed_secs=0)
check("a probe is charged at most its concept's cap, not the flat 20:00",
      D.get_diag(per)["spent_secs"] == D.kc_cap_secs(pkc),
      (D.get_diag(per)["spent_secs"], D.kc_cap_secs(pkc)))
check("with nothing pending the clock quoted is the ceiling",
      D.problem_secs_allowed(per) == D.PER_PROBLEM_SECS)
lo, hi = D.cap_range(per)
check("the picker's range spans the assessed concepts", lo == 300 and hi == 900, (lo, hi))

# A CORRECT answer past the clock (plus grace) is a miss: the charge is capped
# at the clock, so an uncapped answer would be free time and free evidence.
late = UserPracticeState(user_id="late")
D.start(late, hours=6)
q = D.select_probe(late)
lkc = D.get_diag(late)["pending"]["kc"]
_age_pending(late, D.kc_cap_secs(lkc) + D.LATE_GRACE_SECS + 30)
D.record_probe(late, q, "correct")
rec = D.get_diag(late)["probes"][-1]
check("a correct answer landing past the clock is recorded as a miss",
      rec["result"] == "incorrect" and rec["late"] is True, rec)
q = D.select_probe(late)
lkc = D.get_diag(late)["pending"]["kc"]
_age_pending(late, D.kc_cap_secs(lkc) + D.LATE_GRACE_SECS - 30)
D.record_probe(late, q, "correct")
rec = D.get_diag(late)["probes"][-1]
check("inside the grace a correct answer still counts",
      rec["result"] == "correct" and rec["late"] is False, rec)
q = D.select_probe(late)
_age_pending(late, 99_999)
D.record_probe(late, q, "incorrect", elapsed_secs=1)
check("a late miss is a miss, flagged", D.get_diag(late)["probes"][-1]["result"] == "incorrect"
      and D.get_diag(late)["probes"][-1]["late"] is True)

# A resumed probe is handed what is LEFT of its clock, not a fresh one — the
# second-device / cleared-storage case. After a real break (past the grace)
# the serve is restamped once and the clock is whole again, which is what the
# client already does after RETURN_GRACE_SECS.
res = UserPracticeState(user_id="resume-clock")
D.start(res, hours=6)
q = D.select_probe(res)
rkc = D.get_diag(res)["pending"]["kc"]
check("a fresh probe advertises its whole clock",
      D.pending_secs_left(res) == D.kc_cap_secs(rkc))
_age_pending(res, 100)
D.select_probe(res)
left = D.pending_secs_left(res)
check("a resumed probe advertises the time actually left",
      D.kc_cap_secs(rkc) - 102 <= left <= D.kc_cap_secs(rkc) - 99, (left, rkc))
_age_pending(res, D.kc_cap_secs(rkc) + D.LATE_GRACE_SECS + 5)
check("past the clock nothing is left", D.pending_secs_left(res) == 0)
q2 = D.select_probe(res)
rec = D.get_diag(res)["probes"][-1]
check("coming back after a real break: the abandoned probe is a timed-out miss",
      rec["question_id"] == q.id and rec["result"] == "incorrect" and rec["timed_out"] is True
      and rec["secs"] == D.kc_cap_secs(rkc), rec)
check("and a NEW problem is served on a whole clock",
      q2 is not None and q2.id != q.id and D.pending_secs_left(res) == D.kc_cap_secs(D.get_diag(res)["pending"]["kc"]),
      (q2 and q2.id, q.id))

# …and that timed-out record is FINAL: answering the walked-away question
# later (the /submit path takes any id while placement is active) cannot
# refund its charge or replace the miss.
spent_before = D.get_diag(res)["spent_secs"]
D.record_probe(res, q, "correct", elapsed_secs=0)
rec = D.get_diag(res)["probes"][0]
check("a timed-out probe cannot be answered later for credit",
      rec["question_id"] == q.id and rec["result"] == "incorrect" and rec["timed_out"] is True
      and D.get_diag(res)["spent_secs"] == spent_before and len(D.get_diag(res)["probes"]) == 1,
      (rec, D.get_diag(res)["spent_secs"], spent_before))
check("a late record is final too",
      (lambda st: (D.record_probe(st, D.get_question_by_id(st_q := D.get_diag(st)["probes"][0]["question_id"]), "correct", elapsed_secs=0),
                   D.get_diag(st)["probes"][0]["result"] == "incorrect")[1])(late))

# Only the problem ON SCREEN is evidence. /submit takes any id during
# placement; a re-submit of an answered question, or an id never served,
# must change nothing — no record, no refund, no replacement.
only = UserPracticeState(user_id="only-pending")
D.start(only, hours=6)
q1 = D.select_probe(only)
_age_pending(only, 30)
D.record_probe(only, q1, "incorrect")
q2 = D.select_probe(only)
before = json.dumps(D.get_diag(only), sort_keys=True, default=str)
D.record_probe(only, q1, "correct", elapsed_secs=0)
check("re-submitting an answered probe changes nothing",
      json.dumps(D.get_diag(only), sort_keys=True, default=str) == before)
stranger = next(D.get_question_by_id(i) for i in range(1, 2000)
                if D.get_question_by_id(i) is not None and i not in (q1.id, q2.id))
D.record_probe(only, stranger, "dont_know")
check("an id never served records nothing",
      json.dumps(D.get_diag(only), sort_keys=True, default=str) == before)
D.record_probe(only, q2, "correct")
check("the problem on screen still records", D.get_diag(only)["probes"][-1]["question_id"] == q2.id)

# The cap is snapshotted on the pending probe: a cap-table change (or a
# restart with a new table) mid-probe must not move the clock under the
# learner. The table is only consulted for the NEXT problem.
snap = UserPracticeState(user_id="snap")
D.start(snap, hours=6)
q = D.select_probe(snap)
pend = D.get_diag(snap)["pending"]
pend["cap_secs"] = 77
check("the clock displayed and charged is the one served with",
      D.problem_secs_allowed(snap) == 77 and D.pending_secs_left(snap) == 77,
      (D.problem_secs_allowed(snap), D.pending_secs_left(snap)))
_age_pending(snap, 99_999)
D.record_probe(snap, q, "correct")
check("… including what a late answer is judged and charged against",
      D.get_diag(snap)["probes"][-1]["secs"] == 77 and D.get_diag(snap)["probes"][-1]["result"] == "incorrect")

# Walking away is not free: every abandoned probe is charged its whole clock
# and recorded, so waiting probes out drains the plan instead of dodging it,
# and the run ends when the plan does.
drain = UserPracticeState(user_id="drain")
D.start(drain, hours=1)
n = 0
while not D.get_diag(drain)["completed_at"] and n < 100:
    q = D.select_probe(drain)
    if q is None:
        break
    dkc = D.get_diag(drain)["pending"]["kc"]
    _age_pending(drain, D.kc_cap_secs(dkc) + D.LATE_GRACE_SECS + 5)
    n += 1
D.select_probe(drain)
dd = D.get_diag(drain)
check("repeated walk-aways run the plan down and close the run",
      n < 100 and (dd["completed_at"] or D.should_finish(drain))
      and all(p["timed_out"] for p in dd["probes"]) and dd["spent_secs"] >= 3600 - D.MIN_REMAINING_SECS,
      (n, dd["spent_secs"], len(dd["probes"])))


# A probe served BEFORE the table existed (no `cap_secs`) ran on the flat
# 20:00 with a client that handed the whole clock back after a break. The
# deploy must not turn it into a timed-out miss: it is judged on 20:00 and
# restamped once, uncharged.
legacy = UserPracticeState(user_id="legacy-pending")
D.start(legacy, hours=6)
q = D.select_probe(legacy)
lp = D.get_diag(legacy)["pending"]
lp.pop("cap_secs", None)
_age_pending(legacy, 5 * 3600)
check("a legacy pending probe is judged on the 20:00 it was served with",
      D.problem_secs_allowed(legacy) == D.PER_PROBLEM_SECS)
q2 = D.select_probe(legacy)
lp = D.get_diag(legacy)["pending"]
check("… and is resumed whole, once, not recorded as a timed-out miss",
      q2 is not None and q2.id == q.id and not D.get_diag(legacy)["probes"]
      and lp.get("cap_secs") == D.PER_PROBLEM_SECS and lp.get("legacy_restamp") is True
      and D.get_diag(legacy)["spent_secs"] == 0 and D.pending_secs_left(legacy) >= D.PER_PROBLEM_SECS - 1,
      lp)
D.record_probe(legacy, q2, "correct")
check("and its answer on that clock counts", D.get_diag(legacy)["probes"][-1]["result"] == "correct")


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

# --- E. the answer route ----------------------------------------------------
# The route itself, not just the module: which ids /diagnostic/answer accepts
# with and without a pending probe, and that a retry never rewrites a record.
from fastapi import HTTPException
from app.practice.diagnostic_router import diagnostic_answer
from app.practice_schemas import DiagnosticAnswerRequest
from app.adaptive import get_user_state  # the route reads the shared store, so seed it there


class _User:
    def __init__(self, uid): self.id = uid


def _answer(uid, qid, result):
    try:
        return diagnostic_answer(DiagnosticAnswerRequest(question_id=qid, result=result), _User(uid)), None
    except HTTPException as exc:
        return None, exc.status_code


ru = "route-user"
route_state = get_user_state(ru)
D.start(route_state, hours=6)
first = D.select_probe(route_state)
check("an arbitrary id with a probe on screen is refused", _answer(ru, first.id + 100000, "dont_know")[1] in (404, 409))
st, code = _answer(ru, first.id, "correct")
check("the problem on screen is recorded", code is None and st.probes_done == 1, code)
before = [dict(p) for p in D.get_diag(route_state)["probes"]]
st, code = _answer(ru, first.id, "dont_know")
check("a late dont_know behind a graded answer does not rewrite it",
      code is None and D.get_diag(route_state)["probes"][-1]["result"] == "correct"
      and len(D.get_diag(route_state)["probes"]) == 1, (code, D.get_diag(route_state)["probes"]))
check("the replay answers with the current state, not a second record", st is not None and st.probes_done == 1)
second = D.select_probe(route_state)
st, code = _answer(ru, second.id, "incorrect")
check("the next probe records too", code is None and st.probes_done == 2)
check("an OLDER probe's id is refused once a newer one exists", _answer(ru, first.id, "dont_know")[1] == 409)
check("an id never served is refused between probes", _answer(ru, first.id + 100000, "dont_know")[1] in (404, 409))
check("a finished run answers 400", (D.finish(route_state), _answer(ru, second.id, "dont_know")[1])[1] == 400)

print()
if fails:
    print(f"{len(fails)} FAILED: " + ", ".join(fails))
    sys.exit(1)
print("all placement checks passed")
