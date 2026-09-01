#!/usr/bin/env python3
"""test_attempt_attribution.py — the log carries WHY, not only what (2026-09-01).

Covers the two additions that make the encompassing-regression metric
(docs/spec-graph-metadata-audit-layer.md §3a) computable per edge later:

  * `AttemptRow.feature_sources` — the provenance dict `engine_features` builds
    (per-atom `encompassed` mastery, per-KC `prereqs`) rides the attempt row,
    passed through `engine_bridge` -> `record_attempt(sources=...)`.
  * `bkt_update` rows — what the BKT channel did with the same answer, FIRe
    credits included, written from the scoring tail. Never evidence.

The end-to-end group drives `finalize_attempt` through the same call order as
the router (copied from test_engine_bridge.py) because the wire, not the
serializer, is the part that goes missing.

Run: .venv/bin/python scripts/test_attempt_attribution.py
Exits non-zero on any failed assertion. No pytest dependency.
"""
from __future__ import annotations

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
_TMP = tempfile.mkdtemp(prefix="attribution-test-")
os.environ["USER_DATA_DIR"] = _TMP

from app import attempt_log, kc_graph  # noqa: E402
from app.adaptive import UNRATED, UserPracticeState, record_attempt  # noqa: E402
from app.practice.attempt_scoring import finalize_attempt  # noqa: E402
from app.prioritization import record_ladder_outcome  # noqa: E402
from app.questions import get_question_by_id  # noqa: E402

PASS, FAIL = [], []


def check(label, condition, detail=""):
    (PASS if condition else FAIL).append(label)
    print(f"  {'PASS' if condition else 'FAIL'}  {label}" + (f"  — {detail}" if detail else ""))


print("A. UNIT — sources ride the attempt row")
row = attempt_log.record_attempt(
    "unit-user", "kc.x", 1, "faded", {"ability": 0.0},
    type("P", (), {"p": 0.5, "logit_mean": 0.0, "logit_var": 1.0})(),
    True, sources={"encompassed": {"atom-a": 0.42}, "prereqs": {"kc.w": 0.9}},
)
back = list(attempt_log.iter_rows("unit-user"))
check("feature_sources round-trips",
      back and back[-1].feature_sources == {"encompassed": {"atom-a": 0.42}, "prereqs": {"kc.w": 0.9}})
check("row is still evidence", back[-1].is_graded)
legacy = attempt_log.AttemptRow.from_dict(
    {"ts": "2026-08-01T00:00:00Z", "kind": "attempt", "user_id": "unit-user"})
check("legacy row (no field) reads as None", legacy is not None and legacy.feature_sources is None)

print("B. UNIT — bkt_update rows")
r = attempt_log.record_bkt_update("unit-user", 7, {"atom-b": {"atom-b": 0.7, "atom-a": 0.31}})
check("kind is bkt_update", r is not None and r.kind == attempt_log.KIND_BKT_UPDATE)
check("bkt_update is never evidence", not r.is_graded)
check("credits kept per practiced atom",
      r.feature_sources["bkt_changed"]["atom-b"] == {"atom-b": 0.7, "atom-a": 0.31})
check("empty update logs nothing", attempt_log.record_bkt_update("unit-user", 7, {}) is None)
check("bkt_update row round-trips the reader",
      any(x.kind == "bkt_update" for x in attempt_log.iter_rows("unit-user")))

print("C. END TO END — the router's call order writes both")


def _tagged_question():
    for kc in kc_graph._registry():
        for qid in kc_graph.questions_for_kc(kc):
            q = get_question_by_id(qid)
            if q is not None and (getattr(q, "atom_tags", []) or []):
                return q
    raise SystemExit("no question carries atom tags — fixture is wrong, not the code")


q = _tagged_question()
state = UserPracticeState(user_id="e2e-user")
record_attempt(user_state=state, question_id=q.id, subtopic=q.subtopic,
               difficulty_score=q.difficulty_score, correct=True)
record_ladder_outcome(state, q.id, True)
finalize_attempt(state, UNRATED)

rows = list(attempt_log.iter_rows("e2e-user"))
attempts = [x for x in rows if x.kind == "attempt" and x.is_graded]
bkt = [x for x in rows if x.kind == "bkt_update"]
check("a graded attempt row was written", bool(attempts), f"{len(rows)} rows")
check("it carries feature_sources",
      bool(attempts) and isinstance(attempts[-1].feature_sources, dict)
      and "prereqs" in attempts[-1].feature_sources,
      repr(attempts[-1].feature_sources)[:120] if attempts else "no row")
check("encompassed attribution present for a KC'd attempt",
      bool(attempts) and "encompassed" in (attempts[-1].feature_sources or {}))
check("a bkt_update row was written beside it", bool(bkt))
check("its practiced atoms are the question's tags",
      bool(bkt) and set(bkt[-1].feature_sources["bkt_changed"]) ==
      {t["atom_id"] for t in q.atom_tags})

print("D. THE READERS ARE UNDISTURBED")
cal = attempt_log.calibration("e2e-user")
check("calibration counts only graded attempts", cal["n"] == len(attempts), cal["n"])
from app import kc_stats  # noqa: E402
out = kc_stats.kc_stats()
check("kc_stats runs over a log holding all three kinds", isinstance(out["kcs"], dict))

print(f"\n{len(FAIL)} failure(s)")
sys.exit(1 if FAIL else 0)
