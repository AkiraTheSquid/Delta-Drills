#!/usr/bin/env python3
"""Validation suite for activity_router.week_counts (2026-09-01).

Covers: Monday-Sunday bucketing in the learner's timezone, the offset moving
both "which week is current" and each row's bucket (a late-evening UTC row must
land on the learner's local day), that lesson views / in-flight serves /
outside-the-week rows are not counted, and that a featureless-but-answered row
IS counted (the chart is looser than is_graded on purpose). Synthetic logs are
written through `attempt_log.AttemptRow`/`append` itself.

Run: .venv/bin/python scripts/test_activity_week.py
Exits non-zero on any failed assertion. No pytest dependency.
"""
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import attempt_log  # noqa: E402
from app.practice.activity_router import week_counts  # noqa: E402

BASE = Path(tempfile.mkdtemp(prefix="activity_week_test_"))

fails = []


def check(name, cond, detail=""):
    tag = "ok  " if cond else "FAIL"
    print(f"  {tag} {name}" + (f"  {detail}" if (detail and not cond) else ""))
    if not cond:
        fails.append(name)


def put(uid, ts, kind=attempt_log.KIND_ATTEMPT, correct=True, features=None):
    attempt_log.append(
        attempt_log.AttemptRow(
            ts=ts, kind=kind, user_id=uid, kc="numpy.ndarray-model",
            question_id=1, stage="solo",
            features={"ability": 0.0} if features is None else features,
            predicted_p=0.5, correct=correct,
        ),
        base_dir=BASE,
    )


# Frozen clock: Wednesday 2026-09-02 01:30 UTC. In UTC the week is Mon 08-31 …
# Sun 09-06. At tz_offset=+300 (UTC-5, the Americas case) local time is still
# Tuesday 09-01 evening — same local week, different "today".
NOW = datetime(2026, 9, 2, 1, 30, tzinfo=timezone.utc)

print("bucketing:")
U = "learner-a"
put(U, "2026-08-31T10:00:00Z")                      # Mon
put(U, "2026-08-31T11:00:00Z")                      # Mon again
put(U, "2026-09-01T23:10:00Z")                      # Tue (UTC)
put(U, "2026-08-30T10:00:00Z")                      # Sunday BEFORE the week
put(U, "2026-08-01T10:00:00Z")                      # long past
put(U, "2026-09-01T12:00:00Z", kind=attempt_log.KIND_LESSON_VIEW, correct=None)
put(U, "2026-09-01T12:05:00Z", correct=None)        # in-flight serve, no outcome
put(U, "2026-09-01T12:10:00Z", features={})         # answered, no features — counts

utc = week_counts(U, 0, now=NOW, base_dir=BASE)
check("week starts on Monday", utc["week_start"] == "2026-08-31")
check("seven days come back", len(utc["days"]) == 7)
check("today is the UTC Wednesday", utc["today"] == "2026-09-02")
by_date = {d["date"]: d["count"] for d in utc["days"]}
check("Monday counts both attempts", by_date["2026-08-31"] == 2)
check("Tuesday counts answered + featureless", by_date["2026-09-01"] == 2)
check("empty days are zero, not missing", by_date["2026-09-04"] == 0)
check("last week's Sunday is outside", "2026-08-30" not in by_date)
check("lesson views and serves not counted", utc["total"] == 4)

print("timezone:")
# UTC-5: the 23:10Z Tuesday row is 18:10 LOCAL Tuesday (same bucket), and the
# 01:30Z Wednesday "now" is Tuesday evening — today moves, the week does not.
am = week_counts(U, 300, now=NOW, base_dir=BASE)
check("offset keeps the same week here", am["week_start"] == "2026-08-31")
by_date_am = {d["date"]: d["count"] for d in am["days"]}
check("today is still local Tuesday", am["today"] == "2026-09-01")
# 10:00Z/11:00Z Monday → 05:00/06:00 local Monday; 12:10Z Tue → 07:10 local Tue.
check("Monday holds at UTC-5", by_date_am["2026-08-31"] == 2)
check("Tuesday holds at UTC-5", by_date_am["2026-09-01"] == 2)

# UTC+9 (tz_offset=-540): 23:10Z Tuesday → 08:10 local WEDNESDAY.
jp = week_counts(U, -540, now=NOW, base_dir=BASE)
by_date_jp = {d["date"]: d["count"] for d in jp["days"]}
check("late UTC row crosses to local Wednesday", by_date_jp["2026-09-02"] == 1)
check("Tuesday loses it at UTC+9", by_date_jp["2026-09-01"] == 1)

print("empty log:")
empty = week_counts("nobody", 0, now=NOW, base_dir=BASE)
check("no log → all zeros", empty["total"] == 0 and len(empty["days"]) == 7)

if fails:
    print(f"\n{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("\nall checks passed")
