#!/usr/bin/env python3
"""XP = measured learning (app/learning_xp.py, 2026-09-26).

Covers Seth's rules for the yardstick:
  * the course costs the same XP for every learner, each concept its worth
    (more for what the course builds on; 2026-09-29), and there are no levels;
  * explore probes tell the MODEL where the learner is — a correct probe is
    starting credit, not XP;
  * 5 of 7 after the lesson is more learning than 1 of 6;
  * forgetting lowers knowledge, never earned XP, and relearning earns again;
  * the date target spreads what was left at today's open over the days left;
  * the target round-trips through save/load.

Run: .venv/bin/python scripts/test_learning_xp.py
Exits non-zero on any failed assertion. No pytest dependency.
"""
import os
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

os.environ["USER_DATA_DIR"] = tempfile.mkdtemp(prefix="learning_xp_test_")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import adaptive, learning_xp as X  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


UTC = timezone.utc
SCOPE = X.scope_kcs()
CODE_KC = next(k for k in SCOPE if k.startswith("torch."))
# The thresholds below were set when every concept paid 80; since 2026-09-29
# a concept pays its worth, so they scale by CODE_KC's.
F = X.worth(CODE_KC) / 80
START = datetime(2026, 9, 1, 15, 0, tzinfo=UTC)


def state(attempts=None, probes=None, level="beginner"):
    """A learner with no attempt log: the ladder and the placement probes are
    the whole record."""
    ladder = {}
    for kc, rows in (attempts or {}).items():
        ladder[kc] = {"attempts": [
            {"ts": ts.isoformat(), "correct": ok, "stage": "partial", "question_id": 1000 + i,
             **({"probe": True} if probe else {})}
            for i, (ts, ok, probe) in enumerate(rows)]}
    diag = {"probes": [{"kc": kc, "ts": ts.isoformat(), "result": "correct" if ok else "wrong"}
                       for kc, ts, ok in (probes or [])]}
    return SimpleNamespace(user_id=None, kc_ladder=ladder, diagnostic=diag,
                           self_reported_level=level, xp_target=None)


def run(st, now):
    return X.summary(st, UTC, now=now)


def seq(day, pattern, probe=False, gap_min=6):
    """Answers on one day: pattern is a string of 1/0."""
    return [(day + timedelta(minutes=gap_min * i), c == "1", probe) for i, c in enumerate(pattern)]


print("course price")
empty = run(state(), START)
check("course = the sum of its concepts' worth", empty["course"]["total_xp"] == sum(X.worth(k) for k in SCOPE),
      f'{empty["course"]["total_xp"]} for {len(SCOPE)} concepts')
check("a foundation is worth more than a leaf",
      X.worth(min(SCOPE, key=X.worth)) < X.worth(max(SCOPE, key=X.worth)))
check("no levels", not {"level", "into", "need"} & set(empty))
strong = run(state(level="strong"), START)
check("same price for a strong learner", strong["course"]["total_xp"] == empty["course"]["total_xp"])
check("strong prior = more starting knowledge, not XP",
      strong["knowledge"] > empty["knowledge"] and strong["earned"] == 0 == empty["earned"])

print("explore vs exploit")
now = START + timedelta(hours=3)
probe_day = run(state(probes=[(CODE_KC, START + timedelta(minutes=5 * i), True) for i in range(3)]), now)
check("3 correct explore probes: little XP", probe_day["today"]["xp"] < 10 * F, f'{probe_day["today"]["xp"]} XP')
check("…but the model now thinks the learner knows more",
      probe_day["knowledge"] > empty["knowledge"] + 30 * F,
      f'{probe_day["knowledge"]} vs {empty["knowledge"]}')
learn_day = run(state({CODE_KC: seq(START, "0011111")}), now)
check("lesson + 5 of 7 after two misses: real XP", learn_day["today"]["xp"] > 40 * F, f'{learn_day["today"]["xp"]} XP')
one_of_six = run(state({CODE_KC: seq(START, "000001")}), now)
five_of_seven = run(state({CODE_KC: seq(START, "1101101")}), now)
check("1 of 6 earns less than 5 of 7", one_of_six["today"]["xp"] < five_of_seven["today"]["xp"],
      f'{one_of_six["today"]["xp"]} < {five_of_seven["today"]["xp"]}')
single = run(state({CODE_KC: seq(START, "1")}), now)
check("one correct answer is not a whole concept", single["today"]["xp"] < 60 * F, f'{single["today"]["xp"]} XP')
check("…a concept caps at its worth", learn_day["today"]["xp"] <= X.worth(CODE_KC) + 1e-6)


def logged_only(uid, rows, probe, sources=None):
    """Answers that survive ONLY in the attempt log: the 20-row ladder window
    has dropped them."""
    from app import attempt_log
    for i, (ts, ok, _p) in enumerate(rows):
        attempt_log.append(attempt_log.AttemptRow(
            ts=ts.isoformat(), kind=attempt_log.KIND_ATTEMPT, user_id=uid, kc=CODE_KC,
            question_id=2000 + i, stage="partial", correct=ok, probe=probe,
            feature_sources=sources))
    return SimpleNamespace(user_id=uid, kc_ladder={}, diagnostic={},
                           self_reported_level="beginner", xp_target=None)


evicted_probes = run(logged_only("xp-evicted-probe", seq(START, "111"), True), now)
evicted_exploit = run(logged_only("xp-evicted-exploit", seq(START, "111"), False), now)
check("a probe the ladder window dropped is still a probe (attempt-log flag)",
      evicted_probes["today"]["xp"] < 10 * F < evicted_exploit["today"]["xp"],
      f'{evicted_probes["today"]["xp"]} vs {evicted_exploit["today"]["xp"]} XP')
# Rows from before the probe flag (2026-09-19..26): the log recorded whether
# the lesson had ever been read. Never read = measured what they came with.
unread = run(logged_only("xp-old-unread", seq(START, "111"), None, {"days_since_read": None}), now)
read = run(logged_only("xp-old-read", seq(START, "111"), None, {"days_since_read": 0.2}), now)
pre_field = run(logged_only("xp-old-prefield", seq(START, "111"), None), now)
check("an unflagged old answer made before the lesson was ever read is a probe",
      unread["today"]["xp"] < 10 * F < read["today"]["xp"],
      f'{unread["today"]["xp"]} vs {read["today"]["xp"]} XP')
check("…and one older than that field (before explore probes existed) is exploit",
      abs(pre_field["today"]["xp"] - evicted_exploit["today"]["xp"]) < 1e-6)

# Two answers on one concept inside the same second: one a probe, one not.
# Joined per question, they keep their own flags.
same = START + timedelta(microseconds=100)
from app import attempt_log as _al  # noqa: E402
for qid, ts, flag in ((3001, START, True), (3002, same, False)):
    _al.append(_al.AttemptRow(ts=ts.isoformat(), kind=_al.KIND_ATTEMPT, user_id="xp-twins",
                              kc=CODE_KC, question_id=qid, stage="partial", correct=True, probe=flag))
twins = SimpleNamespace(user_id="xp-twins", kc_ladder={}, diagnostic={},
                        self_reported_level="beginner", xp_target=None)
metas = X._answer_meta(twins)
check("two answers in one second keep their own probe flags",
      X._meta_for(metas, CODE_KC, 3001, X.memory_model._to_days(START.isoformat()))[0] is True
      and X._meta_for(metas, CODE_KC, 3002, X.memory_model._to_days(same.isoformat()))[0] is False)

print("problems solved per day")
# 5 of 7 right on day one, 1 of 3 on day two: `solved` counts the right ones,
# `answers` all of them, each on its own day.
two_days = state({CODE_KC: seq(START, "1101101") + seq(START + timedelta(days=1), "010")})
by_day = {d["date"]: d for d in run(two_days, START + timedelta(days=1, hours=2))["days"]}
d1, d2 = by_day[START.date().isoformat()], by_day[(START + timedelta(days=1)).date().isoformat()]
check("solved counts only right answers, per day",
      (d1["solved"], d1["answers"], d2["solved"], d2["answers"]) == (5, 7, 1, 3),
      f'{d1["solved"]}/{d1["answers"]}, {d2["solved"]}/{d2["answers"]}')
# A right placement probe is a solved problem; a wrong one is not.
probed = run(state(probes=[(CODE_KC, START, True), (CODE_KC, START + timedelta(minutes=5), False)]),
             START + timedelta(hours=1))["days"][-1]
check("a right placement probe counts as solved",
      (probed["solved"], probed["answers"]) == (1, 2), f'{probed["solved"]}/{probed["answers"]}')
# One question tagged with two concepts is ONE problem: both ladder rows
# share its question id and time.
OTHER_KC = next(k for k in SCOPE if k != CODE_KC)
pair = state()
pair.kc_ladder = {kc: {"attempts": [{"ts": START.isoformat(), "correct": True, "stage": "partial",
                                     "question_id": 4001}]} for kc in (CODE_KC, OTHER_KC)}
paired = run(pair, START + timedelta(hours=1))["days"][-1]
check("a question tagged with two concepts is one problem solved",
      (paired["solved"], paired["answers"]) == (1, 1), f'{paired["solved"]}/{paired["answers"]}')

print("forgetting and relearning")
back = START + timedelta(days=90)
learned = {CODE_KC: seq(START, "1111")}
before = run(state(learned), START + timedelta(hours=2))
idle = run(state(learned), back)
check("a break lowers knowledge", idle["knowledge"] < before["knowledge"],
      f'{idle["knowledge"]} < {before["knowledge"]}')
check("…and never lowers earned XP", abs(idle["earned"] - before["earned"]) < 1e-6)
relearned = run(state({CODE_KC: seq(START, "1111") + seq(back, "0111")}), back + timedelta(hours=2))
check("relearning after the break earns XP", relearned["today"]["xp"] > 0, f'{relearned["today"]["xp"]} XP')
check("every day's XP is ≥ 0", all(d["xp"] >= 0 for d in relearned["days"]))
check("earned = sum of the days", abs(relearned["earned"] - sum(d["xp"] for d in relearned["days"])) < 0.5)

placed = state(probes=[(CODE_KC, START + timedelta(minutes=5 * i), True) for i in range(3)])
check("placement credit fades with time like any answer",
      run(placed, START + timedelta(days=120))["knowledge"] < probe_day["knowledge"])

print("targets")
check("date target spreads today's-open remainder",
      X.daily_target({"mode": "date", "date": "2026-09-10"}, 1000.0, date(2026, 9, 1)) == 100)
check("date target on the day itself asks for all of it",
      X.daily_target({"mode": "date", "date": "2026-09-01"}, 250.0, date(2026, 9, 1)) == 250)
check("a passed target date asks for nothing, not the whole course today",
      X.daily_target({"mode": "date", "date": "2026-08-31"}, 250.0, date(2026, 9, 1)) is None)
check("fixed daily target", X.daily_target({"mode": "daily", "daily": 90}, 5.0, date(2026, 9, 1)) == 90)
check("no target", X.daily_target(None, 5.0, date(2026, 9, 1)) is None)
st = state()
for bad in (("daily", None, 0), ("date", "2000-01-01", None), ("date", "nope", None), ("weekly", None, None)):
    try:
        X.set_target(st, bad[0], bad[1], bad[2], today=date(2026, 9, 1))
        check(f"rejects {bad}", False)
    except ValueError:
        check(f"rejects {bad}", True)
X.set_target(st, "date", "2026-12-01", today=date(2026, 9, 1))
check("date target stored", st.xp_target == {"mode": "date", "date": "2026-12-01"})

uid = "xp-roundtrip"
real = adaptive.get_user_state(uid)
real.xp_target = {"mode": "daily", "daily": 120}
adaptive.save_user_state(uid)
adaptive._user_states.pop(uid, None) if hasattr(adaptive, "_user_states") else None
reloaded = adaptive._load_user_state(uid) if hasattr(adaptive, "_load_user_state") else adaptive.get_user_state(uid)
check("target survives save/load", getattr(reloaded, "xp_target", None) == {"mode": "daily", "daily": 120})

print("the API")
import uuid  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app import auth  # noqa: E402
from app.main import app  # noqa: E402
from app.models import User  # noqa: E402

api_user = User(id=uuid.uuid4(), email="xp-api-test@x.com", password_hash="x")
app.dependency_overrides[auth.get_current_user] = lambda: api_user
client = TestClient(app)
tz = "tz_offset=300&tz_name=America/Chicago"
got = client.get(f"/api/practice/xp?{tz}")
check("GET /xp answers for a new learner", got.status_code == 200 and "level" not in got.json()
      and got.json()["target"] is None, str(got.status_code))
check("unknown zone name falls back to the offset",
      client.get("/api/practice/xp?tz_offset=0&tz_name=Not/AZone").status_code == 200)
bad = client.post(f"/api/practice/xp-target?{tz}", json={"mode": "date", "date": "2000-01-01"})
check("a passed date is a 400 with a sentence", bad.status_code == 400 and "passed" in bad.json()["detail"])
ok = client.post(f"/api/practice/xp-target?{tz}", json={"mode": "daily", "daily": 90})
check("POST /xp-target saves and answers with the new summary",
      ok.status_code == 200 and ok.json()["target"] == {"mode": "daily", "daily": 90}
      and ok.json()["today"]["target"] == 90)
adaptive._user_states.pop(str(api_user.id), None) if hasattr(adaptive, "_user_states") else None
again = adaptive._load_user_state(str(api_user.id))
check("…and it was written to disk", getattr(again, "xp_target", None) == {"mode": "daily", "daily": 90})
cleared = client.post(f"/api/practice/xp-target?{tz}", json={"mode": "none"})
check("mode none clears it", cleared.status_code == 200 and cleared.json()["target"] is None)
app.dependency_overrides.clear()

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all passed")
