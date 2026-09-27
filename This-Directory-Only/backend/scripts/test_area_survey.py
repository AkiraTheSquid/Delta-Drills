#!/usr/bin/env python3
"""The "what have you done before?" survey (app/area_survey.py, 2026-09-26).

  * each answer sets ONE area's prior centre; unanswered areas stay neutral;
  * the centre moves the explore cost gate (kc_explore.area_known) and a few
    probe answers still overrule it;
  * it is not knowledge: XP reads only answers (learning_xp keeps the level prior);
  * the API stores answers, a skip, and the level only while it is open.

Run: .venv/bin/python scripts/test_area_survey.py
"""
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

os.environ["USER_DATA_DIR"] = tempfile.mkdtemp(prefix="area_survey_test_")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import adaptive, area_survey as S, kc_explore as X, learning_xp as XP  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


def learner(survey=None, level=None):
    return SimpleNamespace(user_id=None, kc_ladder={}, diagnostic={}, self_reported_level=level,
                           xp_target=None, area_survey=survey, study_courses=None)


print("centres")
check("not asked → no centre", S.center(learner(), "torch") is None and not S.answered(learner()))
sk = learner()
S.save(sk, {})
check("skipped → answered, no centre", S.answered(sk) and S.center(sk, "torch") is None)
st = learner({"torch": "lot", "math": "never"})
check("one answer covers its areas", S.center(st, "torch") == 0.8 and S.center(st, "tensor") == 0.8)
check("…and only them", S.center(st, "einops") is None and S.center(st, "raytracing") is None)
check("'some' is the old neutral", S.ANSWERS["some"] == 0.5)
try:
    S.save(learner(), {"torch": "maybe"})
    check("rejects an unknown answer", False)
except ValueError:
    check("rejects an unknown answer", True)
try:
    S.save(learner(), {"rust": "lot"})
    check("rejects an unknown question", False)
except ValueError:
    check("rejects an unknown question", True)
try:
    S.save(learner(), {"leetcode": "lot"})
    check("rejects a question not on offer (LeetCode, ARENA only)", False)
except ValueError:
    check("rejects a question not on offer (LeetCode, ARENA only)", True)
part = learner()
S.save(part, {"torch": "lot"})
check("blank questions are stored as skipped: nothing pending, no centre",
      S.answered(part) and part.area_survey["math"] == S.SKIP and S.center(part, "math") is None)
part.study_courses = ["arena", "leetcode"]
check("a course enabled later asks ONLY its own question",
      [q["id"] for q in S.pending(part)] == ["leetcode"] and not S.answered(part))
S.save(part, {"leetcode": "some"})
check("…and keeps the earlier answers", S.answered(part) and part.area_survey["torch"] == "lot"
      and part.area_survey["leetcode"] == "some")
check("LeetCode question only when studying LeetCode",
      "leetcode" not in [q["id"] for q in S.questions_for(None)]
      and "leetcode" in [q["id"] for q in S.questions_for(["arena", "leetcode"])])

print("explore cost gate")
c = S.centers(st)
check("no survey = the old neutral prior", abs(X.area_known("torch.x", {}) - 0.5) < 1e-9)
check("'a lot' of PyTorch raises the torch prior", X.area_known("torch.x", {}, c) > 0.7)
check("…and makes a cold torch concept worth a probe", X.worth_probing("torch.x", 0.0, {}, c))
check("'never' linear algebra keeps math below the cost gate",
      not X.worth_probing("math.x", -1.0, {}, c) and X.worth_probing("math.x", -1.0, {}))
check("three missed torch probes overrule 'a lot'",
      X.area_known("torch.x", {"torch": (3, 0)}, c) < 0.5)

print("XP")
now = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
plain = XP.summary(learner(level="beginner"), timezone.utc, now=now)
keen = XP.summary(learner({"torch": "lot"}, level="beginner"), timezone.utc, now=now)
check("a survey answer is not knowledge: XP and knowledge unchanged until answers measure it",
      keen["knowledge"] == plain["knowledge"] and keen["earned"] == 0 == plain["earned"])

print("the API")
from fastapi.testclient import TestClient  # noqa: E402

from app import auth  # noqa: E402
from app.main import app  # noqa: E402
from app.models import User  # noqa: E402

user = User(id=uuid.uuid4(), email="survey-test@x.com", password_hash="x")
app.dependency_overrides[auth.get_current_user] = lambda: user
client = TestClient(app)
got = client.get("/api/practice/survey").json()
check("new learner: not answered, level open", not got["answered"] and got["level_open"]
      and [q["id"] for q in got["questions"]] == ["python", "torch", "einops", "math", "nets"])
check("bad level is a 400", client.post("/api/practice/survey", json={"level": "god"}).status_code == 400)
check("bad answer is a 400",
      client.post("/api/practice/survey", json={"answers": {"torch": "x"}}).status_code == 400)
saved = client.post("/api/practice/survey", json={"answers": {"torch": "lot"}, "level": "expert"}).json()
check("answers + level saved", saved["answered"] and saved["answers"]["torch"] == "lot"
      and saved["answers"]["python"] == "skip" and saved["level"] == "expert")
check("…to disk", adaptive._load_user_state(str(user.id)).area_survey["torch"] == "lot"
      and adaptive._load_user_state(str(user.id)).self_reported_level == "strong")
skip = client.post("/api/practice/survey", json={}).json()
check("a skip is stored as answered, earlier answers kept", skip["answered"] and not skip["questions"]
      and skip["answers"]["torch"] == "lot" and skip["answers"]["math"] == "skip")
state = adaptive.get_user_state(str(user.id))
state.diagnostic = {"probes": [{"kc": "torch.x"}], "completed_at": None}
locked = client.post("/api/practice/survey", json={"answers": {}, "level": "novice"}).json()
check("level is not changed once evidence exists", not locked["level_open"] and locked["level"] == "expert")
app.dependency_overrides.clear()

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all passed")
