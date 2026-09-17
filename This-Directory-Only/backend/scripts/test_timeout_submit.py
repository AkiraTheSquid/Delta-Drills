#!/usr/bin/env python3
"""A clock-forced submit is a `timeout`, not a miss.

Run: .venv/bin/python scripts/test_timeout_submit.py   (from backend/)

The answer clock force-submits whatever is in the editor (practice/timer.js
`_forceSubmitOrAdvance`). Before 2026-09-11 that grade was recorded like any
other: 88 of Seth's 97 logged misses were 2:00-clock expiries, and they held
two concepts on the faded rung for weeks. Now `/submit` with `timed_out=true`
and a wrong answer logs `kind=timeout`, moves no ability, no ladder, parks
nothing, and tells the client `scored=false`. A RIGHT answer at the buzzer
still counts, and a placement probe's deadline is still a deadline.
"""
import copy
import os
import sys
import tempfile
import uuid
from pathlib import Path

os.environ["USER_DATA_DIR"] = tempfile.mkdtemp(prefix="timeout_submit_test_")
os.environ.setdefault("KERNEL_BACKEND", "fork")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402

from app import attempt_log, auth, code_runner, kc_graph  # noqa: E402
from app.adaptive import get_user_state  # noqa: E402
from app.main import app  # noqa: E402
from app.models import User  # noqa: E402
from app.questions import get_all_questions  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


user = User(id=uuid.uuid4(), email="timeout-test@x.com", password_hash="x")
app.dependency_overrides[auth.get_current_user] = lambda: user
client = TestClient(app)
uid = str(user.id)
# The app preloads torch at startup; TestClient never runs that hook, and
# without it every torch drill is refused before grading.
code_runner.preload_torch()

# A drill with a concept and test cases, so the ladder has something to move.
q = next(x for x in get_all_questions()
         if x.test_cases and kc_graph.question_kcs(x.id))
kc = kc_graph.question_kcs(q.id)[0]
state = get_user_state(uid)
kc_graph.note_worked_seen(state, kc)  # off the cold-start rung
before_ladder = copy.deepcopy(kc_graph.ladder_view(state, kc))
before_post = copy.deepcopy(state.kc_posteriors)

WRONG = "def solve(*a, **k):\n    return None\n"

# 1. Wrong + timed out: logged as timeout, scored nowhere.
r = client.post("/api/practice/submit",
                json={"question_id": q.id, "user_code": WRONG, "timed_out": True})
check("timed-out wrong submit is 200", r.status_code == 200, r.text[:200])
body = r.json()
check("grade is still reported", body["correct"] is False)
check("scored=false on the wire", body.get("scored") is False)
state = get_user_state(uid)
check("no pending attempt parked", state.pending_attempt is None)
check("ladder row unchanged", kc_graph.ladder_view(state, kc) == before_ladder)
check("kc posteriors unchanged", state.kc_posteriors == before_post)
rows = list(attempt_log.iter_rows(uid))
check("one log row, kind=timeout",
      len(rows) == 1 and rows[0].kind == attempt_log.KIND_TIMEOUT, [r.kind for r in rows])
check("timeout row carries kc, stage, question",
      rows and rows[0].kc == kc and rows[0].question_id == q.id and rows[0].stage is not None,
      rows and (rows[0].kc, rows[0].stage))
check("timeout row is not evidence", rows and not rows[0].is_graded)
check("feedback on it is refused (nothing pending)",
      client.post("/api/practice/feedback",
                  json={"question_id": q.id, "feedback": "not_much"}).status_code == 400)

# 2. Wrong, NOT timed out: the ordinary miss path still runs.
r = client.post("/api/practice/submit",
                json={"question_id": q.id, "user_code": WRONG, "timed_out": False})
body = r.json()
check("ordinary miss is scored", body.get("scored") is True)
state = get_user_state(uid)
check("ordinary miss parks a pending attempt", state.pending_attempt is not None)
check("ordinary miss lands on the ladder",
      len(kc_graph.ladder_view(state, kc).get("attempts", [])) == len(before_ladder.get("attempts", [])) + 1)

# 3. Older client: no field at all → default False → ordinary path.
r = client.post("/api/practice/submit", json={"question_id": q.id, "user_code": WRONG})
check("missing timed_out defaults to scored", r.json().get("scored") is True)

app.dependency_overrides.clear()
print(f"\n{len(fails)} failure(s)" if fails else "\nALL PASS")
sys.exit(1 if fails else 0)
