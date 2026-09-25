#!/usr/bin/env python3
"""A clock-forced submit is a scored MISS, with a `timeout` audit row.

Run: .venv/bin/python scripts/test_timeout_submit.py   (from backend/)

The answer clock force-submits whatever is in the editor (practice/timer.js
`_forceSubmitOrAdvance`). 2026-09-11 to 2026-09-20 a timed-out wrong answer
was logged as `kind=timeout` and scored nowhere; Seth 2026-09-20: "It should
count it as wrong whenever I run out of time". Now `/submit` with
`timed_out=true` and a wrong answer writes the `timeout` row FIRST (so the
audit can tell which misses were the clock), then takes the ordinary miss
path: ladder, pending attempt, felt-difficulty rating. The response says
`timed_out=true` and `scored=true`. A RIGHT answer at the buzzer is just right.
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

WRONG = "def solve(*a, **k):\n    return None\n"

# 1. Wrong + timed out: a scored miss, with the timeout row written first.
r = client.post("/api/practice/submit",
                json={"question_id": q.id, "user_code": WRONG, "timed_out": True})
check("timed-out wrong submit is 200", r.status_code == 200, r.text[:200])
body = r.json()
check("grade is still reported", body["correct"] is False)
check("scored=true, timed_out=true on the wire",
      body.get("scored") is True and body.get("timed_out") is True, body)
state = get_user_state(uid)
check("the miss parks a pending attempt", state.pending_attempt is not None)
after_timeout = len(kc_graph.ladder_view(state, kc).get("attempts", []))
check("the miss lands on the ladder", after_timeout == len(before_ladder.get("attempts", [])) + 1)
rows = list(attempt_log.iter_rows(uid))
check("first log row is kind=timeout",
      rows and rows[0].kind == attempt_log.KIND_TIMEOUT, [r.kind for r in rows])
check("timeout row carries kc, stage, question",
      rows and rows[0].kc == kc and rows[0].question_id == q.id and rows[0].stage is not None,
      rows and (rows[0].kc, rows[0].stage))
check("timeout row itself is not evidence (the miss is)", rows and not rows[0].is_graded)
check("the felt-difficulty rating on it is accepted",
      client.post("/api/practice/feedback",
                  json={"question_id": q.id, "feedback": "not_much"}).status_code == 200)

# 2. Wrong, NOT timed out: the ordinary miss path, no timeout flag.
r = client.post("/api/practice/submit",
                json={"question_id": q.id, "user_code": WRONG, "timed_out": False})
body = r.json()
check("ordinary miss is scored", body.get("scored") is True and body.get("timed_out") is False)
state = get_user_state(uid)
check("ordinary miss parks a pending attempt", state.pending_attempt is not None)
check("ordinary miss lands on the ladder",
      len(kc_graph.ladder_view(state, kc).get("attempts", [])) == after_timeout + 1)
check("no second timeout row",
      sum(r.kind == attempt_log.KIND_TIMEOUT for r in attempt_log.iter_rows(uid)) == 1)

# 3. Older client: no field at all → default False → ordinary path.
r = client.post("/api/practice/submit", json={"question_id": q.id, "user_code": WRONG})
check("missing timed_out defaults to scored", r.json().get("scored") is True)

app.dependency_overrides.clear()
print(f"\n{len(fails)} failure(s)" if fails else "\nALL PASS")
sys.exit(1 if fails else 0)
