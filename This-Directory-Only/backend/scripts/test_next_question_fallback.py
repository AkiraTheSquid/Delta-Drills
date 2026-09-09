"""Regression suite for the /next-question fallback loop (2026-09-09).

The router used to 409 the moment ONE subtopic's pick came back dry; on Seth's
account that bricked practice for good the hour he finished his placement
(`einops.pattern-language`: no `worked` drills, its one rank-0 drill skipped).
Now a dry concept is excluded and the selection asked again; the 409 is kept
for "nothing anywhere". The loop is tested here with the selector and the pick
stubbed, so every branch is reachable without a particular bank state.

Run: .venv/bin/python scripts/test_next_question_fallback.py
Exits non-zero on any failed assertion. No pytest dependency.
"""
import os
import sys
import tempfile
import uuid
from pathlib import Path

os.environ["USER_DATA_DIR"] = tempfile.mkdtemp(prefix="nq_fallback_test_")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app import auth, content_gaps, diagnostic, kc_graph  # noqa: E402
from app.adaptive import get_user_state  # noqa: E402
from app.main import app  # noqa: E402
from app.models import User  # noqa: E402
from app.practice.question_pick import SubtopicDry  # noqa: E402
from app.questions import get_question_by_id  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


qr = sys.modules["app.practice.questions_router"]
user = User(id=uuid.uuid4(), email="nq-fallback-test@x.com", password_hash="x")
app.dependency_overrides[auth.get_current_user] = lambda: user
client = TestClient(app)
state = get_user_state(str(user.id))

# A real drill that carries a concept, so the response tail (ladder fields,
# mastery, expected output) runs for real and `ladder_gap` has a home.
_kc = next(kc for kc in kc_graph._registry() if kc_graph.questions_for_kc(kc))
_q = get_question_by_id(list(kc_graph.questions_for_kc(_kc))[0])
check("fixture drill found", _q is not None, f"{_kc} -> q{getattr(_q, 'id', None)}")

GAP_A = {"kc": "kc.a", "kc_title": "Concept A", "stage": "worked", "seen": 0, "answered": 0, "total": 3}
GAP_B = {"kc": "kc.b", "kc_title": "Concept B", "stage": "faded", "seen": 2, "answered": 2, "total": 2}

_orig = (qr.select_next_subtopic, qr.pick_for_subtopic, diagnostic.should_run, content_gaps.record)
diagnostic.should_run = lambda st: False
recorded = []
content_gaps.record = lambda uid, gap: recorded.append(gap)


def run(select_script, pick_script, focus=None):
    """`select_script`: list of subtopics to answer in order (None = nothing
    left); `pick_script`: per call, either an exception to raise or a gap dict
    (None gap = plain success) to return with the fixture drill. Records the
    kwargs each call saw."""
    calls = {"select": [], "pick": []}
    sel = iter(select_script)
    pk = iter(pick_script)

    def fake_select(st, exclude=None, exclude_kcs=None):
        calls["select"].append((set(exclude or ()), set(exclude_kcs or ())))
        return next(sel, None)

    def fake_pick(uid, st, subtopic, focus_subtopic, exclude_kcs=None):
        calls["pick"].append((subtopic, set(exclude_kcs or ())))
        step = next(pk)
        if isinstance(step, Exception):
            raise step
        return st.get_subtopic_state(_q.subtopic), _q, _kc, step

    qr.select_next_subtopic = fake_select
    qr.pick_for_subtopic = fake_pick
    recorded.clear()
    url = "/api/practice/next-question" + (f"?focus_subtopic={focus}" if focus else "")
    resp = client.get(url)
    return resp, calls


try:
    # A. The lattice's head concept is dry; the next selection serves.
    resp, calls = run(["S1", "S2"], [SubtopicDry(GAP_A), None])
    body = resp.json()
    check("A: dry head concept -> a question is served", resp.status_code == 200, str(body)[:120])
    check("A: the concept was excluded, not its subtopic",
          calls["select"][1] == (set(), {"kc.a"}), str(calls["select"]))
    check("A: the retry pick sees the excluded concept",
          calls["pick"][1] == ("S2", {"kc.a"}), str(calls["pick"]))
    gap = body.get("ladder_gap") or {}
    check("A: the served question says which concept ran dry",
          gap.get("served_from") == "other_concept" and gap.get("kc") == "kc.a"
          and gap.get("served_kc") == _kc, str(gap)[:160])

    # B. The served pick has a rung gap of its own: the FIRST gap still wins.
    resp, calls = run(["S1", "S2"], [SubtopicDry(GAP_A), GAP_B])
    gap = resp.json().get("ladder_gap") or {}
    check("B: originating gap wins, own gap kept",
          gap.get("kc") == "kc.a" and (gap.get("own_gap") or {}).get("kc") == "kc.b", str(gap)[:200])

    # C. The same concept comes back dry from the same subtopic: exclude the
    #    subtopic the second time, so the loop cannot spin.
    resp, calls = run(["S1", "S1", None], [SubtopicDry(GAP_A), SubtopicDry(GAP_A)])
    check("C: repeat-dry concept excludes the subtopic and terminates",
          resp.status_code == 409 and calls["select"][2] == ({"S1"}, {"kc.a"}), str(calls["select"]))
    check("C: the 409 names the first gap", resp.json()["detail"].get("kc") == "kc.a", str(resp.json())[:120])

    # D. A dry pick with no concept to name excludes the subtopic.
    resp, calls = run(["S1", "S2"], [SubtopicDry(None), None])
    check("D: nameless dry excludes the subtopic", calls["select"][1] == ({"S1"}, set()), str(calls["select"]))

    # E. Everything dry and no gap anywhere -> 404, not 409.
    resp, calls = run(["S1", None], [SubtopicDry(None)])
    check("E: nothing servable and no gap -> 404", resp.status_code == 404, str(resp.json())[:120])

    # F. A focused request is not retried elsewhere.
    _orig_by_sub = qr.get_questions_by_subtopic
    qr.get_questions_by_subtopic = lambda name: [_q] if name == "S1" else _orig_by_sub(name)
    try:
        resp, calls = run(["S9"], [SubtopicDry(GAP_A)], focus="S1")
    finally:
        qr.get_questions_by_subtopic = _orig_by_sub
    check("F: focused dry -> immediate 409, selector never asked",
          resp.status_code == 409 and calls["select"] == [] and calls["pick"] == [("S1", set())],
          f"{resp.status_code} {calls}")
    check("F: focused 409 carries the learner message",
          "Concept A" in resp.json()["detail"].get("message", ""), str(resp.json())[:160])
finally:
    qr.select_next_subtopic, qr.pick_for_subtopic, diagnostic.should_run, content_gaps.record = _orig
    app.dependency_overrides.clear()

print()
print(f"{len(fails)} failure(s)" if fails else "ALL PASS")
sys.exit(1 if fails else 0)
