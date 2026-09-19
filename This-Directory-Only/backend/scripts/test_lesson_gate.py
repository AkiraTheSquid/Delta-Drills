#!/usr/bin/env python3
"""Validation suite for the first-encounter exposure guard (Pass 2, 2026-07-18).

Covers: lesson-metadata loading (qmatrix + KC registry + compiled lessons),
gate computation/deduplication, the /exposure GET/POST roundtrip (including
unknown-KC rejection, monotonic timestamps, payload limits, and persistence
across a state reload), the lesson_gate field on /next-question, gate clearing
after exposure, and deterministic diagnostic-probe exclusion.

Run: .venv/bin/python scripts/test_lesson_gate.py
Exits non-zero on any failed assertion. No pytest dependency.
"""
import copy
import os
import sys
import tempfile
import uuid
from pathlib import Path

os.environ["USER_DATA_DIR"] = tempfile.mkdtemp(prefix="lesson_gate_test_")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402

from app import auth, diagnostic, kc_graph, lessons  # noqa: E402
from app.main import app  # noqa: E402
from app.models import User  # noqa: E402
from app.adaptive import get_user_state  # noqa: E402
from app.prioritization import question_is_unlocked  # noqa: E402
from app.questions import get_all_questions, get_every_question  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


# --- metadata loading -------------------------------------------------------
lessons._load()
# 🔴 DERIVED, never a frozen count. Both of these used to be magic numbers
# (416 tagged questions, 63 KCs) and both had been failing for weeks: every
_registry = lessons._read_json("kc_registry.json") or {}
_registry_kcs = {kc["id"] for kc in _registry.get("kcs", [])}
_missing_kp = sorted(_registry_kcs - set(lessons._kc_gate_info))
check("every registered KC has an introducing KP", not _missing_kp,
      f"{len(_missing_kp)} without one: {_missing_kp[:5]}")

_bank_ids = {q.id for q in get_every_question()}
_orphan_tags = sorted(set(lessons._question_target_kcs) - _bank_ids)
check("no qmatrix tag points at a retired question", not _orphan_tags,
      f"{len(_orphan_tags)} orphaned: {_orphan_tags[:5]}")
check("qmatrix loaded the tagged bank", len(lessons._question_target_kcs) > 0,
      f"got {len(lessons._question_target_kcs)}")

gate = lessons.unexposed_target_kcs(1, {})
check("unexposed target KC gates", bool(gate) and gate[0]["kc"] == "torch.argmin-argmax")
check("gate entry carries lesson pointers",
      bool(gate) and all(k in gate[0] for k in ("kc_title", "kp_title", "lesson_id", "lesson_title", "topic")))
_exposed_argmin = {
    "torch.argmin-argmax": "ts",
    **{f"torch.argmin-argmax#{seg['concept_id']}": "ts" for seg in lessons._kc_segments.get("torch.argmin-argmax", [])},
}
check("exposed KC does not gate",
      lessons.unexposed_target_kcs(1, _exposed_argmin) == [])
check("untagged question does not gate", lessons.unexposed_target_kcs(999999, {}) == [])

# 🔴 python.control-flow's page is filed under ARENA 0.0 (topic PyTorch), so
# the topic test alone parked its sixteen library-free drills and the concept
# could never be learned — every cnn.* concept behind it was locked for good
# (replay of Seth's state, 2026-09-19). A python.* concept is pre-library by
# definition.
_cf = [q for q in kc_graph.questions_for_kc("python.control-flow")]
check("python.control-flow has drills", bool(_cf))
check("a python.* drill is pre-library whatever lesson files its page",
      all(lessons.is_prelibrary(q) for q in _cf), f"parked={[q for q in _cf if not lessons.is_prelibrary(q)]}")
from app.questions import get_all_questions  # noqa: E402
_serv = {q.id for q in get_all_questions()}
check("...and so it is served", all(q in _serv for q in _cf), f"unserved={[q for q in _cf if q not in _serv]}")

_duplicate_qid = -1
lessons._question_target_kcs[_duplicate_qid] = [
    "torch.argmin-argmax", "torch.argmin-argmax"
]
try:
    check("duplicate target KC gates once",
          len(lessons.unexposed_target_kcs(_duplicate_qid, {})) == 1)
finally:
    lessons._question_target_kcs.pop(_duplicate_qid, None)

# --- API: exposure roundtrip + next-question gate ---------------------------
user = User(id=uuid.uuid4(), email="lesson-gate-test@x.com", password_hash="x")
app.dependency_overrides[auth.get_current_user] = lambda: user
client = TestClient(app)

resp = client.get("/api/practice/exposure").json()
check("fresh user has empty exposure", resp["exposed"] == {})

resp = client.post("/api/practice/exposure",
                   json={"kcs": ["torch.argmin-argmax", "not.a.real.kc"]}).json()
check("exposure POST records known KC", "torch.argmin-argmax" in resp["exposed"])
check("exposure POST drops unknown KC", "not.a.real.kc" not in resp["exposed"])
first_exposure = resp["exposed"]["torch.argmin-argmax"]
resp = client.post("/api/practice/exposure",
                   json={"kcs": ["torch.argmin-argmax"]}).json()
check("repeat exposure preserves first timestamp",
      resp["exposed"]["torch.argmin-argmax"] == first_exposure)
resp = client.post("/api/practice/exposure", json={"kcs": ["x"] * 65})
check("exposure payload has batch cap", resp.status_code == 422,
      f"got HTTP {resp.status_code}")

# Persistence: drop the in-memory state and reload from disk.
from app import adaptive  # noqa: E402
adaptive._user_states.clear()
resp = client.get("/api/practice/exposure").json()
check("exposure survives state reload", "torch.argmin-argmax" in resp["exposed"])

# Force the normal queue onto a numpy subtopic with the diagnostic disabled.
# (Already imported via app.main — a plain `import app.practice...` here would
# rebind the name `app` and shadow the FastAPI instance.)
# 🔴 The subtopic is DERIVED from what this learner can actually be served,
# not named. It used to be pinned to "Numpy: Core array literacy", and the
# unlock lattice locks every numpy concept behind the python prerequisites, so
# a fresh user has nothing unlocked there: /next-question 404d, the response
# had no `lesson_gate` key at all and the suite died on a KeyError two checks
# from the end. Ask the lattice which subtopic is open instead.
from app.practice import question_pick  # noqa: E402
from app import prioritization  # noqa: E402
_probe_state = get_user_state(str(user.id))
_open = [
    q for q in get_all_questions()
    if question_is_unlocked(_probe_state, q) and lessons.unexposed_target_kcs(q.id, {})
]
check("a fresh learner has an unlocked, gated question to be served", bool(_open),
      f"{len(_open)} unlocked and gated"
      if _open else "nothing unlocked carries a lesson gate — no first drill")
_gate_subtopic = _open[0].subtopic if _open else ""
_orig_select = question_pick.select_next_subtopic
_orig_should_run = diagnostic.should_run
question_pick.select_next_subtopic = lambda st, **kw: _gate_subtopic
prioritization.select_next_subtopic = lambda st, **kw: _gate_subtopic
diagnostic.should_run = lambda st: False
try:
    data = client.get("/api/practice/next-question").json()
    gate_kcs = [e["kc"] for e in data["lesson_gate"]]
    check("next-question carries lesson_gate for unexposed KC", bool(gate_kcs),
          f"qid {data['question_id']} -> {gate_kcs}")

    client.post("/api/practice/exposure", json={"kcs": gate_kcs})
    data2 = client.get("/api/practice/next-question").json()
    again = [e["kc"] for e in data2["lesson_gate"]]
    check("gate clears after exposure", not any(kc in again for kc in gate_kcs),
          f"qid {data2['question_id']} -> {again}")
finally:
    question_pick.select_next_subtopic = _orig_select
    prioritization.select_next_subtopic = _orig_select
    diagnostic.should_run = _orig_should_run
app.dependency_overrides.clear()

# --- API: legacy paused-question context recovery --------------------------
context_user = User(id=uuid.uuid4(), email="question-context-test@x.com", password_hash="x")
app.dependency_overrides[auth.get_current_user] = lambda: context_user
context_state = get_user_state(str(context_user.id))
kc_graph.note_worked_seen(context_state, "torch.tensor-model")
before_context_read = copy.deepcopy(vars(context_state))
resp = client.get("/api/practice/question-context?question_id=482")
context = resp.json()
check("question context endpoint answers", resp.status_code == 200,
      f"got HTTP {resp.status_code}: {context}")
# q482 is an authored faded drill. The faded rung is retired (2026-09-11), so
# a learner resuming it lands on the floor rung and gets the question's own
# starter, not the blanked one — the blanks were that rung's support.
check("question context restores concept on the floor rung",
      context.get("ladder_kc") == "torch.tensor-model"
      and context.get("ladder_stage") == "partial", context.get("ladder_stage"))
check("question context does not hand out the retired faded scaffold",
      "_____" not in (context.get("starter_code") or ""), context.get("starter_code"))
check("question context read does not mutate learner state",
      vars(context_state) == before_context_read)
resp = client.get("/api/practice/question-context?question_id=999999")
check("unknown saved question is rejected", resp.status_code == 404,
      f"got HTTP {resp.status_code}")
app.dependency_overrides.clear()

# --- diagnostic probes are never gated -------------------------------------
probe_user = User(id=uuid.uuid4(), email="lesson-gate-diag@x.com", password_hash="x")
app.dependency_overrides[auth.get_current_user] = lambda: probe_user
qr = sys.modules["app.practice.questions_router"]
from app.questions import get_question_by_id  # noqa: E402
_orig_should_run = diagnostic.should_run
_orig_should_finish = diagnostic.should_finish
_orig_select_probe = diagnostic.select_probe
diagnostic.should_run = lambda st: True
diagnostic.should_finish = lambda st: False
diagnostic.select_probe = lambda st: get_question_by_id(1)
try:
    data = client.get("/api/practice/next-question").json()
    check("diagnostic test exercised probe branch", data.get("diagnostic_active") is True)
    check("diagnostic probe has no lesson_gate", data["lesson_gate"] == [])
finally:
    diagnostic.should_run = _orig_should_run
    diagnostic.should_finish = _orig_should_finish
    diagnostic.select_probe = _orig_select_probe
app.dependency_overrides.clear()

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("ALL PASS")
