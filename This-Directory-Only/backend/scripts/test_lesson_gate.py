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
import os
import sys
import tempfile
import uuid
from pathlib import Path

os.environ["USER_DATA_DIR"] = tempfile.mkdtemp(prefix="lesson_gate_test_")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402

from app import auth, diagnostic, lessons  # noqa: E402
from app.main import app  # noqa: E402
from app.models import User  # noqa: E402
from app.adaptive import get_user_state  # noqa: E402
from app.prioritization import question_is_unlocked  # noqa: E402
from app.questions import get_all_questions  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


# --- metadata loading -------------------------------------------------------
lessons._load()
# 🔴 DERIVED, never a frozen count. Both of these used to be magic numbers
# (416 tagged questions, 63 KCs) and both had been failing for weeks: every
# drill Seth authors moves the first and every retirement moves the second, so
# the numbers rotted on their own and the failure said nothing about the gate.
# The SIZE floors are owned by Local_Deployed_Shared/lessons/watch.py
# (_MIN_KCS, _MIN_TAGGED_QUESTIONS) — one place, and it is the one the deploy
# gate reads. What is left here is the pair of invariants this suite is
# actually about, each stated against another file so it cannot go stale:
#   - every KC the registry declares has an introducing KP, or a learner can
#     reach a concept the gate has no lesson to send them to;
#   - every tagged question id is a real question, or a retirement left tags
#     pointing at drills that no longer exist.
_registry = lessons._read_json("kc_registry.json") or {}
_registry_kcs = {kc["id"] for kc in _registry.get("kcs", [])}
_missing_kp = sorted(_registry_kcs - set(lessons._kc_gate_info))
check("every registered KC has an introducing KP", not _missing_kp,
      f"{len(_missing_kp)} without one: {_missing_kp[:5]}")

_bank_ids = {q.id for q in get_all_questions()}
_orphan_tags = sorted(set(lessons._question_target_kcs) - _bank_ids)
check("no qmatrix tag points at a retired question", not _orphan_tags,
      f"{len(_orphan_tags)} orphaned: {_orphan_tags[:5]}")
check("qmatrix loaded the tagged bank", len(lessons._question_target_kcs) > 0,
      f"got {len(lessons._question_target_kcs)}")

gate = lessons.unexposed_target_kcs(1, {})
check("unexposed target KC gates", bool(gate) and gate[0]["kc"] == "numpy.argmin-argmax")
check("gate entry carries lesson pointers",
      bool(gate) and all(k in gate[0] for k in ("kc_title", "kp_title", "lesson_id", "lesson_title", "topic")))
check("exposed KC does not gate",
      lessons.unexposed_target_kcs(1, {"numpy.argmin-argmax": "ts"}) == [])
check("untagged question does not gate", lessons.unexposed_target_kcs(999999, {}) == [])

_duplicate_qid = -1
lessons._question_target_kcs[_duplicate_qid] = [
    "numpy.argmin-argmax", "numpy.argmin-argmax"
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
                   json={"kcs": ["numpy.argmin-argmax", "not.a.real.kc"]}).json()
check("exposure POST records known KC", "numpy.argmin-argmax" in resp["exposed"])
check("exposure POST drops unknown KC", "not.a.real.kc" not in resp["exposed"])
first_exposure = resp["exposed"]["numpy.argmin-argmax"]
resp = client.post("/api/practice/exposure",
                   json={"kcs": ["numpy.argmin-argmax"]}).json()
check("repeat exposure preserves first timestamp",
      resp["exposed"]["numpy.argmin-argmax"] == first_exposure)
resp = client.post("/api/practice/exposure", json={"kcs": ["x"] * 65})
check("exposure payload has batch cap", resp.status_code == 422,
      f"got HTTP {resp.status_code}")

# Persistence: drop the in-memory state and reload from disk.
from app import adaptive  # noqa: E402
adaptive._user_states.clear()
resp = client.get("/api/practice/exposure").json()
check("exposure survives state reload", "numpy.argmin-argmax" in resp["exposed"])

# Force the normal queue onto a numpy subtopic with the diagnostic disabled.
# (Already imported via app.main — a plain `import app.practice...` here would
# rebind the name `app` and shadow the FastAPI instance.)
# 🔴 The subtopic is DERIVED from what this learner can actually be served,
# not named. It used to be pinned to "Numpy: Core array literacy", and the
# unlock lattice locks every numpy concept behind the python prerequisites, so
# a fresh user has nothing unlocked there: /next-question 404d, the response
# had no `lesson_gate` key at all and the suite died on a KeyError two checks
# from the end. Ask the lattice which subtopic is open instead.
qr = sys.modules["app.practice.questions_router"]
_probe_state = get_user_state(str(user.id))
_open = [
    q for q in get_all_questions()
    if question_is_unlocked(_probe_state, q) and lessons.unexposed_target_kcs(q.id, {})
]
check("a fresh learner has an unlocked, gated question to be served", bool(_open),
      f"{len(_open)} unlocked and gated"
      if _open else "nothing unlocked carries a lesson gate — no first drill")
_gate_subtopic = _open[0].subtopic if _open else ""
_orig_select = qr.select_next_subtopic
_orig_should_run = diagnostic.should_run
qr.select_next_subtopic = lambda st: _gate_subtopic
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
    qr.select_next_subtopic = _orig_select
    diagnostic.should_run = _orig_should_run
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
