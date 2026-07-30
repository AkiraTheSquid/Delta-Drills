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

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


# --- metadata loading -------------------------------------------------------
lessons._load()
# These counts moved and the constants were never updated, so both assertions
# have been failing since before this change:
#   380 -> 374  the structured-dtypes retirement removed that KC's questions,
#               and q480 (curated_additions) added one back.
#   374 -> 380  six new drills for numpy.ndarray-model, the lattice root. It
#               owned two questions against a mastery bar wanting ~7 correct
#               answers, so the queue had to recycle them and the rest of the
#               course stayed locked behind the loop.
#   380 -> 416  36 drills across the nine thinnest concepts on the learner's
#               path, so each of the first ten reaches the ~8 the four-rung
#               ladder needs before it starts recycling.
#    64 -> 63   numpy.structured-dtypes was retired with the torch conversion;
#               kc_registry.json has held 63 KCs since.
_EXPECTED_TAGGED = 416
_EXPECTED_KCS = 63
check("qmatrix loads all easy-topic questions",
      len(lessons._question_target_kcs) == _EXPECTED_TAGGED,
      f"got {len(lessons._question_target_kcs)}, expected {_EXPECTED_TAGGED}")
check("every KC has an introducing KP", len(lessons._kc_gate_info) == _EXPECTED_KCS,
      f"got {len(lessons._kc_gate_info)}, expected {_EXPECTED_KCS}")

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
qr = sys.modules["app.practice.questions_router"]
_orig_select = qr.select_next_subtopic
_orig_should_run = diagnostic.should_run
qr.select_next_subtopic = lambda st: "Numpy: Core array literacy"
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
