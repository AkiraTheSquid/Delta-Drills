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
# The map is "when was this LAST read" (the revisit gate reads it that way),
# so a second reading moves the stamp forward, never back.
check("repeat exposure refreshes the timestamp",
      resp["exposed"]["torch.argmin-argmax"] > first_exposure)
resp = client.post("/api/practice/exposure", json={"kcs": ["x"] * 65})
check("exposure payload has batch cap", resp.status_code == 422,
      f"got HTTP {resp.status_code}")

# --- readiness: a page comes back when the ENGINE says the drill needs it ----
# Seth's own case, 2026-09-19: q650 (torch.dtype-astype, concept s1) read on
# 09-18 00:40, no attempt on the concept until the drill 45.5 h later, missed.
# His posterior on the KC at serve time, off Fly: ability mean -1.785, var
# 1.00, n=1 (one miss on a sibling concept's drill four days earlier). The
# gate is `lesson_readiness`: not ready for the drill as it stands, ready with
# the page re-read — both by the ladder's own bar (lower credible bound >=
# PROMOTE_P). No clock is compared to a threshold anywhere; time reaches the
# decision only through the engine's `lesson` feature fading.
from datetime import datetime, timedelta, timezone  # noqa: E402
from app import lesson_readiness as R  # noqa: E402
from app import logistic_engine as E  # noqa: E402
from app.adaptive import UserPracticeState  # noqa: E402

_rv_kc, _rv_idx = lessons._question_segment[650]
_rv_seg = lessons._kc_segments[_rv_kc][_rv_idx]
_rv_key = f"{_rv_kc}#{_rv_seg['concept_id']}"
_read = datetime(2026, 9, 18, 0, 40, tzinfo=timezone.utc)
_serve = _read + timedelta(hours=45, minutes=30)
_Q650_DIFF = 14


def _seed_prereqs(st, kc, mastery=0.79):
    """Seth's prerequisites on this KC sat near 0.79 (BKT, off Fly). The
    engine's `prereq` term is worth ~+0.26 logits at that level and about
    -0.43 at the cold prior, so a fixture without it is a different learner.
    `kc_mastery` averages the atoms a KC's QUESTIONS exercise, so those are
    what is seeded; the KC's own crosswalk atoms sit at 0.5 (a neutral
    `encompassing` term, which is what his state showed)."""
    from app.questions import get_question_by_id
    for parent in (kc_graph.registry_node(kc) or {}).get("prereqs") or ():
        atoms = {a.get("a") for a in (kc_graph.crosswalk_row(parent) or {}).get("atoms") or []}
        for qid in kc_graph.questions_for_kc(parent):
            q = get_question_by_id(qid)
            atoms.update(t["atom_id"] for t in (getattr(q, "atom_tags", None) or []))
        for atom in atoms:
            if atom:
                st.atom_mastery[atom] = mastery
                st.atom_last_ts[atom] = _serve.isoformat()
    for atom in (kc_graph.crosswalk_row(kc) or {}).get("atoms") or []:
        if atom.get("a") and atom["a"] not in st.atom_mastery:
            st.atom_mastery[atom["a"]] = 0.5
            st.atom_last_ts[atom["a"]] = _serve.isoformat()


def _learner(mean, var, n=1, exposure=None, attempts=None):
    st = UserPracticeState(user_id=f"readiness-{mean}-{var}")
    _seed_prereqs(st, _rv_kc)
    st.kc_exposure = dict(exposure if exposure is not None else {_rv_key: _read.isoformat()})
    st.kc_ladder = {_rv_kc: {"worked_seen": 2, "attempts": list(attempts or [
        {"correct": False, "stage": "partial", "question_id": 651,
         "ts": (_read - timedelta(days=2)).isoformat()},
    ])}}
    st.kc_posteriors = {_rv_kc: {"ability": {
        "mean": mean, "var": var, "n": n, "last_seen": (_read - timedelta(days=2)).isoformat(),
    }}}
    return st


# 🔴 Not his exact mean. On his real state the bounds were 0.430 / 0.557
# against 0.55 — a real decision, but a 0.007 margin that the fixture's
# BKT approximation and any content edit (a new prerequisite, a difficulty
# change) would flip. The fixture sits a little further inside the region
# so the test is about the MECHANISM, and the replay of his state is in the
# session notes for 2026-09-19.
_SETH_MEAN = -1.6
_seth = _learner(_SETH_MEAN, 1.0)
_v = R.readiness(_seth, _rv_kc, 650, difficulty_score=_Q650_DIFF, now=_serve)
_lo_now, _ = _v["now"].interval(E.LADDER_Z)
_lo_after, _ = _v["after_read"].interval(E.LADDER_Z)
check("Seth's q650: below the bar as it stands, over it with the page re-read",
      _lo_now < E.PROMOTE_P <= _lo_after,
      f"lo_now={_lo_now:.3f} lo_after={_lo_after:.3f} bar={E.PROMOTE_P}")
check("Seth's q650: the page is needed", _v["lesson_needed"] is True)
_rv = R.revisit_target_kcs(_seth, 650, difficulty_score=_Q650_DIFF, now=_serve)
check("gate entry is the concept's own page, marked as a revisit",
      len(_rv) == 1 and _rv[0]["exposure_key"] == _rv_key and _rv[0]["revisit"] is True
      and _rv[0]["segment_index"] == _rv_idx,
      f"got {[(e.get('exposure_key'), e.get('revisit')) for e in _rv]}")
check("gate entry carries the stale read_at and the bounds it was decided on",
      bool(_rv) and _rv[0]["read_at"] == _read.isoformat()
      and _rv[0]["ready_lo"] < _rv[0]["ready_bar"] <= _rv[0]["ready_lo_after_read"])

# Time enters only through the model: the same learner an hour after reading.
check("the same learner an hour after reading is not gated",
      not R.revisit_target_kcs(_seth, 650, difficulty_score=_Q650_DIFF, now=_read + timedelta(hours=1)))
# Loop safety: re-reading puts the page at full value, so the gate cannot fire
# again until it fades — whatever the posterior.
_just_read = _learner(_SETH_MEAN, 1.0, exposure={_rv_key: _serve.isoformat()})
check("re-read just now → no gate (the page is at full value)",
      not R.revisit_target_kcs(_just_read, 650, difficulty_score=_Q650_DIFF, now=_serve))
check("...nor ten minutes later",
      not R.revisit_target_kcs(_just_read, 650, difficulty_score=_Q650_DIFF,
                                now=_serve + timedelta(minutes=10)))
check("no posterior anywhere gates a page read a moment ago",
      all(not R.revisit_target_kcs(_learner(m, v, exposure={_rv_key: _serve.isoformat()}),
                                   650, difficulty_score=_Q650_DIFF, now=_serve)
          for m in (-4.0, -2.0, -1.0, 0.0, 1.0, 3.0) for v in (0.1, 0.6, 1.2)))
# Expertise reversal: a learner who has demonstrated the concept is never
# sent back to the page, however stale the read.
_strong = _learner(2.0, 0.3, n=12)
check("a strong posterior is not gated on a stale read",
      not R.revisit_target_kcs(_strong, 650, difficulty_score=_Q650_DIFF, now=_read + timedelta(days=30)))
# The page cannot lift a learner who is far below the bar: that is a missing
# prerequisite (remediation's job), not a stale page, and gating would loop.
_lost = _learner(-4.0, 0.5, n=6)
_vl = R.readiness(_lost, _rv_kc, 650, difficulty_score=_Q650_DIFF, now=_serve)
check("a learner the page would not lift over the bar is not gated",
      not _vl["ready_after_read"] and not _vl["lesson_needed"])
check("nothing read (placement-only exposure) never revisits",
      not R.revisit_target_kcs(_learner(_SETH_MEAN, 1.0, exposure={}), 650,
                               difficulty_score=_Q650_DIFF, now=_serve))
_whole = R.revisit_target_kcs(_learner(_SETH_MEAN, 1.0, exposure={_rv_kc: _read.isoformat()}), 650,
                              difficulty_score=_Q650_DIFF, now=_serve)
check("a pre-split learner holding only the KC key gets the whole-KP step",
      len(_whole) == 1 and _whole[0]["exposure_key"] == _rv_kc)
# A drill not authored under a concept re-teaches the page read most recently.
_untagged = next(q for q, kcs in lessons._question_target_kcs.items()
                 if kcs == [_rv_kc] and q not in lessons._question_segment)
_two = {f"{_rv_kc}#{lessons._kc_segments[_rv_kc][0]['concept_id']}": (_read - timedelta(days=1)).isoformat(),
        _rv_key: _read.isoformat()}
_rv_un = R.revisit_target_kcs(_learner(_SETH_MEAN, 1.0, exposure=_two), _untagged,
                              difficulty_score=_Q650_DIFF, now=_serve)
check("an unsegmented drill re-teaches the most recently read concept",
      len(_rv_un) == 1 and _rv_un[0]["exposure_key"] == _rv_key,
      f"q{_untagged} -> {[e.get('exposure_key') for e in _rv_un]}")
check("page_read_age_days is the feature's input",
      abs(lessons.page_read_age_days(650, _rv_kc, {_rv_key: _read.isoformat()}, now=_serve)
          - 45.5 / 24) < 1e-9
      and lessons.page_read_age_days(650, _rv_kc, {}, now=_serve) is None)
_seg2_key = f"{_rv_kc}#{lessons._kc_segments[_rv_kc][2]['concept_id']}"
_gate = R.lesson_gate(_seth, 651, difficulty_score=_Q650_DIFF, now=_serve)
check("lesson_gate: an unread concept is taught, not revisited",
      len(_gate) == 1 and _gate[0]["exposure_key"] == _seg2_key and not _gate[0].get("revisit"),
      f"got {[(e.get('exposure_key'), e.get('revisit')) for e in _gate]}")
_all_read = _learner(_SETH_MEAN, 1.0, exposure={
    _rv_key: _read.isoformat(),
    f"{_rv_kc}#{lessons._kc_segments[_rv_kc][0]['concept_id']}": "2026-09-01T00:00:00+00:00",
    _seg2_key: "2026-09-01T00:00:00+00:00",
})
_gate = R.lesson_gate(_all_read, 650, difficulty_score=_Q650_DIFF, now=_serve)
check("lesson_gate: all read → the one the engine wants back comes as a revisit",
      len(_gate) == 1 and _gate[0].get("revisit") is True and _gate[0]["exposure_key"] == _rv_key)

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
