#!/usr/bin/env python3
"""The onboarding "which courses do you want to study?" answer (2026-09-25).

Covers: `study_courses` = None reads the Courses tab's ticks (ARENA only
when ticked or nothing is — Seth 2026-09-29, a LeetCode-only learner was
served ARENA); an answered set takes every
other course's concepts out, ARENA's main graph included; stored placement
evidence follows it; POST /study-courses and the Courses tab toggle keep the
set and the shares in step; the save/load round trip.

Run: .venv/bin/python scripts/test_study_courses.py
Exits non-zero on any failed assertion. No pytest dependency.
"""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ["USER_DATA_DIR"] = tempfile.mkdtemp(prefix="study_courses_test_")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import HTTPException  # noqa: E402

from app import adaptive, course_mix, course_registry, diagnostic, kc_graph, kc_prefs  # noqa: E402
from app.adaptive import UserPracticeState  # noqa: E402
import importlib  # noqa: E402
router = importlib.import_module("app.practice.settings_router")
from app.practice_schemas import CourseShareRequest, StudyCoursesRequest  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


REG = kc_graph._registry()
ARENA_KC = next(k for k in REG if course_registry.course_of(k) == "arena")
LC_KC = next(k for k in REG if k.startswith("leetcode."))
DD_KC = course_registry.DELTA_DRILLS_KCS[0]

print("\n--- course_of ---")
check("torch/python concept is ARENA's", course_registry.course_of(ARENA_KC) == "arena", ARENA_KC)
check("leetcode.* is LeetCode's", course_registry.course_of(LC_KC) == "leetcode")
check("deltadrills.* is Delta Drills'", course_registry.course_of(DD_KC) == "delta-drills")
check("normalize_study: registry order, unknown dropped",
      course_registry.normalize_study(["leetcode", "nope", "arena", "arena"]) == ["arena", "leetcode"])

print("\n--- never asked (study_courses None) ---")
st = UserPracticeState(user_id="study-none")
check("nothing ticked: ARENA is the default", not kc_prefs.is_disabled(st, ARENA_KC))
check("LeetCode off while its share is 0", kc_prefs.is_disabled(st, LC_KC))
st.course_shares = {"leetcode": 0.4}
check("LeetCode on once its share is on", not kc_prefs.is_disabled(st, LC_KC))
# 🔴 Seth 2026-09-29: prod user 9f4b132a — LeetCode ticked, ARENA unticked,
# never asked — was served ARENA drills (linear algebra) mid-LeetCode.
check("LeetCode ticked, ARENA unticked: ARENA concept OFF", kc_prefs.is_disabled(st, ARENA_KC))
row = course_mix.status(st, "arena")
check("status: unticked ARENA not studied", row["studied"] is False)
st.course_shares = {"leetcode": 0.4, "arena": 0.4}
check("both ticked: ARENA on", not kc_prefs.is_disabled(st, ARENA_KC))
check("both ticked: LeetCode on", not kc_prefs.is_disabled(st, LC_KC))
st.course_shares = {"arena": 0.4}
check("ARENA alone ticked: LeetCode off", kc_prefs.is_disabled(st, LC_KC))
st = UserPracticeState(user_id="study-none-default")
check("status: default ARENA shows ticked", course_mix.status(st, "arena")["studied"] is True)
course_mix.toggle(st, "leetcode", True)
check("first toggle writes down the default ARENA beside LeetCode",
      st.study_courses == ["arena", "leetcode"])
try:
    course_mix.toggle(UserPracticeState(user_id="study-none-last"), "arena", False)
    check("unticking the default ARENA with nothing else refused", False)
except ValueError:
    check("unticking the default ARENA with nothing else refused", True)

print("\n--- answered: LeetCode only ---")
st = UserPracticeState(user_id="study-lc")
st.study_courses = ["leetcode"]
st.course_shares = {"leetcode": 0.4}
check("ARENA concept OFF", kc_prefs.is_disabled(st, ARENA_KC))
check("LeetCode concept on", not kc_prefs.is_disabled(st, LC_KC))
check("Delta Drills concept off", kc_prefs.is_disabled(st, DD_KC))
kcs = diagnostic.assessed_kcs(st)
check("stored-evidence concepts all leetcode.*", kcs and all(k.startswith("leetcode.") for k in kcs), str(len(kcs)))
rows = {r["course"]: r for r in (course_mix.status(st, c) for c in course_registry.COURSE_IDS)}
check("status: ARENA not studied", rows["arena"]["studied"] is False)
check("status: LeetCode studied", rows["leetcode"]["studied"] is True)

print("\n--- router: POST /study-courses ---")
user = SimpleNamespace(id="router-user")
out = router.set_study_courses(StudyCoursesRequest(courses=["leetcode"]), user=user)
state = adaptive.get_user_state("router-user")
check("study set stored", state.study_courses == ["leetcode"])
check("LeetCode share switched on", course_mix.share(state, "leetcode") == course_mix.DEFAULT_ENABLE_SHARE)
check("ARENA share 0", course_mix.share(state, "arena") == 0.0)
check("response carries study_courses", out["study_courses"] == ["leetcode"])
try:
    router.set_study_courses(StudyCoursesRequest(courses=["nope"]), user=user)
    check("empty pick refused", False)
except HTTPException as e:
    check("empty pick refused", e.status_code == 400)
check("refused pick left the set alone", adaptive.get_user_state("router-user").study_courses == ["leetcode"])

state.course_shares["arena"] = 0.5
router.set_study_courses(StudyCoursesRequest(courses=["arena", "leetcode"]), user=user)
state = adaptive.get_user_state("router-user")
check("ARENA picked keeps its mix", course_mix.share(state, "arena") == 0.5)
check("ARENA concept back on", not kc_prefs.is_disabled(state, ARENA_KC))
router.set_study_courses(StudyCoursesRequest(courses=["leetcode"]), user=user)
check("ARENA left out drops its mix", course_mix.share(adaptive.get_user_state("router-user"), "arena") == 0.0)

router.set_study_courses(StudyCoursesRequest(courses=["arena", "leetcode"]), user=user)
state = adaptive.get_user_state("router-user")
check("re-picked ARENA (share 0) is studied", state.study_courses == ["arena", "leetcode"])
check("re-picked ARENA concept on", not kc_prefs.is_disabled(state, ARENA_KC))
row = course_mix.status(state, "arena")
check("re-picked ARENA: studied, mix still off", row["studied"] is True and row["enabled"] is False)
router.set_study_courses(StudyCoursesRequest(courses=["leetcode"]), user=user)

print("\n--- router: Courses tab toggle keeps the set in step ---")
router.set_course_share(CourseShareRequest(course="arena", enabled=True), user=user)
state = adaptive.get_user_state("router-user")
check("enabling ARENA studies it again", state.study_courses == ["arena", "leetcode"])
check("ARENA concept on after toggle", not kc_prefs.is_disabled(state, ARENA_KC))
router.set_course_share(CourseShareRequest(course="arena", enabled=False), user=user)
state = adaptive.get_user_state("router-user")
check("ARENA unticked stops studying it", state.study_courses == ["leetcode"])
check("ARENA concept off after untick", kc_prefs.is_disabled(state, ARENA_KC))
try:
    router.set_course_share(CourseShareRequest(course="leetcode", enabled=False), user=user)
    check("unticking the last course refused", False)
except HTTPException as e:
    check("unticking the last course refused", e.status_code == 400 and "at least one" in str(e.detail))
state = adaptive.get_user_state("router-user")
check("refused untick changed nothing",
      state.study_courses == ["leetcode"] and course_mix.share(state, "leetcode") > 0)
router.set_course_share(CourseShareRequest(course="arena", enabled=True), user=user)
router.set_course_share(CourseShareRequest(course="leetcode", enabled=False), user=user)
state = adaptive.get_user_state("router-user")
check("LeetCode off stops studying it", state.study_courses == ["arena"])
check("LeetCode concept off", kc_prefs.is_disabled(state, LC_KC))

print("\n--- the queue never serves an unticked course ---")
import datetime  # noqa: E402
from app import questions  # noqa: E402
from app.adaptive import AttemptRecord  # noqa: E402
from app.practice import question_pick  # noqa: E402
questions.get_all_questions()


def served_courses(st, n=30):
    """Serve-and-answer n drills (a wrong answer each, so the course mix's
    turn keeps moving); the set of courses they belonged to."""
    return set().union(*served_sequence(st, n))


def served_sequence(st, n=30):
    """The same replay, one set of courses per served drill, in order."""
    seq = []
    for i in range(n):
        picked, _ = question_pick.run_queue("", st, None, None, record=False)
        if not picked:
            break
        q = picked[1]
        seq.append({course_registry.course_of(k) for k in kc_graph.question_kcs(q.id)})
        sub = st.get_subtopic_state(q.subtopic)
        sub.served_question_ids.append(q.id)
        sub.history.append(AttemptRecord(
            question_id=q.id, subtopic=q.subtopic, difficulty_score=q.difficulty_score or 20,
            grade=0.0, correct=False,
            timestamp=(datetime.datetime(2026, 9, 29) + datetime.timedelta(minutes=i)).isoformat()))
        st.last_served_question_id = q.id
    return seq


st = UserPracticeState(user_id="her-replay")
st.course_shares = {"leetcode": 0.4}
got = served_courses(st)
check("her state (LeetCode ticked, never asked): only LeetCode served", got == {"leetcode"}, str(got))
# The backstop: an upstream path that forgets the course switch (simulated by
# blinding kc_prefs.is_disabled) still cannot get an ARENA drill out.
real = kc_prefs.is_disabled
kc_prefs.is_disabled = lambda *_a, **_k: False
try:
    st = UserPracticeState(user_id="her-replay-blind")
    st.course_shares = {"leetcode": 0.4}
    got = served_courses(st)
    check("upstream filter blinded: backstop still serves only LeetCode", got == {"leetcode"}, str(got))
finally:
    kc_prefs.is_disabled = real
st = UserPracticeState(user_id="arena-replay")
st.course_shares = {"arena": 0.4}
got = served_courses(st)
check("ARENA alone ticked: only ARENA served", got == {"arena"}, str(got))
st = UserPracticeState(user_id="default-replay")
got = served_courses(st)
check("nothing ticked: ARENA served (the default)", got == {"arena"}, str(got))

print("\n--- two courses ticked = an even split (Seth, 2026-10-01) ---")
# 🔴 Before: ARENA + LeetCode both ticked served a fresh learner 30 of 30
# LeetCode, and a learner on the 0.1 target 30 of 30 ARENA.
for label, shares, study in (
    ("both ticked, never asked", {"arena": 0.4, "leetcode": 0.4}, None),
    ("both studied, ARENA exercise mix off", {"arena": 0.0, "leetcode": 0.4}, ["arena", "leetcode"]),
):
    st = UserPracticeState(user_id="split-" + label.replace(" ", "-"))
    st.course_shares = dict(shares)
    st.study_courses = study
    seq = served_sequence(st, 20)
    flat = ["+".join(sorted(c)) for c in seq]
    check(f"{label}: 10 ARENA / 10 LeetCode",
          flat.count("arena") == 10 and flat.count("leetcode") == 10, " ".join(flat))
    check(f"{label}: they alternate", all(a != b for a, b in zip(flat, flat[1:])), " ".join(flat))
st = UserPracticeState(user_id="split-three")
st.course_shares = {"arena": 0.4, "leetcode": 0.4, "delta-drills": 0.4}
flat = ["+".join(sorted(c)) for c in served_sequence(st, 9)]
check("three ticked: each takes a turn in three",
      all(len(set(flat[i:i + 3])) == 3 for i in range(0, 9, 3)), " ".join(flat))
check("one course: no split", course_mix.course_turns(UserPracticeState(user_id="one")) == ["arena"])
# 🔴 The 0.1 target narrows ARENA only: on it, every LeetCode drill used to
# fail practice_targets.allows_question, so the split served ARENA alone.
from app import practice_targets  # noqa: E402
st = UserPracticeState(user_id="split-ray")
st.practice_target = practice_targets.RAY
check("0.1 target leaves LeetCode concepts in scope", practice_targets.includes(st, LC_KC))
check("0.1 target still narrows ARENA",
      not all(practice_targets.includes(st, k) for k in REG if course_registry.course_of(k) == "arena"))
st.course_shares = {"arena": 0.4, "leetcode": 0.4}
flat = ["+".join(sorted(c)) for c in served_sequence(st, 10)]
check("0.1 target + LeetCode ticked: LeetCode is served too", flat.count("leetcode") == 5, " ".join(flat))

print("\n--- save / load ---")
adaptive.write_state_file(state)
check("round trip keeps the set", adaptive.read_state_file("router-user").study_courses == ["arena"])
import json  # noqa: E402
f = adaptive._state_file("router-user")
raw = json.loads(f.read_text())
raw.pop("study_courses", None)
f.write_text(json.dumps(raw))
check("a save from before this field loads as never-asked", adaptive.read_state_file("router-user").study_courses is None)

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
