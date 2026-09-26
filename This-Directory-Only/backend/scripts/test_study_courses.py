#!/usr/bin/env python3
"""The onboarding "which courses do you want to study?" answer (2026-09-25).

Covers: `study_courses` = None keeps the old rules (ARENA always in, a
standalone course only while its share is on); an answered set takes every
other course's concepts out, ARENA's main graph included; the placement area
catalogue follows it; POST /study-courses and the Courses tab toggle keep the
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

from app import adaptive, course_mix, course_registry, kc_graph, kc_prefs, placement_scope  # noqa: E402
from app.adaptive import UserPracticeState  # noqa: E402
import importlib  # noqa: E402
router = importlib.import_module("app.practice.diagnostic_router")
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
check("ARENA concept on", not kc_prefs.is_disabled(st, ARENA_KC))
check("LeetCode off while its share is 0", kc_prefs.is_disabled(st, LC_KC))
st.course_shares = {"leetcode": 0.4}
check("LeetCode on once its share is on", not kc_prefs.is_disabled(st, LC_KC))

print("\n--- answered: LeetCode only ---")
st = UserPracticeState(user_id="study-lc")
st.study_courses = ["leetcode"]
st.course_shares = {"leetcode": 0.4}
check("ARENA concept OFF", kc_prefs.is_disabled(st, ARENA_KC))
check("LeetCode concept on", not kc_prefs.is_disabled(st, LC_KC))
check("Delta Drills concept off", kc_prefs.is_disabled(st, DD_KC))
areas = {a["key"] for a in placement_scope.area_catalog(st)}
check("placement areas = LeetCode only", areas == {"LeetCode"}, str(sorted(areas)))
kcs = placement_scope.kcs_for(st, "all", None)
check("placement concepts all leetcode.*", kcs and all(k.startswith("leetcode.") for k in kcs), str(len(kcs)))
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
check("ARENA mix off keeps ARENA studied", "arena" in state.study_courses)
router.set_course_share(CourseShareRequest(course="leetcode", enabled=False), user=user)
state = adaptive.get_user_state("router-user")
check("LeetCode off stops studying it", state.study_courses == ["arena"])
check("LeetCode concept off", kc_prefs.is_disabled(state, LC_KC))

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
