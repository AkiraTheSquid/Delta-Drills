"""The courses a learner can enable: which KCs each one's own share
(course_mix.py) treats as its pool. New courses are added here only —
course_mix.py, the Courses tab and the KG view all read this list, never a
hardcoded course id.

`milestones()` is the STATIC pool declaration; course_mix.milestones() layers
the practice_target scope narrowing (raytracing-0.1) on top for "arena" only,
the one place that still needs it.
"""
from __future__ import annotations

from typing import List

# The 3-KC Delta-Drills placeholder chain (docs/spec-multi-course-catalog.md).
# mastery -> spaced-repetition -> why-drills, added to lessons/kc_registry.json
# under lessons/deltadrills/.
DELTA_DRILLS_KCS = [
    "deltadrills.mastery",
    "deltadrills.spaced-repetition",
    "deltadrills.why-drills",
]

# `standalone`: the course's concepts sit OUTSIDE the main prerequisite graph,
# so its enable toggle is the only way in — disabled, they are off exactly as a
# learner-disabled concept is (kc_prefs.is_disabled). ARENA is the main graph
# itself: disabling it only drops its share, the graph still leads to it.
COURSES = (
    {"id": "arena", "label": "ARENA", "standalone": False},
    {"id": "delta-drills", "label": "Delta Drills", "standalone": True},
    {"id": "leetcode", "label": "LeetCode Patterns", "standalone": True},
)
COURSE_IDS = tuple(c["id"] for c in COURSES)
_STANDALONE_KCS = {kc: "delta-drills" for kc in DELTA_DRILLS_KCS}

# LeetCode Patterns owns every `leetcode.*` KC in lessons/kc_registry.json —
# the rows scripts/leetcode_course/export_app.py writes (a concept with no
# drills is left out there). A prefix, not a list, so this file never goes
# stale when the exporter adds a pattern; concept-graph/course-registry.js
# reads the same prefix.
LEETCODE_PREFIX = "leetcode."


def _leetcode_kcs() -> List[str]:
    from app import kc_graph
    return [kc for kc in kc_graph._registry() if kc.startswith(LEETCODE_PREFIX)]


def milestones(course_id: str) -> List[str]:
    """The KCs this course's own exercises are tagged to, unscoped. ARENA's
    come from the exercise map read at diagnostic.py; delta-drills is the
    fixed chain above. An unknown id has no pool (share is inert)."""
    if course_id == "arena":
        from app import diagnostic
        return list(diagnostic._arena_links().keys())
    if course_id == "delta-drills":
        return list(DELTA_DRILLS_KCS)
    if course_id == "leetcode":
        return _leetcode_kcs()
    return []


def course_off(user_state, kc: str) -> bool:
    """True for a concept of a STANDALONE course the learner hasn't enabled.
    Read through kc_prefs.is_disabled, so every path that already respects a
    disabled concept — frontier, drill servability, explore probes, reviews,
    placement — respects the course toggle too, with no second rule."""
    course_id = _STANDALONE_KCS.get(kc)
    if course_id is None and kc.startswith(LEETCODE_PREFIX):
        course_id = "leetcode"
    if course_id is None:
        return False
    from app import course_mix
    return course_mix.share(user_state, course_id) <= 0.0
