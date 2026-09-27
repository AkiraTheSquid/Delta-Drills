"""area_survey.py — the quick "what have you done before?" survey.

WHY THIS EXISTS
---------------
Seth, 2026-09-26: drop the placement test as a separate mode, keep
explore/exploit, keep the novice/intermediate/expert question, and on entering
the Practice tab ask a few "simple and easy and fast" questions — have you
studied PyTorch, done a linear algebra course — "a quick way to get a jumpstart
on the prior for the explore/exploit algorithm. It shouldn't just do the prior
across the board."

So each answer sets the CENTRE of one concept area's prior, where there was a
neutral 0.5 for every area:

  * `kc_explore.area_known` shrinks the area's probe hit-rate toward this
    centre (AREA_PRIOR_PSEUDO probes' weight), so two or three probe answers
    overrule the survey. "A lot" of PyTorch makes torch concepts worth probing
    (a right answer skips the lesson); "never" keeps them below the cost gate
    and the lesson comes first.
  * NOT `learning_xp`: a self-report is where to look, not knowledge. XP
    keeps the level prior and moves only on answers ("a lot" of PyTorch would
    otherwise book a quarter of the course before one question).

It is a prior, not evidence: nothing here is an answer to a question.
`UserPracticeState.area_survey` = {question id: answer}; None = never asked.
A question left blank or skipped is stored as "skip" (no centre), so it is not
asked again — but a question the learner has never been OFFERED (the LeetCode
one, before they enable LeetCode) is still pending, and the Practice tab asks
just that one when it appears.
"""
from __future__ import annotations

from typing import Dict, List, Optional

# P(known) an answer stands for. "some" is the old neutral 0.5, so a learner
# who says "a little" everywhere is exactly where the app was before.
ANSWERS: Dict[str, float] = {"never": 0.15, "some": 0.5, "lot": 0.8}
SKIP = "skip"  # offered, not answered: no centre, not asked again

# (id, question, concept areas it speaks for, course it needs). The area is a
# KC id's prefix (`kc_explore.area_of`). `raytracing` and `deltadrills` have no
# question: nobody has done them before this course, and they stay neutral.
QUESTIONS: List[dict] = [
    {"id": "python", "text": "Written Python?", "areas": ["python"], "course": "arena"},
    {"id": "torch", "text": "Used PyTorch or NumPy tensors?", "areas": ["torch", "tensor"], "course": "arena"},
    {"id": "einops", "text": "Used einops or einsum?", "areas": ["einops"], "course": "arena"},
    {"id": "math", "text": "Taken a linear algebra course?", "areas": ["math"], "course": "arena"},
    {"id": "nets", "text": "Trained a neural net or a CNN?", "areas": ["cnn"], "course": "arena"},
    {"id": "leetcode", "text": "Solved LeetCode or interview problems?", "areas": ["leetcode"],
     "course": "leetcode"},
]
_BY_ID = {q["id"]: q for q in QUESTIONS}
_AREA_TO_Q = {a: q["id"] for q in QUESTIONS for a in q["areas"]}


def pending(user_state) -> List[dict]:
    """The offered questions the learner has not answered or skipped yet."""
    survey = getattr(user_state, "area_survey", None) or {}
    return [q for q in questions_for(getattr(user_state, "study_courses", None))
            if q["id"] not in survey]


def answered(user_state) -> bool:
    return not pending(user_state)


def center(user_state, area: str) -> Optional[float]:
    """The prior centre the learner's answer gives `area`, or None (not asked,
    skipped, or an area no question covers)."""
    survey = getattr(user_state, "area_survey", None)
    if not isinstance(survey, dict):
        return None
    q = _AREA_TO_Q.get(area)
    return ANSWERS.get(survey.get(q)) if q else None


def centers(user_state) -> Dict[str, float]:
    """{area: centre} for every area the learner answered."""
    return {a: c for a in _AREA_TO_Q if (c := center(user_state, a)) is not None}


def questions_for(courses: Optional[List[str]]) -> List[dict]:
    """The questions for the courses being studied (None = never asked: ARENA
    only, as `course_registry` treats it)."""
    studied = set(courses) if courses else {"arena"}
    return [q for q in QUESTIONS if q["course"] in studied]


def save(user_state, answers: Dict[str, str]) -> Dict[str, str]:
    """Store the answers to the questions on offer. A question not offered, or
    an answer not in ANSWERS, raises ValueError. Every offered question left
    out is stored as SKIP (an empty dict = skip them all); earlier answers to
    other questions are kept."""
    offered = [q["id"] for q in questions_for(getattr(user_state, "study_courses", None))]
    for qid, ans in (answers or {}).items():
        if qid not in offered:
            raise ValueError(f"Unknown question {qid!r}.")
        if ans not in ANSWERS:
            raise ValueError(f"Answer {qid!r} with never, some or lot.")
    survey = dict(getattr(user_state, "area_survey", None) or {})
    for qid in offered:
        if qid in (answers or {}):
            survey[qid] = answers[qid]
        else:
            survey.setdefault(qid, SKIP)
    user_state.area_survey = survey
    return survey
