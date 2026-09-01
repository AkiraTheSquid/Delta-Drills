"""
Pydantic schemas for the practice / adaptive-learning endpoints.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class NextQuestionResponse(BaseModel):
    question_id: int
    question_text: str
    topic: str
    subtopic: str
    difficulty: float
    target_difficulty: float
    expected_output: str
    solution_code: str
    is_cold_start: bool = False
    subtopic_n: int = 0  # number of completed questions in this subtopic before this one
    p_current: float | None = None  # EWMA accuracy for this subtopic at question time (0–1)
    # Learner-facing concept understanding. This is the BKT/crosswalk KC
    # posterior, NOT the legacy subtopic EWMA and NOT LKT P(correct).
    # Spec honesty rule: any displayed KC posterior must carry both its
    # crosswalk tier and observed-weight coverage.
    kc_mastery: float | None = None
    kc_coverage: float | None = None
    kc_tier: str | None = None
    primary_library: str = "python"
    task_type: str = "stdout_prediction"
    expected_artifact_type: str = "stdout"
    supports_visual_output: bool = False
    function_name: str | None = None
    starter_code: str | None = None
    test_cases: list[dict] = Field(default_factory=list)
    submission_mode: str = "stdout"
    # Authored near-miss outputs shown under the prompt, as
    # [{"call": ..., "output": ..., "why": ...}]. Empty for most questions.
    wrong_examples: list[dict] = Field(default_factory=list)
    hint: str | None = None  # short nudge, revealed by the Show Hint button
    # repo-relative path to the per-question solution Colab (arena-procedural-drills/…);
    # the frontend routes it to GitHub via colabUpstreamHref for the Show Answer button.
    solution_notebook_path: str | None = None
    # repo-relative path to the per-question PROBLEM Colab (starter, no answer).
    # Present for torch questions, which route to Colab instead of the in-app
    # runner; the frontend opens this and offers the solution separately.
    problem_notebook_path: str | None = None
    # ALEKS-style placement diagnostic (diagnostic.py). When active, this
    # question is a placement probe: the frontend shows the placement badge +
    # "I don't know yet" button and skips the felt-difficulty rating.
    diagnostic_active: bool = False
    diagnostic_probe_index: int | None = None   # 1-based index of this probe
    diagnostic_budget: int | None = None        # MAX_PROBES fatigue cap
    diagnostic_area: str | None = None          # topic area being probed
    # First-encounter lesson gate: target KCs of this question the learner has
    # not been exposed to yet. Each entry: {kc, kc_title, kp_title, lesson_id,
    # lesson_title, topic}. The frontend shows the introducing KP(s) BEFORE
    # revealing the question, then POSTs /exposure. Empty = no gate.
    lesson_gate: list[dict] = Field(default_factory=list)
    # Expertise-reversal ladder (kc_graph.kc_stage). `ladder_stage` is one of
    # worked | faded | independent and `ladder_kc` is the concept it was decided
    # for. On "worked" the frontend shows that KC's solved example BEFORE this
    # question — the drill still ships in the same response so no second
    # round-trip is needed once the learner clicks through.
    # `ladder_estimate` is {n, correct, p, ci, worked_seen}: the per-KC interval
    # the stage was chosen from, exposed so the decision is auditable rather
    # than a number the learner has to take on trust.
    ladder_stage: str | None = None
    ladder_kc: str | None = None
    ladder_kc_title: str | None = None
    ladder_estimate: dict | None = None
    # Did this question ACTUALLY arrive with the support its rung promises?
    #
    # The rung is a property of the concept; the scaffold is a property of the
    # question, and they can disagree. A drill whose canonical answer is one
    # statement with no call in it (`return x * 2`) cannot be faded — there is
    # nothing to blank that is not the answer — and no KP wrote an example for
    # it either. Served on the `faded` rung, the strip said "most of the
    # solution is written — supply the rest" over a bare `def solve(x)`, and
    # the learner correctly read it as a solo problem wearing a scaffolded
    # label.
    #
    # Reported rather than fixed by moving the rung: the rung is what the
    # mastery record says and what the next promotion is measured against.
    # Changing it to match the content would mean the ladder skipped a rung
    # because of an authoring gap. The display tells the truth instead.
    ladder_support: bool = True

    # The fifth rung: this problem uses the concept alongside others already
    # taught, rather than exercising it alone. Derived at serve time from the
    # question's supporting concepts and the learner's exposure map — see
    # lessons.is_integrated — and deliberately NOT a fifth `ladder_stage`. The
    # four stage names are what every stored attempt is filed under and what
    # the promotion arithmetic reads back; a fifth would either rewrite that
    # history or invent a rung nothing can be promoted out of.
    ladder_integrated: bool = False

    # Set only when this concept's CURRENT rung had nothing left the learner
    # had not already answered, and the queue reached down a rung for something
    # unseen rather than re-serving a solved problem. Carries the rung it
    # reached to (`served_from`) and how many the spent rung held, so the strip
    # can say so instead of silently looking like a demotion. None is the
    # ordinary case. See prioritization.narrow_to_next_kc.
    ladder_gap: dict | None = None


class SubmitRequest(BaseModel):
    question_id: int
    user_code: str
    # Did the learner actually SEE a worked-example popup in front of this
    # drill? The server scheduled one (`ladder_example.show`) but the client
    # can decline to draw it (no KP page, Colab edition, diagnostic), and an
    # example nobody saw must not be stored as assistance. None = an older
    # client that does not report; the server then falls back to its schedule.
    example_shown: bool | None = None


class LocalEvalSubmitRequest(BaseModel):
    question_id: int
    correct: bool
    example_shown: bool | None = None  # see SubmitRequest
    # Whether this submit is the WHOLE submit. The Colab edition has no
    # felt-difficulty step — running the notebook's checker is the entire
    # interaction — so nothing comes back to close the attempt out and it must
    # be finalized here or never. The einops fallback posts to the same
    # endpoint mid-flow, though, and its felt-difficulty step still follows;
    # finalizing there would consume the pending attempt and make the /feedback
    # that follows 400. Defaults true because the Colab route is the one with
    # no second chance; the in-flow caller opts out explicitly.
    finalize: bool = True


class LocalEvalResponse(BaseModel):
    """What /submit-local-eval did, not just that it returned 200.

    The bug this shape exists to make visible: the endpoint used to answer
    `{"success": true}` while silently parking an attempt nothing ever
    finalized. `finalized` says whether an attempt was actually closed out, and
    the before/after pairs let a caller (or a human reading the network tab)
    see the adaptive state move. They are null whenever nothing was finalized —
    a half-filled pair would read as "it moved to nowhere".
    """
    success: bool
    finalized: bool = False
    # Whether an attempt is now parked waiting for a felt-difficulty rating.
    # The Colab edition asks for one (`finalize=false`) and needs to know that
    # there is something for the rating to land on: during a placement
    # diagnostic no attempt is created at all, and showing the three buttons
    # there would post a /feedback that 400s on an empty pending slot.
    pending: bool = False
    target_difficulty_before: float | None = None
    target_difficulty_after: float | None = None
    p_before: float | None = None
    p_after: float | None = None
    # The rung this question's concept sits on AFTER the outcome is recorded,
    # in backend vocabulary (worked|faded|partial|solo), plus the interval that
    # stage was chosen from. Null for a question the KC map does not claim.
    ladder_stage: str | None = None
    ladder_estimate: dict | None = None


class SubmitResponse(BaseModel):
    correct: bool
    actual_output: str
    expected_output: str
    solution_code: str
    failed_tests: list[dict] = Field(default_factory=list)
    ladder_estimate: dict | None = None


class FeedbackRequest(BaseModel):
    question_id: int
    feedback: Literal["not_much", "somewhat", "a_lot"]


class FeedbackResponse(BaseModel):
    success: bool
    target_difficulty_after: float
    p_after: float = 0.0   # EWMA correctness rate (0–1) after this attempt
    kc_mastery_after: float | None = None
    kc_coverage_after: float | None = None
    kc_tier: str | None = None
    # Fresh per-KC ladder evidence after this graded attempt. Frontend fills
    # current rung immediately while leaving rung label on served problem.
    ladder_estimate: dict | None = None


class OverrideAttemptRequest(BaseModel):
    question_id: int
    correct: bool = Field(default=True)


class OverrideAttemptResponse(BaseModel):
    success: bool


class SubtopicStatsResponse(BaseModel):
    subtopic: str          # full key e.g. "Numpy: Core array literacy"
    topic: str             # e.g. "Numpy" or "Einsum"
    questions_answered: int
    current_difficulty: float
    weight: float          # uniform prioritization weight (1 / total_subtopics)
    learning_rate: float   # EWMA estimate of recent performance change
    gradient: float        # delta = weight × learning_rate
    baseline: float        # running weighted-average score (0–100 scale)
    p: float               # running correctness rate (0–1)


class WeightsUpdateRequest(BaseModel):
    weights: Dict[str, float]   # { "Numpy: Core array literacy": 0.175, ... }


class SubtopicStateSnapshot(BaseModel):
    # Subset of adaptive.SubtopicState fields needed by the frontend bridge
    # (computeAtomReadiness in concept-graph/atom_readiness.js). `history[]`
    # and `served_question_ids[]` are intentionally omitted — they aren't
    # read by the bridge and bloat the response.
    subtopic: str
    n: int
    baseline: float
    p: float
    target_difficulty: float
    last_update_ts: str | None = None


class PracticeStateResponse(BaseModel):
    # Shape mirrors adaptive._save_user_state minus the heavy per-attempt
    # arrays. Frontend reassigns `adaptiveStateJson = JSON.stringify(this)`
    # so the bridge sees `state.subtopic_states[sub].baseline`.
    user_id: str
    subtopic_states: Dict[str, SubtopicStateSnapshot]
    custom_weights: Dict[str, float] = Field(default_factory=dict)
    # Per-atom BKT mastery posterior + last-update timestamps. The frontend
    # readiness/prioritization (concept-graph/atom_readiness.js) prefers these
    # over the per-subtopic EWMA bridge when an atom has a posterior.
    atom_mastery: Dict[str, float] = Field(default_factory=dict)
    atom_last_ts: Dict[str, str] = Field(default_factory=dict)
    # Self-reported experience level: None | "beginner" | "strong".
    self_reported_level: Optional[str] = None


class SelfReportRequest(BaseModel):
    # "beginner" | "default" | "strong" — "default" (or unknown) clears the
    # prior back to the standard BKT p_init.
    level: str


class SelfReportResponse(BaseModel):
    success: bool
    level: Optional[str]  # normalized stored value (None when cleared)


class DiagnosticAreaEstimate(BaseModel):
    topic: str
    theta: float    # posterior-mean ability on the 0-100 difficulty scale
    sd: float       # posterior SD (placement uncertainty)
    probes: int     # probes answered in this area


class DiagnosticStatusResponse(BaseModel):
    active: bool
    completed_at: str | None = None
    declined: bool = False
    probes_done: int = 0
    budget: int
    min_probes: int
    areas: list[DiagnosticAreaEstimate] = Field(default_factory=list)
    atoms_seeded: int | None = None   # set once finished
    can_set_prior: bool = False
    self_reported_level: str | None = None


class DiagnosticAnswerRequest(BaseModel):
    question_id: int
    # "dont_know" is the first-class no-attempt response; correct/incorrect
    # cover self-rated paths (e.g. Colab-routed items).
    result: Literal["dont_know", "correct", "incorrect"]


class CodeRunRequest(BaseModel):
    code: str


class CodeRunResponse(BaseModel):
    stdout: str
    stderr: str
    success: bool


class AIExplanationRequest(BaseModel):
    question_text: str
    solution_code: str
    user_code: str
    actual_output: str
    expected_output: str


class AIExplanationResponse(BaseModel):
    explanation: str


class AIJudgeResponse(BaseModel):
    verdict: str  # "0" = incorrect, "1" = correct


class AITutorMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AITutorRequest(BaseModel):
    """One turn of the post-answer tutor chat.

    The client is stateless: it replays the whole visible thread in
    `messages` on every turn, and the problem context rides along so the
    tutor never has to guess which drill is being discussed. `explanation`
    is the auto-generated AI Explanation already on screen — passed so the
    tutor does not repeat it back at the learner.
    """

    question_text: str
    solution_code: str
    user_code: str
    actual_output: str
    expected_output: str
    explanation: str = ""
    was_correct: Optional[bool] = None
    messages: List[AITutorMessage] = Field(default_factory=list)


class AITutorResponse(BaseModel):
    reply: str


class VisualDebugRequest(BaseModel):
    payload: dict = Field(default_factory=dict)


class VisualDebugResponse(BaseModel):
    success: bool
    latest: dict = Field(default_factory=dict)


class ExposureMarkRequest(BaseModel):
    # One question cannot target anywhere near 64 KCs. Cap payload size to
    # avoid unbounded authenticated requests while retaining batch support.
    kcs: list[str] = Field(max_length=64)


class ExposureResponse(BaseModel):
    # kc_id -> ISO-8601 UTC timestamp of first exposure
    exposed: dict[str, str] = Field(default_factory=dict)


class WorkedSeenRequest(BaseModel):
    """The learner finished reading a worked example for `kc`.

    `question_id` is optional and is NOT what the ladder records — it only says
    which question is on screen, so the response can hand back that question's
    re-staged starter. Marking the example seen promotes the concept off the
    `worked` rung, and the starter the client was given a moment ago was cut for
    the rung the learner has just left.
    """

    kc: str
    question_id: int | None = None


class KcEstimateResponse(BaseModel):
    """Where one concept stands, with no question attached.

    The concept topbar needs this on the lesson screen, which is served before
    any question exists — reading the estimate off a question object works only
    once a drill has been staged, and a lesson may teach several concepts that
    the pending question does not cover.
    """

    kc: str
    ladder_stage: str
    ladder_estimate: dict


class QuestionContextResponse(BaseModel):
    """Server-only ladder fields needed to restore an old paused question."""

    ladder_stage: str
    ladder_kc: str
    ladder_kc_title: str | None = None
    ladder_estimate: dict
    ladder_support: bool = True
    ladder_integrated: bool = False
    starter_code: str | None = None


class WorkedSeenResponse(BaseModel):
    ladder_stage: str
    ladder_estimate: dict | None = None
    # The same question, faded for the rung the learner just climbed onto.
    # None means "keep what you have" (the question's own starter is correct).
    starter_code: str | None = None
