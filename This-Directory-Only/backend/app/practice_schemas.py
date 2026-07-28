"""
Pydantic schemas for the practice / adaptive-learning endpoints.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

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
    primary_library: str = "python"
    task_type: str = "stdout_prediction"
    expected_artifact_type: str = "stdout"
    supports_visual_output: bool = False
    function_name: str | None = None
    starter_code: str | None = None
    test_cases: list[dict] = Field(default_factory=list)
    submission_mode: str = "stdout"
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


class SubmitRequest(BaseModel):
    question_id: int
    user_code: str


class LocalEvalSubmitRequest(BaseModel):
    question_id: int
    correct: bool


class SubmitResponse(BaseModel):
    correct: bool
    actual_output: str
    expected_output: str
    solution_code: str
    failed_tests: list[dict] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    question_id: int
    feedback: Literal["not_much", "somewhat", "a_lot"]


class FeedbackResponse(BaseModel):
    success: bool
    target_difficulty_after: float
    p_after: float = 0.0   # EWMA correctness rate (0–1) after this attempt


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
