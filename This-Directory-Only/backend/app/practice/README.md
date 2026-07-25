# practice

## Purpose
- Hosts every endpoint mounted under `/api/practice` — the adaptive coding-drill API used by the frontend.
- Splits the practice surface into small, single-responsibility routers + helpers so individual concerns (grading, AI calls, subtopic stats, feedback) can move independently.

## Owns
- The `APIRouter` mounted at `/api/practice` and aggregated in `__init__.py`.
- Per-feature route modules: `questions_router`, `feedback_router`, `subtopic_router`, `ai_router`.
- Practice-only support modules: `chatgpt_helpers` (OpenAI key resolution + Responses-with-fallback call), `prompts` (judge/explanation prompt builders), `grading` (run user code, decide correctness via stdout match / function tests / AI judge).

## Does NOT own
- Adaptive-difficulty math, user state persistence, or attempt records — see `app/adaptive.py`.
- Subtopic prioritization weights — see `app/prioritization.py`.
- Pydantic request/response shapes — see `app/practice_schemas.py` (kept at `app/` level so non-practice modules can reuse them).
- Sandboxed Python execution — see `app/code_runner.py`.
- The question catalog and CSV loaders — see `app/questions.py`.
- Authentication — see `app/auth.py`.

## Key Files
- `__init__.py`: builds the aggregated `router = APIRouter(prefix="/api/practice", tags=["practice"])` and `include_router`s every sub-router. `app/main.py` imports `from app.practice import router as practice_router`.
- `questions_router.py`: `/next-question`, `/submit`, `/submit-local-eval`, `/override`. Thin endpoint glue that delegates to `grading.py` and `app/adaptive.py`.
- `feedback_router.py`: `/feedback`, plus `POST` and `GET` for `/visual-debug`. Owns the in-process `_latest_visual_debug_by_user` dict.
- `subtopic_router.py`: `/subtopics` (read user stats), `/weights` (PUT custom per-subtopic weights).
- `ai_router.py`: `/run-code` (sandboxed exec), `/ai-explanation` (gpt-4o), `/ai-judge` (gpt-4o-mini). No auth on these — they reuse `chatgpt_helpers`.
- `grading.py`: `grade_submission`, `select_question_for_difficulty`, `run_and_get_expected_output`. The dispatch order is: stdout-match for `task_type == "stdout_prediction"` → function tests when `submission_mode == "function"` → AI judge fallback.
- `chatgpt_helpers.py`: `load_chatgpt_api_key` (user record → user_settings → env → settings) and `call_chatgpt` (Responses API → Chat Completions fallback, temperature 1).
- `prompts.py`: `build_ai_judge_prompt` and `build_ai_explanation_prompt`. The judge prompt is reused by both `questions_router.submit_answer` and `ai_router.ai_judge` so they cannot drift.
- `watch.py`: structural health checks — verifies every sub-router exposes a top-level `router`, that `__init__.py` mounts each one, and (when fastapi is available) that the aggregated router exposes the expected paths.

## Data & External Dependencies
- OpenAI Responses + Chat Completions APIs via `openai` SDK, called from `chatgpt_helpers.py`.
- SQLAlchemy `Session` (only `chatgpt_helpers` reads from `user_settings`).
- `app/models.User` for the authenticated principal.
- `app/practice_schemas` for every request/response model.
- `app/code_runner` for sandboxed Python execution and function-test runs.

## How It Works (Flow)
1. Frontend calls `GET /api/practice/next-question`. `questions_router.next_question` asks `prioritization.select_next_subtopic`, computes a target difficulty via `adaptive.get_target_difficulty`, picks a question with `grading.select_question_for_difficulty`, marks it served, and returns the payload.
   - Optional `?focus_subtopic=<subtopic>` pins step 1's *selection* to one subtopic — the concept graph's single-KC practice flow ("Practice ⤢"). It overrides only which subtopic is served: scoring, unlock gates, and attempt recording are untouched. A focused request also skips placement probing (see Invariants); an unknown subtopic falls back to the normal weakest-subtopic pick rather than 404ing.
2. User submits code → `POST /api/practice/submit`. `grading.grade_submission` runs user code, picks the right grading strategy, and returns `(correct, actual_output, expected_output, failed_tests)`. `record_attempt` stores a pending attempt.
3. Frontend may render an explanation in parallel by calling `POST /api/practice/ai-explanation` while the judge runs.
4. User confirms via `POST /api/practice/feedback` (or `POST /api/practice/override` to flip correctness first). `apply_feedback` finalizes the pending attempt and updates the adaptive state.
5. `POST /api/practice/visual-debug` lets the frontend stash debug payloads keyed by user; `GET` reads the latest one back.

## Invariants & Constraints
- The aggregated router MUST keep `prefix="/api/practice"` and MUST mount every sub-router. `watch.py::check_invariants` enforces both. `app/main.py` includes the router with no extra prefix.
- Each sub-router file MUST declare a top-level `router = APIRouter(...)` (no prefix at this layer — the parent owns it). `watch.py` text-greps for this.
- Judge and explanation prompts MUST come from `prompts.py`. Never inline them in a router — the prompts diverged historically and that is what motivated the split.
- `chatgpt_helpers` is the only place that resolves an OpenAI API key. Do not read `OPENAI_API_KEY` directly elsewhere.
- `_latest_visual_debug_by_user` is process-local. Treat it as best-effort debug storage, not durable state.
- The placement diagnostic outranks normal subtopic selection on `/next-question` — EXCEPT for a request carrying a valid `focus_subtopic`. Rationale: the learner opened one concept from the graph, so cross-topic probes there look broken and would never move that concept's competency bar. The diagnostic stays active and resumes on the next unfocused request; those attempts still count as evidence. Do not "simplify" this by letting the diagnostic win unconditionally, and do not disable the diagnostic outright when a focus is present.

## Extension Points
- New endpoint, existing area: add it to the matching `*_router.py` and update the endpoint list in `__init__.py`'s docstring + `EXPECTED_PATHS` in `watch.py`.
- New endpoint area entirely: create `something_router.py` that exports `router = APIRouter()`, then `from app.practice.something_router import router as something_router` in `__init__.py` and `router.include_router(something_router)`. Add the file to `SUB_ROUTERS` in `watch.py`.
- New grading strategy: extend `grading.grade_submission` rather than branching inside an endpoint.
- New AI feature reusing the OpenAI client: prefer `call_chatgpt` over instantiating `OpenAI()` directly.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Inline prompt drift between submit and ai-judge** — `RESOLVED`
  - When it happens: maintainer tweaks the AI-judge prompt for one endpoint and forgets the other still has the old copy.
  - Symptom: `/submit`'s embedded judge and `/ai-judge` give different verdicts on the same submission.
  - Root cause: the judge prompt was duplicated as two string literals in the old monolithic `practice_router.py`.
  - Prevention/fix: both call sites now import from `prompts.py`. Do not re-inline.
  - Status: `RESOLVED` (2026-04-27 split).

- **Modulario auto-overwrites unfilled templates** — `ACTIVE`
  - When it happens: editing this folder while `README.md` or `watch.py` still has the marker comment.
  - Symptom: your half-written content gets replaced by a fresh template on next analysis.
  - Root cause: Modulario re-creates marker-prefixed files on each run.
  - Prevention/fix: when filling, always Read → Edit and remove the marker on line 1 in the same pass.
  - Status: `ACTIVE`.

## Recent Changes
- 2026-07-24: `questions_router.next_question` gained the optional `focus_subtopic` query param for the concept graph's single-KC practice flow, and a focused request now bypasses the placement diagnostic. `test_lesson_gate.py` + `test_diagnostic_history.py` still ALL PASS; verified that an unfocused request still serves probes while the diagnostic is active.
- 2026-04-27: Split monolithic `app/practice_router.py` (495 LOC, RED) into the `app/practice/` package: `chatgpt_helpers`, `prompts`, `grading`, `questions_router`, `feedback_router`, `subtopic_router`, `ai_router`, plus an aggregating `__init__.py`. `main.py` import updated from `app.practice_router` to `app.practice`. All resulting files are GREEN/LIME/YELLOW.
