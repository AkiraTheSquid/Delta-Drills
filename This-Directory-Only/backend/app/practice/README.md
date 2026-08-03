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
- `attempt_scoring.py`: `finalize_attempt(user_state, feedback)` — the single exit a graded attempt has. Runs `apply_feedback` (history, `n`, clearing `pending_attempt`), then the per-atom BKT update, then the subtopic-mastery snapshot and the new target difficulty, **in that order**. Both `/feedback` and `/submit-local-eval` go through it; neither may keep a private copy.
- `watch.py`: structural health checks — verifies every sub-router exposes a top-level `router`, that `__init__.py` mounts each one, (when fastapi is available) that the aggregated router exposes the expected paths, and that every graded attempt reaches `finalize_attempt`. That last family parses the routers with `ast` and asserts on real `ast.Call` nodes, because the previous generation of this kind of check matched plain text and was satisfied by the prose comment sitting above the code it was meant to be checking.

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
4. User confirms via `POST /api/practice/feedback` (or `POST /api/practice/override` to flip correctness first). `attempt_scoring.finalize_attempt` closes the pending attempt out and updates the adaptive state.
   - **The Colab edition takes a different exit.** There the notebook's checker IS the submit and no felt-difficulty question is ever asked, so `POST /api/practice/submit-local-eval` finalizes on the spot under the `"unrated"` feedback level. It returns `finalized` plus `target_difficulty_before/after`, `p_before/after`, `ladder_stage` and `ladder_estimate` so the tutor rail can draw the movement it just caused. `finalize: false` opts out, and exactly one caller uses it: the einops/Pyodide fallback, which posts here mid-flow and still has its `/feedback` step to come.
5. `POST /api/practice/visual-debug` lets the frontend stash debug payloads keyed by user; `GET` reads the latest one back.

## Invariants & Constraints
- The aggregated router MUST keep `prefix="/api/practice"` and MUST mount every sub-router. `watch.py::check_invariants` enforces both. `app/main.py` includes the router with no extra prefix.
- Each sub-router file MUST declare a top-level `router = APIRouter(...)` (no prefix at this layer — the parent owns it). `watch.py` text-greps for this.
- Judge and explanation prompts MUST come from `prompts.py`. Never inline them in a router — the prompts diverged historically and that is what motivated the split.
- `chatgpt_helpers` is the only place that resolves an OpenAI API key. Do not read `OPENAI_API_KEY` directly elsewhere.
- `_latest_visual_debug_by_user` is process-local. Treat it as best-effort debug storage, not durable state.
- **Every `record_attempt` MUST be paired with a `finalize_attempt`, on some path.** `record_attempt` only parks the attempt; nothing it does is durable state a learner can see. An unpaired one is overwritten by the next submit and vanishes with no error, no log and no visible symptom other than a mastery model that never moves. `watch.py::check_attempts_are_finalized` pins both exits.
- The scoring tail lives in `attempt_scoring.py` and nowhere else. If a third exit ever appears, route it through `finalize_attempt` rather than copying the sequence — two copies means an answer is worth different amounts depending on which button recorded it.
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
- 2026-08-03: **`/submit-local-eval` never finalized the attempt it recorded.** It called `record_attempt` (which only parks the attempt in `pending_attempt`) and stopped. `/feedback` was the sole caller of the scoring tail, and the Colab edition has no felt-difficulty step to reach it — so on that route every attempt sat pending until the next submit overwrote it: `sub_state.n` never moved, no per-atom BKT posterior was ever updated, `target_difficulty` never moved, and the concept graph reported a learner who had done nothing. Only `record_ladder_outcome` was recording, which is why a concept could show "1/18" in the topbar while every other reading was blank. Fixed by lifting the scoring tail out of `feedback_router` into the new `attempt_scoring.finalize_attempt` and calling it from both exits; the Colab route finalizes under the new `"unrated"` feedback level (`app/adaptive.py`), which is deliberately absent from `FEEDBACK_ALPHA` so it carries no alpha. `LocalEvalSubmitRequest` gained `finalize: bool = True` (the einops fallback sends `false`, because its `/feedback` step still follows and would 400 on a consumed attempt), and the endpoint now returns `LocalEvalResponse` with the before/after adaptive readings and the ladder rung. Verified end to end against `TestClient`: default submit → `finalized: true`, target difficulty 25.0 → 29.0, `n=1`, `pending_attempt` cleared, `feedback == "unrated"`, `alpha is None`; `finalize: false` leaves the attempt pending and the following `/feedback` returns 200; the diagnostic branch finalizes nothing; a second finalize is a no-op. New `watch.py` checks parse both routers with `ast` and fail on the pre-fix code (confirmed against `git show HEAD`). `test_bkt_mastery` / `test_diagnostic_history` / `test_kc_ladder_report` / `test_logistic_engine` PASS; `test_lesson_gate` has one **pre-existing, unrelated** failure (`qmatrix loads all easy-topic questions` expects 416 tagged questions, gets 424 — stale constant).
- 2026-07-30: `subtopic_router.kc_lattice` now returns `ladder_stage` and `ladder_estimate` on every KC row, so the knowledge graph can show a concept's own graded record instead of only the crosswalk posterior (which is a topic proxy for 43 of the 63 KCs). `ladder_estimate` carries `n` / `correct` / `p` / `ci` / `worked_seen` / `last_ts`. Reading a stage or estimate no longer creates ladder rows: `kc_graph.ladder_view` is the read path and `ladder_row` remains the write path, which matters because this endpoint asks all 63 concepts on every load. New `scripts/test_kc_ladder_report.py` (25 checks) covers the non-mutating reads and the payload; `test_lesson_gate` / `test_bkt_mastery` / `test_diagnostic_history` / `test_logistic_engine` still ALL PASS.
- 2026-07-24: `questions_router.next_question` gained the optional `focus_subtopic` query param for the concept graph's single-KC practice flow, and a focused request now bypasses the placement diagnostic. `test_lesson_gate.py` + `test_diagnostic_history.py` still ALL PASS; verified that an unfocused request still serves probes while the diagnostic is active.
- 2026-04-27: Split monolithic `app/practice_router.py` (495 LOC, RED) into the `app/practice/` package: `chatgpt_helpers`, `prompts`, `grading`, `questions_router`, `feedback_router`, `subtopic_router`, `ai_router`, plus an aggregating `__init__.py`. `main.py` import updated from `app.practice_router` to `app.practice`. All resulting files are GREEN/LIME/YELLOW.
