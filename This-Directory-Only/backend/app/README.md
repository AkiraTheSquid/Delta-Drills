# app

## Purpose
- The FastAPI backend for Delta Drills: PDF chapter-splitting jobs and the adaptive coding-practice API.
- This folder is the deployable backend service. `main.py` is the ASGI entry point (`uvicorn app.main:app`).

## Owns
- Application wiring: the `FastAPI` instance, CORS, and router mounting (`main.py`). `main.py` is intentionally thin — endpoint logic lives in `*_router.py` modules.
- Authentication endpoints: `auth_router.py` (`/auth/signup`, `/auth/login`).
- Job endpoints + background worker: `jobs_router.py` (`/jobs` POST/GET, `/jobs/{id}/artifacts`, `process_job`).
- Chapter endpoints: `chapters_router.py` (`/jobs/{id}/chapters`, `/jobs/{id}/chapters/{cid}/download`).
- Authentication primitives: password hashing, JWT issuance, current-user dependency (`auth.py`).
- Persistence: the SQLAlchemy session/engine (`db.py`), ORM models (`models.py`), and Pydantic schemas (`schemas.py`).
- Background processing: PDF auto-TOC and chapter splitting (`processing.py`), artifact capture (`job_artifacts.py`), filesystem layout (`storage.py`).
- Practice domain primitives that aren't router code: adaptive difficulty math (`adaptive.py`), subtopic prioritization (`prioritization.py`), the question catalog (`questions.py`), the sandboxed Python runner (`code_runner.py`), and the practice request/response schemas (`practice_schemas.py`).
- Graph-backed curriculum primitives: explicit concept nodes, prerequisite edges, lesson/problem mappings, and mastery-gated unlocking (`concept_graph.py` + `data/concept_graphs/*.json`).
- The aggregated practice router lives in the `practice/` subpackage.

## Does NOT own
- Practice route handlers, prompt strings, OpenAI call logic, grading dispatch — see `practice/` and its README.
- Frontend code, CSS, or static assets — see `Local_Deployed_Shared/` at the repo root.
- Question CSVs and other data files — sourced from `Local_Deployed_Shared/` at runtime via `questions.py`.
- Deployment scripts and Fly.io config — see `This-Directory-Only/backend/scripts/` and `fly.toml`.

## Key Files
- `main.py`: thin assembly layer. Builds the `FastAPI` app, configures CORS, mounts every `*_router.py` module + the `practice/` package, and exposes `/health`. No endpoint logic lives here.
- `auth_router.py`: `/auth/signup`, `/auth/login`. Owns the IntegrityError-on-duplicate flow.
- `jobs_router.py`: `/jobs` (POST create), `/jobs/{job_id}` (GET), `/jobs/{job_id}/artifacts` (GET). Owns `process_job` — the BackgroundTask worker that runs auto-TOC + `split_chapters`.
- `chapters_router.py`: `/jobs/{job_id}/chapters` and `/jobs/{job_id}/chapters/{chapter_id}/download`.
- `auth.py`: bcrypt password hashing, JWT encode/decode, `get_current_user` FastAPI dependency.
- `db.py`: `SessionLocal`, `engine`, `get_db` dependency. Reads `DATABASE_URL` from `config.py`.
- `models.py`: SQLAlchemy ORM — `User`, `Job`, `Chapter`, `JobArtifact`.
- `schemas.py`: Pydantic shapes for the non-practice endpoints (jobs, chapters, auth tokens).
- `practice_schemas.py`: Pydantic shapes for the practice endpoints. Kept here (not under `practice/`) so any non-practice consumer can import them without a circular dep through the router package.
- `concept_graph.py`: Pydantic schema + JSON loader for the first explicit curriculum graph layer. This is the bridge between hand-seeded concept structure and later graph-aware scheduling.
- `../scripts/export_concept_graph_yed.py`: exports a concept-graph JSON file to GraphML so you can inspect the graph visually in yEd before wiring it into runtime sequencing.
- `adaptive.py`: per-user state, target-difficulty curve, attempt recording, feedback application.
- `prioritization.py`: subtopic-selection weights and gradient-based prioritization.
- `questions.py`: loads the question catalog (CSV-backed) and exposes `get_question_by_id`, `get_questions_by_subtopic`.
- `question_derivation.py`: the pure text/code inference helpers `questions.py` uses while parsing CSVs — difficulty classification, primary-library and task-type inference, fixture/test-case derivation. Stateless: no module globals, no filesystem, no question store. Re-imported by `questions.py`, so callers keep importing these names from `app.questions`.
- `code_runner.py`: subprocess sandbox with a 5s timeout used by the practice grading flow.
- `job_artifacts.py` + `storage.py` + `processing.py`: the PDF-job pipeline.
- `practice/`: see `practice/README.md` — every `/api/practice/*` endpoint lives there.
- `config.py`: settings object (env-driven), incl. `openai_api_key`.

## Data & External Dependencies
- Postgres via SQLAlchemy (`db.py`).
- OpenAI API (used only inside `practice/chatgpt_helpers.py`).
- Local filesystem job tree under `user_data/` (see `storage.py`).
- `Local_Deployed_Shared/` for question CSVs and shared assets.

## How It Works (Flow)
1. `uvicorn app.main:app` boots `main.py`, which constructs `FastAPI`, configures CORS, and mounts four routers: `practice_router`, `auth_router`, `jobs_router`, `chapters_router`.
2. Auth requests hit `/auth/*` → `auth_router.py` issues JWTs via `auth.create_access_token`.
3. Job creation hits `POST /jobs` → `jobs_router.create_job` writes the `Job` row, copies the upload, and queues `jobs_router.process_job` as a `BackgroundTask`. The worker runs `processing.run_auto_toc` / `split_chapters`, captures `JobArtifact` rows via `job_artifacts.capture_auto_toc_artifacts`, and writes `Chapter` rows.
4. Chapter listing/download hits `/jobs/{id}/chapters/*` → `chapters_router.py` reads `Chapter` rows and streams files via `FileResponse`.
5. Practice requests hit `/api/practice/*` → forwarded into the `practice/` package (see its README for the full flow).

## Invariants & Constraints
- `main.py` is an assembly layer, not an endpoint module. It must stay under ~60 LOC and contain no endpoint bodies. `watch.py::check_invariants` enforces the LOC bound. Add new endpoints to the matching `*_router.py` (or create one).
- Each top-level router file MUST declare `router = APIRouter(...)`. `main.py` MUST mount every one via `app.include_router(<module>)`. `watch.py` text-greps both.
- The practice router import path is `from app.practice import router as practice_router`. Do not reintroduce a top-level `practice_router.py` — it was split for a reason (the file went RED at 495 LOC, score 675).
- `practice_schemas.py` lives at this level on purpose; do not move it under `practice/`. Other modules (e.g. tests, scripts) may import its types without paying for the practice routers.
- `code_runner.run_code` enforces a 5-second timeout. Do not call user code in-process from any endpoint.
- API keys come from `practice/chatgpt_helpers.load_chatgpt_api_key`. Do not read `OPENAI_API_KEY` from `os.environ` elsewhere.
- **The ITS serves only questions tagged to a lesson-graph KC** (`lessons.kc_only_serving()`, env `DELTA_KC_ONLY`, **default ON**). The lesson graph is being validated chapter by chapter; a question whose KC does not exist has no prerequisites, difficulty ordering or mastery target to be scheduled against. Today: 380 servable (Numpy/Einops/Einsum), 75 parked (CNN/PyTorch/Autograd/Optimizers). The filter lives in `questions.load_questions()` so every consumer inherits it — **do not add a second gate at a call site**, and do not "fix" a missing `DELTA_KC_ONLY` by defaulting it off. Full rationale: `docs/decision-kc-only-serving.md`.
- **`_questions_by_id` is complete; the serving pools are not.** `get_question_by_id` must keep resolving all 455 questions so history, `served_question_ids` and in-flight client questions still work. Only `_questions` / `_questions_by_subtopic` / `_subtopics` are narrowed. Do not "unify" them.
- `prioritization.select_next_subtopic` is the single source of truth for which subtopic to serve next. Do not introduce override hooks (the previous staleness-review override was deleted on 2026-04-27 because it caused localhost behavior to diverge from the Statistics page).

## Extension Points
- New non-practice endpoint, existing area: add it to the matching `*_router.py` (`auth_router`, `jobs_router`, `chapters_router`). Then update `EXPECTED_PATHS` in `watch.py::check_public_api`.
- New non-practice endpoint area: create `something_router.py` exporting `router = APIRouter(prefix="/something", tags=["something"])`, import + `app.include_router(...)` it in `main.py`, and add `'something_router'` to the iteration in `watch.py::check_invariants` + the import list in `check_imports`.
- New practice endpoint: extend the `practice/` package — see `practice/README.md`.
- New ORM model: define in `models.py`, add a Pydantic shape in `schemas.py`, run a migration via the project's existing scheme (see `schema.sql`).
- New background-job stage: extend `processing.py` and use `job_artifacts.capture_*` helpers to surface results. The dispatcher `process_job` lives in `jobs_router.py`.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Practice router monolith** — `RESOLVED`
  - When it happens: many practice endpoints live in one file, prompt strings inline, helpers sprinkled around.
  - Symptom: `practice_router.py` reached 495 LOC / 9 DEPS / RED, with judge prompts duplicated between `submit` and `ai-judge`.
  - Root cause: organic growth — every new endpoint was appended to the same module.
  - Prevention/fix: split into `app/practice/` with one router per feature group, plus shared `chatgpt_helpers`, `prompts`, and `grading` modules. New endpoints go in the matching sub-router or get a new sub-router file.
  - Status: `RESOLVED` (2026-04-27).

- **`main.py` wiring sprawl** — `RESOLVED`
  - When it happens: every new non-practice endpoint gets appended to `main.py` instead of a dedicated router.
  - Symptom: `main.py` reached 243 LOC / 8 DEPS / ORANGE (score 403) — auth + jobs + chapters + the background worker all sharing the same file with the FastAPI assembly.
  - Root cause: the entry-point file is the path of least resistance for "just add an endpoint".
  - Prevention/fix: split into `auth_router.py`, `jobs_router.py`, `chapters_router.py`. `main.py` is now a thin assembler (≤60 LOC enforced by `watch.py::check_invariants`). New endpoints must go in a `*_router.py` module.
  - Status: `RESOLVED` (2026-04-27).

- **Cycle via `TEMP_staleness_review_REMOVE_LATER/`** — `RESOLVED`
  - When it happens: previously, every `mod query` flagged `[CYCLE] prioritization.py ↔ TEMP_staleness_review_REMOVE_LATER/staleness.py`.
  - Root cause: leftover scaffolding from the disabled staleness-review feature.
  - Prevention/fix: folder deleted along with the `ENABLE_STALENESS_REVIEW` flag and conditional import in `prioritization.py`. If the feature returns, build it inside `prioritization.py` rather than as an external override module.
  - Status: `RESOLVED` (2026-04-27).

## Recent Changes
- 2026-07-30: **The ladder no longer demotes anyone back onto the lesson page.** `kc_graph._stage_from` could return `worked` from two demotion paths — stepping down from a miss recorded at `faded`, and the confidently-struggling branch (Wilson upper under `DEMOTE_HI`). `worked` is not a drill: it is the rung at which `LessonGate` takes the whole screen, so either path replayed a lesson the learner had already read, and because the stage is re-derived from `attempts[-1]` on every request it repeated before every subsequent question on that KC until they answered one correctly. Both paths now floor at `faded`, where the worked example stays on screen beside the problem. `worked` is therefore reachable only from `worked_seen == 0` and now means first contact and nothing else; `_step_down` takes a required `floor` so no future caller can land someone on the lesson page by omission. `scripts/test_kc_ladder_report.py` covers both paths plus a property check over all four stages (32 checks).
- 2026-07-27: **KC-only serving.** `lessons.py` gained `has_target_kcs()` and `kc_only_serving()` (env `DELTA_KC_ONLY`, default ON); `questions.load_questions()` now splits the store into a complete by-id map (455) and servable-only pools (380 across 12 subtopics), parking the 75 CNN/PyTorch/Autograd/Optimizer questions until their chapter of the lesson graph is authored and validated. `lessons/` is now the source of truth; the ARENA concept graph is demoted but retained (it is the diff target for blind-authoring later chapters). Filtering at load means `select_next_subtopic`, the `questions_router` candidate pool and `diagnostic.py`'s probe selection all inherit it; `questions_router.py` (ORANGE) was deliberately not touched. See `docs/decision-kc-only-serving.md`.
- 2026-07-24: Split `questions.py` (740 LOC, over the 700 ceiling) — moved the stateless derivation helpers into `question_derivation.py`, leaving 570 LOC. `compose_full_solution` / `wrap_answer_as_function` deliberately STAYED in `questions.py` because `watch.py::check_public_api` text-greps for those two defs there. Verified behaviour-neutral: all 455 questions produce an identical fingerprint over `(subtopic, difficulty_label, primary_library, task_type, submission_mode, starter_code, test_cases)` before and after.
- 2026-07-24: `GET /api/practice/next-question` gained an optional `focus_subtopic` query param (single-KC practice from the concept graph). See `practice/README.md`.
- 2026-04-27: Split `main.py` (243 LOC, ORANGE) into `auth_router.py`, `jobs_router.py`, `chapters_router.py`. `main.py` is now a 24-LOC assembler (LIME). `watch.py` extended to assert each router exists, declares `router = APIRouter(...)`, is mounted by `main.py`, and that `main.py` stays ≤60 LOC.
- 2026-04-27: Split monolithic `practice_router.py` into the `app/practice/` package; updated `main.py` to import from `app.practice`.
- 2026-04-27: Deleted `TEMP_staleness_review_REMOVE_LATER/` and its hooks in `prioritization.py`, resolving the long-standing import cycle.
