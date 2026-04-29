# practice

## Purpose
Practice-page frontend: loads ARENA-derived coding questions, runs the user's Python in-browser via Pyodide (or via a backend endpoint), grades the result, and renders image previews for visual exercises.

## Owns
- The practice-page UI lifecycle (init, question selection, code editor wiring, run/submit, result display).
- The Pyodide runtime contract and preamble (see `RUNTIME_CONTRACT.md`).
- Image rendering for visual exercises, including the static PNG fallback.
- Adaptive question selection, timing, and progress storage scoped to the practice flow.

## Does NOT own
- Question authoring or curation — lives in `../arena_prereqs_structured.json` and the ARENA source dirs.
- Backend code execution — handled by the Fly.io backend behind `/api/practice/run-code`.
- AI grading prompts — system prompts live in `../function_mode_quality_*.txt` and `../chatgpt/`.
- Auth, navigation, top-level shell — handled by the host app.

## Key Files
- `init.js`: bootstraps the practice page, wires DOM and event handlers.
- `dom.js`: caches DOM element refs used across modules.
- `events.js`: top-level event handlers (run, submit, navigation).
- `engine.js`: question-selection / submission orchestration; talks to `api.js`.
- `api.js`: wrapper around `apiFetch` for `/api/practice/*` endpoints.
- `runner.js`: in-browser Pyodide runtime, `buildPyodidePreamble()`, run-button handler. **Runtime contract documented in `RUNTIME_CONTRACT.md`.**
- `visuals.js`: target-image rendering for visual exercises; multi-path `.npy` lookup, canvas rasterization, static PNG fallback.
- `ui.js`: question rendering into the editor and prompt area.
- `ai.js`: AI-judge submission path.
- `mode.js`: practice mode (backend vs local-pyodide) selection.
- `adaptive.js`: adaptive question difficulty/selection.
- `questions.js`: client-side question caching/normalization.
- `storage.js`: progress / streak persistence.
- `timer.js`, `bars.js`, `bars.css`: per-question timer and progress bars.
- `config.js`: feature flags and tunables.
- `RUNTIME_CONTRACT.md`: declarative contract of what the Pyodide preamble injects; consumed by per-question `runtime_dependencies` arrays.

## Data & External Dependencies
- **Question schema**: `arena_prereqs_structured.json` (sibling); per-entry fields include `starter_code`, `canonical_solution`, `runtime_dependencies`, `runtime_unmet_dependencies`, `supports_visual_output`, `expected_artifact_type`.
- **ARENA source data**: `../delta_numbers.npy` (mirrored to Pyodide FS), `../numbers_stacked.png` (static fallback).
- **Pyodide** (CDN): numpy + micropip + einops loaded at runtime.
- **Backend**: `/api/practice/run-code`, `/api/practice/visual-debug`.
- **Supabase**: progress / streaks via `../supabase-practice.js`.

## How It Works (Flow)
1. `init.js` boots the page; `engine.js` fetches the next question via `api.js`.
2. `ui.js` renders prompt + starter code into the Monaco editor.
3. If the question has `supports_visual_output`, `visuals.js` runs the canonical solution in Pyodide and rasterizes the resulting array onto a canvas. On failure, falls back to `numbers_stacked.png`.
4. User clicks Run → `runner.js` either POSTs to the backend or executes locally in Pyodide using the preamble from `buildPyodidePreamble()`.
5. Submit → `ai.js` (AI-judged) or backend grader returns pass/fail; result feeds `storage.js` and adaptive selection.

## Invariants & Constraints
- The Pyodide preamble in `runner.js` is the **single source of truth** for in-browser injected globals; `RUNTIME_CONTRACT.md` and per-question `runtime_dependencies` must stay in sync.
- Image visuals must degrade gracefully — never leave a blank canvas. Catch path in `visuals.js` must end in either rendered canvas or text note.
- `numbers_stacked.png` and `delta_numbers.npy` must remain at sibling-of-folder served paths; multi-path candidate lookup tolerates moves but `/Local_Deployed_Shared/...` is the canonical path.
- Never inject runtime globals silently without updating both `RUNTIME_CONTRACT.md` and the JSON probes that derive `runtime_dependencies`.
- AI-judged grading must not rely on `assert_*` helpers being injected — those are not in the in-browser contract.

## Extension Points
- **Add a new injected global**: edit `runner.js:buildPyodidePreamble()`, update `RUNTIME_CONTRACT.md`, re-run the dependency-probe script that populates `runtime_dependencies` in the JSON.
- **Add a new question topic**: extend `questionNeedsEinops` / `questionNeedsArenaArray` in `visuals.js`, ensure the topic's data assets exist in `Local_Deployed_Shared/`, and add probes to the annotator script.
- **Add a per-question fallback image**: set `fallback_image_url` on the question; `visuals.js:getArenaNumbersPngCandidates()` will prefer it.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Image preview blank when Pyodide errors** — `RESOLVED`
  - When it happens: visual-output question, Pyodide fetch/exec fails.
  - Symptom: empty canvas with only an error string.
  - Root cause: catch block had no fallback path.
  - Prevention/fix: `visuals.js:renderQuestionVisual` now calls `renderFallbackImage()` before showing the error note. Keep a static PNG at a served path.
  - Status: RESOLVED (2026-04-28).

- **Torch-dependent prereq questions throw `NameError: 't'`** — `ACTIVE`
  - When it happens: einops-flagged question (forced to local Pyodide by `runner.js:148`) also references `t.tensor`/`Tensor`/`assert_all_*`.
  - Symptom: run output is `NameError`. 18 of 27 prereq questions are affected; see `runtime_unmet_dependencies` per question.
  - Root cause: in-browser preamble does not inject torch or ARENA test helpers; routing forces einops questions to Pyodide regardless.
  - Prevention/fix: route questions with non-empty `runtime_unmet_dependencies` to the backend, OR inject a torch/numpy compat shim in `buildPyodidePreamble`. See `RUNTIME_CONTRACT.md` for the full unmet-dep table.
  - Status: ACTIVE.

- **Question gives away answer in starter docstring** — `ACTIVE` (design call)
  - When it happens: function-impl questions where the docstring shows expected output literally (e.g. id 9).
  - Symptom: low cognitive challenge — user can read the answer from the editor.
  - Root cause: ARENA upstream pedagogy embeds expected outputs in docstrings.
  - Prevention/fix: pedagogical trade-off; if changing, scrub docstrings via the export script and verify tests still pass.
  - Status: ACTIVE — keep as-is unless explicitly redesigning.

## Recent Changes
- 2026-04-28: Added `RUNTIME_CONTRACT.md`, `numbers_stacked.png` fallback, `runtime_dependencies` annotations on all 27 prereq questions; fixed id-27 einsum canonical solution.
- 2026-04-27: Initial doc created.
