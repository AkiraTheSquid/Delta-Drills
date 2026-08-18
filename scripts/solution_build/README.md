# solution_build

> Documented 2026-08-18 from source by a session that was passing through, not by
> the folder's author. Purpose, flow and invariants below are read off the
> scripts themselves; the "why" behind an authoring campaign's ordering may be
> thinner than usual.

## Purpose
- The **authoring** half of the question bank: turning a bank question into the artifacts a learner actually opens — a solution notebook, a problem notebook, a hint, a starter stub.
- The domain concern is that a torch drill cannot run in the in-app sandbox. Those questions are routed to Colab, and a question routed to Colab with no notebook behind it is a dead end.
- Most of what lives here is campaign tooling: an agent authors a batch, a validator refuses it until it passes the *real* grader, an assembler emits the notebook. The validators are the part worth preserving; the batches are history.

## Owns
- `dd_questions.json`, the snapshot of the bank every other script here reads.
- Notebook assembly: solution notebooks, problem notebooks, drill solution notebooks, ERE worked/faded tiers.
- The manifests that point the frontend at them (`question_solution_notebooks.jsonl`, `question_problem_notebooks.jsonl`, `question_hints.jsonl`, `drill-hints-manifest.js`).
- Validation of anything an agent authored, against the grader rather than against a human's opinion.

## Does NOT own
- The bank itself. Questions come from `This-Directory-Only/csv files of problems/` plus the override layers in `This-Directory-Only/chatgpt/`; the export here is a **read**. See `Local_Deployed_Shared/pipeline/`.
- Grading. `validate_solutions.py` / `validate_authored.py` import `app.code_runner` precisely so they cannot have their own opinion about correctness.
- Which questions are served or parked — `app/lessons.py` and `app/questions.py`.
- Runtime question repairs from learner feedback — `ops/question_repair/` and `app/feedback_repair_queue.py`.

## Key Files
- `export_questions.py`: bank → `dd_questions.json`. Run from the backend dir with its venv. Everything else here reads that file, so a stale export means every downstream artifact is built against a stale question.
- `validate_solutions.py`: every canonical bank answer against its own grader. Writes `validation_report.json`. Optional arg: a JSON list of ids, for repair re-runs.
- `validate_authored.py`: agent-authored `solution_code` against the same oracle. Function-mode goes through `run_function_tests`; stdout-mode requires a clean run, and an exact match only when `expected_output` is deterministic. Exit 0 iff all pass, so an agent can loop until green.
- `retorch_authored.py`: `authored/*.jsonl` (May 2026, written against the NumPy bank) → `authored_torch/layer.jsonl`, re-dialected onto the torch bank. `solution_code` is taken from the live `answer_code` and never translated; prose is renamed only where the bank's answer still matches the answer that prose described. Writes `retorch_report.json`.
- `build_solution_colabs.py`: `dd_questions.json` + `authored_torch/layer.jsonl` → `arena-procedural-drills/solutions/<topic>/q<ID>-<sub>.ipynb`, plus the id→path and id→hint manifests. Refuses to run if the layer is absent. Also prunes solution notebooks whose question has left the bank.
- `build_problem_colabs.py`: the same questions **without** the answer, as `.problem.ipynb`. Torch-only. Uses the EFFECTIVE starter — overrides merged over the export — so the notebook matches what the backend serves.
- `build_drill_solutions.py`: lifts the answer out of each procedural drill's collapsed `<details>` cell into its stub cell, emitting `<name>.solution.ipynb`. Refuses to emit a notebook that still contains `NotImplementedError` or a cell that will not compile.
- `merge_stubs.py`: authored starter stubs back into `chatgpt/function_mode_overrides_round3.jsonl`, in place, preserving row order and every other field.
- `validate_stubs.py` / `validate_hints.py`: a stub must compile, keep its `print(...)` scaffold, leave a TODO, and **not** be the answer; a hint must be 15–400 chars and name a known drill.
- `ere/`: its own campaign for ERE worked/faded tiers (`build_ere_notebooks.py`, `validate_ere.py`, `verify_exec.py`), cloning boilerplate cells from each atom's template notebook so auth/beacon plumbing matches house style.
- `authored/`, `stubs/`, `drill_hints/`: agent output, one JSONL per batch, with a `.report.json` beside each from its validator.

## Data & External Dependencies
- The backend package and its venv — anything that grades or exports needs torch, so `python3` is not enough.
- `This-Directory-Only/chatgpt/` override layers, read directly by `build_problem_colabs.py`.
- Writes into `arena-procedural-drills/` and `This-Directory-Only/backend/app/data/`, both of which the deployed app reads.
- Notebook paths are rooted at the `arena-procedural-drills/` prefix that `stats/predicted-links.js::colabUpstreamHref()` already routes to GitHub `AkiraTheSquid/Delta-Drills` main — so a new notebook needs no frontend routing, but it does need to be **committed and pushed** before a learner can open it.

## How It Works (Flow)
1. `export_questions.py` snapshots the bank to `dd_questions.json`.
2. An agent authors a batch into `authored/` (or `stubs/`, `drill_hints/`).
3. The matching validator runs the batch through the real grader and writes a report. It exits non-zero until the batch is clean, which is the loop the agent runs against.
4. An assembler turns validated output into notebooks plus a manifest.
5. The notebooks are committed; the frontend resolves them by path.

## Invariants & Constraints
- **Validate against the grader, never against `expected_output`.** For RNG questions the stored literal came from a different seed, so it disagrees with a correct answer. `run_function_tests` is the anchor.
- **A problem notebook must not contain the answer, and a stub must not be the answer.** `validate_stubs.py` enforces the second by requiring the stub to differ from `answer_code` *and* not reproduce the expected output on its own.
- **The export must cover the whole bank, not the servable pool.** `get_all_questions()` is narrowed by `kc_only_serving()` / `torch_only_serving()`; exporting that pool silently skips the parked questions and leaves their notebooks stale forever. (This is what `questions.get_every_question()` exists for.)
- **`dd_questions.json` is committed on purpose** — it is a session artifact that `/tmp` used to lose.
- **Never key a notebook's dialect on `primary_library`.** The backend infers it BEFORE the torch-dialect override layers are applied, so it still reports `numpy` for 437 of the 499 rows whose code is torch. Derive from the code, as `lessons.is_torch_dialect` and `build_solution_colabs.pip_for()` do. Keying on the stored field is exactly how every solution notebook came to tell a torch learner to `%pip install -q numpy`.
- **Prose may not be mechanically translated across a hand-translated answer.** ~30 drills used a numpy function torch cannot spell and were rewritten to a different algorithm. Renaming symbols in the old explanation yields fluent, confident, wrong text — worse than none, because the learner cannot tell. `retorch_authored.py` withholds instead, and reports the id.
- **This whole folder is gitignored** (`.gitignore:75`). Everything here that the build needs at runtime is tracked only because it was `git add -f`'d. A new input file will be silently absent for anyone else — same trap as the `chatgpt/` override layers.
- Run the exporting and grading scripts from `This-Directory-Only/backend` with `.venv/bin/python`.

## Extension Points
- New artifact per question → a new `build_*.py` reading `dd_questions.json` plus an `authored/` JSONL, and a `validate_*.py` that exits non-zero until the batch is clean. Do not skip the validator; it is what makes an authoring campaign safe to hand to an agent.
- New authoring campaign of its own shape → a subfolder, as `ere/` does.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Downstream artifacts built against a stale export** — `ACTIVE`
  - When it happens: a question is rewritten (batch layer, or a runtime repair from learner feedback) and `dd_questions.json` is not re-exported first.
  - Symptom: a solution or problem notebook whose prompt no longer matches the question the app serves.
  - Root cause: every script here reads the snapshot, not the bank, and nothing checks its age.
  - Prevention/fix: re-run `export_questions.py` before any build. For repairs applied in production, the runtime layer must be copied off the Fly volume first — see `Local_Deployed_Shared/pipeline/README.md`.
  - Status: `ACTIVE` — no staleness check exists.

- **Exporting the servable pool instead of the bank** — `RESOLVED`
  - When it happens: using `get_all_questions()` in an offline script.
  - Symptom: 424 questions instead of the full bank; the parked curated additions (ids ~405–479) never get notebooks, and nothing reports a miss.
  - Root cause: `get_all_questions()` returns the SELECTION pool, which serving policy has already narrowed.
  - Prevention/fix: offline coverage tools use `questions.get_every_question()`.
  - Status: `RESOLVED` in `export_questions.py`; the trap is open to any new script here.

## Recent Changes
- 2026-08-18: README created from source. `export_questions.py` moved to `get_every_question()` so the export covers parked questions, and `dd_questions.json` was regenerated (that change belongs to a concurrent session, not to this doc).
- 2026-08-18: **Solution notebooks re-dialected to torch.** All 455 were built in May 2026 against the NumPy bank and never rebuilt after the July conversion, so `Show Answer` on a torch question opened `import numpy as np` — 397 fully, 58 in prose or their `%pip` line. Added `retorch_authored.py`; `build_solution_colabs.py` now reads its layer, derives the pip line from the code instead of `primary_library`, relabels the heading through a mirror of `practice/config.js::TOPIC_DISPLAY_LABELS`, omits a withheld explanation rather than stubbing it, and prunes notebooks for retired questions. 447 rebuilt, 8 pruned (their questions were retired in July), 108 still need a hand-authored "Why this works" — 52 because the answer drifted, 56 because the prose still named a numpy symbol. The only surviving mentions are the three deliberate torch-vs-numpy contrasts in the prompts of q23 / q225 / q233.
