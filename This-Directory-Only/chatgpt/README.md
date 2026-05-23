# chatgpt (runtime)

## Purpose
Runtime artifacts for the AI quality-fix pipeline — secrets, request/response logs, override JSONLs, and quota state. Distinct from `../../Local_Deployed_Shared/chatgpt/` which holds the **code** (committed); this folder holds the **runtime data** the code reads and writes (mostly local-only, never deployed).

## Owns
- `api_key.txt` — OpenAI API key. **Never commit.**
- AI override outputs consumed by the backend on boot:
  - `function_mode_overrides.jsonl` — round-1 function-mode quality fixes (starter_code, test_cases, etc.) keyed by question id.
  - `function_mode_overrides_round2.jsonl` — round-2 manual repairs of validator-flagged failures.
  - `function_mode_overrides_round3.jsonl` — round-3 manual repairs of validator-flagged failures.
  - `einops_prompt_rewrite_overrides.jsonl` — natural-language rewrites of pattern-leaking einops prompts.
  - `numpy_einsum_prompt_rewrite_overrides.jsonl` — natural-language rewrites of numpy + einsum prompts that named the canonical answer function.
  - `function_mode_deleted_ids.json` / `function_mode_broken_ids.json` — id sets the backend skips.
- AI repair pipeline scratch:
  - `function_mode_requests.jsonl`, `function_mode_repair_requests.jsonl` — outbound prompts.
  - `function_mode_rejected.jsonl`, `function_mode_rejected_round2.jsonl` — outputs that failed validation.
  - `function_mode_validation_failures.jsonl` — fed to the next repair round.
  - `repairs/attempt_*` — per-round prompt/output dumps.
- Token-budget state: `usage.md`, `tokens_within_min.txt`, `Minute_Token_limit_reached.txt`, `day_token_limit_reached`.
- Model config: `gpt_model_type`, `gpt_model_type.txt`, `settings.txt`, `gpt_2_settings.txt`.
- Operational logs: `completion_state.txt`, `validator_health.txt`, `output.meta.json`, `outputs/`, `prompts/`, `logprobs/`.

## Does NOT own
- The code that reads/writes these files — that lives in `../../Local_Deployed_Shared/chatgpt/` (`ChatGPT_batch.py`, `function_mode_batch.py`, `ChatGPT.py`).
- The AI repair *orchestrator* — `../scripts/build_function_bank.py` (a shim that delegates to `Local_Deployed_Shared/build_function_bank.py`).
- The system prompts — `../../Local_Deployed_Shared/function_mode_*.txt`, `einops_prompt_rewrite_system.txt`.

## Key Files
- `function_mode_overrides.jsonl` — backend `_load_function_overrides()` reads this on startup; merged with the round-2 / round-3 manual repairs and the prompt-rewrite layers.
- `function_mode_overrides_round2.jsonl` / `function_mode_overrides_round3.jsonl` — manual repair layers; same schema as round-1. Each round is layered on top of the previous, so a later round wins on field-level conflicts for the same id.
- `einops_prompt_rewrite_overrides.jsonl` — produced by `Local_Deployed_Shared/rewrite_einops_prompts.py`; question_text-only records.
- `numpy_einsum_prompt_rewrite_overrides.jsonl` — produced by `Local_Deployed_Shared/rewrite_numpy_einsum_prompts.py`; question_text-only records. Layered last so its rewrites win over the round-1/2/3 overrides on conflicts; runnable starter/test fields from the earlier rounds survive.
- `function_mode_broken_ids.json` — questions the validator could not auto-repair across 3 rounds. Backend skips them entirely; surface as candidates for manual fixing.
- `function_mode_deleted_ids.json` — explicitly excluded ids; same skip path.
- `api_key.txt` — first thing `ChatGPT_batch.load_api_key()` looks for after the env var. Single-line key; trimmed.
- `usage.md` — token-budget bookkeeping; `ChatGPT_batch.UsageTracker` reads/writes it.
- `settings.txt` — model + temperature + per-minute limit overrides; consumed by `parse_settings()`.

## Data & External Dependencies
- **OpenAI API** — every other file in here is a side-effect of API calls.
- **Backend `app.questions._load_function_overrides()`** consumes `*_overrides.jsonl` and the `*_ids.json` files at FastAPI startup.
- **Pipeline orchestrator** — `Local_Deployed_Shared/build_function_bank.py` writes most files here.
- **One-shot rewrite script** — `Local_Deployed_Shared/rewrite_einops_prompts.py` writes `einops_prompt_rewrite_overrides.jsonl` here.

## How It Works (Flow)
1. Pipeline starts: orchestrator runs `export_questions_json.py` → applies any existing overrides from this folder → writes `questions_full.json`.
2. `validate_function_bank.py` runs each question's harness; failures append to `function_mode_validation_failures.jsonl` here.
3. `build_function_mode_repair_requests.py` reads failures, writes `function_mode_repair_requests.jsonl` here.
4. `function_mode_batch.py` calls OpenAI; valid candidates append to `function_mode_overrides.jsonl`, rejected ones to `function_mode_rejected.jsonl`. UsageTracker updates `usage.md`.
5. Orchestrator re-validates. Up to 3 rounds. Anything still failing ends up in `function_mode_broken_ids.json`.
6. Manual repair: still-broken ids are hand-fixed by writing corrected records into `function_mode_overrides_round2.jsonl` or `_round3.jsonl`. Each entry must pass the same harness (`code_runner.run_code` with the `_delta_expected_value = eval(expected_expr, globals())` check) before being committed.
7. Standalone: `rewrite_einops_prompts.py` runs once, writes `einops_prompt_rewrite_overrides.jsonl`.
8. Backend boot: `_load_function_overrides()` merges all four `*_overrides.jsonl` files in order and reads `*_ids.json` to filter the served bank.

## Invariants & Constraints
- **`api_key.txt` must never be committed or copied to `Local_Deployed_Shared/`.** The `This-Directory-Only/watch.py` check enforces this for the public-bundle directory.
- Each line of an `*_overrides.jsonl` must be a JSON object with an integer `id`. Backend `_load_jsonl_overrides()` will skip the whole file on a parse error.
- `function_mode_broken_ids.json` and `function_mode_deleted_ids.json` must be JSON arrays of ints.
- Override files are **append-only by id, never reorder**: question ids are CSV-row-position-derived; reordering CSV rows will silently shift overrides to the wrong questions.
- Rate-limit bookkeeping (`usage.md`, `tokens_within_min.txt`) is owned by `ChatGPT_batch.UsageTracker`. Don't hand-edit these files; the tracker re-derives state on next run.
- The `DELTA_CHATGPT_RUNTIME_DIR` env var can override this path (e.g. for tests); both the orchestrator and backend honor it via `delta_paths.get_chatgpt_runtime_dir()`.

## Extension Points
- **Add a new override layer**: write a new JSONL here (e.g. `numpy_prompt_rewrite_overrides.jsonl`), and update backend `_load_function_overrides()` to merge it. Layer order matters — later writes win on key conflicts.
- **Add a new id-set filter**: write a JSON array, register it in backend `load_questions()` via `_load_id_set("yourfile.json")` and pass to `_load_csv_into`.
- **Track a new model's quota separately**: extend `UsageTracker` in `ChatGPT_batch.py` and add new bookkeeping files here.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Stale overrides outlive their CSV** — `ACTIVE`
  - When it happens: a CSV row gets edited (text change, answer fix), but the old override for that id still points at the old wording.
  - Symptom: question_text or starter_code is stuck on the pre-edit value.
  - Root cause: overrides are append-only and key by id; nothing detects when the underlying CSV row drifted.
  - Prevention/fix: re-run the relevant pipeline (`build_function_bank.py` or `rewrite_einops_prompts.py`) after CSV edits. Or delete the stale line by id from the JSONL.
  - Status: ACTIVE.

- **Two chatgpt folders cause confusion** — `ACTIVE` (architectural)
  - When it happens: someone looks for `api_key.txt` in `Local_Deployed_Shared/chatgpt/` (the code dir) instead of here (the runtime dir).
  - Symptom: `load_api_key()` returns None even though the key is on disk somewhere.
  - Root cause: the code dir and runtime dir are split for deploy reasons but share the name "chatgpt".
  - Prevention/fix: scripts should fall back through `get_chatgpt_runtime_dir()` (the runtime dir) when `load_api_key(code_dir)` returns None. `rewrite_einops_prompts.py:main()` does this.
  - Status: ACTIVE — naming kept for backward compat.

## Recent Changes
- 2026-04-29: `numpy_einsum_prompt_rewrite_overrides.jsonl` added (164 records). Sweep of numpy + einsum prompts that named the canonical answer function (e.g. "Row-wise argmax", "Tile X three times"). Produced by `Local_Deployed_Shared/rewrite_numpy_einsum_prompts.py` + `numpy_einsum_prompt_rewrite_system.txt`. Layered last in the override stack so its question_text wins; runnable scaffolding from the round-1/2/3 layers survives.
- 2026-04-29: 4 from-scratch scaffolds for the unscaffolded numpy-100 holdouts (171 cartesian product, 184 symmetric assignment, 201 point-to-line distance, 209 Game of Life) appended to `function_mode_overrides_round3.jsonl`. Validator broken-id count: 4 → 0; student-facing bank: 384 → 388.
- 2026-04-29: `function_mode_overrides_round3.jsonl` added (25 manual repairs of validator-flagged failures). Loader merge order is now round-1 → round-2 → round-3 → einops prompt rewrite. Validator broken-id count: 29 → 4 (the 4 unscaffolded numpy-100 holdouts).
- 2026-04-28: `einops_prompt_rewrite_overrides.jsonl` added (45 records); doc filled in.
