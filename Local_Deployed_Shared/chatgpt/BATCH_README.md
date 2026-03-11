# ChatGPT Batch Processing

This folder now serves two purposes:

1. General-purpose parallel OpenAI batch requests via `ChatGPT_batch.py`
2. Function-mode conversion support for the Delta Drills question bank

The main new workflow is for question normalization:
- every question in the exported bank is now function-mode
- `chatgpt` is used as an override/repair layer for improving starter code and test cases
- raw model outputs should not be merged blindly without validation

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure your API key is configured (same as for `ChatGPT.py`):
   - Set `OPENAI_API_KEY` environment variable, or
   - Create `api_key.txt` with your key

## General Batch Usage

1. Create numbered prompt files in the `prompts/` directory:
   - `prompts/1_prompt.txt`
   - `prompts/2_prompt.txt`
   - ... up to `prompts/100_prompt.txt`

2. Run the batch processor:
   ```bash
   python ChatGPT_batch.py
   ```

3. Results will be written to:
   - `outputs/1_output.txt`
   - `outputs/2_output.txt`
   - etc.

## Function-Mode Bank Workflow

The current Delta Drills bank has been migrated to function mode across:
- `NumPy`
- `Einsum`
- `Einops`

The active export path is deterministic and lives outside this folder:
- [scripts/export_questions_json.py](/home/stellar-thread/Applications/Delta-Drills-Local/scripts/export_questions_json.py)

That exporter now emits:
- [questions.json](/home/stellar-thread/Applications/Delta-Drills-Local/questions.json)
- [questions_structured.json](/home/stellar-thread/Applications/Delta-Drills-Local/questions_structured.json)

Each question now carries:
- `function_name`
- `starter_code`
- `test_cases`
- `submission_mode`

### Current State

As of the latest conversion pass:
- all `388` questions export as `submission_mode = "function"`
- backend and frontend are both wired to grade function-mode questions with test cases
- `chatgpt` overrides are optional and are meant to improve quality, not define the baseline bank

### Files Added For Function-Mode Conversion

- [function_mode_batch.py](/home/stellar-thread/Applications/Delta-Drills-Local/chatgpt/function_mode_batch.py)
- [scripts/build_function_mode_requests.py](/home/stellar-thread/Applications/Delta-Drills-Local/scripts/build_function_mode_requests.py)
- [scripts/build_function_bank.py](/home/stellar-thread/Applications/Delta-Drills-Local/scripts/build_function_bank.py)
- [scripts/validate_function_bank.py](/home/stellar-thread/Applications/Delta-Drills-Local/scripts/validate_function_bank.py)
- [scripts/test_function_validator.py](/home/stellar-thread/Applications/Delta-Drills-Local/scripts/test_function_validator.py)
- [scripts/build_function_mode_repair_requests.py](/home/stellar-thread/Applications/Delta-Drills-Local/scripts/build_function_mode_repair_requests.py)

### What These Files Do

- `build_function_mode_requests.py`
  - creates `chatgpt/function_mode_requests.jsonl`
  - includes questions that still need LLM-based normalization or repair

- `function_mode_batch.py`
  - reads request JSONL files for conversion or repair
  - asks the model to produce a function-mode override
  - validates the generated override locally against the canonical answer and test harness
  - if validation fails, feeds the failure details back to the model and retries up to the configured attempt limit
  - writes only validated overrides to `function_mode_overrides.jsonl`
  - writes exhausted failures to `function_mode_rejected.jsonl`
  - can add permanently bad IDs to `function_mode_deleted_ids.json`

- `build_function_bank.py`
  - runs the end-to-end function-bank refresh sequence
  - export -> validate -> build repair requests -> run repair retries -> re-export -> re-validate

- `validate_function_bank.py`
  - standalone deterministic validator for the exported `questions.json`
  - checks whether function-mode questions have coherent test fixtures and whether the canonical solution passes the generated tests
  - should be runnable on its own, outside the LLM batch flow
  - writes failures to `chatgpt/function_mode_validation_failures.jsonl`

- `test_function_validator.py`
  - deterministic regression suite for the validator itself
  - tests known-good and known-bad validator cases so the validator does not silently become too strict or too weak
  - writes validator health status to `chatgpt/validator_health.txt`
  - first line is:
    - `0` if the validator regression suite passes
    - `1` if the validator regression suite fails
  - subsequent lines list each regression case by name and whether it passed or failed

## Standalone Validator Workflow

The validator is meant to be used directly, not only through `build_function_bank.py`.

Run the validator regression suite first:
```bash
python3 scripts/test_function_validator.py
```

Then inspect:
- `chatgpt/validator_health.txt`

Interpretation of `validator_health.txt`:
- first line `0` means the validator regression cases passed
- first line `1` means at least one validator regression case failed
- each later line is one named regression case, with expected vs actual validator behavior

Then run the bank validator itself:
```bash
python3 scripts/validate_function_bank.py
```

Then inspect:
- `chatgpt/function_mode_validation_failures.jsonl`

Important:
- `test_function_validator.py` validates the validator
- `validate_function_bank.py` validates the exported question bank
- these are different checks and both should be runnable independently

## Important Warning

Raw model output is still not trusted directly.

Observed issues from the first pass included:
- starter code that still printed instead of returning
- malformed `expected_expr`
- bad `setup_code`
- inconsistent use of `solve()`

So:
- the deterministic export is the current source of truth
- the LLM override file should only contain candidates that passed local validation
- rejected repairs are logged separately
- questions that still fail after the configured retry budget can be removed from export via `function_mode_deleted_ids.json`

If you come back to this later, do not assume `function_mode_overrides.jsonl` is production-ready just because it exists.

## Recommended Next-Day Workflow

1. Regenerate the deterministic bank:
```bash
python3 scripts/export_questions_json.py
```

2. Run the validator regression suite:
```bash
python3 scripts/test_function_validator.py
```

3. Check validator health:
```bash
cat chatgpt/validator_health.txt
```

4. Run the standalone bank validator:
```bash
python3 scripts/validate_function_bank.py
```

5. Generate pending LLM normalization requests:
```bash
python3 scripts/build_function_mode_requests.py
```

6. Run the repair/conversion worker with an interpreter that has `openai` installed.
In this repo, the backend venv currently works:
```bash
/home/stellar-thread/Applications/Delta-Drills-Local/backend/.venv/bin/python3 chatgpt/function_mode_batch.py
```

7. Preferred full loop:
```bash
python3 scripts/build_function_bank.py
```

8. Inspect:
- `chatgpt/function_mode_overrides.jsonl` for validated repairs
- `chatgpt/function_mode_rejected.jsonl` for questions that failed all retries
- `chatgpt/function_mode_deleted_ids.json` for IDs excluded from export
- `chatgpt/function_mode_validation_failures.jsonl` for bank-level validation failures
- `chatgpt/validator_health.txt` for validator-regression status

9. Re-export after any manual changes:
```bash
python3 scripts/export_questions_json.py
```

## Current Caveat

The validator architecture is now explicitly split into:
- validator-regression checks (`test_function_validator.py`)
- bank validation (`validate_function_bank.py`)

That split is intentional and should be preserved.

However, do not assume the validator is fully solved yet just because these scripts exist.
At the moment, the validator runtime still needs more work for some `einops` cases, so:
- the validator infrastructure is in place
- the validator health file is now the quickest way to see if the validator itself is currently trustworthy

## Why Function Mode

Function mode is now the target architecture because it gives:
- deterministic grading
- test harness support
- less dependence on `print(...)`
- less dependence on prompt fixture text being copied by the user
- better long-term support for `NumPy`, `Einsum`, and `Einops`

For visual `Einops` tasks, the app now also supports image rendering in the practice UI.

## Features

### Parallel Processing
- All non-empty prompt files (1-100) are processed simultaneously using async/await
- Much faster than sequential processing

### Rate Limiting
The script automatically tracks and enforces OpenAI's rate limits:

- **Per-minute limits:**
  - 500,000 tokens
  - 500 requests
  
- **Per-day limits:**
  - 5,000,000 tokens

- **Per-batch limit:**
  - 300,000 input tokens (safety limit to prevent excessive single batches)

### Automatic Waiting
If a batch would exceed rate limits, the script will:
- Wait until the minute window resets (for minute limits)
- Wait until midnight (for daily limits)

### Usage Tracking
All usage is tracked in `usage.md`:
- Tokens used within the current minute
- Requests made within the current minute
- Tokens used within the current day
- Flags when limits are reached

## Model Configuration

By default, the script uses `gpt-4o-mini`. You can change this by:
- Setting `OPENAI_MODEL` environment variable, or
- Creating `gpt_model_type.txt` with the model name

## Error Handling

- If a prompt fails, the error is written to its output file
- Other prompts continue processing
- Script exits successfully even if some prompts fail
- For function-mode conversion, malformed model outputs should be expected and validated

## Examples

### Process 5 prompts in parallel
```bash
# Create prompts
echo "What is 2+2?" > prompts/1_prompt.txt
echo "What is the capital of France?" > prompts/2_prompt.txt
echo "Explain quantum computing in one sentence." > prompts/3_prompt.txt
echo "What is the speed of light?" > prompts/4_prompt.txt
echo "Name three primary colors." > prompts/5_prompt.txt

# Run batch
python ChatGPT_batch.py
```

### Check usage
```bash
cat usage.md
```

