# pipeline

## Purpose
- Turns the raw authored problem CSVs into the question bank the app actually serves, and refuses to let a bad bank ship. Everything here is build-time or gate-time; none of it runs while a learner is practising.
- The domain concern is trust in the drill: a question that grades a wrong answer as correct, leaks its own answer, or shows a reference output that contradicts its example is worse than no question at all.

## Owns
- The CSV → `questions.json` / `questions_structured.json` / `questions_full.json` export, including the layered per-id override merge.
- The deploy gates: gameability auditing, grading-harness regression, function-bank validation, atom-graph verification.
- The generation-side harness for authoring/rewriting questions (mechanical candidate gate, prompt expansion, prompt rewrites, regen status reporting).

## Does NOT own
- Runtime grading — that is `This-Directory-Only/backend/app/code_runner.py`. This folder only *tests* it (`test_torch_grading.py`).
- Question serving, selection, or parking policy — `backend/app/questions.py` and `backend/app/lessons.py`.
- Lesson/KC content — `Local_Deployed_Shared/lessons/`.
- The override JSONL files themselves — they live in `This-Directory-Only/chatgpt/` (gitignored dir, individually force-added).

## Key Files
- `export_questions_json.py`: the export. Reads the four CSVs, applies the override layer stack, writes the three bank artifacts. Runs at deploy Step 2.
- `audit_question_bank.py`: `--gate` is deploy Step 2b. Blocks on gameable grading (bare-fixture cheats, degenerate expecteds), broken starters, precompute leaks.
- `test_torch_grading.py`: deploy Step 2c. Regression suite for `code_runner`'s equality harness — tensor equality, rng seeding across setup re-exec, mech-gate non-degeneracy.
- `mech_gate_candidate.py`: innermost deterministic gate for LLM-generated candidates; runs *before* any model review.
- `validate_function_bank.py` / `test_function_validator.py`: function-mode structural validation.
- `verify_atom_graph.py`: deep atom prereq-graph checks (beginner lock state, deadlock sweep, diagnostic-seed reachability). Fast structural subset also runs in the audit gate.
- `regen_status.py`: which questions still need the parameterized-regen rewrite.

## Data & External Dependencies
- Inputs: `This-Directory-Only/csv files of problems/*.csv` (numpy, einsum, einops, cnn) plus the override JSONL stack in `This-Directory-Only/chatgpt/`.
- Outputs: `Local_Deployed_Shared/questions.json`, `questions_structured.json`, `This-Directory-Only/questions_full.json`.
- Needs the backend venv (`This-Directory-Only/backend/.venv`) for anything touching torch — system python has numpy and einops but **not** torch.

## How It Works (Flow)
1. Deploy Step 2 runs `export_questions_json.py`. CSV rows become questions with inferred difficulty/library/task-type and a derived function payload.
2. Override layers merge per id, in list order, last wins. Only whitelisted fields replace CSV values: `function_name`, `starter_code`, `test_cases`, `submission_mode`, `question_text`, `answer_code`, `expected_output`, `task_type`, `expected_artifact_type`, `supports_visual_output`, `difficulty_score`.
3. Step 2b audits the emitted bank and hard-fails the deploy on gameable grading.
4. Step 2c runs the grading-harness regression against `code_runner`.

## Invariants & Constraints
- **`questions.json` is generated, never authored.** It is rewritten on every deploy — hand edits are silently erased. All question changes go in an override layer.
- **The layer list is duplicated and must stay in sync**: `load_function_overrides()` here and `_load_function_overrides()` in `backend/app/questions.py`. Adding a layer to one and not the other makes local and prod disagree about what the bank says.
- **The field whitelist is also duplicated** across those same two files. A field not in the whitelist is silently ignored in an override — it does not error.
- New override layers live in `This-Directory-Only/chatgpt/`, which is **gitignored**. Every sibling layer is individually force-added; a new one needs `git add -f` or it will never reach the deploy.
- A non-empty `function_mode_broken_ids.json` makes the export silently DROP those ids. The deploy script refuses to run on a stale one — do not work around that check.
- Torch-dependent scripts must be run with the backend venv interpreter, not `python3`.

## Extension Points
- New question rewrites → add a JSONL layer in `This-Directory-Only/chatgpt/`, register it last in **both** override-layer lists, `git add -f` it.
- New defect class to block on → add a check to `audit_question_bank.py` (gate-blocking) or `mech_gate_candidate.py` (generation-time).
- New grading-harness behavior → add a case to `test_torch_grading.py` first; it gates the deploy.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Stale `expected_output` in the committed bank** — `ACTIVE (local only)`
  - When it happens: a rewrite changes `answer_code` or the canonical example, and the committed `questions.json` is not re-exported afterwards.
  - Symptom: the reference output shown beside a drill contradicts the drill's own example (e.g. an example printing a 2×3 with a 2×2 output). Grading is unaffected — function-mode grades via `test_cases`.
  - Root cause: the CSV `Output` column was captured out-of-harness and drifts as questions are rewritten.
  - Prevention/fix: **already handled at export time.** `recompute_expected_outputs()` re-runs every stdout-graded question's `answer_code` through the grading harness and overwrites the stored string, so any deploy self-heals. Generators should still derive the value by execution rather than hand-writing it.
  - Status: `ACTIVE` only in the sense that the *committed artifact* drifts between deploys — the last export corrected 189 entries. Measured against the harness immediately after export, exactly **1** question mismatches (#42, genuinely random) and #65 errors by design.
  - **Measurement footgun:** comparing `expected_output` against a plain `exec()` of `answer_code` reports ~142 false positives. The reference is the harness (preamble + seeding), not bare exec — an earlier pass in this repo mis-measured this as "118 stale" for exactly that reason.

- **Precompute leak** — `RESOLVED`
  - When it happens: an `expected_expr` restates the solution, so the test passes without the learner solving anything.
  - Symptom: trivially gameable question; mastery estimates inflate.
  - Root cause: derived test cases mirroring the answer expression.
  - Prevention/fix: `audit_question_bank.py --gate` at deploy Step 2b. Run it locally per batch, not only at deploy.
  - Status: `RESOLVED` as a shipping risk; 7 flagged ids remain as non-blocking warnings (264, 268, 272, 279, 283, 305, 448).

- **Layer-list drift between exporter and backend** — `ACTIVE`
  - When it happens: a layer is registered in one of the two override-layer lists only.
  - Symptom: local and deployed banks disagree; a conversion or fix appears to "not take" in one environment.
  - Root cause: the merge order is intentionally duplicated rather than shared, to keep the backend free of a pipeline import.
  - Prevention/fix: edit both lists in the same commit; both carry a "keep in sync" comment naming the other file.
  - Status: `ACTIVE` — structural, no automated check enforces it.

## Recent Changes
- 2026-07-28: `torch_dialect_overrides_np23.jsonl` registered as the third dialect layer (np-2 + np-3, 120 questions). The bank is now 390 torch / 58 numpy; the remaining numpy is np-4 plus q65, whose `ndarray.flags.writeable` content has no torch equivalent. `audit_question_bank.py --gate` earns its keep here: converted STARTERS carry the same demo block as answers, and a raw list assigned into a tensor (`img[0, 0] = [255.0, ...]`) is a blocking `setup_exec_error` that no other gate sees.
- 2026-07-27: `torch_dialect_overrides.jsonl` now carries all 49 of np-1's questions in the PyTorch dialect. The staleness entry above was corrected: the earlier 118/364 figure was measured against bare `exec` instead of the grading harness, and the real post-export count is 1.
- 2026-07-27: `expected_output` added to the override whitelist in both the exporter and the backend; new `torch_dialect_overrides.jsonl` layer registered last.
- 2026-04-29: Initial doc created.
