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
- Question serving, selection, or parking policy — `backend/app/questions.py` and `backend/app/lessons.py`. Worth knowing while working on the bank: `lessons.torch_only_serving()` DEFAULTS ON as of 2026-07-28 (`DELTA_TORCH_ONLY` defaults to `"1"`), and it drops every question that does not import torch out of the SELECTION pools — the by-subtopic lists the ITS picks from. Lookup by id is deliberately untouched, so history and in-flight attempts still resolve a parked question. Dialect is read off the question's own `answer_code`/`starter_code` rather than a stored field, so it cannot drift out of sync with what the question actually asks, and converting a question unparks it with no schema change. Export a NumPy question today and the app will simply never offer it.
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
- **The layer list is duplicated and must stay in sync**: `load_function_overrides()` here and `_load_function_overrides()` in `backend/app/questions.py`. Adding a layer to one and not the other makes local and prod disagree about what the bank says. **Order matters as much as membership** — layers merge last-wins, so the same filenames in a different sequence still resolve a conflicting id differently, and `watch.py` compares the two lists as ordered sequences for that reason.
- **The field whitelist is also duplicated** across those same two files. A field not in the whitelist is silently ignored in an override — it does not error.
- New override layers live in `This-Directory-Only/chatgpt/`, which is **gitignored**. Every sibling layer is individually force-added; a new one needs `git add -f` or it will never reach the deploy. This is the half of the registration footgun that no check catches: `watch.py` verifies that a registered layer EXISTS on disk, which it does locally the moment the generator writes it, and nothing looks at whether git can see it. A layer can therefore pass every gate here and still be absent from the deployed bank.
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
  - Prevention/fix: edit both lists in the same commit; both carry a "keep in sync" comment naming the other file. `watch.py` (run by `mod watch`) now parses both files — without importing them, since importing the exporter would rewrite the bank as a side effect of a health check — and fails on any difference in membership OR order, and on a registered layer that does not exist on disk.
  - Status: `ACTIVE` for the part a checker cannot see. The divergence itself is caught; what is not caught is a layer that exists locally but was never `git add -f`'d out of the gitignored `chatgpt/` directory, which fails only in the deployed bank.

## Recent Changes
- 2026-08-13: `load_function_overrides()` gained a final layer that no script in this folder produces — `ai_feedback_overrides.jsonl`, written at runtime by the backend when a learner flags a question and Opus 5 repairs it (`backend/app/practice/feedback_ai_improver.py`). It is registered after `curated_overrides.jsonl` in both lists as usual, but breaks the folder's normal assumption in two ways. It is **not read from `CHATGPT_RUNTIME_DIR`**: `_feedback_ai_layer_path()` honours `DELTA_FEEDBACK_AI_DIR`, which production points at the Fly `/data` volume, because anything in the image is rebuilt by a deploy and a runtime write there would be lost. And it is **not in the repo**, so the third step of registering a layer (`git add -f`) does not apply — on a dev box the file is simply absent and the layer is empty. To fold production's repairs into the shipped bank, copy `ai_feedback_overrides.jsonl` down off the volume before exporting; `ai_feedback_revisions.jsonl` beside it holds the before/after for each one and is what to read before deciding whether a repair deserves promoting.
- 2026-07-30: Eight questions authored for the lesson ladder (q523–q530) through `curated_additions.csv` + `curated_overrides.jsonl`; the exporter minted the ids positionally as documented. Two things to know before doing this again. First, `export_questions_json.py` recomputes `expected_output` by running each answer, and it corrected **40 pre-existing** stale values along the way — all on unused CNN/Autograd/Optimizer drills (ids 405–460) whose stored Output predated their function-mode rewrite, e.g. q412 `'bias is None: True'` → `'True'`. That correction rides in on whatever commit happens to run the export next; it is the exporter working, not a regression, but it means an unrelated bank diff is normal here. Second, **do not run `validate_function_bank.py` on this machine.** It cannot execute torch `expected_expr`s outside the fork runner, so it fails all 499 questions and writes `chatgpt/function_mode_broken_ids.json` — the file the deploy script refuses to start with, and which would silently EXCLUDE every id from the export if the deploy were forced. Delete it if you hit this. `audit_question_bank.py --gate` plus `run_function_tests` from `backend/app/code_runner.py` are the checks that actually work offline; new questions also need a row in `backend/app/data/question_atom_tags.jsonl` (the gate blocks on `atom_tag_missing` — untagged questions move no mastery) followed by `export_kc_atom_crosswalk.py`.
- 2026-07-28: `torch_dialect_overrides_np4.jsonl` (45 np-4 drills) and `torch_dialect_overrides_parked.jsonl` (the 17 untagged CNN/backprop drills) registered as the fourth and fifth dialect layers, in that order, in **both** override-layer lists. That finishes the migration: the bank is a single dialect, 448 questions, 448 torch, no NumPy left. The seven questions that could not cross — the six `numpy.structured-dtypes` drills plus q65's `ndarray.flags.writeable` — were retired through `function_mode_deleted_ids.json` rather than converted, which is why the bank went 455 → 448. Registering a new layer is still the sharpest edge in this folder and now has three steps, not two: add it to `load_function_overrides()` here, add it in the same position to `_load_function_overrides()` in `backend/app/questions.py`, and `git add -f` the file itself. `watch.py` catches the first two failures; nothing catches the third.
- 2026-07-28: `torch_dialect_overrides_np23.jsonl` registered as the third dialect layer (np-2 + np-3, 120 questions). The bank is now 390 torch / 58 numpy; the remaining numpy is np-4 plus q65, whose `ndarray.flags.writeable` content has no torch equivalent. `audit_question_bank.py --gate` earns its keep here: converted STARTERS carry the same demo block as answers, and a raw list assigned into a tensor (`img[0, 0] = [255.0, ...]`) is a blocking `setup_exec_error` that no other gate sees.
- 2026-07-27: `torch_dialect_overrides.jsonl` now carries all 49 of np-1's questions in the PyTorch dialect. The staleness entry above was corrected: the earlier 118/364 figure was measured against bare `exec` instead of the grading harness, and the real post-export count is 1.
- 2026-07-27: `expected_output` added to the override whitelist in both the exporter and the backend; new `torch_dialect_overrides.jsonl` layer registered last.
- 2026-04-29: Initial doc created.
