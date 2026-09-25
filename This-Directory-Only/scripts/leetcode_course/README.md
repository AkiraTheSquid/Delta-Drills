# leetcode_course (scripts)

## Purpose
- Builds the LeetCode Patterns course's problem bank: every problem tested,
  bucketed into a Grokking-pattern concept, and difficulty-rated within its concept.
- No lessons (Seth, 2026-09-25): the course is drills only.

## Owns
- The pipeline `build_pool.py` → `verify.py` → `gen_missing.py` → `tag.py` →
  `select_bank.py` → `rate_difficulty.py`, and the graph check `check_graph.py`.
- The generated-problem runner `harness.py` (plain-function and design shapes,
  linked-list / tree / cyclic-list argument encodings).

## Does NOT own
- The concept graph itself and the finished bank: `../../leetcode_course/`
  (`patterns.json`, `tags.csv`, `bank.jsonl`).
- The Jev pairwise judge / king-of-the-hill / rescale: `../typesafe_difficulty/`
  (reused unchanged; only its context, criteria and cache path are swapped).
- App wiring (course registry, kc_registry, question bank): not built yet.

## Key Files
- `build_pool.py`: joins the sheets to LeetCodeDataset → `RAW_DIR/pool.jsonl`.
- `verify.py`: runs every reference solution against its own tests.
- `gen_missing.py`: codex writes statement + optimal + brute-force solutions + inputs
  for core problems with no dataset row; kept only if both solutions agree.
- `harness.py`: runs a generated problem's solution over its inputs.
- `tag.py`: one Jev request per problem, one Noul per concept + 2 clean-up questions.
- `select_bank.py`: concept assignment, clean-up filter, fallback top-up to FLOOR.
- `rate_difficulty.py`: per-concept 15..100 difficulty via `../typesafe_difficulty/`.
- `check_graph.py`: DAG / encompassing ⊆ prereqs / redundancy / reachability.

## Data & External Dependencies
- Raw inputs outside the repo in `RAW_DIR` = `~/.cache/delta-drills/leetcode/`
  (`LEETCODE_RAW_DIR` overrides): the NeetCode 250 / Striver A2Z / big LeetCode /
  Codewars CSVs from Seth's Downloads and `newfacade/LeetCodeDataset` (Apache-2.0,
  `train.jsonl` + `test.jsonl` from Hugging Face). Caches live there too.
- `codex` CLI (gen_missing; runs on Codex usage, not Claude). `typesafe-sdk` in
  `backend/.venv`, key in `../../typesafe_difficulty/.env`.

## How It Works (Flow)
1. `build_pool.py`: core = NeetCode 250 ∪ Striver LeetCode problems minus Top 100
   Liked; fallback = rest of the big sheet (+ Codewars, currently unused).
2. `verify.py` → `RAW_DIR/verified.jsonl`.
3. `gen_missing.py` (27 core LeetCode + 123 Striver GfG/TUF, Beginner step skipped).
4. `tag.py --fallback` → `tags.csv`; 5. `select_bank.py` → `bank.jsonl`;
   6. `rate_difficulty.py` writes `difficulty_score` into `bank.jsonl`.

## Invariants & Constraints
- No problem ships unless its reference solution passes its tests; a generated
  problem's expected outputs come from RUNNING two independent solutions that agree,
  never from a model writing outputs.
- Codewars and fallback problems only top up concepts below FLOOR (Seth: no bulk
  easy problems).
- Run everything with `nice`; `gen_missing.py` keeps at most two codex processes.

## Extension Points
- New concept: add a row to `patterns.json` (with `describe`), run `check_graph.py`,
  re-run `tag.py` (every problem re-asks: the question set is in the fingerprint).
- New argument encoding: `harness.py` `_IN` / `_OUT` + the `arg_types` text in
  `gen_missing.PROMPT`.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Dataset preamble imports sortedcontainers** — `RESOLVED`
  - When it happens: verifying LeetCodeDataset rows.
  - Symptom: 43 problems fail with `No module named 'sortedcontainers'`.
  - Root cause: the preamble imports `SortedList` for every problem.
  - Prevention/fix: `build_pool.py` drops the import unless the solution names `Sorted*`.
  - Status: `RESOLVED`.

## Recent Changes
- 2026-09-25: `export_app.py` added, the last pipeline step. It grades every drill
  (the reference passes, the bare starter fails) and writes problems.json, ids.json, the
  `leetcode.*` registry rows, the qmatrix slice, glossary kcLesson, clock caps, `lc-*` atoms,
  edges and atom tags. `select_bank.py` runs without generated.jsonl.
- 2026-09-25: Folder created: pool, verify, codex generation, TypeSafe tagging,
  selection and difficulty rating for the LeetCode Patterns course.
