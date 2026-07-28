# csv files of problems

Raw question sources for the Delta Drills bank. Every drill the learner ever
sees begins as a row in one of these files.

## Intent

These CSVs are the **base layer** of the bank, and nothing else. They carry the
question text, a canonical answer, a difficulty number and a topic/subtopic
label — the material as it was originally imported. Everything that makes a row
servable (a `solve()` signature, a starter, graded test cases, a PyTorch
dialect) is applied on top as an override layer in `../chatgpt/*.jsonl`. Read a
row here as "what the question is about", never as "what the learner is shown".

## The invariant that matters: ids are positional

There is no id column. A question's id is its **ordinal position** in the
concatenation of the loaded CSVs, counted in a fixed source order, starting at
1. As of 2026-07-28:

| source | skip_rows | rows | id range |
|---|---|---|---|
| `Export of numpy problems with outputs.csv` | 2 | 242 | 1–242 |
| `einsum_problems.csv` | 0 | 70 | 243–312 |
| `einops_problems.csv` | 0 | 92 | 313–404 |
| `cnn_problems.csv` | 0 | 75 | 405–479 |
| `curated_additions.csv` | 0 | 1 | 480 |

The counter advances on every row that is *read*, including rows that are later
dropped as excluded or deleted — which is why the maximum id (480) exceeds the
449 questions actually exported.

**Consequence: inserting or deleting a row anywhere but the end renumbers every
question after it.** Those ids are referenced by

- `Local_Deployed_Shared/lessons/qmatrix_tags.json` (which KC each drill teaches),
- KP frontmatter `faded:` / `guided:` / `independent:` lists,
- every override record in `../chatgpt/*.jsonl`,
- and `served_question_ids` in stored per-learner state, on disk and in Neon.

A renumber silently re-points all of it. There is no checksum that would catch
it. Treat these files as append-only.

## Adding a question

Append a row to `curated_additions.csv` — that source exists to be last, so
appending mints an id above the entire imported bank and cannot renumber
anything. Then:

1. Run `This-Directory-Only/scripts/export_questions_json.py` and read the new
   id out of `questions_full.json`; you cannot predict it reliably by hand.
2. Add a record for that id to `../chatgpt/curated_overrides.jsonl` with
   `function_name`, `starter_code`, `answer_code` and `test_cases`. Without it
   the exporter *infers* a payload, and the inference produces `starter_code:
   null` and zero test cases — the question loads but cannot be graded.
3. Tag it: reference the id from a KP page's `faded:` / `guided:` /
   `independent:` list, then run `scripts/build_qmatrix.py`. An untagged easy-topic
   question makes that script exit non-zero.
4. Gate it: `Local_Deployed_Shared/pipeline/audit_question_bank.py --gate`. The
   deploy runs this and refuses to ship on a gameable check.

## Integration

Two independent readers parse these files and **must agree**:

- `Local_Deployed_Shared/pipeline/export_questions_json.py::CSV_SOURCES` — builds
  the static bank the frontend and offline Pyodide grading use.
- `This-Directory-Only/backend/app/questions.py::load_questions` — builds the bank
  the Fly backend serves and grades against.

Each hard-codes the source list, the order, and the `skip_rows` value. They are
kept in lockstep by comment only. If they ever disagree, the same question gets
a different id in the two banks, which corrupts tags and stored state at once.
The `pipeline/watch.py` invariant check catches a divergence in the *override
layer* list; it does not currently check the CSV source list.

The directory is gitignored (`.gitignore:107`). It still reaches production —
`This-Directory-Only/Dockerfile` COPYs it into the backend image from the local
build context, which honours `.dockerignore`, not `.gitignore`. Files here are
therefore live in prod while being invisible to `git status`.
`curated_additions.csv` is force-added as an exception, because it is the only
copy of the questions written for this course.

## Gotchas

- **`skip_rows: 2` on the numpy export.** That file has two banner rows before
  the header. Changing it shifts every id in the bank.
- **`einops_problems_with_outputs.csv` is preferred if present, else
  `einops_problems.csv`.** Only the latter exists today. Dropping the former in
  changes which file is read — and if its row count differs, every id from 313
  up moves.
- **The `Output` column is not trustworthy.** It was captured from unseeded,
  out-of-harness runs; 85 of 274 stored strings were unreachable by any honest
  solution. `recompute_expected_outputs` re-derives stdout under the real
  grading harness at export time, so the column is effectively advisory.
- **Rows with an empty Question or Subtopic are skipped**, but still consume an
  id. Do not delete blank rows to tidy the file.
- **Not every file here is loaded.** `Export of numpy problems.csv`, `einsum and
  numpy problems.csv`, `deep_research_knowledge_graph_tool.md`,
  `generate_outputs.py` and `generate_einops_outputs.py` are historical inputs
  and one-off generators, retained for provenance. Only the five files in the
  table above are read at build time.

## Recent Changes

- **2026-07-28** — Added `curated_additions.csv` as a fifth, deliberately-last
  CSV source, and registered it in both readers. It holds q480, the independent
  drill for `numpy.ndarray-model`, which gave the course's first KC a real
  faded→independent ladder instead of one faded item and nothing after it.
