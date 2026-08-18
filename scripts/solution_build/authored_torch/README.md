# authored_torch

## Purpose
- The torch-dialect authoring layer for solution notebooks: the May-2026 agent-authored hints and explanations, re-expressed for the bank as it exists *after* the July PyTorch conversion.
- The domain concern is that a solution notebook is teaching material. A notebook can be syntactically fine and still be wrong to publish, because its prose explains an algorithm the question no longer asks for. This folder is where that judgement is recorded, per question.

## Owns
- `layer.jsonl` — one row per bank question that has authoring, and the decision about whether its prose is still usable.
- The distinction between "we have prose" and "we have prose we trust".

## Does NOT own
- The prose itself. `../authored/` holds the original agent batches; this is derived from them and is regenerable.
- The answer. `solution_code` here is copied verbatim from the bank's `answer_code` — the graded contract — and is never translated. See `../README.md`.
- Notebook assembly → `../build_solution_colabs.py`.
- Whether a question exists at all → `This-Directory-Only/backend/app/questions.py`.

## Key Files
- `layer.jsonl`: `{id, solution_code, explanation, hint, needs_authoring, drift_ratio}`. An empty `explanation` or `hint` is a deliberate withholding, not missing data — `build_solution_colabs.py` omits the section rather than stubbing it. Regenerate with `retorch_authored.py`; never hand-edit.

## Data & External Dependencies
- Reads `../dd_questions.json` (the bank snapshot) and `../authored/*.jsonl`.
- Prose renaming borrows `This-Directory-Only/scripts/torchify_np_prose.py::convert`, which only touches symbols torch spells identically — the same conservative table the July lesson conversion used.

## How It Works (Flow)
1. `retorch_authored.py` pairs each authored row with its live bank question.
2. The authored `solution_code` is compared against the live `answer_code`, normalised so dialect alone is not a difference.
3. Prose is renamed; anything still naming a numpy symbol is flagged.
4. Rows whose answer drifted, or whose prose kept numpy residue, ship with that field blanked and `needs_authoring: true`. The audit trail goes to `../retorch_report.json`.

## Invariants & Constraints
- **`solution_code` is copied, never translated.** A translated answer that drifts from the bank is a notebook that disagrees with the grader the learner is being scored by.
- **Withhold, never guess.** Fluent wrong prose is worse than absent prose, because the learner cannot tell it is wrong. Every blank field here is that rule firing.
- **Text similarity is not behavioural equivalence.** `dim=0` → `dim=1` scores ~0.99. Behaviour-bearing tokens are compared separately, with the known-equivalent spellings (`axis=`≡`dim=`, `dtype=bool`≡`dtype=t.bool`) folded first so a mechanical rename is not misread as drift.
- **This folder is gitignored** (`.gitignore:75` covers all of `scripts/solution_build/`). `layer.jsonl` is tracked only because it was `git add -f`'d, and `build_solution_colabs.py` refuses to run without it.

## Extension Points
- A new authoring campaign → add batches under `../authored/` and re-run `retorch_authored.py`; it picks up any id present in the bank.
- Clearing the backlog → author against the ids listed in `../retorch_report.json` under `drifted` and `residue`, write them into an `authored/` batch, re-run. `needs_authoring` flips to false on its own.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Prose that survived a hand-translated answer** — `RESOLVED`
  - When it happens: the July conversion rewrote ~30 drills to a different algorithm because torch cannot spell the numpy function they used (`ogrid`, `nditer`, `apply_along_axis`, `argpartition`, `intersect1d`, ufunc `out=`/`where=`).
  - Symptom: a "Why this works" that confidently explains an approach the reference solution does not take.
  - Root cause: renaming symbols in prose is not the same operation as converting the code it describes.
  - Prevention/fix: compare the answers, withhold on drift. Never batch-rename explanations across a conversion.
  - Status: `RESOLVED` — 75 rows withheld on drift.

- **A coverage gap read as a deletion signal** — `RESOLVED`
  - When it happens: a bank question has no authored prose, so no row is emitted here.
  - Symptom: the build would delete that question's existing notebook, and exit 0.
  - Root cause: pruning keyed on "did this run build it" instead of "does the question exist".
  - Prevention/fix: `build_solution_colabs.py::prune_orphans` keys on bank membership. A missing row means *skip*, never *delete*.
  - Status: `RESOLVED` — caught by a critic pass, covered by a canary check.

## Recent Changes
- 2026-08-18: Folder created. 447 rows; 319 clean, 75 withheld on answer drift, 53 on numpy residue in the prose. 52 bank questions have no authoring at all and are left alone.
