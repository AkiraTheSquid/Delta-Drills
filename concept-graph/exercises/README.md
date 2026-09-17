# exercises

## Purpose
Per-exercise atom tags for every ARENA notebook in the curriculum. Each `0_<part>_<index>.json` file maps one notebook to the set of atoms it teaches, with role + evidence quote per atom.

## Owns
- The mapping notebook → atoms (which atoms a given exercise teaches).
- The role distinction per tag — `core` (the lesson) vs `incidental` (used but not taught).
- The evidence quote per tag — short snippet from the notebook justifying the tag.
- The seed-FP record (`atoms_in_seed_but_not_actually_present`) used in calibration reports.

## Does NOT own
- The atom vocabulary — lives in `../vocab/atoms.json`.
- The prereq DAG — lives in `../vocab/prereqs.json`.
- Calibration reports — co-located here (`CALIBRATION_REPORT*.md`) but they're outputs of the tagging passes, not the tag data itself.

## Key Files
- `0_<part>_<index>.json`: one tagged exercise. Schema `{exercise_id, title, source_notebook, arena_chapter, arena_part, arena_index, difficulty, importance, atoms: [{id, role, evidence}], atoms_in_seed_but_not_actually_present, tagged_by, tagged_at, notes}`.
- `CALIBRATION_REPORT_PART<n>.md`: per-part stats (mean atoms/exercise, FP rate, DD coverage breakdown).
- `watch.py`: schema + atom-ref check, runs on every edit.

## Data & External Dependencies
- Reads `../vocab/atoms.json` to resolve atom ids in tags.
- Source notebooks live outside this repo (path recorded per file in `source_notebook`).
- 79 files for chapter 0; chapter 1+ will extend this list.

## How It Works (Flow)
1. Tagger reads the notebook end-to-end.
2. Tagger writes a new `0_<part>_<index>.json` with one atom entry per concept taught (`role: core`) or used (`role: incidental`).
3. `watch.py` (and `../scripts/validate.py`) verifies every `atoms[].id` exists in `../vocab/atoms.json`.
4. Calibration report aggregates per-part stats once the batch is done.

## Invariants & Constraints
- Every `atoms[].id` must resolve to an atom in `../vocab/atoms.json`. Hard error otherwise.
- Every tag must have `id`, `role ∈ {core, incidental}`, non-empty `evidence`.
- No duplicate atom ids inside a single exercise's `atoms` array.
- `arena_chapter`/`arena_part`/`arena_index` form a unique tuple; filename `0_<part>_<index>.json` reflects this.

## Extension Points
- New exercise: drop a new `0_<part>_<index>.json` following the schema; run `python ../scripts/validate.py`.
- New chapter: bump filename prefix from `0_` to `1_`; atoms vocab will extend as new ids appear.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Seed atoms tagged without notebook evidence** — `RESOLVED`
  - When it happens: copying part2's seed-extrapolated atom list into tags before reading the notebook.
  - Symptom: high FP rate (part2 batch 2 ran at 0.83 FP).
  - Root cause: topic-extrapolated seed vocab.
  - Prevention/fix: hand-draft seed from a full notebook read (parts 1/3/4/5 ran at 0.20-0.33 FP). Always include a literal quote in `evidence`.

## Recent Changes
- 2026-05-19: watch.py filled with schema + atom-ref invariants.
- 2026-05-18: 79 chapter-0 exercises tagged end-to-end across 5 parts.
