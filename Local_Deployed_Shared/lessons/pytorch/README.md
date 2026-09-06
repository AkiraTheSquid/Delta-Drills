# pytorch

## Purpose
- The `tr-1` lesson, "Rays as tensors (ARENA 0.1)": the four knowledge points a learner
  needs before ARENA Chapter 0.1's first exercise, `make_rays_1d`, and that exercise
  itself as the capstone. Authored 2026-09-06 from the supplementary notebook in
  Seth's ARENA fork (`0.1_Ray_Tracing_exercises_with_supplementary_exercises.ipynb`).
- Sits ABOVE the numpy course in the lattice: every prerequisite is a `numpy.*` KC.

## Owns
- `kp-out-argument.md` — `torch.out-argument`: `t.linspace(..., out=view)` / `t.arange(..., out=view)`.
- `kp-slice-assignment.md` — `torch.slice-assignment`: `x[sel] = tensor` (broadcast write) and `copy_`.
- `kp-ray-parametrisation.md` — `raytracing.ray-parametrisation`: O + u·D, the (2, 3) ray slab.
- `kp-make-rays-1d.md` — `raytracing.make-rays-1d`: the (n, 2, 3) fan, zeros then two column writes.
- Drills q798–q841 (44), topic `PyTorch`, subtopic `Rays as tensors`, in
  `This-Directory-Only/csv files of problems/curated_additions.csv` + `chatgpt/curated_overrides.jsonl`.

## Does NOT own
- The registry rows (`../kc_registry.json`, lesson `tr-1`), the glossary terms (`../glossary.js`:
  "ray", "out=", "broadcast write"), the atom graph (`backend/app/data/concept_graphs/
  arena_drillable_v1.json`: atoms `broadcast-slice-assignment`, `make-rays-1d`; reuses
  `linspace-out-param`, `ray-parametric-form`) and the atom tags
  (`backend/app/data/question_atom_tags.jsonl`).
- The ARENA notebook these pages point at — that is the fork mirror
  (`scripts/sync_arena_fork.sh` → `content/ARENA_3.0-fork/`).

## Key Files
- The four `kp-*.md` pages above. Same contract as `../numpy/` (see `../AUTHORING.md`).

## Data & External Dependencies
- Every fence runs under torch (`scripts/validate_lessons.py`, backend venv).
- `new_syntax` here is deliberately small: `torch.arange#out`, `torch.linspace#out`,
  `Tensor.copy_`. The raytracing pages declare none — they are discipline/geometry
  nodes like `einops.*` and `numpy.random-threading`.

## How It Works (Flow)
1. Learner reaches `make_rays_1d` in the in-app ARENA 0.1 notebook and presses
   "Practice this exercise" (practice session shell, `practice/`).
2. The session scope is `raytracing.make-rays-1d` + its lattice ancestors; a wrong
   answer falls back down the rung ladder onto these prerequisite pages.

## Invariants & Constraints
- 🔴 A symbol used in a drill must appear in at least one ARENA notebook
  (`check_arena_grounding_ratchet`). `Tensor.fill_` and `Tensor.repeat_interleave`
  do NOT — they were removed from this lesson on 2026-09-06; do not reintroduce them.
- `out=` with a mismatched target does not raise: PyTorch warns, resizes the view over
  the canvas's storage and writes the values into the WRONG slots (verified 2026-09-06:
  4 values into a 5-slot column land across rows 0–1). The pages teach this; keep it.
  🔴 An earlier draft said "the canvas stays untouched" — codex caught it; it is false.
- Bank starters carry no blanks (`return None`); the hand-blanked faded starters
  live only in the KP fences. `blank_new_syntax` still runs over them at compile.
- Faded ≥ 2 per segment, Solo ≥ 6, Integrated ≥ 3 (met: 2 / 6 / 3 on every page).

## Extension Points
- Next exercises in 0.1 (`intersect_ray_1d`, `intersect_rays_1d`, `make_rays_2d`) get
  their own KCs under `tr-1` with `raytracing.make-rays-1d` as prerequisite. Add the
  registry row, the glossary `kcLesson` line, an atom + edges, tag rows, then run the
  pipeline in `../README.md` order.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Course-order debt on function-mode scaffolds** — `ACTIVE`
  - When it happens: every new drill here is function-mode (`def solve`, `*example`).
  - Symptom: `audit_solution_prereqs.py` reports `syntax.star-args` / `syntax.none` as
    "taught by NO lesson" for each new drill.
  - Root cause: `python.defining-functions` comes later in course order than the
    scaffold every drill uses — the same debt recorded for q568+ (see CLAUDE.md).
  - Prevention/fix: re-record `scripts/solution_prereq_baseline.json` after authoring;
    the real fix is course order, not more drills.
  - Status: `ACTIVE`

## Recent Changes
- 2026-09-06: Lesson created — 4 KPs, 44 drills (q798–q841), atoms and glossary wired.
