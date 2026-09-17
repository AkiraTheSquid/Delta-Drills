# scripts

## Purpose
Builder scripts that generate procedural-drill `.ipynb` files from canonical Python source. Keep the notebooks themselves machine-generated — edit the builder, re-run, commit both.

## Owns
- One `build_<atom-id>.py` per drill notebook. Each script is self-contained: imports nothing from sibling scripts, produces exactly one `.ipynb`.
- Notebook structure helpers (`md()`, `code()` inline in each builder).

## Does NOT own
- The notebooks themselves → `../prereqs_<topic>/`.
- Solution-validation logic — each builder validates its own solutions via a smoke-run in the docstring, not as a separate script.

## Key Files
- `build_einops_rearrange.py`: emits `prereqs_einops/einops-rearrange.ipynb`. Five exercises with hardcoded test harness.

## Data & External Dependencies
- No external data files — all exercise tensors are constructed inline.
- Builder deps: stdlib only (`json`, `pathlib`).
- Generated notebook deps: `torch`, `einops`, `numpy` (only relevant at notebook execution time, not at build time).

## How It Works (Flow)
1. `python build_<atom>.py` constructs the cell list as a Python `dict`.
2. Each cell gets a stable `cell-NN` id so `nbformat.validate` doesn't warn.
3. Writes `json.dumps(nb, indent=1)` to `../prereqs_<topic>/<atom>.ipynb`.
4. Smoke-test solutions by running them as Python (see the standalone solution-check pattern at the bottom of `build_einops_rearrange.py`).

## Invariants & Constraints
- **Builder scripts are the source of truth.** Never hand-edit the generated `.ipynb` — your edits will be erased on re-build.
- **Cell IDs must be stable** (use `cell-NN` index, not random) so re-builds produce minimal diffs.
- **`metadata.delta_drills` block must include `atom_id`, `subtopic`, `drill_kind`, `template_version`.** The frontend reads these to route the notebook.

## Extension Points
- Add a new drill: copy `build_einops_rearrange.py` → rename → swap `ATOM_ID`, `SUBTOPIC`, `TITLE` constants → rewrite the exercise cells. Solutions section at the bottom should also be updated to keep the smoke-test honest.

## Recent Changes
- 2026-05-23: `build_einops_rearrange.py` ships.
