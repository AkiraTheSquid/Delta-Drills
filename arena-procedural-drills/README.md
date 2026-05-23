# arena-procedural-drills

## Purpose
Atom-keyed procedural drill notebooks for ARENA chap-0 prerequisite concepts. When a learner's `computeAtomReadiness(atom_id)` for an atom is low, the Delta Drills "stuck → drill" loop surfaces the matching notebook here. Sibling to `arena-book-colab/` (which is ARENA-exercise-keyed).

## Owns
- `.ipynb` files keyed by iter-5 v2 atom-id (e.g. `einops-rearrange.ipynb`).
- The procedural-drill notebook **template** — header layout, Delta Drills auth cell, exercise/solution/assertion pattern, completion beacon.
- Builder scripts under `scripts/` that emit the notebooks from canonical Python source.

## Does NOT own
- ARENA-exercise notebooks → `arena-book-colab/`.
- The atom → bank-subtopic bridge → `Local_Deployed_Shared/concept-graph/atom_readiness.js`.
- The concept graph itself → `This-Directory-Only/backend/app/data/concept_graphs/arena_iter5_v2.json`.
- The `arena-rating` endpoint that consumes beacons → `backend/app/practice/arena_rating_router.py`.

## Key Files
- `prereqs_einops/einops-rearrange.ipynb`: drills 5 patterns for `einops.rearrange` (identity → swap → flatten → unfold → patchify). Bridges to bank subtopic `"Einops: Rearrange"`.
- `scripts/build_einops_rearrange.py`: regenerates the notebook from canonical source. Re-run after editing.

## Data & External Dependencies
- Atom-ids must exist in `concept_graphs/arena_iter5_v2.json`.
- Beacon POSTs to `/api/practice/arena-rating` on the Delta Drills backend with bridged subtopic name(s).
- Notebook deps: `torch`, `einops`, `numpy` (Colab-compatible).

## How It Works (Flow)
1. User clicks "Drill this concept" in Delta Drills frontend → opens notebook on Colab.
2. User pastes `DD_TOKEN`, runs exercises top-to-bottom.
3. Each assertion adds its tag to `_dd_passed`. Final cell calls `report_completion()`.
4. If all 5 exercises pass + token present, POST → `/api/practice/arena-rating` → backend bumps EWMA for the bridged subtopic via `apply_feedback`.
5. Next time `computeAtomReadiness(atom_id)` runs, the higher subtopic baseline lifts the atom score.

## Invariants & Constraints
- **Atom-id in metadata must match the filename stem.** `prereqs_einops/einops-rearrange.ipynb` → `metadata.delta_drills.atom_id == "einops-rearrange"`.
- **Subtopic in metadata must exist in the question bank.** Otherwise `arena-rating` accepts the name silently but no future bank question maps to it → orphan EWMA state.
- **All assertions must verify behavior, not pattern strings.** Compare against `torch.reshape` / `x.T` / hand-built ground truth — never `rearrange == rearrange`.
- **Solutions live in `<details>` markdown blocks.** Never as runnable code cells (would auto-overwrite the student's stub).
- **The completion beacon fires only when `_dd_passed == _DD_REQUIRED`.** Partial completion never reports.

## Extension Points
- **New atom drill:** copy `scripts/build_einops_rearrange.py` → swap `ATOM_ID`, `SUBTOPIC`, exercises, solutions. Make sure the atom-id resolves via `atom_readiness.js` (token-bridge or topic-alias) or has signal in `subtopic_states` directly.
- **New topic folder:** mirror `prereqs_einops/` layout. Topic folder name matches the modal bank topic (e.g. `prereqs_numpy/`, `prereqs_einsum/`).
- **Difficulty ladder:** keep 5 exercises ramping from ⚪⚪⚪⚪⚪ → 🔴🔴🔴🔴⚪. Last exercise should be the closest analog to a real ARENA use of the atom.

## Recent Changes
- 2026-05-23: Initial drill `einops-rearrange.ipynb` shipped. Defines the procedural-drill template for all future atoms.
