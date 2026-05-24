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
- `prereqs_einops/einops-reduce.ipynb`: drills 5 patterns for `einops.reduce` (channel mean → global mean → keepdim max → 2×2 avg-pool → softmax stabilize). Bridges to bank subtopic `"Einops: Reduce"`.
- `prereqs_einops/einops-repeat.ipynb`: drills 5 patterns for `einops.repeat` (broadcast → per-token → vertical stretch → horizontal tile → 2×2 nearest upsample). Bridges to bank subtopic `"Einops: Repeat"`.
- `prereqs_einops/einops-einsum.ipynb`: drills 5 patterns for `einops.einsum` (Hadamard → matmul → row sum → batched matmul → attention QK^T). Bridges to bank subtopic `"Einops: Deep Learning"` (einsum problems live in that subtopic in the bank).
- `prereqs_numpy/vector-normalisation.ipynb`: drills 5 patterns for unit-vector arithmetic (L2 norm → unit → keepdim → batch normalize → safe normalize). Bridges to bank subtopic `"Numpy: Applied patterns and advanced"`.
- `prereqs_numpy/softmax-from-logits.ipynb`: drills 5 numerically-stable softmax patterns (naive → subtract-max → row-wise → log-softmax → stable CE). Bridges to bank subtopic `"Numpy: Applied patterns and advanced"`.
- `prereqs_numpy/broadcasting-rules.ipynb`: drills 5 broadcasting patterns (shape prediction → row broadcast → column broadcast → axis insertion → outer product). Bridges to bank subtopic `"Numpy: Vectorization and broadcasting"`.
- `prereqs_numpy/as-strided-noncontig-source.ipynb`: drills 5 stride / contiguity patterns (read strides → transpose breaks contig → view fails → contiguous fixes → as_strided rolling window). Bridges to bank subtopic `"Numpy: Applied patterns and advanced"`.
- `prereqs_numpy/tensor-zeros-init.ipynb`: drills 5 allocation patterns (1-D zeros → multi-axis → `zeros_like` shape+dtype mirror → `dtype=long` index buffer → allocate-then-scatter for the canonical Ray Tracing per-ray output buffer). Bridges to bank subtopic `"Numpy: Core array literacy"`.
- `prereqs_numpy/tensor-unbind.ipynb`: drills 5 axis-peel patterns (default dim → explicit dim → equivalence with `select` → tuple destructure → ray-equation evaluate `o + t·d` via `(N, 2, 3).unbind(dim=1)`). Bridges to bank subtopic `"Numpy: Indexing and selection"`.
- `prereqs_numpy/boolean-mask-identity-replace.ipynb`: drills 5 mask-and-substitute patterns (mask from compare → masked scalar write → row-zero via 1-D mask → identity substitute in `(B, N, N)` → safe batched solve where singular slots come out zero). Bridges to bank subtopic `"Numpy: Indexing and selection"`.
- `prereqs_numpy/rotation-matrix-3d-y-axis.ipynb`: drills 5 Y-axis rotation patterns (`cos`/`sin` → assemble `R_y` → rotate a vector → composition law `R(α)·R(β) = R(α+β)` → rotate a `(N, 3)` batch via `points @ R.T`). Right-hand convention. Bridges to bank subtopic `"Numpy: Applied patterns and advanced"`.
- `prereqs_numpy/tensor-item-scalar.ipynb`: drills 5 `.item()` patterns (0-D extract → single-elem-any-shape → dtype preservation across float/int/bool → `.item()` vs `.tolist()` branching → tensor → Python control flow via reduce-then-`.item()`). Bridges to bank subtopic `"Numpy: Core array literacy"`.
- `scripts/build_*.py`: per-drill builders. Each calls the shared `verify_solutions()` gate before writing. Re-run after editing.

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
- **Builder must self-verify before emitting the notebook.** Each `scripts/build_*.py` calls `verify_solutions(EXERCISE_SPECS)` immediately before writing the `.ipynb`. The function execs each `solution_body` against its `test_body`; any shape/value/import/syntax failure aborts the build (`SystemExit(1)`) and the notebook is NOT written. Requires `torch + numpy + einops` in the build env. Catches stub/solution drift, broken test ground-truth, and assertion mismatches — does NOT catch one-liner-cheats that happen to pass.

## Extension Points
- **New atom drill:** copy `scripts/build_einops_rearrange.py` → swap `ATOM_ID`, `SUBTOPIC`, exercises, solutions. Make sure the atom-id resolves via `atom_readiness.js` (token-bridge or topic-alias) or has signal in `subtopic_states` directly. The build-time `verify_solutions()` gate (see Invariants) will catch most authoring slips before the notebook ships.
- **New topic folder:** mirror `prereqs_einops/` layout. Topic folder name matches the modal bank topic (e.g. `prereqs_numpy/`, `prereqs_einsum/`).
- **Difficulty ladder:** keep 5 exercises ramping from ⚪⚪⚪⚪⚪ → 🔴🔴🔴🔴⚪. Last exercise should be the closest analog to a real ARENA use of the atom.

## Recent Changes
- 2026-05-23: Initial drill `einops-rearrange.ipynb` shipped. Defines the procedural-drill template for all future atoms.
- 2026-05-24: Added build-time `verify_solutions()` gate to `build_einops_rearrange.py`. Builder execs every canonical solution against its in-notebook test and aborts before write on any failure. Sabotage test confirmed: wrong `'b c -> c b'` solution raises `AssertionError('shape mismatch: torch.Size([4, 3]) vs torch.Size([3, 4])')` → SystemExit(1), notebook untouched.
- 2026-05-24: Shipped `einops-reduce.ipynb`, `einops-repeat.ipynb`, `einops-einsum.ipynb` — completes the 4-atom einops family. All inherit the verify gate (5/5 pass per drill). einsum drill bridges to `Einops: Deep Learning` rather than a per-pattern subtopic (the bank stores einsum problems in that catch-all sub).
- 2026-05-24: Shipped first 4 non-einops drills in `prereqs_numpy/`: `vector-normalisation.ipynb`, `softmax-from-logits.ipynb`, `broadcasting-rules.ipynb`, `as-strided-noncontig-source.ipynb` — covers the Ray-Tracing / Backprop / CNN prereq subset that token-bridges to topic `Numpy`. CNN-specific atoms (`conv-output-size-formula`, `conv2d-module`, etc.) were skipped because the question bank has no CNN subtopic — drilling them would create orphan EWMA state. Verify gate caught one authoring slip mid-build (stale `import pytest_simulated` line in a test_body) — fixed before notebook emitted, exactly the failure mode the gate is designed to prevent.
- 2026-05-24: Shipped 3 more Ray Tracing prereq drills in `prereqs_numpy/`: `tensor-zeros-init.ipynb` (8 plinks — most-linked Ray Tracing atom; allocation patterns for per-ray output buffers, bridges to `Numpy: Core array literacy`), `tensor-unbind.ipynb` (5 plinks — peeling axes into named slices, integrative ex5 evaluates the parametric ray equation `o + t·d`, bridges to `Numpy: Indexing and selection`), `boolean-mask-identity-replace.ipynb` (5 plinks — culminates in `safe_solve` that swaps singular `A` slices with identity + zeros the matching `b` rows so a batched `linalg.solve` doesn't die on one bad sample; bridges to `Numpy: Indexing and selection`). All three passed verify gate first try.
- 2026-05-24: Shipped final 2 Numpy-bridged Ray Tracing prereq drills: `rotation-matrix-3d-y-axis.ipynb` (4 plinks — Y-axis rotation matrix construction, vector rotation, composition law, batch rotation via `points @ R.T`; right-hand convention; bridges to `Numpy: Applied patterns and advanced`) and `tensor-item-scalar.ipynb` (4 plinks — `.item()` extraction across dtypes + branching `.item()` vs `.tolist()` + reduce-then-`.item()` for Python control flow; bridges to `Numpy: Core array literacy`). 13 procedural drills total (4 einops + 9 numpy = 65 exercises = 65 KCs). Easy Numpy-bridged Ray Tracing atoms now exhausted — next high-yield direction is authoring CNN bank questions to unlock `conv-output-size-formula` (20 plinks), `conv2d-module` (14), etc.
