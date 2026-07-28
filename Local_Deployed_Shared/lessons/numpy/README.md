# numpy

## Purpose
- The array-programming knowledge-point lessons (44 KPs across registry lessons np-1..np-4): first-encounter teaching for every array KC in the drill course.
- **Mid-migration to the PyTorch dialect.** np-1 ("Arrays from the ground up", 12 KPs), np-2 ("Indexing and selection", 11 KPs) and np-3 ("Vectorization and broadcasting", 11 KPs) are converted: they teach `import torch as t`, matching ARENA 0.0, which the 64-KC graph mirrors. Only np-4 ("Applied patterns", 9 KPs) still teaches NumPy. The folder name is historical.

## Owns
- One `kp-<slug>.md` per NumPy KC — concept prose, worked examples, faded/guided/independent exercise assignments, misconceptions.

## Does NOT own
- Format rules and registry (`../AUTHORING.md`, `../kc_registry.json`), tooling (`scripts/` at repo root), einsum/einops content (sibling folders).

## Key Files
- `kp-ndarray-model.md`, `kp-constructors.md`, `kp-ranges.md`, `kp-dtype-astype.md`: np-1 openers — the pilot of the strict segment rhythm (one concept → one worked example → learner solves one) awaiting Seth's learner review.
- `kp-diag-triangles.md`: reference example of a 4-segment split (extract diag / trace / build diag / triangles).
- Remaining `kp-*.md`: mix of restructured (segments) and legacy single-segment files pending conversion.

## Data & External Dependencies
- Exercise ids reference `../../questions_structured.json`; segment structure is validated/compiled by `scripts/validate_lessons.py` / `compile_lessons.py`.

## How It Works (Flow)
1. Edit a KP following `../AUTHORING.md` (repeatable `## Concept:` segments).
2. Validate (`--coverage`) then compile; the app's lesson gate pages through segments.

## Invariants & Constraints
- **One dialect per page, and it must match the drills it fades into.** A page teaching `np.repeat` while its exercise grades `t.repeat_interleave` still passes every test and simply confuses the learner, so `watch.py` compares each KP's fence imports against the `answer_code` of every question id in its frontmatter. Convert a lesson and its bank questions in the same pass.
- One concept per segment; exactly ONE worked example per segment; every segment has a faded exercise whose solution passes bank tests.
- Concept prose teaches the GENERAL procedure before the example (never example-only).
- Fences execute top-to-bottom in a shared namespace per file — later segments may rely on earlier definitions, so don't reorder segments without re-validating.

## Extension Points
- Splitting a legacy KP: reuse its guided/independent bank ids as per-segment faded exercises (dump contracts with the scratchpad `dumpq.py`/`avail.py` helpers or read the bank directly); update frontmatter lists to match.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Faded starter written against assumed contract** — `ACTIVE`
  - When it happens: promoting a guided/independent qid to faded without reading its test_cases.
  - Symptom: validator FAILs the solution (e.g. q64's tuple order is (allclose, array_equal), not the "obvious" order).
  - Root cause: question text/tests differ from what the prose suggests.
  - Prevention/fix: always dump the bank question's test_cases first; let the validator be the gate.

- **Exact float equality in a torch fence** — `ACTIVE`
  - When it happens: writing `assert x.tolist() == [0.1, 0.4, 0.9]` in a converted KP.
  - Symptom: `validate_lessons.py` reports `fence failed: AssertionError` with no further detail; the code is correct.
  - Root cause: torch's default float is **float32**, so `0.4` round-trips as `0.4000000059604645`. NumPy's float64 hid this.
  - Prevention/fix: use dyadic values in worked examples (0.25, 0.5, 0.75, 3.5, 9.5 — exact in binary), or assert with a tolerance. The grader itself is tolerant; only the lesson fences use bare `==`.

## Recent Changes
- 2026-07-28: np-2 and np-3's 22 KPs converted to the PyTorch dialect alongside their 120 bank questions. Several pages changed in SUBSTANCE, not just spelling, because the torch behaviour is the opposite of what they taught: `kp-centering` (torch `std` divides by n−1, NumPy by n — the population version now needs `correction=0`), `kp-inplace-out` (torch marks in-place with a trailing underscore, and `x.sort()` is NOT in place — there is no `sort_`), `kp-nonzero-argwhere` (torch's default is coordinate rows; the NumPy tuple layout needs `as_tuple=True`), `kp-topk-selection` (no partition/argpartition — `t.topk` returns values AND indices, already sorted), `kp-where-select` and `kp-rescaling` (no ufunc `where=` — masked assignment instead), `kp-sliding-windows` (`Tensor.unfold`, no stride_tricks or `convolve`), `kp-pad-borders` (`t.nn.functional.pad` takes LAST-dimension-first pairs), `kp-index-grids` (no `ogrid`), `kp-cumulative-diff` (`cummax`/`cummin` instead of `ufunc.accumulate`; `cumsum` requires `dim`) and `kp-unique` (no `intersect1d`/`union1d`/`setdiff1d` — compose `unique` with `isin`). q65 was dropped from `kp-inplace-out`'s independent list: `ndarray.flags.writeable` has no torch equivalent, so that one drill stays NumPy.
- 2026-07-27: np-1's 12 KPs converted to the PyTorch dialect alongside their 49 bank questions. Several KPs gained torch-specific content the NumPy version had no reason to teach: `t.flip` (negative slice steps are rejected), `.repeat` vs `repeat_interleave` (the names swap meaning against NumPy), `t.sort` returning a (values, indices) pair, `meshgrid`'s mandatory `indexing=`, and float32-vs-float64 defaults. `watch.py` filled with the lesson↔drill dialect check.
- 2026-07-19: ~20 KPs restructured into single-concept segments; np-1 openers trimmed to one-worked-example rhythm (pilot for Seth's review).
- 2026-07-15: Initial 33 KPs authored (Pass 1).
