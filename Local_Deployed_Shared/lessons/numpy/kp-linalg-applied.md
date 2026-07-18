---
kc: numpy.linalg-applied
title: Applied linear algebra — multi-RHS, batched solve, block matrices
supporting: [numpy.linalg-basics, numpy.constructors]
new_syntax: []
faded: [173]
guided: [208]
independent: [190, 197, 118]
---

## Concept

The np-1 linalg KP solved one system. Real workloads batch:

**Many right-hand sides, one matrix.** `np.linalg.solve(a, B)` accepts an
(n, m) matrix B — each COLUMN is an independent right-hand side, all m
systems solved in one factorization of `a`. This is why "solve for several
b's" should never loop: the expensive part (factorizing a) is shared.

**Many matrices — batched (stacked) operations.** NumPy's linalg functions
broadcast over leading axes: `np.linalg.solve(ms, vs)` with ms of shape
(p, n, n) and vs (p, n, 1) solves p systems at once. The same convention
makes `@` batch-multiply stacks. When a batched product needs a REDUCTION
over the batch too (e.g. Σᵢ Mᵢ @ vᵢ), matmul alone can't say it — either
sum the batched products (`(m @ v).sum(axis=0)`) or reach for `np.einsum`,
whose subscript string ('pij,pjk->ik') names exactly which axes pair up and
which collapse. Treat that as a preview: einsum is the next course topic,
and batched-linalg tasks are where it starts paying.

**Structure builders and diagnostics.**

- `np.linalg.matrix_rank(z)` — the number of linearly independent
  rows/columns (as a plain int) — the standard "is this system degenerate?"
  probe.
- **Block-diagonal assembly**: allocate the zero canvas of summed shape,
  then paste each block with slice assignment at its running offset —
  constructors + slicing doing matrix carpentry (scipy has block_diag, but
  the drills build it by hand on purpose).
- Special matrices from broadcast formulas — e.g. the Cauchy matrix
  `1 / (x[:, None] - y[None, :])` — are the index-grids pattern applied to
  linear algebra.

## Worked example

Task: solve a system for three right-hand sides at once; then a batched
solve over a stack of systems.

```python
import numpy as np

a = np.array([[2.0, 0.0],
              [0.0, 4.0]])
B = np.array([[2.0, 4.0, 6.0],
              [8.0, 12.0, 16.0]])       # three RHS columns

# One call, all three systems: column j of X solves a @ x = B[:, j].
X = np.linalg.solve(a, B)
assert X.tolist() == [[1.0, 2.0, 3.0],
                      [2.0, 3.0, 4.0]]
assert np.allclose(a @ X, B)            # verify all three at once

# Batched: p independent systems, stacked on a leading axis.
ms = np.stack([np.eye(2) * 2.0, np.eye(2) * 4.0])   # (2, 2, 2)
vs = np.array([[[2.0], [4.0]],
               [[4.0], [8.0]]])                     # (2, 2, 1)
xs = np.linalg.solve(ms, vs)                        # (2, 2, 1): per-system
assert xs[:, :, 0].tolist() == [[1.0, 2.0], [1.0, 2.0]]

# Batch-product-then-reduce: sum over the batch of m[i] @ v[i].
total = (ms @ vs).sum(axis=0)          # 2*[2,4] + 4*[4,8] = [20, 40]
assert total.tolist() == [[20.0], [40.0]]
# einsum preview — same contraction, named explicitly:
assert np.array_equal(np.einsum('pij,pjk->ik', ms, vs), total)
```

Why each step:

1. `a @ X ≈ B` verifies THIRTY numbers in one allclose — the substitute-back
   check scales with the batching for free. Keep it.
2. In the batched solve, watch the shapes: (p, n, n) and (p, n, 1) share the
   leading p — that agreement is what NumPy broadcasts over. A (p, n) vs
   would be interpreted differently (and wrongly); the trailing 1 keeps the
   vectors as columns.
3. The two spellings of Σ mᵢ @ vᵢ — batched-matmul-then-sum vs einsum —
   compute identically; the einsum string documents the intent (`p` appears
   on both inputs but NOT the output = summed over). When you meet einsum
   formally, this example is your anchor.

## Faded practice

### q173
One matrix, m right-hand sides (columns of b), all solved simultaneously.

```python starter
import numpy as np

def solve(a, b):
    """(n, m) solution X: column j solves a @ x = b[:, j]."""
    return np.linalg._____(a, b)
```

```python solution
import numpy as np

def solve(a, b):
    """(n, m) solution X: column j solves a @ x = b[:, j]."""
    return np.linalg.solve(a, b)
```

## Guided practice

### q208
1. A stack of p products m[i] @ v[i], SUMMED over the batch — batched matmul
   handles the products; what reduces the p axis?
2. `(m @ v).sum(axis=0)` — or the einsum whose output subscripts omit p.
3. Check shapes: (p,n,n) @ (p,n,1) → (p,n,1); summing axis 0 leaves (n,1).

## Independent practice

From the drill bank: q190 (matrix rank as a plain int), q197 (block-diagonal
from a list of arbitrary 2-D blocks — zero canvas + running-offset slice
assignment), q118 (the Cauchy matrix 1/(xᵢ−yⱼ) and its determinant — a
broadcast build plus one linalg call).

## Misconceptions

- **"Multiple right-hand sides need a loop over solve."** — solve accepts
  the (n, m) RHS matrix directly and shares the factorization. The loop
  costs m factorizations for the price of... m factorizations. Don't.
- **"Batched linalg means vectorize=True somewhere."** — It's implicit:
  leading axes broadcast, in solve, inv, det, and @ alike. Stack your
  problems, call once.
- **"rank(a) = number of nonzero rows."** — Rank counts linearly INDEPENDENT
  rows — [[1,2],[2,4]] has two nonzero rows and rank 1. Let
  `np.linalg.matrix_rank` (SVD-based, tolerance-aware) decide.
