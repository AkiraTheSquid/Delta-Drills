---
kc: numpy.linalg-applied
title: Applied linear algebra — multi-RHS, batched solve, block matrices
supporting: [numpy.linalg-basics, numpy.constructors]
new_syntax: []
faded: [173, 208, 190]
guided: []
independent: [197, 118]
---

## Concept: many right-hand sides, one matrix

The np-1 linalg KP solved one system. Real workloads batch.

**Many right-hand sides, one matrix.** `np.linalg.solve(a, B)` accepts an
(n, m) matrix B — each COLUMN is an independent right-hand side, all m
systems solved in one factorization of `a`. This is why "solve for several
b's" should never loop: the expensive part (factorizing a) is shared.

## Worked example

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
```

Why: `a @ X ≈ B` verifies all the numbers in one allclose — the
substitute-back check scales with the batching for free. Keep it.

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

## Concept: batched (stacked) linalg — leading axes broadcast

**Many matrices.** NumPy's linalg functions broadcast over leading axes:
`np.linalg.solve(ms, vs)` with ms of shape (p, n, n) and vs (p, n, 1) solves
p systems at once. The same convention makes `@` batch-multiply stacks.

When a batched product needs a REDUCTION over the batch too (e.g.
Σᵢ Mᵢ @ vᵢ), matmul alone can't say it — either sum the batched products
(`(m @ v).sum(axis=0)`) or reach for `np.einsum`, whose subscript string
('pij,pjk->ik') names exactly which axes pair up and which collapse. Treat
that as a preview: einsum is the next course topic, and batched-linalg
tasks are where it starts paying.

## Worked example

```python
import numpy as np

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

Why: watch the shapes — (p, n, n) and (p, n, 1) share the leading p, and
that agreement is what NumPy broadcasts over. A (p, n) vs would be
interpreted differently (and wrongly); the trailing 1 keeps the vectors as
columns.

## Faded practice

### q208
Σᵢ m[i] @ v[i]: batched matmul, then reduce the batch axis.

```python starter
import numpy as np

def solve(m, v):
    """Sum over the batch of the p products m[i] @ v[i]."""
    return (m @ v).sum(axis=_____)
```

```python solution
import numpy as np

def solve(m, v):
    """Sum over the batch of the p products m[i] @ v[i]."""
    return (m @ v).sum(axis=0)
```

## Concept: diagnostics and structure builders

- **`np.linalg.matrix_rank(z)`** — the number of linearly independent
  rows/columns (as a plain int) — the standard "is this system degenerate?"
  probe. Rank counts INDEPENDENT rows, not nonzero rows: [[1,2],[2,4]] has
  two nonzero rows and rank 1. The function is SVD-based and
  tolerance-aware; let it decide.
- **Block-diagonal assembly**: allocate the zero canvas of summed shape,
  then paste each block with slice assignment at its running offset —
  constructors + slicing doing matrix carpentry (scipy has block_diag, but
  the drills build it by hand on purpose).
- Special matrices from broadcast formulas — e.g. the Cauchy matrix
  `1 / (x[:, None] - y[None, :])` — are the index-grids pattern applied to
  linear algebra.

## Worked example

```python
import numpy as np

# Rank: independent rows, not nonzero rows.
z = np.array([[1.0, 2.0],
              [2.0, 4.0]])          # second row = 2 x first
assert int(np.linalg.matrix_rank(z)) == 1

full = np.array([[1.0, 0.0],
                 [0.0, 1.0]])
assert int(np.linalg.matrix_rank(full)) == 2
```

Why: the [[1,2],[2,4]] example is the canonical trap — visually "two rows
of data", mathematically one direction.

## Faded practice

### q190
Matrix rank as a plain int.

```python starter
import numpy as np

def solve(z):
    """Number of linearly independent rows/columns of z."""
    return int(np.linalg._____(z))
```

```python solution
import numpy as np

def solve(z):
    """Number of linearly independent rows/columns of z."""
    return int(np.linalg.matrix_rank(z))
```

## Independent practice

From the drill bank: q197 (block-diagonal from a list of arbitrary 2-D
blocks — zero canvas + running-offset slice assignment), q118 (the Cauchy
matrix 1/(xᵢ−yⱼ) and its determinant — a broadcast build plus one linalg
call).

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
