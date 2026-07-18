---
kc: einsum.batch-dims
title: Batch dimensions — carrying axes through
supporting: [einsum.matvec-matmul, einsum.outer-products]
new_syntax: []
faded: [265]
guided: [268]
independent: [276, 297, 243, 287, 284]
---

## Concept

Real tensors arrive in batches: (batch, n, m) stacks of matrices, (batch, d)
stacks of vectors. In einsum, batching costs exactly one letter:

> **A letter appearing in every operand AND the output is a batch axis** —
> the whole contraction runs independently at each of its positions.

Take any spec you know and prepend the letter:

- matmul `'ij,jk->ik'` → **batched matmul** `'aij,ajk->aik'`
  (pairs t[0] with u[0], t[1] with u[1], …).
- matvec `'ij,j->i'` → `'bij,bj->bi'` — each matrix times ITS OWN vector.
- outer `'i,j->ij'` → `'bi,bj->bij'` — one outer product per batch item.

Contrast three fates of a batch-like letter, using (b, i, k) × (?, k, j):

1. **In both inputs and output** (`'bik,bkj->bij'`) — parallel independent
   products: the batch.
2. **In one input only, kept** (`'bij,jk->bik'`) — the OTHER operand is
   shared across the batch: one constant matrix applied to every item.
   No tiling, no loop; the letter's absence from the second input does it.
3. **In both inputs but dropped** (`'bi,bj->ij'`) — the results are SUMMED
   over the batch: an aggregated contraction (Σₙ outer(vₙ, vₙ) — the Gram
   accumulation). Dropping the batch letter is how "sum over the dataset"
   enters the spec.

That three-way distinction — parallel / shared / summed — is the entire
content of batching, and the spec states it unambiguously where looped code
buries it.

## Worked example

Task: batched matmul; a constant matrix applied per item; a batch-summed
outer product.

```python
import numpy as np

t = np.arange(12.0).reshape(2, 2, 3)     # (a=2, i=2, j=3)
u = np.arange(18.0).reshape(2, 3, 3)     # (a=2, j=3, k=3)

# 1. PARALLEL: a everywhere -> item-by-item matmul.
batched = np.einsum('aij,ajk->aik', t, u)
assert batched.shape == (2, 2, 3)
assert np.allclose(batched[0], t[0] @ u[0])   # each slice is its own matmul
assert np.allclose(batched[1], t[1] @ u[1])

# 2. SHARED: m has no batch letter -> the same m for every item.
m = np.arange(9.0).reshape(3, 3)
shared = np.einsum('bij,jk->bik', t, m)
assert np.allclose(shared[0], t[0] @ m)
assert np.allclose(shared[1], t[1] @ m)       # same m both times

# 3. SUMMED: batch letter dropped -> aggregate over the batch.
v = np.array([[1.0, 2.0],
              [3.0, 4.0]])                    # (b=2, d=2)
gram = np.einsum('bi,bj->ij', v, v)
by_hand = np.outer(v[0], v[0]) + np.outer(v[1], v[1])
assert np.allclose(gram, by_hand)
```

Why each step:

1. The per-slice asserts (`batched[0] == t[0] @ u[0]`) are the definition of
   "parallel over the batch" — and the test you should write whenever a
   batched spec feels uncertain.
2. In the SHARED case, notice what you did NOT do: no np.tile, no
   broadcasting gymnastics, no loop. Omitting the batch letter from one
   operand IS the sharing.
3. The SUMMED case reads two ways that must agree: algebraically (Σₙ over
   outer products) and mechanically (b shared → multiplied along; b absent
   from output → summed). When both readings match, the spec is right.

## Faded practice

### q265
Item-by-item product of two matrix batches.

```python starter
import numpy as np

def solve(t, u):
    """(a,i,j) x (a,j,k) -> (a,i,k): the matmul spec plus a batch letter."""
    return np.einsum('_____', t, u)
```

```python solution
import numpy as np

def solve(t, u):
    """(a,i,j) x (a,j,k) -> (a,i,k): the matmul spec plus a batch letter."""
    return np.einsum('aij,ajk->aik', t, u)
```

## Guided practice

### q268
1. A batch of matrices times a batch of vectors, pairwise — start from the
   matvec spec 'ij,j->i'.
2. The batch letter goes on BOTH inputs and the output.
3. `'bij,bj->bi'` — check: does each output row depend only on its own
   batch item?

## Independent practice

From the drill bank: q276 (batched matmul again, different letter names —
fluency check), q297 (ONE constant matrix applied to a whole batch — the
shared pattern), q243 (batch of outer products), q287 (batch of SELF outer
products), q284 (per-item sum of squares — a batched dot of each vector
with itself).

## Misconceptions

- **"Batching needs a loop or np.vectorize."** — One letter, present
  everywhere, batches any contraction. The loop exists only in compiled
  code.
- **"All operands must mention every letter."** — An operand OMITTING the
  batch letter is broadcast across the batch — that's the shared-weights
  pattern, not an error. (Only the OUTPUT omitting it changes the math, by
  summing.)
- **"'bi,bj->ij' is a batch of outer products."** — The batch letter is
  dropped, so the outers are SUMMED — one (i, j) matrix, not (b, i, j).
  Keep b in the output ('bi,bj->bij') for the per-item version. One letter
  in the output = the difference between a dataset aggregate and per-sample
  results.
