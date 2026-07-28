---
kc: einsum.matrix-forms
title: Matrix forms — quadratic, Gram, covariance
supporting: [einsum.matvec-matmul, einsum.batch-dims, numpy.centering]
new_syntax: []
faded: [252]
guided: [258]
independent: [259, 248, 306]
---

## Concept

Classical matrix expressions from statistics and geometry compress into
short specs — this KP is a tour of the ones the drills (and ML papers) use,
as translation practice between math notation and einsum:

- **Quadratic form** xᵀWx = Σᵢⱼ xᵢWᵢⱼxⱼ → `'i,ij,j->'` — three operands,
  every letter dropped: a fully contracted scalar. The spec is literally
  the double sum with the Σs removed.
- **Gram matrix** XᵀX for X of shape (n, d): entry [p, q] = column p ·
  column q = Σₙ XₙₚXₙₙq → `'nd,ne->de'` — the same tensor twice, the shared
  observation axis n contracted, the two FEATURE axes kept under different
  letters (d, e — same size, distinct roles).
- **Sample covariance**: the Gram of the COLUMN-CENTERED data, divided by
  n−1 — `t.einsum('nd,ne->de', xc, xc) / (n - 1)` where
  `xc = x - x.mean(axis=0)`. einsum does the contraction; centering
  (np-3) and the 1/(n−1) live outside — the familiar division-outside rule.
- **Pairwise row dots** between two sets: `'nd,md->nm'` — the linear-algebra
  core of similarity matrices (np-4's cosine, pre-normalization).
- **Batch aggregates**: Σₙ xₙxₙᵀ → `'bi,bj->ij'` (drop the batch letter);
  per-item quadratic forms vᵢᵀMvᵢ for a stack of vectors →
  `'bi,ij,bj->b'` — b kept, i and j contracted against a SHARED M.

The through-line: **repeated math indices = repeated einsum letters;
summation signs = letters missing from the output.** Any Σ-expression you
can write on paper transliterates directly. When you meet an unfamiliar
matrix identity, writing its einsum is often the fastest way to both
understand and implement it.

## Worked example

Task: a quadratic form, a Gram matrix, and a covariance — each checked
against its classical spelling.

```python
import torch as t

x = t.tensor([1.0, 2.0])
w = t.tensor([[3.0, 1.0],
              [0.0, 2.0]])

# x^T W x: the double sum, fully contracted.
qf = t.einsum('i,ij,j->', x, w, x)
assert qf == float(x @ w @ x)
assert qf == 13.0        # 1*3*1 + 1*1*2 + 2*0*1 + 2*2*2

# Gram matrix of columns: same tensor twice, observation axis contracted.
data = t.tensor([[1.0, 10.0],
                 [2.0, 20.0],
                 [3.0, 30.0]])          # (n=3, d=2)
gram = t.einsum('nd,ne->de', data, data)
assert t.allclose(gram, data.T @ data)
assert gram[0, 1] == 1*10 + 2*20 + 3*30    # col 0 . col 1

# Sample covariance: center columns first, contract, divide by n-1.
xc = data - data.mean(dim=0)
cov = t.einsum('nd,ne->de', xc, xc) / (data.shape[0] - 1)
assert t.allclose(cov, t.cov(data.T))
```

Why each step:

1. Expanding the quadratic form by hand once (all four terms) demystifies
   the three-operand spec: each (i, j) pair contributes xᵢWᵢⱼxⱼ; einsum
   enumerates and sums them. The `x @ w @ x` twin confirms it.
2. In the Gram spec, the deliberate oddity is d vs e for axes of the SAME
   tensor: einsum needs distinct names for distinct roles (output row vs
   column), even when sizes coincide. 'nd,nd->dd' would instead walk a
   diagonal — wrong operation.
3. The covariance assembles three lessons — centering (np-3), the Gram
   contraction, division outside — and lands exactly on `t.cov`. Building
   library functions from primitives, then checking against the library, is
   the final form of validate-don't-assert.

## Faded practice

### q252
The quadratic form xᵀWx.

```python starter
import torch as t

def solve(x, w):
    """Sum_ij x_i W_ij x_j — fully contracted."""
    return t.einsum('_____', x, w, x)
```

```python solution
import torch as t

def solve(x, w):
    """Sum_ij x_i W_ij x_j — fully contracted."""
    return t.einsum('i,ij,j->', x, w, x)
```

## Guided practice

### q258
1. XᵀX, entry [p, q] = column p dotted with column q — which axis is shared
   between the two copies of x, and which axes survive?
2. The two surviving axes need DIFFERENT letters even though both come from
   x's column axis.
3. `'nd,ne->de'` — check one off-diagonal entry against a hand dot.

## Independent practice

From the drill bank: q259 (sample covariance — centering + Gram + n−1),
q248 (all pairwise row dots between two matrices), q306 (a quadratic form
per batch item against one shared matrix — 'bi,ij,bj->b').

q310 (Σ of self outer products over a batch) teaches the same spec from the
batching side — a batch letter dropped from the output — so `einsum.batch-dims`
owns it as that segment's faded exercise. A question can be claimed by exactly
one KP; two claims abort `build_qmatrix.py`.

## Misconceptions

- **"Two axes of the same size should share a letter."** — Letters encode
  ROLE, not size. The Gram's output row and column both come from x's
  feature axis but must be d and e; sharing the letter would compute a
  diagonal instead. Size-match is necessary for sharing, never sufficient.
- **"Covariance is an einsum one-liner."** — The CONTRACTION is; centering
  and the 1/(n−1) are not (einsum neither subtracts nor divides). Three
  short lines, each doing one thing.
- **"Math-to-einsum translation is ad hoc."** — It's mechanical: subscripts
  become letters, Σ over an index = omit that letter from the output,
  independent products = separate operands. Practice the transliteration
  direction math→spec; the reverse (reading) then comes free.
