---
kc: einsum.matvec-matmul
title: Matrix-vector and matrix-matrix products
supporting: [einsum.dot-frobenius, numpy.linalg-basics]
new_syntax: []
concepts: [matvec-contract-row, matmul-row-meets-column]
faded: [312, 286, 260]
guided: [262]
independent: [264, 311, 294]
---

## Concept: matrix-vector 'ij,j->i'

You already know the dot: `'i,i->'` lines two vectors up on a shared letter,
multiplies, and sums it away. A matrix-vector product is that SAME move, done
once per row.

Label the axes. A matrix is `(i, j)`; a vector is `(j,)`. Write them side by
side: `'ij,j->i'`. Now read each letter:

- **j appears in BOTH inputs** — it's the shared letter. einsum sums over any
  letter that is shared and does not appear in the output. So j is multiplied
  aligned and summed away. That summing IS a dot: for a fixed row, `Σ_j a[i,j]·v[j]`
  is row i dotted with v.
- **i appears only in the matrix, and in the output** — it's private and kept.
  It picks WHICH row. One surviving letter → one number per row.

So `'ij,j->i'` says: for each row i, dot that row with v. The notation didn't
memorize "matrix times vector" — it fell out of "sum the shared letter, keep
the private one."

## Watch out

- **Which axis is shared depends on the shapes, not habit.** A *vector times a
  matrix* (`v @ M`, shapes `(i,)` and `(i,j)`) shares the FIRST matrix axis:
  `'i,ij->j'`. Don't reflexively write `'ij,j->i'` — check which axis the
  vector's length lines up with, then make THAT the shared letter.

## Worked example

```python
import torch as t

a = t.tensor([[1.0, 2.0],
              [3.0, 4.0]])
v = t.tensor([10.0, 1.0])

# 'ij,j->i': j is shared+dropped (row . v); i is private+kept (one per row).
mv = t.einsum('ij,j->i', a, v)
assert mv.tolist() == [12.0, 34.0]
assert t.equal(mv, a @ v)   # the @-twin: always check while learning
print("'ij,j->i' ->", mv, "| a @ v ->", a @ v)
```

Why: trace entry [0] straight from the rule — i=0 fixed, j ranges:
`Σ_j a[0,j]·v[j] = 1·10 + 2·1 = 12`. That is row 0 dotted with v. Entry [1] is
row 1 dotted with v: `3·10 + 4·1 = 34`. The result is one dot per row — nothing
memorized beyond "sum the shared letter."

## Faded practice

### q312
A permutation matrix `p` (each row is a single 1, the rest 0) times a vector
`v`. It looks like it *shuffles* v rather than dotting anything — but a
permutation matrix is still a matrix, so this is still a matrix-vector product.
Write the spec.

```python starter
import torch as t

def solve(p, v):
    """p @ v — each row of p has one 1, so row i just picks out one entry of v."""
    return t.einsum('_____', p, v)
```

```python solution
import torch as t

def solve(p, v):
    """p @ v — each row of p has one 1, so row i just picks out one entry of v."""
    return t.einsum('ij,j->i', p, v)
```

### q286
Same contraction, arguments the other way round: a single query vector `q` of
length d against a memory matrix `m` of shape (n, d), giving one similarity
per memory row. The example's spec will not transfer letter for letter — the
vector is the FIRST operand here, and the shared axis is d, not the matrix's
first axis. Name the axes of each operand as they actually are, then keep the
one the output needs.

```python starter
import torch as t

def solve(q, m):
    """Length-n vector: entry i is the dot product of q with row i of m."""
    return t.einsum('_____', q, m)
```

```python solution
import torch as t

def solve(q, m):
    """Length-n vector: entry i is the dot product of q with row i of m."""
    return t.einsum('d,nd->n', q, m)
```

## Concept: matrix-matrix 'ik,kj->ij'

Matmul is a matrix-vector product done once per *column* of the second matrix —
i.e. a whole grid of dots, one for every (row, column) pair. The notation says
exactly that.

Label the axes: A is `(i, k)`, B is `(k, j)`. Write them: `'ik,kj->ij'`. Read
the letters:

- **k appears in both inputs and NOT in the output** — shared and dropped, so
  einsum sums over it. k is the "inner" axis, the one A and B have in common; it
  gets contracted away. `Σ_k a[i,k]·b[k,j]` is row i of A dotted with column j
  of B.
- **i appears only in A and survives** — it picks the row.
- **j appears only in B and survives** — it picks the column.

Two survivors → a 2-D result, indexed `[i, j]`. So
`result[i, j] = Σ_k a[i,k]·b[k,j]` = "row i of A meets column j of B", for every
i and j. That's the definition of matmul, read straight off the spec.

The notation even encodes the shape rule: the shared letter k is the inner
dimension that must match and then vanishes; the two survivors i and j become
the output shape `(i, j)`. `(m,k) @ (k,n) → (m,n)` is just "drop the shared
letter, keep the private ones."

## Watch out

- **One transposed letter is a silent, wrong matmul.** `'ik,kj->ij'` contracts
  B's FIRST axis (correct). `'ik,jk->ij'` contracts B's SECOND axis — that's
  `a @ b.T`. Same output shape, different numbers, no error raised. This is THE
  einsum bug. The `@`-twin assert is how you catch it while learning.

## Worked example

```python
import torch as t

a = t.tensor([[1.0, 2.0],
              [3.0, 4.0]])
b = t.tensor([[5.0, 6.0],
              [7.0, 8.0]])

# 'ik,kj->ij': k is shared+dropped (the contraction); i, j are private+kept.
mm = t.einsum('ik,kj->ij', a, b)
assert t.equal(mm, a @ b)   # the @-twin
print("'ik,kj->ij' — k contracted away, i and j kept:")
print(mm)
```

Why: trace entry [0, 0] by the rule — i=0, j=0 fixed, k ranges:
`Σ_k a[0,k]·b[k,0] = 1·5 + 2·7 = 19` = row 0 of A dotted with column 0 of B.
Entry [1, 0] uses row 1 and column 0: `3·5 + 4·7 = 43`. Each output cell is one
row-dot-column; matmul is the whole grid of them.

## Faded practice

### q260
Matrix product of `c` with shape `(1, 2)` and `d` with shape `(2, 1)`. The
result is `(1, 1)` — a single number in a 2-D box. It *looks* like a plain dot
of a row and a column, so it's tempting to write `'i,i->'`. But both inputs are
2-D matrices, so it's a matmul: name every axis and let the survivors decide the
output shape.

```python starter
import torch as t

def solve(c, d):
    """(1,2) @ (2,1) -> (1,1). Two matrices -> matmul, not a bare dot."""
    return t.einsum('_____', c, d)
```

```python solution
import torch as t

def solve(c, d):
    """(1,2) @ (2,1) -> (1,1). Two matrices -> matmul, not a bare dot."""
    return t.einsum('ik,kj->ij', c, d)
```

## Guided practice

### q262
1. Two inputs: a matrix with axes `(i, j)` and a vector with axis `(j,)`. Which
   letter do they share?
2. The shared letter j is summed away (each row dots the vector); i is private
   to the matrix and survives, one result per row.
3. `'ij,j->i'`.

## Independent practice

These COMBINE matmul rather than introduce a new atom — drill them unaided once
the two specs above are automatic:

- q264: a triple product `a @ b @ c` written as ONE einsum — two contraction
  letters chained like dominoes, each adjacent pair sharing exactly one letter.
- q311: `a @ diag(v) @ b` with the diagonal kept as a bare vector in the spec —
  a matmul whose middle factor never gets materialized.

Also from the bank: q294 (contract a conv-shaped (o, i, h, w) weight against
an (i, h, w) input; three axes pair at once).

