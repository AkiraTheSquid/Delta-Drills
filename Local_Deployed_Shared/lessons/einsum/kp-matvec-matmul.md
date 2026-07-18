---
kc: einsum.matvec-matmul
title: Matrix-vector and matrix-matrix products
supporting: [einsum.dot-frobenius, numpy.linalg-basics]
new_syntax: []
faded: [262, 260]
guided: [264]
independent: [312, 311]
---

## Concept

With dots (shared+dropped) and outers (unshared+kept) in hand, the classical
products are just MIXTURES — some letters contract while others survive:

- **Matrix–vector** `'ij,j->i'`: j is shared and dropped (each row dots the
  vector); i survives (one result per row). Exactly `a @ b` for (i,j)@(j,).
- **Matrix–matrix** `'ik,kj->ij'`: k shared+dropped (the contraction), i
  and j each private to one input and kept — every row of the first meets
  every column of the second. This is THE matmul spec; note how it encodes
  the (m,k)@(k,n)→(m,n) shape rule: the shared letter is the vanishing
  inner dimension.
- **Chains in one call**: `'ik,km,mp->ip'` — three inputs, two contraction
  letters. einsum contracts the whole chain a @ b @ c at once, and (a real
  advantage) chooses a good multiplication ORDER internally via
  `optimize=True` when arrays are large.
- **Structured middles**: a diagonal matrix in the middle of a product
  doesn't need materializing — `'ij,j,jk->ik'` computes a @ diag(v) @ b
  with v staying a vector: the shared j scales each column/row pair
  directly. Cheaper and clearer than building diag(v).

The reading discipline for ANY multi-input spec: for each letter ask
(1) which inputs carry it, (2) does it survive? Contract the dead ones,
distribute the live ones. Ten seconds per spec, no memorization.

## Worked example

Task: matvec, matmul, and a three-matrix chain — each cross-checked against
`@`.

```python
import numpy as np

a = np.array([[1.0, 2.0],
              [3.0, 4.0]])
v = np.array([10.0, 1.0])

# 'ij,j->i': j contracts (row . vector), i survives.
mv = np.einsum('ij,j->i', a, v)
assert mv.tolist() == [12.0, 34.0]
assert np.array_equal(mv, a @ v)

# 'ik,kj->ij': the matmul. k is the inner dimension — shared, dropped.
b = np.array([[5.0, 6.0],
              [7.0, 8.0]])
mm = np.einsum('ik,kj->ij', a, b)
assert np.array_equal(mm, a @ b)

# Chain a @ b @ c in one spec: two contraction letters, k and m.
c = np.array([[1.0, 0.0],
              [0.0, -1.0]])
chain = np.einsum('ik,km,mp->ip', a, b, c)
assert np.array_equal(chain, a @ b @ c)

# Diagonal middle without building the matrix: a @ diag(d) @ b.
d = np.array([2.0, 3.0])
diag_mid = np.einsum('ij,j,jk->ik', a, d, b)
assert np.array_equal(diag_mid, a @ np.diag(d) @ b)
```

Why each step:

1. Every einsum here is asserted against its `@` twin — while learning,
   ALWAYS write the check; the specs that pass silently wrong (transposed
   letters!) are the dangerous ones.
2. In the matmul spec, trace entry [0, 0] by the rules: i=0, j=0 fixed,
   k ranges → Σₖ a[0,k]·b[k,0] — the row-dot-column definition falls out of
   the notation rather than being memorized beside it.
3. The diagonal-middle example is the first spec that's EASIER than its
   classical spelling — no diag allocation, and the vector's role (scale
   index j) is visible. This "structure stays implicit" trick is much of
   einsum's practical value.

## Faded practice

### q262
Matrix–vector product.

```python starter
import numpy as np

def solve(a, b):
    """(i,j) @ (j,) -> (i,): contract the shared letter."""
    return np.einsum('_____', a, b)
```

```python solution
import numpy as np

def solve(a, b):
    """(i,j) @ (j,) -> (i,): contract the shared letter."""
    return np.einsum('ij,j->i', a, b)
```

### q260
Matrix product of (i, k) and (k, j).

```python starter
import numpy as np

def solve(c, d):
    """The matmul spec: inner dimension shared and dropped."""
    return np.einsum('_____', c, d)
```

```python solution
import numpy as np

def solve(c, d):
    """The matmul spec: inner dimension shared and dropped."""
    return np.einsum('ik,kj->ij', c, d)
```

## Guided practice

### q264
1. a @ b @ c with shapes (i,k), (k,m), (m,p) — one spec, three inputs.
2. Two letters contract (the two inner dimensions); two survive.
3. `'ik,km,mp->ip'` — each adjacent pair shares exactly one letter, like
   dominoes.

## Independent practice

From the drill bank: q312 (permutation matrix times vector — mathematically
just a matvec; notice WHICH spec, then reflect that p @ v with a 0/1 matrix
is a reordering), q311 (a @ diag(v) @ b without materializing the diagonal —
the three-input spec with a bare vector in the middle).

## Misconceptions

- **"'ij,jk->ik' and 'ij,kj->ik' are basically the same."** — The second
  pairs b's FIRST axis with nothing and contracts j against b's second axis
  — it computes a @ b.T. One transposed letter, silently different math;
  this is THE einsum bug, and the `@`-twin assert is how you catch it.
- **"Chained products should be separate einsum/@ calls."** — One spec is
  clearer and (with optimize=True) lets einsum pick the cheaper
  association order — (a@b)@c vs a@(b@c) can differ by orders of magnitude
  in FLOPs.
- **"A diagonal matrix must be built to multiply by it."** — A bare vector
  in the spec ('ij,j,jk->ik') applies the diagonal implicitly. Matrices
  with structure (diagonal, permutation, one-hot) often stay implicit in
  einsum.
