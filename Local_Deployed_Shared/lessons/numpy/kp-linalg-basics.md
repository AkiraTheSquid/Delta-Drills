---
kc: numpy.linalg-basics
title: Matrix multiply and np.linalg basics
supporting: [numpy.aggregations, numpy.elementwise-ufuncs]
new_syntax: [matmul-operator]
faded: [107]
guided: [239]
independent: []
---

## Concept

Two different "multiplications" exist for matrices, and NumPy gives each its
own operator:

- **`a * b` — elementwise**: multiplies corresponding entries; shapes must
  match (or broadcast). No summing happens.
- **`a @ b` — matrix multiplication**: row-times-column with a sum inside.
  For `a` of shape (m, k) and `b` of shape (k, n), the result is (m, n):
  entry `[i, j]` is the dot product of row i of `a` with column j of `b`.
  The inner dimensions (k) must agree, and they disappear in the output.

The shape rule `(m, k) @ (k, n) → (m, n)` is worth chanting: it predicts both
whether a product is legal and what comes out. It also covers matrix–vector:
`(m, k) @ (k,) → (m,)`.

Beyond `@`, the `np.linalg` submodule holds the "real linear algebra":

- **`np.linalg.solve(a, b)`** — solve the system `a @ x = b` for `x`.
  This is THE way to compute "a⁻¹ b". Numerically, solving directly is both
  faster and more accurate than `np.linalg.inv(a) @ b`; computing an explicit
  inverse is almost never what you want.
- `np.linalg.inv`, `np.linalg.det`, `np.linalg.matrix_rank`,
  `np.linalg.norm`, `np.linalg.eig` — inverse, determinant, rank, norms,
  eigendecomposition, when a task genuinely asks for them.

Sanity-checking a solve is one line: plug `x` back in and compare
`a @ x` with `b` using `np.allclose` (float arithmetic — never `==`).

## Worked example

Task: multiply two matrices, then solve a linear system and verify the
solution.

```python
import numpy as np

a = np.array([[1.0, 2.0],
              [3.0, 4.0]])
b = np.array([[5.0, 6.0],
              [7.0, 8.0]])

# Elementwise vs matrix product — same operands, different operations:
elem = a * b            # [[5, 12], [21, 32]] — corresponding entries
mat = a @ b             # row·column with a sum inside
assert elem.tolist() == [[5.0, 12.0], [21.0, 32.0]]
assert mat.tolist() == [[19.0, 22.0], [43.0, 50.0]]
# Check one entry by hand: mat[0,0] = 1*5 + 2*7 = 19. Row 0 · column 0.

# Shape rule: (2,3) @ (3,2) -> (2,2); the inner 3s must match and vanish.
p = np.ones((2, 3)) @ np.ones((3, 2))
assert p.shape == (2, 2)

# Solve a @ x = b_vec — NOT by computing an inverse.
a_sys = np.array([[2.0, 0.0],
                  [0.0, 4.0]])
b_vec = np.array([6.0, 8.0])
x = np.linalg.solve(a_sys, b_vec)
assert x.tolist() == [3.0, 2.0]

# Verification pattern: substitute back, compare with float tolerance.
assert np.allclose(a_sys @ x, b_vec)
```

Why each step:

1. Computing `mat[0, 0]` by hand once (row 0 of `a` dotted with column 0 of
   `b`) is the fastest way to internalize what `@` does beyond the shape rule.
2. The `(2,3) @ (3,2)` example isolates the shape rule from the values —
   predict shapes BEFORE running, and mismatches become design errors you
   catch on paper.
3. `solve` + `allclose` verification: the pair costs one line and catches
   both wrong answers and ill-conditioned systems.

## Faded practice

### q107
Solve the linear system a @ x = b (a is invertible).

```python starter
import numpy as np

def solve(a, b):
    """Return x such that a @ x = b (use a solver, not an inverse)."""
    return np.linalg._____(a, b)
```

```python solution
import numpy as np

def solve(a, b):
    """Return x such that a @ x = b (use a solver, not an inverse)."""
    return np.linalg.solve(a, b)
```

## Guided practice

### q239
1. Matrix product of shapes (m, k) and (k, n) — which operator contracts the
   shared k axis?
2. Not `*` — that's elementwise and would fail on (m,k)×(k,n) shapes anyway.
3. One binary operator between the two arrays does it.

## Misconceptions

- **"`*` multiplies matrices."** — `*` is elementwise; `@` is the matrix
  product. Mixing them up usually *doesn't* crash (broadcasting can make `*`
  legal), it just silently computes the wrong thing — the worst kind of bug.
- **"To solve a @ x = b, compute inv(a) @ b."** — `np.linalg.solve(a, b)` is
  more accurate and faster; explicit inverses amplify rounding error and cost
  more. Reach for `inv` only when the inverse itself is the deliverable.
- **"If `@` runs, the shapes were right."** — `@` between wrong-but-compatible
  shapes (e.g. transposed operands, square matrices) runs happily and returns
  garbage. Predict `(m, k) @ (k, n) → (m, n)` on paper first.
