---
kc: numpy.linalg-basics
title: Matrix multiply and t.linalg basics
supporting: [numpy.aggregations, numpy.elementwise-ufuncs]
new_syntax: [matmul-operator, syntax.matmul, torch.linalg, torch.linalg.solve]
faded: [239, 107]
guided: [508, 509]
independent: [510, 511, 512, 513]
---

## Concept: two multiplications — * vs @

Two different "multiplications" exist for matrices, and PyTorch gives each its
own operator:

- **`a * b` — elementwise**: multiplies corresponding entries; shapes must
  match (or broadcast). No summing happens.
- **`a @ b` — matrix multiplication**: row-times-column with a sum inside.
  For `a` of shape (m, k) and `b` of shape (k, n), the result is (m, n):
  entry `[i, j]` is the dot product of row i of `a` with column j of `b`.
  The inner dimensions (k) must agree, and they disappear in the output.

```python
import torch as t

a = t.tensor([[1.0, 2.0], [3.0, 4.0]])
b = t.tensor([[5.0, 6.0], [7.0, 8.0]])
print("a * b (elementwise)")
print(a * b)
print("a @ b (matrix product)")
print(a @ b)
```

`a*b` entry [0,0] is 1·5. `a@b` entry [0,0] is 1·5 + 2·7 = 19 — the row met
the column and the k axis was summed away.

The shape rule `(m, k) @ (k, n) → (m, n)` is worth chanting: it predicts
both whether a product is legal and what comes out. It also covers
matrix–vector: `(m, k) @ (k,) → (m,)`.

```python
print((t.ones((2, 3)) @ t.ones((3, 4))).shape)     # (2,4): the 3s vanish
print((t.ones((2, 3)) @ t.ones(3)).shape)          # (2,): matrix times vector

try:
    t.ones((2, 3)) @ t.ones((2, 3))                # inner dims 3 vs 2
except RuntimeError as err:
    print("RuntimeError:", err)
```

`t.matmul(a, b)` is the same operation spelled as a function, and in model
code you will meet `a.T` for the transpose that so often precedes it.

## Worked example

```python
import torch as t

a = t.tensor([[1.0, 2.0],
              [3.0, 4.0]])
b = t.tensor([[5.0, 6.0],
              [7.0, 8.0]])

# Elementwise vs matrix product — same operands, different operations:
elem = a * b            # [[5, 12], [21, 32]] — corresponding entries
mat = a @ b             # row·column with a sum inside
assert elem.tolist() == [[5.0, 12.0], [21.0, 32.0]]
assert mat.tolist() == [[19.0, 22.0], [43.0, 50.0]]
# Check one entry by hand: mat[0,0] = 1*5 + 2*7 = 19. Row 0 · column 0.
print("a * b")
print(elem)
print("a @ b")
print(mat)

# Shape rule: (2,3) @ (3,2) -> (2,2); the inner 3s must match and vanish.
p = t.ones((2, 3)) @ t.ones((3, 2))
assert p.shape == (2, 2)
```

Why: computing `mat[0, 0]` by hand once (row 0 of `a` dotted with column 0
of `b`) is the fastest way to internalize what `@` does beyond the shape
rule — and predicting shapes BEFORE running makes mismatches design errors
you catch on paper.

## Faded practice

### q239
Matrix product of shapes (m, k) and (k, n).

```python starter
import torch as t

def solve(a, b):
    """The (m, n) matrix product of a (m, k) and b (k, n)."""
    return a _____ b
```

```python solution
import torch as t

def solve(a, b):
    """The (m, n) matrix product of a (m, k) and b (k, n)."""
    return a @ b
```

## Concept: t.linalg.solve — never build the inverse

The `t.linalg` submodule holds the "real linear algebra", and its names match
NumPy's `np.linalg` almost one for one:

- **`t.linalg.solve(a, b)`** — solve the system `a @ x = b` for `x`.
  This is THE way to compute "a⁻¹ b". Numerically, solving directly is both
  faster and more accurate than `t.linalg.inv(a) @ b`; computing an
  explicit inverse is almost never what you want.
- `t.linalg.inv`, `t.linalg.det`, `t.linalg.matrix_rank`,
  `t.linalg.norm`, `t.linalg.eig` — inverse, determinant, rank, norms,
  eigendecomposition, when a task genuinely asks for them.

One dtype caveat that is easy to trip over here: these routines want floats,
and the default float is 32-bit. Ill-conditioned systems lose accuracy sooner
than the float64 you may be used to from NumPy — if a solve looks wrong,
checking the dtype is a reasonable first move.

```python
import torch as t

a_sys = t.tensor([[3.0, 1.0],
                  [1.0, 2.0]])
b_vec = t.tensor([9.0, 8.0])
x = t.linalg.solve(a_sys, b_vec)
print("x =", x)
```

Sanity-checking a solve is one line: plug `x` back in and compare
`a @ x` with `b` using `t.allclose` (float arithmetic — never `==`).

```python
print("a @ x =", a_sys @ x, " b =", b_vec)
print("exactly equal? ", bool(t.equal(a_sys @ x, b_vec)))
print("close enough?  ", bool(t.allclose(a_sys @ x, b_vec)))
assert t.allclose(a_sys @ x, b_vec)
```

The inverse route reaches the same answer and does more work to get there —
run it once so the equivalence is concrete, then stop writing it:

```python
via_inverse = t.linalg.inv(a_sys) @ b_vec
print("solve:  ", x)
print("inverse:", via_inverse)
assert t.allclose(x, via_inverse)
```

## Worked example

```python
import torch as t

# Solve a @ x = b_vec — NOT by computing an inverse.
a_sys = t.tensor([[2.0, 0.0],
                  [0.0, 4.0]])
b_vec = t.tensor([6.0, 8.0])
x = t.linalg.solve(a_sys, b_vec)
assert x.tolist() == [3.0, 2.0]
print("x =", x)

# Verification pattern: substitute back, compare with float tolerance.
assert t.allclose(a_sys @ x, b_vec)
print("a @ x =", a_sys @ x, " b =", b_vec)
```

Why: `solve` + `allclose` verification — the pair costs one line and
catches both wrong answers and ill-conditioned systems.

## Faded practice

### q107
Solve the linear system a @ x = b (a is invertible).

```python starter
import torch as t

def solve(a, b):
    """Return x such that a @ x = b (use a solver, not an inverse)."""
    return t.linalg._____(a, b)
```

```python solution
import torch as t

def solve(a, b):
    """Return x such that a @ x = b (use a solver, not an inverse)."""
    return t.linalg.solve(a, b)
```

## Guided practice

### q508
1. This is the multiplication that is NOT matrix multiplication.
2. Entry [i][j] depends only on the two entries at [i][j] — nothing is summed,
   so no axis is contracted.
3. `a * b`.

### q509
1. Both answers come from the same two matrices; only the operator changes.
2. `*` pairs entries in place; `@` contracts a's columns against b's rows.
3. `((a * b).tolist(), (a @ b).tolist())`.

## Independent practice

From the drill bank: q510 (transpose a matrix and read back its new shape).
From the drill bank: q511 (matrix @ vector — which axis vanishes?).
From the drill bank: q512 (solve a @ x = b, then verify a @ x really does reproduce b).
From the drill bank: q513 (apply a whole BATCH of matrices to one vector with a single @).

## Misconceptions

- **"`*` multiplies matrices."** — `*` is elementwise; `@` is the matrix
  product. Mixing them up usually *doesn't* crash (broadcasting can make `*`
  legal), it just silently computes the wrong thing — the worst kind of bug.
- **"To solve a @ x = b, compute inv(a) @ b."** — `t.linalg.solve(a, b)` is
  more accurate and faster; explicit inverses amplify rounding error and cost
  more. Reach for `inv` only when the inverse itself is the deliverable.
- **"If `@` runs, the shapes were right."** — `@` between wrong-but-compatible
  shapes (e.g. transposed operands, square matrices) runs happily and returns
  garbage. Predict `(m, k) @ (k, n) → (m, n)` on paper first.
- **"Integer tensors are fine for linalg."** — They are not; the solvers
  require floating point and will raise. Convert with `.to(t.float32)` first.
