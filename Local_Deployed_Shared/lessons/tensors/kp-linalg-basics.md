---
kc: torch.linalg-basics
title: Matrix multiply and t.linalg basics
supporting: [torch.aggregations, torch.elementwise-ops]
new_syntax: [syntax.matmul, torch.linalg.inv, torch.linalg.solve]
faded: [239, 107, 1722, 1723, 1724, 1725, 1726, 1727]
guided: [508, 509]
independent: [510, 511, 512, 513, 1571, 1572, 1573, 1574]
integrated: [1521, 1522, 1523, 1575, 1662, 1663, 1664, 1665]
---

## Concept: two multiplications — * vs @

Two different "multiplications" exist for matrices, and PyTorch gives each its
own operator:

- **`a * b` — elementwise**: pairs the entries at the same row and column
  and multiplies them; for this comparison use matrices of equal shape. No
  summing happens.
- **`a @ b` — matrix multiplication**: row-times-column with a sum inside.
  For `a` of shape (m, k) and `b` of shape (k, n), the result is (m, n): to
  get entry `[i, j]`, multiply the corresponding entries of row i of `a` and
  column j of `b`, then add those products. The inner dimensions (k) must
  agree, and they disappear in the output.

```python
import torch as t

a = t.tensor([[1.0, 2.0], [3.0, 4.0]])
b = t.tensor([[5.0, 6.0], [7.0, 8.0]])
print("a * b (elementwise)")
print(a * b)
print("a @ b (matrix product)")
print(a @ b)
# Hidden checks
assert _delta_output == 'a * b (elementwise)\ntensor([[ 5., 12.],\n        [21., 32.]])\na @ b (matrix product)\ntensor([[19., 22.],\n        [43., 50.]])\n'
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
# Hidden checks
assert _delta_output.startswith("torch.Size([2, 4])\ntorch.Size([2])\nRuntimeError:")
assert "cannot be multiplied" in _delta_output
```

`t.matmul(a, b)` is the same operation spelled as a function, and in model
code you will meet `a.T` for the transpose that so often precedes it.

## Worked example

The example below runs both products on the same two matrices, then
checks how the input shapes fix the shape of the matrix product.

```python
import torch as t

a = t.tensor([[1.0, 2.0],
              [3.0, 4.0]])
b = t.tensor([[5.0, 6.0],
              [7.0, 8.0]])

# Elementwise vs matrix product — same operands, different operations:
elem = a * b            # [[5, 12], [21, 32]] — corresponding entries
mat = a @ b             # row·column with a sum inside
# Check one entry by hand: mat[0,0] = 1*5 + 2*7 = 19. Row 0 · column 0.
print("a * b")
print(elem)
print("a @ b")
print(mat)

# Shape rule: (2,3) @ (3,2) -> (2,2); the inner 3s must match and vanish.
p = t.ones((2, 3)) @ t.ones((3, 2))
# Hidden checks
assert elem.tolist() == [[5.0, 12.0], [21.0, 32.0]]
assert mat.tolist() == [[19.0, 22.0], [43.0, 50.0]]
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

### q1722
Return the matrix–vector product of the floating-point PyTorch tensors `a`, shape `(m, k)`, and `v`, shape `(k,)`, as a PyTorch tensor of shape `(m,)`: entry `i` multiplies row `i` of `a` with `v` entry by entry and adds the products.

```python starter
import torch as t

def solve(a, v):
    """Matrix times vector: entry i is row i of a combined with v."""
    return a _____ v
```

```python solution
import torch as t

def solve(a, v):
    """Matrix times vector: entry i is row i of a combined with v."""
    return a @ v
```

### q1723
Return the vector–matrix product of the floating-point PyTorch tensors `v`, shape `(k,)`, and `b`, shape `(k, n)`, as a plain Python list of `n` floats: entry `j` multiplies `v` with column `j` of `b` entry by entry and adds the products.

```python starter
import torch as t

def solve(v, b):
    """Vector on the left: a (k,) row combined with each column of b."""
    return (v _____ b).tolist()
```

```python solution
import torch as t

def solve(v, b):
    """Vector on the left: a (k,) row combined with each column of b."""
    return (v @ b).tolist()
```

### q1724
Return the matrix product of the floating-point PyTorch tensors `a`, `b` and `c`, in that order, as a PyTorch tensor of shape `(m, p)`. Their shapes are `(m, k)`, `(k, n)` and `(n, p)`: `k` and `n` are both contracted away.

```python starter
import torch as t

def solve(a, b, c):
    """A chain of two matrix products: the inner sizes vanish twice."""
    return a _____ b _____ c
```

```python solution
import torch as t

def solve(a, b, c):
    """A chain of two matrix products: the inner sizes vanish twice."""
    return a @ b @ c
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
# Hidden checks
assert _delta_output == 'x = tensor([2., 3.])\n'
```

Sanity-checking a solve is one line: plug `x` back in and compare
`a @ x` with `b` using `t.allclose` (float arithmetic — never `==`).

```python
print("a @ x =", a_sys @ x, " b =", b_vec)
print("exactly equal? ", bool(t.equal(a_sys @ x, b_vec)))
print("close enough?  ", bool(t.allclose(a_sys @ x, b_vec)))
# Hidden checks
assert t.allclose(a_sys @ x, b_vec)
```

The inverse route reaches the same answer and does more work to get there —
run it once so the equivalence is concrete, then stop writing it:

```python
via_inverse = t.linalg.inv(a_sys) @ b_vec
print("solve:  ", x)
print("inverse:", via_inverse)
# Hidden checks
assert t.allclose(x, via_inverse)
```

## Worked example

The example below recovers an unknown vector from two linear equations,
then substitutes the result back into those equations to check it.

```python
import torch as t

# Solve a @ x = b_vec — NOT by computing an inverse.
a_sys = t.tensor([[2.0, 0.0],
                  [0.0, 4.0]])
b_vec = t.tensor([6.0, 8.0])
x = t.linalg.solve(a_sys, b_vec)
print("x =", x)

# Verification pattern: substitute back, compare with float tolerance.
print("a @ x =", a_sys @ x, " b =", b_vec)
# Hidden checks
assert x.tolist() == [3.0, 2.0]
assert t.allclose(a_sys @ x, b_vec)
```

Why: `solve` + `allclose` verification — the pair costs one line and catches
a wrong answer. Substituting back checks that the candidate approximately
satisfies the equations; it does not diagnose an ill-conditioned system, where
a small change in the inputs moves the solution a long way.

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

### q1725
Return the floating-point PyTorch tensor `x` of shape `(n, r)` with `a @ x = b`, in ONE call and without forming an inverse. `a` is an invertible floating-point PyTorch tensor of shape `(n, n)`; `b` is one of shape `(n, r)`, `r` right-hand sides side by side.

```python starter
import torch as t

def solve(a, b):
    """Solve a @ x = b for r right-hand sides at once."""
    return t.linalg._____(a, b)
```

```python solution
import torch as t

def solve(a, b):
    """Solve a @ x = b for r right-hand sides at once."""
    return t.linalg.solve(a, b)
```

### q1726
Recover the vector `x` with `a @ x = y` and return its entries as a plain Python list of `n` floats — without ever building an inverse. `a` is an invertible floating-point PyTorch tensor of shape `(n, n)`; `y = a @ x` is one of shape `(n,)`.

```python starter
import torch as t

def solve(a, y):
    """Recover the vector a was applied to: undo y = a @ x."""
    return t.linalg._____(a, y).tolist()
```

```python solution
import torch as t

def solve(a, y):
    """Recover the vector a was applied to: undo y = a @ x."""
    return t.linalg.solve(a, y).tolist()
```

### q1727
Return the total of every solution entry across all the systems, as a 0-d floating-point PyTorch tensor: solve every system `mats[i] @ x_i = b[i]` in ONE call, then add up all entries of the result. `mats` is a floating-point PyTorch tensor of shape `(batch, n, n)`, every matrix invertible; `b` is one of shape `(batch, n)`, one right-hand side per matrix.

```python starter
import torch as t

def solve(mats, b):
    """One solve per batch entry, in a single call; then total everything."""
    return t.linalg._____(mats, b).sum()
```

```python solution
import torch as t

def solve(mats, b):
    """One solve per batch entry, in a single call; then total everything."""
    return t.linalg.solve(mats, b).sum()
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

## Solo practice

### q510
Transpose a matrix and read back its new shape.

### q511
A matrix applied to a vector: which axis vanishes?

### q512
Solve a @ x = b, then verify a @ x really does reproduce b.

### q513
Apply a whole BATCH of matrices to one vector with a single operator.

### q1571
Matrix times matrix: which axis disappears, and what shape is left.

### q1572
The same two vectors, multiplied two ways — one keeps the length, one collapses it.

### q1573
The inverse of a matrix — not the reciprocal of its entries.

### q1574
Solve a @ x = b — the system, not a product.

## Integrated practice

### q1521
Solve a system whose matrix is the transpose of the one you are handed.

### q1522
Several right-hand sides at once, then verify the whole block.

### q1523
One solver call for a whole batch of systems sharing a right-hand side.

### q1575
A batch of products, then the batch of solves that undoes them — one call each, no loop.

### q1662
A vector on both sides of a matrix: two products chained, and why * cannot do it.

### q1663
The inverse applied twice: the matrix acts on y two times, so the solve has to be undone twice.

### q1664
Two matrices act in turn: one product to combine them, one system to undo both.

### q1665
Both products of the same two matrices, and whether their top-left entries happen to agree.

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
