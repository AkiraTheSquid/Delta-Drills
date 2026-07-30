---
kc: numpy.dot-matmul-patterns
title: Dot products and matrix-multiply patterns
supporting: [numpy.linalg-basics, numpy.axis-reductions, numpy.broadcasting-rules]
new_syntax: [torch.dot, torch.isclose, torch.linalg.norm]
faded: [37, 144, 121, 5]
guided: [514]
independent: [141, 515]
---

## Concept: the dot product — multiply, then sum

The **dot product** of two equal-length vectors — multiply corresponding
entries, add them up — is the atom that all of linear algebra's products are
built from. PyTorch spells it three interchangeable ways:

```python no-run
t.dot(a, b)      ==  a @ b  ==  (a * b).sum()
```

The third spelling is the important one conceptually: *dot = elementwise
multiply + reduction.* Holding the decomposition lets you build variants
(weighted dots, masked dots, batch dots) instead of hunting for a function
that may not exist.

## Worked example

```python
import torch as t

a = t.tensor([1.0, 2.0, 3.0])
b = t.tensor([4.0, -5.0, 6.0])

# The atom, three spellings — same number.
d = float(t.dot(a, b))
assert d == float(a @ b) == float((a * b).sum()) == 12.0
```

Why: verifying the three dot spellings agree once buys permanent fluency:
when you see `(x * w).sum()` in someone's code, you now read "dot".

## Faded practice

### q37
Dot product of two vectors, as a plain float.

```python starter
import torch as t

def solve(a, b):
    """The dot product of vectors a and b."""
    return float(t._____(a, b))
```

```python solution
import torch as t

def solve(a, b):
    """The dot product of vectors a and b."""
    return float(t.dot(a, b))
```

## Concept: matrix @ vector — one dot per row

**Matrix @ vector** (`z @ v`, shapes (n, m) @ (m,)): one dot per row of z —
"each row dotted with v" in a single call. Result shape (n,). (Matrix @
matrix is one dot per (row, column) pair — the previous KP.)

Check one output by hand and the shape rule follows: matmul = a dot per
row, so an (n, m) matrix against a length-m vector yields n dots.

## Worked example

```python
import torch as t

# Matrix @ vector: row i of the result = (row i of z) . v
z = t.tensor([[1.0, 2.0],
              [3.0, 4.0]])
v = t.tensor([10.0, 1.0])
zv = z @ v
assert zv.tolist() == [12.0, 34.0]        # 1*10+2*1, 3*10+4*1
```

Why: checking one output by hand (1·10 + 2·1 = 12) anchors "matmul = a dot
per row" — and predicts the result's shape (n,) without memorizing another
rule.

## Faded practice

### q144
Each row of z dotted with v.

```python starter
import torch as t

def solve(z, v):
    """Length-n array: entry i = (row i of z) . v."""
    return z _____ v
```

```python solution
import torch as t

def solve(z, v):
    """Length-n array: entry i = (row i of z) . v."""
    return z @ v
```

## Concept: when @ doesn't fit — multiply, then reduce an axis

When the pattern you need is *not* one of the packaged shapes, the
decomposition rescues you. "Dot each row of `a` with the CORRESPONDING row
of `b`" (same shapes) is not `a @ b` — matmul dots every row with every
COLUMN. But per the atom: multiply elementwise, then reduce each row:

```python no-run
(a * b).sum(dim=1)          # row-wise dots
```

That multiply-then-reduce-an-axis maneuver covers the "batch of dots" tasks
— and it is the exact pattern einsum notation will name concisely in the
next course topic.

## Worked example

```python
import torch as t

# Row-wise dots of two SAME-SHAPE matrices: NOT a matmul.
p = t.tensor([[1.0, 2.0],
              [3.0, 4.0]])
q = t.tensor([[5.0, 6.0],
              [7.0, 8.0]])
row_dots = (p * q).sum(dim=1)
assert row_dots.tolist() == [17.0, 53.0]  # 1*5+2*6, 3*7+4*8
```

Why: this case is deliberately a trap — `p @ q` runs on these square
matrices and returns the WRONG thing (full matrix product). Decomposing to
multiply+sum(axis=1) is the general escape whenever the packaged products
don't match the pairing you need.

## Faded practice

### q121
Dot each row of a with the corresponding row of b.

```python starter
import torch as t

def solve(a, b):
    """Length-n array: entry i = (row i of a) . (row i of b)."""
    return (a * b).sum(dim=_____)
```

```python solution
import torch as t

def solve(a, b):
    """Length-n array: entry i = (row i of a) . (row i of b)."""
    return (a * b).sum(dim=1)
```

## Concept: norms — a dot with itself, rooted

**Norms** are the same atom again: a vector's Euclidean length is
`t.sqrt(v @ v)`. Applied to a whole matrix's entries — √(sum of all
squares) — it's the **Frobenius norm**, `t.linalg.norm(z)` with no
arguments (as if the matrix were one long vector). Operator norms exist
behind `ord=`, but Frobenius is the drills' default meaning of "the norm".

## Worked example

```python
import torch as t

# Frobenius norm: sqrt of the sum of ALL squared entries.
f = t.tensor([[3.0, 4.0],
              [0.0, 0.0]])
assert t.linalg.norm(f) == 5.0
assert t.isclose(t.linalg.norm(f), t.sqrt((f * f).sum()))
```

Why: the first-principles spelling `t.sqrt((z * z).sum())` is
multiply+reduce again — the whole KP is one atom wearing different hats.

## Faded practice

### q5
The Frobenius norm of a matrix.

```python starter
import torch as t

def solve(z):
    """Square root of the sum of squares of ALL entries of z."""
    return t.linalg._____(z)
```

```python solution
import torch as t

def solve(z):
    """Square root of the sum of squares of ALL entries of z."""
    return t.linalg.norm(z)
```

## Independent practice

From the drill bank: q141 (diagonal of a @ b WITHOUT computing the full
product — think about which dots the diagonal actually needs: row i of a
with COLUMN i of b).

From the drill bank: q515 (rescale every row to unit length).

## Guided practice

### q514
1. Do not reach for the built-in dot product — the question wants the two
   steps it is made of.
2. Multiply the pairs elementwise, then collapse the result to one number.
3. `float((a * b).sum())`.

## Misconceptions

- **"Row-wise dots of two matrices = a @ b."** — Matmul dots every row with
  every COLUMN. Corresponding-rows pairing is elementwise-multiply + row
  reduction: `(a * b).sum(axis=1)`.
- **"The dot product is its own primitive."** — It's multiply + sum. Holding
  the decomposition lets you build variants (weighted dots, masked dots,
  batch dots) instead of hunting for a function that may not exist.
- **"norm of a matrix = largest row norm."** — Default `t.linalg.norm(z)` on
  2-D is FROBENIUS: all entries squared, summed, rooted — as if the matrix
  were one long vector. Operator norms exist behind `ord=`, but Frobenius is
  the drills' default meaning of "the norm".
