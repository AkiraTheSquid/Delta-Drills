---
kc: numpy.diag-triangles
title: Diagonals, triangles, and trace
supporting: [numpy.constructors, numpy.aggregations]
new_syntax: []
faded: [77, 237, 47, 31]
guided: [78]
independent: [140, 16, 4, 160, 180]
---

## Concept: extracting a diagonal with t.diag

Give `t.diag` a **2-D array** and it *extracts* a diagonal as a 1-D array.
Which diagonal is picked by one convention you'll reuse everywhere: the
**offset k**.

- `k = 0` (the default) is the **main diagonal** — top-left to bottom-right.
- **Positive k** counts diagonals *above* the main one: `t.diag(z, k=1)` is
  the superdiagonal.
- **Negative k** counts diagonals *below*: `t.diag(z, k=-1)` is the
  subdiagonal.

Every function in this family (`diag`, `trace`, `triu`, `tril`, `t.eye`'s
`k=`) shares this exact convention — learn it once here.

## Worked example

Task: extract the main, upper, and lower diagonals of a 3×3 matrix.

```python
import torch as t

z = t.arange(1, 10).reshape(3, 3)   # [[1,2,3],[4,5,6],[7,8,9]]

# The offset convention in action: up is positive, down is negative.
assert t.diag(z).tolist() == [1, 5, 9]        # k=0, main
assert t.diag(z, diagonal=1).tolist() == [2, 6]      # one above
assert t.diag(z, diagonal=-1).tolist() == [4, 8]     # one below
```

Why: `t.arange(1, 10).reshape(3, 3)` is the perfect test matrix — every
entry announces its own position, so you can *see* which diagonal came out.
When unsure about a k, test on it.

## Faded practice

### q77
The k-th diagonal of a square matrix, using the standard sign convention.

```python starter
import torch as t

def solve(z, k):
    """The k-th diagonal of z (k=0 main, positive above, negative below)."""
    return t.diag(z, _____)
```

```python solution
import torch as t

def solve(z, k):
    """The k-th diagonal of z (k=0 main, positive above, negative below)."""
    return t.diag(z, diagonal=k)
```

## Concept: t.trace — sum of the main diagonal

`t.trace(z)` is the **sum of the main diagonal**, returned as a scalar. It's
equivalent to `t.diag(z).sum()` — extract, then reduce — collapsed into one
call. It also accepts `offset=` for other diagonals, with the same sign
convention as `t.diag`'s k.

## Worked example

```python
import torch as t

z = t.arange(1, 10).reshape(3, 3)

# Trace = main diagonal summed: 1 + 5 + 9.
assert t.trace(z) == 15

# Same thing spelled as extract-then-reduce:
assert t.diag(z).sum() == 15

# t.trace has NO offset argument — for any other diagonal, compose:
assert t.diagonal(z, offset=1).sum() == 8
```

Why: "sum of the k-th diagonal" is the composition habit this whole toolkit
builds — *structure function, then reduce*. `t.trace` is just the shortcut
for the most common case.

## Faded practice

### q237
The trace of a square matrix, as a scalar.

```python starter
import torch as t

def solve(z):
    """Sum of z's main-diagonal entries."""
    return t._____(z)
```

```python solution
import torch as t

def solve(z):
    """Sum of z's main-diagonal entries."""
    return t.trace(z)
```

## Concept: building a diagonal matrix with t.diag

`t.diag` is a **two-way street** — its behavior depends on the input's rank.
You've seen the 2-D direction (extract). Give it a **1-D array** and it
*builds*: a square matrix with those values ON the diagonal, zeros elsewhere.
`t.diag(t.tensor([1, 2, 3]))` is a 3×3 diagonal matrix (and unlike NumPy it
insists on a tensor, not a bare list); `t.diag(vals, diagonal=-1)` places
them just below the main diagonal (the result grows to fit: length n on
offset k gives shape (n+|k|, n+|k|)).

Build-vs-extract is decided by the INPUT's rank, not by an argument — passing
a vector where you meant a matrix silently switches modes, so always know
which one your data is.

## Worked example

```python
import torch as t

# BUILD (1-D input): same function, other direction.
d = t.diag(t.tensor([1.0, 2.0, 3.0]))
assert d.tolist() == [[1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0],
                      [0.0, 0.0, 3.0]]

# With an offset, the matrix grows to fit: 2 values on k=-1 -> 3x3.
assert t.diag(t.tensor([7, 8]), diagonal=-1).tolist() == [[0, 0, 0],
                                          [7, 0, 0],
                                          [0, 8, 0]]
```

Why: the offset build is how you make shift/step matrices (e.g. the values
1..n-1 just below the diagonal) in one call — no loops, no indexing.

## Faded practice

### q47
A k×k matrix with the given values on its main diagonal.

```python starter
import torch as t

def solve(vals):
    """Square matrix with vals on the diagonal, zeros elsewhere."""
    return t._____(t.as_tensor(vals))
```

```python solution
import torch as t

def solve(vals):
    """Square matrix with vals on the diagonal, zeros elsewhere."""
    return t.diag(t.as_tensor(vals))
```

## Concept: triangles — t.triu and t.tril

`t.triu(z, k=0)` keeps the **upper triangle** — everything ON and ABOVE
diagonal k — and zeroes the rest; `t.tril(z, k=0)` keeps the lower. The `k`
shifts the cut line, with the same sign convention as always:
`t.tril(z, k=-1)` keeps only *strictly below* the main diagonal.

To build a triangular matrix from scratch, compose with a constructor:
`t.triu(t.ones((n, n)))` is the upper-triangular matrix of ones — pass
`dtype=bool` to `ones` and it's a boolean mask instead. Constructor →
structure function is the standard recipe.

## Worked example

```python
import torch as t

z = t.arange(1, 10).reshape(3, 3)

# Keep upper (with diagonal), zero the rest...
assert t.triu(z).tolist() == [[1, 2, 3],
                               [0, 5, 6],
                               [0, 0, 9]]

# ...or keep ONLY the strictly-below part by shifting the cut to k=-1.
assert t.tril(z, diagonal=-1).tolist() == [[0, 0, 0],
                                     [4, 0, 0],
                                     [7, 8, 0]]

# Compose with a constructor: ones cut to a triangle.
tri = t.triu(t.ones((3, 3)))
assert tri.tolist() == [[1.0, 1.0, 1.0],
                        [0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0]]
```

Why: "strictly above/below" is expressed by shifting the cut (k=1 / k=-1),
not by post-processing. Most triangle tasks are one call with the right k.

## Faded practice

### q31
The n×n upper-triangular matrix of ones (diagonal included).

```python starter
import torch as t

def solve(n):
    """Upper triangle of ones, including the main diagonal."""
    return t.triu(t._____((n, n)))
```

```python solution
import torch as t

def solve(n):
    """Upper triangle of ones, including the main diagonal."""
    return t.triu(t.ones((n, n)))
```

## Guided practice

### q78
1. Upper-triangular INCLUDING the diagonal is the default offset — no
   `diagonal=` argument needed.
2. Build a full tensor of the right dtype first, then let the triangular
   helper zero out the lower part.
3. `t.triu(t.ones((n, n), dtype=t.bool))` — set the dtype at construction;
   converting afterwards is an extra pass, and an int tensor would fail
   the bool check.

## Independent practice

From the drill bank: q140 (SUM of the k-th diagonal — extract then sum, mind
the plain-int contract), q16 (ones strictly BELOW the diagonal — which
function, which k?), q4 (the values 1..n-1 placed just below the diagonal —
diag's build mode with an offset).

Also from the bank: q160 (lower-triangular matrix whose row i is the first
i+1 entries of v), q180 (the tridiagonal stencil — d0 on the main
diagonal, d1 on both neighbours).

## Misconceptions

- **"t.diag always extracts."** — With 1-D input it BUILDS a matrix. The
  function is two-way, switched by input rank; a stray reshape can flip you
  into the wrong mode.
- **"triu deletes the diagonal too."** — Default k=0 KEEPS the diagonal.
  "Strictly above/below" needs k=1 / k=-1 — read the task for whether the
  diagonal is in or out.
- **"Positive k means below."** — Positive is ABOVE the main diagonal,
  negative below, consistently across diag/trace/triu/tril/eye. When unsure,
  test on `t.arange(9).reshape(3,3)` where every entry announces its
  position.
