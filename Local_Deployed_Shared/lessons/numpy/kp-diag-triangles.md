---
kc: numpy.diag-triangles
title: Diagonals, triangles, and trace
supporting: [numpy.constructors, numpy.aggregations]
new_syntax: []
faded: [237, 47]
guided: [77]
independent: [140, 31, 16, 4]
---

## Concept

Matrix structure along and around the main diagonal has a compact toolkit,
organized by one convention: the **offset k** — `k = 0` is the main diagonal,
positive k counts diagonals *above* it, negative k *below*.

**`np.diag` — a two-way street.** Its behavior depends on input rank:

- Give it a **2-D array**: it *extracts* the k-th diagonal as a 1-D array.
  `np.diag(z)` is the main diagonal; `np.diag(z, k=1)` the superdiagonal.
- Give it a **1-D array**: it *builds* a matrix with those values ON the k-th
  diagonal, zeros elsewhere. `np.diag([1, 2, 3])` is a 3×3 diagonal matrix;
  `np.diag(vals, k=-1)` places them just below the main diagonal (the result
  grows to fit: length n on offset k gives shape (n+|k|, n+|k|)).

**`np.trace(z)`** — the sum of the main diagonal, as a scalar. Equivalent to
`np.diag(z).sum()`, and it accepts `offset=` for other diagonals.

**Triangles.** `np.triu(z, k=0)` keeps the upper triangle (everything ON and
ABOVE diagonal k), zeroing the rest; `np.tril(z, k=0)` keeps the lower. The
`k` shifts the cut line: `np.tril(z, k=-1)` keeps only *strictly below* the
main diagonal. Building triangular masks/matrices from scratch composes them
with constructors: `np.triu(np.ones((n, n)))` is the upper-triangular matrix
of ones — pass `dtype=bool` to `ones` and it's a boolean mask instead.

The composition habit to build: **constructor → structure function →
(optionally) reduce.** "Sum of the k-th diagonal" = extract then sum.
"Upper-triangular boolean mask" = bool ones then triu.

## Worked example

Task: extract diagonals at several offsets, compute a trace, build a
diagonal matrix from a vector, and cut a matrix into triangles.

```python
import numpy as np

z = np.arange(1, 10).reshape(3, 3)   # [[1,2,3],[4,5,6],[7,8,9]]

# EXTRACT (2-D input): the offset convention in action.
assert np.diag(z).tolist() == [1, 5, 9]        # k=0, main
assert np.diag(z, k=1).tolist() == [2, 6]      # one above
assert np.diag(z, k=-1).tolist() == [4, 8]     # one below

# Trace = main diagonal summed, directly.
assert np.trace(z) == 15

# BUILD (1-D input): same function, other direction.
d = np.diag(np.array([1.0, 2.0, 3.0]))
assert d.tolist() == [[1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0],
                      [0.0, 0.0, 3.0]]

# Triangles: keep upper (with diagonal), zero the rest...
assert np.triu(z).tolist() == [[1, 2, 3],
                               [0, 5, 6],
                               [0, 0, 9]]
# ...or keep ONLY the strictly-below-diagonal part by shifting the cut:
# k=-1 keeps everything on and below the diagonal one step down.
assert np.tril(z, k=-1).tolist() == [[0, 0, 0],
                                     [4, 0, 0],
                                     [7, 8, 0]]

# Compose with a constructor: ones cut to a triangle.
tri = np.triu(np.ones((3, 3)))
assert tri.tolist() == [[1.0, 1.0, 1.0],
                        [0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0]]
```

Why each step:

1. The three `np.diag` extractions pin the offset convention — up is
   positive. Every function in this family (`diag`, `trace`, `triu`, `tril`,
   `np.eye`'s `k=`) shares it, so learn it once.
2. Build-vs-extract is decided by the INPUT's rank, not by an argument —
   passing a vector where you meant a matrix silently switches modes, so
   know which one your data is.
3. The `k=-1` tril shows how "strictly below" is expressed: shift the cut,
   don't post-process. Most triangle tasks are one call with the right k.

## Faded practice

### q237
The trace of a square matrix, as a scalar.

```python starter
import numpy as np

def solve(z):
    """Sum of z's main-diagonal entries."""
    return np._____(z)
```

```python solution
import numpy as np

def solve(z):
    """Sum of z's main-diagonal entries."""
    return np.trace(z)
```

### q47
A k×k matrix with the given values on its main diagonal.

```python starter
import numpy as np

def solve(vals):
    """Square matrix with vals on the diagonal, zeros elsewhere."""
    return np._____(np.asarray(vals))
```

```python solution
import numpy as np

def solve(vals):
    """Square matrix with vals on the diagonal, zeros elsewhere."""
    return np.diag(np.asarray(vals))
```

## Guided practice

### q77
1. The k-th diagonal of a square matrix, with the standard sign convention —
   this is `np.diag`'s extract mode.
2. The offset travels as the keyword `k=`.
3. One call; the convention in the prompt matches NumPy's exactly.

## Independent practice

From the drill bank: q140 (SUM of the k-th diagonal — extract then reduce,
mind the plain-int contract), q31 (upper triangle of ones including the
diagonal), q16 (ones strictly BELOW the diagonal — which function, which k?),
q4 (the values 1..n-1 placed just below the diagonal — diag's build mode
with an offset).

## Misconceptions

- **"np.diag always extracts."** — With 1-D input it BUILDS a matrix. The
  function is two-way, switched by input rank; a stray reshape can flip you
  into the wrong mode.
- **"triu deletes the diagonal too."** — Default k=0 KEEPS the diagonal.
  "Strictly above/below" needs k=1 / k=-1 — read the task for whether the
  diagonal is in or out.
- **"Positive k means below."** — Positive is ABOVE the main diagonal,
  negative below, consistently across diag/trace/triu/tril/eye. When unsure,
  test on `np.arange(9).reshape(3,3)` where every entry announces its
  position.
