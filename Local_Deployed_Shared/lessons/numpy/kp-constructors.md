---
kc: numpy.constructors
title: Array constructors — zeros, ones, full, eye, *_like
supporting: [numpy.ndarray-model]
new_syntax: []
faded: [227, 212, 41, 228]
guided: []
independent: [225, 50, 213]
---

## Concept: constructors and the tuple-shape convention

`np.array` converts data you already have. Just as often you need an array
**built from scratch** — a canvas of zeros to fill in, a mask of ones. NumPy
has one constructor per pattern, and they all share the same calling
convention:

> **constructor(shape, dtype=...)** — say how big, optionally say what type.

- **`np.zeros(shape)`** — all entries `0.0`. The default "empty canvas".
- **`np.ones(shape)`** — all entries `1.0`.
- **`np.full(shape, v)`** — all entries equal to your value `v`.
- **`np.empty(shape)`** — allocates *without initializing* (contents are
  whatever bytes were in memory). Only worth it when you will overwrite
  every entry immediately.

**Shape is a tuple for anything above 1-D.** `np.zeros(5)` is a length-5
vector, but a 2-D array needs `np.zeros((2, 3))` — note the inner
parentheses. `np.zeros(2, 3)` is a `TypeError`, because the second
positional argument is the dtype.

## Worked example

Build a 3×4 canvas of zeros:

```python
import numpy as np

# A 2-D shape must be passed as ONE tuple argument: (rows, cols).
board = np.zeros((3, 4))
assert board.shape == (3, 4)
```

Why: the tuple-shape convention (`(3, 4)`) is the single most common
beginner syntax error with constructors — the constructor takes *one* shape
argument, not one argument per dimension.

## Faded practice

### q227
All-zeros float vector of a given length (must also work for length 0).

```python starter
import numpy as np

def solve(n):
    """Return a 1-D float array of n zeros."""
    return np._____(n)
```

```python solution
import numpy as np

def solve(n):
    """Return a 1-D float array of n zeros."""
    return np.zeros(n)
```

## Concept: the dtype is float64 unless you say otherwise

**The default dtype is `float64`,** even though the values print like
integers. Pass `dtype=` to override: `np.ones((2, 2), dtype=bool)` is a
matrix of `True` (1 as a boolean is `True`); `np.zeros(4, dtype=int)` is
integer zeros. Checking `dtype` right after construction is the habit that
catches the float-by-default surprise before it propagates.

## Worked example

An all-`True` boolean mask — ones, with the dtype said out loud:

```python
import numpy as np

# ones gives every entry the value 1 — and 1 as a boolean is True.
mask = np.ones((3, 4), dtype=bool)
assert mask.all() and mask.dtype == np.bool_
```

Why: without `dtype=bool` this would be a float array of 1.0s that merely
*prints* like what you wanted — say the type when it matters.

## Faded practice

### q212
A rows×cols array where every entry is the boolean `True`.

```python starter
import numpy as np

def solve(rows, cols):
    """All-True boolean matrix of shape (rows, cols)."""
    return np.ones((rows, cols), dtype=_____)
```

```python solution
import numpy as np

def solve(rows, cols):
    """All-True boolean matrix of shape (rows, cols)."""
    return np.ones((rows, cols), dtype=bool)
```

## Concept: *_like — copy shape AND dtype from an existing array

The **`*_like` variants** (`np.zeros_like(x)`, `np.ones_like(x)`,
`np.full_like(x, v)`) copy both the shape *and the dtype* from an existing
array — the right tool whenever the question is "give me a blank array
shaped like this one". Reaching for `*_like` is both shorter and safer than
reading off `.shape` and `.dtype` yourself.

## Worked example

```python
import numpy as np

# "Blank array shaped like x" — zeros_like copies shape AND dtype,
# so an int32 input yields an int32 result, not the float default.
x = np.array([[3, -1, 4], [1, 5, -9]], dtype=np.int32)
blank = np.zeros_like(x)
assert blank.shape == x.shape
assert blank.dtype == np.int32
```

Why: `np.zeros(x.shape)` would lose the dtype (float default) — `_like`
keeps both properties in one call.

## Faded practice

### q41
Blank array matching BOTH the shape and dtype of an existing array.

```python starter
import numpy as np

def solve(x):
    """Return an all-zeros array with x's shape and x's dtype."""
    return np._____(x)
```

```python solution
import numpy as np

def solve(x):
    """Return an all-zeros array with x's shape and x's dtype."""
    return np.zeros_like(x)
```

## Concept: np.eye — the identity matrix

**`np.eye(n)`** is the n×n identity matrix: `1.0` on the main diagonal,
`0.0` elsewhere. It's the seed for anything diagonal-shaped: `v * np.eye(n)`
puts a constant v on the diagonal, and `np.eye(n, k=1)` shifts the ones off
the main diagonal (same offset convention as `np.diag`).

## Worked example

```python
import numpy as np

I = np.eye(3)
assert I.tolist() == [[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]]
```

Why: the identity is a constructor, not something you assemble by loop —
and scaling it (`v * np.eye(n)`) is the one-liner for "v on the diagonal".

## Faded practice

### q228
The n×n identity matrix.

```python starter
import numpy as np

def solve(n):
    """n-by-n identity matrix."""
    return np._____(n)
```

```python solution
import numpy as np

def solve(n):
    """n-by-n identity matrix."""
    return np.eye(n)
```

## Independent practice

From the drill bank: q225 (all-ones vector), q50 (v on the diagonal —
combine `np.eye` with a scalar multiply), q213 (one-hot basis vector — a
zeros canvas plus one assignment).

## Misconceptions

- **"`np.zeros(2, 3)` makes a 2×3 array."** — It raises a `TypeError`: the
  second positional argument is the dtype. Multi-dimensional shapes are ONE
  tuple: `np.zeros((2, 3))`.
- **"Constructors give me integers if I write `np.ones(5)`."** — The default
  dtype is `float64` regardless of how the values look. If the grader (or your
  model) needs ints or bools, say so with `dtype=`.
- **"`np.empty` means an array with no elements."** — It means *uninitialized
  memory* of the full requested shape: garbage values, not zeros, not empty.
  Use `np.zeros` unless you will overwrite everything.
