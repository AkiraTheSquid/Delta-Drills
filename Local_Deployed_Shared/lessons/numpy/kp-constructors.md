---
kc: numpy.constructors
title: Array constructors — zeros, ones, full, eye, *_like
supporting: [numpy.ndarray-model]
new_syntax: []
faded: [227, 41]
guided: [212]
independent: [225, 228, 50, 213]
---

## Concept

`np.array` converts data you already have. Just as often you need an array
**built from scratch** — a canvas of zeros to fill in, a mask of ones, an
identity matrix. NumPy has one constructor per pattern, and they all share the
same calling convention:

> **constructor(shape, dtype=...)** — say how big, optionally say what type.

The core family:

- **`np.zeros(shape)`** — all entries `0.0`. The default "empty canvas".
- **`np.ones(shape)`** — all entries `1.0`.
- **`np.full(shape, v)`** — all entries equal to your value `v`.
- **`np.eye(n)`** — the n×n identity matrix: `1.0` on the main diagonal,
  `0.0` elsewhere.
- **`np.empty(shape)`** — allocates *without initializing* (contents are
  whatever bytes were in memory). Only worth it when you will overwrite every
  entry immediately.

Two conventions to internalize:

1. **Shape is a tuple for anything above 1-D.** `np.zeros(5)` is a length-5
   vector, but a 2-D array needs `np.zeros((2, 3))` — note the inner
   parentheses. `np.zeros(2, 3)` is a `TypeError`, because the second
   positional argument is the dtype.
2. **The default dtype is `float64`,** even though the values print like
   integers you might not want. Pass `dtype=` to override:
   `np.ones((2, 2), dtype=bool)` is a matrix of `True`.

Finally, the **`*_like` variants** (`np.zeros_like(x)`, `np.ones_like(x)`,
`np.full_like(x, v)`) copy both the shape *and the dtype* from an existing
array — the right tool whenever the question is "give me a blank array shaped
like this one".

## Worked example

Task: build a 3×4 scoreboard of zeros, an all-`True` mask of the same shape,
and a blank integer array matching an existing one.

```python
import numpy as np

# 1. A 2-D shape must be passed as ONE tuple argument: (rows, cols).
board = np.zeros((3, 4))
assert board.shape == (3, 4)
# 2. We didn't ask for a dtype, so we got the float default.
assert board.dtype == np.float64

# 3. Same shape, but boolean: ones + dtype=bool gives all True.
#    (np.full((3, 4), True) is equivalent.)
mask = np.ones((3, 4), dtype=bool)
assert mask.all() and mask.dtype == np.bool_

# 4. "Blank array shaped like x" — zeros_like copies shape AND dtype,
#    so an int32 input yields an int32 result, not the float default.
x = np.array([[3, -1, 4], [1, 5, -9]], dtype=np.int32)
blank = np.zeros_like(x)
assert blank.shape == x.shape
assert blank.dtype == np.int32
```

Why each step:

1. The tuple-shape convention (`(3, 4)`) is the single most common beginner
   syntax error with constructors — the constructor takes *one* shape
   argument, not one argument per dimension.
2. Checking `dtype` right after construction is the habit that catches the
   float-by-default surprise before it propagates.
3. Any constant array is either `ones`/`zeros` with a dtype, or `np.full` with
   the value — pick whichever states your intent most directly.
4. When a task says "matching shape and dtype of the input", reaching for
   `*_like` is both shorter and safer than reading off `.shape` and `.dtype`
   yourself.

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

## Guided practice

### q212
1. You need a rows×cols array where every entry is the boolean `True`.
2. `np.ones` gives every entry the value 1 — and 1 as a boolean is `True`.
   You just have to ask for the right dtype.
3. Remember the shape convention: both sizes go inside one tuple,
   `np.ones((rows, cols), dtype=...)`.

## Independent practice

From the drill bank: q225 (all-ones vector), q228 (identity matrix),
q50 (v on the diagonal — combine `np.eye` with a scalar multiply),
q213 (one-hot basis vector — a zeros canvas plus one assignment).

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
