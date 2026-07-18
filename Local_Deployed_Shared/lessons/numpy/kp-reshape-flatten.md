---
kc: numpy.reshape-flatten
title: Reshape, ravel, and element order
supporting: [numpy.ndarray-model, numpy.ranges]
new_syntax: []
faded: [46]
guided: [36]
independent: [23]
---

## Concept

An array's data is one flat block of memory; the shape is just metadata saying
how to read it. **Reshaping changes the metadata without touching the data** —
which is why it's free (no copy) and why the one hard rule is:

> the new shape's element count must equal the old one
> (`2 × 6 = 12 = 3 × 4` ✓, but 12 → `(5, 3)` ✗ raises an error).

The two directions:

- **`x.reshape(shape)`** — reinterpret the flat data as a new shape.
  A convenience worth memorizing: one dimension may be **`-1`**, meaning
  "compute this one for me": `x.reshape(3, -1)` figures out the columns.
- **`x.ravel()`** (or `x.flatten()`, which always copies) — collapse any shape
  back to 1-D.

The question that makes reshape make sense is: **in what order do elements
fill the new shape?** NumPy's default is **row-major order ("C order")**: the
*last* axis varies fastest. Reading a 2-D array in row-major order means
walking across row 0 left to right, then row 1, and so on. So

```python no-run
np.arange(6).reshape(2, 3)   # → [[0, 1, 2],
                             #    [3, 4, 5]]
```

fills row 0 first. This single fact explains most reshape results, including
higher-dimensional ones: `reshape(2, 2, 3)` fills the last axis (length 3)
fastest, the first axis slowest.

Column-major ("Fortran") order — first axis fastest — exists behind the
`order='F'` keyword on both `reshape` and `ravel`. You rarely want it, but
"read this matrix column by column" is exactly `z.ravel(order='F')` (or
equivalently, transpose then ravel).

## Worked example

Task: build the classic "counting matrix" — an n×n array containing 0..n²-1
reading left-to-right, top-to-bottom — then flatten it back both ways.

```python
import numpy as np

n = 3
# Step 1: make the flat sequence 0..8. It's 1-D — shape (9,).
flat = np.arange(n * n)

# Step 2: reshape to (3, 3). Row-major fill means 0,1,2 land in row 0 —
# exactly the "reading order" the task describes. No data is copied.
grid = flat.reshape(n, n)
assert grid.tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

# The -1 shortcut: "3 rows, you work out the columns."
assert grid.tolist() == np.arange(9).reshape(3, -1).tolist()

# Step 3: ravel undoes it — row-major walk gives back 0..8 in order.
assert grid.ravel().tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8]

# Column-major walk reads DOWN each column instead:
assert grid.ravel(order='F').tolist() == [0, 3, 6, 1, 4, 7, 2, 5, 8]
```

Why each step:

1. `arange` + `reshape` is the standard two-step for "matrix containing the
   numbers 0..k in reading order" — generate the flat values, then organize
   them. It works because reshape's fill order IS reading order.
2. `-1` earns its keep when one dimension is derived: you state the part you
   know and let NumPy check the arithmetic.
3. The two ravels show that "flatten" is not one operation until you say the
   order. Default C order matches how the matrix prints; `'F'` order is the
   column-by-column read.

## Faded practice

### q46
n×n matrix containing 0..n²-1 in reading order.

```python starter
import numpy as np

def solve(n):
    """Return the n x n matrix of 0..n*n-1 in row-major reading order."""
    return np.arange(_____).reshape(_____, _____)
```

```python solution
import numpy as np

def solve(n):
    """Return the n x n matrix of 0..n*n-1 in row-major reading order."""
    return np.arange(n * n).reshape(n, n)
```

## Guided practice

### q36
1. You get a 1-D array and a target 3-D shape whose product equals its length —
   this is a pure reorganize-the-metadata task.
2. "Same row-major order (the last axis varies fastest)" in the prompt is
   describing reshape's DEFAULT fill order — no reordering needed on your part.
3. One method call on the input array does the whole job.

## Independent practice

From the drill bank: q23 (list all entries in COLUMN-major order — recall
which keyword, or which transpose trick, changes the walk order).

## Misconceptions

- **"Reshape can rearrange values."** — Reshape never moves data; it only
  re-labels positions. The flat sequence of elements is identical before and
  after. If the values need to move (transpose, sort, roll), reshape is the
  wrong tool.
- **"reshape(3, 4) on 11 elements will pad or truncate."** — It raises
  `ValueError`. Counts must match exactly; `-1` only *derives* a dimension,
  it can't invent elements.
- **"Flattening reads down the columns."** — Default order is row-major: across
  row 0 first. Column-major is opt-in (`order='F'`).
