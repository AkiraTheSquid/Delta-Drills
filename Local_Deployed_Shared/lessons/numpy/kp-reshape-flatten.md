---
kc: numpy.reshape-flatten
title: Reshape, flatten, and element order
supporting: [numpy.ndarray-model, numpy.ranges]
new_syntax: []
faded: [46]
guided: [36]
independent: [23]
---

## Concept

A tensor's data is one flat block of memory; the shape is just metadata saying
how to read it. **Reshaping changes the metadata without touching the data** —
which is why it's usually free (no copy) and why the one hard rule is:

> the new shape's element count must equal the old one
> (`2 × 6 = 12 = 3 × 4` ✓, but 12 → `(5, 3)` ✗ raises an error).

The two directions:

- **`x.reshape(shape)`** — reinterpret the flat data as a new shape.
  A convenience worth memorizing: one dimension may be **`-1`**, meaning
  "compute this one for me": `x.reshape(3, -1)` figures out the columns.
- **`x.flatten()`** — collapse any shape back to 1-D.

You will also meet **`x.view(shape)`**, which is reshape's stricter sibling: it
*only* ever re-labels the existing memory and raises if that is impossible.
`reshape` falls back to copying in that case. Prefer `reshape` unless you
specifically want the error.

The question that makes reshape make sense is: **in what order do elements
fill the new shape?** PyTorch's order is **row-major ("C order")**: the
*last* axis varies fastest. Reading a 2-D tensor in row-major order means
walking across row 0 left to right, then row 1, and so on. So

```python no-run
t.arange(6).reshape(2, 3)   # → [[0, 1, 2],
                            #    [3, 4, 5]]
```

fills row 0 first. This single fact explains most reshape results, including
higher-dimensional ones: `reshape(2, 2, 3)` fills the last axis (length 3)
fastest, the first axis slowest.

Column-major ("Fortran") order — first axis fastest — has **no keyword** in
PyTorch. NumPy spells it `order='F'`; here you get it by changing the axis
order first and then flattening: `z.T.flatten()` reads the matrix column by
column.

## Worked example

Task: build the classic "counting matrix" — an n×n tensor containing 0..n²-1
reading left-to-right, top-to-bottom — then flatten it back both ways.

```python
import torch as t

n = 3
# Step 1: make the flat sequence 0..8. It's 1-D — shape (9,).
flat = t.arange(n * n)

# Step 2: reshape to (3, 3). Row-major fill means 0,1,2 land in row 0 —
# exactly the "reading order" the task describes. No data is copied.
grid = flat.reshape(n, n)
assert grid.tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

# The -1 shortcut: "3 rows, you work out the columns."
assert grid.tolist() == t.arange(9).reshape(3, -1).tolist()

# Step 3: flatten undoes it — row-major walk gives back 0..8 in order.
assert grid.flatten().tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8]

# Column-major walk reads DOWN each column instead. No order= keyword
# exists, so transpose first and let the row-major walk do the work.
assert grid.T.flatten().tolist() == [0, 3, 6, 1, 4, 7, 2, 5, 8]
```

Why each step:

1. `arange` + `reshape` is the standard two-step for "matrix containing the
   numbers 0..k in reading order" — generate the flat values, then organize
   them. It works because reshape's fill order IS reading order.
2. `-1` earns its keep when one dimension is derived: you state the part you
   know and let PyTorch check the arithmetic.
3. The two flattens show that "flatten" is not one operation until you say the
   order. The default matches how the matrix prints; the column-by-column read
   is a transpose away.

## Faded practice

### q46
n×n matrix containing 0..n²-1 in reading order.

```python starter
import torch as t

def solve(n):
    """Return the n x n matrix of 0..n*n-1 in row-major reading order."""
    return t.arange(_____).reshape(_____, _____)
```

```python solution
import torch as t

def solve(n):
    """Return the n x n matrix of 0..n*n-1 in row-major reading order."""
    return t.arange(n * n).reshape(n, n)
```

## Guided practice

### q36
1. You get a 1-D tensor and a target 3-D shape whose product equals its length —
   this is a pure reorganize-the-metadata task.
2. "Same row-major order (the last axis varies fastest)" in the prompt is
   describing reshape's DEFAULT fill order — no reordering needed on your part.
3. One method call on the input tensor does the whole job.

## Independent practice

From the drill bank: q23 (list all entries in COLUMN-major order — there is no
`order=` keyword to reach for, so which axis rearrangement gets you there?).

## Misconceptions

- **"Reshape can rearrange values."** — Reshape never *reorders* values: the
  flat row-major sequence of elements is identical before and after, and only
  the shape metadata changes. (It may still copy that sequence into fresh
  memory when the input is a non-contiguous view — same order, new buffer.)
  If the values need to move (transpose, sort, flip), reshape is the wrong
  tool.
- **"reshape(3, 4) on 11 elements will pad or truncate."** — It raises a
  `RuntimeError`. Counts must match exactly; `-1` only *derives* a dimension,
  it can't invent elements.
- **"Flattening reads down the columns."** — The order is row-major: across
  row 0 first. Column-major means transposing first.
- **"`order='F'` works, like in NumPy."** — PyTorch's `flatten` takes no order
  argument at all. `z.T.flatten()` is the column-major read.
