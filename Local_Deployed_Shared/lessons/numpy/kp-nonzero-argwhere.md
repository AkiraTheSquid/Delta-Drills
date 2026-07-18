---
kc: numpy.nonzero-argwhere
title: Finding positions — nonzero and argwhere
supporting: [numpy.boolean-masking]
new_syntax: []
faded: [40]
guided: [86]
independent: [110]
---

## Concept

Masks answer "*which values* satisfy the condition?". Just as often you need
"*at which positions*?" — indices, not values. Two functions return them, in
two different layouts; choosing is a matter of what you'll do with the
coordinates:

- **`np.nonzero(z)`** returns a **tuple of index arrays, one per axis**.
  For 1-D: a 1-tuple holding the positions. For 2-D: `(rows, cols)` — two
  parallel arrays where `(rows[i], cols[i])` is the i-th hit in row-major
  order. This layout plugs straight back into indexing: `z[np.nonzero(z)]`
  gives the nonzero values. Because a boolean is just 0/1, `np.nonzero(mask)`
  works on any condition: `np.nonzero(x > 5)`.
- **`np.argwhere(z)`** returns **one (k, n_dims) array of coordinate rows** —
  for 2-D, k rows of `[row, col]`. This layout is for *reading* coordinates
  (iterate them, report them, save them); it does NOT plug back into
  indexing directly.

Same information, transposed packaging: `np.argwhere(z)` equals
`np.transpose(np.nonzero(z))`.

Once you have per-axis index arrays, whole-array geometry questions become
min/max over them — e.g. the **bounding box** of the nonzero region of a 2-D
mask is `rows.min()..rows.max()` × `cols.min()..cols.max()`.

One wrinkle for plain Python inputs: these functions accept lists, but when a
task hands you a list and wants array semantics, convert explicitly with
`np.asarray(x)` first — it's free for arrays and makes intent visible.

## Worked example

Task: find where a vector is nonzero (numpy's native tuple form), then list
the (row, col) coordinates of every nonzero cell in a 2-D mask.

```python
import numpy as np

x = np.asarray([0, 3, 0, 0, 7, 0, -2])

# nonzero returns a TUPLE of index arrays — one per axis, so 1-D gives a
# 1-tuple. This is the same structure you get from np.where(cond).
pos = np.nonzero(x)
assert isinstance(pos, tuple) and len(pos) == 1
assert pos[0].tolist() == [1, 4, 6]

# The tuple layout plugs back into indexing: the nonzero VALUES.
assert x[pos].tolist() == [3, 7, -2]

# 2-D: argwhere gives coordinate ROWS — one [row, col] pair per hit,
# in row-major scan order. Made for reading, not for indexing.
z = np.array([[0, 1],
              [1, 0]])
coords = np.argwhere(z)
assert coords.tolist() == [[0, 1], [1, 0]]

# Same info via nonzero, as parallel per-axis arrays:
rows, cols = np.nonzero(z)
assert rows.tolist() == [0, 1] and cols.tolist() == [1, 0]

# Geometry from index arrays: bounding box of the nonzero region.
assert (rows.min(), rows.max(), cols.min(), cols.max()) == (0, 1, 0, 1)
```

Why each step:

1. Checking `isinstance(pos, tuple)` once makes the 1-D quirk stick: the
   answer to "where?" is always per-axis arrays, even when there's one axis.
   (Some drills require exactly this tuple structure.)
2. `x[pos]` closes the loop — positions in nonzero-format are *designed* to be
   used as indices.
3. The rows/cols unpacking plus min/max shows why the tuple layout wins for
   computation: each axis's coordinates are already a vector you can reduce.

## Faded practice

### q40
Positions of all nonzero entries of a plain Python list, in numpy's
tuple-of-index-arrays structure.

```python starter
import numpy as np

def solve(x):
    """Return numpy's tuple-of-index-arrays for nonzero positions of list x."""
    return np.nonzero(np._____(x))
```

```python solution
import numpy as np

def solve(x):
    """Return numpy's tuple-of-index-arrays for nonzero positions of list x."""
    return np.nonzero(np.asarray(x))
```

## Guided practice

### q86
1. You need a (k, 2) array of [row, col] coordinates in scan order — which of
   the two functions produces coordinate rows rather than per-axis arrays?
2. Zero rows for an all-zero mask happens automatically — an empty result is
   shape (0, 2).
3. One call, no post-processing.

## Independent practice

From the drill bank: q110 (bounding box of the nonzero region as four plain
ints — get per-axis index arrays, then reduce each).

## Misconceptions

- **"nonzero and argwhere return the same thing."** — Same information,
  different layout: per-axis tuple (indexing-ready) vs coordinate rows
  (reading-ready). Graders check the structure, and code that indexes with
  argwhere output breaks.
- **"For a 1-D array, nonzero returns the indices directly."** — It returns a
  1-TUPLE containing the index array. Unpack with `pos[0]` or
  `idx, = np.nonzero(x)` when you want the bare array.
- **"I need a loop to find positions matching a condition."** —
  `np.nonzero(condition)` does it: a mask is already the 0/1 array nonzero
  scans.
