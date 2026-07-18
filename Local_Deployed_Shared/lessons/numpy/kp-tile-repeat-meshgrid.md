---
kc: numpy.tile-repeat-meshgrid
title: Tiling and repetition — tile, repeat, meshgrid
supporting: [numpy.reshape-flatten]
new_syntax: []
faded: [35, 34]
guided: [217]
independent: [69, 29]
---

## Concept

"Make copies of this data" splits into two genuinely different operations, and
telling them apart is the entire skill:

- **`np.repeat(x, k)` — each ELEMENT repeats.**
  `[1, 2, 3]`, k=3 → `[1, 1, 1, 2, 2, 2, 3, 3, 3]`.
  Elements stay grouped; the *fine structure* is duplicated.
  `repeat` also accepts a per-element count array:
  `np.repeat([1, 2, 3], [1, 3, 2])` → `[1, 2, 2, 2, 3, 3]` — and a count of 0
  drops that element.
- **`np.tile(x, k)` — the WHOLE BLOCK repeats.**
  `[1, 2, 3]`, k=2 → `[1, 2, 3, 1, 2, 3]`.
  The sequence as a unit is laid end to end; the *coarse structure* is
  duplicated.

Reading a task, ask: *does the example output interleave copies of each
element (repeat), or does the full pattern recur (tile)?*

`tile` generalizes to 2-D with a tuple of reps per axis:
`np.tile(block, (2, 3))` lays the block out in a 2-row × 3-column mosaic —
checkerboards, texture patterns, repeated stamps.

**`np.meshgrid(x, y)`** is repetition in service of coordinates: given axis
values `x` (n long) and `y` (m long), it returns two (m, n) matrices — `X`
where every row is a copy of `x`, and `Y` where every column is a copy of `y`
— so `(X[i, j], Y[i, j])` walks every point of the grid. Any time you must
evaluate `f(x, y)` over all combinations, meshgrid (or its lazy sibling
`np.ogrid`) sets up the coordinates.

## Worked example

Task: from the vector `[1, 2, 3]`, produce (a) each element three times in a
row, (b) the whole sequence twice, and (c) a 2×2 mosaic of a 2-D block.

```python
import numpy as np

x = np.array([1, 2, 3])

# (a) Fine structure duplicated: 1 1 1 2 2 2 3 3 3.
# Each element finishes all its copies before the next element starts.
assert np.repeat(x, 3).tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3]

# (b) Coarse structure duplicated: 1 2 3 1 2 3.
# The block as a unit is laid end to end.
assert np.tile(x, 2).tolist() == [1, 2, 3, 1, 2, 3]

# (c) 2-D tiling: reps per axis as a tuple -> a (2*2) x (2*2) mosaic.
block = np.array([[0, 1],
                  [1, 0]])
mosaic = np.tile(block, (2, 2))
assert mosaic.tolist() == [[0, 1, 0, 1],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [1, 0, 1, 0]]

# Coordinates over a grid: X repeats x along rows, Y repeats y down columns.
X, Y = np.meshgrid(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0]))
assert X.shape == Y.shape == (2, 3)
assert X.tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
assert Y.tolist() == [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]
```

Why each step:

1. (a) vs (b) is the repeat/tile discrimination — say the two output patterns
   aloud ("ones then twos then threes" vs "one-two-three, one-two-three")
   until the mapping is automatic.
2. The 2-D tile shows why the reps argument mirrors shape: one repetition
   count per axis, `(reps_rows, reps_cols)`.
3. In meshgrid's output, X varies along columns and Y varies along rows —
   pairing `X[i, j]` with `Y[i, j]` is what makes them coordinates.

## Faded practice

### q35
Each element appears k times in a row.

```python starter
import numpy as np

def solve(x, k):
    """Each element of x, repeated k times consecutively."""
    return np._____(x, k)
```

```python solution
import numpy as np

def solve(x, k):
    """Each element of x, repeated k times consecutively."""
    return np.repeat(x, k)
```

### q34
The entire sequence repeated k times end to end.

```python starter
import numpy as np

def solve(x, k):
    """The whole of x laid end-to-end k times."""
    return np._____(x, k)
```

```python solution
import numpy as np

def solve(x, k):
    """The whole of x laid end-to-end k times."""
    return np.tile(x, k)
```

## Guided practice

### q217
1. A 2-D block tiled reps_r times vertically and reps_c times horizontally —
   which of the two functions handles whole-block copies?
2. In 2-D, the repetition counts travel as a tuple, one per axis.
3. `np.tile(block, (reps_r, reps_c))` — check the output shape formula in the
   prompt against what tile produces.

## Independent practice

From the drill bank: q69 (per-element repetition COUNTS as an array — repeat's
second form), q29 (build coordinate matrices for a grid — the meshgrid
contract, note which input becomes rows vs columns).

## Misconceptions

- **"repeat and tile do the same thing."** — `repeat` duplicates each element
  in place (`1 1 2 2`); `tile` duplicates the whole block (`1 2 1 2`). The
  example output in the task tells you which one is being asked for.
- **"To tile in 2-D I call tile twice."** — Once, with a tuple:
  `np.tile(block, (r, c))`. The tuple mirrors the shape convention used
  everywhere else.
- **"meshgrid returns one array of pairs."** — It returns one matrix PER
  input axis, shaped (len(y), len(x)); the pairing is positional between them.
  Note the row count comes from the SECOND argument.
