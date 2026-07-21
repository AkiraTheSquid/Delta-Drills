---
kc: numpy.tile-repeat-meshgrid
title: Tiling and repetition — tile, repeat, meshgrid
supporting: [numpy.reshape-flatten]
new_syntax: []
concepts: [repeat-elements, tile-blocks, coordinate-grids]
faded: [35, 34, 29]
guided: [217]
independent: [69]
---

## Concept: Repeat each element with np.repeat

`np.repeat(x, k)` repeats each element of `x` before moving to the next
element.

```text
[1, 2, 3] → [1, 1, 1, 2, 2, 2, 3, 3, 3]
```

Use it when output groups copies of each individual value. `k` may also be an
array containing one repetition count per element.

## Watch out

Read expected output from left to right. If one value finishes all its copies
before the next value appears, use `repeat`.

## Worked example: Repeat every reading three times

Task: repeat each element of `[4, 7, 9]` three consecutive times.

```python
import numpy as np

x = np.array([4, 7, 9])
repeated = np.repeat(x, 3)

assert repeated.tolist() == [4, 4, 4, 7, 7, 7, 9, 9, 9]
print(repeated)
```

Why: `repeat` completes three copies of `4`, then three copies of `7`, then
three copies of `9`.

## Faded practice

### q35
Each element appears `k` times consecutively.

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

## Concept: Repeat a whole block with np.tile

`np.tile(x, k)` repeats `x` as one complete block.

```text
[1, 2, 3] → [1, 2, 3, 1, 2, 3]
```

For a 2-D array, pass one repetition count per axis:
`np.tile(block, (row_repeats, column_repeats))`.

## Watch out

The repetition tuple follows axis order. First number repeats rows; second
number repeats columns.

## Worked example: Build a checkerboard mosaic

Task: repeat one 2×2 checker block twice vertically and twice horizontally.

```python
import numpy as np

block = np.array([[0, 1],
                  [1, 0]])
mosaic = np.tile(block, (2, 2))

assert mosaic.tolist() == [[0, 1, 0, 1],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [1, 0, 1, 0]]
print(mosaic)
```

Why: `(2, 2)` lays out two copies along rows and two copies along columns.

## Faded practice

### q34
Repeat the entire sequence `k` times end to end.

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

## Concept: Build coordinate grids with np.meshgrid

`np.meshgrid(x, y)` turns two 1-D coordinate axes into two 2-D matrices.

- `X` repeats the `x` coordinates across every row.
- `Y` repeats each `y` coordinate down its matching row.

Pairing `X[i, j]` with `Y[i, j]` gives one point from every possible
combination of `x` and `y`.

## Watch out

`meshgrid` returns one coordinate matrix per input axis—not one array of
coordinate pairs. Both output matrices have shape `(len(y), len(x))`.

## Worked example: Enumerate a rectangular grid

Task: build coordinate matrices for three x-values and two y-values.

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([10.0, 20.0])
X, Y = np.meshgrid(x, y)

assert X.tolist() == [[1.0, 2.0, 3.0],
                      [1.0, 2.0, 3.0]]
assert Y.tolist() == [[10.0, 10.0, 10.0],
                      [20.0, 20.0, 20.0]]
print(X)
print(Y)
```

Why: each column chooses an x-coordinate; each row chooses a y-coordinate.
Their matching positions enumerate the full grid.

## Faded practice

### q29
Return coordinate matrices built from 1-D arrays `x` and `y`.

```python starter
import numpy as np

def solve(x, y):
    """Return the tuple (X, Y) of 2-D coordinate grids."""
    return np._____(x, y)
```

```python solution
import numpy as np

def solve(x, y):
    """Return the tuple (X, Y) of 2-D coordinate grids."""
    return np.meshgrid(x, y)
```

## Guided practice

### q217
Tile a 2-D block `reps_r` times vertically and `reps_c` times horizontally.
Use one `np.tile` call with a two-value repetition tuple.

## Independent practice

From the drill bank: q69 (pass an array of repetition counts to `np.repeat`,
so each input element may receive a different number of copies).
