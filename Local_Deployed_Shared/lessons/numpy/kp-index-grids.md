---
kc: numpy.index-grids
title: Index-pattern grids — checkerboards and coordinate masks
supporting: [numpy.slicing-views, numpy.constructors, numpy.boolean-masking]
new_syntax: []
faded: [2, 116]
guided: []
independent: [112, 72, 11]
---

## Concept: periodic patterns — strided slice assignment

A family of tasks asks you to build or mask a matrix based on **each cell's
coordinates**: checkerboards, distance-from-center maps, bands around the
diagonal. When the pattern has a **fixed period**, strided slices do it.

Slices accept a step, so `z[::2]` is "every even row" and `z[1::2, ::2]` is
"odd rows, even columns". A checkerboard is exactly two such assignments on
a zeros canvas:

```python no-run
z[1::2, ::2] = 1    # odd rows, even columns
z[::2, 1::2] = 1    # even rows, odd columns
```

(`z[::2]` is every SECOND row, not the first two — `z[:2]` is that. Step
lives in the third slot: start:stop:step.)

## Worked example

```python
import numpy as np

# Checkerboard, period-2 pattern -> strided slice assignment.
rows, cols = 3, 4
z = np.zeros((rows, cols), dtype=int)
z[1::2, ::2] = 1        # cells where row is odd  and col is even
z[::2, 1::2] = 1        # cells where row is even and col is odd
assert z.tolist() == [[0, 1, 0, 1],
                      [1, 0, 1, 0],
                      [0, 1, 0, 1]]
# Sanity: (i + j) odd <=> cell is 1 — exactly "no equal neighbors".
```

Why: the two assignments partition the 1-cells by row parity — walking one
cell ("row 1 is odd, col 0 is even → 1") verifies the slice choice faster
than staring at the pattern.

## Faded practice

### q2
Checkerboard of 0s and 1s, 0 in the top-left corner.

```python starter
import numpy as np

def solve(rows, cols):
    """Checkerboard with z[0,0] == 0, alternating both directions."""
    z = np.zeros((rows, cols), dtype=int)
    z[1::2, _____] = 1
    z[_____, 1::2] = 1
    return z
```

```python solution
import numpy as np

def solve(rows, cols):
    """Checkerboard with z[0,0] == 0, alternating both directions."""
    z = np.zeros((rows, cols), dtype=int)
    z[1::2, ::2] = 1
    z[::2, 1::2] = 1
    return z
```

## Concept: coordinate formulas — ogrid arithmetic

When the pattern is a **FORMULA in i and j**, build the row-index and
column-index vectors and lean on their shapes: `np.ogrid[:n, :n]` returns a
**column** `y` of shape (n, 1) and a **row** `x` of shape (1, n). Any
arithmetic between them produces the full (n, n) matrix of `f(i, j)` values
— each cell computed from its own coordinates. (WHY a (n,1)-by-(1,n)
operation yields (n,n) is broadcasting, next lesson's opening KP — here, use
it as "the coordinate-grid recipe".) Examples:

- Manhattan distance from the center: `np.abs(y - c) + np.abs(x - c)`.
- Diagonal band mask: `np.abs(y - x) <= 1` — True within one step of the
  main diagonal; multiply by `z` or use as a mask to keep the band.
- "Every cell = its row index": `y + 0 * x` (or explicit `np.repeat`).

The decision rule: periodic pattern → strided slices; coordinate formula →
ogrid arithmetic. Both build the structure without visiting cells in Python.

## Worked example

```python
import numpy as np

# Coordinate formula -> ogrid. y is a column (n,1), x is a row (1,n).
n = 3
y, x = np.ogrid[:n, :n]
assert y.shape == (3, 1) and x.shape == (1, 3)
c = n // 2
manhattan = np.abs(y - c) + np.abs(x - c)
assert manhattan.tolist() == [[2, 1, 2],
                              [1, 0, 1],
                              [2, 1, 2]]

# Band mask: cell (i, j) survives iff |i - j| <= 1.
m = np.arange(16).reshape(4, 4)
band = np.abs(np.arange(4)[:, None] - np.arange(4)[None, :]) <= 1
kept = m * band
assert kept.tolist() == [[0, 1, 0, 0],
                         [4, 5, 6, 0],
                         [0, 9, 10, 11],
                         [0, 0, 14, 15]]
```

Why: the SHAPES carry the meaning — the column `y` varies down, the row `x`
varies across, and combining them touches every (i, j) pair exactly once.
The band example shows formula→mask→apply; diagonal bands, wedges, and
"within k of the anti-diagonal" all fall to the same three moves.

## Faded practice

### q116
Manhattan distance of each cell from the center of an odd n×n grid.

```python starter
import numpy as np

def solve(n):
    """Entry [i, j] = |i - n//2| + |j - n//2|."""
    y, x = np.ogrid[:n, :n]
    c = n // 2
    return np.abs(y - c) _____ np.abs(x - c)
```

```python solution
import numpy as np

def solve(n):
    """Entry [i, j] = |i - n//2| + |j - n//2|."""
    y, x = np.ogrid[:n, :n]
    c = n // 2
    return np.abs(y - c) + np.abs(x - c)
```

## Independent practice

From the drill bank: q112 (keep values within one step of the main diagonal —
band mask times the matrix), q72 (checkerboard with a CHOSEN top-left value —
same slices, think about which parity gets 1... or an (i+j) formula),
q11 (every entry equals its own row index — the simplest coordinate formula).

## Misconceptions

- **"Patterned matrices need nested loops."** — Periodic patterns are strided
  slice assignments; coordinate formulas are ogrid arithmetic. Python-level
  cell visits are never required in this family.
- **"ogrid returns full matrices."** — It returns SKINNY vectors — shape
  (n, 1) and (1, n) — that expand only when combined. (The full-matrix
  sibling is `np.mgrid`/`meshgrid`; ogrid is the memory-light default.)
- **"`z[::2]` means the first two rows."** — It means every SECOND row
  (step 2, from row 0). `z[:2]` is the first two. Step lives in the third
  slot: start:stop:step.
