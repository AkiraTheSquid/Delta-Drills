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
import torch as t

# Checkerboard, period-2 pattern -> strided slice assignment.
rows, cols = 3, 4
z = t.zeros((rows, cols), dtype=t.int64)
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
import torch as t

def solve(rows, cols):
    """Checkerboard with z[0,0] == 0, alternating both directions."""
    z = t.zeros((rows, cols), dtype=t.int64)
    z[1::2, _____] = 1
    z[_____, 1::2] = 1
    return z
```

```python solution
import torch as t

def solve(rows, cols):
    """Checkerboard with z[0,0] == 0, alternating both directions."""
    z = t.zeros((rows, cols), dtype=t.int64)
    z[1::2, ::2] = 1
    z[::2, 1::2] = 1
    return z
```

## Concept: coordinate formulas — index-vector arithmetic

When the pattern is a **FORMULA in i and j**, build the row-index and
column-index vectors and lean on their shapes: `t.arange(n)[:, None]` is a
**column** `y` of shape (n, 1) and `t.arange(n)[None, :]` is a **row** `x` of
shape (1, n). Any arithmetic between them produces the full (n, n) matrix of
`f(i, j)` values — each cell computed from its own coordinates. (WHY a
(n,1)-by-(1,n) operation yields (n,n) is broadcasting, next lesson's opening
KP — here, use it as "the coordinate-grid recipe".) Examples:

- Manhattan distance from the center: `t.abs(y - c) + t.abs(x - c)`.
- Diagonal band mask: `t.abs(y - x) <= 1` — True within one step of the
  main diagonal; multiply by `z` or use as a mask to keep the band.
- "Every cell = its row index": `y + 0 * x`.

If you would rather have both coordinates as full (n, n) matrices,
`t.meshgrid(t.arange(n), t.arange(n), indexing="ij")` returns exactly that —
`indexing="ij"` is the row-major convention these tasks assume, and leaving
it off gives you the transpose.

The decision rule: periodic pattern → strided slices; coordinate formula →
index-vector arithmetic. Both build the structure without visiting cells in
Python.

## Worked example

```python
import torch as t

# Coordinate formula -> index vectors. y is a column (n,1), x a row (1,n).
n = 3
y = t.arange(n)[:, None]
x = t.arange(n)[None, :]
assert tuple(y.shape) == (3, 1) and tuple(x.shape) == (1, 3)
c = n // 2
manhattan = t.abs(y - c) + t.abs(x - c)
assert manhattan.tolist() == [[2, 1, 2],
                              [1, 0, 1],
                              [2, 1, 2]]

# Band mask: cell (i, j) survives iff |i - j| <= 1.
m = t.arange(16).reshape(4, 4)
band = t.abs(t.arange(4)[:, None] - t.arange(4)[None, :]) <= 1
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
import torch as t

def solve(n):
    """Entry [i, j] = |i - n//2| + |j - n//2|."""
    y = t.arange(n)[:, None]
    x = t.arange(n)[None, :]
    c = n // 2
    return t.abs(y - c) _____ t.abs(x - c)
```

```python solution
import torch as t

def solve(n):
    """Entry [i, j] = |i - n//2| + |j - n//2|."""
    y = t.arange(n)[:, None]
    x = t.arange(n)[None, :]
    c = n // 2
    return t.abs(y - c) + t.abs(x - c)
```

## Independent practice

From the drill bank: q112 (keep values within one step of the main diagonal —
band mask times the matrix), q72 (checkerboard with a CHOSEN top-left value —
same slices, think about which parity gets 1... or an (i+j) formula),
q11 (every entry equals its own row index — the simplest coordinate formula).

## Misconceptions

- **"Patterned matrices need nested loops."** — Periodic patterns are strided
  slice assignments; coordinate formulas are index-vector arithmetic.
  Python-level cell visits are never required in this family.
- **"The coordinate vectors have to be full matrices."** — They are SKINNY —
  shape (n, 1) and (1, n) — and expand only when combined, so the (n, n)
  grid never exists in memory. `t.meshgrid` builds the full pair when you
  actually want it.
- **"`z[::2]` means the first two rows."** — It means every SECOND row
  (step 2, from row 0). `z[:2]` is the first two. Step lives in the third
  slot: start:stop:step.
