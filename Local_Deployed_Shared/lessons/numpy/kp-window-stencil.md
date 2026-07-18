---
kc: numpy.window-stencil
title: 2-D windows and stencils — block sums, correlation, Life
supporting: [numpy.sliding-windows, numpy.pad-borders, numpy.cumulative-diff]
new_syntax: []
faded: [195]
guided: [156]
independent: [167, 176, 196, 209]
---

## Concept

The 1-D window toolkit generalizes to grids, where three related shapes of
task appear:

**Non-overlapping blocks — `np.add.reduceat`, once per axis.**
`np.add.reduceat(z, starts, axis=0)` sums SEGMENTS of rows beginning at the
given start indices — so with `starts = np.arange(0, nrows, k)` it collapses
every k rows into one. Apply along axis 0, then axis 1, and you have k×k
block sums — and because reduceat just segments at the starts, **trailing
partial blocks handle themselves** when the shape isn't a multiple of k
(the reshape-based alternative `z.reshape(r//k, k, c//k, k).sum(axis=(1,3))`
is neat but demands exact divisibility).

**Overlapping windows — 2-D `sliding_window_view`.**
`sliding_window_view(z, (h, w))` yields a 4-D array indexed by
(window-row, window-col, within-row, within-col): every h×w submatrix.
Reduce the last two axes for sliding statistics; flatten the first two for
"all submatrices as a stack". **Cross-correlation** ("slide a kernel, sum
the products") is the window view times the kernel, summed over the window
axes: `(wins * kern).sum(axis=(-2, -1))` — the 'valid' convolution shape
(r−h+1, c−w+1) falls out of the view.

**Stencils — shifted neighbors via padding.**
When each cell needs its NEIGHBORS' aggregate (Game of Life counts,
smoothing), pad the grid (borders KP: dead cells beyond every edge = pad
with 0), then either take the window view of the padded grid, or sum the
eight shifted slices. From neighbor counts, Life's update is pure masking:
survive = alive & (count == 2 or 3); birth = dead & (count == 3).

The **integral image** (summed-area table) belongs to this family as the
precomputation that answers arbitrary rectangle-sum queries in O(1):
`np.cumsum(np.cumsum(z, axis=0), axis=1)` — cumsum down, then across, so
entry [i, j] holds the sum of the whole rectangle from the origin.

## Worked example

Task: k×k block sums of a grid whose size ISN'T a multiple of k; then a
valid cross-correlation.

```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

z = np.ones((5, 5), dtype=int)

# Block sums with k=2 on a 5x5: starts 0, 2, 4 per axis — the last
# "block" is the ragged 1-wide remainder, handled automatically.
r = np.add.reduceat(z, np.arange(0, 5, 2), axis=0)   # collapse row bands
blocks = np.add.reduceat(r, np.arange(0, 5, 2), axis=1)  # then col bands
assert blocks.tolist() == [[4, 4, 2],
                           [4, 4, 2],
                           [2, 2, 1]]     # corner block is just 1 cell

# Valid cross-correlation: every kernel-shaped window times the kernel.
x = np.arange(16, dtype=float).reshape(4, 4)
kern = np.array([[1.0, 0.0],
                 [0.0, 1.0]])             # picks top-left + bottom-right
wins = sliding_window_view(x, kern.shape)             # (3, 3, 2, 2)
corr = (wins * kern).sum(axis=(-2, -1))               # reduce window axes
assert corr.shape == (3, 3)
assert corr[0, 0] == x[0, 0] + x[1, 1]                # 0 + 5
assert corr[2, 2] == x[2, 2] + x[3, 3]                # 10 + 15
```

Why each step:

1. Reading the block-sum output against the geometry (4 = full 2×2 of ones,
   2 = ragged edge, 1 = corner) verifies the starts arithmetic better than
   any formula — sketch the 5×5 with cuts at rows/cols 0, 2, 4.
2. The window view's 4-D shape is worth pausing on: axes 0-1 say WHICH
   window, axes 2-3 say WHERE INSIDE it. Reductions always target the
   inside pair — `axis=(-2, -1)` names them robustly.
3. In the correlation, broadcasting aligns kern (2, 2) against wins
   (3, 3, 2, 2) from the right — the same right-alignment rule, now doing
   image processing.

## Faded practice

### q195
Sums of non-overlapping k×k blocks, ragged edges included.

```python starter
import numpy as np

def solve(z, k):
    """Block sums tiling from the top-left; partial edge blocks count."""
    r = np.add.reduceat(z, np.arange(0, z.shape[0], k), axis=_____)
    return np.add.reduceat(r, np.arange(0, z.shape[1], k), axis=_____)
```

```python solution
import numpy as np

def solve(z, k):
    """Block sums tiling from the top-left; partial edge blocks count."""
    r = np.add.reduceat(z, np.arange(0, z.shape[0], k), axis=0)
    return np.add.reduceat(r, np.arange(0, z.shape[1], k), axis=1)
```

## Guided practice

### q156
1. Entry [i, j] = sum of everything above-and-left inclusive — a cumulative
   quantity, in TWO directions.
2. One cumsum handles one direction; the directions compose.
3. `np.cumsum(np.cumsum(z, axis=0), axis=1)` — check the bottom-right entry
   equals z.sum().

## Independent practice

From the drill bank: q167 (ALL k×k contiguous submatrices as a stack — window
view, then merge the two window-index axes), q176 (per-window maxima —
window view + max over the inside axes), q196 (valid cross-correlation with
an arbitrary kernel), q209 (Game of Life step — pad, count the eight
neighbors, apply the birth/survival masks).

## Misconceptions

- **"Block operations require divisible shapes."** — reshape-based blocking
  does; `add.reduceat` with arange starts doesn't — segments simply end
  where the next starts (or the array does). Check the task for ragged
  edges before choosing.
- **"2-D windows mean nested loops over (i, j)."** — One
  `sliding_window_view(z, (h, w))` call materializes every window as the
  last two axes; statistics are reductions from there.
- **"Convolution and correlation are interchangeable."** — Convolution flips
  the kernel; correlation doesn't. The window-view formula computes
  CORRELATION; flip the kernel (`kern[::-1, ::-1]`) if a task genuinely
  wants convolution. Drills usually say "cross-correlation" precisely to
  spare you the flip.
