---
kc: numpy.window-stencil
title: 2-D windows and stencils — block sums, correlation, Life
supporting: [numpy.sliding-windows, numpy.pad-borders, numpy.cumulative-diff]
new_syntax: []
faded: [195]
guided: [156]
independent: [167, 176, 196, 209, 126]
---

## Concept

The 1-D window toolkit generalizes to grids, where three related shapes of
task appear:

**Non-overlapping blocks — `split`, once per axis.**
`z.split(k, dim=0)` cuts the rows into consecutive groups of k and hands back
a tuple of chunks; summing each over dim 0 and stacking collapses every k rows
into one. Do it along dim 0, then dim 1, and you have k×k block sums — and
because `split` just cuts every k, **a trailing partial block handles itself**
when the shape isn't a multiple of k (the last chunk is simply shorter). The
reshape-based alternative `z.reshape(r//k, k, c//k, k).sum(dim=(1, 3))` is
neater but demands exact divisibility.

(NumPy spells this `np.add.reduceat(z, starts, axis=0)`, passing the start
indices explicitly. Torch has no `reduceat`; `split` says the same thing by
giving the chunk SIZE instead of the cut points.)

**Overlapping windows — `Tensor.unfold`, once per axis.**
`z.unfold(0, h, 1).unfold(1, w, 1)` yields a 4-D tensor indexed by
(window-row, window-col, within-row, within-col): every h×w submatrix. Each
`unfold(dim, size, step)` call windows ONE dimension, so a 2-D window is two
calls — and the windowed length lands at the END, which is why the within-window
axes come last. Reduce those last two axes for sliding statistics; flatten the
first two for "all submatrices as a stack". **Cross-correlation** ("slide a
kernel, sum the products") is the window view times the kernel, summed over the
window axes: `(wins * kern).sum(dim=(-2, -1))` — the 'valid' convolution shape
(r−h+1, c−w+1) falls out of the view.

**Stencils — shifted neighbors via padding.**
When each cell needs its NEIGHBORS' aggregate (Game of Life counts,
smoothing), pad the grid (borders KP: dead cells beyond every edge = pad
with 0), then either take the window view of the padded grid, or sum the
eight shifted slices. From neighbor counts, Life's update is pure masking:
survive = alive & (count == 2 or 3); birth = dead & (count == 3).

The **integral image** (summed-area table) belongs to this family as the
precomputation that answers arbitrary rectangle-sum queries in O(1):
`t.cumsum(t.cumsum(z, dim=0), dim=1)` — cumsum down, then across, so
entry [i, j] holds the sum of the whole rectangle from the origin. (`t.cumsum`
REQUIRES the dim; there is no flatten-by-default form.)

## Worked example

Task: k×k block sums of a grid whose size ISN'T a multiple of k; then a
valid cross-correlation.

```python
import torch as t

z = t.ones((5, 5), dtype=t.int64)

# Block sums with k=2 on a 5x5: chunks of 2, 2, 1 per axis — the last
# "block" is the ragged 1-wide remainder, handled automatically.
r = t.stack([c.sum(dim=0) for c in z.split(2, dim=0)])            # row bands
blocks = t.stack([c.sum(dim=1) for c in r.split(2, dim=1)], dim=1)  # col bands
assert blocks.tolist() == [[4, 4, 2],
                           [4, 4, 2],
                           [2, 2, 1]]     # corner block is just 1 cell

# Valid cross-correlation: every kernel-shaped window times the kernel.
x = t.arange(16, dtype=t.float32).reshape(4, 4)
kern = t.tensor([[1.0, 0.0],
                 [0.0, 1.0]])             # picks top-left + bottom-right
wins = x.unfold(0, 2, 1).unfold(1, 2, 1)              # (3, 3, 2, 2)
corr = (wins * kern).sum(dim=(-2, -1))                # reduce window axes
assert tuple(corr.shape) == (3, 3)
assert corr[0, 0] == x[0, 0] + x[1, 1]                # 0 + 5
assert corr[2, 2] == x[2, 2] + x[3, 3]                # 10 + 15
print("2x2 block sums of a 5x5 — last band is the ragged remainder:")
print(blocks)
print("windows", tuple(wins.shape), "-> correlation", tuple(corr.shape))
print(corr)
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
import torch as t

def solve(z, k):
    """Block sums tiling from the top-left; partial edge blocks count."""
    rows = t.stack([chunk.sum(dim=0) for chunk in z.split(k, dim=_____)])
    return t.stack([chunk.sum(dim=1) for chunk in rows.split(k, dim=_____)], dim=1)
```

```python solution
import torch as t

def solve(z, k):
    """Block sums tiling from the top-left; partial edge blocks count."""
    rows = t.stack([chunk.sum(dim=0) for chunk in z.split(k, dim=0)])
    return t.stack([chunk.sum(dim=1) for chunk in rows.split(k, dim=1)], dim=1)
```

## Guided practice

### q156
1. Entry [i, j] = sum of everything above-and-left inclusive — a cumulative
   quantity, in TWO directions.
2. One cumsum handles one direction; the directions compose.
3. `t.cumsum(t.cumsum(z, dim=0), dim=1)` — check the bottom-right entry
   equals z.sum().

## Independent practice

From the drill bank: q167 (ALL k×k contiguous submatrices as a stack — window
view, then merge the two window-index axes), q176 (per-window maxima —
window view + max over the inside axes), q196 (valid cross-correlation with
an arbitrary kernel), q209 (Game of Life step — pad, count the eight
neighbors, apply the birth/survival masks).

Also from the bank: q126 (block sums over non-overlapping bh x bw tiles).

## Misconceptions

- **"Block operations require divisible shapes."** — reshape-based blocking
  does; `split` doesn't — the last chunk is simply shorter than the rest.
  Check the task for ragged edges before choosing.
- **"2-D windows mean nested loops over (i, j)."** — Two `unfold` calls
  materialize every window in the last two axes; statistics are reductions
  from there.
- **"Convolution and correlation are interchangeable."** — Convolution flips
  the kernel; correlation doesn't. The window-view formula computes
  CORRELATION; flip the kernel (`kern.flip(0).flip(1)` — torch rejects a
  negative slice step) if a task genuinely wants convolution. Drills usually
  say "cross-correlation" precisely to spare you the flip.
