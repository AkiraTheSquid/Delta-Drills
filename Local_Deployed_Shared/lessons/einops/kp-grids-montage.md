---
kc: einops.grids-montage
title: Laying batches out as grids
supporting: [einops.split-axes, einops.merge-axes]
new_syntax: []
faded: [389]
guided: [364]
independent: [329, 382, 318, 381, 322, 371, 531]
integrated: [909, 910, 911, 912]
---

## Concept

"Tile these b images into a g1-row grid" — the montage — is the flagship
split-then-merge pattern:

> `'(g1 g2) h w c -> (g1 h) (g2 w) c', g1=rows`

Read it in two moves:

1. **Split the batch into grid coordinates.** `(g1 g2)` on the left declares
   the batch axis packs g1 rows of g2 images, ROW-MAJOR (g1 slow: images
   0..g2-1 form grid row 0). One keyword fixes both factors.
2. **Merge each grid coordinate with its image dimension.** `(g1 h)`: grid
   row with within-image height — grid row 0's images occupy output rows
   0..h-1. `(g2 w)`: grid column with width. The output is one big
   ((g1·h) × (g2·w)) image.

Every montage variant is a small edit to this template:

- Channels-first data: same idea around the c axis —
  `'(g1 g2) c h w -> c (g1 h) (g2 w)'`.
- A single row of images ("side by side") is the degenerate g1=1 case —
  which collapses to the merge-KP pattern `'b h w c -> h (b w) c'`.
- Column-major filling would be `(g2 g1)` on the left — the packing-order
  question from the split KP, again decided by the task's words
  ("row-major", "grid position (i, j) holds image i·cols + j").

The montage also runs in REVERSE — carving a grid image back into a batch —
by swapping the pattern's sides: `'(g1 h) (g2 w) c -> (g1 g2) h w c'` with
two keywords, since neither factor of each merged axis is inferable alone.

## Worked example

Task: six images into a 3-row × 2-column grid, row-major — verified by
locating specific images.

```python
import torch as t
import einops

# Six 2x2 single-channel images; image k is constant k (easy to locate).
imgs = t.stack([t.full((2, 2, 1), float(k)) for k in range(6)])
assert imgs.shape == (6, 2, 2, 1)

grid = einops.rearrange(imgs, '(g1 g2) h w c -> (g1 h) (g2 w) c', g1=3)
assert grid.shape == (6, 4, 1)               # (3*2, 2*2, 1)

# Row-major placement: grid row 0 holds images 0,1; row 1 -> 2,3; row 2 -> 4,5.
assert grid[0, 0, 0] == 0.0                  # top-left block = image 0
assert grid[0, 2, 0] == 1.0                  # top-right block = image 1
assert grid[2, 0, 0] == 2.0                  # second row starts image 2
assert grid[4, 2, 0] == 5.0                  # bottom-right = image 5

# Reverse: carve the montage back into the batch. Each merged input axis
# hides two unknowns, so each group needs one keyword: g1 (fixes h) AND
# g2 (fixes w).
back = einops.rearrange(grid, '(g1 h) (g2 w) c -> (g1 g2) h w c', g1=3, g2=2)
assert t.equal(back, imgs)
print("6 images", tuple(imgs.shape), "-> montage", tuple(grid.shape))
print(grid[:, :, 0])          # each image is a constant block, so read them off
print("carved back to the batch exactly:", bool(t.equal(back, imgs)))
```

Why each step:

1. Constant-valued test images turn placement checking into value lookups:
   `grid[0, 2]` sitting in grid-row 0, grid-col 1 must equal image 1 under
   row-major packing. Build such fixtures whenever a layout task confuses
   you — arange or constants, never random.
2. The g1=3 keyword does double duty: fixes g2=2 AND documents "3 rows" —
   matching the task's phrasing decides WHICH factor you pass.
3. The reverse pattern needs a keyword PER GROUP (g1 and g2) because each
   merged input axis hides two unknowns and einops solves exactly one
   unknown per parenthesized group.

## Faded practice

### q389
Six images → 3×2 grid, row-major.

```python starter
import torch as t
import einops

def solve(imgs):
    """(6, h, w, c) -> ((3h), (2w), c), images placed row-major."""
    return einops.rearrange(imgs, '_____', g1=3)
```

```python solution
import torch as t
import einops

def solve(imgs):
    """(6, h, w, c) -> ((3h), (2w), c), images placed row-major."""
    return einops.rearrange(imgs, '(g1 g2) h w c -> (g1 h) (g2 w) c', g1=3)
```

## Guided practice

### q364
1. A channels-LAST batch (b, h, w, c), b divisible by b1, tiled into a
   b1-row montage — the template with different names.
2. Which factor is given (b1 rows), which inferred?
3. `'(b1 b2) h w c -> (b1 h) (b2 w) c', b1=b1`.

## Independent practice

From the drill bank: q329 (channels-FIRST montage — where does c sit?),
q382 (3×2 grid again, channels-last — fluency rep), q318 (twelve
channels-first images in a 4×3 grid), q381 (four SEPARATE image arguments
into a 2×2 grid — combine the list-as-axis trick with the montage).

Also from the bank: q322 (carve H into hs strips and W into ws strips,
subgrid indices moved OUT into the batch axis, subgrid-major), q371 (the
inverse — subgrids packed in the batch axis, unpacked back into space),
q531 (the same montage filled COLUMN-major — the misconception below, as a
drill: the edit is on the input side).

## Integrated practice

### q909
grid with a given row count

### q910
grid with a given column count

### q911
grayscale grid

### q912
column-major grid

## Misconceptions

- **"Montages need a loop pasting images into a canvas."** — The
  split-merge pattern is the whole operation. If you're computing paste
  offsets, the pattern replaces that code.
- **"g1 is rows because it's named g1."** — It's rows because it's the SLOW
  factor of the batch split AND merges with h. Rename freely; position does
  the work. (Corollary: to fill column-major, swap the split order, not the
  names.)
- **"The grid pattern only works for images."** — Any batch × per-item-2D
  data montages the same way; and the same split-merge shape reappears in
  patches, space-to-depth, and pooling. Learn it as geometry, not as an
  image trick.
