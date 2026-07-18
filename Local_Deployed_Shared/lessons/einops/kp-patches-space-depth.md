---
kc: einops.patches-space-depth
title: Patches, space-to-depth, depth-to-space
supporting: [einops.split-axes, einops.grids-montage]
new_syntax: []
faded: [323]
guided: [313]
independent: [401, 404, 343, 350]
---

## Concept

Vision architectures constantly trade SPACE for DEPTH (or batch): cut each
image into little patches and treat them as tokens (ViT), fold pixel blocks
into channels (space-to-depth), or unfold them back (depth-to-space /
pixel-shuffle). All are the grid pattern of the last KP with the roles
recast:

**Patch extraction** — split each spatial axis into (blocks × within-block),
then pull the block coordinates out as a patch index:

> `'(h p1) (w p2) c -> (h w) p1 p2 c', p1=P, p2=P`

Height splits into h blocks of p1 rows; width likewise; merging (h w)
row-major gives the patch list. **Reassembly is the same pattern reversed**
— it was your grid-KP faded exercise (q323), now recognized as
depth-to-space's cousin.

**Space-to-depth** — the block coordinates fold into the CHANNEL axis
instead of a patch index:

> `'b c (h p) (w q) -> b (c p q) h w', p=P, q=P`

Each p×q pixel block's values become extra channels; spatial dims shrink by
P, channels grow by P². (The output channel packing order — c slow, then p,
then q — is dictated by the paren order; tasks specify theirs.)

**Depth-to-space** mirrors it: `'b (c p q) h w -> b c (h p) (w q)'` — the
channel axis DECLARES its factorization, and blocks unfold back into space.
This is pixel-shuffle upsampling.

The discipline for all of them: write the INPUT side to describe how the
data is actually packed (which factor is slow), the OUTPUT side to describe
what the task wants — then hand einops the block sizes. When the packing
order is ambiguous in your head, build a tiny arange example and round-trip.

## Worked example

Task: extract 2×2 patches from an image, reassemble them, and run a
space-to-depth.

```python
import numpy as np
import einops

img = np.arange(16.0).reshape(4, 4, 1)     # (H, W, c=1), values = positions

# PATCHES: split H into 2 blocks of 2, W likewise; block coords -> patch axis.
patches = einops.rearrange(img, '(h p1) (w p2) c -> (h w) p1 p2 c', p1=2, p2=2)
assert patches.shape == (4, 2, 2, 1)
# Patch 0 = top-left 2x2 block, row-major patch order:
assert patches[0, :, :, 0].tolist() == [[0.0, 1.0], [4.0, 5.0]]
assert patches[1, 0, 0, 0] == 2.0          # patch 1 starts at column 2

# REASSEMBLE: the same pattern, sides swapped (this is q323's shape).
back = einops.rearrange(patches, '(h w) p1 p2 c -> (h p1) (w p2) c', h=2)
assert np.array_equal(back, img)

# SPACE-TO-DEPTH on a batch: blocks fold into channels; H, W halve.
x = np.arange(32.0).reshape(1, 2, 4, 4)    # (b, c=2, H, W)
s2d = einops.rearrange(x, 'b c (h p) (w q) -> b (c p q) h w', p=2, q=2)
assert s2d.shape == (1, 8, 2, 2)           # channels x4, spatial /2
# The new channel block for output pixel (0,0) holds input block [0:2, 0:2]:
assert s2d[0, :4, 0, 0].tolist() == [0.0, 1.0, 4.0, 5.0]
```

Why each step:

1. Position-valued pixels make each check readable: patch 1 starting at
   value 2.0 confirms row-major patch order and a correct width split.
   This fixture technique is how to debug ANY packing dispute with einops.
2. The reassembly line being the extraction line reversed (with the
   keyword moving to the other side's unknowns) cements the symmetry —
   patches/grids/s2d are one bijection family, direction chosen by which
   side carries the parens you're UNPACKING.
3. In space-to-depth, verify the channel packing: c slow, p, then q — the
   four values 0,1,4,5 are block (0,0) in row-major order, sitting after
   channel 0's... here c=... the first 4 output channels come from input
   channel 0. Reading packed channel layouts element-by-element once
   inoculates against the classic s2d ordering bug.

## Faded practice

### q323
Reassemble a row-major tile stack into the image.

```python starter
import numpy as np
import einops

def solve(patches, h, w):
    """(h*w, p1, p2, c) tiles, row-major -> ((h p1), (w p2), c) image."""
    return einops.rearrange(patches, '_____', h=h, w=w)
```

```python solution
import numpy as np
import einops

def solve(patches, h, w):
    """(h*w, p1, p2, c) tiles, row-major -> ((h p1), (w p2), c) image."""
    return einops.rearrange(patches, '(h w) p1 p2 c -> (h p1) (w p2) c', h=h, w=w)
```

## Guided practice

### q313
1. Space-to-depth on (b, c, h, w) with block size p: spatial dims shrink by
   p, channels multiply by p².
2. Split each spatial axis into (blocks × p); fold the two p-factors into
   the channel group — the task states the required channel packing order.
3. `'b c (h p1) (w p2) -> b (c p1 p2) h w', p1=p, p2=p` — or the order the
   prompt demands; verify one block.

## Independent practice

From the drill bank: q401 (all non-overlapping 3×3 patches of an (H, W, C)
image), q404 (extract patches from a channels-FIRST image), q343
(depth-to-space: channel groups ordered c-slowest unfold into space),
q350 (transpose WITHIN each 2×2 patch — split both axes, swap the two
within-patch names, merge back).

## Misconceptions

- **"Patch extraction needs sliding-window machinery."** — NON-overlapping
  patches are a pure reshape (split + merge); no windows, no copies of
  copies. Sliding (overlapping) windows are the numpy KP's tool — different
  task, check the word "non-overlapping".
- **"Space-to-depth loses spatial information."** — It's a bijection: every
  pixel gets a unique (channel, position) address, and depth-to-space
  inverts it exactly. What changes is which axis "sees" the detail.
- **"The channel packing order after s2d doesn't matter."** — Downstream
  code (or the grader) reads channels by index; (c p q) vs (p q c) are
  different tensors with equal shapes. The paren order IS the file format —
  get it from the task, verify with a position-valued example.
