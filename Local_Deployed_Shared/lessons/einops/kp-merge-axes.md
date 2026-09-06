---
kc: einops.merge-axes
title: Merging axes with (parentheses)
supporting: [einops.pattern-language, numpy.reshape-flatten]
new_syntax: [einops-axis-composition]
faded: [391, 347]
guided: [357]
independent: [342, 314, 346, 353, 373, 380, 392, 400]
integrated: [889, 890, 891, 892, 905, 906, 907, 908]
---

## Concept

Parentheses on the **output** side of a pattern MERGE axes into one:

> `'c h w -> c (h w)'`
> — h and w fuse into a single axis of length h·w.

Two things define what you get:

1. **Order inside the parens = nesting order.** The LEFT name varies
   slowest, the rightmost fastest (exactly row-major reshape). `(h w)`
   walks: h=0 with all w's, then h=1 with all w's… — each row in full, row
   by row. `(w h)` would walk columns instead. When a task says
   "row-major", "reading order", or "all of X's block before the next X",
   it is dictating the paren order.
2. **Which axes you merge — and they need not be adjacent in the input.**
   `'b h w c -> h (b w) c'` merges batch INTO width: because b is the slow
   (left) name, image 0's columns come first, then image 1's — the images
   laid side by side. Merging non-adjacent axes quietly includes the
   transpose that brings them together; the pattern spells the whole move.

The classic instances:

- Flatten spatial: `'c h w -> c (h w)'` — per-channel row-major flattening.
- Stack a batch vertically: `'b h w c -> (b h) w c'` — batch into height,
  image after image.
- Side-by-side concatenation: `'b h w c -> h (b w) c'`.
- Channel unroll: `'c h w -> (c h) w'` — all of channel 0's rows, then
  channel 1's… (c slow, h fast).

In raw PyTorch each of these is a permute+reshape pair you must derive;
in einops the pattern is the derivation.

## Worked example

Task: flatten an image's spatial axes per channel; lay a batch out side by
side.

```python
import torch as t
import einops

img = t.arange(12).reshape(3, 2, 2)      # (c, h, w)

# Merge h and w, h varying slowest: each channel flattens in reading order.
flat = einops.rearrange(img, 'c h w -> c (h w)')
assert flat.shape == (3, 4)
assert flat[0].tolist() == [0, 1, 2, 3]   # row 0 then row 1 of channel 0

# Paren order matters: (w h) reads DOWN the columns instead.
flat_cols = einops.rearrange(img, 'c h w -> c (w h)')
assert flat_cols[0].tolist() == [0, 2, 1, 3]

# Merge NON-adjacent axes: batch into width -> images side by side.
batch = t.arange(16).reshape(2, 2, 2, 2)  # (b, h, w, c)
wide = einops.rearrange(batch, 'b h w c -> h (b w) c')
assert wide.shape == (2, 4, 2)
# Row 0: image 0's two columns, THEN image 1's two columns (b is slow).
assert wide[0, :, 0].tolist() == [0, 2, 8, 10]
print("'(h w)' reads across rows:", flat[0])
print("'(w h)' reads down columns:", flat_cols[0])
print("batch merged into width", tuple(wide.shape), "-> row 0:",
      wide[0, :, 0])
```

Why each step:

1. The `(h w)` vs `(w h)` pair on the same data is the fastest way to burn
   in "left = slow": identical merge, different walk, different output.
   When unsure, test both on arange data — the values ARE their original
   positions.
2. For the side-by-side merge, predict before running: b slow means all of
   image 0's width before image 1's — that's "side by side, image 0 on the
   left". If you wanted interleaved columns, b would go FAST: `(w b)`.
3. Notice `wide`'s shape (2, 4, 2) contains the arithmetic (b·w = 4) — a
   merged axis's length is always the product, a free sanity check.

## Faded practice

### q391
Flatten spatial axes per channel, reading order.

```python starter
import torch as t
import einops

def solve(img):
    """(c, h, w) -> (c, h*w), rows before columns."""
    return einops.rearrange(img, '_____')
```

```python solution
import torch as t
import einops

def solve(img):
    """(c, h, w) -> (c, h*w), rows before columns."""
    return einops.rearrange(img, 'c h w -> c (h w)')
```

### q347
Same input, a different merge: lay the channels out HORIZONTALLY, so channel
0's whole image sits left of channel 1's. Shape (c, h, w) -> (h, c·w). Two
decisions the flatten above did not ask for — WHICH pair of axes merges (they
are not adjacent in the input), and which of them is the slow one.

```python starter
import torch as t
import einops

def solve(img):
    """(c, h, w) -> (h, c*w): channel 0's image, then channel 1's, side by side."""
    return einops.rearrange(img, '_____')
```

```python solution
import torch as t
import einops

def solve(img):
    """(c, h, w) -> (h, c*w): channel 0's image, then channel 1's, side by side."""
    return einops.rearrange(img, 'c h w -> h (c w)')
```

## Guided practice

### q357
1. (C, H, W) → ((C·H), W): channels and height merge into one tall strip,
   channel 0's rows first.
2. "Channel 0's rows first, then channel 1's" tells you which name is slow
   inside the parens.
3. `'c h w -> (c h) w'`.

## Independent practice

From the drill bank: q342 (batch merged into height), q314 (batch side by
side in a channels-LAST layout).

Also from the bank: q346 (side by side WITHIN channels-first: 'b c h w ->
c h (b w)'), q353 (merge the chunk axes of (b, n, p, d) back into one
sequence), q373 (unroll the CHANNEL axis along height, plus a trailing
singleton — '(c h) w ()'), q380 (batch into height AND channels-last in
the same pattern), q392 (stack vertically in channels-first — 'c (b h)
w'), q400 (batch merged into width with the batch index INNERMOST, so
columns interleave instead of images concatenating).

## Integrated practice

### q889
batch side by side

### q890
vertical stack, channels-last

### q891
channels stacked vertically

### q892
batch and space into one axis

### q905
channels interleaved by row

### q906
channels interleaved by column

### q907
channels side by side, per image

### q908
transposed channels side by side

## Misconceptions

- **"`(h w)` and `(w h)` give the same flattening."** — Same length,
  different order: left-slow/right-fast. The task's phrase "row-major" /
  "column by column" / "X's block first" picks the order for you.
- **"Axes must be adjacent to merge."** — The pattern happily merges
  distant axes ('b h w c -> h (b w) c'); einops inserts the implied
  transpose. By hand you'd have to permute them together first — that
  two-step is exactly what the pattern hides.
- **"Merging loses information."** — It's a pure relabeling; every element
  keeps a unique address. The inverse operation (splitting, next KP)
  recovers the original — provided you remember one of the factor sizes.
