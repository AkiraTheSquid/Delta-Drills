---
kc: einops.repeat-model
title: einops.repeat — new axes and stretched axes
supporting: [einops.split-axes, numpy.tile-repeat-meshgrid, numpy.broadcasting-rules]
new_syntax: []
faded: [317]
guided: [351]
independent: [338, 348, 385, 341, 339, 383, 352, 355]
---

## Concept

The third einops function, **`einops.repeat`, is the mirror image of
reduce: the output may contain names (or factors) the input DOESN'T have**
— and the data is copied to fill them. Two distinct moves live under it:

**1. A brand-new axis** — a name on the right that's absent on the left:

> `repeat(cls, 'b d -> b t d', t=8)`
> — each (b, d) embedding is broadcast across 8 new time steps. This is
> einops' answer to `broadcast_to` / `unsqueeze` + `expand`: the classic use is
> spreading a class token or per-token weights across a new dimension.

**2. Stretching an existing axis** — a factor inside output parens that
wasn't in the input:

> `repeat(img, 'c h w -> c (h 3) w')` — each ROW appears 3× consecutively
> (h slow, 3 fast: row, its copies, next row). Nearest-neighbor upscaling
> is this on both axes: `'h w c -> (h 2) (w 2) c'`.
> `repeat(img, 'c h w -> c (3 h) w')` — the WHOLE image stacked 3 times
> (3 slow: full copy, then full copy). PyTorch's
> `repeat_interleave`-vs-`repeat` distinction, decided by paren ORDER
> instead of function choice.

The order rule is the one you already own from merging: **left = slow =
blocks, right = fast = within-block.** `(h k)` interleaves copies per row;
`(k h)` concatenates whole copies. Literal numbers can sit in the parens
directly, or be keywords (`(h k), k=3`).

New-axis literals work too: `'h w -> h w 3'` copies a grayscale image into
3 identical channels.

## Worked example

Task: broadcast an embedding across time; stretch rows; whole-image stack —
the three moves, distinguished.

```python
import torch as t
import einops

# 1. NEW AXIS: (b, d) -> (b, t, d), each embedding copied t times.
cls = t.tensor([[1.0, 2.0],
                [3.0, 4.0]])                # (b=2, d=2)
seq = einops.repeat(cls, 'b d -> b t d', t=3)
assert seq.shape == (2, 3, 2)
assert seq[0].tolist() == [[1.0, 2.0]] * 3   # identical copies down t

# 2. STRETCH, factor fast: each ROW repeats consecutively.
img = t.tensor([[1, 2],
                [3, 4]])                     # (h, w) for clarity
rows3 = einops.repeat(img, 'h w -> (h r) w', r=2)
assert rows3.tolist() == [[1, 2],
                          [1, 2],
                          [3, 4],
                          [3, 4]]            # row, its copy, next row

# 3. STRETCH, factor slow: the WHOLE block repeats.
whole = einops.repeat(img, 'h w -> (r h) w', r=2)
assert whole.tolist() == [[1, 2],
                          [3, 4],
                          [1, 2],
                          [3, 4]]            # full image, then again

# Nearest-neighbor 2x upscale: both axes, factor fast on each.
up = einops.repeat(img, 'h w -> (h a) (w b)', a=2, b=2)
assert up.tolist() == [[1, 1, 2, 2],
                       [1, 1, 2, 2],
                       [3, 3, 4, 4],
                       [3, 3, 4, 4]]
```

Why each step:

1. `(h r)` vs `(r h)` on the same data is the `repeat_interleave`-vs-`repeat` shootout
   resettled by one convention (left slow) instead of two function names —
   run both once, then trust the rule.
2. The new-axis case allocates real copies (unlike `expand`'s
   virtual stretch) — fine for drills; in memory-tight code you'd reach
   for broadcast_to semantics knowingly.
3. The upscale's per-axis factors (a, b) show the moves composing — each
   axis independently gets the "pixel becomes a block" treatment, and the
   2×2 blocks in the output are the proof.

## Faded practice

### q317
Triple an image's height by repeating each row consecutively.

```python starter
import torch as t
import einops

def solve(img):
    """(c, h, w) -> (c, 3h, w): each row appears 3x in a row."""
    return einops.repeat(img, 'c h w -> c (_____) w')
```

```python solution
import torch as t
import einops

def solve(img):
    """(c, h, w) -> (c, 3h, w): each row appears 3x in a row."""
    return einops.repeat(img, 'c h w -> c (h 3) w')
```

## Guided practice

### q351
1. Double both spatial dims with each pixel becoming a 2×2 block —
   stretch, and which order makes copies stay WITH their pixel?
2. Factor fast: `(h 2) (w 2)`.
3. `'h w c -> (h 2) (w 2) c'` — check one corner block.

## Independent practice

From the drill bank: q338 (broadcast class embeddings across time), q348
(rows repeated 3× consecutively, batched),
q385 (columns repeated 4× IN SEQUENCE — which order?), q341 (whole image
stacked k times vertically — the other order), q339 (per-token scalar
weights (b, t) expanded to (b, t, d)), q383 (duplicate the whole channel
BLOCK r times — slow or fast?).

Also from the bank: q352 (stretch vertically by duplicating each ROW in
place — '(h k)', new index FASTEST), q355 (slice, stack two images
vertically, then repeat the strip horizontally).

## Misconceptions

- **"repeat is rearrange with bigger numbers."** — rearrange is a
  bijection (every input element appears once); repeat COPIES. The
  found-on-the-right-only names/factors are what license the copying.
- **"'(h 2)' and '(2 h)' both just double the axis."** — Same length,
  different layout: h-slow keeps copies adjacent to their source row
  (stretch); 2-slow lays whole blocks end to end (tile). Tasks say
  "consecutively"/"each row" vs "whole image again" — map the words to
  the order.
- **"Broadcasting makes repeat unnecessary."** — Broadcasting stretches
  virtually within an operation; repeat materializes an actual tensor with
  the new shape — which is what a drill's return-shape contract (and much
  downstream code) requires. Know which one you're producing.
