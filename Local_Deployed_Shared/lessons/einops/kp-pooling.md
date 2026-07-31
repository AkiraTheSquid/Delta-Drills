---
kc: einops.pooling
title: Pooling with factored axes
supporting: [einops.reduce-model, einops.split-axes]
new_syntax: []
faded: [324]
guided: [363]
independent: [386, 368, 354, 336, 377]
---

## Concept

Combine reduce's aggregation with split's parentheses and you get POOLING —
window-wise downsampling — as pure notation:

> `einops.reduce(x, 'b c (h h2) (w w2) -> b c h w', 'mean', h2=2, w2=2)`

Read the input side as a split: height factors into (h blocks × h2 rows),
width into (w × w2). The output keeps the block coordinates (h, w) and
DROPS the within-window names (h2, w2) — so each output pixel aggregates
its h2×w2 window. That's non-overlapping average pooling; `'max'` makes it
max pooling. The mental model:

> **split the axis into (keep × window), reduce away the window.**

Variants the drills exercise:

- **Any window size / rectangle**: `h2=3, w2=3` for 3×3; the factors need
  not match.
- **Any rank**: a 5-D volume pools with three factored axes —
  `'b c (x a) (y b2) (z c2) -> b c x y z'` — nothing new, one more group.
- **Pool one axis only**: halve the width by averaging adjacent column
  pairs: `'b h (w w2) c -> b h w c', w2=2` — "adjacent pairs" is a
  length-2 window on that axis alone. Temporal downsampling
  ('b c (t two) -> b c t') is the same idea on sequences.
- **Pooling + flatten, etc.**: since it's all one pattern language, pooling
  composes freely with merges in the same call.

One requirement: non-overlapping windows must TILE the axis — sizes must
divide exactly (drills guarantee it; real code pads first). Overlapping /
strided pooling is outside reduce's power — that's `x.unfold(...)`
and the pooling layers' territory.

## Worked example

Task: 2×2 average pooling on a batch; then halving width by averaging
adjacent column pairs.

```python
import torch as t
import einops

x = t.arange(16.0).reshape(1, 1, 4, 4)      # (b, c, H, W)

# Split H into (2 blocks x 2 rows), W likewise; reduce the window names.
pooled = einops.reduce(x, 'b c (h h2) (w w2) -> b c h w', 'mean', h2=2, w2=2)
assert pooled.shape == (1, 1, 2, 2)
# Window (0,0) = mean of [[0,1],[4,5]] = 2.5:
assert pooled[0, 0].tolist() == [[2.5, 4.5],
                                 [10.5, 12.5]]

# Max pooling is the same pattern, different aggregation.
mx = einops.reduce(x, 'b c (h h2) (w w2) -> b c h w', 'max', h2=2, w2=2)
assert mx[0, 0].tolist() == [[5.0, 7.0],
                             [13.0, 15.0]]

# One-axis pooling: average adjacent COLUMN pairs, everything else intact.
imgs = t.arange(8.0).reshape(1, 2, 4, 1)    # (b, h, w=4, c)
halved = einops.reduce(imgs, 'b h (w w2) c -> b h w c', 'mean', w2=2)
assert halved.shape == (1, 2, 2, 1)
assert halved[0, 0, :, 0].tolist() == [0.5, 2.5]   # (0+1)/2, (2+3)/2
print(x[0, 0], "\n")
print("mean-pooled 2x2 ->\n", pooled[0, 0])
print("max-pooled  2x2 ->\n", mx[0, 0])
print("column pairs averaged:", imgs[0, 0, :, 0].tolist(), "->",
      halved[0, 0, :, 0].tolist())
```

Why each step:

1. Hand-verify ONE window ([[0,1],[4,5]] → 2.5) and trust the pattern for
   the rest — the same one-element discipline as every layout KP, now with
   an aggregation attached.
2. mean→max changing only the string argument shows where the operation
   lives: geometry in the pattern, semantics in the aggregation. Swap
   either independently.
3. In the one-axis case, note which name went INSIDE the parens: the axis
   being pooled. Everything outside parens rides along — that's how the
   pattern scales to 5-D volumes without new ideas.

## Faded practice

### q324
2×2 non-overlapping average pooling on (B, C, H, W).

```python starter
import torch as t
import einops

def solve(x):
    """Halve H and W by averaging each 2x2 window."""
    return einops.reduce(x, '_____', 'mean', h2=2, w2=2)
```

```python solution
import torch as t
import einops

def solve(x):
    """Halve H and W by averaging each 2x2 window."""
    return einops.reduce(x, 'b c (h h2) (w w2) -> b c h w', 'mean', h2=2, w2=2)
```

## Guided practice

### q363
1. 3×3 average pooling on a channels-first single image (c, h, w) — same
   split-and-reduce, no batch axis.
2. Window factors are 3 this time; both spatial axes factor.
3. `'c (h h3) (w w3) -> c h w', h3=3, w3=3` with 'mean'.

## Independent practice

From the drill bank: q386 (halve width by averaging adjacent column pairs —
channels-last), q368 (2×2×2 max-pool of a 5-D volume — three factored
axes), q354 (subtract each (batch, channel) pair's spatial MINIMUM from its
own map — a reduce with kept singletons feeding a broadcast subtraction,
not a pooling; recognize the difference).

Also from the bank: q336 (max-pool AND tile the batch into a grid in ONE
reduce), q377 (max-pool then flatten everything but the batch, also in one
reduce).

## Misconceptions

- **"Pooling needs a framework (torch.nn.AvgPool2d) or a loop."** — Non-
  overlapping pooling is reshape+reduce, which is exactly what the factored
  pattern states. Frameworks add padding/stride options; the core is this.
- **"The window names (h2, w2) must be called that."** — Any names; what
  matters is they appear inside input parens and NOT in the output. The
  kept block-count names are the ones that survive.
- **"reduce can do stride-1 (overlapping) pooling too."** — No: factored
  axes tile the input disjointly. Overlap = sliding_window_view + reduction
  on `x.unfold(...)`. "Non-overlapping" in a task is your green light for
  the einops form.
