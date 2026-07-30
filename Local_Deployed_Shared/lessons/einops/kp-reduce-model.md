---
kc: einops.reduce-model
title: einops.reduce — dropping axes with an aggregation
supporting: [einops.pattern-language, numpy.axis-reductions]
new_syntax: []
faded: [325]
guided: [328]
independent: [326, 367, 399, 402, 332, 340, 370]
---

## Concept

Where rearrange must keep every axis, **`einops.reduce` is allowed to drop
them — and you say HOW the dropped values collapse**:

> `einops.reduce(x, 'b c h w -> b', 'mean')`
> — c, h, w vanish from the pattern, so each output element is the mean
> over everything that vanished. The third argument names the aggregation:
> `'mean'`, `'max'`, `'min'`, `'sum'`, `'prod'`.

This is PyTorch's `dim=` reductions with the einsum-style deletion rule, in
einops clothing — three notations, one concept:

- `x.mean(dim=(1, 2, 3))` — axes by number.
- `t.einsum('bchw->b', x) / (c*h*w)` — sum by deletion, mean by hand.
- `reduce(x, 'b c h w -> b', 'mean')` — deletion by name, aggregation
  declared. (Note: unlike einsum, reduce does means/maxes natively — no
  divide-outside dance.)

Details that matter in the drills:

- **Partial drops**: `'b h w c -> b h w'` maxes only over channels;
  `'b c h w -> b c'` averages each channel map. Any subset of axes can go.
- **Keep a singleton**: writing `1` (or `()` — same thing) in the output
  where the dropped axis was — `'h w c -> 1 w c'` — keeps the result
  broadcast-ready against the input, einops' keepdim. This is exactly the
  `keepdim=True` story: reduce-then-broadcast pipelines (subtract each
  column's max…) want the singleton kept.
- **Reduce + rearrange compose**: the pattern can still permute the
  survivors while reducing (`'b c h w -> c b'` is legal), and — the next
  KP — parenthesized factors turn reduce into pooling.

## Worked example

Task: per-image scalar means; per-column max keeping a singleton row; and a
grayscale via mean-over-channels with the channel kept.

```python
import torch as t
import einops

x = t.arange(24.0).reshape(2, 3, 2, 2)      # (b, c, h, w)

# All of c, h, w collapse under 'mean' -> one number per image.
per_image = einops.reduce(x, 'b c h w -> b', 'mean')
assert per_image.tolist() == [5.5, 17.5]
assert t.allclose(per_image, x.mean(dim=(1, 2, 3)))   # the numpy twin

# Partial drop: mean over spatial only -> per-channel statistics.
per_channel = einops.reduce(x, 'b c h w -> b c', 'mean')
assert per_channel.shape == (2, 3)
assert per_channel[0, 0] == x[0, 0].mean()

# Keep-a-singleton: per-column max of an image, row axis kept as 1.
img = t.tensor([[1.0, 5.0],
                [7.0, 2.0]])
colmax = einops.reduce(img, 'h w -> 1 w', 'max')
assert colmax.shape == (1, 2)
assert colmax.tolist() == [[7.0, 5.0]]
# Why keep it: the (1, 2) result broadcasts straight back against (2, 2).
assert (img - colmax).shape == (2, 2)

# Grayscale keeping a trailing singleton channel: (b,h,w,c) -> (b,h,w,1).
imgs = t.ones((2, 2, 2, 3))
gray = einops.reduce(imgs, 'b h w c -> b h w 1', 'mean')
assert gray.shape == (2, 2, 2, 1)
```

Why each step:

1. The `dim=` twin assert carries your existing axis intuition into the new
   notation: names deleted ↔ axis numbers listed. After a few reps the
   named form usually reads faster — especially at rank 4+.
2. In the singleton example, the follow-up subtraction is the POINT: the
   kept `1` is what makes reduce-then-operate pipelines shape-safe, same as
   keepdims in np-3.
3. The grayscale line matches a drill's exact contract ('-> b h w 1');
   note reduce handles mean natively — resist the einsum habit of dividing
   afterwards.

## Faded practice

### q325
Each image of a channels-first batch → one scalar mean.

```python starter
import torch as t
import einops

def solve(arr):
    """(b, c, h, w) -> (b,): mean over channels and pixels."""
    return einops.reduce(arr, '_____', 'mean')
```

```python solution
import torch as t
import einops

def solve(arr):
    """(b, c, h, w) -> (b,): mean over channels and pixels."""
    return einops.reduce(arr, 'b c h w -> b', 'mean')
```

## Guided practice

### q328
1. (h, w, c) image, average over the HEIGHT axis only → (w, c).
2. One name disappears from the pattern; the aggregation string says how.
3. `einops.reduce(img, 'h w c -> w c', 'mean')`.

## Independent practice

From the drill bank: q326 (max over channels of a channels-last batch),
q367 (per-channel spatial means → (b, c)), q399 (grayscale with trailing
singleton — the '-> b h w 1' form), q402 (per-column max keeping the height
axis as a singleton — '() w c' or '1 w c'), q332 (row maxima with BOTH other
axes as singletons — read the required output shape carefully).

Also from the bank: q340 (centre each (batch, channel) map by its OWN mean
— reduce to 'b c 1 1' so it broadcasts back), q370 (centre each channel
across the WHOLE batch — '1 c 1 1', BatchNorm-style; contrast with q340).

## Misconceptions

- **"reduce is rearrange with a mode argument."** — rearrange forbids
  dropping names; reduce requires it (something must reduce). They share
  the pattern grammar, not the contract.
- **"Means still need dividing outside, like einsum."** — reduce's third
  argument does real means/maxes/mins natively. The divide-outside habit
  is einsum-specific.
- **"'h w -> w' and 'h w -> 1 w' are the same reduction."** — Same numbers,
  different SHAPE: (w,) vs (1, w). The singleton version survives
  broadcasting against the original; the bare version may misalign
  (np-3's keepdims lesson, verbatim). Drills specify which they grade.
