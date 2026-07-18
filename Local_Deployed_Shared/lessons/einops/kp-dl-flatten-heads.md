---
kc: einops.dl-flatten-heads
title: Deep-learning shapes — flattening and attention heads
supporting: [einops.merge-axes, einops.split-axes, einsum.attention-patterns]
new_syntax: []
faded: [356, 384]
guided: [396]
independent: [394, 349]
---

## Concept

Two shape manoeuvres appear in virtually every model's forward pass, and
both are single einops patterns:

**The classifier flatten.** Convolutional features (b, c, h, w) must become
(b, c·h·w) before a linear layer:

> `'b c h w -> b (c h w)'`

— batch stays, everything else merges (c slow: all of channel 0's pixels,
then channel 1's — the order PyTorch's `.view(b, -1)` produces, so weights
transfer). A single image entering a batch-expecting classifier combines
this with the singleton trick: `'c h w -> 1 (c h w)'`. The full-tensor
collapse `'b c h w -> b'`-style sums are reduce; TOTAL flattening to a
scalar count of axes is `'... -> (...)'`-shaped merges — same grammar
throughout.

**The attention-head split/merge.** Multi-head attention stores per-head
outputs as (b, nh, t, d); tokens want them CONCATENATED as (b, t, nh·d):

> merge: `'b nh t d -> b t (nh d)'`
> — head index slow: head 0's d features first within each token vector.
> split (the inverse, entering attention): `'b t (nh d) -> b nh t d', nh=N`
> — the packed feature axis DECLARES its (heads × per-head) structure.

These two patterns are why einops is beloved in transformer code: the
head-packing convention (nh slow) is load-bearing — get it wrong and
weights trained with one convention silently misread activations from the
other — and the pattern states it where a reshape hides it.

Everything here is the merge/split KPs at model-shaped rank; what this KP
adds is fluency with the CONVENTIONS (batch first, which name is slow) that
the drills — and real checkpoints — assume.

## Worked example

Task: flatten conv features for a classifier; merge attention heads; split
them back.

```python
import numpy as np
import einops

feats = np.arange(24.0).reshape(2, 3, 2, 2)     # (b, c, h, w)

# Classifier flatten: batch survives, (c h w) merge in that order.
flat = einops.rearrange(feats, 'b c h w -> b (c h w)')
assert flat.shape == (2, 12)
# c slow: the first 4 entries of item 0 are channel 0's pixels.
assert flat[0, :4].tolist() == feats[0, 0].ravel().tolist()

# Heads merge: (b, nh, t, d) -> (b, t, (nh d)).
heads = np.arange(16.0).reshape(1, 2, 2, 4)     # (b, nh=2, t, d=4)
merged = einops.rearrange(heads, 'b nh t d -> b t (nh d)')
assert merged.shape == (1, 2, 8)
# Token 0's vector = head 0's features then head 1's (nh slow):
assert merged[0, 0].tolist() == (heads[0, 0, 0].tolist()
                                 + heads[0, 1, 0].tolist())

# The inverse split — declare how the packed axis factors.
unmerged = einops.rearrange(merged, 'b t (nh d) -> b nh t d', nh=2)
assert np.array_equal(unmerged, heads)          # round trip exact
```

Why each step:

1. The `flat[0, :4]` check pins the merge order to something inspectable —
   and matches the framework convention, which is the practical point:
   these patterns interoperate with real model weights.
2. The token-vector concatenation check reads the head-packing convention
   off actual data: head 0 first. When a drill (or paper) says "heads
   concatenated per token", this is the layout it means.
3. The round trip closes the loop: merge and split with matching
   conventions are exact inverses. If your split of someone else's packed
   tensor doesn't round-trip through their merge, your nh-slow assumption
   is wrong — debug the CONVENTION, not the code.

## Faded practice

### q356
Flatten each image of a batch for a linear layer.

```python starter
import numpy as np
import einops

def solve(x):
    """(b, c, h, w) -> (b, c*h*w), standard framework order."""
    return einops.rearrange(x, '_____')
```

```python solution
import numpy as np
import einops

def solve(x):
    """(b, c, h, w) -> (b, c*h*w), standard framework order."""
    return einops.rearrange(x, 'b c h w -> b (c h w)')
```

### q384
Concatenate attention heads back together per token.

```python starter
import numpy as np
import einops

def solve(x_heads):
    """(b, nh, t, d) -> (b, t, nh*d), head 0's features first."""
    return einops.rearrange(x_heads, '_____')
```

```python solution
import numpy as np
import einops

def solve(x_heads):
    """(b, nh, t, d) -> (b, t, nh*d), head 0's features first."""
    return einops.rearrange(x_heads, 'b nh t d -> b t (nh d)')
```

## Guided practice

### q396
1. The packed sequence (b, t, nh·d), head index slowest, must SPLIT back
   into heads — inverse of the merge you just did.
2. The parens move to the input side; one keyword names the head count.
3. `'b t (nh d) -> b nh t d', nh=nh` — round-trip it against a merge to be
   sure.

## Independent practice

From the drill bank: q394 (single image → flattened prediction shape
(1, c·h·w) — flatten + singleton batch in one pattern), q349 (sum EVERY
element of a 4-D tensor via reduce's empty output — '(b c h w -> )'-shaped;
which function, which aggregation?).

## Misconceptions

- **"Any flatten order works if shapes match."** — The linear layer's
  weights were trained against ONE packing order. (c h w) vs (h w c)
  flattens produce equal shapes and incompatible semantics. Follow the
  framework convention the drill names.
- **"Head merge/split order is a style choice."** — nh-slow is the
  transformer ecosystem's convention; violating it silently permutes
  features. The einops pattern documents the choice — that's half its
  value.
- **"These need reshape + transpose + reshape chains."** — Each is ONE
  pattern. If your solution has intermediate .transpose calls around
  reshapes, the pattern can absorb them (and check the shapes while it's
  at it).
