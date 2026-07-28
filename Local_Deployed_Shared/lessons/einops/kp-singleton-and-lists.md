---
kc: einops.singleton-and-lists
title: Singleton axes and lists as an axis
supporting: [einops.pattern-language]
new_syntax: []
faded: [361]
guided: [360]
independent: [358, 374, 376]
---

## Concept

Two small pattern features that finish the rearrange grammar:

**`1` — a literal singleton axis.** Writing `1` in a pattern inserts (on
the right) or consumes (on the left) a length-1 axis:

- `'h w -> 1 h w'` — add a leading channel/batch axis (the einops spelling
  of `x[None]`).
- `'b h w c -> b 1 h w c'` — insert one mid-tensor.
- `'1 h w -> h w'` — squeeze a known singleton, with verification: if that
  axis isn't length 1, einops errors instead of silently squeezing the
  wrong thing.

Keeping a REDUCED axis as a singleton (`'h w c -> 1 w c'` in reduce) also
uses this — the reduce KP picks that up.

**A Python list as the first axis.** Handing einops a LIST of same-shape
tensors makes the list index axis 0 — pattern it like any other axis:

- `einops.rearrange([img_a, img_b], 'b h w c -> h (b w) c')` — two images
  side by side, no explicit t.stack first.
- `'b h w c -> b h w c'` on a list is exactly t.stack: the identity
  pattern, with the list→tensor conversion as the entire point.

Together these subsume t.stack / t.unsqueeze / t.squeeze with the
same pattern language you're already using — one notation for the whole
shape-plumbing toolbox.

## Worked example

Task: stack a list of images into a batch; add a singleton channel axis;
combine both in one pattern.

```python
import torch as t
import einops

imgs = [t.ones((2, 3, 1)) * i for i in range(4)]   # list of (h, w, c)

# List -> batch axis: the identity pattern DOES the stacking.
batch = einops.rearrange(imgs, 'b h w c -> b h w c')
assert batch.shape == (4, 2, 3, 1)
assert batch[2, 0, 0, 0] == 2.0                      # list order preserved

# Singleton insertion: a plain 2-D tensor gains a leading axis.
x2d = t.arange(6).reshape(2, 3)
x3d = einops.rearrange(x2d, 'h w -> 1 h w')
assert x3d.shape == (1, 2, 3)

# Both at once: stack a list AND lay the images out side by side.
pair = [t.zeros((2, 2, 1)), t.ones((2, 2, 1))]
wide = einops.rearrange(pair, 'b h w c -> h (b w) c')
assert wide.shape == (2, 4, 1)
assert wide[0, :, 0].tolist() == [0.0, 0.0, 1.0, 1.0]  # a then b, left to right

# Squeeze with verification: consuming a '1' that isn't there fails loudly.
try:
    einops.rearrange(x2d, '1 h w -> h w')            # x2d is 2-D — no 1 axis
    raised = False
except Exception:
    raised = True
assert raised
```

Why each step:

1. The identity pattern on a list looks like a no-op and isn't — the
   conversion is the operation. Reading einops code, remember the input
   TYPE is part of the semantics.
2. The combined example is the idiom to keep: list-stack + merge in one
   declarative step replaces stack-then-rearrange chains. Note b defaults
   to slow in `(b w)`: first list element leftmost.
3. The verified squeeze failing on 2-D data is einops' shape checking
   again — `'1 h w -> h w'` documents an EXPECTATION about the input, and
   the library enforces it. `t.squeeze` would have silently done something.

## Faded practice

### q361
A Python list of (h, w, c) images → one (b, h, w, c) batch.

```python starter
import torch as t
import einops

def solve(imgs):
    """Stack the list: the list index becomes axis b."""
    return einops.rearrange(imgs, '_____')
```

```python solution
import torch as t
import einops

def solve(imgs):
    """Stack the list: the list index becomes axis b."""
    return einops.rearrange(imgs, 'b h w c -> b h w c')
```

## Guided practice

### q360
1. (h, w) → (1, h, w): nothing moves; one axis appears.
2. The literal `1` on the output side inserts it.
3. `'h w -> 1 h w'`.

## Independent practice

From the drill bank: q358 (insert a singleton just after the batch axis),
q374 (list of images concatenated side by side — list axis + merge),
q376 (exactly two images side by side — a two-element list works).

## Misconceptions

- **"I must t.stack a list before einops can touch it."** — A list of
  same-shape tensors is accepted directly; its index becomes the first
  axis. The stack is the pattern's job.
- **"`1` in a pattern is a size I'm asserting for a normal axis."** — It's
  a LITERAL singleton: inserted on the right, consumed (with verification)
  on the left. Naming it (like 'c') instead would bind a real axis.
- **"Squeezing with einops is overkill — t.squeeze is fine."** —
  t.squeeze removes ALL singletons, including ones you didn't expect
  (a batch that happens to be size 1!). `'1 h w -> h w'` removes exactly
  the declared one and errors otherwise — overkill is the feature.
