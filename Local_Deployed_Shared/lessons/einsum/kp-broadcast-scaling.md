---
kc: einsum.broadcast-scaling
title: Weighted sums and per-axis scaling
supporting: [einsum.batch-dims, einsum.reductions]
new_syntax: []
faded: [269]
guided: [291]
independent: [251, 304, 309, 255]
---

## Concept

A hugely common tensor operation: **one small vector of weights applied
along one axis of a big tensor** — per-channel scales, per-timestep weights,
head-importance vectors. In einsum, the vector simply declares WHICH axis it
rides on, by using that axis's letter; the output side then decides between
the operation's two flavors:

- **Weighted SUM (collapse the axis):**
  `'chw,c->hw'` — the image's channel axis pairs with the weight vector,
  and c is dropped: each output pixel is Σ_c w[c]·img[c,h,w]. A grayscale
  conversion is exactly this. Likewise `'btd,t->bd'` (weights per timestep,
  summed over time), `'btd,d->bt'` (project features onto a vector).
- **Weighted SCALE (keep the axis):**
  `'chw,c->chw'` — same pairing, but c survives: channel k is multiplied
  by s[k], nothing summed. The broadcasting equivalent is
  `img * s[:, None, None]` — einsum saves you counting the Nones, because
  the LETTER finds the axis by name.

The recipe for any such task: (1) letter the big tensor meaningfully,
(2) give the vector the letter of the axis it describes, (3) keep or drop
that letter per whether the task says "scale/weight each…" (keep) or
"weighted sum/average over…" (drop). A weighted AVERAGE divides by
`w.sum()` outside the spec (einsum never divides — same as means).

These little specs are constant companions in model code — attention-head
weighting `'bthd,h->btd'` is exactly the same shape of thought as grayscale
conversion.

## Worked example

Task: weighted sum over channels (grayscale-style), and per-channel scaling
— same operands, one letter's fate apart.

```python
import numpy as np

img = np.arange(12.0).reshape(3, 2, 2)     # (c=3, h=2, w=2)
w = np.array([0.5, 0.25, 0.25])            # one weight per channel

# COLLAPSE: c pairs with the weights and is dropped -> weighted sum.
gray = np.einsum('chw,c->hw', img, w)
assert gray.shape == (2, 2)
# pixel (0,0): 0.5*img[0,0,0] + 0.25*img[1,0,0] + 0.25*img[2,0,0]
assert gray[0, 0] == 0.5 * 0.0 + 0.25 * 4.0 + 0.25 * 8.0

# KEEP: same pairing, c survives -> per-channel scaling.
s = np.array([1.0, 10.0, 100.0])
scaled = np.einsum('chw,c->chw', img, s)
assert scaled.shape == img.shape
assert scaled[1, 0, 0] == img[1, 0, 0] * 10.0     # channel 1 scaled by 10
# Broadcasting twin — the letters replaced None-counting:
assert np.allclose(scaled, img * s[:, None, None])

# Weighted AVERAGE of rows: weighted sum / total weight — divide outside.
a = np.array([[1.0, 2.0],
              [3.0, 4.0]])
wr = np.array([3.0, 1.0])
wavg = np.einsum('n,nd->d', wr, a) / wr.sum()
assert np.allclose(wavg, [1.5, 2.5])
```

Why each step:

1. The two specs differ ONLY in whether c appears after `->` — pausing on
   that pair is the fastest way to internalize keep-vs-drop as the
   sum-vs-scale switch.
2. The hand-computed pixel check is the standard verification for weighted
   ops: pick one output element, expand its formula, compare. One element
   suffices — the spec treats all positions identically.
3. In the weighted average, einsum handles the numerator only. The
   denominator (`w.sum()`) lives outside — remembering WHERE the division
   goes is most of q255.

## Faded practice

### q269
Weighted sum over channels: (c,h,w) and weights (c,) → (h,w).

```python starter
import numpy as np

def solve(img, w):
    """Per-pixel weighted sum across channels."""
    return np.einsum('_____', img, w)
```

```python solution
import numpy as np

def solve(img, w):
    """Per-pixel weighted sum across channels."""
    return np.einsum('chw,c->hw', img, w)
```

## Guided practice

### q291
1. Scale each channel by s[k], image shape unchanged — is the channel letter
   kept or dropped?
2. Same left-hand side as the weighted sum; the output keeps everything.
3. `'chw,c->chw'` — check one entry of a scaled channel.

## Independent practice

From the drill bank: q251 (project every token's features onto a vector —
which letter pairs, which survive?), q304 (per-timestep weights, summed over
time), q309 (weight attention heads by importance and sum them out),
q255 (weighted average of rows — numerator by einsum, denominator outside).

## Misconceptions

- **"The weight vector needs reshaping to match the tensor."** — In einsum
  the LETTER aligns it; `'chw,c->…'` is the whole alignment. Reshaping with
  None is the broadcasting spelling of the same thing — fine too, but count
  your Nones.
- **"Scale vs weighted-sum are different kinds of operation."** — Same
  multiply along the paired axis; the output side's keep/drop is the only
  difference. Task words map directly: "scale/weight each X" → keep;
  "combine/sum/average over X" → drop.
- **"einsum can compute the weighted average directly."** — No division in
  the notation. Weighted sum inside, `/ w.sum()` outside — and use the
  WEIGHTS' total, not the item count.
