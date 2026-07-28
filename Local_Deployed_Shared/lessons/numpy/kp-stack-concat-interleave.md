---
kc: numpy.stack-concat-interleave
title: Stacking, concatenating, interleaving
supporting: [numpy.reshape-flatten, numpy.slicing-views]
new_syntax: []
faded: [84]
guided: [146]
independent: [89, 238, 159]
---

## Concept

Combining arrays into one splits on a single question: **does the result have
a NEW axis, or grow an EXISTING one?**

- **Grow an existing dimension — `t.cat` and its 2-D shorthands.**
  `t.vstack([a, b])` stacks rows (b's rows below a's);
  `t.hstack([a, b])` extends rows sideways. Shapes must agree on the other
  axis; the combined axis just adds up. General form:
  `t.cat([a, b], dim=k)`.
- **Create a new axis — `t.stack`.**
  `t.stack([a, b], axis=0)` piles k same-shape arrays into a (k, …) array.
  Nothing merges; you gain a dimension. This is the bridge to *reductions
  over the pile*: the elementwise average of two arrays is
  `t.stack([a, b]).mean(axis=0)` — stack, then reduce the new axis. Any
  "combine k arrays by taking the elementwise mean/max/median" is this
  two-step.
- **Interleave — stack + reshape, or strided assignment.**
  Alternating elements `[a0, b0, a1, b1, …]` has two idiomatic spellings:
  - `t.column_stack((a, b)).ravel()` — pair up (each row `[a_i, b_i]`),
    then read row-major: the pairs unroll in exactly alternating order.
    (Reshape's fill order doing real work!)
  - Preallocate and stride: `out[0::2] = a; out[1::2] = b` — allocate the
    full-length result, then write each source into its residue class.
    Generalizes cleanly to 3+ sources (`0::3`, `1::3`, `2::3`) and to
    "insert nz zeros between entries" (`out[::nz+1] = z` into a zeros
    canvas).

Choosing: piles that keep identity → stack; seams along an axis → concat
family; alternating patterns → column_stack+ravel or strided slots.

## Worked example

Task: stack two matrices vertically and horizontally; average them
elementwise via a stack; interleave two vectors.

```python
import torch as t

a = t.tensor([[1.0, 2.0],
              [3.0, 4.0]])
b = t.tensor([[5.0, 6.0],
              [7.0, 8.0]])

# Grow axis 0 (rows below) / axis 1 (columns to the right).
v = t.vstack([a, b])
h = t.hstack([a, b])
assert v.shape == (4, 2) and h.shape == (2, 4)

# NEW axis then reduce it: elementwise average of the two arrays.
piled = t.stack([a, b], dim=0)        # shape (2, 2, 2) — nothing merged
assert piled.shape == (2, 2, 2)
avg = piled.mean(dim=0)                # collapse the pile
assert avg.tolist() == [[3.0, 4.0], [5.0, 6.0]]

# Interleave two vectors: pair rows, then row-major ravel unrolls
# them alternately.
x = t.tensor([1, 3, 5])
y = t.tensor([2, 4, 6])
inter = t.column_stack((x, y)).ravel()
assert inter.tolist() == [1, 2, 3, 4, 5, 6]

# Same result by strided assignment — the form that scales to 3+ streams.
out = t.empty(6, dtype=x.dtype)
out[0::2] = x
out[1::2] = y
assert out.tolist() == [1, 2, 3, 4, 5, 6]
```

Why each step:

1. Track shapes: vstack (2,2)+(2,2)→(4,2) grew an axis; stack →(2,2,2) added
   one. The shape arithmetic is the reliable way to tell which operation a
   task describes.
2. stack-then-reduce turns "elementwise average/max of k arrays" into the
   axis machinery you already own — no dedicated function needed, and it
   generalizes from mean to any reduction.
3. Both interleave spellings matter: column_stack+ravel is elegant for two
   streams; residue-class assignment (`empty` first — safe here because
   every slot gets written) reads mechanically but handles any number of
   streams and irregular spacings.

## Faded practice

### q84
Elementwise average of two same-shape arrays, via a new axis.

```python starter
import torch as t

def solve(a, b):
    """Elementwise average: stack on a new axis, then reduce it."""
    return t.stack([a, b], dim=0)._____(dim=0)
```

```python solution
import torch as t

def solve(a, b):
    """Elementwise average: stack on a new axis, then reduce it."""
    return t.stack([a, b], dim=0).mean(dim=0)
```

## Guided practice

### q146
1. Three streams interleaved position by position — the pair-and-ravel trick
   still works, but the strided form is clearer: what are the three residue
   classes?
2. Allocate the result (`t.empty(3 * n, dtype=...)` — dtype from the inputs
   via `t.result_type`), then one slice assignment per stream.
3. `out[0::3] = a; out[1::3] = b; out[2::3] = c`.

## Independent practice

From the drill bank: q89 (alternate two vectors), q238 (vertical AND
horizontal combination as a tuple),
q159 (nz zeros between consecutive entries — a zeros canvas plus ONE strided
assignment; derive the canvas length first).

## Misconceptions

- **"stack and concatenate are synonyms."** — concatenate grows an existing
  axis (no new dimension); stack creates a new one. (2,3)+(2,3): concat
  axis-0 → (4,3); stack → (2,2,3). The task's result shape tells you which.
- **"Interleaving needs a Python loop."** — Either pair-and-ravel
  (column_stack + row-major flatten) or strided slice assignment. Both are
  single-pass, loop-free.
- **"t.empty is dangerous here."** — It's uninitialized memory, which is
  fine EXACTLY when every slot gets written before any read — as in the
  residue-class pattern. If any slot might stay untouched (the zeros-between
  drill!), start from `t.zeros` instead.
