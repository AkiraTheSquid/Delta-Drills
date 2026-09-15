---
kc: numpy.stack-concat-interleave
title: Stacking, concatenating, interleaving
supporting: [numpy.reshape-flatten, numpy.slicing-views]
new_syntax: [Tensor.ravel, torch.cat, torch.cat#dim, torch.column_stack, torch.empty, torch.empty#dtype, torch.hstack, torch.stack, torch.stack#dim, torch.vstack]
concepts: [grow-existing-dimension, create-new-axis, interleave-streams]
faded: [974, 975, 84, 976]
guided: [146, 977]
independent: [89, 238, 159, 978, 979, 980]
integrated: [981, 982, 983]
---

## Concept: Grow an existing dimension with cat, vstack, and hstack

Combining arrays into one splits on a single question: **does the result have a NEW axis, or grow an EXISTING one?**

When you want to extend an existing dimension, use **`t.cat`** or its 2-D shorthands:
- `t.vstack([a, b])` stacks rows vertically (`b`'s rows placed below `a`'s), growing dimension 0.
- `t.hstack([a, b])` extends columns horizontally (`b`'s columns placed to the right of `a`'s), growing dimension 1.

All other dimension sizes must match exactly; the chosen axis simply adds up. The general form is `t.cat([a, b], dim=k)`. The 2-D shorthands are simply that call with `dim` fixed.

```python
import torch as t

pair = [t.tensor([[1, 2]]), t.tensor([[3, 4]])]
print("dim=0 (taller):", t.cat(pair, dim=0).shape, t.cat(pair, dim=0).tolist())
print("dim=1 (wider): ", t.cat(pair, dim=1).shape, t.cat(pair, dim=1).tolist())
print("vstack is dim=0:", t.equal(t.vstack(pair), t.cat(pair, dim=0)))
# Hidden checks
assert _delta_output == 'dim=0 (taller): torch.Size([2, 2]) [[1, 2], [3, 4]]\ndim=1 (wider):  torch.Size([1, 4]) [[1, 2, 3, 4]]\nvstack is dim=0: True\n'
```

## Worked example

Task: Join two matrices vertically (taller) and horizontally (wider).

```python
import torch as t

a = t.tensor([[1.0, 2.0],
              [3.0, 4.0]])
b = t.tensor([[5.0, 6.0],
              [7.0, 8.0]])

# Grow axis 0 (rows below) vs axis 1 (columns to the right)
v = t.vstack([a, b])
h = t.hstack([a, b])
print("vstack shape:", tuple(v.shape))
print("hstack shape:", tuple(h.shape))
# Hidden checks
assert v.shape == (4, 2) and h.shape == (2, 4)
assert v.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
assert h.tolist() == [[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]]
```

Why each step:
1. `t.vstack` joins along dimension 0: two `(2, 2)` matrices become `(4, 2)`.
2. `t.hstack` joins along dimension 1: two `(2, 2)` matrices become `(2, 4)`.
3. In both cases, the number of dimensions stays 2; only one existing axis gets longer.

## Faded practice

### q974
A seam along an axis that already exists: the row counts add up and the
column count is unchanged.

```python starter
import torch as t


def solve(a, b):
    """The two matrices joined top to bottom."""
    return t._____([a, b], dim=0)
```

```python solution
import torch as t


def solve(a, b):
    """The two matrices joined top to bottom."""
    return t.cat([a, b], dim=0)
```

### q975
The same kind of seam along the other axis, written with the 2-D shorthand
that fixes the axis for you.

```python starter
import torch as t


def solve(a, b):
    """The two matrices joined side by side."""
    return t._____([a, b])
```

```python solution
import torch as t


def solve(a, b):
    """The two matrices joined side by side."""
    return t.hstack([a, b])
```

## Concept: Create a new axis with stack

When you want to keep arrays distinct rather than merging them along an existing axis, use **`t.stack`**.

- `t.stack([a, b], dim=0)` piles k same-shape arrays into a new `(k, ...)` array.
- Nothing merges: you gain a brand-new dimension. Every input must have the exact same shape.

This is the bridge to **reductions over a collection of arrays**:
To compute the elementwise average of two arrays, stack them into a new axis and take the mean along that new axis: `t.stack([a, b], dim=0).mean(dim=0)`. Any "combine k arrays by taking the elementwise mean, max, or sum" is this two-step pattern.

```python
import torch as t

a = t.tensor([[1.0, 2.0], [3.0, 4.0]])
b = t.tensor([[5.0, 6.0], [7.0, 8.0]])

piled = t.stack([a, b], dim=0)
print("stacked shape:", tuple(piled.shape))
avg = piled.mean(dim=0)
print("elementwise average:", avg.tolist())
# Hidden checks
assert piled.shape == (2, 2, 2)
assert avg.tolist() == [[3.0, 4.0], [5.0, 6.0]]
```

## Worked example

Task: Combine two same-shape matrices into a pile, then compute their elementwise mean.

```python
import torch as t

a = t.tensor([[1.0, 2.0],
              [3.0, 4.0]])
b = t.tensor([[5.0, 6.0],
              [7.0, 8.0]])

# NEW axis (dim=0), then reduce it across that axis.
piled = t.stack([a, b], dim=0)
avg = piled.mean(dim=0)

print("pile shape:", tuple(piled.shape))
print("avg shape:", tuple(avg.shape))
print("averaged:", avg.tolist())
# Hidden checks
assert piled.shape == (2, 2, 2)
assert avg.shape == (2, 2)
assert avg.tolist() == [[3.0, 4.0], [5.0, 6.0]]
```

Why each step:
1. Two `(2, 2)` matrices stacked along `dim=0` produce `(2, 2, 2)`. A new axis is created at index 0.
2. `.mean(dim=0)` collapses that new axis, averaging across the two matrices to return a single `(2, 2)` matrix.

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

## Concept: Interleave arrays via pairing or strided assignment

Interleaving means alternating elements from multiple arrays: given `a = [a0, a1, a2]` and `b = [b0, b1, b2]`, produce `[a0, b0, a1, b1, a2, b2]`.

There are two idiomatic, loop-free approaches in PyTorch.

### Solution Version 1: Pair and Unroll (column_stack + ravel)

This approach works in two steps:
1. Pair elements side-by-side into columns using `t.column_stack((a, b))`. Each row `i` holds `[a[i], b[i]]`.
2. Flatten the 2D tensor using `.ravel()` (or `.flatten()`). Because PyTorch stores and reads tensors in row-major order (reading across row 0, then row 1, etc.), reading row-by-row naturally yields the alternating sequence: `a0, b0, a1, b1, ...`.

```python
import torch as t

a = t.tensor([1, 3, 5])
b = t.tensor([2, 4, 6])

paired = t.column_stack((a, b))
print("paired columns:\n", paired.tolist())

interleaved = paired.ravel()
print("unrolled row-major:", interleaved.tolist())
# Hidden checks
assert paired.tolist() == [[1, 2], [3, 4], [5, 6]]
assert interleaved.tolist() == [1, 2, 3, 4, 5, 6]
```

### Solution Version 2: Preallocate and Strided Assignment

Instead of reshaping, allocate the final output container and write each stream directly into its designated slots using strided slicing (`start::step`):
1. Preallocate an empty tensor of the total required size: `out = t.empty(len(a) + len(b), dtype=a.dtype)`.
2. Assign the first stream into even indices: `out[0::2] = a` (slots 0, 2, 4, ...).
3. Assign the second stream into odd indices: `out[1::2] = b` (slots 1, 3, 5, ...).

`t.empty` is uninitialized memory, which is completely safe here because every single slot is guaranteed to be overwritten before being read.

```python
import torch as t

a = t.tensor([1, 3, 5])
b = t.tensor([2, 4, 6])

out = t.empty(len(a) + len(b), dtype=a.dtype)
out[0::2] = a
out[1::2] = b

print("strided assignment:", out.tolist())
# Hidden checks
assert out.tolist() == [1, 2, 3, 4, 5, 6]
```

## Worked example

Task: Interleave two vectors, comparing both solution strategies.

```python
import torch as t

x = t.tensor([1, 3, 5])
y = t.tensor([2, 4, 6])

# Solution Version 1: Pair columns, then unroll row-major
sol1 = t.column_stack((x, y)).ravel()

# Solution Version 2: Preallocate and assign with stride 2
sol2 = t.empty(len(x) + len(y), dtype=x.dtype)
sol2[0::2] = x
sol2[1::2] = y

print("Solution 1 (pair & unroll):", sol1.tolist())
print("Solution 2 (strided assign):", sol2.tolist())
print("Both solutions match:", t.equal(sol1, sol2))
# Hidden checks
assert sol1.tolist() == [1, 2, 3, 4, 5, 6]
assert sol2.tolist() == [1, 2, 3, 4, 5, 6]
assert t.equal(sol1, sol2)
```

Comparing the two solution versions:
1. **Solution 1 (`column_stack` + `ravel`)** is compact and expressive for 2 or 3 same-shape streams. It relies on the memory layout of row-major traversal.
2. **Solution 2 (strided assignment)** is explicit and flexible. It easily generalizes to:
   - 3 or more streams: `out[0::3] = a`, `out[1::3] = b`, `out[2::3] = c`.
   - Spacing patterns and zero insertion: allocate `t.zeros(...)` and write with `out[::nz+1] = z`.

## Faded practice

### q976
Pair-and-unroll: once each row holds one entry from each source, reading
row by row IS the alternating order.

```python starter
import torch as t


def solve(a, b):
    """The two vectors alternated, a first."""
    return t.column_stack((a, b))._____()
```

```python solution
import torch as t


def solve(a, b):
    """The two vectors alternated, a first."""
    return t.column_stack((a, b)).ravel()
```

## Guided practice

### q146
1. Three streams interleaved position by position — the pair-and-ravel trick
   still works, but the strided form is clearer: what are the three residue
   classes?
2. Allocate the result (`t.empty(3 * n, dtype=...)` — dtype from the inputs
   via `t.result_type`), then one slice assignment per stream.
3. `out[0::3] = a; out[1::3] = b; out[2::3] = c`.


### q977
1. The two joining tools differ on one question: does the result gain an axis, or does an existing one get longer? Here every input has to stay identifiable, so it gains one.
2. `t.stack` takes the list and the position of the new axis. Piling k tensors of shape (r, c) at position 0 gives (k, r, c).
3. `t.stack(mats, dim=0)`, then `tuple(...shape)` on the result.

## Independent practice

From the drill bank: q89 (alternate two vectors), q238 (vertical AND
horizontal combination as a tuple),
q159 (nz zeros between consecutive entries — a zeros canvas plus ONE strided
assignment; derive the canvas length first).


From the drill bank: q978 (elementwise maximum across a list of tensors).
From the drill bank: q979 (three vectors interleaved in order).
From the drill bank: q980 (one constant column added on each side of a matrix).


## Integrated practice

### q981
Return both vertical and horizontal joins, with their shapes.

### q982
Insert a fixed number of zeros between consecutive entries.

### q983
Return the tensor pile, its elementwise mean, and its count.

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
