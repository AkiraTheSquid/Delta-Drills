---
kc: numpy.axis-reductions
title: Reductions along an axis — and keepdims
supporting: [numpy.aggregations, numpy.broadcasting-rules]
new_syntax: [Tensor.mean#dim, Tensor.mean#keepdim, Tensor.sum#dim]
faded: [220, 135]
guided: [503, 504]
independent: [108, 130, 505, 174]
---

## Concept: axis= — the axis you name disappears

Whole-array reductions collapse everything to one number. Add **`axis=`** and
the reduction collapses **only that axis**, leaving the rest of the shape
intact:

> **The axis you name is the axis that DISAPPEARS.**

For a (r, c) matrix:

- `x.sum(axis=0)` — axis 0 (rows) disappears → shape (c,): **column sums**
  (you summed *down* each column).
- `x.sum(axis=1)` — axis 1 disappears → shape (r,): **row sums**.

The naming feels backwards until you anchor it: `axis=0` does NOT mean
"per-row results", it means "reduce ALONG axis 0" — the r rows are collapsed
on top of each other. Predict the output shape first (cross the named axis
out of the shape tuple) and the direction sorts itself out.

Everything from the aggregation KP takes `dim=`: `mean`, `amin`, `amax`,
`std`, `any`, `all`, `argmax`, plus `t.quantile` and friends. Two PyTorch
wrinkles carry through this whole KP: `mean` refuses an integer tensor (cast
with `.to(t.float32)` first), and `std` divides by n−1 unless you pass
`correction=0`.

## Worked example

```python
import torch as t

x = t.tensor([[1.0, 2.0, 3.0],
              [10.0, 20.0, 30.0]])    # shape (2, 3) — float, so mean works

# dim=0 -> the 2 rows collapse onto each other -> one sum PER COLUMN.
col_sums = x.sum(dim=0)
assert tuple(col_sums.shape) == (3,)  # (2, 3) with dim 0 crossed out
assert col_sums.tolist() == [11.0, 22.0, 33.0]

# dim=1 -> the 3 columns collapse -> one value PER ROW.
row_means = x.mean(dim=1)
assert tuple(row_means.shape) == (2,)
assert row_means.tolist() == [2.0, 20.0]
```

Why: for each reduction, the assert on `.shape` comes BEFORE the values —
that's the recommended order in your own code too: predict the shape by
crossing out the named axis, then check the numbers.

## Faded practice

### q220
One sum per column.

```python starter
import torch as t

def solve(x):
    """Column sums of a 2-D matrix: which axis disappears?"""
    return x.sum(dim=_____)
```

```python solution
import torch as t

def solve(x):
    """Column sums of a 2-D matrix: which axis disappears?"""
    return x.sum(dim=0)
```

## Concept: tuples of axes, and keepdims

Higher-rank arrays allow a *tuple* of axes — `x.sum(axis=(-2, -1))` collapses
the last two dimensions at once (e.g. summing each image of a batch), and
negative indices count from the end just like in indexing. That makes
"per-image" reductions one call, robust to how many leading batch axes exist.

One more switch on the same call: **`keepdims=True`** keeps the reduced axis
as length 1 instead of deleting it — shape (r, c) → (r, 1) rather than (r,).
Why you'd want that: a (r, 1) result broadcasts back against the original
(r, c) *by row*. The reduce → keepdims → operate pipeline is the heart of the
next KP (centering), where you'll practice it.

## Worked example

```python
import torch as t

# Tuple of axes on a 4-D batch (a, b, c, d): collapse the last two ->
# one total per (a, b) slice. Negative axes save counting.
batch = t.arange(24).reshape(2, 3, 2, 2)
totals = batch.sum(dim=(-2, -1))
assert totals.shape == (2, 3)
assert totals[0, 0] == 0 + 1 + 2 + 3

# keepdim preview: the reduced dim survives as 1, so the result still
# lines up against the original for broadcasting.
# (t.mean needs a float tensor — it will not promote ints the way numpy does.)
x = t.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
rm = x.mean(dim=1, keepdim=True)
assert tuple(rm.shape) == (2, 1)
centered = x - rm                     # (2,3) - (2,1): broadcasts by row
assert centered[0].tolist() == [-1.0, 0.0, 1.0]
```

Why: a bare `(2,)` row-mean would align against the WRONG axis when
broadcast (right-aligned → columns) — the source of a classic silent bug
when r = c. keepdims makes the intended alignment explicit.

## Faded practice

### q135
Per-slice totals of a 4-D batch: collapse the LAST two axes in one call.

```python starter
import torch as t

def solve(x):
    """(a, b, c, d) -> (a, b): total of each c*d slice."""
    return x.sum(dim=_____)
```

```python solution
import torch as t

def solve(x):
    """(a, b, c, d) -> (a, b): total of each c*d slice."""
    return x.sum(dim=(-2, -1))
```

## Independent practice

From the drill bank: q108 (row quantiles — `t.quantile` takes `axis=` like
everything else; watch which axis gives per-row results), q130 (index of each
row's first nonzero — build a boolean mask, then argmax along the right axis;
why does argmax find the FIRST True?).

From the drill bank: q505 (make every row sum to 1 — this is what keepdim was for).

Also from the bank: q174 (per-row median with an odd column count — a
reduction torch spells differently from mean).

## Guided practice

### q503
1. One number per row means the COLUMN axis has to go.
2. The axis you name in `dim=` is the one that disappears, so name the one
   you are collapsing, not the one you are keeping.
3. `x.sum(dim=1)`.

### q504
1. Same reduction as before, but the result must still have two axes.
2. There is a keyword that leaves the collapsed axis behind at length 1,
   which is exactly what a later broadcast needs.
3. `x.sum(dim=1, keepdim=True)`.

## Misconceptions

- **"axis=0 gives row sums."** — axis=0 REMOVES axis 0: the rows collapse
  together, yielding one result per column. Cross the axis out of the shape
  tuple and read what's left.
- **"keepdims is cosmetic."** — It preserves alignment for broadcasting.
  `x - x.mean(dim=1)` on a square matrix runs WITHOUT error and quietly
  subtracts along the wrong axis; `keepdims=True` (shape (r,1)) makes the
  intended row-wise alignment explicit and correct.
- **"Reducing two axes needs two calls."** — `axis=(1, 2)` collapses both in
  one pass. Chained single-axis calls also shift the axis numbering between
  calls — a tuple avoids that trap entirely.
