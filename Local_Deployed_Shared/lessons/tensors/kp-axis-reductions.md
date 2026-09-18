---
kc: torch.axis-reductions
title: Reductions along an axis — and keepdims
supporting: [torch.aggregations, torch.broadcasting-rules]
new_syntax: [Tensor.mean#dim, Tensor.mean#keepdim, Tensor.sum#dim, syntax.bool-literal]
faded: [220, 1363, 1364, 135, 1365, 1366]
guided: [503, 504]
independent: [108, 505, 174, 1367, 1368, 1369, 1370, 1471, 1472, 1473, 1475, 1477, 1479, 1480, 1482]
integrated: [1371, 1372, 1373]
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

The example below takes one (2, 3) float matrix and reduces it twice: once
across its columns to get a total per row, once down its rows to get a mean
per column. Before each print, predict the output shape by crossing the named
axis out of `(2, 3)`.

```python
import torch as t

x = t.tensor([[1.0, 2.0, 3.0],
              [10.0, 20.0, 30.0]])    # shape (2, 3) — float, so mean works

# dim=1 -> the 3 columns collapse -> one total PER ROW.
row_sums = x.sum(dim=1)

# dim=0 -> the 2 rows collapse onto each other -> one mean PER COLUMN.
col_means = x.mean(dim=0)
print("x shape", tuple(x.shape))
print("sum(dim=1) ", row_sums,  "shape", tuple(row_sums.shape))
print("mean(dim=0)", col_means, "shape", tuple(col_means.shape))
# Hidden checks
assert tuple(row_sums.shape) == (2,)  # (2, 3) with dim 1 crossed out
assert row_sums.tolist() == [6.0, 60.0]
assert tuple(col_means.shape) == (3,)
assert col_means.tolist() == [5.5, 11.0, 16.5]
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
    return x.sum(_____=_____)
```

```python solution
import torch as t

def solve(x):
    """Column sums of a 2-D matrix: which axis disappears?"""
    return x.sum(dim=0)
```

### q1363
Three axes now: (n, r, c) becomes (n, r), one total per row of every
matrix.

```python starter
import torch as t

def solve(x):
    """Row totals for every one of the n matrices."""
    return x._____(_____=_____)
```

```python solution
import torch as t

def solve(x):
    """Row totals for every one of the n matrices."""
    return x.sum(dim=-1)
```

### q1364
Same tensor, opposite question: every position averaged across the n
matrices, (n, r, c) → (r, c).

```python starter
import torch as t

def solve(x):
    """Average the n matrices down to one."""
    return x._____(_____=_____)
```

```python solution
import torch as t

def solve(x):
    """Average the n matrices down to one."""
    return x.mean(dim=0)
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

The example below reduces a 4-D batch over TWO axes in one call, then shows
what `keepdim=True` changes when a row mean is subtracted back from its
matrix. Both halves hinge on the output shape, so predict it first.

```python
import torch as t

# Tuple of axes on a 4-D batch (a, b, c, d): collapse the two LEADING axes
# in one call -> one total per (c, d) position.
batch = t.arange(24).reshape(2, 3, 2, 2)
totals = batch.sum(dim=(0, 1))
print("(2,3,2,2) summed over the first two ->", tuple(totals.shape))
print(totals)
# Hidden checks
assert totals.shape == (2, 2)
assert totals.tolist() == [[60, 66], [72, 78]]
```

Two axes named, two axes gone: `(2, 3, 2, 2)` with the first two crossed out
is `(2, 2)`, and each entry is a total over all 6 (a, b) combinations.
Negative axes would name the trailing pair instead — the per-image case the
concept text described. Now the keepdim switch, on a float matrix (`mean`
refuses ints):

```python
x = t.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
rm = x.mean(dim=1, keepdim=True)      # (2, 1), not (2,)
centered = x - rm                     # (2,3) - (2,1): broadcasts by row
print("keepdim=True keeps the axis as 1:", tuple(rm.shape), "->", rm.tolist())
print(centered)
# Hidden checks
assert tuple(rm.shape) == (2, 1)
assert centered[0].tolist() == [-1.0, 0.0, 1.0]
```

Why: a bare `(2,)` row-mean would align against the WRONG axis when
broadcast (right-aligned → columns) — the source of a classic silent bug
when r = c. keepdims makes the intended alignment explicit.

## Faded practice

### q135
Per-slice totals of a 4-D batch: (a, b, c, d) → (a, b), the total of each
(c, d) slice.

```python starter
import torch as t

def solve(x):
    """(a, b, c, d) -> (a, b): total of each c*d slice."""
    return x.sum(_____=_____)
```

```python solution
import torch as t

def solve(x):
    """(a, b, c, d) -> (a, b): total of each c*d slice."""
    return x.sum(dim=(-2, -1))
```

### q1365
The reduction is the easy part; the SHAPE of the result is the point — column
means as (1, c), not (c,).

```python starter
import torch as t

def solve(x):
    """Column means that still have two axes."""
    return x._____(_____=_____, _____=_____)
```

```python solution
import torch as t

def solve(x):
    """Column means that still have two axes."""
    return x.mean(dim=0, keepdim=True)
```

### q1366
One mean per (h, w) slice, and the result keeps all three axes:
(n, h, w) → (n, 1, 1).

```python starter
import torch as t

def solve(x):
    """One mean per slice, shaped to broadcast back over it."""
    return x._____(_____=_____, _____=_____)
```

```python solution
import torch as t

def solve(x):
    """One mean per slice, shaped to broadcast back over it."""
    return x.mean(dim=(-2, -1), keepdim=True)
```

## Solo practice

### q108
One quantile value for each row of a matrix.

### q505
Rescale each row so its entries sum to 1.

### q174
Per-row median with an odd column count — a reduction torch spells differently
from mean.

### q1367
A per-axis reduction beside a whole-tensor one — and a scalar broadcasts
against anything.

### q1368
Per-image means of a (b, h, w) batch: two axes have to disappear and one has
to survive.

### q1369
Reduce, then use the result against the original — the shape you keep decides
which axis it lines up with.

### q1370
A reduction feeding another reduction: the first one chooses the axis, the
second has none left to choose.

### q1471
Per-row totals from a matrix.

### q1472
Per-column totals from a matrix.

### q1473
Per-row means from a matrix.

### q1475
Per-image sums over both spatial axes.

### q1477
Row sums retaining length-one axis for broadcasting.

### q1479
A tuple reduction over the first two axes.

### q1480
Column means retaining the reduced axis.

### q1482
Per-image sums retaining every reduced axis.

## Integrated practice

### q1371
Each row becomes shares of its own total, then the average share of each
column — returned as a (1, c) tensor.

### q1372
Every image divided by its own total, so each image's pixels sum to 1.

### q1373
The same batch reduced three ways — every axis choice on the page at once.

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
