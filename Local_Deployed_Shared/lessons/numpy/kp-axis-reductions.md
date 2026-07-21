---
kc: numpy.axis-reductions
title: Reductions along an axis — and keepdims
supporting: [numpy.aggregations, numpy.broadcasting-rules]
new_syntax: []
faded: [220, 135]
guided: []
independent: [108, 130]
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

Everything from the aggregation KP takes `axis=`: `mean`, `min`, `max`,
`std`, `any`, `all`, `argmax`, plus `np.quantile` and friends.

## Worked example

```python
import numpy as np

x = np.array([[1, 2, 3],
              [10, 20, 30]])          # shape (2, 3)

# axis=0 -> the 2 rows collapse onto each other -> one sum PER COLUMN.
col_sums = x.sum(axis=0)
assert col_sums.shape == (3,)         # (2, 3) with axis 0 crossed out
assert col_sums.tolist() == [11, 22, 33]

# axis=1 -> the 3 columns collapse -> one value PER ROW.
row_means = x.mean(axis=1)
assert row_means.shape == (2,)
assert row_means.tolist() == [2.0, 20.0]
```

Why: for each reduction, the assert on `.shape` comes BEFORE the values —
that's the recommended order in your own code too: predict the shape by
crossing out the named axis, then check the numbers.

## Faded practice

### q220
One sum per column.

```python starter
import numpy as np

def solve(x):
    """Column sums of a 2-D matrix: which axis disappears?"""
    return x.sum(axis=_____)
```

```python solution
import numpy as np

def solve(x):
    """Column sums of a 2-D matrix: which axis disappears?"""
    return x.sum(axis=0)
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
import numpy as np

# Tuple of axes on a 4-D batch (a, b, c, d): collapse the last two ->
# one total per (a, b) slice. Negative axes save counting.
batch = np.arange(24).reshape(2, 3, 2, 2)
totals = batch.sum(axis=(-2, -1))
assert totals.shape == (2, 3)
assert totals[0, 0] == 0 + 1 + 2 + 3

# keepdims preview: the reduced axis survives as 1, so the result still
# lines up against the original for broadcasting.
x = np.array([[1, 2, 3], [10, 20, 30]])
rm = x.mean(axis=1, keepdims=True)
assert rm.shape == (2, 1)
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
import numpy as np

def solve(x):
    """(a, b, c, d) -> (a, b): total of each c*d slice."""
    return x.sum(axis=_____)
```

```python solution
import numpy as np

def solve(x):
    """(a, b, c, d) -> (a, b): total of each c*d slice."""
    return x.sum(axis=(-2, -1))
```

## Independent practice

From the drill bank: q108 (row quantiles — `np.quantile` takes `axis=` like
everything else; watch which axis gives per-row results), q130 (index of each
row's first nonzero — build a boolean mask, then argmax along the right axis;
why does argmax find the FIRST True?).

## Misconceptions

- **"axis=0 gives row sums."** — axis=0 REMOVES axis 0: the rows collapse
  together, yielding one result per column. Cross the axis out of the shape
  tuple and read what's left.
- **"keepdims is cosmetic."** — It preserves alignment for broadcasting.
  `x - x.mean(axis=1)` on a square matrix runs WITHOUT error and quietly
  subtracts along the wrong axis; `keepdims=True` (shape (r,1)) makes the
  intended row-wise alignment explicit and correct.
- **"Reducing two axes needs two calls."** — `axis=(1, 2)` collapses both in
  one pass. Chained single-axis calls also shift the axis numbering between
  calls — a tuple avoids that trap entirely.
