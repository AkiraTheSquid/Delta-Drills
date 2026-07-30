---
kc: numpy.centering
title: Centering and standardizing rows/columns
supporting: [numpy.axis-reductions, numpy.broadcasting-rules]
new_syntax: []
faded: [15, 7, 106]
guided: [154]
independent: [109, 218]
---

## Concept: the template — statistic, then operate

An enormous share of data preprocessing is one sentence: **compute a
statistic, then subtract/divide it back into the data.** Reduction produces
the statistic; broadcasting spreads it back. The template to internalize:

> `result = z  OP  z.STATISTIC(...)`

Pick the statistic (mean, std, min, max…), pick the operation (subtract to
center, divide to scale). The simplest case is **global centering**:
`z - z.mean()` — a scalar statistic, broadcasts everywhere; the result's
overall mean is 0, whatever z's shape.

## Worked example

```python
import torch as t

z = t.tensor([[1.0, 2.0, 3.0],
              [10.0, 20.0, 30.0]])

# Global centering: one scalar mean, subtracted everywhere.
centered = z - z.mean()
assert t.allclose(centered.mean(), t.tensor(0.0))
```

Why: the postcondition assertion (`mean ≈ 0`) restates the task's
*definition* — a cheap self-check, and `allclose` (not `==`) because float
arithmetic.

## Faded practice

### q15
Subtract the global mean (any input shape).

```python starter
import torch as t

def solve(z):
    """z minus its overall mean; result's mean is 0."""
    return z - z._____()
```

```python solution
import torch as t

def solve(z):
    """z minus its overall mean; result's mean is 0."""
    return z - z.mean()
```

## Concept: row and column centering — where keepdims earns its keep

Per-group centering picks an axis: which group shares a statistic?
Reading a task, the phrase "each row's …" means axis=1; "each column's …"
means axis=0.

- **Row centering**: `z - z.mean(axis=1, keepdims=True)` — the (r, 1)
  statistic broadcasts across each row; every row of the result averages 0.
- **Column centering**: `z - z.mean(axis=0)` — the (c,) statistic
  right-aligns against z's last axis, hitting each column.

Note the asymmetry: columns work WITHOUT keepdims because right-alignment
happens to be correct; rows NEED keepdims (or `[:, None]`). When in doubt,
keepdims is never wrong.

## Worked example

```python
import torch as t

z = t.tensor([[1.0, 2.0, 3.0],
              [10.0, 20.0, 30.0]])

# ROW centering: statistic per row -> dim=1, kept as a column (2, 1)
# so it broadcasts back across each row.
row_mu = z.mean(dim=1, keepdim=True)
assert row_mu.tolist() == [[2.0], [20.0]]
centered = z - row_mu
assert centered.tolist() == [[-1.0, 0.0, 1.0],
                             [-10.0, 0.0, 10.0]]
# Postcondition worth checking in real code: every row now averages 0.
assert t.allclose(centered.mean(dim=1), t.zeros(2))
```

Why: materializing `row_mu` and asserting its SHAPE (2, 1) before
subtracting is the discipline that prevents the classic square-matrix bug
(bare (2,) statistic aligning against the wrong axis).

## Faded practice

### q7
Subtract each row's own mean.

```python starter
import torch as t

def solve(z):
    """Every entry minus its row's mean; each result row averages 0."""
    return z - z.mean(dim=_____, keepdim=_____)
```

```python solution
import torch as t

def solve(z):
    """Every entry minus its row's mean; each result row averages 0."""
    return z - z.mean(dim=1, keepdim=True)
```

## Concept: standardizing — apply the template twice

**Standardizing** (z-scores) is centering AND scaling:
`(x - x.mean(axis=0)) / x.std(axis=0)` gives each column mean 0, std 1.
Both reductions share the same axis — mixing axes between the mean and the
std is a real bug seen in the wild; the template keeps them locked together.

One refinement for real data: when the divisor can be zero (a constant
column has std 0), guard the division with the `where=`/`out=` pattern from
the where-select KP — statistics and safe division compose cleanly.

## Worked example

```python
import torch as t

# COLUMN standardizing: both statistics per column (dim=0).
x = t.tensor([[1.0, 10.0],
              [3.0, 30.0]])
zscores = (x - x.mean(dim=0)) / x.std(dim=0, correction=0)
assert t.allclose(zscores, t.tensor([[-1.0, -1.0],
                                     [1.0, 1.0]]))
assert t.allclose(zscores.mean(dim=0), t.zeros(2))
assert t.allclose(zscores.std(dim=0, correction=0), t.ones(2))
```

Why: the two postconditions (`mean ≈ 0`, `std ≈ 1`) catch dim mistakes
instantly. Note the `correction=0`: PyTorch's `std` defaults to the SAMPLE
standard deviation (divide by n−1), so the population version these drills
want has to be asked for by name. Leaving it off does not raise — it just
returns slightly different numbers, which is the worst way to be wrong.

## Faded practice

### q106
Standardize each column (no constant columns).

```python starter
import torch as t

def solve(x):
    """Each column standardized: mean 0, std 1."""
    return (x - x.mean(dim=_____)) / x.std(dim=_____, correction=0)
```

```python solution
import torch as t

def solve(x):
    """Each column standardized: mean 0, std 1."""
    return (x - x.mean(dim=0)) / x.std(dim=0, correction=0)
```

## Guided practice

### q154
1. Two steps, in order: standardize per COLUMN, then clip. Column
   statistics reduce over dim=0.
2. Population std means dividing by n, not n-1 — torch's default is the
   sample version, so one keyword has to change.
3. `(z - z.mean(dim=0)) / z.std(dim=0, correction=0)`, then `t.clip(...,
   -limit, limit)`.

## Independent practice

From the drill bank: q109 (column centering — contrast with the row version
you faded), q218 (column standardize where constant columns must come out
ZERO, not NaN — compose the template with safe division).

## Misconceptions

- **"`z - z.mean(dim=1)` centers the rows."** — On a non-square matrix it
  ERRORS; on a square one it silently centers the wrong way (the bare (r,)
  aligns with columns). Row statistics need `keepdim=True` (or `[:, None]`).
- **"`std()` divides by n."** — PyTorch divides by n−1 by default
  (`correction=1`, the SAMPLE std). The population version is
  `std(correction=0)`. This is the opposite of NumPy's default, so a formula
  carried over from the numpy dialect changes its answer without complaining
  — always state `correction` explicitly.
- **"Center-then-scale needs a loop over rows."** — The statistic/broadcast
  template does any per-row/per-column normalization in one expression;
  loops over rows are a sign the axis machinery isn't being used.
