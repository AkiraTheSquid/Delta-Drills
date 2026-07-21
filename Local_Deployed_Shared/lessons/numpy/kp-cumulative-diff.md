---
kc: numpy.cumulative-diff
title: Cumulative ops and discrete differences
supporting: [numpy.aggregations, numpy.axis-reductions]
new_syntax: []
faded: [234, 152, 22]
guided: []
independent: [82, 149]
---

## Concept: cumsum — running totals

`np.cumsum(x)` returns the running sum: entry i is `x[0] + … + x[i]`. Same
length as the input; the last entry equals `x.sum()`. It is the vectorized
replacement for the "total so far" loop. `np.cumprod` is the running
product. On 2-D data both take `axis=`: `z.cumsum(axis=1)` runs along each
row.

Reading a task: "running / so far / cumulative" is the tell for this family.

## Worked example

```python
import numpy as np

sales = np.array([2, 3, 5, 1, 4])

# Running total: same length, entry i = sum of entries 0..i.
totals = np.cumsum(sales)
assert totals.tolist() == [2, 5, 10, 11, 15]
assert totals[-1] == sales.sum()          # the last entry IS the total
```

Why: `totals[-1] == sales.sum()` is the sanity anchor connecting cumsum to
the plain reduction — cumsum is "all the partial answers", sum is the last
one.

## Faded practice

### q234
Running total of a 1-D array.

```python starter
import numpy as np

def solve(x):
    """Entry i = total of x[0..i] inclusive."""
    return np._____(x)
```

```python solution
import numpy as np

def solve(x):
    """Entry i = total of x[0..i] inclusive."""
    return np.cumsum(x)
```

## Concept: ufunc.accumulate — running ANYTHING

cumsum is one member of a general form: **`np.ufunc.accumulate`** — any
binary ufunc can run cumulatively. `np.maximum.accumulate(x)` is the
*running maximum* ("largest value seen so far"), a pattern with no dedicated
function of its own; `np.minimum.accumulate` the running minimum. All take
`axis=` on matrices.

When a task asks for a running-anything, ask "which ufunc, accumulated?"
before writing a loop.

## Worked example

```python
import numpy as np

sales = np.array([2, 3, 5, 1, 4])

# Running maximum: maximum.accumulate — "record high so far".
records = np.maximum.accumulate(sales)
assert records.tolist() == [2, 3, 5, 5, 5]

# Per-row on a matrix: the accumulation runs along axis 1.
z = np.array([[3, 1, 4], [2, 6, 0]])
assert np.maximum.accumulate(z, axis=1).tolist() == [[3, 3, 4], [2, 6, 6]]
```

Why: `max(axis=...)` collapses the axis to ONE value; the running version
keeps every prefix's answer. "So far" in the task text is the tell that you
want accumulate, not a reduction.

## Faded practice

### q152
Running maximum of each row, scanning left to right.

```python starter
import numpy as np

def solve(z):
    """Entry [i, j] = max of row i's columns 0..j."""
    return np.maximum._____(z, axis=1)
```

```python solution
import numpy as np

def solve(z):
    """Entry [i, j] = max of row i's columns 0..j."""
    return np.maximum.accumulate(z, axis=1)
```

## Concept: np.diff — adjacent differences

`np.diff(x)` returns `x[1:] - x[:-1]`: entry i is the step from element i to
i+1. Length shrinks by 1 — n elements have n−1 adjacent gaps. It's the
discrete derivative, the inverse-ish of cumsum (`np.diff(np.cumsum(x))`
recovers `x[1:]`). Options mirror the family: `axis=` chooses the direction
on matrices, and `n=k` applies the operation k times (second differences of
a quadratic sequence are constant — a classic check).

The manual spelling `x[1:] - x[:-1]` is worth recognizing too:
shifted-slice arithmetic generalizes to gaps other than 1. Reading a task:
"successive / adjacent / change between neighbors" → diff.

## Worked example

```python
import numpy as np

sales = np.array([2, 3, 5, 1, 4])

# Adjacent differences: one shorter, entry i = step i -> i+1.
changes = np.diff(sales)
assert changes.tolist() == [1, 2, -4, 3]
assert len(changes) == len(sales) - 1

# diff undoes cumsum (up to the first element):
assert np.diff(np.cumsum(sales)).tolist() == sales[1:].tolist()

# On matrices, axis= picks the direction. Along each ROW:
z = np.arange(6).reshape(2, 3)
assert np.diff(z, axis=1).tolist() == [[1, 1], [1, 1]]
```

Why: the diff/cumsum round-trip and the explicit length bookkeeping (n vs
n−1) preempt the two standard off-by-one surprises in this family.

## Faded practice

### q22
Differences between adjacent columns within each row: result (r, c-1).

```python starter
import numpy as np

def solve(z):
    """Entry [i, j] = z[i, j+1] - z[i, j]."""
    return np.diff(z, axis=_____)
```

```python solution
import numpy as np

def solve(z):
    """Entry [i, j] = z[i, j+1] - z[i, j]."""
    return np.diff(z, axis=1)
```

## Independent practice

From the drill bank: q82 (running total along each row), q149 (k-th order
difference — one keyword, or the operation applied k times).

## Misconceptions

- **"cumsum needs a loop with a running variable."** — `np.cumsum` IS that
  loop, in compiled code. Same for running max/min/product via
  `ufunc.accumulate`.
- **"diff returns the same length."** — One shorter per application: n
  elements have n−1 adjacent gaps. `diff(x, n=k)` shrinks by k. Plan output
  shapes accordingly.
- **"Running maximum = np.max with axis."** — `max(axis=...)` collapses the
  axis to ONE value; the running version keeps every prefix's answer. "So
  far" in the task text is the tell that you want accumulate, not a
  reduction.
