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

Given a 1-D tensor `x` — a single row of numbers, like five days of sales —
`t.cumsum(x, dim=0)` returns its running sum: entry i is `x[0] + … + x[i]`.
The result is the same length as `x`, and its last entry equals `x.sum()`.
It is the vectorized replacement for the "total so far" loop.

`dim=0` says which direction to accumulate along. `x` has only one direction,
so it is the only choice here — but unlike NumPy, torch REQUIRES you to say
so: there is no flatten-by-default.

Reading a task: "running / so far / cumulative" is the tell for this family.

## Worked example

```python
import torch as t

sales = t.tensor([2, 3, 5, 1, 4])

# Running total: same length, entry i = sum of entries 0..i.
totals = t.cumsum(sales, dim=0)
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
import torch as t

def solve(x):
    """Entry i = total of x[0..i] inclusive."""
    return t._____(x, dim=0)
```

```python solution
import torch as t

def solve(x):
    """Entry i = total of x[0..i] inclusive."""
    return t.cumsum(x, dim=0)
```

## Concept: the cum* family — running ANYTHING

cumsum has siblings. **`t.cummax(x, dim)`** is the *running maximum*
("largest value seen so far") and **`t.cummin`** the running minimum;
`t.cumprod` the running product. Like `t.sort` and `t.topk`, the cum-extrema
return a `(values, indices)` pair, so the running values are `.values` and
`.indices` tells you WHERE each record was set.

NumPy generalizes this as `np.ufunc.accumulate` over any binary ufunc;
PyTorch instead ships the four that matter as named functions, and every one
of them requires an explicit `dim`.

When a task asks for a running-anything, reach for a `cum*` before writing a
loop.

## Worked example

```python
import torch as t

sales = t.tensor([2, 3, 5, 1, 4])

# Running maximum: cummax — "record high so far". Note .values.
records = t.cummax(sales, dim=0).values
assert records.tolist() == [2, 3, 5, 5, 5]

# Per-row on a matrix: the accumulation runs along dim 1.
z = t.tensor([[3, 1, 4], [2, 6, 0]])
assert t.cummax(z, dim=1).values.tolist() == [[3, 3, 4], [2, 6, 6]]
```

Why: `amax(dim=...)` collapses the dimension to ONE value; the running
version keeps every prefix's answer. "So far" in the task text is the tell
that you want a `cum*`, not a reduction.

## Faded practice

### q152
Running maximum of each row, scanning left to right.

```python starter
import torch as t

def solve(z):
    """Entry [i, j] = max of row i's columns 0..j."""
    return t._____(z, dim=1).values
```

```python solution
import torch as t

def solve(z):
    """Entry [i, j] = max of row i's columns 0..j."""
    return t.cummax(z, dim=1).values
```

## Concept: t.diff — adjacent differences

`t.diff(x)` returns `x[1:] - x[:-1]`: entry i is the step from element i to
i+1. Length shrinks by 1 — n elements have n−1 adjacent gaps. It's the
discrete derivative, the inverse-ish of cumsum (`t.diff(t.cumsum(x, dim=0))`
recovers `x[1:]`). Options mirror the family: `axis=` chooses the direction
on matrices, and `n=k` applies the operation k times (second differences of
a quadratic sequence are constant — a classic check).

The manual spelling `x[1:] - x[:-1]` is worth recognizing too:
shifted-slice arithmetic generalizes to gaps other than 1. Reading a task:
"successive / adjacent / change between neighbors" → diff.

## Worked example

```python
import torch as t

sales = t.tensor([2, 3, 5, 1, 4])

# Adjacent differences: one shorter, entry i = step i -> i+1.
changes = t.diff(sales)
assert changes.tolist() == [1, 2, -4, 3]
assert len(changes) == len(sales) - 1

# diff undoes cumsum (up to the first element):
assert t.diff(t.cumsum(sales, dim=0)).tolist() == sales[1:].tolist()

# On matrices, dim= picks the direction. Along each ROW:
z = t.arange(6).reshape(2, 3)
assert t.diff(z, dim=1).tolist() == [[1, 1], [1, 1]]
```

Why: the diff/cumsum round-trip and the explicit length bookkeeping (n vs
n−1) preempt the two standard off-by-one surprises in this family.

## Faded practice

### q22
Differences between adjacent columns within each row: result (r, c-1).

```python starter
import torch as t

def solve(z):
    """Entry [i, j] = z[i, j+1] - z[i, j]."""
    return t.diff(z, dim=_____)
```

```python solution
import torch as t

def solve(z):
    """Entry [i, j] = z[i, j+1] - z[i, j]."""
    return t.diff(z, dim=1)
```

## Independent practice

From the drill bank: q82 (running total along each row), q149 (k-th order
difference — one keyword, or the operation applied k times).

## Misconceptions

- **"cumsum needs a loop with a running variable."** — `t.cumsum` IS that
  loop, in compiled code. Same for running max/min/product via
  `ufunc.accumulate`.
- **"diff returns the same length."** — One shorter per application: n
  elements have n−1 adjacent gaps. `diff(x, n=k)` shrinks by k. Plan output
  shapes accordingly.
- **"Running maximum = amax with dim."** — `amax(dim=...)` collapses the
  axis to ONE value; the running version keeps every prefix's answer. "So
  far" in the task text is the tell that you want accumulate, not a
  reduction.
