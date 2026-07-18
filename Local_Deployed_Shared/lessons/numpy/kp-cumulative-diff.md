---
kc: numpy.cumulative-diff
title: Cumulative ops and discrete differences
supporting: [numpy.aggregations, numpy.axis-reductions]
new_syntax: []
faded: [234]
guided: [152]
independent: [82, 149, 22]
---

## Concept

Two families turn a sequence into a same-length (or nearly) sequence that
looks *along* it rather than across it:

**Running totals — `np.cumsum` and the accumulate family.**
`np.cumsum(x)` returns the running sum: entry i is `x[0] + … + x[i]`. Same
length as the input; last entry equals `x.sum()`. It is the vectorized
replacement for the "total so far" loop. Relatives: `np.cumprod` (running
product), and the general form **`np.ufunc.accumulate`** — any binary ufunc
can run cumulatively, so `np.maximum.accumulate(x)` is the *running maximum*
("largest value seen so far"), a pattern with no dedicated function of its
own. All take `axis=` on 2-D data: `z.cumsum(axis=1)` runs along each row.

**Adjacent differences — `np.diff`.**
`np.diff(x)` returns `x[1:] - x[:-1]`: entry i is the step from element i to
i+1. Length shrinks by 1. It's the discrete derivative, the inverse-ish of
cumsum (`np.diff(np.cumsum(x))` recovers `x[1:]`). Options mirror the family:
`axis=` chooses the direction on matrices, and `n=k` applies the operation k
times (second differences of a quadratic sequence are constant — a classic
check). Note the manual spelling `x[1:] - x[:-1]` is worth recognizing too:
shifted-slice arithmetic generalizes to gaps other than 1.

Reading a task: "running / so far / cumulative" → accumulate family;
"successive / adjacent / change between neighbors" → diff.

## Worked example

Task: running totals of daily sales; running maximum (record so far); daily
changes.

```python
import numpy as np

sales = np.array([2, 3, 5, 1, 4])

# Running total: same length, entry i = sum of entries 0..i.
totals = np.cumsum(sales)
assert totals.tolist() == [2, 5, 10, 11, 15]
assert totals[-1] == sales.sum()          # the last entry IS the total

# Running maximum: maximum.accumulate — "record high so far".
records = np.maximum.accumulate(sales)
assert records.tolist() == [2, 3, 5, 5, 5]

# Adjacent differences: one shorter, entry i = step i -> i+1.
changes = np.diff(sales)
assert changes.tolist() == [1, 2, -4, 3]
assert len(changes) == len(sales) - 1

# diff undoes cumsum (up to the first element):
assert np.diff(totals).tolist() == sales[1:].tolist()

# On matrices both take axis=. Along each ROW:
z = np.arange(6).reshape(2, 3)
assert z.cumsum(axis=1).tolist() == [[0, 1, 3], [3, 7, 12]]
assert np.diff(z, axis=1).tolist() == [[1, 1], [1, 1]]
```

Why each step:

1. `totals[-1] == sales.sum()` is the sanity anchor connecting cumsum to the
   plain reduction — cumsum is "all the partial answers", sum is the last
   one.
2. `maximum.accumulate` demonstrates the general principle: ANY binary ufunc
   accumulates. When a task asks for a running-anything, ask "which ufunc,
   accumulated?" before writing a loop.
3. The diff/cumsum round-trip and the explicit length bookkeeping (n vs n−1)
   preempt the two standard off-by-one surprises in this family.

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

## Guided practice

### q152
1. "Running maximum of each row, scanning left to right" — running-anything
   means an accumulate; which ufunc?
2. Per-row means the accumulation runs along axis 1.
3. `np.maximum.accumulate(z, axis=1)`.

## Independent practice

From the drill bank: q82 (running total along each row), q149 (k-th order
difference — one keyword, or the operation applied k times), q22 (differences
between adjacent COLUMNS — mind the axis and the output shape (r, c-1)).

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
