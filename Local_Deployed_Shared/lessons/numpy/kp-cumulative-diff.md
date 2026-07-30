---
kc: numpy.cumulative-diff
title: Cumulative ops and discrete differences
supporting: [numpy.aggregations, numpy.axis-reductions]
new_syntax: [torch.cumsum, torch.cumsum#dim, torch.cummax, torch.cummax#dim, torch.cummin, torch.cumprod, torch.diff, torch.diff#dim, torch.diff#n]
faded: [234, 152, 22]
guided: []
independent: [82, 149]
---

## Concept: cumsum — running totals

Take a 1-D tensor `x` — one row of numbers, like five days of sales.
`t.cumsum(x, dim=0)` returns its **running total**: entry i is
`x[0] + … + x[i]`. The result is the same length as `x`, and its last entry is
the same number `x.sum()` would give you. It is the vectorized replacement for
the "total so far" loop.

`dim=0` says which direction to add along. A 1-D tensor has only one direction,
so `dim=0` is the only choice here — but torch makes you say it. NumPy would
have flattened the tensor and guessed; torch never guesses.

Reading a task: "running / so far / cumulative" is the tell for this family.

## Worked example

```python
import torch as t

# Five days of sales, as a 1-D tensor — one number per day.
sales = t.tensor([2, 3, 5, 1, 4])
print(sales)

# The running total. Entry i is the sum of days 0 through i, so entry 2 is
# 2 + 3 + 5 = 10. Same length as sales: one running total per day.
totals = t.cumsum(sales, dim=0)
print(totals)

# The LAST running total is the total of everything...
print(totals[-1])

# ...which is exactly what sum() gives you in one step.
print(sales.sum())
```

```output
tensor([2, 3, 5, 1, 4])
tensor([ 2,  5, 10, 11, 15])
tensor(15)
tensor(15)
```

Why: those last two prints are the anchor. `sum` gives you the final answer;
`cumsum` gives you every partial answer along the way, and its last entry is
the final one. Same computation, different amount of it kept.

## Faded practice

### q234
Running total of a 1-D tensor.

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

cumsum has siblings, and they all mean "so far". `t.cummax(x, dim)` is the
running **maximum** — the largest value seen up to that point. `t.cummin` is
the running minimum, and `t.cumprod` the running product.

The two extrema return a *pair*, not a single tensor: `.values` holds the
running values and `.indices` tells you where each new record was set. That is
the same shape `t.sort` and `t.topk` return, so the `.values` step will keep
coming up.

Every one of them requires an explicit `dim`, exactly like cumsum.

## Worked example

```python
import torch as t

# The same five days of sales.
sales = t.tensor([2, 3, 5, 1, 4])
print(sales)

# cummax returns a PAIR, so this is not the tensor you want yet.
record_pair = t.cummax(sales, dim=0)
print(record_pair)

# .values pulls out the running maximum: the best day so far. It never goes
# down — once you have seen a 5, the best-so-far stays at least 5.
records = record_pair.values
print(records)

# .indices says WHICH day set each record. Days 3 and 4 did not beat day 2,
# so both still point back at index 2.
print(record_pair.indices)
```

```output
tensor([2, 3, 5, 1, 4])
torch.return_types.cummax(
values=tensor([2, 3, 5, 5, 5]),
indices=tensor([0, 1, 2, 2, 2]))
tensor([2, 3, 5, 5, 5])
tensor([0, 1, 2, 2, 2])
```

Why: `sales.amax()` collapses the whole tensor to one number, 5. The running
version keeps the answer at every point instead. "So far" in a task is the
tell that you want a `cum*` and not a plain reduction.

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

`t.diff(x)` gives the **step between neighbours**: entry i is
`x[i+1] - x[i]`. The result is one shorter than the input, because n numbers
have only n−1 gaps between them.

It is the opposite of cumsum. cumsum adds a sequence up; diff reads back the
individual steps. `n=k` applies diff k times, and on a matrix `dim=` picks the
direction, the same argument cumsum takes.

Reading a task: "successive / adjacent / change between neighbours" → diff.

## Worked example

```python
import torch as t

# The same five days of sales.
sales = t.tensor([2, 3, 5, 1, 4])
print(sales)

# The day-to-day change. 2 -> 3 is +1, 3 -> 5 is +2, 5 -> 1 is -4, 1 -> 4 is
# +3. Four gaps between five days, so the result is one shorter.
changes = t.diff(sales)
print(changes)

# diff undoes cumsum: the steps of a running total ARE the original numbers,
# starting from the second one.
print(t.diff(t.cumsum(sales, dim=0)))

# ...which is sales without its first entry.
print(sales[1:])
```

```output
tensor([2, 3, 5, 1, 4])
tensor([ 1,  2, -4,  3])
tensor([3, 5, 1, 4])
tensor([3, 5, 1, 4])
```

Why: the round trip is the point — cumsum and diff are inverses, and the
length bookkeeping (n going to n−1) is the off-by-one this family is famous
for.

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
  loop, in compiled code. Same for the running max, min and product.
- **"diff returns the same length."** — One shorter per application: n
  numbers have n−1 gaps. `t.diff(x, n=k)` shrinks by k. Plan output shapes
  accordingly.
- **"Running maximum = amax with a dim."** — `amax(dim=...)` collapses the
  dimension to ONE value; the running version keeps the answer at every point.
  "So far" in the task text is the tell that you want a `cum*`.
- **"cummax returns a tensor."** — It returns a `(values, indices)` pair.
  Forgetting `.values` is the standard first mistake, and the error it causes
  shows up later, wherever the pair is finally used as a tensor.
