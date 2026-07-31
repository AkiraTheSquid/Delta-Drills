---
kc: numpy.ranges
title: Numeric ranges — arange and linspace
supporting: [numpy.constructors, numpy.slicing-views]
new_syntax: [torch.arange, torch.linspace]
faded: [229, 242, 214]
guided: []
independent: [53]
---

## Concept: t.arange — the stop is exclusive

When you know the **step**, use **`t.arange(start, stop, step)`** (step
defaults to 1). It counts from `start` in increments of `step` and — exactly
like Python's `range` — **stops BEFORE `stop`**. The endpoint is never
included, even with a step: `t.arange(0, 10, 2)` is `[0, 2, 4, 6, 8]` — 10
is left out.

```python
import torch as t

print(t.arange(5))
print(t.arange(0, 10, 2))          # 10 is the stop, so 10 is missing
print(t.arange(0, 11, 2))          # push the stop past it to get it back
```

That exclusive stop is the whole trick, and the source of most range bugs.
When a task wants the endpoint *included*, you have to extend the stop past
where you want to end.

`t.arange` over integers gives you an **integer** tensor (`int64`), which
matters because integer tensors are what you index with.

```python
idx = t.arange(3)
letters = t.tensor([10, 20, 30, 40])
print(idx.dtype, "->", letters[idx])
assert idx.dtype == t.int64
```

## Worked example

```python
import torch as t

# Counting by 2 up to 10 — but 10 is the stop, so it's EXCLUDED.
evens = t.arange(0, 10, 2)
assert evens.tolist() == [0, 2, 4, 6, 8]
assert evens.dtype == t.int64
print(evens, evens.dtype, "  <- no 10")
```

Why: notice 10 never appears. The step doesn't change the rule — `arange`
always halts one step short of `stop`.

## Faded practice

### q229
Every integer from `start` to `end`, **including both endpoints**. (Watch the
exclusive stop — how do you make `end` appear?)

```python starter
import torch as t

def solve(start, end):
    """Integers start..end inclusive, in order."""
    return t.arange(start, _____)
```

```python solution
import torch as t

def solve(start, end):
    """Integers start..end inclusive, in order."""
    return t.arange(start, end + 1)
```

## Concept: t.linspace — you know the number of points

When you know the **number of points** instead of the step, use
**`t.linspace(start, stop, num)`**: exactly `num` evenly spaced values, and
this time **both endpoints are included**. Mind the fencepost — `num` points
make `num − 1` gaps, so `t.linspace(0.0, 1.0, 5)` has step 1/4, not 1/5.

```python
import torch as t

grid = t.linspace(0.0, 1.0, 5)
print(grid)
print("gaps:", t.diff(grid))
assert grid[0].item() == 0.0 and grid[-1].item() == 1.0
```

Both ends present, four gaps between five points. Side by side with `arange`
the two conventions are hard to confuse again:

```python
print("arange  ", t.arange(0.0, 1.0, 0.25))     # stop excluded -> 4 values
print("linspace", t.linspace(0.0, 1.0, 5))      # stop included -> 5 values
assert len(t.arange(0.0, 1.0, 0.25)) == 4
assert len(t.linspace(0.0, 1.0, 5)) == 5
```

## Worked example

```python
import torch as t

# 5 points from 0 to 1, endpoints INCLUDED -> 4 equal gaps of 0.25.
grid = t.linspace(0.0, 1.0, 5)
assert grid.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
print(grid, " 5 points,", len(grid) - 1, "gaps")
```

Why: both 0.0 and 1.0 are present — that's the opposite of `arange`. Count
the points, not the intervals.

## Faded practice

### q242
The `n` evenly spaced breakpoints **strictly inside** (0, 1) — exclude 0.0 and
1.0. (linspace includes the endpoints; how do you get `n` points *between*
them?)

```python starter
import torch as t

def solve(n):
    """n interior breakpoints of (0, 1), endpoints excluded."""
    return t.linspace(0.0, 1.0, n + _____)[1:-1]
```

```python solution
import torch as t

def solve(n):
    """n interior breakpoints of (0, 1), endpoints excluded."""
    return t.linspace(0.0, 1.0, n + 2)[1:-1]
```

## Concept: float steps drift — count, then scale

`t.arange` with a **float** step is a trap: each element is built by repeated
addition, so rounding error accumulates and the endpoint may or may not show
up. It is a sharper trap here than in NumPy, because the default float is
32-bit and has fewer digits to lose.

The robust recipe is to turn the step-question into a count-question:
figure out how many points there are, generate exact **integers**, and scale
them once — `t.arange(n_points) * step`. One multiply per element, no drift.

```python
import torch as t

drifty = t.arange(0.0, 1.0 + 0.1, 0.1)
print(drifty)
print("last value:", drifty[-1].item(), "| point count:", len(drifty))
```

The last entry is not the clean `1.0` the call asked for, and whether an
eleventh point appears at all is decided by rounding error. Counting first
removes the gamble:

```python
n = int(round(1.0 / 0.1)) + 1
exact = t.arange(n) * 0.1
print(exact)
print("point count:", len(exact))
assert len(exact) == 11
```

The count uses the exclusive-stop insight again: an inclusive range of
`step`-spaced points from 0 to `stop` has `round(stop / step) + 1` of them.

## Worked example

```python
import torch as t

# 0 to 1 inclusive, spacing 0.25. Count the points, scale integers.
n = int(round(1.0 / 0.25)) + 1        # 5 points: 0, 0.25, 0.5, 0.75, 1.0
grid = t.arange(n) * 0.25
assert grid.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
print("n =", n, "->", grid)
```

Why: `t.arange(0, 1.0 + 0.25, 0.25)` would gamble on the endpoint;
`t.arange(n) * step` is exact because the integers are exact.

## Faded practice

### q214
`solve(stop, step)`: the inclusive float range 0, step, 2·step, …, up to **and
including** `stop` (an exact multiple of `step`). (Why does the point count
need a `+ 1`?)

```python starter
import torch as t

def solve(stop, step):
    """0, step, ..., stop inclusive — exact, no float drift."""
    n = int(round(stop / step)) + _____
    return t.arange(n) * step
```

```python solution
import torch as t

def solve(stop, step):
    """0, step, ..., stop inclusive — exact, no float drift."""
    n = int(round(stop / step)) + 1
    return t.arange(n) * step
```

## Independent practice

From the drill bank: q53 (n evenly spaced values from exactly 0.0 to exactly
1.0 — which tool has that endpoints-included contract?).

## Misconceptions

- **"`t.arange(3, 8)` includes 8."** — Like Python's `range`, the stop is
  exclusive. Endpoint bugs from this are the most common range mistake; when
  a task says "inclusive", plan the `+ step` (or switch to `linspace`).
- **"Float steps in arange are fine."** — Each element is built by repeated
  float addition, so the endpoint may or may not appear and interior values
  drift. Scale exact integers (`t.arange(n) * step`) or use `linspace`.
- **"`linspace(0, 1, 5)` has step 1/5."** — It has step 1/4: five points means
  FOUR gaps. `linspace` counts points, not intervals.
- **"`t.arange(5)` gives floats."** — Integer arguments give an `int64` tensor.
  That is what you want for indexing; if you need floats, ask for them
  (`t.arange(5.0)` or `dtype=t.float32`).
