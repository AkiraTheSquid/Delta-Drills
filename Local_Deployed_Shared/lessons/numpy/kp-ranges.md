---
kc: numpy.ranges
title: Numeric ranges — arange and linspace
supporting: [numpy.constructors, numpy.slicing-views]
new_syntax: [torch.arange, torch.linspace]
faded: [229, 242, 214, 959, 960, 961, 962, 963]
guided: [524, 964, 965]
independent: [53, 966, 967, 968, 969, 970]
integrated: [971, 972, 973]
---

## Concept: t.arange — the stop is exclusive

`t.arange` counts upward and stops **before** its stop value. Predict the last value:

```python
import torch as t
values = t.arange(5)
print(values)
# Hidden checks
assert values.tolist() == [0, 1, 2, 3, 4]
```

The result ends at 4. With one argument, counting starts at zero and takes steps of one.
Give three arguments to choose the start, stop, and step:

```python
values = t.arange(0, 10, 2)
print(values)
# Hidden checks
assert values.tolist() == [0, 2, 4, 6, 8]
```

Each value is two larger than the last. The stop is still excluded: 10 does not appear.

## Worked example

How can we include 10? Move the boundary past it:

```python
import torch as t
values = t.arange(0, 11, 2)
print(values)
# Hidden checks
assert values.tolist() == [0, 2, 4, 6, 8, 10]
assert values.dtype == t.int64
```

Now 10 fits before the boundary at 11. Integer arguments produce integer values.

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


### q959
The three-argument form, with the step chosen by the caller. The stop is
still exclusive, whatever the step is.

```python starter
import torch as t


def solve(start, end, k):
    """Every k-th integer from start up to but not including end."""
    return t._____(start, end, k)
```

```python solution
import torch as t


def solve(start, end, k):
    """Every k-th integer from start up to but not including end."""
    return t.arange(start, end, k)
```

### q960
Choosing all three arguments at once: where to start, where to stop, and how
far to move. Work out the stop from how many values you want.

```python starter
import torch as t


def solve(n):
    """The first n odd numbers, in increasing order."""
    return t._____(1, 2 * n, 2)
```

```python solution
import torch as t


def solve(n):
    """The first n odd numbers, in increasing order."""
    return t.arange(1, 2 * n, 2)
```

## Concept: t.linspace — choose the number of points

`t.linspace(start, stop, steps)` places that many points between two endpoints, **including both ends**.
Predict the middle value of these five points:

```python
import torch as t
grid = t.linspace(0.0, 1.0, 5)
print(grid)
# Hidden checks
assert grid.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
```

The middle is 0.5. Five points leave **four gaps**, so each gap spans one quarter:

```python
gaps = grid[1:] - grid[:-1]
print(gaps)
# Hidden checks
assert gaps.tolist() == [0.25, 0.25, 0.25, 0.25]
```

Subtracting each point from the next reveals the spacing. This is why the denominator is `steps - 1`.

## Worked example

Predict three points from 2 to 8. There are two gaps to divide the distance:

```python
import torch as t
grid = t.linspace(2.0, 8.0, 3)
print(grid)
# Hidden checks
assert grid.tolist() == [2.0, 5.0, 8.0]
```

The distance is 6, each gap is 3. Use `linspace` when you know the **count**, `arange` when you know the **step**.

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


### q961
The plain form: two endpoints and a count, both ends kept.

```python starter
import torch as t


def solve(a, b, n):
    """n evenly spaced points from a to b, both endpoints included."""
    return t._____(a, b, n)
```

```python solution
import torch as t


def solve(a, b, n):
    """n evenly spaced points from a to b, both endpoints included."""
    return t.linspace(a, b, n)
```

### q962
n pieces need n + 1 edges. Once the edges exist, every midpoint is the
average of an edge and the one after it — a slice against a shifted slice.

```python starter
import torch as t


def solve(n):
    """The midpoints of the n equal sub-intervals of 0 to 1."""
    edges = t._____(0.0, 1.0, n + 1)
    return (edges[:-1] + edges[1:]) / 2
```

```python solution
import torch as t


def solve(n):
    """The midpoints of the n equal sub-intervals of 0 to 1."""
    edges = t.linspace(0.0, 1.0, n + 1)
    return (edges[:-1] + edges[1:]) / 2
```

## Concept: Count first, then scale

Floating-point numbers approximate most decimal fractions. If the **number of points** matters, choose that count directly instead of extending a floating stop and hoping to include the endpoint.

First, count five positions:

```python
import torch as t
positions = t.arange(5)
print(positions)
# Hidden checks
assert positions.tolist() == [0, 1, 2, 3, 4]
```

Now scale each position by the desired spacing. Predict the final value:

```python
grid = positions * 0.25
print(grid)
# Hidden checks
assert grid.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
```

Four steps of 0.25 reach 1.0. The count is controlled by the integers; other decimal spacings can still have rounding error.

## Worked example

From 0 to 2 inclusive, spacing 0.5 gives four gaps and **five points**:

```python
import torch as t
step = 0.5
n = int(round(2.0 / step)) + 1
grid = t.arange(n) * step
print(grid)
# Hidden checks
assert n == 5
assert grid.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]
```

The `+ 1` includes the starting point. This recipe assumes the interval is a whole number of steps; use `linspace` when both endpoints and the point count are given.

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


### q963
The same count-then-scale recipe with the spacing given as a RATE: the
spacing is one over it.

```python starter
import torch as t


def solve(n, hz):
    """The n sample times of a signal sampled at hz samples per second."""
    return t._____(n) * (1.0 / hz)
```

```python solution
import torch as t


def solve(n, hz):
    """The n sample times of a signal sampled at hz samples per second."""
    return t.arange(n) * (1.0 / hz)
```

## Guided practice

### q524
1. The exclusive stop does not care which way you are counting. Going down,
   "stops before the stop" means the stop has to sit one step PAST `low`.
2. A negative step reverses the direction: `t.arange(high, ?, -1)`. Ask what
   `?` makes `low` the last value actually produced.
3. `t.arange(high, low - 1, -1)` — the `- 1` is the inclusive-endpoint fix
   from q229, pointed the other way.


### q964
1. A negative step is the arange answer to counting down, and it brings the exclusive-stop problem with it. Here you are given the COUNT, not the step.
2. The endpoints-and-count tool does not require the first endpoint to be the smaller one. Hand it the two ends in the order you want them produced.
3. `t.linspace(high, low, n)`.

### q965
1. Build the range first — the question is about what it CONTAINS, and the range itself is the authority on that.
2. Its final element is the one at index minus one. Convert that single value to a Python int before returning it.
3. `int(t.arange(start, stop, step)[-1])`.

## Independent practice

From the drill bank: q53 (n evenly spaced values from exactly 0.0 to exactly
1.0 — which tool has that endpoints-included contract?).


From the drill bank: q966 (edges dividing an interval into equal bins).
From the drill bank: q967 (an integer countdown ending at one).
From the drill bank: q968 (an exact inclusive float range).
From the drill bank: q969 (evenly spaced points including zero, excluding the far end).
From the drill bank: q970 (equally spaced positions around a circle).


## Integrated practice

### q971
Return an inclusive range and its length.

### q972
Return equal-bin edges and centres.

### q973
Return an evenly spaced grid and its successive gaps.

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
