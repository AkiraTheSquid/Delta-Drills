---
kc: numpy.ranges
title: Numeric ranges — arange and linspace
supporting: [numpy.constructors, numpy.slicing-views]
new_syntax: []
faded: [229]
guided: [214]
independent: [53, 242]
---

## Concept

Sequences of evenly spaced numbers come up constantly — integer indices, time
steps, plot axes, probability breakpoints. NumPy gives you two generators, and
choosing between them comes down to one question:

> **Do you know the step size, or the number of points?**

- **`np.arange(start, stop, step)`** — you know the **step**. Counts from
  `start` in increments of `step`, and — exactly like Python's `range` —
  **excludes `stop`**. `np.arange(3, 8)` → `[3, 4, 5, 6, 7]`.
- **`np.linspace(start, stop, num)`** — you know the **number of points**.
  Produces exactly `num` evenly spaced values, and **includes both
  endpoints**. `np.linspace(0.0, 1.0, 5)` → `[0.0, 0.25, 0.5, 0.75, 1.0]`.

The exclusive/inclusive difference is the whole trick. Most range bugs are
off-by-one at the endpoint:

- To make `arange` include its endpoint with integer steps, extend the stop:
  `np.arange(start, end + 1)`.
- With **float** steps, prefer not to use `arange` at all: accumulated
  floating-point error makes the last point unreliable (sometimes `stop`
  sneaks in, sometimes the point before it is off by 1e-16). The robust
  pattern is to generate exact **integers** and scale them:
  `np.arange(n_points) * step` — or use `linspace`, which is float-exact at
  both endpoints by construction.

`linspace` also composes well with slicing when you need *interior* points:
generate the inclusive grid, then cut the ends off with `[1:-1]`.

## Worked example

Task: build (a) all integers from -2 through 3 inclusive, and (b) the grid
0.0, 0.5, 1.0, ..., 10.0 — inclusive of 10.0 — without float drift.

```python
import numpy as np

# (a) Integer range, inclusive of the endpoint.
# arange excludes stop, so push stop one past the end we want.
ints = np.arange(-2, 3 + 1)
assert ints.tolist() == [-2, -1, 0, 1, 2, 3]

# (b) Float grid with step 0.5 from 0 to 10 inclusive.
# DON'T: np.arange(0, 10 + 0.5, 0.5) — float steps make the endpoint a gamble.
# DO: count how many points there are, generate exact integers, scale once.
stop, step = 10.0, 0.5
n = int(round(stop / step)) + 1     # 21 points: 0.0, 0.5, ..., 10.0
grid = np.arange(n) * step
assert len(grid) == 21
assert grid[0] == 0.0 and grid[-1] == 10.0

# Equivalent linspace formulation: we know the point count, so:
grid2 = np.linspace(0.0, 10.0, 21)
assert np.allclose(grid, grid2)
```

Why each step:

1. In (a) the `+ 1` is doing the real work — spelling it `3 + 1` instead of
   `4` documents *why* the stop is what it is.
2. In (b), `int(round(stop / step)) + 1` converts a step-question into a
   count-question. The multiplication `np.arange(n) * step` performs ONE float
   operation per element instead of accumulating additions, so the endpoint is
   exact.
3. When both endpoints must land exactly, `linspace` is the most direct tool —
   its contract is "exactly `num` points, endpoints included".

## Faded practice

### q229
Every integer from start to end, INCLUDING both endpoints.

```python starter
import numpy as np

def solve(start, end):
    """Return the integers start..end inclusive, in order."""
    return np.arange(start, _____)
```

```python solution
import numpy as np

def solve(start, end):
    """Return the integers start..end inclusive, in order."""
    return np.arange(start, end + 1)
```

## Guided practice

### q214
1. The sequence is 0, step, 2·step, …, stop — you're asked for an INCLUSIVE
   float range, which is exactly the fragile case for `np.arange`.
2. Convert it to a count: how many multiples of `step` lie in [0, stop]?
   (`stop` is promised to be an exact multiple of `step`.)
3. Generate those integers with `np.arange(n)` and multiply by `step` once —
   integer counting is exact, so the endpoint is too.

## Independent practice

From the drill bank: q53 (n evenly spaced values from exactly 0.0 to exactly
1.0 — which tool has that contract?), q242 (the n interior breakpoints of
(0, 1) — generate the inclusive grid, then slice the endpoints away).

## Misconceptions

- **"`np.arange(3, 8)` includes 8."** — Like Python's `range`, the stop is
  exclusive. Endpoint bugs from this are the most common range mistake; when
  a task says "inclusive", plan the `+ step` (or switch to `linspace`).
- **"Float steps in arange are fine."** — Each element is built by repeated
  float addition, so the endpoint may or may not appear and interior values
  drift. Scale exact integers (`np.arange(n) * step`) or use `linspace`.
- **"`linspace(0, 1, 5)` has step 1/5."** — It has step 1/4: five points means
  FOUR gaps. `linspace` counts points, not intervals.
