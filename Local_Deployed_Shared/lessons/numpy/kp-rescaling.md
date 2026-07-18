---
kc: numpy.rescaling
title: Rescaling — min-max, unit norm, probability rows
supporting: [numpy.axis-reductions, numpy.where-select]
new_syntax: []
faded: [80]
guided: [162]
independent: [6, 97, 153, 10]
---

## Concept

Centering shifts data; **rescaling** maps it into a target range or size.
Three canonical scalings cover nearly every drill, each defined by what the
result must satisfy:

- **Min-max to [0, 1]**: `(z - z.min()) / (z.max() - z.min())` — smallest
  entry becomes exactly 0, largest exactly 1, everything else keeps its
  relative position. ("Normalize to [a, b]" is this, then `* (b - a) + a`.)
- **Unit Euclidean length**: `v / np.linalg.norm(v)` — same direction, length
  1. Scaling to length L is `v * (L / norm)`. `np.linalg.norm` is the
  square-root-of-sum-of-squares; on matrices it takes `axis=` (and
  `keepdims=`) exactly like a reduction.
- **Probability distribution**: `z / z.sum()` — non-negative entries summing
  to 1. Per-row: divide by `z.sum(axis=1, keepdims=True)`.

All three are the same template as centering — *statistic, then divide* —
so per-row/per-column versions come from `axis=` + `keepdims=True`, no new
machinery.

The recurring production concern: **the divisor can be zero** (all-zero row,
constant array). Bare division emits warnings and produces NaN/Inf; drills
phrased "zero rows must remain zeros, with no warnings" want the safe-divide
composition:

```python no-run
sums = z.sum(axis=1, keepdims=True)
np.divide(z, sums, out=np.zeros_like(z), where=sums != 0)
```

— the `where=` skips the bad rows, the `out=` zeros fill them.

## Worked example

Task: min-max a vector to [0, 1]; make a unit vector; convert score rows to
probability rows, zero rows staying zero.

```python
import numpy as np

# 1. Min-max: affine map sending min -> 0 and max -> 1.
z = np.array([2.0, 4.0, 6.0])
mm = (z - z.min()) / (z.max() - z.min())
assert mm.tolist() == [0.0, 0.5, 1.0]

# 2. Unit length: divide by the Euclidean norm. Direction is preserved —
# the entries keep their ratios (3:4 here).
v = np.array([3.0, 4.0])
unit = v / np.linalg.norm(v)
assert np.allclose(unit, [0.6, 0.8])
assert np.isclose(np.linalg.norm(unit), 1.0)     # the defining postcondition

# 3. Probability rows with a zero row — the safe-divide composition.
scores = np.array([[1.0, 3.0],
                   [0.0, 0.0]])
sums = scores.sum(axis=1, keepdims=True)          # (2, 1): [[4.], [0.]]
probs = np.divide(scores, sums,
                  out=np.zeros_like(scores), where=sums != 0)
assert probs.tolist() == [[0.25, 0.75],
                          [0.0, 0.0]]
assert np.isclose(probs[0].sum(), 1.0)
```

Why each step:

1. Each scaling is verified against its own DEFINITION (endpoints 0 and 1;
   norm 1; row sums 1) — write these postconditions as asserts while
   practicing and axis/formula errors have nowhere to hide.
2. In the min-max formula, both statistics come from the SAME array before
   any modification — compute them first (or inline), never after a partial
   in-place update.
3. Step 3 is three prior KPs snapping together: axis reduction (row sums),
   keepdims (alignment), where/out (safety). Recognizing tasks as
   compositions of known moves — rather than new tricks — is the skill this
   lesson is building.

## Faded practice

### q80
Linear rescale so min → 0.0 and max → 1.0.

```python starter
import numpy as np

def solve(z):
    """Min-max rescale to [0, 1]."""
    return (z - _____) / (_____ - _____)
```

```python solution
import numpy as np

def solve(z):
    """Min-max rescale to [0, 1]."""
    return (z - z.min()) / (z.max() - z.min())
```

## Guided practice

### q162
1. Each row becomes a probability distribution — divide by what, per row,
   kept in which shape?
2. Zero rows must SURVIVE as zeros with no warnings — that's the
   `where=`/`out=` division, not an if-statement.
3. `np.divide(z, sums, out=np.zeros_like(z), where=sums != 0)` with
   `sums = z.sum(axis=1, keepdims=True)`.

## Independent practice

From the drill bank: q6 (unit vector), q97 (rescale to a target length L —
one multiplicative constant), q153 (unit-length ROWS with zero rows safe —
norm takes axis/keepdims), q10 (global min-max on an array of ANY shape —
does the formula even care about shape?).

## Misconceptions

- **"Normalize means divide by the max."** — `z / z.max()` sends the max to 1
  but the min to min/max, not 0. True min-max subtracts the min first. Read
  which endpoints the task pins down.
- **"norm(v) is the sum of absolute values."** — Default is the EUCLIDEAN
  (L2) norm: √Σx². The L1 norm is `np.linalg.norm(v, 1)` — and a
  "probability" scaling divides by the plain SUM, which for non-negative
  data equals L1.
- **"Guard zero divisors with `if z.sum() == 0`."** — Per-row that's a loop
  in disguise. The vectorized guard is `np.divide(..., out=..., where=...)`,
  which handles mixed zero/nonzero rows in one call.
