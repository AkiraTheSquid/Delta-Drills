---
kc: numpy.aggregations
title: Whole-array aggregations and Python scalars
supporting: [numpy.elementwise-ufuncs]
new_syntax: []
faded: [26]
guided: [28]
independent: [62, 64]
---

## Concept

Where a ufunc maps an array to a same-shaped array, an **aggregation
(reduction)** collapses an array down to a single number:

- `x.sum()`, `x.mean()`, `x.min()`, `x.max()`, `x.std()` — the workhorses.
- Boolean reducers: `x.any()` (is at least one entry True?) and `x.all()`
  (are they all True?). These pair naturally with comparisons:
  `(x > 0).all()` asks "is everything positive?".
- Whole-array comparisons: `np.array_equal(a, b)` (exact match of shape and
  values) and `np.allclose(a, b)` (equal within floating-point tolerance —
  the right check after float arithmetic).

Called with no arguments, each of these reduces over **all** elements
regardless of shape — a 2-D array's `x.max()` is the max of the whole matrix.
(Reducing along just one axis is the `axis=` keyword, which gets its own KP in
the broadcasting lesson — walk before running.)

One practical wrinkle: reductions return **NumPy scalar types**
(`np.float64`, `np.int64`, `np.bool_`), which print like Python numbers but
aren't quite them. Graders, JSON encoders, and `if` statements occasionally
care. When a task says "return a plain Python int/float/bool", convert
explicitly:

> `float(x.mean())`, `int(x.sum())`, `bool((x > 0).any())`
> — or `x.item()` for the generic "unwrap this 0-d result".

## Worked example

Task: summarize a matrix of sensor readings — global min and max, the mean as
a plain float, and whether any reading was negative.

```python
import numpy as np

readings = np.array([[3.5, -2.0, 7.25],
                     [0.0,  9.5, -8.75]])

# min/max with no axis argument scan the WHOLE array, ignoring shape.
lo, hi = readings.min(), readings.max()
assert (lo, hi) == (-8.75, 9.5)

# mean returns np.float64 — a NumPy scalar. Usually fine, but when the
# contract says "a single float scalar", unwrap it explicitly.
avg = float(readings.mean())
assert isinstance(avg, float)
assert np.isclose(avg, 1.5833333333333333)

# Boolean pipeline: comparison (elementwise) then reduction (any).
# Read it aloud: "readings less than zero — any?"
has_negative = bool((readings < 0).any())
assert has_negative is True

# Float-safe equality between arrays: after arithmetic, prefer allclose.
a = np.array([0.1, 0.2]) + np.array([0.2, 0.1])
b = np.array([0.3, 0.3])
assert not np.array_equal(a, b)      # bitwise-exact? no — float error
assert np.allclose(a, b)             # equal within tolerance? yes
```

Why each step:

1. The comparison-then-reduce pattern (`(x < 0).any()`) is the standard way to
   ask yes/no questions about arrays. The comparison builds a boolean array
   (previous KP); the reduction collapses it to one answer.
2. `float(...)` / `bool(...)` at the boundary: keep NumPy types inside your
   computation, convert exactly where a plain Python value is required.
3. The `array_equal` vs `allclose` pair is worth one deliberate look: exact
   equality is for ints/bools and provenance checks; `allclose` is for
   anything that went through float arithmetic.

## Faded practice

### q26
Global min and max of a 2-D array, returned as a (min, max) pair.

```python starter
import numpy as np

def solve(x):
    """Return (smallest, largest) element of the whole 2-D array."""
    return (x._____(), x._____())
```

```python solution
import numpy as np

def solve(x):
    """Return (smallest, largest) element of the whole 2-D array."""
    return (x.min(), x.max())
```

## Guided practice

### q28
1. The arithmetic mean of a vector is one method call.
2. The task says "a single float scalar" — what type does `.mean()` actually
   return, and does that satisfy a strict grader?
3. Wrap the result in `float(...)` to hand back a plain Python float.

## Independent practice

From the drill bank: q62 (sum of an integer array as a plain Python int),
q64 (exact equality AND tolerant closeness of two arrays, as two plain bools).

## Misconceptions

- **"`x.max()` on a matrix gives per-row maxima."** — With no arguments it
  reduces over everything: one scalar for the whole array. Per-row/column
  reductions need `axis=`, covered in the broadcasting lesson.
- **"Reductions return normal Python numbers."** — They return NumPy scalars
  (`np.float64` etc.). Mostly interchangeable, but "return a plain int/float"
  contracts require `int(...)`/`float(...)`/`.item()`.
- **"`==` tells me whether two arrays are equal."** — `a == b` is ELEMENTWISE,
  yielding a boolean array (and `if` on it raises an error). Whole-array
  verdicts are `np.array_equal` (exact) or `np.allclose` (float-tolerant).
