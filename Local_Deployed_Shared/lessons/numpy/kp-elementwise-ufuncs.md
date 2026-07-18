---
kc: numpy.elementwise-ufuncs
title: Elementwise math (ufuncs)
supporting: [numpy.ndarray-model]
new_syntax: []
faded: [49]
guided: [67]
independent: [43, 192]
---

## Concept

The core promise of NumPy: **write the formula once, and it applies to every
element** — no loop. Operations with this behavior are called *universal
functions* (ufuncs), and they come in three flavors you'll combine constantly:

1. **Operators.** `+ - * / ** %` between an array and a scalar, or between two
   same-shaped arrays, work element by element: `z * 2` doubles everything;
   `a * b` multiplies corresponding entries (NOT matrix multiplication —
   that's `@`).
2. **Named math functions.** `np.sqrt`, `np.abs`, `np.exp`, `np.log`,
   `np.sin`, … — each maps over the whole array. Rounding is a family:
   `np.round` (nearest), `np.floor` (largest integer ≤ x — so −0.3 → −1.0),
   `np.ceil`, `np.trunc` (toward zero).
3. **Elementwise choosers.** `np.maximum(a, b)` / `np.minimum(a, b)` pick the
   larger/smaller *at each position* (contrast with `a.max()`, which reduces
   the whole array to one number — different KP). `z.clip(min=lo, max=hi)`
   limits values to a range; `clip(min=0)` is exactly ReLU.

The general procedure for any "transform each entry" task:

> Express the rule for ONE element as a formula, then write that formula with
> the whole array in place of the element.

"Replace each x by x² − 1" → `z**2 - 1`. "Floor of each entry" →
`np.floor(z)`. If you find yourself writing `for i in range(len(z))`, stop —
there is almost always a ufunc spelling, and it is both shorter and orders of
magnitude faster (the loop happens in compiled code).

All these return **new arrays** and leave the input untouched — which is what
"do not modify the input" tasks expect.

## Worked example

Task: given exam scores, apply a curve — add 5 points, cap at 100, and floor
the result to whole points.

```python
import numpy as np

scores = np.array([71.5, 88.25, 97.0, 99.5])

# One formula, applied to every element at once:
# 1. +5 broadcasts the scalar across the array,
# 2. clip caps anything above 100,
# 3. floor drops the fractional part (largest integer <= x).
curved = np.floor((scores + 5).clip(max=100.0))

assert curved.tolist() == [76.0, 93.0, 100.0, 100.0]

# The input is untouched — every step above built a new array.
assert scores.tolist() == [71.5, 88.25, 97.0, 99.5]

# Elementwise chooser between TWO arrays: keep the larger at each slot.
a = np.array([1.0, 5.0, 2.0])
b = np.array([3.0, 4.0, 2.5])
assert np.maximum(a, b).tolist() == [3.0, 5.0, 2.5]
```

Why each step:

1. The pipeline reads exactly like the per-element rule: add, cap, floor.
   Composing ufuncs left-to-right is normal style — each stage maps over the
   whole array.
2. `clip` is the idiomatic "cap/limit" tool; spelling it as
   `np.minimum(x, 100)` is equivalent, and `clip(min=0)` ==
   `np.maximum(x, 0)`. Recognizing these equivalences helps you read others'
   code.
3. `np.maximum` (two arrays → same-shape array) vs `a.max()` (one array → one
   scalar) is a naming trap worth noticing NOW; the reduction version is the
   next KP.

## Faded practice

### q49
Floor every entry (note what floor does to negatives).

```python starter
import numpy as np

def solve(z):
    """Replace each entry by the largest integer value <= it."""
    return np._____(z)
```

```python solution
import numpy as np

def solve(z):
    """Replace each entry by the largest integer value <= it."""
    return np.floor(z)
```

## Guided practice

### q67
1. Every negative entry becomes 0.0, non-negatives pass through — state that
   as a per-element rule first.
2. That rule is "limit from below at 0", which is one method with one keyword
   argument (or an elementwise chooser against the scalar 0).
3. `z.clip(min=0)` or `np.maximum(z, 0.0)` — both return a new array, which is
   what "do not modify the input" needs.

## Independent practice

From the drill bank: q43 (elementwise larger of two arrays), q192 (elementwise
cube).

## Misconceptions

- **"`a * b` on two matrices is matrix multiplication."** — It is elementwise.
  Matrix product is `a @ b`. This distinction matters enough that it gets its
  own KP later.
- **"`np.maximum` and `np.max` are the same."** — `np.maximum(a, b)` compares
  two arrays position-by-position (returns an array); `np.max(a)` reduces one
  array to its single largest value (returns a scalar).
- **"floor and truncate are the same."** — Only for positives. For negatives,
  floor moves AWAY from zero (−0.3 → −1.0) while trunc/astype(int) move toward
  it (−0.3 → 0). Read the task's example values to see which is being asked.
