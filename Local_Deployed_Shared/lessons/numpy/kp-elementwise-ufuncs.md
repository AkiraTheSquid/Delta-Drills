---
kc: numpy.elementwise-ufuncs
title: Elementwise math (ufuncs)
supporting: [numpy.ndarray-model]
new_syntax: []
faded: [192, 49, 67]
guided: []
independent: [43]
---

## Concept: write the formula once — operators are elementwise

The core promise of NumPy: **write the formula once, and it applies to every
element** — no loop. Operators `+ - * / ** %` between an array and a scalar,
or between two same-shaped arrays, work element by element: `z * 2` doubles
everything; `a * b` multiplies corresponding entries (NOT matrix
multiplication — that's `@`).

The general procedure for any "transform each entry" task:

> Express the rule for ONE element as a formula, then write that formula
> with the whole array in place of the element.

"Replace each x by x² − 1" → `z**2 - 1`. If you find yourself writing
`for i in range(len(z))`, stop — the ufunc spelling is shorter and orders of
magnitude faster (the loop happens in compiled code). All of these return
**new arrays** and leave the input untouched.

## Worked example

```python
import numpy as np

z = np.array([1.0, 2.0, 3.0])

# The per-element rule "x**2 - 1", written once for the whole array:
out = z**2 - 1
assert out.tolist() == [0.0, 3.0, 8.0]

# The input is untouched — the expression built a new array.
assert z.tolist() == [1.0, 2.0, 3.0]
```

Why: the expression reads exactly like the per-element rule — that
transliteration IS the method.

## Faded practice

### q192
Elementwise cube.

```python starter
import numpy as np

def solve(x):
    """Each entry raised to the third power."""
    return x _____ 3
```

```python solution
import numpy as np

def solve(x):
    """Each entry raised to the third power."""
    return x ** 3
```

## Concept: named math functions and the rounding family

Beyond operators, named ufuncs map over the whole array: `np.sqrt`,
`np.abs`, `np.exp`, `np.log`, `np.sin`, …

Rounding is a *family*, and the members differ on negatives:

- `np.round` — nearest;
- `np.floor` — largest integer ≤ x, so −0.3 → −1.0 (away from zero);
- `np.ceil` — smallest integer ≥ x;
- `np.trunc` — toward zero, so −0.3 → 0.0 (same as `astype(int)`).

Read the task's example values to see which member is being asked for.

## Worked example

```python
import numpy as np

v = np.array([1.7, -0.3, 2.5])

assert np.floor(v).tolist() == [1.0, -1.0, 2.0]   # floor moves DOWN
assert np.trunc(v).tolist() == [1.0, -0.0, 2.0]   # trunc moves toward zero
assert np.sqrt(np.array([4.0, 9.0])).tolist() == [2.0, 3.0]
```

Why: floor vs trunc only disagree on negatives — that's exactly where tasks
(and graders) check.

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

## Concept: elementwise choosers — maximum, minimum, clip

`np.maximum(a, b)` / `np.minimum(a, b)` pick the larger/smaller *at each
position* (contrast with `a.max()`, which reduces the whole array to one
number — different KP). `z.clip(min=lo, max=hi)` limits values to a range.

These are interchangeable spellings worth recognizing in others' code:
`clip(max=100)` == `np.minimum(x, 100)`, and `clip(min=0)` ==
`np.maximum(x, 0)` — which is exactly ReLU.

## Worked example

```python
import numpy as np

# Elementwise chooser between TWO arrays: keep the larger at each slot.
a = np.array([1.0, 5.0, 2.0])
b = np.array([3.0, 4.0, 2.5])
assert np.maximum(a, b).tolist() == [3.0, 5.0, 2.5]

# Pipeline: curve exam scores — add 5, cap at 100, floor to whole points.
scores = np.array([71.5, 88.25, 97.0, 99.5])
curved = np.floor((scores + 5).clip(max=100.0))
assert curved.tolist() == [76.0, 93.0, 100.0, 100.0]
```

Why: composing ufuncs left-to-right is normal style — each stage maps over
the whole array, and the pipeline reads exactly like the per-element rule:
add, cap, floor.

## Faded practice

### q67
Negatives become 0.0, non-negatives pass through (ReLU).

```python starter
import numpy as np

def solve(z):
    """Each negative entry replaced by 0.0 (new array; z unmodified)."""
    return z.clip(_____=0)
```

```python solution
import numpy as np

def solve(z):
    """Each negative entry replaced by 0.0 (new array; z unmodified)."""
    return z.clip(min=0)
```

## Independent practice

From the drill bank: q43 (elementwise larger of two arrays).

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
