---
kc: numpy.where-select
title: Conditional values — np.where and where= arguments
supporting: [numpy.boolean-masking]
new_syntax: []
faded: [100]
guided: []
independent: [94]
---

## Concept

Masked assignment (previous KP) *overwrites part of an array*. Its functional
sibling builds a **new array by choosing, position by position, between two
sources**:

> **`np.where(condition, value_if_true, value_if_false)`**

Read it as a vectorized if/else: for each position, take from the second
argument where the condition is True, from the third where it is False.
Either argument can be a scalar or an array (broadcast as needed):

- `np.where(z < 0, 0.0, z)` — ReLU, spelled as a choice.
- `np.where(y > t, -1.0, z)` — replace z's entries by −1 wherever a
  *different* array exceeds a threshold. (Notice: condition on `y`, values
  from `z` — the three arguments don't have to involve the same array.)

Because it returns a new array, `np.where` is the natural fit for "return a
new array equal to X except…" tasks — no `.copy()` choreography needed.

The same idea appears as a **keyword argument on ufuncs**: `where=` restricts
*where the operation happens at all*, and `out=` supplies the array that fills
the untouched positions. The canonical use is division that must not divide
by zero:

```python no-run
np.divide(a, b, out=np.zeros_like(a), where=b != 0)
```

At positions where `b == 0`, no division is performed (so no warning, no
inf/NaN) and the `out` array's value (0.0) shows through. Unlike
`np.where(b != 0, a / b, 0.0)` — which computes `a / b` *everywhere* first
and only then selects, warning included — the `where=` form genuinely skips
the bad positions.

## Worked example

Task: rewrite negative readings as 0 (choice form), then compute a safe
elementwise ratio a/b that yields 0.0 where b is zero, with no warnings.

```python
import numpy as np

z = np.array([-2.0, 0.5, 3.0, -1.0])

# Vectorized if/else: condition, value-if-true, value-if-false.
relu = np.where(z < 0, 0.0, z)
assert relu.tolist() == [0.0, 0.5, 3.0, 0.0]
assert z.tolist() == [-2.0, 0.5, 3.0, -1.0]   # input untouched — new array

a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 0.0, 4.0])

# Safe division: out= provides the default values, where= limits the
# operation to positions whose divisor is nonzero. b == 0 slots are never
# divided at all — that's what keeps the warning from firing.
ratio = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
assert ratio.tolist() == [0.5, 0.0, 0.75]
```

Why each step:

1. In the `np.where` call, walk one position through the sentence: "z[0] is
   −2.0; is it < 0? yes → take 0.0." The whole array is that sentence at
   every position simultaneously.
2. `out=np.zeros_like(a)` does double duty: allocates the result AND sets the
   fill for skipped positions. `zeros_like` (not `zeros(shape)`) keeps the
   dtype aligned with `a`.
3. Choosing between the two forms: need to *select between computed values*?
   `np.where`. Need to *avoid computing* somewhere (division by zero, log of
   negative)? ufunc `where=`.

## Faded practice

### q100
Elementwise a/b, but exactly 0.0 where b is zero — and no warnings raised.

```python starter
import numpy as np

def solve(a, b):
    """a / b elementwise; 0.0 where b == 0; no divide warnings."""
    return np.divide(a, b, out=_____, where=_____)
```

```python solution
import numpy as np

def solve(a, b):
    """a / b elementwise; 0.0 where b == 0; no divide warnings."""
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)
```

## Independent practice

From the drill bank: q94 (new array equal to z except −1.0 where a DIFFERENT
array y exceeds a threshold — either `np.where` or copy + masked assignment;
notice the condition and the values come from different arrays).

## Misconceptions

- **"`np.where(cond, a, b)` short-circuits like if/else."** — Both `a` and
  `b` are fully evaluated first; `where` only selects afterwards. If
  evaluating one side is the problem (1/0, log(-1)), use the ufunc's `where=`
  keyword, which actually skips positions.
- **"One-argument `np.where(cond)` is the same thing."** — With a single
  argument it returns *indices* of True positions (like `np.nonzero`), not
  values. The choose-between-values form always takes three arguments.
- **"I need `.copy()` with np.where."** — `np.where` already builds a fresh
  array; copying is for the masked-assignment style.
