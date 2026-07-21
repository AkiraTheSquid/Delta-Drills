---
kc: numpy.where-select
title: Conditional values — np.where and where= arguments
supporting: [numpy.boolean-masking]
new_syntax: []
faded: [94, 100]
guided: []
independent: []
---

## Concept: Choose values with np.where

Masked assignment overwrites part of an array. `np.where` instead builds a
**new array** by choosing a value at every position:

> **`np.where(condition, value_if_true, value_if_false)`**

Read it as vectorized if/else. Where the condition is `True`, NumPy takes the
second argument. Everywhere else, it takes the third. Values may be scalars or
arrays; normal broadcasting rules apply.

For example, `np.where(z < 0, 0.0, z)` replaces negative values with zero while
leaving `z` unchanged.

## Watch out

`np.where(condition)` with only one argument does something different: it
returns indices of `True` positions. Choosing values always needs three
arguments.

## Worked example: Replace negative readings

Task: build a new array equal to `z`, except negative readings become `0.0`.

```python
import numpy as np

z = np.array([-2.0, 0.5, 3.0, -1.0])

relu = np.where(z < 0, 0.0, z)

assert relu.tolist() == [0.0, 0.5, 3.0, 0.0]
assert z.tolist() == [-2.0, 0.5, 3.0, -1.0]
print(relu)
```

Why: at `z[0]`, `z[0] < 0` is `True`, so NumPy chooses `0.0`. At
`z[1]`, the condition is `False`, so NumPy chooses `z[1]`. Same choice runs at
every position. `np.where` returns a new array, so no `.copy()` is needed.

## Faded practice

### q94
Return `z` with `-1.0` wherever a different array, `y`, exceeds a threshold.

```python starter
import numpy as np

def solve(z, y, threshold):
    """Return z with -1.0 wherever y exceeds the threshold."""
    return np.where(_____, _____, _____)
```

```python solution
import numpy as np

def solve(z, y, threshold):
    """Return z with -1.0 wherever y exceeds the threshold."""
    return np.where(y > threshold, -1.0, z)
```

## Concept: Skip unsafe ufunc positions with where=

`np.where` chooses between already-computed values. A ufunc's `where=` keyword
does something different: it controls **where the operation runs**.

For safe division, combine it with `out=`:

```python no-run
np.divide(a, b, out=np.zeros_like(a), where=b != 0)
```

`where=b != 0` runs division only where the divisor is nonzero. `out=` supplies
both the result array and the values retained at skipped positions. Starting
with zeros therefore makes every skipped result `0.0`.

## Watch out

`np.where(b != 0, a / b, 0.0)` does **not** prevent division by zero. Python
evaluates `a / b` first; `np.where` selects afterward. Ufunc `where=` skips the
unsafe operation itself.

## Worked example: Divide safely

Task: compute `a / b`, producing `0.0` where `b` is zero without division
warnings.

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 0.0, 4.0])

ratio = np.divide(a, b, out=np.zeros_like(a), where=b != 0)

assert ratio.tolist() == [0.5, 0.0, 0.75]
print(ratio)
```

Why: division runs at positions 0 and 2. Position 1 is skipped, so its initial
`out` value—`0.0`—remains. No invalid division occurs.

## Faded practice

### q100
Compute elementwise `a / b`, but return exactly `0.0` where `b` is zero.

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
