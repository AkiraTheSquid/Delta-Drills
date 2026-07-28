---
kc: numpy.where-select
title: Conditional values — t.where and where= arguments
supporting: [numpy.boolean-masking]
new_syntax: []
faded: [94, 100]
guided: []
independent: []
---

## Concept: Choose values with t.where

Masked assignment overwrites part of an array. `t.where` instead builds a
**new array** by choosing a value at every position:

> **`t.where(condition, value_if_true, value_if_false)`**

Read it as vectorized if/else. Where the condition is `True`, PyTorch takes the
second argument. Everywhere else, it takes the third. Values may be scalars or
arrays; normal broadcasting rules apply.

For example, `t.where(z < 0, 0.0, z)` replaces negative values with zero while
leaving `z` unchanged.

## Watch out

`t.where(condition)` with only one argument does something different: it
returns indices of `True` positions. Choosing values always needs three
arguments.

## Worked example: Replace negative readings

Task: build a new array equal to `z`, except negative readings become `0.0`.

```python
import torch as t

z = t.tensor([-2.0, 0.5, 3.0, -1.0])

relu = t.where(z < 0, 0.0, z)

assert relu.tolist() == [0.0, 0.5, 3.0, 0.0]
assert z.tolist() == [-2.0, 0.5, 3.0, -1.0]
print(relu)
```

Why: at `z[0]`, `z[0] < 0` is `True`, so PyTorch chooses `0.0`. At
`z[1]`, the condition is `False`, so PyTorch chooses `z[1]`. Same choice runs at
every position. `t.where` returns a new array, so no `.copy()` is needed.

## Faded practice

### q94
Return `z` with `-1.0` wherever a different array, `y`, exceeds a threshold.

```python starter
import torch as t

def solve(z, y, threshold):
    """Return z with -1.0 wherever y exceeds the threshold."""
    return t.where(_____, _____, _____)
```

```python solution
import torch as t

def solve(z, y, threshold):
    """Return z with -1.0 wherever y exceeds the threshold."""
    return t.where(y > threshold, -1.0, z)
```

## Concept: Skip unsafe positions with masked assignment

`t.where` chooses between already-computed values. Sometimes you need the
unsafe value never to be COMPUTED — division by zero being the standard case.

PyTorch has no `where=` keyword on its operators (NumPy's ufuncs do). The
torch spelling is a zeros canvas plus masked assignment, which runs the
operation only at the selected positions:

```python no-run
out = t.zeros_like(a)
nz = b != 0
out[nz] = a[nz] / b[nz]
```

`a[nz] / b[nz]` divides only the safe entries — the zero divisors are never
handed to the division at all. The canvas supplies the value left at every
skipped position, so starting from `zeros_like` makes each of them `0.0`.

## Watch out

`t.where(b != 0, a / b, 0.0)` does **not** prevent division by zero. Python
evaluates `a / b` first — producing `inf` or `nan` — and `t.where` only
selects afterward. Masked assignment skips the unsafe operation itself.

## Worked example: Divide safely

Task: compute `a / b`, producing `0.0` where `b` is zero without division
warnings.

```python
import torch as t

a = t.tensor([1.0, 2.0, 3.0])
b = t.tensor([2.0, 0.0, 4.0])

ratio = t.zeros_like(a)
nz = b != 0
ratio[nz] = a[nz] / b[nz]

assert ratio.tolist() == [0.5, 0.0, 0.75]
print(ratio)
```

Why: division runs at positions 0 and 2 only — `a[nz]` and `b[nz]` are
length-2 tensors that never contain the zero divisor. Position 1 keeps the
canvas value `0.0`. No invalid division occurs.

## Faded practice

### q100
Compute elementwise `a / b`, but return exactly `0.0` where `b` is zero.

```python starter
import torch as t

def solve(a, b):
    """a / b elementwise; 0.0 where b == 0; no division by zero."""
    out = t.zeros_like(a)
    nz = _____
    out[nz] = a[nz] / b[nz]
    return out
```

```python solution
import torch as t

def solve(a, b):
    """a / b elementwise; 0.0 where b == 0; no division by zero."""
    out = t.zeros_like(a)
    nz = b != 0
    out[nz] = a[nz] / b[nz]
    return out
```
