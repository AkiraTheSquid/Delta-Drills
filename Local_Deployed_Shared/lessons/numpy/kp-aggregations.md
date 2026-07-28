---
kc: numpy.aggregations
title: Whole-tensor aggregations and Python scalars
supporting: [numpy.elementwise-ufuncs]
new_syntax: []
faded: [26, 28, 64]
guided: []
independent: [62]
---

## Concept: reductions — collapsing a tensor to one number

Where an elementwise operation maps a tensor to a same-shaped tensor, an
**aggregation (reduction)** collapses a tensor down to a single number:
`x.sum()`, `x.mean()`, `x.min()`, `x.max()`, `x.std()` — the workhorses.

Called with no arguments, each of these reduces over **all** elements
regardless of shape — a 2-D tensor's `x.max()` is the max of the whole matrix.
(Reducing along just one axis is the `dim=` keyword, which gets its own KP in
the broadcasting lesson — walk before running.)

## Worked example

Task: global min and max of a matrix of sensor readings.

```python
import torch as t

readings = t.tensor([[3.5, -2.0, 7.25],
                     [0.0,  9.5, -8.75]])

# min/max with no dim argument scan the WHOLE tensor, ignoring shape.
lo, hi = readings.min(), readings.max()
assert (lo.item(), hi.item()) == (-8.75, 9.5)
```

Why: no dim argument = one value for the whole tensor, shape ignored. That's
the default to internalize before `dim=` complicates things.

## Faded practice

### q26
Global min and max of a 2-D tensor, returned as a (min, max) pair of plain
Python numbers.

```python starter
import torch as t

def solve(x):
    """Return (smallest, largest) element of the whole 2-D tensor."""
    return (x._____().item(), x._____().item())
```

```python solution
import torch as t

def solve(x):
    """Return (smallest, largest) element of the whole 2-D tensor."""
    return (x.min().item(), x.max().item())
```

## Concept: 0-dimensional tensors vs plain Python numbers

One practical wrinkle, and it bites harder here than in NumPy: reductions
return a **0-dimensional tensor**, not a number. It prints as
`tensor(1.5833)` and it still carries a dtype, a device, and possibly a
gradient. Graders, JSON encoders, and f-strings care.

When a task says "return a plain Python int/float/bool", convert explicitly:

> `float(x.mean())`, `int(x.sum())`, `bool((x > 0).any())`
> — or `x.item()`, the generic "unwrap this 0-d result".

Keep tensors *inside* your computation; convert exactly at the boundary
where a plain Python value is required. Unwrapping early is how you
accidentally break the autograd chain in real model code.

## Worked example

```python
import torch as t

readings = t.tensor([[3.5, -2.0, 7.25],
                     [0.0,  9.5, -8.75]])

# mean returns a 0-d TENSOR. Usually fine, but when the contract says
# "a single float scalar", unwrap it explicitly.
raw = readings.mean()
assert raw.ndim == 0 and isinstance(raw, t.Tensor)

avg = float(raw)
assert isinstance(avg, float)
assert abs(avg - 1.5833333) < 1e-5
```

Why: `float(...)` at the boundary — the computation stays in torch, only the
returned value is unwrapped.

## Faded practice

### q28
The arithmetic mean of a vector, as a plain Python float.

```python starter
import torch as t

def solve(x):
    """Mean of x as a plain Python float."""
    return _____(x.mean())
```

```python solution
import torch as t

def solve(x):
    """Mean of x as a plain Python float."""
    return float(x.mean())
```

## Concept: whole-tensor yes/no verdicts

Yes/no questions about tensors have two standard shapes:

- **Comparison, then reduce.** A comparison builds a boolean tensor (previous
  KP); `x.any()` (is at least one entry True?) or `x.all()` (are they all
  True?) collapses it to one answer. `(x > 0).all()` asks "is everything
  positive?".
- **Whole-tensor equality.** `t.equal(a, b)` is an exact match of shape
  and values; `t.allclose(a, b)` is equality within floating-point tolerance
  — the right check after float arithmetic.

`a == b` alone is NOT a verdict — it's elementwise and yields a boolean
tensor (and `if` on it raises an error).

## Worked example

```python
import torch as t

readings = t.tensor([[3.5, -2.0, 7.25],
                     [0.0,  9.5, -8.75]])

# Boolean pipeline: comparison (elementwise) then reduction (any).
# Read it aloud: "readings less than zero — any?"
has_negative = bool((readings < 0).any())
assert has_negative is True

# Float-safe equality: after arithmetic, prefer allclose. Ten 0.1s summed
# in float32 land just past 1.0.
a = t.full((10,), 0.1).sum()
b = t.tensor(1.0)
assert not t.equal(a, b)         # bitwise-exact? no — accumulated float error
assert t.allclose(a, b)          # equal within tolerance? yes
```

Why: exact equality is for ints/bools and provenance checks; `allclose` is
for anything that went through float arithmetic — and float32 has fewer
digits to spare than the float64 you may be used to.

## Faded practice

### q64
Tolerant closeness AND exact equality of two tensors, as two plain bools.

```python starter
import torch as t

def solve(a, b):
    """(close within float tolerance?, exactly equal?) as plain bools."""
    return (bool(t._____(a, b)), bool(t._____(a, b)))
```

```python solution
import torch as t

def solve(a, b):
    """(close within float tolerance?, exactly equal?) as plain bools."""
    return (bool(t.allclose(a, b)), bool(t.equal(a, b)))
```

## Independent practice

From the drill bank: q62 (sum of an integer tensor as a plain Python int —
reduction plus the scalar boundary in one task).

## Misconceptions

- **"`x.max()` on a matrix gives per-row maxima."** — With no arguments it
  reduces over everything: one value for the whole tensor. Per-row/column
  reductions need `dim=`, covered in the broadcasting lesson.
- **"Reductions return normal Python numbers."** — They return 0-dimensional
  tensors. Mostly interchangeable in arithmetic, but "return a plain
  int/float" contracts require `int(...)`/`float(...)`/`.item()`.
- **"`==` tells me whether two tensors are equal."** — `a == b` is ELEMENTWISE,
  yielding a boolean tensor (and `if` on it raises an error). Whole-tensor
  verdicts are `t.equal` (exact) or `t.allclose` (float-tolerant).
- **"`t.array_equal` is the exact check."** — That's the NumPy name; there is
  no such function here. It's `t.equal`.
