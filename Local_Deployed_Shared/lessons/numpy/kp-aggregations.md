---
kc: numpy.aggregations
title: Whole-tensor aggregations and Python scalars
supporting: [numpy.elementwise-ufuncs]
new_syntax: [Tensor.any, Tensor.max, Tensor.mean, Tensor.min, Tensor.sum, torch.Tensor, torch.allclose, syntax.compare]
faded: [26, 28, 64, 984, 985, 986]
guided: [495, 496]
independent: [62, 497, 498, 987, 988, 989, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324]
integrated: [990, 991, 992, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332]
---

## Concept: reductions — collapsing a tensor to one number

Where an elementwise operation maps a tensor to a same-shaped tensor, an
**aggregation (reduction)** collapses a tensor down to a single number:
`x.sum()`, `x.mean()`, `x.min()`, `x.max()`, `x.std()` — the workhorses.

Called with no arguments, each of these reduces over **all** elements
regardless of shape — a 2-D tensor's `x.max()` is the max of the whole matrix.
(Reducing along just one axis is the `dim=` keyword, which gets its own KP in
the broadcasting lesson — walk before running.)

```python
import torch as t

grid = t.tensor([[3.0, 8.0, 1.0],
                 [6.0, 2.0, 9.0]])
print("sum ", grid.sum())
print("mean", grid.mean())
print("min ", grid.min())
print("max ", grid.max())
# Hidden checks
assert _delta_output == 'sum  tensor(29.)\nmean tensor(4.8333)\nmin  tensor(1.)\nmax  tensor(9.)\n'
```

Six values in, one value out, every time — and notice the shape never came
into it:

```python
flat = grid.reshape(6)
print(grid.shape, "and", flat.shape, "→ same answers")
# Hidden checks
assert t.equal(flat.max(), grid.max())
assert t.equal(flat.sum(), grid.sum())
```

## Worked example

Task: global min and max of a matrix of sensor readings.

```python
import torch as t

readings = t.tensor([[3.5, -2.0, 7.25],
                     [0.0,  9.5, -8.75]])

# min/max with no dim argument scan the WHOLE tensor, ignoring shape.
lo, hi = readings.min(), readings.max()
print("min", lo.item(), "| max", hi.item())
# Hidden checks
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


### q984
Two whole-tensor reductions in one answer, each crossing back to a plain
Python number.

```python starter
import torch as t


def solve(x):
    """(total, average) of the whole tensor as plain Python numbers."""
    return (x._____().item(), x._____().item())
```

```python solution
import torch as t


def solve(x):
    """(total, average) of the whole tensor as plain Python numbers."""
    return (x.sum().item(), x.mean().item())
```

## Concept: 0-dimensional tensors vs plain Python numbers

One practical wrinkle, and it bites harder here than in NumPy: reductions
return a **0-dimensional tensor**, not a number. It prints as
`tensor(1.5833)` and it still carries a dtype, a device, and possibly a
gradient. Graders, JSON encoders, and f-strings care.

When a task says "return a plain Python int/float/bool", convert explicitly:

> `float(x.mean())`, `int(x.sum())`, `bool((x > 0).any())`
> — or `x.item()`, the generic "unwrap this 0-d result".

```python
import torch as t

grid = t.tensor([[3.0, 8.0, 1.0],
                 [6.0, 2.0, 9.0]])
raw = grid.mean()
print(raw, "| ndim", raw.ndim, "| type", type(raw).__name__)
print(float(raw), "| type", type(float(raw)).__name__)
# Hidden checks
assert _delta_output == 'tensor(4.8333) | ndim 0 | type Tensor\n4.833333492279053 | type float\n'
```

The first line still says `tensor(...)`. That is the whole distinction:

```python
print("raw.ndim", raw.ndim, "-> still a tensor:", isinstance(raw, t.Tensor))
print("int(grid.sum())", int(grid.sum()), type(int(grid.sum())).__name__)
print("bool((grid > 0).all())", bool((grid > 0).all()),
      type(bool((grid > 0).all())).__name__)
# Hidden checks
assert raw.ndim == 0 and isinstance(raw, t.Tensor)
assert isinstance(raw.item(), float)
assert int(grid.sum()) == 29
assert bool((grid > 0).all()) is True
```

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

avg = float(raw)
print("as a tensor:", raw, "| as a float:", avg)
# Hidden checks
assert raw.ndim == 0 and isinstance(raw, t.Tensor)
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


### q985
The same boundary as the mean drill, crossed after a different reduction.

```python starter
import torch as t


def solve(x):
    """The largest entry of x as a plain Python float."""
    return _____(x.max())
```

```python solution
import torch as t


def solve(x):
    """The largest entry of x as a plain Python float."""
    return float(x.max())
```

## Concept: whole-tensor yes/no verdicts

Yes/no questions about tensors have two standard shapes:

- **Comparison, then reduce.** A comparison builds a boolean tensor of the
  same shape: `x > 0` marks positive entries with `True`. The operators `<`
  and `>` exclude the boundary; `<=` and `>=` include it.
  `x.any()` (is at least one entry True?) or `x.all()` (are they all
  True?) collapses it to one answer. `(x > 0).all()` asks "is everything
  positive?".
- **Whole-tensor equality.** `t.equal(a, b)` is an exact match of shape
  and values; `t.allclose(a, b)` is equality within floating-point tolerance
  — the right check after float arithmetic.

```python
import torch as t

grid = t.tensor([[3.0, -8.0, 1.0],
                 [6.0, 2.0, -9.0]])
print("negatives present?", bool((grid < 0).any()))
print("all positive?    ", bool((grid > 0).all()))
# Hidden checks
assert _delta_output == 'negatives present? True\nall positive?     False\n'
```

`a == b` alone is NOT a verdict — it's elementwise and yields a boolean
tensor (and `if` on it raises an error).

```python
a = t.tensor([1.0, 2.0])
b = t.tensor([1.0, 5.0])
print(a == b)                 # a tensor, not an answer
print(t.equal(a, b))          # one bool
# Hidden checks
assert _delta_output == 'tensor([ True, False])\nFalse\n'
```

And the reason `allclose` exists at all — float arithmetic does not land
where the arithmetic says it should:

```python
summed = t.full((10,), 0.1).sum()
print(summed.item(), "vs", 1.0)
# Hidden checks
assert not t.equal(summed, t.tensor(1.0))
assert t.allclose(summed, t.tensor(1.0))
```

## Worked example

```python
import torch as t

readings = t.tensor([[3.5, -2.0, 7.25],
                     [0.0,  9.5, -8.75]])

# Boolean pipeline: comparison (elementwise) then reduction (any).
# Read it aloud: "readings less than zero — any?"
has_negative = bool((readings < 0).any())

# Float-safe equality: after arithmetic, prefer allclose. Ten 0.1s summed
# in float32 land just past 1.0.
a = t.full((10,), 0.1).sum()
b = t.tensor(1.0)
print("any negative?", has_negative)
print("ten 0.1s summed:", a.item(), "| equal:", bool(t.equal(a, b)),
      "| allclose:", bool(t.allclose(a, b)))
# Hidden checks
assert has_negative is True
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


### q986
A whole-tensor verdict beside a piece of metadata — one is reduced out of
the values, the other was known before any of them were read.

```python starter
import torch as t


def solve(z):
    """(is any entry true?, how many entries there are)."""
    return (bool(z._____()), z._____())
```

```python solution
import torch as t


def solve(z):
    """(is any entry true?, how many entries there are)."""
    return (bool(z.any()), z.numel())
```

## Independent practice

From the drill bank: q62 (sum of an integer tensor as a plain Python int —
reduction plus the scalar boundary in one task).

From the drill bank: q497 (any element above a threshold, and how many — one mask, two answers).
From the drill bank: q498 (min, max and the span between them, as plain Python floats).


From the drill bank: q987 (centre and scale a tensor using whole-tensor statistics).
From the drill bank: q988 (whether two tensors hold equal totals within tolerance).
From the drill bank: q989 (the average and whether the tensor is constant).

### q1313
Total squared magnitude.

### q1314
Count zero entries.

### q1315
Fraction above a threshold.

### q1316
Midpoint of the observed range.

### q1317
Population variance.

### q1318
Prediction error.

### q1319
Opposite-sign pairs.

### q1320
Difference between batch totals.

### q1321
Count above the global average.

### q1322
Share of entries at the maximum.

### q1323
Total positive contribution.

### q1324
Root mean square.

## Guided practice

### q495
1. Two steps: collapse the tensor to one number, then leave PyTorch behind.
2. A whole-tensor reduction returns a 0-dimensional TENSOR, not a Python
   number — the test checks which one you handed back.
3. `x.sum().item()`.

### q496
1. Both halves are whole-tensor questions, so neither needs an axis.
2. The count is metadata you already know how to read — it is not a
   reduction, and it does not need .item().
3. `(x.mean().item(), x.numel())`.


## Integrated practice

### q990
Return total, average, minimum, and maximum as Python floats.

### q991
Return a centred tensor, its removed average, and a centring verdict.

### q992
Return span, average, and whether the tensor is constant.

### q1325
Weighted average.

### q1326
Average across unequal batches.

### q1327
Normalized mass with original total.

### q1328
Calibrate an observed range.

### q1329
Best single scale factor.

### q1330
Correct a constant sensor offset.

### q1331
Weighted spread.

### q1332
Accuracy within a tolerance.

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
