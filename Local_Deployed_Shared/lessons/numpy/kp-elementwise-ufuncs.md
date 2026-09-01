---
kc: numpy.elementwise-ufuncs
title: Elementwise math
supporting: [numpy.ndarray-model]
new_syntax: [Tensor.clamp, Tensor.clamp#max, Tensor.clamp#min, torch.ceil, torch.floor, torch.maximum, torch.minimum, torch.round, torch.sqrt, torch.trunc]
previews: [Tensor.max, syntax.matmul]
faded: [192, 49, 67, 630, 631, 632]
guided: [487, 488]
independent: [43, 489, 63, 633, 634, 635]
integrated: [636, 637, 638]
---

## Concept: write the formula once — operators are elementwise

The core promise of tensor programming: **write the formula once, and it
applies to every element** — no loop. Operators `+ - * / ** %` between a
tensor and a scalar, or between two same-shaped tensors, work element by
element: `z * 2` doubles everything; `a * b` multiplies corresponding entries
(NOT matrix multiplication — that's `@`).

The general procedure for any "transform each entry" task:

> Express the rule for ONE element as a formula, then write that formula
> with the whole tensor in place of the element.

```python
import torch as t

z = t.tensor([1.0, 2.0, 3.0, 4.0])
print(z * 2)
print(z ** 2 - 1)
print(z % 2)
```

"Replace each x by x² − 1" → `z**2 - 1`. If you find yourself writing
`for i in range(len(z))`, stop — the elementwise spelling is shorter and
orders of magnitude faster (the loop happens in compiled code, and on a GPU it
happens in parallel). All of these return **new tensors** and leave the input
untouched.

```python
before = z.tolist()
squared = z ** 2 - 1
print("z after the expression:", z)
assert z.tolist() == before          # nothing was written back
```

And the one operator that is NOT elementwise, so the contrast lands early:

```python
a = t.tensor([[1.0, 2.0], [3.0, 4.0]])
print("a * a (elementwise):")
print(a * a)
print("a @ a (matrix product):")
print(a @ a)
assert not t.equal(a * a, a @ a)
```

## Worked example

```python
import torch as t

z = t.tensor([1.0, 2.0, 3.0])

# The per-element rule "x**2 - 1", written once for the whole tensor:
out = z**2 - 1
assert out.tolist() == [0.0, 3.0, 8.0]

# The input is untouched — the expression built a new tensor.
assert z.tolist() == [1.0, 2.0, 3.0]
print("z  ", z)
print("out", out)
```

Why: the expression reads exactly like the per-element rule — that
transliteration IS the method.

## Faded practice

### q192
Elementwise cube.

```python starter
import torch as t

def solve(x):
    """Each entry raised to the third power."""
    return x _____ 3
```

```python solution
import torch as t

def solve(x):
    """Each entry raised to the third power."""
    return x ** 3
```

### q630
A line through every entry — one formula, no loop.

```python starter
import torch as t

def solve(x, m, b):
    """Return m * x + b for every entry."""
    return m _____ x _____ b
```

```python solution
import torch as t

def solve(x, m, b):
    """Return m * x + b for every entry."""
    return m * x + b
```

## Concept: named math functions and the rounding family

Beyond operators, named functions map over the whole tensor: `t.sqrt`,
`t.abs`, `t.exp`, `t.log`, `t.sin`, …

Rounding is a *family*, and the members differ on negatives:

- `t.round` — nearest;
- `t.floor` — largest integer ≤ x, so −0.3 → −1.0 (away from zero);
- `t.ceil` — smallest integer ≥ x;
- `t.trunc` — toward zero, so −0.3 → −0.0 (same as `.to(t.int64)`).

Read the task's example values to see which member is being asked for — and
the fastest way to tell them apart is to run all four on the same negatives:

```python
import torch as t

v = t.tensor([1.7, -0.3, 2.5, -2.5])
print("input", v)
print("round", t.round(v))
print("floor", t.floor(v))
print("ceil ", t.ceil(v))
print("trunc", t.trunc(v))
```

The only column where floor and trunc disagree is the negative one, which is
exactly where a grader will look:

```python
assert t.floor(v).tolist() == [1.0, -1.0, 2.0, -3.0]
assert t.trunc(v).tolist() == [1.0, -0.0, 2.0, -2.0]
assert t.equal(t.trunc(v).to(t.int64), v.to(t.int64))
print("at -0.3: floor ->", t.floor(v)[1].item(),
      "but trunc ->", t.trunc(v)[1].item())
print("trunc == .to(t.int64):", bool(t.equal(t.trunc(v).to(t.int64),
                                             v.to(t.int64))))
```

## Worked example

```python
import torch as t

v = t.tensor([1.7, -0.3, 2.5])

assert t.floor(v).tolist() == [1.0, -1.0, 2.0]   # floor moves DOWN
assert t.trunc(v).tolist() == [1.0, -0.0, 2.0]   # trunc moves toward zero
assert t.sqrt(t.tensor([4.0, 9.0])).tolist() == [2.0, 3.0]
print("input", v)
print("floor", t.floor(v))
print("trunc", t.trunc(v), "  <- differs only at the negative")
```

Why: floor vs trunc only disagree on negatives — that's exactly where tasks
(and graders) check.

## Faded practice

### q49
Floor every entry (note what floor does to negatives).

```python starter
import torch as t

def solve(z):
    """Replace each entry by the largest integer value <= it."""
    return t._____(z)
```

```python solution
import torch as t

def solve(z):
    """Replace each entry by the largest integer value <= it."""
    return t.floor(z)
```

### q631
The fractional part, sign kept — pick the rounding family member that cuts toward zero.

```python starter
import torch as t

def solve(x):
    """Return the fractional part of every entry, keeping its sign."""
    return x - t._____(x)
```

```python solution
import torch as t

def solve(x):
    """Return the fractional part of every entry, keeping its sign."""
    return x - t.trunc(x)
```

## Concept: elementwise choosers — maximum, minimum, clamp

`t.maximum(a, b)` / `t.minimum(a, b)` pick the larger/smaller *at each
position* (contrast with `a.max()`, which reduces the whole tensor to one
number — different KP). `z.clamp(min=lo, max=hi)` limits values to a range;
NumPy calls this `clip`, and PyTorch accepts that spelling as an alias, but
`clamp` is the name you will read in model code.

```python
import torch as t

a = t.tensor([1.0, 5.0, 2.0])
b = t.tensor([3.0, 4.0, 2.5])
print("maximum:", t.maximum(a, b))     # one answer per position
print("a.max():", a.max())             # one answer, full stop
```

These overlap: `clamp(max=100)` and `t.minimum(x, ...)` compute the same
thing, and `clamp(min=0)` is ReLU. Prefer `clamp` for a constant bound — it
takes a plain Python number, whereas `t.minimum` wants a second tensor, and a
tensor you construct on the spot lands on the CPU and will not match an input
living on a GPU.

```python
readings = t.tensor([-2.0, 0.5, 7.0, 12.0])
print("clamp(min=0):     ", readings.clamp(min=0.0))       # ReLU
print("clamp(max=10):    ", readings.clamp(max=10.0))
print("clamp(0, 10):     ", readings.clamp(min=0.0, max=10.0))
assert t.equal(readings.clamp(max=10.0), t.minimum(readings, t.tensor(10.0)))
```

## Worked example

```python
import torch as t

# Elementwise chooser between TWO tensors: keep the larger at each slot.
a = t.tensor([1.0, 5.0, 2.0])
b = t.tensor([3.0, 4.0, 2.5])
assert t.maximum(a, b).tolist() == [3.0, 5.0, 2.5]

# Pipeline: curve exam scores — add 5, cap at 100, floor to whole points.
scores = t.tensor([71.5, 88.25, 97.0, 99.5])
curved = t.floor((scores + 5).clamp(max=100.0))
assert curved.tolist() == [76.0, 93.0, 100.0, 100.0]
print("raw   ", scores)
print("curved", curved)

# The other bound. `min=` sets a FLOOR, `max=` sets a CEILING — and the two
# read backwards from how they sound: min= raises everything below it.
readings = t.tensor([-2.0, 0.5, 7.0])
assert readings.clamp(min=0.0).tolist() == [0.0, 0.5, 7.0]
print("readings   ", readings)
print("clamp(min=0)", readings.clamp(min=0.0), " <- this is ReLU")
```

Why: composing these left-to-right is normal style — each stage maps over
the whole tensor, and the pipeline reads exactly like the per-element rule:
add, cap, floor.

The last line is the one worth memorising: `clamp(min=0.0)` is ReLU, the
single most common nonlinearity in the whole field.

## Faded practice

### q67
Negatives become 0.0, non-negatives pass through (ReLU).

```python starter
import torch as t

def solve(z):
    """Each negative entry replaced by 0.0 (new tensor; z unmodified)."""
    return z.clamp(_____=0.0)
```

```python solution
import torch as t

def solve(z):
    """Each negative entry replaced by 0.0 (new tensor; z unmodified)."""
    return z.clamp(min=0.0)
```

### q632
Position by position, keep the smaller of the two.

```python starter
import torch as t

def solve(a, b):
    """Return the smaller of each pair of corresponding entries."""
    return t._____(a, b)
```

```python solution
import torch as t

def solve(a, b):
    """Return the smaller of each pair of corresponding entries."""
    return t.minimum(a, b)
```

## Guided practice

### q487
1. The whole tensor at once — there is no index to loop over.
2. An arithmetic operator applied to a tensor is applied to every element.
3. `x * 2`.

### q488
1. You need the magnitude of each element, sign discarded.
2. It is a method on the tensor, and it returns a NEW tensor rather than
   editing yours — which is what keeps the input unchanged.
3. `x.abs()`.

## Solo practice

### q43
The larger of each pair of corresponding entries.

### q489
Pull every entry into [lo, hi] — both bounds at once.

### q63
The integer part of positive entries — several members of the family agree here.

### q633
Pythagoras on every pair — square, add, root.

### q634
The two integers each entry sits between.

### q635
A floor under every entry, written as a maximum against a constant.

maximum compares two tensors position by position. Pair x with a tensor of
zeros shaped like it and you have ReLU without clamp:

```python worked
import torch as t

x = t.tensor([-2.0, 0.5, 3.0])
zeros = t.zeros_like(x)
print("x     ", x)
print("zeros ", zeros)
print("relu  ", t.maximum(x, zeros))
assert t.equal(t.maximum(x, zeros), x.clamp(min=0.0))
```

## Integrated practice

### q636
Clamp, round, and report whether the clamp had anything to do.

### q637
Larger, smaller, and the gap — three elementwise results from one pair.

### q638
One conversion formula, then two different elementwise clean-ups.

## Misconceptions

- **"`a * b` on two matrices is matrix multiplication."** — It is elementwise.
  Matrix product is `a @ b`. This distinction matters enough that it gets its
  own KP later.
- **"`t.maximum` and `t.max` are the same."** — `t.maximum(a, b)` compares
  two tensors position-by-position (returns a tensor); `t.max(a)` reduces one
  tensor to its single largest value.
- **"floor and truncate are the same."** — Only for positives. For negatives,
  floor moves AWAY from zero (−0.3 → −1.0) while trunc moves toward it
  (−0.3 → −0.0). Read the task's example values to see which is being asked.
- **"An in-place version is just a style choice."** — Methods ending in an
  underscore (`z.clamp_(min=0)`) modify the tensor you were handed. That is a
  different contract from `z.clamp(min=0)`, and it is how you accidentally
  mutate a caller's data.
