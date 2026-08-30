---
kc: numpy.constructors
title: Tensor constructors — zeros, ones, full, eye, *_like
supporting: [numpy.ndarray-model]
new_syntax: [Tensor.all, torch.bool, torch.eye, torch.full, torch.full_like, torch.int32, torch.ones, torch.ones#dtype, torch.ones_like, torch.zeros, torch.zeros#dtype, torch.zeros_like]
previews: [syntax.matmul]
faded: [227, 212, 41, 228, 639, 640, 641, 642]
guided: [48]
independent: [225, 50, 213, 643, 644, 645]
integrated: [646, 647, 648]
---

## Concept: constructors and the shape argument

`t.tensor` converts data you already have. Just as often you need a tensor
**built from scratch** — a canvas of zeros to fill in, a mask of ones. PyTorch
has one constructor per pattern, and they all share the same calling
convention:

> **constructor(shape, dtype=...)** — say how big, optionally say what type.

- **`t.zeros(shape)`** — all entries `0.0`. The default "empty canvas".
- **`t.ones(shape)`** — all entries `1.0`.
- **`t.full(shape, v)`** — all entries equal to your value `v`.
- **`t.empty(shape)`** — allocates *without initializing* (contents are
  whatever bytes were in memory). Only worth it when you will overwrite
  every entry immediately.

Three of them side by side — same call shape, different fill:

```python
import torch as t

print(t.zeros((2, 3)))
print(t.ones((2, 3)))
print(t.full((2, 3), 7.0))
```

PyTorch accepts the shape **either way**: `t.zeros(2, 3)` and `t.zeros((2, 3))`
both give a 2×3 tensor. That is worth noticing precisely because NumPy does
*not* allow it — `np.zeros(2, 3)` is a `TypeError`, since NumPy reads the
second positional argument as the dtype. Code translated from NumPy will use
the tuple form, and it keeps working.

```python
loose = t.zeros(3, 4)
tupled = t.zeros((3, 4))
assert loose.shape == tupled.shape == (3, 4)
print(loose.shape, "==", tupled.shape)
```

## Worked example

Build a 3×4 canvas of zeros:

```python
import torch as t

# Both spellings mean the same thing in PyTorch.
board = t.zeros((3, 4))
same = t.zeros(3, 4)
assert board.shape == (3, 4) == same.shape
print(board)
print("both spellings give", tuple(board.shape), "and", tuple(same.shape))
```

Why: shape comes first and everything else is keyword-only, so the constructor
never has to guess whether you meant a dimension or a dtype.

## Faded practice

### q227
All-zeros float vector of a given length (must also work for length 0).

```python starter
import torch as t

def solve(n):
    """Return a 1-D float tensor of n zeros."""
    return t._____(n)
```

```python solution
import torch as t

def solve(n):
    """Return a 1-D float tensor of n zeros."""
    return t.zeros(n)
```

### q639
A grid of one value — say how big, then say the value.

```python starter
import torch as t

def solve(rows, cols, v):
    """Return a rows x cols tensor of v, as a nested list."""
    return t._____((rows, cols), v).tolist()
```

```python solution
import torch as t

def solve(rows, cols, v):
    """Return a rows x cols tensor of v, as a nested list."""
    return t.full((rows, cols), v).tolist()
```

## Concept: the dtype is float32 unless you say otherwise

**The default floating dtype is `torch.float32`,** even though the values print
like integers. This is a real difference from NumPy, whose default is
`float64` — the same line of code gives you half the precision here, which is
deliberate, because neural networks are trained in 32-bit.

```python
import torch as t

default = t.ones(3)
print(default, default.dtype)
assert default.dtype == t.float32
```

The values printed as `1.` with a trailing dot, and that dot is the whole
warning: they are floats, not the integers they look like.

Pass `dtype=` to override: `t.ones((2, 2), dtype=t.bool)` is a matrix of `True`
(1 as a boolean is `True`); `t.zeros(4, dtype=t.int64)` is integer zeros.
Checking `dtype` right after construction is the habit that catches the
float-by-default surprise before it propagates.

```python
flags = t.ones((2, 2), dtype=t.bool)
counts = t.zeros(4, dtype=t.int64)
print(flags)
print(counts, counts.dtype)
assert bool(flags.all()) and flags.dtype == t.bool
```

## Worked example

An all-`True` boolean mask — ones, with the dtype said out loud:

```python
import torch as t

# ones gives every entry the value 1 — and 1 as a boolean is True.
mask = t.ones((3, 4), dtype=t.bool)
assert bool(mask.all()) and mask.dtype == t.bool
print(mask)

# The default, for contrast — float32, not float64 and not int.
assert t.ones(3).dtype == t.float32
print("asked for bool:", mask.dtype, "| default:", t.ones(3).dtype)
```

Why: without `dtype=t.bool` this would be a float tensor of 1.0s that merely
*prints* like what you wanted — say the type when it matters.

## Faded practice

### q212
A rows×cols tensor where every entry is the boolean `True`.

```python starter
import torch as t

def solve(rows, cols):
    """All-True boolean matrix of shape (rows, cols)."""
    return t.ones((rows, cols), dtype=_____)
```

```python solution
import torch as t

def solve(rows, cols):
    """All-True boolean matrix of shape (rows, cols)."""
    return t.ones((rows, cols), dtype=t.bool)
```

### q640
Zeros that are actually integers.

```python starter
import torch as t

def solve(n):
    """Return (n integer zeros as a list, their dtype name)."""
    z = t._____(n, _____=t.int64)
    return (z.tolist(), str(z.dtype))
```

```python solution
import torch as t

def solve(n):
    """Return (n integer zeros as a list, their dtype name)."""
    z = t.zeros(n, dtype=t.int64)
    return (z.tolist(), str(z.dtype))
```

## Concept: *_like — copy shape AND dtype from an existing tensor

The **`*_like` variants** (`t.zeros_like(x)`, `t.ones_like(x)`,
`t.full_like(x, v)`) copy both the shape *and the dtype* from an existing
tensor — the right tool whenever the question is "give me a blank tensor
shaped like this one". Reaching for `*_like` is both shorter and safer than
reading off `.shape` and `.dtype` yourself, and in real model code it also
carries across the device the original lives on.

```python
import torch as t

x = t.tensor([[3, -1, 4], [1, 5, -9]], dtype=t.int32)
print("zeros_like:", t.zeros_like(x).dtype)
print("zeros(x.shape):", t.zeros(x.shape).dtype)
```

Same shape from both, but only one of them still knows the tensor was
integer. Every `*_like` behaves that way:

```python
print(t.ones_like(x))
print(t.full_like(x, 7))
assert t.full_like(x, 7).dtype == x.dtype == t.int32
```

## Worked example

```python
import torch as t

# "Blank tensor shaped like x" — zeros_like copies shape AND dtype,
# so an int32 input yields an int32 result, not the float default.
x = t.tensor([[3, -1, 4], [1, 5, -9]], dtype=t.int32)
blank = t.zeros_like(x)
assert blank.shape == x.shape
assert blank.dtype == t.int32
print(blank)
print("copied dtype:", blank.dtype, "| t.zeros(x.shape) would give:",
      t.zeros(x.shape).dtype)
```

Why: `t.zeros(x.shape)` would lose the dtype (float default) — `_like`
keeps both properties in one call.

## Faded practice

### q41
Blank tensor matching BOTH the shape and dtype of an existing tensor.

```python starter
import torch as t

def solve(x):
    """Return an all-zeros tensor with x's shape and x's dtype."""
    return t._____(x)
```

```python solution
import torch as t

def solve(x):
    """Return an all-zeros tensor with x's shape and x's dtype."""
    return t.zeros_like(x)
```

### q641
Shaped and typed like x, filled with v.

```python starter
import torch as t

def solve(x, v):
    """Return (a tensor of v shaped AND typed like x, its dtype name)."""
    filled = t._____(x, v)
    return (filled.tolist(), str(filled.dtype))
```

```python solution
import torch as t

def solve(x, v):
    """Return (a tensor of v shaped AND typed like x, its dtype name)."""
    filled = t.full_like(x, v)
    return (filled.tolist(), str(filled.dtype))
```

## Concept: t.eye — the identity matrix

**`t.eye(n)`** is the n×n identity matrix: `1.0` on the main diagonal,
`0.0` elsewhere. It's the seed for anything diagonal-shaped: `v * t.eye(n)`
puts a constant v on the diagonal, and indexing its rows with a permutation
turns it into a permutation matrix (a trick you will use in the random-number
KP).

```python
import torch as t

print(t.eye(3))
print(5.0 * t.eye(3))
```

The permutation trick, since it is the least obvious one — reordering the
identity's ROWS builds the matrix that reorders a vector the same way:

```python
order = t.tensor([2, 0, 1])
P = t.eye(3)[order]
v = t.tensor([10.0, 20.0, 30.0])
print(P)
print(P @ v)
assert (P @ v).tolist() == [30.0, 10.0, 20.0]
```

## Worked example

```python
import torch as t

I = t.eye(3)
assert I.tolist() == [[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]]
print(I)
```

Why: the identity is a constructor, not something you assemble by loop —
and scaling it (`v * t.eye(n)`) is the one-liner for "v on the diagonal".

## Faded practice

### q228
The n×n identity matrix.

```python starter
import torch as t

def solve(n):
    """n-by-n identity matrix."""
    return t._____(n)
```

```python solution
import torch as t

def solve(n):
    """n-by-n identity matrix."""
    return t.eye(n)
```

### q642
The identity, and its shape.

```python starter
import torch as t

def solve(n):
    """Return (the n x n identity as a nested list, its shape)."""
    I = t._____(n)
    return (I.tolist(), tuple(I.shape))
```

```python solution
import torch as t

def solve(n):
    """Return (the n x n identity as a nested list, its shape)."""
    I = t.eye(n)
    return (I.tolist(), tuple(I.shape))
```

## Guided practice

### q48
1. Nothing is being constructed from scratch here — you need a COPY of v
   that you are allowed to write into.
2. Odd indices are a slice with a step, and assigning a scalar into a
   slice broadcasts it across every selected position.
3. `out = v.clone()` then `out[1::2] = fill`. Skipping the clone mutates
   the caller's tensor, which the drill checks for.

## Solo practice

### q225
n ones in the default float dtype.

### q50
v on the diagonal, zeros elsewhere — the identity, scaled.

### q213
A one-hot vector: zeros first, then one write.

### q643
An all-False mask.

### q644
Ones shaped like x, counted and typed.

### q645
An integer identity — eye takes dtype= like the others.

Every constructor on this page takes the same `dtype=` keyword, and `eye`
is no exception:

```python worked
import torch as t

flags = t.eye(3, dtype=t.bool)
print(flags)
print(flags.dtype)
assert flags.dtype == t.bool
```

## Integrated practice

### q646
Two grids of the same shape, and what each reports about itself.

### q647
The whole _like family on one tensor.

### q648
Three constructors, three dtypes, one call.

## Misconceptions

- **"Constructors give me integers if I write `t.ones(5)`."** — The default
  dtype is `torch.float32` regardless of how the values look. If the grader (or
  your model) needs ints or bools, say so with `dtype=`.
- **"float is float — precision doesn't change when I port NumPy code."** —
  `np.ones(4)` is float64 and `t.ones(4)` is float32. Exact equality checks
  written against NumPy output can fail here for no reason other than the
  narrower dtype.
- **"`dtype=bool` works, like in NumPy."** — PyTorch wants its own dtype
  objects: `t.bool`, `t.int64`, `t.float32`. Python's builtin `bool` and `int`
  are accepted in some places but `t.*` is the spelling to learn.
- **"`t.empty` means a tensor with no elements."** — It means *uninitialized
  memory* of the full requested shape: garbage values, not zeros, not empty.
  Use `t.zeros` unless you will overwrite everything.
