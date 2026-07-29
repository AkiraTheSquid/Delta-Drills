---
kc: numpy.ndarray-model
title: What a tensor is — data + shape + dtype
supporting: []
new_syntax: [torch.tensor, torch.tensor#dtype, Tensor.shape, Tensor.dtype, Tensor.ndim, Tensor.numel, torch.int64, torch.float32]
faded: [224, 482, 484]
guided: []
independent: [480, 481, 483, 485, 486]
---

## Concept: a tensor is one block of one type

PyTorch's core object is the **tensor** (n-dimensional array). Everything else in
this course — indexing, broadcasting, einsum, einops — is a way of manipulating
this one object, so it pays to know exactly what it is.

A Python list is a bag of pointers: each element can be a different type, live
anywhere in memory, and even be another list of a different length. A tensor is
the opposite: **one block of memory holding elements that are all the same
type**, plus a small amount of metadata describing how to interpret that block.
A freshly built tensor lays those elements out contiguously; operations that
only re-describe the block — transposing, slicing with a step — hand back a
tensor that shares the same memory in a different reading order, which is why
`.contiguous()` exists.

Why this design? Because when every element is the same type and sits at a
predictable memory address, PyTorch can hand whole-tensor operations to fast
compiled kernels instead of interpreting Python code element by element. That is
the entire performance story — and the reason the idiomatic style you will learn
here avoids writing Python `for` loops over elements.

The general procedure for turning existing Python data into a tensor is
`t.tensor(data)`: it walks the (possibly nested) sequence, finds a common
element type, and copies the values into one block.

By convention PyTorch is imported once per file as `import torch as t`. That
short alias is what the ARENA exercises use, so every `t.` you see below is the
same library you would import as `torch`.

## Worked example

Turn a nested Python list into a 2-D tensor:

```python
import torch as t

# A nested Python list: two inner lists of three integers each.
rows = [[1, 2, 3], [4, 5, 6]]

# t.tensor walks the nesting and copies the values into one block.
a = t.tensor(rows)

# The values survive the trip unchanged — .tolist() reads them back out.
assert a.tolist() == [[1, 2, 3], [4, 5, 6]]
```

You never told `t.tensor` how big the result should be, and you never told it
what type to use. It read both off the data. The next two segments are about
those two answers, because they are where almost every tensor bug lives.

## Faded practice

### q224
Turn a nested Python list of equal-length integer rows into a 2-D tensor.

```python starter
import torch as t

def solve(rows):
    """Return a 2-D tensor whose i-th row holds rows[i]."""
    return t._____(rows)
```

```python solution
import torch as t

def solve(rows):
    """Return a 2-D tensor whose i-th row holds rows[i]."""
    return t.tensor(rows)
```

## Concept: nesting becomes axes — shape, ndim, numel

The **shape** is a tuple giving the length along each dimension (axis). A 3×4
matrix has `shape == (3, 4)`: axis 0 has length 3 (rows), axis 1 has length 4
(columns). Axis 0 is always the outermost nesting level, so a flat list becomes
1-D, a list of equal-length lists becomes 2-D, and a list of those becomes 3-D.

Two smaller readings come off the same metadata:

- **`ndim`** — how many axes there are. This is the nesting depth, and it is an
  attribute, not a call.
- **`numel()`** — the total element count, which is the *product* of the shape.
  This is a method, so it needs the parentheses. For a 2×3 tensor `ndim` is 2
  and `numel()` is 6 — six numbers arranged as two rows, not two of anything.

Getting those two confused is the classic first-week error, and it is worth
fixing now: `ndim` counts axes, `numel()` counts numbers.

## Worked example

```python
import torch as t

flat = t.tensor([4, 1, 7])
grid = t.tensor([[1, 2, 3], [4, 5, 6]])

# One nesting level -> one axis. Three numbers in it.
assert flat.shape == (3,)
assert flat.ndim == 1
assert flat.numel() == 3

# Two nesting levels -> two axes, (rows, columns). numel is 2 * 3, not 2.
assert grid.shape == (2, 3)
assert grid.ndim == 2
assert grid.numel() == 6

# shape reports as torch.Size, which IS a tuple subclass — so it compares
# equal to a plain tuple, and tuple() converts it when you need the real thing.
assert tuple(grid.shape) == (2, 3)
```

## Faded practice

### q482
Count the axes and the elements. Remember which one takes parentheses.

```python starter
import torch as t

def solve(rows):
    """Return (number of axes, total element count) for the tensor from rows."""
    a = t.tensor(rows)
    return (a._____, a._____())
```

```python solution
import torch as t

def solve(rows):
    """Return (number of axes, total element count) for the tensor from rows."""
    a = t.tensor(rows)
    return (a.ndim, a.numel())
```

## Concept: dtype is a property of the whole block

The **dtype** is the single element type shared by every entry — `torch.int64`,
`torch.float32`, `torch.bool`, and so on. There is exactly one per tensor,
because there is exactly one block of memory and every slot in it is the same
size.

That has a consequence people meet by accident: when you build a tensor from
mixed Python numbers, PyTorch cannot keep some entries as ints and some as
floats. It picks ONE type that can hold everything, so a single float anywhere
in the input turns the whole tensor into `torch.float32`. All-integer input
gives you `torch.int64` instead. Ordinary division still works on that —
`a / 2` quietly hands back a float tensor — but anything that has to *write* a
float back into an integer block does not: `a /= 2` and `a.mean()` both raise
rather than silently rounding. When you know you want floats, say so up front
by passing `dtype=` instead of relying on how the input happened to be typed.

Two tensors can hold the same numbers in the same layout and still disagree on
dtype. Shape and dtype are independent, and code that checks only one of them
is checking half the question.

## Worked example

```python
import torch as t

ints = t.tensor([[1, 2], [3, 4]])
mixed = t.tensor([[1, 2.5], [3, 4]])

# All-integer input -> one integer type for all four entries.
assert ints.dtype == t.int64

# ONE float in the input decides the type of the whole block.
assert mixed.dtype == t.float32

# Same shape, different dtype: the two questions are independent.
assert ints.shape == mixed.shape
assert ints.dtype != mixed.dtype

# dtype= overrides the inference rather than relying on the input's spelling.
assert t.tensor([[1, 2], [3, 4]], dtype=t.float32).dtype == t.float32
```

## Faded practice

### q484
Two tensors, two independent questions. Compare each piece of metadata on its
own.

```python starter
import torch as t

def solve(rows_a, rows_b):
    """Return (do the shapes match?, do the dtypes match?)."""
    a = t.tensor(rows_a)
    b = t.tensor(rows_b)
    return (a._____ == b._____, a._____ == b._____)
```

```python solution
import torch as t

def solve(rows_a, rows_b):
    """Return (do the shapes match?, do the dtypes match?)."""
    a = t.tensor(rows_a)
    b = t.tensor(rows_b)
    return (a.shape == b.shape, a.dtype == b.dtype)
```

## Misconceptions

- **"A tensor is just a faster list."** — A list stores anything, a tensor
  stores exactly one dtype in one memory block. That's why `t.tensor([1, 2.5])`
  changes your integer to a float: PyTorch must pick ONE type for the block.
- **"I need to tell PyTorch the shape when converting data."** — `t.tensor`
  infers shape from the nesting. You only specify shapes with from-scratch
  constructors (`t.zeros((2, 3))`), covered next.
- **"shape is (columns, rows)."** — It is (axis 0, axis 1) = (rows, columns)
  for a matrix. Axis 0 is always the outermost nesting level.
- **"`ndim` and `numel()` are two names for the size."** — `ndim` counts axes,
  `numel()` counts elements. A 2×3 tensor has `ndim == 2` and `numel() == 6`.
