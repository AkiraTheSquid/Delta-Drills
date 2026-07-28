---
kc: numpy.ndarray-model
title: What a tensor is — data + shape + dtype
supporting: []
new_syntax: [torch.tensor, Tensor.shape, Tensor.dtype, Tensor.ndim, Tensor.numel, torch.int64, torch.float32]
faded: [224]
guided: []
independent: [480]
---

## Concept

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
`.contiguous()` exists. The three pieces of metadata you will use constantly:

- **`shape`** — a tuple giving the length along each dimension (axis). A 3×4
  matrix has `shape == (3, 4)`: axis 0 has length 3 (rows), axis 1 has length 4
  (columns).
- **`dtype`** — the single element type shared by every entry (`torch.int64`,
  `torch.float32`, `torch.bool`, …).
- **`ndim` / `numel()`** — the number of axes, and the total element count
  (the product of the shape).

Why this design? Because when every element is the same type and sits at a
predictable memory address, PyTorch can hand whole-tensor operations to fast
compiled kernels instead of interpreting Python code element by element. That is
the entire performance story — and the reason the idiomatic style you will learn
here avoids writing Python `for` loops over elements.

The general procedure for turning existing Python data into a tensor is
`t.tensor(data)`: it walks the (possibly nested) sequence, finds a common
element type, and copies the values into one block. Nesting depth becomes the
number of dimensions — a flat list becomes a 1-D tensor, a list of equal-length
lists becomes a 2-D tensor, and so on.

By convention PyTorch is imported once per file as:

```python
import torch as t
```

That short alias is what the ARENA exercises use, so every `t.` you see below is
the same library you would import as `torch`.

## Worked example

Turn a nested Python list into a 2-D tensor:

```python
import torch as t

# A nested Python list: two inner lists of three integers each.
rows = [[1, 2, 3], [4, 5, 6]]

# The input is still an ordinary Python list — nothing has changed yet.
print("input:", rows)

# t.tensor walks the nesting: outer list -> axis 0, inner lists -> axis 1.
a = t.tensor(rows)

# Printing a tensor shows the values AND the structure the nesting produced.
print(a)

# shape is a tuple: (length of axis 0, length of axis 1) = (rows, columns).
print("shape:", a.shape)

# dtype is the ONE element type shared by every entry. Every input was an
# int, so PyTorch picked a single integer type for the whole block.
print("dtype:", a.dtype)

# ndim counts the axes; numel() counts the elements (the product of shape).
print("ndim:", a.ndim, "numel:", a.numel())

# Mixing an int with a float forces ONE common type for the whole block:
# the integer is widened to a float rather than kept as an int.
print("mixed dtype:", t.tensor([1, 2.5]).dtype)
```

Run that and PyTorch prints:

```output
input: [[1, 2, 3], [4, 5, 6]]
tensor([[1, 2, 3],
        [4, 5, 6]])
shape: torch.Size([2, 3])
dtype: torch.int64
ndim: 2 numel: 6
mixed dtype: torch.float32
```

Read the output line by line. You never told `t.tensor` the shape — the
nesting set it, and `torch.Size([2, 3])` is PyTorch reporting back what it
inferred. `torch.int64` is the single type it chose for all six entries. The
last line is the same rule biting: ask for one int and one float in one
tensor and you get `torch.float32` for both, because the block can only hold
one type. Printing `shape` and `dtype` right after you build a tensor should
become a reflex — most tensor bugs are a shape or dtype surprise, and they are
invisible until you look.

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

## Misconceptions

- **"A tensor is just a faster list."** — A list stores anything, a tensor
  stores exactly one dtype in one memory block. That's why `t.tensor([1, 2.5])`
  changes your integer to a float: PyTorch must pick ONE type for the block.
- **"I need to tell PyTorch the shape when converting data."** — `t.tensor`
  infers shape from the nesting. You only specify shapes with from-scratch
  constructors (`t.zeros((2, 3))`), covered next.
- **"shape is (columns, rows)."** — It is (axis 0, axis 1) = (rows, columns)
  for a matrix. Axis 0 is always the outermost nesting level.
- **"`.size` is the element count, like in NumPy."** — In PyTorch `.size` is a
  *method* that returns the shape; the element count is `.numel()`. Reaching for
  NumPy's `.size` attribute here gets you a bound method, not a number.
