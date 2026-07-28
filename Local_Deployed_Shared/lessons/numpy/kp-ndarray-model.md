---
kc: numpy.ndarray-model
title: What a tensor is — data + shape + dtype
supporting: []
new_syntax: [import-torch-as-t, tensor-attributes]
faded: [224]
guided: []
independent: []
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

rows = [[1, 2, 3],
        [4, 5, 6]]

# t.tensor walks the nesting: outer list -> axis 0, inner lists -> axis 1.
a = t.tensor(rows)

assert a.shape == (2, 3)      # 2 rows, 3 columns — inferred from the nesting
assert a.dtype == t.int64     # every input was an int -> one int dtype for all
assert a.ndim == 2 and a.numel() == 6
```

Why: you don't tell `t.tensor` the shape — nesting sets the dimensions.
Checking `shape` and `dtype` right after should become a reflex; most tensor
bugs are shape or dtype surprises.

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
