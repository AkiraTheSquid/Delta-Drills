---
kc: numpy.ndarray-model
title: What an ndarray is — data + shape + dtype
supporting: []
new_syntax: [import-numpy-as-np, array-attributes]
faded: [224]
guided: []
independent: []
---

## Concept

NumPy's core object is the **ndarray** (n-dimensional array). Everything else in
this course — indexing, broadcasting, einsum, einops — is a way of manipulating
this one object, so it pays to know exactly what it is.

A Python list is a bag of pointers: each element can be a different type, live
anywhere in memory, and even be another list of a different length. An ndarray is
the opposite: **one contiguous block of memory holding elements that are all the
same type**, plus a small amount of metadata describing how to interpret that
block. The three pieces of metadata you will use constantly:

- **`shape`** — a tuple giving the length along each dimension (axis). A 3×4
  matrix has `shape == (3, 4)`: axis 0 has length 3 (rows), axis 1 has length 4
  (columns).
- **`dtype`** — the single element type shared by every entry (`int64`,
  `float64`, `bool`, …).
- **`ndim` / `size`** — the number of axes, and the total element count
  (the product of the shape).

Why this design? Because when every element is the same type and sits at a
predictable memory address, NumPy can hand whole-array operations to fast
compiled loops instead of interpreting Python code element by element. That is
the entire performance story of NumPy — and the reason the idiomatic style you
will learn here avoids writing Python `for` loops over array elements.

The general procedure for turning existing Python data into an array is
`np.array(data)`: it walks the (possibly nested) sequence, finds a common
element type, and copies the values into one block. Nesting depth becomes the
number of dimensions — a flat list becomes a 1-D array, a list of equal-length
lists becomes a 2-D array, and so on.

By convention NumPy is imported once per file as:

```python
import numpy as np
```

## Worked example

Turn a nested Python list into a 2-D array:

```python
import numpy as np

rows = [[1, 2, 3],
        [4, 5, 6]]

# np.array walks the nesting: outer list -> axis 0, inner lists -> axis 1.
a = np.array(rows)

assert a.shape == (2, 3)      # 2 rows, 3 columns — inferred from the nesting
assert a.dtype == np.int64    # every input was an int -> one int dtype for all
```

Why: you don't tell `np.array` the shape — nesting sets the dimensions.
Checking `shape` and `dtype` right after should become a reflex; most NumPy
bugs are shape or dtype surprises.

## Faded practice

### q224
Turn a nested Python list of equal-length integer rows into a 2-D array.

```python starter
import numpy as np

def solve(rows):
    """Return a 2-D NumPy array whose i-th row holds rows[i]."""
    return np._____(rows)
```

```python solution
import numpy as np

def solve(rows):
    """Return a 2-D NumPy array whose i-th row holds rows[i]."""
    return np.array(rows)
```

## Misconceptions

- **"An array is just a faster list."** — A list stores anything, an array
  stores exactly one dtype in one memory block. That's why `np.array([1, 2.5])`
  changes your integer to a float: NumPy must pick ONE type for the block.
- **"I need to tell NumPy the shape when converting data."** — `np.array`
  infers shape from the nesting. You only specify shapes with from-scratch
  constructors (`np.zeros((2, 3))`), covered next.
- **"shape is (columns, rows)."** — It is (axis 0, axis 1) = (rows, columns)
  for a matrix. Axis 0 is always the outermost nesting level.
