---
kc: torch.slice-assignment
title: Writing tensors into slices: broadcasting on the left
supporting: [numpy.slicing-views, numpy.broadcasting-rules, numpy.constructors, numpy.ranges]
new_syntax: [Tensor.copy_]
previews: []
faded: [809, 810]
guided: []
independent: [811, 812, 813, 814, 815, 816]
integrated: [817, 818, 819]
---

## Concept: `x[sel] = value` writes in place, broadcasting the value

You have met `x[2:5] = 0.0`: assigning to a slice writes through the view into
`x`. That page used a scalar. Here the right-hand side becomes a **tensor**,
the slice becomes **multi-axis**, and two rules make this the workhorse of
tensor building:

1. **The right-hand side broadcasts to the slice.** A scalar fills every slot;
   a 1-D tensor of the right length fills a column or a row; a full tensor of
   the slice's shape is copied element by element. Ordinary broadcasting
   rules, applied to the *slice's* shape.
2. **The write is in place.** Every other view of the same storage sees it,
   and the tensor's shape and dtype never change (a float written into an int
   tensor is truncated, not upcast).

```python
import torch as t

x = t.zeros((3, 4))
x[:, 1] = 3.0                 # scalar → whole column
x[0] = t.arange(4)            # 1-D of length 4 → one row
x[1:, 2:] = t.ones((2, 2))    # block of matching shape
print(x)
```

The same write has a **method spelling** you will see in library code:
`x[0].copy_(src)` is `x[0] = src`, with `src` broadcast to the slice. The
trailing underscore is PyTorch's mark for *in place* — it acts on the view,
so it writes into `x`:

```python
import torch as t

x = t.zeros((2, 3))
x[1].copy_(t.tensor([1.0, 2.0, 3.0]))
x[:, 0].copy_(t.tensor(7.0))        # a 0-d tensor broadcasts like a scalar
print(x)
```

Shape still has to work. `x[:, 1] = t.arange(5)` on a 3-row `x` is a
`RuntimeError` — five values cannot broadcast to three slots — and, unlike
`out=`, assignment **does** raise, which is why it is the safer of the two.

## Worked example

Three slabs of shape (2, 3). Give row 1 of every slab an x of 1 and a y equal
to the slab's own index, then scale the middle slab's row 1 by 10 — three
writes, one canvas:

```python
import torch as t

slabs = t.zeros((3, 2, 3))
slabs[:, 1, 0] = 1.0                 # scalar broadcast to 3 slots
slabs[:, 1, 1] = t.arange(3)         # 1-D of length 3 into 3 slots
slabs[1, 1] = slabs[1, 1] * 10       # a whole row, rewritten from itself
print(slabs)
assert slabs[1, 1].tolist() == [10.0, 10.0, 0.0]
```

| write | slice shape | RHS shape | result |
| --- | --- | --- | --- |
| `x[:, 0] = 1.0` on 3×2 | `(3,)` | `()` | column 0 all ones |
| `x[0] = t.arange(2)` on 3×2 | `(2,)` | `(2,)` | row 0 = `[0, 1]` |
| `x[:, 0] = t.arange(2)` on 3×2 | `(3,)` | `(2,)` | `RuntimeError` |

## Faded practice

### q809
The slice is the target; the scalar broadcasts to every slot of it.

```python starter
def solve(n, v, col):
    """One column set to v by assigning into the slice."""
    canvas = t.zeros((n, 3))
    canvas[_____, _____] = v
    return canvas.tolist()
```

```python solution
def solve(n, v, col):
    """One column set to v by assigning into the slice."""
    canvas = t.zeros((n, 3))
    canvas[:, col] = v
    return canvas.tolist()
```

### q810
copy_ takes a tensor; wrap the list first.

```python starter
def solve(m, row, row_values):
    """One row overwritten with copy_."""
    canvas = t.zeros((m, len(row_values)))
    canvas[row]._____(t.tensor(row_values))
    return canvas.tolist()
```

```python solution
def solve(m, row, row_values):
    """One row overwritten with copy_."""
    canvas = t.zeros((m, len(row_values)))
    canvas[row].copy_(t.tensor(row_values))
    return canvas.tolist()
```

## Solo practice

### q811
Two scalar broadcasts into two columns.

### q812
A 1-D tensor assigned into a column.

### q813
A write through one view shows in every view.

### q814
A block write via two slices.

### q815
Assign a row, then copy_ a column; order matters at the crossing.

### q816
A float assigned into an int tensor is truncated, not upcast.

## Integrated practice

### q817
Three column writes, the third built from the second.

### q818
A row-vector broadcast into a stepped row slice.

### q819
copy_ a block, then rewrite it from itself.

## Misconceptions

- **"`x[:, 0] = v` makes a new tensor."** It does not; it writes into the
  storage `x` already has, and every view of that storage sees the change.
  Rebinding (`x = ...`) is what makes new tensors.
- **"Rows and columns are interchangeable in a write."** `x[0]` is a row,
  `x[:, 0]` a column. The shapes differ, so a 1-D right-hand side that fits
  one will raise on the other whenever the tensor is not square.
- **"Assignment upcasts."** The tensor's dtype is fixed; a float written into
  an int tensor is truncated toward zero.
