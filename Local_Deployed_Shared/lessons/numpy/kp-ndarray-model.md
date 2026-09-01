---
kc: numpy.ndarray-model
title: What a tensor is — data + shape + dtype
supporting: []
new_syntax: [Tensor.dtype, Tensor.item, Tensor.ndim, Tensor.numel, Tensor.shape, Tensor.tolist, torch.equal, torch.float32, torch.int64, torch.tensor, torch.tensor#dtype]
faded: [224, 482, 484, 532, 533, 534, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546]
guided: []
independent: [480, 481, 483, 485, 486, 547, 548, 549, 550, 551, 554, 555, 556, 557, 559]
integrated: [560, 562, 563, 564, 566, 567]
---

## Concept: a tensor is one block of one type


PyTorch's core object is the **tensor** (n-dimensional array). Every other
thing this course teaches — selecting parts of one, doing arithmetic on whole
ones at a time, rearranging them — is a way of manipulating this one object, so
it pays to know exactly what it is.

A Python list is a bag of pointers: each element can be a different type, live
anywhere in memory, and even be another list of a different length. A tensor is
the opposite: **one block of memory holding elements that are all the same
type**, plus a small amount of metadata describing how to interpret that block.

By convention PyTorch is imported once per file as `import torch as t`. That
short alias is what the ARENA exercises use, so every `t.` below is the same
library you would import as `torch`.

```python
import torch as t

# A list can hold three different types at once.
print([type(item).__name__ for item in [1, "two", [3]]])

# A tensor holds one, for every element, and says which one.
a = t.tensor([[1, 2, 3], [4, 5, 6]])
print(a)
print("dtype of the whole block:", a.dtype)
```

That one block is also why some operations are nearly free. Re-describing how
the block should be read costs almost nothing, while producing a block of your
own costs a write of every element. Which operations are which — and how to
ask a tensor the question — is the concept after next, and this page does not
assume any of it.

Why this design? Because when every element is the same type and sits at a
predictable memory address, PyTorch can hand whole-tensor operations to fast
compiled kernels instead of interpreting Python code element by element. That is
the entire performance story — and the reason the idiomatic style you will learn
here avoids writing Python `for` loops over elements.

The general procedure for turning existing Python data into a tensor is
`t.tensor(data)`: it walks the (possibly nested) sequence, finds a common
element type, and copies the values into one block. Nesting of any depth goes
through the same procedure.

```python
cube = t.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("shape:", cube.shape)

# A cell ending in a bare expression prints its value, the way a notebook does.
cube * 10
```

Two ways of reading a tensor back out come up constantly from here on, so name
them now. **`t.equal(x, y)`** answers "same shape AND same values?" as one
bool — the whole-tensor comparison, which is what a check wants. **`x.item()`**
pulls a single element out as a plain Python number, and it refuses unless the
tensor holds exactly one; that refusal is the point, because a silent "first
element" would be a guess.

```python
same = t.tensor([[1, 2], [3, 4]])
also = t.tensor([[1, 2], [3, 4]])
print("t.equal ->", t.equal(same, also), "  one answer for the whole tensor")

one = t.tensor([7])
print("one.item() ->", one.item(), "as a", type(one.item()).__name__)
try:
    same.item()
except RuntimeError as exc:
    print("four elements ->", type(exc).__name__, "- item() wants exactly one")
```

## Worked example


The problem below asks you to turn a nested Python list into a 2-D tensor. This
example shows that move once. Start with the list — three inner lists of two
integers each — and hand it to `t.tensor`:

```python
import torch as t

rows = [[7, 8], [9, 10], [11, 12]]
a = t.tensor(rows)
print(a)
```

`t.tensor` walked the nesting and copied the values into one block. Nothing was
rounded, reordered, or dropped, and `.tolist()` reads them straight back out —
which is the check worth running whenever you are not sure a conversion did what
you meant:

```python
print("back to a list:", a.tolist())
assert a.tolist() == [[7, 8], [9, 10], [11, 12]]
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

### q532
Read the tensor's numbers back out as a plain nested Python list.

```python starter
import torch as t

def solve(rows):
    """Return the tensor's contents as a plain nested Python list."""
    a = t.tensor(rows)
    return a.tolist()
```

```python solution
import torch as t

def solve(rows):
    """Return the tensor's contents as a plain nested Python list."""
    a = t.tensor(rows)
    return a.tolist()
```

### q533
The tensor holds exactly one number. Hand that number back as a plain Python value.

```python starter
import torch as t

def solve(values):
    """Return the tensor's only element as a plain Python number."""
    a = t.tensor(values)
    return a.item()
```

```python solution
import torch as t

def solve(values):
    """Return the tensor's only element as a plain Python number."""
    a = t.tensor(values)
    return a.item()
```

### q534
One bool for the whole pair: same layout AND same numbers?

```python starter
import torch as t

def solve(rows_a, rows_b):
    """Return True when the two tensors match on shape AND values."""
    a = t.tensor(rows_a)
    b = t.tensor(rows_b)
    return t.equal(a, b)
```

```python solution
import torch as t

def solve(rows_a, rows_b):
    """Return True when the two tensors match on shape AND values."""
    a = t.tensor(rows_a)
    b = t.tensor(rows_b)
    return t.equal(a, b)
```

## Concept: nesting becomes axes — shape, ndim, numel


The **shape** is a tuple giving the length along each dimension (axis). A 3×4
matrix has `shape == (3, 4)`: axis 0 has length 3 (rows), axis 1 has length 4
(columns). Axis 0 is always the outermost nesting level, so a flat list becomes
1-D, a list of equal-length lists becomes 2-D, and a list of those becomes 3-D.

```python
import torch as t

grid = t.tensor([[1, 2, 3], [4, 5, 6]])
print("shape:", grid.shape)
print("axis 0 (rows)   :", grid.shape[0])
print("axis 1 (columns):", grid.shape[1])
```

Two smaller readings come off the same metadata:

- **`ndim`** — how many axes there are. This is the nesting depth, and it is an
  attribute, not a call.
- **`numel()`** — the total element count, which is the *product* of the shape.
  This is a method, so it needs the parentheses. For a 2×3 tensor `ndim` is 2
  and `numel()` is 6 — six numbers arranged as two rows, not two of anything.

Getting those two confused is the classic first-week error, and it is worth
fixing now: `ndim` counts axes, `numel()` counts numbers.

```python
# Four numbers, three nestings, three different answers for ndim — and the same
# answer for numel every time.
for data in ([1, 2, 3, 4], [[1, 2], [3, 4]], [[[1], [2]], [[3], [4]]]):
    x = t.tensor(data)
    print(tuple(x.shape), "ndim", x.ndim, "numel", x.numel())
```

## Worked example


The problem below asks for a tensor's number of axes and its element count,
in that order: `(ndim, numel)`. Build the tensor, then read those two pieces of
metadata directly:

```python
import torch as t

rows = [[1, 2, 3], [4, 5, 6]]
a = t.tensor(rows)

print("ndim:", a.ndim)
print("numel:", a.numel())
assert (a.ndim, a.numel()) == (2, 6)
```

Two nesting levels give `ndim == 2`. Six numbers give `numel() == 6`.
Tuple position matters: `(2, 6)` is correct; `(6, 2)` reverses the requested
order.

## Faded practice


### q482
Count the axes and the elements. Return them in this exact order:
`(ndim, numel)`.

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

### q537
Return the tensor's shape as a plain Python tuple of ints.

```python starter
import torch as t

def solve(rows):
    """Return the tensor's shape as a plain tuple of ints."""
    a = t.tensor(rows)
    return tuple(a.shape)
```

```python solution
import torch as t

def solve(rows):
    """Return the tensor's shape as a plain tuple of ints."""
    a = t.tensor(rows)
    return tuple(a.shape)
```

### q538
Return the length along axis 0 and the length along axis 1.

```python starter
import torch as t

def solve(rows):
    """Return (length along axis 0, length along axis 1)."""
    a = t.tensor(rows)
    return (a.shape[0], a.shape[1])
```

```python solution
import torch as t

def solve(rows):
    """Return (length along axis 0, length along axis 1)."""
    a = t.tensor(rows)
    return (a.shape[0], a.shape[1])
```

### q539
Return how many axes the tensor has, as a plain int.

```python starter
import torch as t

def solve(data):
    """Return the tensor's number of axes."""
    a = t.tensor(data)
    return a.ndim
```

```python solution
import torch as t

def solve(data):
    """Return the tensor's number of axes."""
    a = t.tensor(data)
    return a.ndim
```

### q540
Two counts that are not the same question: every number in the tensor, and the rows in the input.

```python starter
import torch as t

def solve(rows):
    """Return (total element count, number of outer rows)."""
    a = t.tensor(rows)
    return (a.numel(), len(rows))
```

```python solution
import torch as t

def solve(rows):
    """Return (total element count, number of outer rows)."""
    a = t.tensor(rows)
    return (a.numel(), len(rows))
```

### q541
One bool: does the tensor's shape equal the tuple you were handed?

```python starter
import torch as t

def solve(rows, wanted):
    """Return True when the tensor's shape equals `wanted`."""
    a = t.tensor(rows)
    return tuple(a.shape) == tuple(wanted)
```

```python solution
import torch as t

def solve(rows, wanted):
    """Return True when the tensor's shape equals `wanted`."""
    a = t.tensor(rows)
    return tuple(a.shape) == tuple(wanted)
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
gives you `torch.int64` instead.

```python
import torch as t

print(t.tensor([1, 2, 3]).dtype)
print(t.tensor([1, 2.5, 3]).dtype)
```

All-integer input becomes `torch.int64`. One float makes the whole tensor
`torch.float32`. You can also choose the dtype explicitly:

```python
print(t.tensor([1, 2, 3], dtype=t.float32).dtype)
```

Division shows why one dtype for the whole block matters. `a / 2` halves every
number and creates a new float tensor. It does not change `a`:

```python
a = t.tensor([2, 4, 6])
halved = a / 2

print(a.tolist())
print(halved.tolist())
print(halved.dtype)
```

`a = a / 2` would make that new float tensor, then rebind the name `a` to it.
`a /= 2` instead tries to write float results back into the original integer
tensor. PyTorch rejects that in-place mutation rather than silently rounding:

```python
a = t.tensor([2, 4, 6])

try:
    a /= 2
except RuntimeError as exc:
    print(type(exc).__name__)
```

Two tensors can hold the same numbers in the same layout and still disagree on
dtype. Shape and dtype are independent, and code that checks only one of them
is checking half the question.

## Worked example


The problem below asks you to compare two tensors on shape and dtype. Those are
two separate checks.

Start with two tensors built from the same numbers in the same layout:

```python
import torch as t

a = t.tensor([[20, 21, 22], [23, 24, 25]])
b = t.tensor([[30, 31, 32], [33, 34, 35]])

print(a.shape == b.shape)
print(a.dtype == b.dtype)
assert (a.shape == b.shape, a.dtype == b.dtype) == (True, True)
```

Both checks are `True`. Now change one number to a float. Layout stays same, so
shapes still match. Dtypes do not:

```python
c = t.tensor([[30, 31.5, 32], [33, 34, 35]])

print(a.shape == c.shape)
print(a.dtype == c.dtype)
assert (a.shape == c.shape, a.dtype == c.dtype) == (True, False)
```

Last, keep integer dtype but change layout. Dtypes match; shapes do not:

```python
d = t.tensor([[40, 41], [42, 43], [44, 45]])

print(a.shape == d.shape)
print(a.dtype == d.dtype)
assert (a.shape == d.shape, a.dtype == d.dtype) == (False, True)
```

`.shape` and `.dtype` are attributes; neither takes parentheses.

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

### q542
Return the name of the tensor's element type as a string.

```python starter
import torch as t

def solve(values):
    """Return str() of the tensor's dtype."""
    a = t.tensor(values)
    return str(a.dtype)
```

```python solution
import torch as t

def solve(values):
    """Return str() of the tensor's dtype."""
    a = t.tensor(values)
    return str(a.dtype)
```

### q543
One bool: is the whole block held as 64-bit integers?

```python starter
import torch as t

def solve(values):
    """Return True when the tensor's dtype is the 64-bit integer one."""
    a = t.tensor(values)
    return a.dtype == t.int64
```

```python solution
import torch as t

def solve(values):
    """Return True when the tensor's dtype is the 64-bit integer one."""
    a = t.tensor(values)
    return a.dtype == t.int64
```

### q544
Build the tensor as 32-bit floats rather than letting the input decide, then read it back as a list.

```python starter
import torch as t

def solve(values):
    """Return the values as a float32 tensor, read back as a list."""
    a = t.tensor(values, dtype=t.float32)
    return a.tolist()
```

```python solution
import torch as t

def solve(values):
    """Return the values as a float32 tensor, read back as a list."""
    a = t.tensor(values, dtype=t.float32)
    return a.tolist()
```

### q545
Two answers: is the block float32, and how many numbers are in it?

```python starter
import torch as t

def solve(values):
    """Return (is the block float32?, how many elements)."""
    a = t.tensor(values)
    return (a.dtype == t.float32, a.numel())
```

```python solution
import torch as t

def solve(values):
    """Return (is the block float32?, how many elements)."""
    a = t.tensor(values)
    return (a.dtype == t.float32, a.numel())
```

### q546
One bool: do the two tensors agree on element type?

```python starter
import torch as t

def solve(values_a, values_b):
    """Return True when the two tensors share a dtype."""
    a = t.tensor(values_a)
    b = t.tensor(values_b)
    return a.dtype == b.dtype
```

```python solution
import torch as t

def solve(values_a, values_b):
    """Return True when the two tensors share a dtype."""
    a = t.tensor(values_a)
    b = t.tensor(values_b)
    return a.dtype == b.dtype
```

## Solo practice

### q481
A flat list becomes a 1-D tensor, however many numbers are in it.

### q480
Build it, describe its shape, and say whether one float re-typed the block.

### q483
Name the element type and count the elements — two answers, one of which the input decides.

### q485
Three levels of nesting. Report the axes, the shape and the element count.

### q486
Inferred versus forced: build the same numbers twice and compare the element types.

### q547
A flat list, described three ways: axes, elements, shape.

### q548
Name the element type, and say whether it is the float one.

### q549
Three levels of nesting. Show that the shape multiplies out to the element count.

Multiplying a shape out is a plain Python loop over the tuple — nothing
tensor-specific about it. The element count is the answer that loop should
reach:

```python worked
import torch as t

a = t.tensor([[1, 2, 3], [4, 5, 6]])
shape = tuple(a.shape)

product = 1
for length in shape:
    product = product * length

print(shape, "multiplies out to", product, "and numel is", a.numel())
assert product == a.numel()
```

### q550
Pull out the single element — but only when there really is exactly one.

`.item()` refuses on anything but a one-element tensor, and that refusal is
the point — a silent "first element" would be a guess. Ask the element count
first and you never have to catch anything:

```python worked
import torch as t

for values in ([7], [1, 2, 3]):
    a = t.tensor(values)
    answer = a.item() if a.numel() == 1 else None
    print(values, "-> numel", a.numel(), "->", answer)
```

### q551
Agreeing on shape is half a question. Answer both halves.

### q554
Force an integer type onto floats and report what survived.

Forcing an integer type is not rounding — it truncates toward zero, and it
does it to every element at once because there is one type for the whole
block:

```python worked
import torch as t

a = t.tensor([1.9, -2.7, 3.2], dtype=t.int64)
print(str(a.dtype), a.tolist())
assert a.tolist() == [1, -2, 3]
```

### q555
What PyTorch infers versus what you asked for.

`t.tensor` reads the dtype off the data unless you tell it otherwise. Naming
the dtype up front is how you stop the input's spelling from deciding it:

```python worked
import torch as t

values = [1, 2, 3]
inferred = t.tensor(values)
forced = t.tensor(values, dtype=t.float32)

print("inferred:", str(inferred.dtype))
print("forced  :", str(forced.dtype))
assert inferred.dtype != forced.dtype
```

### q556
Same count of numbers, different arrangement — two separate bools.

### q557
Describe ONE row of a 2-D tensor.

Indexing a tensor with a single int hands back one slice along axis 0, with
that axis gone. A row of a 2-D tensor is therefore 1-D — same numbers, one
fewer axis:

```python worked
import torch as t

a = t.tensor([[1, 2, 3], [4, 5, 6]])
row = a[0]

print("a   :", tuple(a.shape), "ndim", a.ndim)
print("a[0]:", tuple(row.shape), "ndim", row.ndim)
assert row.ndim == a.ndim - 1
```

### q559
A one-element tensor is still 1-D. Show it.

## Integrated practice

### q560
The three answers that fully describe a tensor, in one tuple.

### q562
Shape, dtype and whole-tensor equality are three separate verdicts. The third is not implied by the other two.

### q563
Choose the element type from an argument, then report the tensor three ways.

### q564
Every piece of metadata for a triply nested input.

### q566
Count, value and element type — where the value only exists for a one-element tensor.

### q567
Does the tensor give the input back unchanged? It depends on the type you asked for.

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
