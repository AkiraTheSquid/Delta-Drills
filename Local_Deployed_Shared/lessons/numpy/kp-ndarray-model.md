---
kc: numpy.ndarray-model
title: What a tensor is — data + shape + dtype
supporting: []
new_syntax: [Tensor.T, Tensor.contiguous, Tensor.data_ptr, Tensor.dtype, Tensor.is_contiguous, Tensor.item, Tensor.ndim, Tensor.numel, Tensor.shape, Tensor.tolist, torch.equal, torch.float32, torch.int64, torch.tensor, torch.tensor#dtype]
faded: [224, 482, 484]
guided: [523]
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

A freshly built tensor lays those elements out contiguously. Operations that
only re-describe the block — transposing, slicing with a step — hand back a
tensor that *shares the same memory* in a different reading order. That is
cheap, and it is why `.contiguous()` exists: it is how you ask for the copy.

```python
at = a.T  # `a` is still defined — this cell continues the one above.

# Same numbers, same memory, read down the columns instead of along the rows.
print("shares storage with a:", at.data_ptr() == a.data_ptr())
print("still in reading order:", at.is_contiguous())

packed = at.contiguous()
print("the copy owns its memory:", packed.data_ptr() != a.data_ptr())
assert t.equal(packed, at)
```

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

Turn a nested Python list into a 2-D tensor:

```python
import torch as t

# A nested Python list: three inner lists of two integers each.
rows = [[7, 8], [9, 10], [11, 12]]

# t.tensor walks the nesting and copies the values into one block.
a = t.tensor(rows)

# The values survive the trip unchanged — .tolist() reads them back out.
assert a.tolist() == [[7, 8], [9, 10], [11, 12]]
print(a)
print("back to a list:", a.tolist())
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

```python
import torch as t

flat = t.tensor([4, 1, 7, 2, 9])
grid = t.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# One nesting level -> one axis. Five numbers in it.
assert flat.shape == (5,)
assert flat.ndim == 1
assert flat.numel() == 5

# Two nesting levels -> two axes, (rows, columns). numel is 3 * 4, not 3.
assert grid.shape == (3, 4)
assert grid.ndim == 2
assert grid.numel() == 12

# shape reports as torch.Size, which IS a tuple subclass — so it compares
# equal to a plain tuple, and tuple() converts it when you need the real thing.
assert tuple(grid.shape) == (3, 4)
print("flat: shape", flat.shape, "ndim", flat.ndim, "numel", flat.numel())
print("grid: shape", grid.shape, "ndim", grid.ndim, "numel", grid.numel())
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
gives you `torch.int64` instead.

```python
import torch as t

print(t.tensor([1, 2, 3]).dtype)
print(t.tensor([1, 2.5, 3]).dtype)   # one float decides the whole block

# Say what you want up front rather than relying on how the input is spelled.
print(t.tensor([1, 2, 3], dtype=t.float32).dtype)
```

Ordinary division still works on an integer tensor — `a / 2` quietly hands back
a *new* float tensor — but anything that has to write a float back into the
integer block does not. `a /= 2` raises rather than silently rounding, and the
error is the useful kind: it happens where the type is wrong, not three steps
later where the numbers are.

```python
ints = t.tensor([2, 4, 6])
print("ints / 2   ->", (ints / 2).dtype, "(a new tensor)")

try:
    ints /= 2
except RuntimeError as exc:
    print("ints /= 2  ->", type(exc).__name__)
```

Two tensors can hold the same numbers in the same layout and still disagree on
dtype. Shape and dtype are independent, and code that checks only one of them
is checking half the question.

## Worked example

The problem below asks you to compare two tensors on shape and on dtype and
report a `True`/`False` for each. The point of this example is to show that
those two questions are genuinely independent — two tensors can agree on one
and disagree on the other — and that the answer depends entirely on what you
feed in. So rather than one block of code, here are three pairs, each printed.

Start with two tensors built from the same numbers in the same layout:

```python
import torch as t

a = t.tensor([[1, 2], [3, 4]])
b = t.tensor([[5, 6], [7, 8]])

print("shapes:", a.shape, b.shape, "-> match?", a.shape == b.shape)
print("dtypes:", a.dtype, b.dtype, "-> match?", a.dtype == b.dtype)
assert (a.shape == b.shape, a.dtype == b.dtype) == (True, True)
```

That pair answers `(True, True)`. Now change ONE number to a float. Nothing
about the layout changed, so the shapes still agree — but a tensor holds one
type for every element, so that single `2.5` re-types the whole block:

```python
c = t.tensor([[1, 2.5], [3, 4]])

print("c is", c.dtype, "because of one float, not just that one entry")
print("shape match?", a.shape == c.shape, " dtype match?", a.dtype == c.dtype)
assert (a.shape == c.shape, a.dtype == c.dtype) == (True, False)
```

`(True, False)` — same answer to the shape question, opposite answer to the
dtype question. Now the other way round: keep the type and change the layout.

```python
d = t.tensor([[1, 2, 3], [4, 5, 6]])

print("a is", tuple(a.shape), "and d is", tuple(d.shape))
print("shape match?", a.shape == d.shape, " dtype match?", a.dtype == d.dtype)
assert (a.shape == d.shape, a.dtype == d.dtype) == (False, True)
```

Three inputs, three different answers: `(True, True)`, `(True, False)`,
`(False, True)`. That is why the problem gives you two arguments and asks for
two booleans — there is no single right answer to memorise, only two checks to
run. `.shape` is an attribute and `.dtype` is an attribute; neither takes
parentheses.

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

## Guided practice

### q523
```python worked
import torch as t

a = t.tensor([[1, 2], [3, 4], [5, 6]])

# .reshape re-describes the SAME block: 6 numbers, read 2 rows of 3 instead
# of 3 rows of 2. Nothing is copied, so both names read one buffer.
view = a.reshape(2, 3)
assert view.data_ptr() == a.data_ptr()

# .clone() is the opposite request: give me my own block.
copy = a.clone()
assert copy.data_ptr() != a.data_ptr()

# Sharing is about memory, not values — the copy holds the same numbers.
assert copy.tolist() == a.tolist()
print("view shares a's block:", view.data_ptr() == a.data_ptr())
print("copy shares a's block:", copy.data_ptr() == a.data_ptr())
print("view reads as:", view.tolist())
```

Your turn: the same three questions, but for the TRANSPOSE and for
`.contiguous()` rather than for `.reshape` and `.clone()`.

1. Two of the three answers are memory questions, not value questions: does
   the tensor you got back read the ORIGINAL block, or a fresh one?
   Transposing never moves data; `.contiguous()` exists precisely to ask for
   the move.
2. `a.T` is the transpose and `a.T.contiguous()` the packed copy;
   `.data_ptr()` reports which block each one reads.
3. `view = a.T`, `packed = view.contiguous()`, then return
   `(view.data_ptr() == a.data_ptr(), packed.data_ptr() == a.data_ptr(),
   packed.tolist())`.

## Applied practice

### q481
The problem below hands you a FLAT list and asks for the tensor plus its number
of axes. This example is here so you have seen the move once; the problem is the
same move on different data, and you write the whole function yourself.

```python worked
import torch as t

# One level of nesting -> one axis, however many numbers are in it.
flat = t.tensor([3, 1, 4, 1, 5])
print(flat, "-> ndim", flat.ndim)
assert flat.ndim == 1

# Adding a nesting level is what adds an axis. The COUNT of numbers does not.
nested = t.tensor([[3, 1], [4, 1]])
print(nested, "-> ndim", nested.ndim)
assert nested.ndim == 2
```

### q483
The problem below asks for the dtype's NAME as a string and the element count.
Two inputs here, because the dtype depends on what is in the list — that is the
whole point of the question.

```python worked
import torch as t

# All integers -> torch.int64. str() is what turns the dtype into its name.
ints = t.tensor([2, 4, 6, 8])
print(str(ints.dtype), ints.numel())
assert (str(ints.dtype), ints.numel()) == ("torch.int64", 4)

# One float re-types the whole block, and numel is unchanged by that.
mixed = t.tensor([2, 4.5, 6, 8])
print(str(mixed.dtype), mixed.numel())
assert (str(mixed.dtype), mixed.numel()) == ("torch.float32", 4)
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
