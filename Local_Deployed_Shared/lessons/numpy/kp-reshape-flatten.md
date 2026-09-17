---
kc: numpy.reshape-flatten
title: Reshape, flatten, and element order
supporting: [numpy.ndarray-model, numpy.ranges]
new_syntax: [Tensor.flatten, Tensor.reshape]
faded: [46, 36, 491, 619, 490, 620, 621, 622, 623, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]
guided: []
independent: [23, 492, 493, 494, 624, 625, 626, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1354, 1355, 1356]
integrated: [627, 628, 629, 1357, 1358, 1359, 1360, 1361, 1362]
---

## Concept: reshape re-describes the same run of numbers

A contiguous tensor stores its values consecutively in reading order; its
shape describes how to group that run into axes. **Reshaping preserves the
reading-order sequence while changing those groups.** For contiguous input,
this shares storage without copying values. Other layouts may require a copy.
Either way, the one hard rule is:

> the new shape must account for exactly the same number of elements
> (`2 × 6 = 12 = 3 × 4` ✓, but 12 → `(5, 3)` ✗ raises an error).

The spelling is **`x.reshape(shape)`**, and the shape can be given loose
(`x.reshape(3, 4)`) or as a tuple (`x.reshape((3, 4))`):

```python
import torch as t

x = t.arange(12)
print(x)
print(x.reshape(3, 4))
print("(2, 6) ", x.reshape(2, 6).shape)
print("(3, 4) ", x.reshape((3, 4)).shape)
# Hidden checks
assert _delta_output == 'tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])\ntensor([[ 0,  1,  2,  3],\n        [ 4,  5,  6,  7],\n        [ 8,  9, 10, 11]])\n(2, 6)  torch.Size([2, 6])\n(3, 4)  torch.Size([3, 4])\n'
```

The count rule is not a guideline. A mismatch raises rather than padding or
truncating:

```python
try:
    x.reshape(5, 3)                      # 15 != 12
except RuntimeError as err:
    print("RuntimeError:", err)
# Hidden checks
assert _delta_output.startswith("RuntimeError:")
assert "12" in _delta_output and "5, 3" in _delta_output
```

One dimension may be **`-1`**, meaning "work this one out for me":
`x.reshape(3, -1)` fixes three rows and lets PyTorch derive the columns. It
only ever *derives* a length — it cannot invent elements — and only one `-1`
is allowed per call.

```python
print("(3, -1) ->", tuple(x.reshape(3, -1).shape))
print("(-1, 2) ->", tuple(x.reshape(-1, 2).shape))
# Hidden checks
assert x.reshape(3, -1).shape == x.reshape(3, 4).shape
```

The fixed dimensions describe the groups you want to keep. For a tensor with
shape `(batch, rows, cols)`, keeping `batch` as the first length gives one long
row per batch; keeping `cols` as the last length gives one row per original
row. Both preserve the same flat sequence, but they answer different questions.
A dimension of length one adds a grouping level, not another value. Its
position determines which part of the result gets wrapped in another pair of
brackets. For example, `(1, 2, 3)` wraps one whole matrix, while `(2, 1, 3)`
wraps each row separately.

## Worked example

Task: build the classic "counting matrix" — an n×n tensor containing 0..n²-1
reading left-to-right, top-to-bottom.

```python
import torch as t

n = 3
# Step 1: make the flat run 0..8. It's 1-D — shape (9,).
flat = t.arange(n * n)

# Step 2: reshape to (3, 3). The first three numbers become row 0, the next
# three row 1 — exactly the "reading order" the task describes. No data
# is copied; only the note about the shape changed.
grid = flat.reshape(n, n)

# The -1 shortcut: "3 rows, you work out the columns."
print(grid)
# Hidden checks
assert grid.tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
assert grid.tolist() == t.arange(9).reshape(3, -1).tolist()
```

Why each step:

1. `arange` + `reshape` is the standard two-step for "matrix containing the
   numbers 0..k in reading order" — generate the flat values, then cut them.
2. `-1` earns its keep when one dimension is derived: you state the part you
   know and let PyTorch check the arithmetic.

## Faded practice

### q46
n×n matrix containing 0..n²-1 in reading order.

```python starter
import torch as t

def solve(n):
    """Return the n x n matrix of 0..n*n-1 in row-major reading order."""
    return t.arange(_____)._____(_____, _____)
```

```python solution
import torch as t

def solve(n):
    """Return the n x n matrix of 0..n*n-1 in row-major reading order."""
    return t.arange(n * n).reshape(n, n)
```

### q36
A 1-D run cut into a 3-D shape whose product matches its length.

```python starter
import torch as t

def solve(a, shape):
    """Return a re-described with the given 3-D shape."""
    return a._____(shape)
```

```python solution
import torch as t

def solve(a, shape):
    """Return a re-described with the given 3-D shape."""
    return a.reshape(shape)
```

### q491
You know the column count. Do not compute the rows — let -1 do it.

```python starter
import torch as t

def solve(x, cols):
    """Return x reshaped to `cols` columns, inferring the row count."""
    return x._____(_____, cols)
```

```python solution
import torch as t

def solve(x, cols):
    """Return x reshaped to `cols` columns, inferring the row count."""
    return x.reshape(-1, cols)
```

### q619
The same run of numbers, re-described as `rows` rows of `cols`.

```python starter
import torch as t

def solve(x, rows, cols):
    """Return x re-described as (rows, cols), as a plain nested list."""
    return x._____(rows, cols).tolist()
```

```python solution
import torch as t

def solve(x, rows, cols):
    """Return x re-described as (rows, cols), as a plain nested list."""
    return x.reshape(rows, cols).tolist()
```


### q1333
Infer the middle axis.

```python starter
import torch as t

def solve(x, rows, cols):
    return x._____(rows, _____, cols).tolist()
```

```python solution
import torch as t

def solve(x, rows, cols):
    return x.reshape(rows, -1, cols).tolist()
```

### q1334
One outer container per batch.

```python starter
import torch as t

def solve(x):
    return x._____(_____, x.shape[0], x.shape[1]).tolist()
```

```python solution
import torch as t

def solve(x):
    return x.reshape(1, x.shape[0], x.shape[1]).tolist()
```

### q1335
Keep each batch together.

```python starter
import torch as t

def solve(x):
    return x._____(x.shape[_____], _____).tolist()
```

```python solution
import torch as t

def solve(x):
    return x.reshape(x.shape[0], -1).tolist()
```

## Concept: flatten, and the order the numbers come out in

**`x.flatten()`** is the other direction: any shape back to one axis. Both
operations answer the same question — **in what order do elements fill a
shape?** — and PyTorch's answer is **row-major** ("C order"): the *last* axis
moves fastest. Reading a 2-D tensor in row-major order means walking across
row 0 left to right, then row 1, and so on.

```python
import torch as t

z = t.arange(6).reshape(2, 3)
print(z)
print("flatten ->", z.flatten())
# Hidden checks
assert z.flatten().tolist() == [0, 1, 2, 3, 4, 5]
```

The same rule explains higher-dimensional shapes: `reshape(2, 2, 3)` fills the
last axis (length 3) fastest and the first axis slowest.

```python
print(t.arange(12).reshape(2, 2, 3))
# Hidden checks
assert _delta_output == 'tensor([[[ 0,  1,  2],\n         [ 3,  4,  5]],\n\n        [[ 6,  7,  8],\n         [ 9, 10, 11]]])\n'
```

Read that print bottom-up: the innermost brackets (length 3) count by one, so
that axis moves fastest; the outermost changes only once.

Column-major ("Fortran") order — first axis fastest — has **no keyword** in
PyTorch. NumPy spells it `order='F'`; here you get it by swapping the axes
first and then flattening: `z.T.flatten()` reads the matrix column by column.

```python
print("row-major   ", z.flatten())
print("column-major", z.T.flatten())
# Hidden checks
assert z.T.flatten().tolist() == [0, 3, 1, 4, 2, 5]
```

Each original row occupies one interval in the flat run. With width `w`, row
`r` starts at position `r * w` and ends just before `(r + 1) * w`; its last
entry is at `(r + 1) * w - 1`. This explains why row starts are `w` positions
apart. Taking a slice before flattening selects groups, whereas taking it
after flattening selects individual values. Decide which unit the task names
before choosing where to index.

```python
letters_as_numbers = t.tensor([[14, 25, 36], [47, 58, 69]])
print("one original row:", letters_as_numbers[1:].flatten().tolist())
print("all but one entry:", letters_as_numbers.flatten()[1:].tolist())
# Hidden checks
assert letters_as_numbers[1:].flatten().tolist() == [47, 58, 69]
assert letters_as_numbers.flatten()[1:].tolist() == [25, 36, 47, 58, 69]
```

## Worked example

Task: take the counting matrix apart again, both ways.

```python
import torch as t

grid = t.arange(9).reshape(3, 3)

# Step 1: flatten undoes the reshape — the row-major walk gives 0..8 back.

# Step 2: the column-major walk reads DOWN each column instead. No order=
# keyword exists, so transpose first and let the row-major walk do the work.
print(grid)
print("row-major   ", grid.flatten())
print("column-major", grid.T.flatten())
# Hidden checks
assert grid.flatten().tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8]
assert grid.T.flatten().tolist() == [0, 3, 6, 1, 4, 7, 2, 5, 8]
```

Why: "flatten" is not one operation until you say the order. The default
matches how the matrix prints; the column-by-column read is a transpose away.

## Faded practice

### q490
Any shape in, exactly one axis out.

```python starter
import torch as t

def solve(x):
    """Return x collapsed to a single axis."""
    return x._____()
```

```python solution
import torch as t

def solve(x):
    """Return x collapsed to a single axis."""
    return x.flatten()
```

### q620
The k-th number in reading order — flatten, then index.

```python starter
import torch as t

def solve(x, k):
    """Return the k-th number of x in reading order (row-major)."""
    return x._____()[k].item()
```

```python solution
import torch as t

def solve(x, k):
    """Return the k-th number of x in reading order (row-major)."""
    return x.flatten()[k].item()
```

### q621
Two reads of one grid: across the rows, then down the columns.

```python starter
import torch as t

def solve(n, rows):
    """Return (row-major read, column-major read) of the counting matrix, as lists."""
    g = t.arange(n)._____(rows, -1)
    return (g._____().tolist(), g.T._____().tolist())
```

```python solution
import torch as t

def solve(n, rows):
    """Return (row-major read, column-major read) of the counting matrix, as lists."""
    g = t.arange(n).reshape(rows, -1)
    return (g.flatten().tolist(), g.T.flatten().tolist())
```


### q1336
Keep every second reading-order value.

```python starter
import torch as t

def solve(x):
    return x._____()[::_____].tolist()
```

```python solution
import torch as t

def solve(x):
    return x.flatten()[::2].tolist()
```

### q1337
Flatten one batch.

```python starter
import torch as t

def solve(x, k):
    return x[k]._____().tolist()
```

```python solution
import torch as t

def solve(x, k):
    return x[k].flatten().tolist()
```

### q1338
Find the end of the first row.

```python starter
import torch as t

def solve(x):
    return x._____()[x.shape[_____] - 1].item()
```

```python solution
import torch as t

def solve(x):
    return x.flatten()[x.shape[1] - 1].item()
```

## Concept: reshape is a view of the same memory

For contiguous input, reshape can describe the new shape using the original
storage — the result is a **view**. Writing into this view writes the original:

```python
import torch as t

base = t.zeros(6)
grid = base.reshape(2, 3)
grid[0, 0] = 9.0
print("grid:", grid)
print("base:", base)
# Hidden checks
assert base[0].item() == 9.0
```

That sharing is the reason this reshape avoids a copy, and it is also the thing to
remember when a "copy" turns out not to be one. When the run cannot be
re-described in place — a transposed tensor, say, whose numbers are not in
reading order — `reshape` quietly copies instead. You will also meet
**`x.view(shape)`**, reshape's stricter sibling: it *only* ever re-labels the
existing memory and raises if that is impossible. Prefer `reshape` unless you
specifically want the error.

A round trip changes nothing: flatten, then reshape back to the old shape, and
you have the same numbers in the same arrangement.

```python
z = t.arange(6).reshape(2, 3)
back = z.flatten().reshape(z.shape)
print(back)
# Hidden checks
assert t.equal(back, z)
```

Two different shapes can therefore describe the same changing data. A write
through either view appears through both, even if their row boundaries differ.
To record what the values were **before** a write, call `tolist()` before the
assignment: the resulting Python list contains copied values. Calling it
afterward records the changed values instead. Every new mutation drill here
specifies contiguous input, so the required sharing is guaranteed for its
reshapes; it does not rely on the behavior of a transposed or sliced layout.

```python
original = t.tensor([13, 24, 35, 46])
before = original.tolist()
paired = original.reshape(2, 2)
paired[0, 1] = 70
print("saved list:", before)
print("current run:", original.tolist())
# Hidden checks
assert before == [13, 24, 35, 46]
assert original.tolist() == [13, 70, 35, 46]
```

## Worked example

Task: prove the view shares memory, then prove the round trip is exact.

```python
import torch as t

x = t.tensor([[1, 2], [3, 4]])

# Step 1: a reshaped view of a tensor in reading order shares its memory.
flat = x.reshape(-1)
flat[0] = 99

# Step 2: the round trip — flatten, reshape back — is the same tensor.
again = x.flatten().reshape(x.shape)
print("x after the write:", x.tolist())
print("round trip equal: ", t.equal(again, x))
# Hidden checks
assert x.tolist() == [[99, 2], [3, 4]]      # the write landed in x
assert t.equal(again, x)
```

Why: step 1 is the difference between reshape and a copy, and it is what
makes a stray write into a "temporary" reshaped tensor show up in the original.

## Faded practice

### q622
Write through the reshaped view and read the original.

```python starter
import torch as t

def solve(x, v):
    """Reshape x to 1-D, write v into the LAST slot of the result, return x's values."""
    flat = x._____(-1)
    flat[-1] = v
    return x.tolist()
```

```python solution
import torch as t

def solve(x, v):
    """Reshape x to 1-D, write v into the LAST slot of the result, return x's values."""
    flat = x.reshape(-1)
    flat[-1] = v
    return x.tolist()
```

### q623
Flatten, then reshape back — the round trip changes nothing.

```python starter
import torch as t

def solve(x):
    """Flatten x and reshape it back; return (equal to x?, shape of the round trip)."""
    back = x._____()._____(x.shape)
    return (t.equal(back, x), tuple(back.shape))
```

```python solution
import torch as t

def solve(x):
    """Flatten x and reshape it back; return (equal to x?, shape of the round trip)."""
    back = x.flatten().reshape(x.shape)
    return (t.equal(back, x), tuple(back.shape))
```


### q1339
Change one row through a flat view.

```python starter
import torch as t

def solve(x, k, value):
    flat = x._____(-1)
    width = x.shape[1]
    flat[_____:(k + 1) * width] = value
    return x.tolist()
```

```python solution
import torch as t

def solve(x, k, value):
    flat = x.reshape(-1)
    width = x.shape[1]
    flat[k * width:(k + 1) * width] = value
    return x.tolist()
```

### q1340
Write through a matrix view.

```python starter
import torch as t

def solve(x, cols, value):
    matrix = x._____(-1, cols)
    matrix[_____, _____] = value
    return x.tolist()
```

```python solution
import torch as t

def solve(x, cols, value):
    matrix = x.reshape(-1, cols)
    matrix[0, -1] = value
    return x.tolist()
```

### q1341
Read the changed entry through another shape.

```python starter
import torch as t

def solve(x, k, value):
    matrix = x._____(x.shape[0], -1)
    matrix[_____, 0] = value
    return x.tolist()
```

```python solution
import torch as t

def solve(x, k, value):
    matrix = x.reshape(x.shape[0], -1)
    matrix[k, 0] = value
    return x.tolist()
```

## Solo practice

### q23
Every entry in COLUMN-major order — there is no `order=` keyword, so which
axis rearrangement gets you there?

### q492
Lay 0..n-1 out with a given row count — build the run, then reshape it.

### q493
The flattened values AND the shape they came from — everything needed to
rebuild the tensor.

### q494
Write into a reshaped result and see what happens to the original.

### q624
Any rank in, one axis out — and count the axes on both sides.

### q625
Reshape to the transposed SHAPE is not a transpose. Show both.

### q626
Three axes, two of them given — infer the last with -1.

-1 works in any position and at any rank: it stands for "the one length
that makes the count come out". Here it fills a THIRD axis:

```python worked
import torch as t

x = t.arange(12)

g = x.reshape(2, 2, -1)      # 12 = 2 * 2 * ?, so the last axis is 3
print(g)
print("shape:", tuple(g.shape))
# Hidden checks
assert tuple(g.shape) == (2, 2, 3)
```


### q1342
Gather one position from every batch.

### q1343
Preserve the final feature axis.

### q1344
Each number gets its own row.

### q1345
Insert an axis between rows and columns.

### q1346
Remove the batch grouping.

### q1347
Read one complete block.

### q1348
Inspect the first item in each block.

### q1349
Skip the first reading-order value.

### q1350
Read row starts in the flat run.

### q1351
Separate row ends from row starts.

### q1352
Write every row start through one axis.

### q1353
Write every row end through one axis.

### q1354
Rewrite one whole batch.

### q1355
Restore a saved matrix shape.

### q1356
Two valid layouts of the same run.

## Integrated practice

### q627
Build the counting grid, then describe it every way you know.

### q628
Flatten it, re-cut it into `cols` columns, and count the rows you got.

### q629
Three axes from one run, and back again.


### q1357
Edit a block and inspect both layouts.

### q1358
Regroup a batch and propagate a write.

### q1359
Overwrite a prefix crossing a row boundary.

### q1360
Preserve a snapshot before a view write.

### q1361
Compare regrouped shapes after a write.

### q1362
Snapshot one batch then change its first row.

## Misconceptions

- **"Reshape can rearrange values."** — Reshape never *reorders* values: the
  flat row-major sequence of elements is identical before and after, and only
  the shape metadata changes. (It may still copy that sequence into fresh
  memory when the input is a non-contiguous view — same order, new buffer.)
  If the values need to move (transpose, sort, flip), reshape is the wrong
  tool.
- **"reshape(3, 4) on 11 elements will pad or truncate."** — It raises a
  `RuntimeError`. Counts must match exactly; `-1` only *derives* a dimension,
  it can't invent elements.
- **"Flattening reads down the columns."** — The order is row-major: across
  row 0 first. Column-major means transposing first.
- **"`order='F'` works, like in NumPy."** — PyTorch's `flatten` takes no order
  argument at all. `z.T.flatten()` is the column-major read.
