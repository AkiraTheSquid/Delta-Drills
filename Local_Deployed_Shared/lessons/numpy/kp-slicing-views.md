---
kc: numpy.slicing-views
title: Selecting and changing parts of a tensor
supporting: [numpy.ndarray-model]
new_syntax: [Tensor.clone, syntax.multi-axis-index, syntax.slice, syntax.slice-step, torch.flip, torch.rot90]
concepts: [slice-bounds, rows-and-columns, reversing, quarter-turns, writing-through-views, copying-before-writing]
faded: [506, 507, 233, 75, 231, 74]
guided: [76]
independent: [189]
---

## Concept: Choose a stretch of values

A slice selects a stretch: `x[start:stop]` includes `start`, stops **before** `stop`.
Predict the three numbers below, then run.

```python
import torch as t
x = t.tensor([10, 20, 30, 40, 50])
part = x[1:4]
print(part)
# Hidden checks
assert part.tolist() == [20, 30, 40]
```

Indices 1, 2, 3 give `20, 30, 40`. Index 4 is the boundary, so `50` stays out.
Leaving out `start` means “from the beginning.” What will `x[:2]` include?

```python
part = x[:2]
print(part)
# Hidden checks
assert part.tolist() == [10, 20]
```

Two values, at indices 0 and 1. A negative index counts from the end: `-1` names the last value.

## Worked example

Take the last two values by starting two places from the end:

```python
import torch as t
x = t.tensor([10, 20, 30, 40, 50])
tail = x[-2:]
print(tail)
# Hidden checks
assert tail.tolist() == [40, 50]
```

The missing stop means “through the end.” Before moving on, predict what `x[-3:]` would select.

## Faded practice

### q506
Return the first k values.

```python starter
import torch as t

def solve(x, k):
    """Return the first k elements."""
    return x[:_____]
```

```python solution
import torch as t

def solve(x, k):
    """Return the first k elements."""
    return x[:k]
```

## Concept: Select rows or columns

Inside brackets, the first slot chooses **rows**, the second chooses **columns**.
A colon on its own means “all.” First, look at the matrix:

```python
import torch as t
z = t.tensor([[1, 2, 3], [4, 5, 6]])
print(z)
# Hidden checks
assert z.tolist() == [[1, 2, 3], [4, 5, 6]]
```

Predict the middle column. Keep all rows, choose column 1:

```python
column = z[:, 1]
print(column)
# Hidden checks
assert column.tolist() == [2, 5]
```

The result is `[2, 5]`: one value from each row. An integer selects one column and removes that axis.

## Worked example

A slice keeps the column axis, even when it selects just one column:

```python
import torch as t
z = t.tensor([[1, 2, 3], [4, 5, 6]])
column = z[:, 1:2]
print(column)
# Hidden checks
assert column.tolist() == [[2], [5]]
assert tuple(column.shape) == (2, 1)
```

Notice the extra brackets: two rows, one column. What would `z[0, :]` select instead?

## Faded practice

### q507
Return one column and its shape.

```python starter
import torch as t

def solve(x, col):
    """Return one column of a 2-D tensor, and its shape."""
    c = x[_____, col]
    return (c.tolist(), tuple(c.shape))
```

```python solution
import torch as t

def solve(x, col):
    """Return one column of a 2-D tensor, and its shape."""
    c = x[:, col]
    return (c.tolist(), tuple(c.shape))
```

## Concept: Reverse along an axis

`t.flip` reverses the order along the axes you name. For a vector, its only axis is 0.
Predict which value moves to the front:

```python
import torch as t
x = t.tensor([1, 2, 3, 4])
reversed_x = t.flip(x, [0])
print(reversed_x)
# Hidden checks
assert reversed_x.tolist() == [4, 3, 2, 1]
assert x.tolist() == [1, 2, 3, 4]
```

The last value becomes the first. `flip` returns a new tensor; `x` stays unchanged.
PyTorch rejects negative slice steps such as `x[::-1]`; use `flip` for reversal.

## Worked example

For a matrix, axis 1 runs across each row. Predict this left-right mirror:

```python
import torch as t
z = t.tensor([[1, 2, 3], [4, 5, 6]])
mirrored = t.flip(z, [1])
print(mirrored)
# Hidden checks
assert mirrored.tolist() == [[3, 2, 1], [6, 5, 4]]
```

Each row reads backwards. Axis 0 would reverse the row order instead: `[4, 5, 6]` would be on top.

## Faded practice

### q233
Return a reversed copy of the vector.

```python starter
import torch as t

def solve(x):
    return t._____(x, [0])
```

```python solution
import torch as t

def solve(x):
    return t.flip(x, [0])
```

## Concept: Turn a matrix a quarter-turn

`t.rot90` turns a matrix 90° **counterclockwise**. Picture lifting its right edge upward.
Start with this small rectangle:

```python
import torch as t
z = t.tensor([[1, 2, 3], [4, 5, 6]])
print(z)
# Hidden checks
assert z.tolist() == [[1, 2, 3], [4, 5, 6]]
```

Before running, predict the new top row. The old rightmost column moves there:

```python
turned = t.rot90(z)
print(turned)
# Hidden checks
assert turned.tolist() == [[3, 6], [2, 5], [1, 4]]
```

`[3, 6]` is now on top. Two rows by three columns became three rows by two columns.

## Worked example

`k` counts quarter-turns. Predict where `1` ends up after two turns:

```python
import torch as t
z = t.tensor([[1, 2, 3], [4, 5, 6]])
turned = t.rot90(z, k=2)
print(turned)
# Hidden checks
assert turned.tolist() == [[6, 5, 4], [3, 2, 1]]
```

Two turns make 180°: `1` lands at the bottom right. Four turns return to the start; `k=-1` turns clockwise.

## Faded practice

### q75
Return a 90° counterclockwise rotation.

```python starter
import torch as t

def solve(z):
    return t._____(z)
```

```python solution
import torch as t

def solve(z):
    return t.rot90(z)
```

## Concept: Write through a slice

A slice is a **view**: another way to reach the same values. It does not copy them.
First, select the middle two values:

```python
import torch as t
x = t.tensor([10, 20, 30, 40])
window = x[1:3]
print(window)
# Hidden checks
assert window.tolist() == [20, 30]
```

`window[0]` and `x[1]` reach the same place. Predict which number changes in `x`:

```python
window[0] = 99
print(x)
# Hidden checks
assert x.tolist() == [10, 99, 30, 40]
```

Changing the view changed the original. This is useful when you intend to update a whole region.

## Worked example

Assign to a slice directly. Which positions will become zero?

```python
import torch as t
x = t.tensor([10, 20, 30, 40])
x[1:3] = 0
print(x)
# Hidden checks
assert x.tolist() == [10, 0, 0, 40]
```

Indices 1 and 2 change. Index 3 is outside the slice. One assignment fills every selected position.

## Faded practice

### q231
Fill the selected range in place.

```python starter
import torch as t

def solve(x, start, stop, value):
    x[_____:_____] = value
    return x
```

```python solution
import torch as t

def solve(x, start, stop, value):
    x[start:stop] = value
    return x
```

## Concept: Copy before changing values

When the original must stay unchanged, `.clone()` gives you separate storage.
We will mark every second value. The third slice slot is the step: `::2` visits indices 0, 2, 4.

```python
import torch as t
x = t.tensor([10, 20, 30, 40, 50])
selected = x[::2]
print(selected)
# Hidden checks
assert selected.tolist() == [10, 30, 50]
```

The step skips one value between each selection. Predict which entries would be selected by `x[1::2]`.

## Worked example

Make the copy first, then mark its selected positions:

```python
import torch as t
x = t.tensor([10, 20, 30, 40, 50])
marked = x.clone()
marked[::2] = -1
print(marked)
# Hidden checks
assert marked.tolist() == [-1, 20, -1, 40, -1]
```

Only the copy should change. Run the original to check that prediction:

```python
print(x)
# Hidden checks
assert x.tolist() == [10, 20, 30, 40, 50]
```

The original still holds all five values. Without `.clone()`, assigning `marked = x` would give the same tensor a second name.

## Faded practice

### q74
Replace every step-th value in a copy; preserve the input.

```python starter
import torch as t

def solve(z, step, v):
    out = z._____()
    out[::step] = v
    return out
```

```python solution
import torch as t

def solve(z, step, v):
    out = z.clone()
    out[::step] = v
    return out
```

## Guided practice

### q76
Mirror each row left-right, then mirror the row order top-bottom. Choose one axis for each result.

## Independent practice

From the drill bank: q189 (rotate a matrix by k quarter-turns).

## Misconceptions

- The stop is a boundary, not a selected index.
- Negative indices count from the end; negative slice steps are unsupported.
- Slices share storage. Clone before writing when the original must stay unchanged.
