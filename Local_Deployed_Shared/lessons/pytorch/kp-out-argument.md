---
kc: torch.out-argument
title: Filling a tensor in place with out=
supporting: [torch.ranges, torch.slicing-views, torch.constructors]
new_syntax: [torch.arange#out, torch.linspace#out]
previews: []
faded: [798, 799]
guided: []
independent: [800, 801, 802, 803, 804, 805, 1598, 1599]
integrated: [806, 807, 808, 1600, 1674, 1675, 1676, 1677]
---

## Concept: out= writes into storage you already own

Every constructor you know so far **allocates** a fresh tensor and hands it
back. `t.linspace(0, 1, 5)` makes five new floats somewhere in memory. Often
that is exactly right. But when you have already built the tensor that will
hold the result — a canvas of zeros with the right shape — a fresh allocation
is one extra step: make the values, then copy them into place.

`out=` removes that step. Pass the **target** as `out=` and the function writes
its result **directly into that tensor's storage** instead of allocating:

```python
import torch as t

canvas = t.zeros((5, 3))
t.linspace(0, 1, 5, out=canvas[:, 1])
print(canvas)
# Hidden checks
assert _delta_output == 'tensor([[0.0000, 0.0000, 0.0000],\n        [0.0000, 0.2500, 0.0000],\n        [0.0000, 0.5000, 0.0000],\n        [0.0000, 0.7500, 0.0000],\n        [0.0000, 1.0000, 0.0000]])\n'
```

Three things to notice.

- The target was a **slice**, `canvas[:, 1]`. A slice is a view of the
  canvas, so writing into the view wrote into the canvas — column 1 is
  filled and the other columns are untouched.
- The **return value is the target itself**, not a new tensor. Catching it in
  a name gives you the same view back:

```python
import torch as t

canvas = t.zeros(4)
view = canvas[1:3]
filled = t.arange(2, out=view)
print(filled is view)
print(canvas)
# Hidden checks
assert _delta_output == 'True\ntensor([0., 0., 1., 0.])\n'
```

- The target's **shape must already match** what the call would produce.
  `t.linspace(0, 1, 5)` is five values, so the target must hold five. If it
  does not, PyTorch does **not** raise: it warns, **resizes the target** to a
  flat run of the canvas's own storage, and writes the values there — which
  is *not* the column you named. The numbers land in the wrong slots of the
  canvas, silently. Count the slots of the slice before you call:

```python
import torch as t

canvas = t.zeros((5, 3))
# 4 values into a 5-slot column
t.linspace(0, 1, 4, out=canvas[:, 1])
# the values sit across row 0..1, not down column 1
print(canvas)
# Hidden checks
assert canvas.shape == (5, 3)
assert t.allclose(canvas.flatten()[1:5], t.tensor([0., 1/3, 2/3, 1.]))
assert canvas.flatten()[5:].count_nonzero().item() == 0
assert canvas.flatten()[0].item() == 0
```

`out=` works the same on `t.arange`, `t.linspace`, and most other tensor
producing functions. The dtype follows the target: `t.arange(3, out=floats)`
writes `0., 1., 2.` because the target is float.

## Worked example

Build five y-coordinates as one column of a zero canvas, in one call:

```python
import torch as t

points = t.zeros((5, 3))
# column 1 = y
t.linspace(-1.0, 1.0, 5, out=points[:, 1])
print(points)
# Hidden checks
assert points[0, 1] == -1.0 and points[-1, 1] == 1.0
assert points[:, 0].sum() == 0.0            # other columns untouched
```

Input → output of the same idea, on a shorter canvas:

| call | canvas afterwards |
| --- | --- |
| `t.linspace(0, 1, 3, out=c[:, 0])` on `c = t.zeros((3, 2))` | `[[0, 0], [0.5, 0], [1, 0]]` |
| `t.arange(3, out=c[:, 1])` on the same `c` | `[[0, 0], [0.5, 1], [1, 2]]` |

## Faded practice

### q798
One call writes the column; the canvas is the thing you return.

```python starter
def solve(n, lo, hi, col):
    """n evenly spaced values from lo to hi, written into column col."""
    canvas = t.zeros((n, 3))
    t._____(lo, hi, n, _____=canvas[:, col])
    return canvas.tolist()
```

```python solution
def solve(n, lo, hi, col):
    """n evenly spaced values from lo to hi, written into column col."""
    canvas = t.zeros((n, 3))
    t.linspace(lo, hi, n, out=canvas[:, col])
    return canvas.tolist()
```

### q799
The target is float, so the integers land as 0., 1., 2.

```python starter
def solve(n, row):
    """The integers 0..n-1 written into row `row` of a 2×n canvas."""
    canvas = t.zeros((2, n))
    t._____(n, _____=canvas[row])
    return canvas.tolist()
```

```python solution
def solve(n, row):
    """The integers 0..n-1 written into row `row` of a 2×n canvas."""
    canvas = t.zeros((2, n))
    t.arange(n, out=canvas[row])
    return canvas.tolist()
```

## Solo practice

### q800
Two columns of one canvas, each filled where it lives.

### q801
out= hands back the target view; the canvas shows the write.

### q802
Three out= writes, three columns.

### q803
Only the top k entries of one column change.

### q804
An integer canvas; one row gets a run of consecutive integers.

### q805
One position of every slab, filled in a single write.

### q1598
out= targets a row this time — the last one.

### q1599
Which slice of the canvas is its last column, and can the values land there directly?

## Integrated practice

### q806
Every other row of one column: count how many slots that is.

### q807
A column partitioned between two out= writes.

### q808
A ones-column and a linspace-column, both via out=.

### q1600
Two out= writes into two rows of one canvas, then read one value back.

### q1674
Two ramps meeting at the middle: two writes into the canvas whose stretches share one cell.

### q1675
A two-row counting grid: one range written into each row, the second starting where the first stopped.

### q1676
A column ramp and a row count crossing: two writes into two views, the later one wins.

### q1677
A falling ramp into the tail of an existing tensor: the slice starts at n - k, the head survives.

## Misconceptions

- **"`out=` is just a faster way to get a return value."** The return value
  is the *target*. What changes is that the target's storage is written. If
  you throw the return value away, the write still happened.
- **"A shape mismatch raises."** It does not. PyTorch warns, resizes the
  target over the canvas's storage and writes the values into the wrong
  slots — a corrupted canvas, not an error. Check the slice's slot count
  against the number of values you pass.
- **"Row 0 is column 0."** `canvas[0]` is a row; `canvas[:, 0]` is a column.
  Both are valid `out=` targets and both are the wrong one half the time.
