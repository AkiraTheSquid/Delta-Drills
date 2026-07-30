---
kc: numpy.tile-repeat-meshgrid
title: Tiling and repetition — repeat, repeat_interleave, meshgrid
supporting: [numpy.reshape-flatten]
new_syntax: []
concepts: [repeat-elements, tile-blocks, coordinate-grids]
faded: [35, 34, 29]
guided: [217]
independent: [69, 155]
---

## Concept: Repeat each element with t.repeat_interleave

`t.repeat_interleave(x, k)` repeats each element of `x` before moving to the
next element.

```text
[1, 2, 3] → [1, 1, 1, 2, 2, 2, 3, 3, 3]
```

Use it when output groups copies of each individual value. `k` may also be a
tensor containing one repetition count per element.

## Watch out

Read expected output from left to right. If one value finishes all its copies
before the next value appears, use `repeat_interleave`.

This is the operation NumPy calls `repeat` — and PyTorch also has a `repeat`,
which does something else entirely. Getting the two names straight is the
whole point of this KP.

## Worked example: Repeat every reading three times

Task: repeat each element of `[4, 7, 9]` three consecutive times.

```python
import torch as t

x = t.tensor([4, 7, 9])
repeated = t.repeat_interleave(x, 3)

assert repeated.tolist() == [4, 4, 4, 7, 7, 7, 9, 9, 9]
print(repeated)
```

Why: `repeat_interleave` completes three copies of `4`, then three copies of
`7`, then three copies of `9`.

## Faded practice

### q35
Each element appears `k` times consecutively.

```python starter
import torch as t

def solve(x, k):
    """Each element of x, repeated k times consecutively."""
    return t._____(x, k)
```

```python solution
import torch as t

def solve(x, k):
    """Each element of x, repeated k times consecutively."""
    return t.repeat_interleave(x, k)
```

## Concept: Repeat a whole block with x.repeat

`x.repeat(k)` repeats `x` as one complete block.

```text
[1, 2, 3] → [1, 2, 3, 1, 2, 3]
```

For a 2-D tensor, pass one repetition count per axis:
`block.repeat(row_repeats, column_repeats)`. `t.tile` is a NumPy-compatible
alias for the same behaviour, and it takes the counts as a tuple.

## Watch out

**`x.repeat` is NOT NumPy's `repeat`.** It is NumPy's `tile`: it lays whole
copies end to end. The elementwise one is `repeat_interleave`. If you
translate `np.repeat(x, 3)` to `x.repeat(3)` you get the right length and the
wrong order — a bug that survives any test that only checks shape.

The repetition counts follow axis order. First number repeats rows; second
number repeats columns.

## Worked example: Build a checkerboard mosaic

Task: repeat one 2×2 checker block twice vertically and twice horizontally.

```python
import torch as t

block = t.tensor([[0, 1],
                  [1, 0]])
mosaic = block.repeat(2, 2)

assert mosaic.tolist() == [[0, 1, 0, 1],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [1, 0, 1, 0]]
print(mosaic)

# Same thing, NumPy-style spelling.
assert t.tile(block, (2, 2)).tolist() == mosaic.tolist()

# And the contrast that matters — same input, other operation:
assert t.repeat_interleave(t.tensor([1, 2, 3]), 2).tolist() == [1, 1, 2, 2, 3, 3]
assert t.tensor([1, 2, 3]).repeat(2).tolist() == [1, 2, 3, 1, 2, 3]
```

Why: `(2, 2)` lays out two copies along rows and two copies along columns.

## Faded practice

### q34
Repeat the entire sequence `k` times end to end.

```python starter
import torch as t

def solve(x, k):
    """The whole of x laid end-to-end k times."""
    return x._____(k)
```

```python solution
import torch as t

def solve(x, k):
    """The whole of x laid end-to-end k times."""
    return x.repeat(k)
```

## Concept: Build coordinate grids with t.meshgrid

`t.meshgrid(x, y, indexing='xy')` turns two 1-D coordinate axes into two 2-D
matrices.

- `X` repeats the `x` coordinates across every row.
- `Y` repeats each `y` coordinate down its matching row.

Pairing `X[i, j]` with `Y[i, j]` gives one point from every possible
combination of `x` and `y`.

## Watch out

`meshgrid` returns one coordinate matrix per input axis—not one tensor of
coordinate pairs.

**State the `indexing` argument.** With `indexing='xy'` the outputs have shape
`(len(y), len(x))`, matching NumPy. With `indexing='ij'` they come out
transposed, shape `(len(x), len(y))`. PyTorch will not guess for you quietly —
omitting the argument warns and uses `'ij'`, so NumPy code ported without it
silently transposes.

## Worked example: Enumerate a rectangular grid

Task: build coordinate matrices for three x-values and two y-values.

```python
import torch as t

x = t.tensor([1.0, 2.0, 3.0])
y = t.tensor([10.0, 20.0])
X, Y = t.meshgrid(x, y, indexing='xy')

assert X.tolist() == [[1.0, 2.0, 3.0],
                      [1.0, 2.0, 3.0]]
assert Y.tolist() == [[10.0, 10.0, 10.0],
                      [20.0, 20.0, 20.0]]
print(X)
print(Y)
```

Why: each column chooses an x-coordinate; each row chooses a y-coordinate.
Their matching positions enumerate the full grid.

## Faded practice

### q29
Return coordinate matrices built from 1-D tensors `x` and `y`.

```python starter
import torch as t

def solve(x, y):
    """Return the tuple (X, Y) of 2-D coordinate grids."""
    return t.meshgrid(x, y, indexing=_____)
```

```python solution
import torch as t

def solve(x, y):
    """Return the tuple (X, Y) of 2-D coordinate grids."""
    return t.meshgrid(x, y, indexing='xy')
```

## Guided practice

### q217
Tile a 2-D block `reps_r` times vertically and `reps_c` times horizontally.
One `block.repeat(reps_r, reps_c)` call does it — and note that this is the
*block* operation, not the per-element one.

## Independent practice

From the drill bank: q69 (pass a tensor of repetition counts to
`t.repeat_interleave`, so each input element may receive a different number of
copies).

Also from the bank: q155 (decode a run-length encoding — values repeated
by their counts, zero counts vanishing).

## Misconceptions

- **"`x.repeat(3)` repeats each element three times."** — That is NumPy's
  `repeat`. In PyTorch `.repeat` tiles whole copies; per-element repetition is
  `t.repeat_interleave`. The two produce the same *length* from the same
  input, so length checks won't catch the mix-up.
- **"`t.meshgrid(x, y)` matches `np.meshgrid(x, y)`."** — Only with
  `indexing='xy'`. The default is `'ij'`, which gives you the transpose.
