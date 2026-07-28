---
kc: numpy.nonzero-argwhere
title: Finding positions — nonzero and argwhere
supporting: [numpy.boolean-masking]
new_syntax: []
faded: [40]
guided: [86]
independent: [110]
---

## Concept

Masks answer "*which values* satisfy the condition?". Just as often you need
"*at which positions*?" — indices, not values. Two functions return them, in
two different layouts; choosing is a matter of what you'll do with the
coordinates:

PyTorch puts both layouts behind ONE function, `t.nonzero`, and a flag picks
between them:

- **`t.nonzero(z, as_tuple=True)`** returns a **tuple of index tensors, one
  per dimension**. For 1-D: a 1-tuple holding the positions. For 2-D:
  `(rows, cols)` — two parallel tensors where `(rows[i], cols[i])` is the
  i-th hit in row-major order. This layout plugs straight back into indexing:
  `z[t.nonzero(z, as_tuple=True)]` gives the nonzero values. Because a
  boolean is just 0/1, it works on any condition:
  `t.nonzero(x > 5, as_tuple=True)`.
- **`t.nonzero(z)`** — the DEFAULT — returns **one (k, n_dims) tensor of
  coordinate rows**, for 2-D k rows of `[row, col]`. This layout is for
  *reading* coordinates (iterate them, report them, save them); it does NOT
  plug back into indexing directly. `t.argwhere(z)` is a second name for
  exactly this.

Same information, transposed packaging. Note which way round the default
falls: bare `t.nonzero` gives coordinate ROWS, so the tuple form — the one
NumPy hands you by default — is the one you have to ask for.

Once you have per-axis index arrays, whole-array geometry questions become
min/max over them — e.g. the **bounding box** of the nonzero region of a 2-D
mask is `rows.min()..rows.max()` × `cols.min()..cols.max()`.

One wrinkle for plain Python inputs: these functions accept lists, but when a
task hands you a list and wants array semantics, convert explicitly with
`t.as_tensor(x)` first — it's free for tensors and makes intent visible.

## Worked example

Task: find where a vector is nonzero (the tuple form), then list the
(row, col) coordinates of every nonzero cell in a 2-D mask.

```python
import torch as t

x = t.as_tensor([0, 3, 0, 0, 7, 0, -2])

# as_tuple=True returns a TUPLE of index tensors — one per dim, so 1-D gives
# a 1-tuple. This is the same structure you get from t.where(cond).
pos = t.nonzero(x, as_tuple=True)
assert isinstance(pos, tuple) and len(pos) == 1
assert pos[0].tolist() == [1, 4, 6]

# The tuple layout plugs back into indexing: the nonzero VALUES.
assert x[pos].tolist() == [3, 7, -2]

# 2-D: the DEFAULT gives coordinate ROWS — one [row, col] pair per hit,
# in row-major scan order. Made for reading, not for indexing.
# t.nonzero(z) and t.argwhere(z) are the same call.
z = t.tensor([[0, 1],
              [1, 0]])
coords = t.argwhere(z)
assert coords.tolist() == [[0, 1], [1, 0]]

# Same info as parallel per-dim tensors:
rows, cols = t.nonzero(z, as_tuple=True)
assert rows.tolist() == [0, 1] and cols.tolist() == [1, 0]

# Geometry from index arrays: bounding box of the nonzero region.
assert (rows.min(), rows.max(), cols.min(), cols.max()) == (0, 1, 0, 1)
```

Why each step:

1. Checking `isinstance(pos, tuple)` once makes two things stick: the tuple
   layout is opt-in, and even one dimension still comes back wrapped.
   (Some drills require exactly this tuple structure.)
2. `x[pos]` closes the loop — positions in nonzero-format are *designed* to be
   used as indices.
3. The rows/cols unpacking plus min/max shows why the tuple layout wins for
   computation: each dimension's coordinates are already a vector you can
   reduce.

## Faded practice

### q40
Positions of all nonzero entries of a plain Python list, in the
tuple-of-index-tensors structure.

```python starter
import torch as t

def solve(x):
    """Return the tuple-of-index-tensors for nonzero positions of list x."""
    return t.nonzero(t.as_tensor(x), _____=True)
```

```python solution
import torch as t

def solve(x):
    """Return the tuple-of-index-tensors for nonzero positions of list x."""
    return t.nonzero(t.as_tensor(x), as_tuple=True)
```

## Guided practice

### q86
1. You need a (k, 2) tensor of [row, col] coordinates in scan order — which
   spelling produces coordinate rows rather than per-dim tensors?
2. Zero rows for an all-zero mask happens automatically — an empty result is
   shape (0, 2).
3. One call, no post-processing.

## Independent practice

From the drill bank: q110 (bounding box of the nonzero region as four plain
ints — get per-axis index arrays, then reduce each).

## Misconceptions

- **"`t.nonzero` behaves like NumPy's."** — It does not. NumPy's default is
  the per-axis tuple; PyTorch's default is coordinate rows (what NumPy calls
  argwhere). Passing `as_tuple=True` is what recovers the NumPy layout, and
  code that indexes with the default output breaks.
- **"For a 1-D tensor, as_tuple=True returns the indices directly."** — It
  returns a 1-TUPLE containing the index tensor. Unpack with `pos[0]` or
  `idx, = t.nonzero(x, as_tuple=True)` when you want the bare tensor.
- **"I need a loop to find positions matching a condition."** —
  `t.nonzero(condition, as_tuple=True)` does it: a mask is already the 0/1
  tensor nonzero scans.
