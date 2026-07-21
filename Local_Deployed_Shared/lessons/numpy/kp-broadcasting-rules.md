---
kc: numpy.broadcasting-rules
title: Broadcasting rules
supporting: [numpy.elementwise-ufuncs, numpy.reshape-flatten]
new_syntax: [none-newaxis-indexing]
faded: [111, 151]
guided: []
independent: [60, 81]
---

## Concept: the right-alignment rule

Elementwise operations "require matching shapes" — except NumPy will
**stretch** certain mismatched shapes to fit, following one mechanical rule
set called **broadcasting**. It is the engine under half of idiomatic NumPy,
and it is fully predictable:

> **Align the two shapes from the RIGHT. For each axis pair, the sizes are
> compatible if they are equal, or if one of them is 1** (that axis gets
> conceptually copied to match). **A missing leading axis counts as 1.**

Work `(m, 1) + (1, n)` by hand: align → axis 0 is m vs 1 (stretch the 1 → m),
axis 1 is 1 vs n (stretch → n) — result shape `(m, n)`, where entry [i, j] =
`a[i, 0] + b[0, j]`. Every pairwise-combination table, distance matrix, and
outer product starts exactly like this.

The stretching is *virtual* — no copies are made; NumPy just reuses the
single row/column while iterating. Two everyday cases you have already been
using: scalar-with-array (`z * 2`) and matrix-with-row (`z - row` where row
has shape (n,) — aligned right, it matches z's last axis).

## Worked example

```python
import numpy as np

a = np.arange(3).reshape(3, 1)      # column: [[0], [1], [2]]           (3, 1)
b = np.arange(4).reshape(1, 4)      # row:    [[0, 1, 2, 3]]            (1, 4)

# Align right:  (3, 1)
#               (1, 4)
# axis 1: 1 vs 4 -> stretch a across columns; axis 0: 3 vs 1 -> stretch b
# down rows. Result (3, 4): the "addition table" of the two vectors.
table = a + b
assert table.shape == (3, 4)
assert table.tolist() == [[0, 1, 2, 3],
                          [1, 2, 3, 4],
                          [2, 3, 4, 5]]
```

Why: writing the two shapes one above the other, right-aligned, and
resolving each column IS the method — do it on paper until it's automatic.
The result shape falls out before any code runs.

## Faded practice

### q111
The broadcast sum of a column (m, 1) and a row (1, n).

```python starter
import numpy as np

def solve(a, b):
    """(m, n) table where entry [i, j] = a[i, 0] + b[0, j]."""
    return _____ + _____
```

```python solution
import numpy as np

def solve(a, b):
    """(m, n) table where entry [i, j] = a[i, 0] + b[0, j]."""
    return a + b
```

## Concept: placing the 1s yourself with None

The craft skill is **placing the 1s yourself**. Indexing with `None` (alias
`np.newaxis`) inserts a length-1 axis: `v[:, None]` turns shape (n,) into a
**column** (n, 1); `v[None, :]` makes an explicit **row** (1, n). When an
operation needs a vector to run *down* rather than *across* (or to hit a
specific axis of a 3-D array), you reshape it with `None` until the alignment
says what you mean.

When shapes are incompatible (say (3,) with (4,)), NumPy raises rather than
guessing — a broadcast error means your alignment is wrong, and the fix is
almost always a well-placed `None`. Never reshape at random until the error
goes away: work the right-alignment on paper, decide where the 1 belongs,
and place it deliberately.

## Worked example

```python
import numpy as np

# The addition table again, from FLAT vectors — we place the 1-axes:
va, vb = np.arange(3), np.arange(4)
table = va[:, None] + vb[None, :]
assert table.shape == (3, 4)

# 3-D case: image (h, w, c) scaled per-PIXEL by map (h, w).
# Align right: (2, 2, 3) vs (2, 2) -> trailing axes are 3 vs 2: INCOMPATIBLE.
# The map needs its stretch-axis at the END: scale[:, :, None] is (2, 2, 1).
img = np.ones((2, 2, 3))
scale = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
scaled = img * scale[:, :, None]
assert scaled.shape == (2, 2, 3)
assert scaled[1, 0].tolist() == [3.0, 3.0, 3.0]   # whole pixel scaled by 3
```

Why: the `va[:, None] + vb[None, :]` form is the general recipe for "all
pairs f(a_i, b_j)". In the 3-D case, the naive `img * scale` FAILS the
alignment check — working the rule shows the 1 must go at the end.

## Faded practice

### q151
Scale each pixel of an (h, w, c) image by a per-pixel (h, w) map.

```python starter
import numpy as np

def solve(a, b):
    """(h, w, c) image a scaled per-pixel by (h, w) map b."""
    return a * b[_____]
```

```python solution
import numpy as np

def solve(a, b):
    """(h, w, c) image a scaled per-pixel by (h, w) map b."""
    return a * b[:, :, None]
```

## Independent practice

From the drill bank: q60 (matrix whose every row is 0..cols-1 — one arange
plus broadcasting against a zeros column... or think about what `+` does),
q81 (a matrix whose rows are all copies of v — broadcasting or tile, then
consider which is cheaper).

## Misconceptions

- **"Broadcasting matches shapes from the left."** — From the RIGHT. `(3,)`
  against `(3, 4)` aligns 3-with-4 and fails; `(4,)` against `(3, 4)` aligns
  4-with-4 and works. Most surprise errors are left-alignment intuition.
- **"Stretching copies the data."** — The stretch is virtual; memory is
  reused, not duplicated. Broadcasting a (10000, 1) against (1, 10000) does
  NOT allocate 10⁸ intermediate elements for the inputs.
- **"When shapes don't broadcast, reshape until the error goes away."** —
  Random reshaping produces silently WRONG results more often than errors.
  Work the right-alignment on paper, decide where the 1 belongs, and place it
  with `None` deliberately.
