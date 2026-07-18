---
kc: numpy.pad-borders
title: Borders and padding
supporting: [numpy.slicing-views, numpy.constructors]
new_syntax: []
faded: [17]
guided: [27]
independent: [90]
---

## Concept

Border tasks come in two mirror-image forms, each with its own tool:

**Growing — add a border AROUND the data: `np.pad`.**

> `np.pad(z, pad_width=1, mode="constant", constant_values=0)`

wraps `z` in a one-cell-thick frame of zeros: shape (r, c) → (r+2, c+2).
`pad_width` can differ per side (`((top, bottom), (left, right))`),
`constant_values` sets the fill, and other modes (`edge`, `reflect`, `wrap`)
extend the data instead of a constant — `pad` is the standard prelude to
sliding-window and stencil operations, where the window must not fall off the
edge.

**Marking — modify the border WITHIN the existing shape: slice assignment.**
The frame of a matrix is four slices, or — the cleaner inverse — everything
*except* the frame is ONE slice, the interior `z[1:-1, 1:-1]`. So:

- ring of ones, hollow inside: start from `np.ones`, zero the interior:
  `z[1:-1, 1:-1] = 0`.
- overwrite the frame with a value: start from the data, assign the four
  edge slices `z[0, :]`, `z[-1, :]`, `z[:, 0]`, `z[:, -1]`.

The interior-slice trick has a built-in kindness: for 1×n or 2×n arrays,
`z[1:-1, 1:-1]` is an *empty* selection, so the assignment quietly does
nothing — exactly the right behavior when "everything is border", no special
case needed.

Choosing: does the output's shape GROW (pad) or stay the same (slice
assignment)? Read the task's shape contract first.

## Worked example

Task: (a) surround a matrix of ones with a zero border — shape grows;
(b) build a same-shape "picture frame": ones on the border, zeros inside.

```python
import numpy as np

z = np.ones((2, 2))

# (a) GROW: pad adds cells around the data. (2,2) -> (4,4).
framed = np.pad(z, pad_width=1, mode="constant", constant_values=0)
assert framed.shape == (4, 4)
assert framed.tolist() == [[0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0]]

# (b) SAME SHAPE: build all-ones, then blank the interior with ONE slice.
n = 4
ring = np.ones((n, n))
ring[1:-1, 1:-1] = 0.0            # interior = rows 1..-2, cols 1..-2
assert ring.tolist() == [[1.0, 1.0, 1.0, 1.0],
                         [1.0, 0.0, 0.0, 1.0],
                         [1.0, 0.0, 0.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0]]

# The degenerate case is free: a 2x2 has no interior, so nothing changes.
small = np.ones((2, 2))
small[1:-1, 1:-1] = 0.0           # empty selection — assignment is a no-op
assert small.tolist() == [[1.0, 1.0], [1.0, 1.0]]
```

Why each step:

1. In (a) note who supplies the border values: `pad` does — the original
   array is unchanged in the middle. Padding never mutates; it returns a new,
   larger array.
2. In (b) the insight is *invert the selection*: four border slices are
   fiddly, one interior slice is clean. `1:-1` reads "skip the first, skip
   the last" on each axis.
3. The 2×2 no-op is why the interior-slice formulation is preferred over
   explicit border slices in tasks that say "for n = 1 or 2 the whole matrix
   is border": empty slices make the edge cases disappear.

## Faded practice

### q17
Zero border AROUND the data: (r, c) → (r+2, c+2).

```python starter
import numpy as np

def solve(z):
    """Surround z with a one-cell-thick border of zeros."""
    return np.pad(z, pad_width=_____, mode="constant", constant_values=_____)
```

```python solution
import numpy as np

def solve(z):
    """Surround z with a one-cell-thick border of zeros."""
    return np.pad(z, pad_width=1, mode="constant", constant_values=0)
```

## Guided practice

### q27
1. Ones on the outer ring, zeros inside, same shape — grow or modify?
2. Build the ones first; the interior is a single two-axis slice.
3. `z[1:-1, 1:-1] = 0.0` — and check the task's promise about n=1, n=2
   against what an empty slice assignment does.

## Independent practice

From the drill bank: q90 (border of a FILL VALUE around existing data —
pad again, one keyword different).

## Misconceptions

- **"np.pad modifies in place."** — It returns a new, bigger array (it must —
  the shape changes). The original is one of its ingredients, not its victim.
- **"The border needs four assignments."** — Assigning the border is four
  slices, but most frame tasks invert: fill everything, then overwrite the
  single interior slice `z[1:-1, 1:-1]`.
- **"Small matrices need an if-statement."** — `z[1:-1, 1:-1]` on a 1×n or
  2×n array selects nothing, and assigning to an empty selection is a legal
  no-op. The slice formulation handles the degenerate cases by construction.
