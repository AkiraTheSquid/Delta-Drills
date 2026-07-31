---
kc: numpy.pad-borders
title: Borders and padding
supporting: [numpy.slicing-views, numpy.constructors]
new_syntax: [torch.nn, torch.nn.functional, torch.nn.functional.pad, torch.nn.functional.pad#mode, torch.nn.functional.pad#value]
faded: [17]
guided: [27]
independent: [90, 186]
---

## Concept

Border tasks come in two mirror-image forms, each with its own tool:

**Growing — add a border AROUND the data: `t.nn.functional.pad`.**

> `t.nn.functional.pad(z, (1, 1, 1, 1), mode="constant", value=0)`

wraps `z` in a one-cell-thick frame of zeros: shape (r, c) → (r+2, c+2).

Mind the argument order — it is the one real trap on this page. The pad
tuple runs **last dimension first**, in (before, after) pairs:
`(left, right, top, bottom)` for a 2-D tensor. Pass a shorter tuple and only
the trailing dimensions get padded, which is a quiet way to pad columns and
wonder where your rows went. `value` sets the fill, and other modes
(`replicate`, `reflect`, `circular`) extend the data instead of a constant —
padding is the standard prelude to sliding-window and stencil operations,
where the window must not fall off the edge.

**Marking — modify the border WITHIN the existing shape: slice assignment.**
The frame of a matrix is four slices, or — the cleaner inverse — everything
*except* the frame is ONE slice, the interior `z[1:-1, 1:-1]`. So:

- ring of ones, hollow inside: start from `t.ones`, zero the interior:
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
import torch as t

z = t.ones((2, 2))

# (a) GROW: pad adds cells around the data. (2,2) -> (4,4).
framed = t.nn.functional.pad(z, (1, 1, 1, 1), mode="constant", value=0)
assert tuple(framed.shape) == (4, 4)
assert framed.tolist() == [[0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0]]

# (b) SAME SHAPE: build all-ones, then blank the interior with ONE slice.
n = 4
ring = t.ones((n, n))
ring[1:-1, 1:-1] = 0.0            # interior = rows 1..-2, cols 1..-2
assert ring.tolist() == [[1.0, 1.0, 1.0, 1.0],
                         [1.0, 0.0, 0.0, 1.0],
                         [1.0, 0.0, 0.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0]]

# The degenerate case is free: a 2x2 has no interior, so nothing changes.
small = t.ones((2, 2))
small[1:-1, 1:-1] = 0.0           # empty selection — assignment is a no-op
assert small.tolist() == [[1.0, 1.0], [1.0, 1.0]]
print("(a) padded — shape grew to", tuple(framed.shape))
print(framed)
print("(b) same shape, interior blanked")
print(ring)
print("(c) 2x2 has no interior, so nothing changed:", small.tolist())
```

Why each step:

1. In (a) note who supplies the border values: `pad` does — the original
   tensor is unchanged in the middle. Padding never mutates; it returns a
   new, larger tensor. The four 1s are (left, right, top, bottom).
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
import torch as t

def solve(z):
    """Surround z with a one-cell-thick border of zeros."""
    return t.nn.functional.pad(z, _____, mode="constant", value=_____)
```

```python solution
import torch as t

def solve(z):
    """Surround z with a one-cell-thick border of zeros."""
    return t.nn.functional.pad(z, (1, 1, 1, 1), mode="constant", value=0)
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

Also from the bank: q186 (an odd-sized window centred anywhere, with fill
wherever it hangs off the edge).

## Misconceptions

- **"pad modifies in place."** — It returns a new, bigger tensor (it must —
  the shape changes). The original is one of its ingredients, not its victim.
- **"The pad tuple is (top, bottom, left, right)."** — It is LAST DIMENSION
  FIRST: `(left, right, top, bottom)` for 2-D. Getting this backwards pads a
  valid-looking tensor with the frame on the wrong axis.
- **"The border needs four assignments."** — Assigning the border is four
  slices, but most frame tasks invert: fill everything, then overwrite the
  single interior slice `z[1:-1, 1:-1]`.
- **"Small matrices need an if-statement."** — `z[1:-1, 1:-1]` on a 1×n or
  2×n array selects nothing, and assigning to an empty selection is a legal
  no-op. The slice formulation handles the degenerate cases by construction.
