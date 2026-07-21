---
kc: numpy.slicing-views
title: Slicing, views, and slice assignment
supporting: [numpy.ndarray-model]
new_syntax: [slice-notation, multi-axis-indexing]
faded: [233]
guided: [76]
independent: [231, 75]
---

## Concept

Slicing is how you name a rectangular piece of an array. The syntax
generalizes Python's list slicing in two ways, and adds one semantic twist
that trips everyone at least once.

**Syntax.** A slice is `start:stop:step` (stop exclusive, any part omittable),
and a multi-dimensional array takes **one slice per axis, separated by
commas** inside a single pair of brackets:

- `x[2:5]` — elements 2, 3, 4 of a vector.
- `z[0, :]` — row 0, all columns. `z[:, -1]` — every row, last column.
- `x[::-1]` — the whole axis, stepped backwards: a reversed view.
- `z[:, ::-1]` — every row reversed (mirror left-right);
  `z[::-1, :]` — row order reversed (mirror top-bottom).

Negative indices count from the end (`-1` is the last element), exactly as in
Python.

**The twist: slices are *views*, not copies.** A slice doesn't copy data — it
is a new window onto the *same* memory block. Two consequences:

1. **Writing through a slice writes the original.** That enables the single
   most useful idiom in this KP, **slice assignment**:
   `x[start:stop] = value` sets a whole range at once (the scalar is
   broadcast to every selected position — no loop).
2. **"Return a new array" tasks need an explicit `.copy()`** if you would
   otherwise be returning or mutating a view of the caller's data. Rule of
   thumb: mutate → `.copy()` first, unless the task says to modify in place.

Some named helpers are just packaged slices: `np.rot90(z)` (rotate 90° CCW),
`np.flipud`/`np.fliplr` — all expressible with `::-1` and transposes.

## Worked example

Task: given a vector, produce a reversed copy; then blank out the middle of
another vector in place.

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0])

# Reverse = take the whole axis with step -1. This is a VIEW of x — no copy
# has happened yet, and building it costs nothing regardless of length.
rev = x[::-1]
assert rev.tolist() == [4.0, 3.0, 2.0, 1.0]

# Proof it's a view: writing through rev changes x itself.
rev[0] = 99.0                    # rev[0] is x[-1]!
assert x[-1] == 99.0
x[-1] = 4.0                      # undo

# The task said "the input must not be modified" — so hand back a copy.
safe = x[::-1].copy()
safe[0] = -1                     # touches only the copy
assert x[-1] == 4.0

# Slice ASSIGNMENT: set positions 1..3 (stop 4 exclusive) to 0 — in place,
# no loop. The scalar 0.0 is broadcast across the selected range.
y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
y[1:4] = 0.0
assert y.tolist() == [1.0, 0.0, 0.0, 0.0, 5.0, 6.0]
```

Why each step:

1. `[::-1]` reads as "everything, stepping backwards" — start/stop omitted,
   step −1. The same pattern reverses any single axis of any array; you pick
   *which* axis by which comma slot it sits in.
2. The view demonstration is the mental model to keep: a slice is a window,
   not a photocopy. Cheap to make, dangerous to mutate casually.
3. Slice assignment replaces the `for i in range(start, stop)` loop entirely —
   and it is the building block for border/checkerboard/striping patterns in
   the next lesson.

## Faded practice

### q233
Reversed copy of a 1-D array (input unmodified).

```python starter
import numpy as np

def solve(x):
    """Return a new array with x's elements in reverse order."""
    return x[_____]
```

```python solution
import numpy as np

def solve(x):
    """Return a new array with x's elements in reverse order."""
    return x[::-1]
```

## Guided practice

### q76
1. Two mirrors of a 2-D array: left-right (reverse within each row) and
   top-bottom (reverse the order of rows). Both are single slices.
2. In `z[rows, cols]` notation: which slot gets the `::-1` to reverse each
   row? Which to reverse the row order?
3. `z[:, ::-1]` and `z[::-1, :]` — check whether the task requires copies
   (it says the input must not be modified — are you modifying it, or just
   returning views?).

## Independent practice

From the drill bank: q231 (assign through a slice in place), q75 (rotate a
matrix 90° counterclockwise — either the
dedicated helper or a transpose composed with a flip).

## Misconceptions

- **"A slice is a copy."** — It's a view of the same memory. Mutating a slice
  mutates the original. When a task says "return a new array" or "do not
  modify the input" and you plan to write into the result, `.copy()` first.
- **"2-D indexing is `z[i][j]`."** — That works but chains two operations;
  the NumPy idiom is one bracket, comma-separated: `z[i, j]`, `z[i, :]`,
  `z[:, j]`. The chained form also breaks down for slice-then-assign patterns.
- **"Reversing needs a loop or `reversed()`."** — `[::-1]` on the relevant
  axis. Omitted start/stop with a negative step means "whole axis, backwards".
