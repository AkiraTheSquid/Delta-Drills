---
kc: numpy.slicing-views
title: Slicing, views, and slice assignment
supporting: [numpy.ndarray-model]
new_syntax: [slice-notation, multi-axis-indexing, syntax.slice, syntax.slice-step, torch.flip]
faded: [233]
guided: [76, 506]
independent: [231, 75, 507]
---

## Concept

Slicing is how you name a rectangular piece of a tensor. The syntax
generalizes Python's list slicing in two ways, and adds one semantic twist
that trips everyone at least once.

**Syntax.** A slice is `start:stop:step` (stop exclusive, any part omittable),
and a multi-dimensional tensor takes **one slice per axis, separated by
commas** inside a single pair of brackets:

- `x[2:5]` — elements 2, 3, 4 of a vector.
- `z[0, :]` — row 0, all columns. `z[:, -1]` — every row, last column.
- `x[::2]` — every second element.

Negative *indices* count from the end (`-1` is the last element), exactly as in
Python.

**Negative *steps* are the exception.** NumPy reverses an axis with `x[::-1]`;
PyTorch refuses — it raises `ValueError: step must be greater than zero`. This
is probably the single most common surprise when moving NumPy habits to torch.
Reversal has its own function:

- **`t.flip(x, [0])`** — reverse along axis 0. `t.flip(z, [1])` mirrors each
  row left-right; `t.flip(z, [0])` reverses the row order (mirror top-bottom).
- **`t.rot90(z)`** — rotate 90° counterclockwise, the composition of a
  transpose and a flip.

**The twist: slices are *views*, not copies.** A slice doesn't copy data — it
is a new window onto the *same* memory block. Two consequences:

1. **Writing through a slice writes the original.** That enables the single
   most useful idiom in this KP, **slice assignment**:
   `x[start:stop] = value` sets a whole range at once (the scalar is
   broadcast to every selected position — no loop).
2. **"Return a new tensor" tasks need an explicit `.clone()`** if you would
   otherwise be returning or mutating a view of the caller's data. Rule of
   thumb: mutate → `.clone()` first, unless the task says to modify in place.

`t.flip` is not in that category: it always returns a **copy**, so writing into
its result never touches the input.

## Worked example

Task: given a vector, produce a reversed copy; then blank out the middle of
another vector in place.

```python
import torch as t

x = t.tensor([1.0, 2.0, 3.0, 4.0])

# The NumPy reflex does not work here.
try:
    x[::-1]
    raised = False
except ValueError:
    raised = True
assert raised, "torch rejects negative slice steps"

# Reverse with flip instead — and flip hands back a COPY.
rev = t.flip(x, [0])
assert rev.tolist() == [4.0, 3.0, 2.0, 1.0]
rev[0] = 99.0                    # writes only into rev
assert x.tolist() == [1.0, 2.0, 3.0, 4.0]

# A plain slice, by contrast, IS a view: writing through it writes x.
window = x[1:3]
window[0] = 99.0                 # window[0] is x[1]!
assert x[1] == 99.0
x[1] = 2.0                       # undo

# Slice ASSIGNMENT: set positions 1..3 (stop 4 exclusive) to 0 — in place,
# no loop. The scalar 0.0 is broadcast across the selected range.
y = t.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
y[1:4] = 0.0
assert y.tolist() == [1.0, 0.0, 0.0, 0.0, 5.0, 6.0]
```

Why each step:

1. The failed `x[::-1]` is worth writing once deliberately. It is the fastest
   way to stop reaching for it by reflex later.
2. `t.flip(x, dims)` names the axes to reverse as a list — you pick *which*
   axis by what you put in that list, the same choice you would have made by
   which comma slot got the `::-1`.
3. The view demonstration is the mental model to keep: a slice is a window,
   not a photocopy. Cheap to make, dangerous to mutate casually.
4. Slice assignment replaces the `for i in range(start, stop)` loop entirely —
   and it is the building block for border/checkerboard/striping patterns in
   the next lesson.

## Faded practice

### q233
Reversed copy of a 1-D tensor (input unmodified).

```python starter
import torch as t

def solve(x):
    """Return a new tensor with x's elements in reverse order."""
    return t._____(x, [0])
```

```python solution
import torch as t

def solve(x):
    """Return a new tensor with x's elements in reverse order."""
    return t.flip(x, [0])
```

## Guided practice

### q76
1. Two mirrors of a 2-D tensor: left-right (reverse within each row) and
   top-bottom (reverse the order of rows). Both are single `t.flip` calls.
2. `t.flip` takes the axes to reverse as a list. Which axis number reverses
   each row? Which reverses the row order?
3. `t.flip(z, [1])` and `t.flip(z, [0])` — and because flip copies, the
   "input must not be modified" requirement is already satisfied.

### q506
1. A slice, not an index — you want a run of elements, not one element.
2. Leaving the start empty means 'from the beginning'.
3. `x[:k]`.

## Independent practice

From the drill bank: q231 (assign through a slice in place), q75 (rotate a
matrix 90° counterclockwise — either the dedicated helper or a transpose
composed with a flip).

From the drill bank: q507 (one column of a matrix — note which axis indexing with an int removes).

## Misconceptions

- **"`x[::-1]` reverses a tensor."** — It raises `ValueError: step must be
  greater than zero`. PyTorch supports no negative slice steps at all;
  `t.flip(x, [0])` is the operation.
- **"A slice is a copy."** — It's a view of the same memory. Mutating a slice
  mutates the original. When a task says "return a new tensor" or "do not
  modify the input" and you plan to write into the result, `.clone()` first.
- **"`.copy()` makes the copy."** — That's the NumPy name. Tensors clone with
  `.clone()`.
- **"2-D indexing is `z[i][j]`."** — That works but chains two operations;
  the idiom is one bracket, comma-separated: `z[i, j]`, `z[i, :]`, `z[:, j]`.
  The chained form also breaks down for slice-then-assign patterns.
