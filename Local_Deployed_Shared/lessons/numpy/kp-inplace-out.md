---
kc: numpy.inplace-out
title: In-place operations and the out= argument
supporting: [numpy.elementwise-ufuncs, numpy.slicing-views, numpy.fancy-indexing]
new_syntax: []
faded: [235, 138]
guided: [59]
independent: [65]
---

## Concept

Most NumPy expressions allocate a fresh array per step. Usually fine —
but "in place" tasks (and memory-tight code) need the alternatives, which
form a small toolkit:

- **In-place methods and operators.** `x.sort()` (vs `np.sort`),
  `x += 1`, `x *= 2` — these modify the existing buffer. The augmented
  operators (`+=`) reuse memory where their spelled-out forms
  (`x = x + 1`) allocate and rebind.
- **Whole-array in-place assignment: `z[:] = expr`.** The subtle one.
  `z = z[p]` REBINDS the local name — the caller's array is untouched.
  `z[:] = z[p]` writes the values INTO the existing buffer through a
  full-array slice, so every other reference to that array sees the change.
  "Modify the array object passed in" is this spelling. (Safe with `z[p]` on
  the right because fancy indexing copies first.)
- **`out=` on ufuncs.** Every ufunc accepts a destination:
  `np.add(a, b, out=b)` computes a+b and stores it in b's buffer — zero new
  allocations. Chains of these (`np.divide(a, 2, out=a)`,
  `np.negative(a, out=a)`, …) evaluate multi-step formulas entirely within
  the input buffers; each step's comment should track *what the buffer now
  holds*, because the names stop matching their original contents.
- **Write-protection.** `z.flags.writeable = False` freezes an array — any
  later assignment raises `ValueError`. Useful for guarding shared data
  (and the direct subject of one drill).

The mirror-image rule from the slicing KP still applies: "do not modify the
input" → copy first. This KP is the deliberate OPPOSITE — recognize which
contract a task states before choosing tools.

## Worked example

Task: sort an array in place; permute a matrix's rows in place; then
evaluate ((a + b) · (−a/2)) with no new allocations.

```python
import numpy as np

# 1. In-place sort: the METHOD, and mind that it returns None.
x = np.array([3.0, 1.0, 2.0])
x.sort()
assert x.tolist() == [1.0, 2.0, 3.0]

# 2. In-place row permutation: write INTO the buffer via z[:].
z = np.arange(6).reshape(3, 2)
alias = z                        # a second reference to the same buffer
p = np.array([2, 0, 1])
z[:] = z[p]                      # rebinding (z = z[p]) would NOT affect alias
assert alias.tolist() == [[4, 5], [0, 1], [2, 3]]

# 3. ((a + b) * (-a / 2)) with out= only. Track buffer contents per step:
a = np.array([1.0, 2.0])
b = np.array([3.0, 4.0])
np.add(a, b, out=b)              # b now holds a + b
np.divide(a, 2.0, out=a)         # a now holds a / 2
np.negative(a, out=a)            # a now holds -a/2
np.multiply(a, b, out=a)         # a now holds (a+b) * (-a/2)
assert a.tolist() == [-2.0, -6.0]

# 4. Freezing an array against writes:
frozen = np.zeros(3)
frozen.flags.writeable = False
try:
    frozen[0] = 1.0
    raised = False
except ValueError:
    raised = True
assert raised
```

Why each step:

1. The `alias` variable in step 2 is the proof of what "in place" means:
   both names watch the same memory, so only the `z[:] =` spelling changes
   what `alias` sees. This distinction is the entire point of the drill
   family.
2. In the out= chain, the comments tracking "what the buffer holds now" are
   not decoration — after `np.add(a, b, out=b)`, the NAME b no longer means
   the original b. Losing track of that is how out= code goes wrong.
3. Order matters in the chain: b must absorb (a+b) BEFORE a is halved, since
   the addition needs the original a. Reordering the same four calls breaks
   the result — dataflow, not formula layout, dictates sequence.

## Faded practice

### q235
Ascending order, in place — the passed-in object itself must change.

```python starter
import numpy as np

def solve(x):
    """Sort x itself (no new array), then return it."""
    x._____()
    return x
```

```python solution
import numpy as np

def solve(x):
    """Sort x itself (no new array), then return it."""
    x.sort()
    return x
```

### q138
Reorder rows by permutation p, in place.

```python starter
import numpy as np

def solve(z, p):
    """Row i becomes old row p[i] — INSIDE z's own buffer."""
    z_____ = z[p]
    return z
```

```python solution
import numpy as np

def solve(z, p):
    """Row i becomes old row p[i] — INSIDE z's own buffer."""
    z[:] = z[p]
    return z
```

## Guided practice

### q59
1. ((a + b) * (-a / 2)) with ZERO allocations — every step must land in a's
   or b's buffer via out=.
2. Dataflow first: which value must be captured before which buffer gets
   overwritten? (The sum needs the original a; the halving destroys it.)
3. add→(into b), divide→(into a), negative→(into a), multiply→(into a).
   Four calls, two buffers.

## Independent practice

From the drill bank: q65 (make an array refuse subsequent writes — one flag).

## Misconceptions

- **"`z = z[p]` modifies z in place."** — It rebinds the NAME; the original
  buffer (and every other reference to it) is unchanged. In-place is
  `z[:] = z[p]` — assignment through the full slice.
- **"`x = x.sort()` sorts and keeps the array."** — The method returns None;
  that assignment throws the data away. In-place: call and DON'T assign.
  New-array: `np.sort(x)`.
- **"out= is just an optimization hint."** — It's a hard contract: the
  result is written into that exact buffer. Aliasing consequences included —
  `np.add(a, b, out=b)` destroys the old b for all readers. Powerful,
  deliberate, and exactly what no-allocation drills demand.
