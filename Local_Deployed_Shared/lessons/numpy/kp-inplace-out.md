---
kc: numpy.inplace-out
title: In-place operations and the out= argument
supporting: [numpy.elementwise-ufuncs, numpy.slicing-views, numpy.fancy-indexing]
new_syntax: []
faded: [235, 138, 59]
guided: []
independent: [65]
---

## Concept: in-place methods and operators

Most NumPy expressions allocate a fresh array per step. Usually fine — but
"in place" tasks (and memory-tight code) need the alternatives.

The first pair of tools: **in-place methods and augmented operators.**
`x.sort()` (vs `np.sort(x)`, which returns a sorted copy), `x += 1`,
`x *= 2` — these modify the existing buffer. The augmented operators (`+=`)
reuse memory where their spelled-out forms (`x = x + 1`) allocate and
rebind. Mind that in-place methods return `None`: `x = x.sort()` throws the
data away.

The mirror-image rule from the slicing KP still applies: "do not modify the
input" → copy first. This KP is the deliberate OPPOSITE — recognize which
contract a task states before choosing tools.

## Worked example

```python
import numpy as np

# In-place sort: the METHOD, and mind that it returns None.
x = np.array([3.0, 1.0, 2.0])
x.sort()
assert x.tolist() == [1.0, 2.0, 3.0]

# Augmented operator: same buffer, values doubled.
x *= 2
assert x.tolist() == [2.0, 4.0, 6.0]
```

Why: in-place = call and DON'T assign. New-array = `np.sort(x)`. The two
spellings answer two different task contracts.

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

## Concept: z[:] = expr — writing INTO the buffer

The subtle one: **whole-array in-place assignment.** `z = z[p]` REBINDS the
local name — the caller's array is untouched. `z[:] = z[p]` writes the
values INTO the existing buffer through a full-array slice, so every other
reference to that array sees the change. "Modify the array object passed
in" is this spelling. (Safe with `z[p]` on the right because fancy indexing
copies first.)

## Worked example

```python
import numpy as np

# In-place row permutation: write INTO the buffer via z[:].
z = np.arange(6).reshape(3, 2)
alias = z                        # a second reference to the same buffer
p = np.array([2, 0, 1])
z[:] = z[p]                      # rebinding (z = z[p]) would NOT affect alias
assert alias.tolist() == [[4, 5], [0, 1], [2, 3]]
```

Why: the `alias` variable is the proof of what "in place" means — both
names watch the same memory, so only the `z[:] =` spelling changes what
`alias` sees. This distinction is the entire point of the drill family.

## Faded practice

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

## Concept: out= — zero-allocation ufunc chains

Every ufunc accepts a destination: `np.add(a, b, out=b)` computes a+b and
stores it in b's buffer — zero new allocations. Chains of these
(`np.divide(a, 2, out=a)`, `np.negative(a, out=a)`, …) evaluate multi-step
formulas entirely within the input buffers.

Two disciplines make out= code correct: track *what each buffer now holds*
(after `np.add(a, b, out=b)`, the NAME b no longer means the original b),
and order by DATAFLOW — a value must be captured before the buffer holding
its ingredient is overwritten.

A related switch: `z.flags.writeable = False` freezes an array — any later
assignment raises `ValueError`. Useful for guarding shared data.

## Worked example

```python
import numpy as np

# ((a + b) * (-a / 2)) with out= only. Track buffer contents per step:
a = np.array([1.0, 2.0])
b = np.array([3.0, 4.0])
np.add(a, b, out=b)              # b now holds a + b
np.divide(a, 2.0, out=a)         # a now holds a / 2
np.negative(a, out=a)            # a now holds -a/2
np.multiply(a, b, out=a)         # a now holds (a+b) * (-a/2)
assert a.tolist() == [-2.0, -6.0]
```

Why: order matters — b must absorb (a+b) BEFORE a is halved, since the
addition needs the original a. Reordering the same four calls breaks the
result; dataflow, not formula layout, dictates sequence.

## Faded practice

### q59
((a + b) * (-a / 2)) with ZERO new allocations — every step lands in a's or
b's buffer.

```python starter
import numpy as np

def solve(a, b):
    """Compute ((a + b) * (-a / 2)) using only the two given buffers."""
    np.add(a, b, out=_____)
    np.divide(a, 2.0, out=a)
    np.negative(a, out=a)
    np.multiply(a, b, out=a)
    return a
```

```python solution
import numpy as np

def solve(a, b):
    """Compute ((a + b) * (-a / 2)) using only the two given buffers."""
    np.add(a, b, out=b)
    np.divide(a, 2.0, out=a)
    np.negative(a, out=a)
    np.multiply(a, b, out=a)
    return a
```

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
