---
kc: numpy.argmin-argmax
title: Locating extremes — argmin and argmax
supporting: [numpy.aggregations, numpy.slicing-views]
new_syntax: []
faded: [38, 219, 98]
guided: []
independent: [24, 61]
---

## Concept: argmin/argmax — the index, not the value

`min`/`max` tell you the extreme **value**; the `arg` twins tell you **where
it lives**:

- **`np.argmin(v)` / `v.argmin()`** — index of the smallest element.
- **`np.argmax(v)` / `v.argmax()`** — index of the largest.

**Ties break to the first occurrence.** If the extreme value appears more
than once, you get the smallest index — deterministically. Many drills state
"replace only the first occurrence"; argmin/argmax gives exactly that for
free.

Like other reductions, the result is a NumPy scalar; wrap in `int(...)` when
a plain Python int is required. (Per-row/column argmax with `axis=` appears
in the broadcasting lesson — same idea, one axis at a time.)

## Worked example

```python
import numpy as np

v = np.array([4.0, 2.0, 7.0, 2.0, 9.0])

# Index of the minimum. 2.0 appears twice — argmin reports the FIRST.
i = int(np.argmin(v))
assert i == 1

# The value at that index is the min itself.
assert v[i] == v.min()
```

Why: `int(...)` at the boundary again — graders asking for "a plain Python
int" reject `np.int64`. And `v[v.argmax()]` is how you get the value back
when you need both.

## Faded practice

### q38
Index of the smallest element (first occurrence on ties), as a plain int.

```python starter
import numpy as np

def solve(v):
    """Index of v's smallest element, first occurrence on ties."""
    return int(np._____(v))
```

```python solution
import numpy as np

def solve(v):
    """Index of v's smallest element, first occurrence on ties."""
    return int(np.argmin(v))
```

## Concept: the index as a handle for surgery

"Replace the largest entry with 0" is: copy (if the input must survive), then
`out[out.argmax()] = 0`. One read, one write, no scanning loop — the index is
a *handle* you use to edit the array.

The habit to build: **protect the input, then operate.** Copy first, then
assign through the index. It reads cleanest and never backfires.

## Worked example

```python
import numpy as np

v = np.array([4.0, 2.0, 7.0, 2.0, 9.0])

# Replace the max with 0 on a copy: the index is the handle.
out = v.copy()
out[out.argmax()] = 0.0          # argmax -> 4; out[4] = 0
assert out.tolist() == [4.0, 2.0, 7.0, 2.0, 0.0]
assert v.tolist() == [4.0, 2.0, 7.0, 2.0, 9.0]   # input intact
```

Why: the copy-then-assign order matters when the input must survive —
`v[v.argmax()] = 0` would mutate the caller's array.

## Faded practice

### q219
Largest entry replaced with 0 (first occurrence only), input untouched.

```python starter
import numpy as np

def solve(x):
    """x with its largest entry replaced by 0.0, without mutating x."""
    result = x._____()
    result[np.argmax(x)] = 0.0
    return result
```

```python solution
import numpy as np

def solve(x):
    """x with its largest entry replaced by 0.0, without mutating x."""
    result = x.copy()
    result[np.argmax(x)] = 0.0
    return result
```

## Concept: closest-to-target — argmin on a transformed array

argmin/argmax compose with transformed arrays. The pattern
`np.argmin(np.abs(z - target))` answers "which entry is *closest to*
target?" — build the quantity you want minimized, then ask where its minimum
sits. Any "closest / best / most-similar" task is this pattern with a
different transform.

Keep the roles straight: the *transformed* array chooses the index; the
*original* array supplies the value at that index.

## Worked example

```python
import numpy as np

v = np.array([4.0, 2.0, 7.0, 2.0, 9.0])

# "Closest to target" = argmin of a transformed array.
target = 6.5
j = int(np.argmin(np.abs(v - target)))
assert j == 2                     # |7.0 - 6.5| = 0.5 is the smallest gap
closest_value = v[j]
assert closest_value == 7.0
```

Why: no sorting needed — sorting is O(n log n) and loses positions;
`argmin(|v - t|)` is one pass and keeps them.

## Faded practice

### q98
The INDEX of the entry closest to target, as a plain int.

```python starter
import numpy as np

def solve(z, target):
    """Index of the entry of z closest to target."""
    return int(np.argmin(np._____(z - target)))
```

```python solution
import numpy as np

def solve(z, target):
    """Index of the entry of z closest to target."""
    return int(np.argmin(np.abs(z - target)))
```

## Independent practice

From the drill bank: q24 (the max-replacement surgery, stated slightly
differently), q61 (the VALUE closest to v — transform, argmin, then read the
original).

## Misconceptions

- **"argmax returns the maximum."** — It returns the *index* of the maximum.
  `v[v.argmax()]` is the value; needing both is common and costs one extra
  read.
- **"Ties are an error / unspecified."** — Ties resolve to the first
  occurrence, deterministically. Tasks that say "first occurrence" are
  describing argmin/argmax's default, not asking for extra work.
- **"Closest-to-target needs sorting."** — Sorting is O(n log n) and loses
  positions; `argmin(|v - t|)` is one pass and keeps them. Save sorting for
  when you need full order, not one winner.
