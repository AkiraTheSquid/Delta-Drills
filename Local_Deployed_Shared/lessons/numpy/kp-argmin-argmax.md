---
kc: numpy.argmin-argmax
title: Locating extremes — argmin and argmax
supporting: [numpy.aggregations, numpy.slicing-views]
new_syntax: []
faded: [38]
guided: [219]
independent: [24, 61, 98]
---

## Concept

`min`/`max` tell you the extreme **value**; the `arg` twins tell you **where
it lives**:

- **`np.argmin(v)` / `v.argmin()`** — index of the smallest element.
- **`np.argmax(v)` / `v.argmax()`** — index of the largest.

Three properties do all the work in practice:

1. **Ties break to the first occurrence.** If the extreme value appears more
   than once, you get the smallest index. Many drills state "replace only the
   first occurrence" — argmin/argmax gives exactly that for free.
2. **The index is a handle for surgery.** "Replace the largest entry with 0"
   is: copy (if the input must survive), then `out[out.argmax()] = 0`. One
   read, one write, no scanning loop.
3. **They compose with transformed arrays.** The pattern
   `np.argmin(np.abs(z - target))` answers "which entry is *closest to*
   target?" — build the quantity you want minimized, then ask where its
   minimum sits. Any "closest / best / most-similar" task is this pattern
   with a different transform.

Like other reductions, the result is a NumPy scalar; wrap in `int(...)` when
a plain Python int is required. (Per-row/column argmax with `axis=` appears
in the broadcasting lesson — same idea, one axis at a time.)

## Worked example

Task: find the index of the smallest reading; then, without touching the
original, zero out the first occurrence of the maximum; then find which entry
is closest to a target value.

```python
import numpy as np

v = np.array([4.0, 2.0, 7.0, 2.0, 9.0])

# 1. Index of the minimum. 2.0 appears twice — argmin reports the FIRST.
i = int(np.argmin(v))
assert i == 1

# 2. Replace the max with 0 on a copy: the index is the handle.
out = v.copy()
out[out.argmax()] = 0.0          # argmax -> 4; out[4] = 0
assert out.tolist() == [4.0, 2.0, 7.0, 2.0, 0.0]
assert v.tolist() == [4.0, 2.0, 7.0, 2.0, 9.0]   # input intact

# 3. "Closest to target" = argmin of a transformed array.
target = 6.5
j = int(np.argmin(np.abs(v - target)))
assert j == 2                     # |7.0 - 6.5| = 0.5 is the smallest gap
closest_value = v[j]
assert closest_value == 7.0
```

Why each step:

1. `int(...)` at the boundary again — graders asking for "a plain Python int"
   reject `np.int64`.
2. The copy-then-assign order matters: index first or copy first both work
   here, but the habit "protect the input, then operate" reads cleanest and
   never backfires.
3. In step 3, note which array each part comes from: the *transformed* array
   (`np.abs(v - target)`) chooses the index; the *original* array supplies
   the value at that index. Keeping those roles straight is the whole trick.

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

## Guided practice

### q219
1. "Largest entry replaced with 0, first occurrence only" — you need the
   position of the max, not its value.
2. The input must keep its values — which array do you write into?
3. `result[np.argmax(x)] = 0.0` on a copy.

## Independent practice

From the drill bank: q24 (same surgery, stated slightly differently),
q61 (the VALUE closest to v — transform, argmin, then read the original),
q98 (the INDEX closest to target — same pattern, different return).

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
