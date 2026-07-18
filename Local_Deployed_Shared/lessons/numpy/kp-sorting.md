---
kc: numpy.sorting
title: Sorting arrays
supporting: [numpy.ndarray-model, numpy.slicing-views]
new_syntax: []
faded: [58]
guided: []
independent: []
---

## Concept

Sorting in NumPy comes in two forms, and the difference is who gets modified:

- **`np.sort(z)`** — the function. Returns a **new** sorted array; `z` is
  untouched. This is what "return a new array, leave the input unmodified"
  tasks want.
- **`z.sort()`** — the method. Sorts **in place**, returns `None` (a classic
  trap: `z = z.sort()` leaves you holding `None`). This is what "rearrange the
  array object itself" tasks want.

Both sort ascending. There is no `descending=` flag — the idiom for descending
order is to sort ascending and reverse with the slice you already know:
`np.sort(z)[::-1]`.

Two related tools to know about now (each gets real coverage later):

- `np.argsort(z)` returns the *indices* that would sort `z` — the key to
  "sort one thing by another" tasks (order-statistics KP).
- On 2-D arrays, `np.sort(z, axis=...)` sorts each row or column
  independently — mind that this scrambles rows as units; reordering whole
  rows is argsort + fancy indexing, not `sort`.

## Worked example

Task: produce a sorted copy of a vector, confirm the original is intact, then
get the same values descending.

```python
import numpy as np

z = np.array([0.4, 0.1, 0.9])

# The FUNCTION returns a new array...
asc = np.sort(z)
assert asc.tolist() == [0.1, 0.4, 0.9]
# ...and the input keeps its original order (the grader often checks this).
assert z.tolist() == [0.4, 0.1, 0.9]

# Descending = ascending + reverse. No keyword exists for it.
desc = np.sort(z)[::-1]
assert desc.tolist() == [0.9, 0.4, 0.1]

# The METHOD sorts in place and returns None — never assign its result.
w = np.array([3.0, 1.0, 2.0])
result = w.sort()
assert result is None
assert w.tolist() == [1.0, 2.0, 3.0]   # w itself was reordered
```

Why each step:

1. Function vs method is a *contract* question, not a style question: read the
   task for "new array" vs "in place" and pick accordingly.
2. The descending idiom reuses `[::-1]` — one more payoff of the slicing KP.
3. The `result is None` check is worth seeing once, because `z = z.sort()`
   silently destroys your data reference and is a genuinely common bug.

## Faded practice

### q58
Sorted copy, smallest to largest, input left unmodified.

```python starter
import numpy as np

def solve(z):
    """Return a NEW array with z's values in ascending order."""
    return np._____(z)
```

```python solution
import numpy as np

def solve(z):
    """Return a NEW array with z's values in ascending order."""
    return np.sort(z)
```

## Misconceptions

- **"`z.sort()` returns the sorted array."** — It returns `None` and sorts
  `z` in place. If you need a sorted copy, use the function `np.sort(z)`.
- **"There's a reverse/descending flag."** — There isn't; use
  `np.sort(z)[::-1]`.
- **"Sorting a 2-D array sorts the rows as units."** — `np.sort(z, axis=1)`
  sorts *within* each row independently, destroying row integrity. Keeping
  rows intact while reordering them is an argsort + indexing pattern (later
  KP).
