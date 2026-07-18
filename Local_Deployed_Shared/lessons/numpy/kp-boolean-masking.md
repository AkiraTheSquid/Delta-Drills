---
kc: numpy.boolean-masking
title: Boolean masks — compare, count, filter
supporting: [numpy.slicing-views, numpy.elementwise-ufuncs, numpy.aggregations]
new_syntax: [boolean-mask-indexing]
faded: [236, 52]
guided: [12]
independent: [232]
---

## Concept

A comparison applied to an array is itself elementwise: `x > 2.5` produces a
**boolean array** the same shape as `x` — `True` where the condition holds.
That boolean array is called a **mask**, and it is the pivot of a
three-step pattern behind almost every "find/count/change the entries
where…" task:

> **1. Build the mask** — write the condition on the whole array.
> **2. (Optionally) combine masks** — with `&` (and), `|` (or), `~` (not).
>    NOT Python's `and`/`or`/`not`, which fail on arrays. Because `&`/`|`
>    bind tighter than comparisons, each comparison needs parentheses:
>    `(x > 3) & (x < 8)`.
> **3. Use the mask**, one of three ways:
>    - **count / reduce**: `np.count_nonzero(mask)` or `mask.sum()` (how
>      many?), `mask.any()` / `mask.all()` (yes/no questions);
>    - **filter**: `x[mask]` returns a 1-D array of just the selected
>      elements (a *copy*, unlike slices);
>    - **assign**: `x[mask] = value` (or `x[mask] *= -1`) rewrites only the
>      selected positions, in place.

Divisibility, sign, range membership, "equal to any of…" — anything you can
phrase as an elementwise condition becomes a mask. Note that mask indexing
for *assignment* mutates the original array, so the usual contract applies:
"do not modify the input" ⇒ `.copy()` first, then assign through the mask on
the copy.

## Worked example

Task: given readings, mark which exceed a threshold, count them, pull them
out, and (on a copy) negate everything in the range (3, 8).

```python
import numpy as np

x = np.array([1, 4, 6, 9, 3, 7])

# 1. The comparison itself is the mask — same shape as x, dtype bool.
mask = x > 5
assert mask.tolist() == [False, False, True, True, False, True]

# 2a. Count: True behaves as 1, so both spellings work.
assert np.count_nonzero(mask) == 3
assert mask.sum() == 3

# 2b. Filter: mask indexing keeps just the True positions (as a copy).
assert x[mask].tolist() == [6, 9, 7]

# 3. Combined condition + masked assignment, on a copy to protect x.
#    Parentheses around EACH comparison are mandatory with & and |.
out = x.copy()
out[(out > 3) & (out < 8)] *= -1
assert out.tolist() == [1, -4, -6, 9, 3, -7]
assert x.tolist() == [1, 4, 6, 9, 3, 7]      # input untouched
```

Why each step:

1. Saying "the comparison IS the mask" kills the urge to loop: there is no
   separate "test each element" step to write.
2. `count_nonzero`/`sum` on a mask is the standard "how many satisfy…?" —
   and wrapping in `int(...)` hands back a plain Python int when required.
3. In the combined condition, try removing the parentheses mentally:
   `out > 3 & out < 8` would evaluate `3 & out` first (bitwise on ints!) —
   the precedence trap is why the parenthesized form should become muscle
   memory.

## Faded practice

### q236
Boolean array marking entries strictly greater than a threshold.

```python starter
import numpy as np

def solve(x, threshold):
    """True exactly where x exceeds threshold."""
    return x _____ threshold
```

```python solution
import numpy as np

def solve(x, threshold):
    """True exactly where x exceeds threshold."""
    return x > threshold
```

### q52
Number of True entries in a boolean array, as a plain int.

```python starter
import numpy as np

def solve(z):
    """Count the True entries of boolean array z."""
    return int(np._____(z))
```

```python solution
import numpy as np

def solve(z):
    """Count the True entries of boolean array z."""
    return int(np.count_nonzero(z))
```

## Guided practice

### q12
1. "Strictly greater than 3 AND strictly less than 8" — two comparisons
   combined. Which operator combines masks, and what punctuation does each
   comparison need?
2. The selected entries get negated in place via the mask:
   `arr[mask] *= -1`.
3. The input must survive unmodified — where does the `.copy()` go?

## Independent practice

From the drill bank: q232 (divisibility mask — the condition is a modulo
comparison).

## Misconceptions

- **"Combine conditions with `and`/`or`."** — Those are Python's short-circuit
  operators and raise on arrays. Masks combine with `&`, `|`, `~` — and each
  comparison must be parenthesized because `&` binds tighter than `>`.
- **"`x[mask]` keeps the array's shape."** — It returns a 1-D array of just
  the selected elements, however many there are. Only masked *assignment*
  leaves the shape intact.
- **"Counting Trues needs a loop or list.count."** — A mask is 0s and 1s:
  `mask.sum()` or `np.count_nonzero(mask)`. For yes/no rather than how-many,
  `any()`/`all()`.
