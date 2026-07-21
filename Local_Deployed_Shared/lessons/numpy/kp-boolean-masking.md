---
kc: numpy.boolean-masking
title: Boolean masks — compare, count, filter
supporting: [numpy.slicing-views, numpy.elementwise-ufuncs, numpy.aggregations]
new_syntax: [boolean-mask-indexing]
faded: [236, 52, 12]
guided: []
independent: [232]
---

## Concept: the comparison IS the mask

A comparison applied to an array is itself elementwise: `x > 2.5` produces a
**boolean array** the same shape as `x` — `True` where the condition holds.
That boolean array is called a **mask**.

Saying "the comparison IS the mask" kills the urge to loop: there is no
separate "test each element" step to write. Divisibility, sign, range
membership, "equal to any of…" — anything you can phrase as an elementwise
condition becomes a mask in one expression.

## Worked example

```python
import numpy as np

x = np.array([1, 4, 6, 9, 3, 7])

# The comparison itself is the mask — same shape as x, dtype bool.
mask = x > 5
assert mask.tolist() == [False, False, True, True, False, True]

# Any elementwise condition works the same way, e.g. divisibility:
even = x % 2 == 0
assert even.tolist() == [False, True, True, False, False, False]
```

Why: one expression, no loop — the condition is written on the whole array
at once, and the result carries a True/False verdict per element.

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

## Concept: using a mask — count and filter

Once you have a mask, two of its three uses are read-only:

- **count / reduce**: `np.count_nonzero(mask)` or `mask.sum()` (how many? —
  True behaves as 1), `mask.any()` / `mask.all()` (yes/no questions). Wrap in
  `int(...)` when a plain Python int is required.
- **filter**: `x[mask]` returns a 1-D array of just the selected elements
  (a *copy*, unlike slices) — however many there are, shape not preserved.

## Worked example

```python
import numpy as np

x = np.array([1, 4, 6, 9, 3, 7])
mask = x > 5

# Count: True behaves as 1, so both spellings work.
assert np.count_nonzero(mask) == 3
assert mask.sum() == 3

# Filter: mask indexing keeps just the True positions (as a copy).
assert x[mask].tolist() == [6, 9, 7]
```

Why: `count_nonzero`/`sum` on a mask is the standard "how many satisfy…?";
`x[mask]` is the standard "give me those entries".

## Faded practice

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

## Concept: combining masks and masked assignment

Masks combine with `&` (and), `|` (or), `~` (not) — NOT Python's
`and`/`or`/`not`, which fail on arrays. Because `&`/`|` bind tighter than
comparisons, each comparison needs parentheses: `(x > 3) & (x < 8)`.

The third use of a mask is **assignment**: `x[mask] = value` (or
`x[mask] *= -1`) rewrites only the selected positions, *in place*. That
mutates the original array, so the usual contract applies: "do not modify
the input" ⇒ `.copy()` first, then assign through the mask on the copy.

## Worked example

```python
import numpy as np

x = np.array([1, 4, 6, 9, 3, 7])

# Combined condition + masked assignment, on a copy to protect x.
# Parentheses around EACH comparison are mandatory with & and |.
out = x.copy()
out[(out > 3) & (out < 8)] *= -1
assert out.tolist() == [1, -4, -6, 9, 3, -7]
assert x.tolist() == [1, 4, 6, 9, 3, 7]      # input untouched
```

Why: try removing the parentheses mentally: `out > 3 & out < 8` would
evaluate `3 & out` first (bitwise on ints!) — the precedence trap is why the
parenthesized form should become muscle memory.

## Faded practice

### q12
Entries strictly between 3 and 8 negated, input untouched.

```python starter
import numpy as np

def solve(z):
    """z with entries strictly between 3 and 8 negated (z unmodified)."""
    out = z.copy()
    out[(out > 3) _____ (out < 8)] *= -1
    return out
```

```python solution
import numpy as np

def solve(z):
    """z with entries strictly between 3 and 8 negated (z unmodified)."""
    out = z.copy()
    out[(out > 3) & (out < 8)] *= -1
    return out
```

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
