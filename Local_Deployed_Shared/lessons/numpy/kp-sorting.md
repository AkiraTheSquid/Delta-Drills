---
kc: numpy.sorting
title: Sorting tensors
supporting: [numpy.ndarray-model, numpy.slicing-views]
new_syntax: []
faded: [58]
guided: []
independent: []
---

## Concept

**`t.sort(z)` does not return a sorted tensor.** It returns a *pair* — the
sorted values and the indices that produced them — and forgetting that is the
mistake this KP exists to prevent:

```python no-run
t.sort(z)          # -> torch.return_types.sort(values=..., indices=...)
t.sort(z).values   # the sorted tensor you actually wanted
```

You can unpack it either way: `values, indices = t.sort(z)`, or reach for
`.values` / `.indices` by name. The indices half is not a consolation prize —
it is what "sort one thing by another" tasks need, and it is the same thing
`t.argsort(z)` gives you on its own.

Sorting never modifies the input: `t.sort(z)` and the method form `z.sort()`
both leave `z` alone and hand back a new pair. (There is no in-place `sort_`.)

Two more things worth knowing now:

- **`descending=True`** sorts largest-first — a real keyword, unlike NumPy,
  where you have to sort ascending and reverse afterwards.
- On 2-D tensors, **`dim=`** sorts each row or column independently — mind
  that this scrambles rows as units; reordering whole rows is argsort plus
  fancy indexing, not `sort`.

## Worked example

Task: produce a sorted copy of a vector, confirm the original is intact, then
get the same values descending.

```python
import torch as t

z = t.tensor([0.5, 0.25, 0.75])

# sort returns a PAIR — take .values for the sorted tensor.
result = t.sort(z)
assert result.values.tolist() == [0.25, 0.5, 0.75]
assert result.indices.tolist() == [1, 0, 2]     # where each value came from

# The input keeps its original order (the grader often checks this).
assert z.tolist() == [0.5, 0.25, 0.75]

# Descending is a keyword here — no reverse step needed.
desc = t.sort(z, descending=True).values
assert desc.tolist() == [0.75, 0.5, 0.25]

# Unpacking works too, and reads well when you want both halves.
values, indices = t.sort(z)
assert values.tolist() == [0.25, 0.5, 0.75]
```

Why each step:

1. Taking `.values` is the habit to build. Returning `t.sort(z)` straight from
   a function hands the caller a pair, and the failure looks like a type error
   far from the line that caused it.
2. The indices are the bridge to the order-statistics KP: they say *where*
   each sorted value came from, which is how you carry a second tensor along.
3. `descending=True` is one of the places PyTorch is friendlier than NumPy —
   worth knowing so you don't write the reverse-slice workaround (which
   wouldn't work here anyway, since negative slice steps are rejected).

## Faded practice

### q58
Sorted copy, smallest to largest, input left unmodified.

```python starter
import torch as t

def solve(z):
    """Return a NEW tensor with z's values in ascending order."""
    return t.sort(z)._____
```

```python solution
import torch as t

def solve(z):
    """Return a NEW tensor with z's values in ascending order."""
    return t.sort(z).values
```

## Misconceptions

- **"`t.sort(z)` returns the sorted tensor."** — It returns a
  `(values, indices)` pair. Take `.values`, or unpack both.
- **"`z.sort()` sorts in place, like the NumPy method."** — It does not. There
  is no in-place sort; the input is never modified.
- **"There's no descending option, so I'll reverse the result."** — There is:
  `descending=True`. And the NumPy reversal idiom `[::-1]` would raise here
  anyway.
- **"Sorting a 2-D tensor sorts the rows as units."** — `t.sort(z, dim=1)`
  sorts *within* each row independently, destroying row integrity. Keeping
  rows intact while reordering them is an argsort + indexing pattern (later
  KP).
