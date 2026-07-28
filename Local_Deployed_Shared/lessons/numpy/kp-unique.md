---
kc: numpy.unique
title: Distinct values — t.unique and friends
supporting: [numpy.boolean-masking, numpy.sorting]
new_syntax: []
faded: [14]
guided: [241]
independent: [102, 148, 79]
---

## Concept

**`t.unique(z)`** returns the distinct values of an array, **sorted
ascending** — deduplication and ordering in one call. Its real depth is in
the optional outputs, each answering a different question about the
duplicates it collapsed:

- **`return_counts=True`** — how many times does each distinct value occur?
  Returns `(values, counts)`, aligned by position. This is the histogram of
  the data's actual values.
- **`return_index=True`** — where did each distinct value FIRST appear?
  Returns indices into the original array (handy for order-of-first-appearance
  reconstructions).
- **`return_inverse=True`** — for each original element, which distinct value
  is it? (`values[inverse]` rebuilds the input — a factorization into
  vocabulary + codes.)

Two generalizations worth knowing now:

- **Rows as units**: `t.unique(z, dim=0)` deduplicates whole ROWS of a 2-D
  tensor (sorted lexicographically — first column, then second…). Without
  `dim=`, a 2-D input is flattened and you get distinct *scalars*.
- **Set operations between tensors**: PyTorch ships only the membership test,
  **`t.isin(a, b)`** — a boolean mask over `a`. There is no `intersect1d`,
  `union1d` or `setdiff1d`, and you do not need them: composing `unique` with
  `isin` covers the family.

  - intersection: `ua = t.unique(a); ua[t.isin(ua, b)]`
  - difference:   `ua = t.unique(a); ua[~t.isin(ua, b)]`
  - union:        `t.unique(t.cat([a, b]))`

The mental model: `unique` hands back a canonical (sorted, deduplicated)
form, and `isin` filters it — if a task says "each value exactly once,
ascending", it is describing this family.

## Worked example

Task: get the vocabulary of a measurement vector with occurrence counts, and
find which values two arrays share.

```python
import torch as t

z = t.tensor([3, 1, 2, 3, 1, 3])

# Distinct values, sorted — and their aligned counts.
values, counts = t.unique(z, return_counts=True)
assert values.tolist() == [1, 2, 3]
assert counts.tolist() == [2, 1, 3]      # counts[i] belongs to values[i]

# The pair answers "most common value" without a Python Counter:
assert values[counts.argmax()] == 3

# return_inverse: codes that rebuild the input from the vocabulary.
values2, inverse = t.unique(z, return_inverse=True)
assert values2[inverse].tolist() == z.tolist()

# Set intersection of two tensors: shared values, once each, ascending.
# unique gives the sorted vocabulary; isin filters it down to the shared part.
a = t.tensor([4, 1, 3, 2, 3])
b = t.tensor([3, 4, 4, 8, 0])
ua = t.unique(a)
assert ua[t.isin(ua, b)].tolist() == [3, 4]
```

Why each step:

1. `values`/`counts` alignment is positional — index i of each refers to the
   same distinct value. Downstream questions (most common, rarest, values
   occurring exactly once) are reductions over `counts` followed by indexing
   into `values`.
2. `values[counts.argmax()]` chains last KP's argmax with this KP's aligned
   arrays — this composition is the standard "mode" idiom for small unique
   sets.
3. `intersect1d` returning `[3, 4]` (not `[4, 3]`, not `[3, 4, 4]`)
   demonstrates the canonical-form contract: sorted, each value once —
   regardless of input order or multiplicity.

## Faded practice

### q14
Distinct values ascending, with aligned occurrence counts.

```python starter
import torch as t

def solve(z):
    """(distinct values ascending, counts aligned with them)."""
    return t.unique(z, _____=True)
```

```python solution
import torch as t

def solve(z):
    """(distinct values ascending, counts aligned with them)."""
    return t.unique(z, return_counts=True)
```

## Guided practice

### q241
1. Values appearing in BOTH arrays, each exactly once, ascending — that
   sentence is the contract of one set function.
2. The `*1d` family: intersect, union, setdiff. Which one?
3. Empty intersection falls out naturally as an empty array — no special
   case needed.

## Independent practice

From the drill bank: q102 (values + counts again, stated as a tuple contract),
q148 (indices of FIRST occurrence of each distinct value — which optional
output?), q79 (distinct ROWS, lexicographic — remember the axis keyword).

## Misconceptions

- **"unique preserves input order."** — It sorts. If you need
  order-of-first-appearance, combine `return_index=True` with a sort of those
  indices — the sorted-values default is the contract, not a coincidence.
- **"unique on a matrix gives unique rows."** — Without `axis=0` the array is
  flattened first and you get distinct scalars. Row-level deduplication is
  explicitly `t.unique(z, axis=0)`.
- **"Intersection = loop with `in`."** — `t.unique(a)` filtered by
  `t.isin(...)` (or `t.isin(a, b)` alone when you want a mask on `a`). The
  Python-level loop is quadratic and unvectorized.
