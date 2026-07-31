---
kc: numpy.argsort-ranking
title: Order statistics — argsort, ranks, sort-by
supporting: [numpy.sorting, numpy.fancy-indexing]
new_syntax: []
faded: [92, 114, 119]
guided: []
independent: [183]
---

## Concept: sort-by — argsort as a row order

`t.sort` rearranges values. **`t.argsort` returns the indices that WOULD
sort them** — and that indirection is the tool.

First use: **sort one thing by another ("sort-by")**. To reorder a matrix's
ROWS so that column k comes out ascending: `z[z[:, k].argsort()]`. Read it
inside out — argsort of the key column gives the row order; fancy indexing
applies that order to whole rows. The rows travel intact; only their sequence
changes. Any "sort records by field" task is this two-step.

Note what is NOT happening: no row contents are sorted — `t.sort(z, axis=0)`
would destroy the records by sorting each column independently.

## Worked example

```python
import torch as t

z = t.tensor([[10, 3],
              [20, 1],
              [30, 2]])

# Sort ROWS by column 1. Inside out:
key = z[:, 1]                    # the key column: [3, 1, 2]
order = key.argsort()            # row order that sorts it: [1, 2, 0]
sorted_rows = z[order]           # fancy indexing moves whole rows
assert sorted_rows.tolist() == [[20, 1], [30, 2], [10, 3]]
# One-liner form you'll actually write: z[z[:, 1].argsort()]
print("key column", key, "-> row order", order)
print(sorted_rows)
```

Why: unpacking the sort-by into key/order/apply once makes the one-liner
readable forever after.

## Faded practice

### q92
Reorder rows so that column k is ascending.

```python starter
import torch as t

def solve(z, k):
    """Rows of z reordered so column k comes out ascending."""
    return z[z[:, k]._____()]
```

```python solution
import torch as t

def solve(z, k):
    """Rows of z reordered so column k comes out ascending."""
    return z[z[:, k].argsort()]
```

## Concept: ranks — argsort twice

**Ranks.** The composition `t.argsort(t.argsort(z))` assigns each element
its 0-based rank (0 = smallest). First argsort: "who is in each sorted
position"; second: "what position does each element hold" — the inverse
permutation. For distinct values this is THE rank formula worth memorizing.

One argsort is *not* ranks: it answers "which element is at sorted position
i?", ranks answer the inverse question — hence argsort twice.

## Worked example

```python
import torch as t

# Ranks (0 = smallest) via double argsort — the inverse permutation.
v = t.tensor([30.0, 10.0, 20.0])
ranks = t.argsort(t.argsort(v))
assert ranks.tolist() == [2, 0, 1]       # 30 is largest -> rank 2
print("values", v)
print("ranks ", ranks, " (0 = smallest)")
```

Why: trace one element — 30.0 sits at sorted position 2, so its rank is 2.
The first argsort maps positions→elements; the second inverts it to
elements→positions.

## Faded practice

### q114
Each element's 0-based rank (values distinct), reported at its own position.

```python starter
import torch as t

def solve(z):
    """0-based rank of each element: smallest -> 0, at each element's slot."""
    return z.argsort()._____()
```

```python solution
import torch as t

def solve(z):
    """0-based rank of each element: smallest -> 0, at each element's slot."""
    return z.argsort().argsort()
```

## Concept: top-k — the tail of an argsort

**Top-k.** The k largest values' indices: `t.argsort(z)[-k:][::-1]`
(ascending order → take the tail → flip to largest-first). When k is small
and n is huge, `t.topk(z, k)` finds the top-k in linear
time — partial ordering for free; sort just those k afterwards if order
matters.

Direction control, since argsort has no `descending=`: argsort the negated
array (`t.argsort(-z)`) or flip the result (`[::-1]`) — same options as
sort.

## Worked example

```python
import torch as t

# Indices of the top-2 values, largest first.
w = t.tensor([5.0, 9.0, 1.0, 7.0])
top2 = t.argsort(w)[-2:].flip(0)
assert top2.tolist() == [1, 3]           # 9.0 at index 1, then 7.0 at 3
assert w[top2].tolist() == [9.0, 7.0]    # indices recover the values
print("w", w)
print("top-2 indices", top2, "-> values", w[top2])
```

Why: each stage of the chain is checkable — ascending indices, keep the last
k, reverse. When the task says "any order is fine", argpartition saves the
final sort.

## Faded practice

### q119
Indices of the k largest elements, largest first (values distinct).

```python starter
import torch as t

def solve(z, k):
    """Indices of the k largest entries of z, largest first."""
    return t.argsort(z)[-k:]_____
```

```python solution
import torch as t

def solve(z, k):
    """Indices of the k largest entries of z, largest first."""
    return t.argsort(z)[-k:].flip(0)
```

## Independent practice

From the drill bank: q183 (per-ROW ranks of a 2-D array — the double argsort
with `axis=1` on both).

## Misconceptions

- **"argsort returns sorted values."** — It returns INDICES. `z[t.argsort(z)]`
  is the sorted array; the indices themselves are the tool for sort-by, ranks,
  and top-k.
- **"Sorting a table by a column = t.sort(z, axis=0)."** — That sorts every
  column independently, tearing rows apart. Row-preserving sort-by is
  argsort-the-key + fancy-index-the-rows.
- **"One argsort gives ranks."** — One argsort answers "which element is at
  sorted position i?" Ranks answer the inverse question, hence argsort twice.
  For distinct values they coincide only when the array was already sorted.
