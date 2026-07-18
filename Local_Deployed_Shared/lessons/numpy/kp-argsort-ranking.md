---
kc: numpy.argsort-ranking
title: Order statistics — argsort, ranks, sort-by
supporting: [numpy.sorting, numpy.fancy-indexing]
new_syntax: []
faded: [92]
guided: [114]
independent: [119, 183]
---

## Concept

`np.sort` rearranges values. **`np.argsort` returns the indices that WOULD
sort them** — and that indirection is what makes three whole task families
tractable:

1. **Sort one thing by another ("sort-by").** To reorder a matrix's ROWS so
   that column k comes out ascending: `z[z[:, k].argsort()]`. Read it inside
   out — argsort of the key column gives the row order; fancy indexing
   applies that order to whole rows. The rows travel intact; only their
   sequence changes. Any "sort records by field" task is this two-step.
2. **Ranks.** The composition `np.argsort(np.argsort(z))` assigns each
   element its 0-based rank (0 = smallest). First argsort: "who is in each
   sorted position"; second: "what position does each element hold" — the
   inverse permutation. For distinct values this is THE rank formula worth
   memorizing.
3. **Top-k.** The k largest values' indices: `np.argsort(z)[-k:][::-1]`
   (ascending order → take the tail → flip to largest-first). When k is
   small and n is huge, `np.argpartition(z, -k)` finds an unordered top-k in
   linear time — partial ordering for free; sort just those k afterwards if
   order matters.

Direction control, since argsort has no `descending=`: argsort the negated
array (`np.argsort(-z)`) or flip the result (`[::-1]`) — same options as
sort.

## Worked example

Task: sort a table of rows by its second column; rank a vector; find the
indices of the top 2 entries.

```python
import numpy as np

z = np.array([[10, 3],
              [20, 1],
              [30, 2]])

# 1. Sort ROWS by column 1. Inside out:
key = z[:, 1]                    # the key column: [3, 1, 2]
order = key.argsort()            # row order that sorts it: [1, 2, 0]
sorted_rows = z[order]           # fancy indexing moves whole rows
assert sorted_rows.tolist() == [[20, 1], [30, 2], [10, 3]]
# One-liner form you'll actually write: z[z[:, 1].argsort()]

# 2. Ranks (0 = smallest) via double argsort — the inverse permutation.
v = np.array([30.0, 10.0, 20.0])
ranks = np.argsort(np.argsort(v))
assert ranks.tolist() == [2, 0, 1]       # 30 is largest -> rank 2

# 3. Indices of the top-2 values, largest first.
w = np.array([5.0, 9.0, 1.0, 7.0])
top2 = np.argsort(w)[-2:][::-1]
assert top2.tolist() == [1, 3]           # 9.0 at index 1, then 7.0 at 3
assert w[top2].tolist() == [9.0, 7.0]    # indices recover the values
```

Why each step:

1. Unpacking the sort-by into key/order/apply once makes the one-liner
   readable forever after. Note what is NOT happening: no row contents are
   sorted — `np.sort(z, axis=0)` would destroy the records by sorting each
   column independently.
2. For ranks, trace one element: 30.0 sits at sorted position 2, so its rank
   is 2. The first argsort maps positions→elements; the second inverts it to
   elements→positions.
3. In the top-k chain, each stage is checkable: ascending indices, keep the
   last k, reverse. When the task says "any order is fine", argpartition
   saves the final sort.

## Faded practice

### q92
Reorder rows so that column k is ascending.

```python starter
import numpy as np

def solve(z, k):
    """Rows of z reordered so column k comes out ascending."""
    return z[z[:, k]._____()]
```

```python solution
import numpy as np

def solve(z, k):
    """Rows of z reordered so column k comes out ascending."""
    return z[z[:, k].argsort()]
```

## Guided practice

### q114
1. Each element's 0-based rank in ascending order — smallest gets 0, largest
   gets n-1. Values are distinct, so no tie policy needed.
2. One argsort tells you which element belongs at each rank; you need the
   inverse of that mapping.
3. argsort applied twice is that inverse: `np.argsort(np.argsort(z))`.

## Independent practice

From the drill bank: q119 (indices of the k largest, ordered largest first —
tail of an argsort, flipped), q183 (per-ROW ranks of a 2-D array — the double
argsort with `axis=1` on both).

## Misconceptions

- **"argsort returns sorted values."** — It returns INDICES. `z[np.argsort(z)]`
  is the sorted array; the indices themselves are the tool for sort-by, ranks,
  and top-k.
- **"Sorting a table by a column = np.sort(z, axis=0)."** — That sorts every
  column independently, tearing rows apart. Row-preserving sort-by is
  argsort-the-key + fancy-index-the-rows.
- **"One argsort gives ranks."** — One argsort answers "which element is at
  sorted position i?" Ranks answer the inverse question, hence argsort twice.
  For distinct values they coincide only when the array was already sorted.
