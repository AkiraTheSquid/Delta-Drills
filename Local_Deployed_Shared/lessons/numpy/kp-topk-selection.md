---
kc: numpy.topk-selection
title: Top-k selection — partition vs sort
supporting: [numpy.argsort-ranking, numpy.sorting]
new_syntax: []
faded: [206]
guided: []
independent: [187, 194]
---

## Concept

"The k largest values" does not require sorting everything — and NumPy makes
the cheaper operation explicit:

**`np.partition(z, i)`** rearranges z just enough that position i holds what
it WOULD hold after a full sort, everything before it is ≤, everything after
is ≥ — in linear time, order within the two sides unspecified. So:

> **the n largest values, unordered: `np.partition(z, -n)[-n:]`**
> (position −n is the n-th largest; the tail beyond it is the top-n set).

Need them ordered? Sort just those n: `np.sort(np.partition(z, -n)[-n:])` —
total cost O(len + n log n) instead of O(len log len). For small k on big
arrays this is the difference that the word "efficiently" in a task is
pointing at.

**`np.argpartition`** is the index twin (as argsort is to sort): the same
tail trick yields the top-k *indices*, which you then order by value if the
task demands largest-first:

```python no-run
idx = np.argpartition(z, -k)[-k:]      # top-k indices, unordered
idx[np.argsort(z[idx])[::-1]]          # ordered: largest's index first
```

Per-row top-k on matrices: all four functions take `axis=` — a common combo
is `np.argsort(z, axis=1)[:, -k:]` for per-row top-k index sets, or a
partition when rows are wide.

Decision rule: **full order needed → sort/argsort; only membership in the
top k needed → partition/argpartition** (sort the k afterwards if the task
wants order too).

## Worked example

Task: the 2 largest values in ascending order, via a partial sort; then their
indices ordered largest-first.

```python
import numpy as np

z = np.array([5, 1, 9, 3, 7])

# Step 1: partition around position -2. After this, the last 2 slots hold
# the top-2 SET {9, 7} — in some order; the rest is "everything smaller".
part = np.partition(z, -2)
assert set(part[-2:].tolist()) == {7, 9}

# Step 2: the task wants ascending order — sort just those two.
top2 = np.sort(part[-2:])
assert top2.tolist() == [7, 9]

# Index version: which POSITIONS hold the top-2, largest first?
w = np.array([5.0, 9.0, 1.0, 7.0])
idx = np.argpartition(w, -2)[-2:]          # {1, 3} in some order
ordered = idx[np.argsort(w[idx])[::-1]]    # order those two by value, desc
assert ordered.tolist() == [1, 3]
assert w[ordered].tolist() == [9.0, 7.0]
```

Why each step:

1. Asserting on the SET first (not the order) mirrors what partition
   guarantees — internalizing the weaker contract prevents relying on
   accidental orderings that differ across NumPy versions.
2. The final sort touches only k elements; that asymmetry (linear scan +
   tiny sort) is the entire efficiency argument, and stating it is usually
   what the drill's "as a partial sort" phrasing wants.
3. In the index version, note the two-level indexing: `w[idx]` fetches just
   the candidates, argsort orders *within* them, and the outer `idx[...]`
   maps back to original positions — a miniature of the sort-by pattern.

## Faded practice

### q206
The n largest values, ascending, efficiently.

```python starter
import numpy as np

def solve(z, n):
    """n largest values, ascending — partition, then sort only the tail."""
    return np.sort(np._____(z, -n)[-n:])
```

```python solution
import numpy as np

def solve(z, n):
    """n largest values, ascending — partition, then sort only the tail."""
    return np.sort(np.partition(z, -n)[-n:])
```

## Independent practice

From the drill bank: q187 (0/1 mask marking each row's k largest — per-row
top-k indices + a canvas to scatter 1s onto), q194 ((rows, k) matrix of each
row's top-k indices, largest first — the argpartition/argsort combo with
axis=1, or argsort alone if you accept the extra log factor).

## Misconceptions

- **"Top-k requires sorting the array."** — Partition finds the top-k SET in
  linear time; sort only if (and only what) the task orders. On a million
  elements with k=10 that's the difference between one pass and a full
  N log N shuffle.
- **"partition(z, -n) puts the max last."** — It guarantees position −n is
  correct and the tail is the top-n SET; order within the tail is
  unspecified. Never read order out of a partition without the follow-up
  sort.
- **"argpartition output is ordered by value."** — Same caveat, index
  flavored: it hands you WHICH indices, not in which order. The
  `idx[np.argsort(z[idx])]` step is mandatory when order matters.
