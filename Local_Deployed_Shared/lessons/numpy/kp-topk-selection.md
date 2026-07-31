---
kc: numpy.topk-selection
title: Top-k selection — topk vs sort
supporting: [numpy.argsort-ranking, numpy.sorting]
new_syntax: [topk]
faded: [206]
guided: []
independent: [187, 194]
---

## Concept

"The k largest values" does not require sorting everything, and PyTorch packs
the whole operation into one call:

> **`t.topk(z, k)`** → a `(values, indices)` pair holding the k largest
> entries, **largest first**.

Both halves come back together, so the two questions a top-k task can ask —
*which values* and *at which positions* — are answered by reading `.values`
or `.indices` off the same result. Cost is O(len + k log k) rather than a
full O(len log len) sort; for small k on a big tensor that gap is what the
word "efficiently" in a task is pointing at.

Two knobs matter:

- **`largest=False`** flips it to the k SMALLEST.
- **`sorted=False`** drops the ordering guarantee, returning the top-k SET
  slightly cheaper — use it when you only need membership.

Because the default is already sorted descending, "largest first" needs no
follow-up step, and ascending order is just a `.flip(0)`:

```python no-run
t.topk(z, k).values.flip(0)      # k largest, ascending
t.topk(z, k, dim=1).indices      # per-row top-k indices, largest first
```

Per-row top-k on matrices: `topk` takes `dim=`, so `t.topk(z, k, dim=1)`
gives a (rows, k) block directly. For the k-th largest *single* value,
`t.kthvalue(z, k)` is the one-element cousin.

Decision rule: **full order over everything → sort/argsort; only the top k →
topk** (and `.flip(0)` if the task wants them ascending).

## Worked example

Task: the 2 largest values in ascending order; then their indices ordered
largest-first.

```python
import torch as t

z = t.tensor([5, 1, 9, 3, 7])

# topk returns both halves at once, largest first.
top = t.topk(z, 2)
assert top.values.tolist() == [9, 7]
assert top.indices.tolist() == [2, 4]

# The task wants ascending — reverse the (already sorted) k values.
assert top.values.flip(0).tolist() == [7, 9]

# Index version: which POSITIONS hold the top-2, largest first?
w = t.tensor([5.0, 9.0, 1.0, 7.0])
ordered = t.topk(w, 2).indices
assert ordered.tolist() == [1, 3]
assert w[ordered].tolist() == [9.0, 7.0]
print("z", z, "-> topk", top)
print("ascending instead:", top.values.flip(0))
print("w", w, "-> top-2 positions", ordered, "-> values", w[ordered])
```

Why each step:

1. Reading `.values` and `.indices` off one result is the habit to build —
   NumPy needed two separate calls (`partition` and `argpartition`) that
   could disagree; here they cannot.
2. `flip(0)` costs nothing on k elements. Reaching for a second `sort` is the
   common reflex and is pure waste, since topk already ordered them.
3. Only k elements are ever ordered; that asymmetry (linear scan + tiny sort)
   is the entire efficiency argument, and stating it is usually what a
   drill's "efficiently" phrasing wants.

## Faded practice

### q206
The n largest values, ascending, efficiently.

```python starter
import torch as t

def solve(z, n):
    """n largest values, ascending — topk gives them descending."""
    return t.topk(z, n).values._____(0)
```

```python solution
import torch as t

def solve(z, n):
    """n largest values, ascending — topk gives them descending."""
    return t.topk(z, n).values.flip(0)
```

## Independent practice

From the drill bank: q187 (0/1 mask marking each row's k largest — per-row
topk indices plus a zeros canvas to scatter 1s onto), q194 ((rows, k) matrix
of each row's top-k indices, largest first — which is exactly what
`topk(..., dim=1).indices` already returns).

## Misconceptions

- **"Top-k requires sorting the tensor."** — `topk` finds and orders only k
  entries. On a million elements with k=10 that's the difference between one
  pass and a full N log N shuffle.
- **"topk returns them smallest-first."** — Largest first by default. Pass
  `largest=False` for the other end, and `.flip(0)` when you want the k
  largest in ascending order.
- **"You need argsort to get top-k indices."** — `.indices` comes back from
  the same call, already ordered by value. NumPy's separate
  partition/argpartition dance has no equivalent here, and reproducing it is
  strictly more work.
