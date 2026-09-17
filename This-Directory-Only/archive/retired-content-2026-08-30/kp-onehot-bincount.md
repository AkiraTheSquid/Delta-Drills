---
kc: numpy.onehot-bincount
title: Labels — one-hot encoding and bincount
supporting: [numpy.fancy-indexing, numpy.constructors, numpy.argmin-argmax]
new_syntax: [torch.bincount]
faded: [93]
guided: [134]
independent: [150, 124, 172]
---

## Concept

Integer **class labels** (0, 1, …, K−1) have two canonical transformations,
and both are one-liners once you see the trick.

**One-hot encoding: `t.eye(k)[labels]`.**
A one-hot row for class c is a length-k vector of zeros with a 1 in slot c —
which is precisely **row c of the k×k identity matrix**. So encoding a whole
label vector is a lookup-table read (fancy-indexing KP) where the table is
`t.eye(k)`: each label picks its identity row, and the result stacks them
into shape (len(labels), k). Need integers instead of floats? Build the table
that way: `t.eye(k, dtype=int)`. Don't know k? The labels tell you:
`k = labels.max() + 1`.

**Counting labels: `t.bincount(labels)`.**
Returns an array where entry v is *how many times value v occurs* — a
histogram over the non-negative integers 0..max. Unlike `t.unique`'s counts
(which list only values that appear), bincount's output is **dense**: absent
values get an explicit 0, and the position IS the value. That density powers
compositions:

- **Mode** (most frequent value): `t.bincount(x).argmax()` — and because
  argmax breaks ties at the first index, ties resolve to the SMALLEST value
  automatically.
- **Weighted sums per class**: `t.bincount(labels, weights=v)` sums v's
  entries per class — grouped aggregation in one call (the applied lesson
  builds on this).

The two are inverses in spirit: one-hot *spreads* a label into a row;
bincount *collapses* a label vector into per-class totals. (Indeed
`onehot.sum(axis=0)` equals the bincount.)

## Worked example

Task: one-hot a label vector; count label occurrences; find the mode with
smallest-value tie-breaking.

```python
import torch as t

labels = t.tensor([0, 2, 1, 2])
k = 3

# One-hot: the identity matrix as lookup table, labels as row selectors.
onehot = t.eye(k)[labels]
assert onehot.shape == (4, 3)
assert onehot.tolist() == [[1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]]

# bincount: entry v = multiplicity of v. Dense — class 0,1,2 all present.
counts = t.bincount(labels)
assert counts.tolist() == [1, 1, 2]

# The two views agree: summing one-hot rows counts the classes.
assert t.equal(onehot.sum(dim=0), counts)

# Mode with smallest-on-tie: argmax over the dense counts.
x = t.tensor([3, 1, 3, 2, 3, 1])
mode = int(t.bincount(x).argmax())
assert mode == 3
# Tie case: 1 and 2 both appear twice -> argmax hits index 1 first.
assert int(t.bincount(t.tensor([1, 2, 1, 2])).argmax()) == 1
print("labels", labels)
print(onehot)
print("bincount", counts, "| one-hot columns summed", onehot.sum(dim=0))
print("mode of", x.tolist(), "is", mode)
```

Why each step:

1. Seeing `t.eye(k)[labels]` as "lookup table = identity" connects three
   prior KPs (constructors, fancy indexing) into an idiom you can re-derive
   under exam conditions rather than memorize.
2. The `onehot.sum(axis=0) == bincount` identity is a genuine consistency
   check — worth one assert when correctness matters.
3. The tie-break behavior isn't luck: bincount's index-is-value layout plus
   argmax's first-occurrence rule together GUARANTEE smallest-value ties.
   Reading composition behavior off the parts is the skill.

## Faded practice

### q93
One-hot rows for a label vector, class count given.

```python starter
import torch as t

def solve(labels, k):
    """(len(labels), k) one-hot matrix: row i encodes labels[i]."""
    return t._____(k)[labels]
```

```python solution
import torch as t

def solve(labels, k):
    """(len(labels), k) one-hot matrix: row i encodes labels[i]."""
    return t.eye(k)[labels]
```

## Guided practice

### q134
1. The most frequent value of a non-negative integer array, smallest on ties
   — which counting tool gives you position-is-value output?
2. Once counts are dense, "most frequent value" is the INDEX of the largest
   count.
3. `int(t.bincount(x).argmax())` — convince yourself why ties come out
   smallest for free.

## Independent practice

From the drill bank: q150 (one-hot where k must be INFERRED from the labels,
integer dtype required), q124 (per-row argmax one-hot of a matrix — a 2-D
cousin: zeros canvas + fancy indexing with `t.arange(rows)` paired against
the row argmaxes).

Also from the bank: q172 (per-row MODE, ties broken toward the smallest
value).

## Misconceptions

- **"One-hot needs a loop setting out[i, labels[i]] = 1."** — That loop is
  exactly what `t.eye(k)[labels]` performs in one vectorized gather. (The
  explicit-canvas form does have its place — see the per-row variant in
  q124.)
- **"bincount == unique counts."** — unique's counts are COMPACT (only values
  present, paired with a values array); bincount is DENSE (every integer
  0..max gets a slot, position = value). Mode-finding and class-vector tasks
  want the dense layout.
- **"bincount works on any integers."** — Non-negative only; negatives raise.
  Shift first (`x - x.min()`) if the data can dip below zero, and shift the
  interpretation back afterwards.
