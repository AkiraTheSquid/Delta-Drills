---
kc: numpy.fancy-indexing
title: Fancy indexing — index arrays and lookup tables
supporting: [numpy.slicing-views]
new_syntax: [integer-array-indexing]
faded: [139, 71, 25]
guided: [136]
independent: [70, 30, 101, 177, 184]
---

## Concept: index with an array — select and reorder at once

Slices select *regular* pieces — ranges with a stride. **Fancy indexing**
lifts that restriction: index with an **array of integers**, and you get the
elements at exactly those positions, *in the order you listed them,
repetitions allowed*:

```python no-run
x[t.tensor([3, 0, 0, 2])]   # elements 3, 0, 0, 2 — any order, any repeats
```

**Reordering is indexing.** Rows of a matrix in a new order: `z[perm]` where
`perm` is a permutation of row indices — row i of the result is row
`perm[i]` of `z`. The index array describes *where results come from*, not
where they go. (On 2-D arrays a single index array selects whole ROWS;
columns are `z[:, idx]`.) Negative indices work just like plain indexing —
and unlike a slice, fancy indexing always returns a **copy**.

## Worked example

```python
import torch as t

z = t.arange(6).reshape(3, 2)      # [[0,1],[2,3],[4,5]]

# Rows in the order 2, 0, 1. Read it as: "give me row 2, then row 0,
# then row 1" — the index array IS the new row order.
perm = t.tensor([2, 0, 1])
reordered = z[perm]
assert reordered.tolist() == [[4, 5], [0, 1], [2, 3]]

# Fancy indexing copies — mutating the result leaves z alone.
reordered[0, 0] = 99
assert z[2, 0] == 4
print("order", perm)
print(reordered, " <- 99 written here")
print("z unchanged:", z.tolist())
```

Why: verbalizing `z[perm]` ("row perm[i] lands at position i") resolves the
direction confusion that otherwise haunts permutation tasks.

## Faded practice

### q139
Exactly the rows named by idx, in idx's order (repeats and negatives legal).

```python starter
import torch as t

def solve(x, idx):
    """Rows of x selected and ordered by idx."""
    return x[_____]
```

```python solution
import torch as t

def solve(x, idx):
    """Rows of x selected and ordered by idx."""
    return x[idx]
```

## Concept: a lookup table is indexing

If `values` is a table of length K and `labels` holds class ids 0..K-1, then
`values[labels]` replaces every label with its looked-up value — the output
is shaped like `labels`; the indexed table just supplies the entries. Any
"map each id to its value" task is one indexing expression.

Note which array is inside the brackets: the LABELS index, the TABLE is
indexed. Output shape always follows the indexer.

## Worked example

```python
import torch as t

# Lookup table: labels index into values. Output has labels' shape.
values = t.tensor([10, 20, 30])
labels = t.tensor([0, 2, 1, 2, 0])
decoded = values[labels]
assert decoded.tolist() == [10, 30, 20, 30, 10]
print("labels ", labels, "shape", tuple(labels.shape))
print("decoded", decoded, "shape", tuple(decoded.shape), "<- labels' shape")
```

Why: this is a single vectorized gather in C — if you're writing
`t.tensor([values[l] for l in labels])`, replace it.

## Faded practice

### q71
Replace every label by its value from a lookup table.

```python starter
import torch as t

def solve(labels, values):
    """values[labels[i]] for every i — as one indexing expression."""
    return _____[_____]
```

```python solution
import torch as t

def solve(labels, values):
    """values[labels[i]] for every i — as one indexing expression."""
    return values[labels]
```

## Concept: swaps are simultaneous

`out[[0, -1]] = out[[-1, 0]]` exchanges first and last rows in one
statement: the right side is gathered *before* the left side is written, so
nothing is clobbered mid-swap. The swap works *because* fancy indexing
copies on read.

Contrast: the pure-Python idiom `z[0], z[-1] = z[-1], z[0]` is NOT safe on
arrays — each `z[i]` is a view, so the first assignment overwrites data the
second still needs.

## Worked example

```python
import torch as t

# Simultaneous swap of first and last rows — RHS gathered before write.
w = t.tensor([[1, 2], [3, 4], [5, 6]])
w[[0, -1]] = w[[-1, 0]]
assert w.tolist() == [[5, 6], [3, 4], [1, 2]]
print(w)
```

Why: the list form `[[0, -1]]` is fancy indexing with a 2-element index
tensor — by the time PyTorch writes `w[[0, -1]]`, the old rows are already
safely gathered.

## Faded practice

### q25
First and last rows exchanged (new array, input unmodified).

```python starter
import torch as t

def solve(x):
    """x with its first and last rows swapped, x unmodified."""
    out = x.clone()
    out[[0, -1]] = out[_____]
    return out
```

```python solution
import torch as t

def solve(x):
    """x with its first and last rows swapped, x unmodified."""
    out = x.clone()
    out[[0, -1]] = out[[-1, 0]]
    return out
```

## Guided practice

### q136
1. Row swap is a fancy-index assignment: read the two rows in one order,
   write them back in the other.
2. Work on a clone — the drill checks the input is untouched — and
   remember i may equal j, which must still be a no-op.
3. `out = z.clone()`, then `out[[i, j]] = out[[j, i]]`. The right-hand
   side is materialized before the write, so no aliasing problem.

## Independent practice

From the drill bank: q70 (reorder rows by a permutation — the segment-1
pattern as a full task), q30 (swap first and last COLUMNS — same idea, other
axis, mind the `:` slot), q101 (sample k distinct rows with rng.choice + row
indexing).

Also from the bank: q177 (remap values through a dict WITHOUT looping over
the data), q184 (set a symmetric pair z[i, j] and z[j, i] together,
diagonal case included).

## Misconceptions

- **"Fancy indexing returns a view like slicing."** — It returns a COPY.
  Consequence: `z[idx][0] = 5` modifies the copy and silently discards it;
  write through the original (`z[idx_of_target] = 5`) when mutation is the
  goal.
- **"Swapping rows needs a temporary."** — `z[[i, j]] = z[[j, i]]` is safe:
  the right side is fully read before the write. (The pure-Python idiom
  `z[i], z[j] = z[j], z[i]` is NOT safe on arrays — the first assignment
  overwrites data the second still needs, because each `z[i]` is a view.)
- **"values[labels] loops over labels under the hood in Python."** — It's a
  single vectorized gather in C. If you're writing
  `t.tensor([values[l] for l in labels])`, replace it.
