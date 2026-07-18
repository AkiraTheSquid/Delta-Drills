---
kc: numpy.fancy-indexing
title: Fancy indexing — index arrays and lookup tables
supporting: [numpy.slicing-views]
new_syntax: [integer-array-indexing]
faded: [71, 139]
guided: [70]
independent: [25, 30, 101]
---

## Concept

Slices select *regular* pieces — ranges with a stride. **Fancy indexing**
lifts that restriction: index with an **array of integers**, and you get the
elements at exactly those positions, *in the order you listed them,
repetitions allowed*:

```python no-run
x[np.array([3, 0, 0, 2])]   # elements 3, 0, 0, 2 — any order, any repeats
```

Three consequences give it its power:

1. **Reordering is indexing.** Rows of a matrix in a new order:
   `z[perm]` where `perm` is a permutation of row indices — row i of the
   result is row `perm[i]` of `z`. No loop, no copying rows one by one.
   (On 2-D arrays a single index array selects whole ROWS; columns are
   `z[:, idx]`.)
2. **A lookup table is indexing.** If `values` is a table of length K and
   `labels` holds class ids 0..K-1, then `values[labels]` replaces every
   label with its looked-up value — the output is shaped like `labels`, the
   indexed table just supplies the entries. Any "map each id to its value"
   task is one indexing expression.
3. **Swaps are simultaneous.** `out[[0, -1]] = out[[-1, 0]]` exchanges first
   and last rows in one statement: the right side is gathered *before* the
   left side is written, so nothing is clobbered mid-swap.

Two contrasts with slicing to keep straight: fancy indexing always returns a
**copy** (a slice returns a view), and it accepts negative indices just like
plain indexing.

## Worked example

Task: reorder a matrix's rows by a permutation; decode a label vector through
a lookup table; swap two rows in place.

```python
import numpy as np

z = np.arange(6).reshape(3, 2)      # [[0,1],[2,3],[4,5]]

# 1. Rows in the order 2, 0, 1. Read it as: "give me row 2, then row 0,
#    then row 1" — the index array IS the new row order.
perm = np.array([2, 0, 1])
reordered = z[perm]
assert reordered.tolist() == [[4, 5], [0, 1], [2, 3]]

# Fancy indexing copies — mutating the result leaves z alone.
reordered[0, 0] = 99
assert z[2, 0] == 4

# 2. Lookup table: labels index into values. Output has labels' shape.
values = np.array([10, 20, 30])
labels = np.array([0, 2, 1, 2, 0])
decoded = values[labels]
assert decoded.tolist() == [10, 30, 20, 30, 10]

# 3. Simultaneous swap of first and last rows — RHS gathered before write.
w = np.array([[1, 2], [3, 4], [5, 6]])
w[[0, -1]] = w[[-1, 0]]
assert w.tolist() == [[5, 6], [3, 4], [1, 2]]
```

Why each step:

1. Verbalizing `z[perm]` ("row perm[i] lands at position i") resolves the
   direction confusion that otherwise haunts permutation tasks — the index
   array describes *where results come from*, not where they go.
2. In `values[labels]`, note which array is inside the brackets: the LABELS
   index, the TABLE is indexed. Output shape always follows the indexer.
3. The swap works *because* fancy indexing copies on read: by the time
   NumPy writes `w[[0, -1]]`, the old rows are already safely gathered.
   The list form `[[0, -1]]` is fancy indexing with a 2-element index array.

## Faded practice

### q71
Replace every label by its value from a lookup table.

```python starter
import numpy as np

def solve(labels, values):
    """values[labels[i]] for every i — as one indexing expression."""
    return _____[_____]
```

```python solution
import numpy as np

def solve(labels, values):
    """values[labels[i]] for every i — as one indexing expression."""
    return values[labels]
```

### q139
Exactly the rows named by idx, in idx's order (repeats and negatives legal).

```python starter
import numpy as np

def solve(x, idx):
    """Rows of x selected and ordered by idx."""
    return x[_____]
```

```python solution
import numpy as np

def solve(x, idx):
    """Rows of x selected and ordered by idx."""
    return x[idx]
```

## Guided practice

### q70
1. "Row i of the result is row perm[i] of z" — that sentence is the
   definition of indexing z with perm.
2. No loop: the permutation array goes straight into the brackets.
3. Check against the example: perm [2, 0, 1] must put z's row 2 first.

## Independent practice

From the drill bank: q25 (swap first and last ROWS — simultaneous-swap
pattern), q30 (swap first and last COLUMNS — same, other axis, mind the
`:` slot), q101 (sample k distinct rows with rng.choice + row indexing).

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
  `np.array([values[l] for l in labels])`, replace it.
