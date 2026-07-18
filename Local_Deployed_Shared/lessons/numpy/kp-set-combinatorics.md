---
kc: numpy.set-combinatorics
title: Set operations, cartesian products, run-length encoding
supporting: [numpy.unique, numpy.cumulative-diff, numpy.tile-repeat-meshgrid, numpy.fancy-indexing]
new_syntax: []
faded: [185]
guided: [171]
independent: [199, 205, 143]
---

## Concept

Three "discrete structure" patterns that keep appearing, all buildable from
tools you own:

**Run-length encoding (RLE).** Compress `[1, 1, 2, 3, 3, 3]` into
(values `[1, 2, 3]`, counts `[2, 1, 3]`). The vectorized derivation:

> 1. **Where do runs start?** At index 0, and wherever an element differs
>    from its predecessor: `a[1:] != a[:-1]` (shifted-slice comparison),
>    positions via `np.flatnonzero` (+1 for the shift).
> 2. **Values** = the elements at the start positions: `a[starts]`.
> 3. **Counts** = differences between consecutive starts (with the array
>    length appended as the final fence): `np.diff(np.r_[starts, a.size])`.

`np.r_[...]` is the row-concatenation shorthand (`np.r_[0, idx]` prepends a
0). The whole encoder is three lines of slicing, nonzero, and diff — a
model example of composing primitives. Related one-liners: collapse
consecutive duplicates = `a[np.r_[True, a[1:] != a[:-1]]]` (keep each run's
first element); decode an RLE = `np.repeat(values, counts)`.

**Cartesian products.** Every combination of one element per array:
`np.meshgrid(*arrays, indexing='ij')` builds one coordinate grid per input
(`'ij'` keeps the first array's axis FIRST — the ordering that makes output
rows lexicographic), then ravel each grid and stack as columns → shape
(∏ lengths, n_arrays).

**Row-level set operations.** unique-with-axis handles distinct rows; for
"rows in both a and b" the standard trick converts each row to a single
comparable unit — e.g. a *structured view* (each row becomes one record) so
`np.intersect1d` applies, or for small integer alphabets, encode rows as
scalars (base-K positional encoding) and intersect those. Order-of-first-
appearance variants combine `return_index=True` with a sort of the indices
(as in the unique KP).

## Worked example

Task: RLE-encode a vector, verify by decoding; build a cartesian product of
two arrays.

```python
import numpy as np

a = np.array([1, 1, 2, 3, 3, 3])

# 1. Run starts: index 0 plus every "different from predecessor" position.
starts = np.r_[0, 1 + np.flatnonzero(a[1:] != a[:-1])]
assert starts.tolist() == [0, 2, 3]

# 2-3. Values at the starts; counts as fenced differences.
values = a[starts]
counts = np.diff(np.r_[starts, a.size])
assert values.tolist() == [1, 2, 3]
assert counts.tolist() == [2, 1, 3]

# Round-trip check: repeat decodes RLE exactly.
assert np.repeat(values, counts).tolist() == a.tolist()

# Cartesian product of [1,2] x [10,20,30]: 6 rows, lexicographic.
arrays = [np.array([1, 2]), np.array([10, 20, 30])]
grids = np.meshgrid(*arrays, indexing='ij')
prod = np.stack([g.ravel() for g in grids], axis=1)
assert prod.shape == (6, 2)
assert prod.tolist() == [[1, 10], [1, 20], [1, 30],
                         [2, 10], [2, 20], [2, 30]]
```

Why each step:

1. The `+1` on the flatnonzero result compensates for the slice shift
   (`a[1:] != a[:-1]` compares position i+1 with i) — narrate it or you'll
   drop it. The prepended 0 encodes "the first run starts at 0", which the
   comparison can't see.
2. The fence `np.r_[starts, a.size]` turns "count = next start − this start"
   into a uniform diff, final run included — the same prepend/append move as
   the cumsum windows trick.
3. In the cartesian product, `indexing='ij'` is what makes the FIRST array
   vary slowest (lexicographic rows); the default 'xy' swaps the first two
   axes and scrambles the expected order.

## Faded practice

### q185
Run-length encoding as (values, counts).

```python starter
import numpy as np

def solve(a):
    """(run values in order, run lengths)."""
    starts = np.r_[0, 1 + np.flatnonzero(a[1:] != a[:-1])]
    values = a[_____]
    counts = np.diff(np.r_[_____, a.size])
    return values, counts
```

```python solution
import numpy as np

def solve(a):
    """(run values in order, run lengths)."""
    starts = np.r_[0, 1 + np.flatnonzero(a[1:] != a[:-1])]
    values = a[starts]
    counts = np.diff(np.r_[starts, a.size])
    return values, counts
```

## Guided practice

### q171
1. Every combination of one element per input array, as rows — meshgrid
   generalizes to any number of inputs via unpacking.
2. `indexing='ij'` keeps the first input's values varying slowest — check
   the required row order in the prompt.
3. Ravel each grid, stack as columns: shape (∏ lengths, len(arrays)).

## Independent practice

From the drill bank: q199 (rows common to two 2-D arrays — make rows
comparable units first), q205 (distinct rows in order of FIRST appearance —
unique axis=0 + return_index + sort), q143 (collapse consecutive duplicates —
the keep-run-starts mask).

## Misconceptions

- **"RLE needs itertools.groupby."** — The shifted-slice comparison +
  flatnonzero + diff pipeline is fully vectorized; groupby iterates in
  Python. Same information, orders of magnitude apart on long arrays.
- **"np.unique solves 'distinct rows in appearance order'."** — Bare unique
  SORTS (lexicographically with axis=0). Appearance order requires
  return_index and re-sorting by those indices — two extra steps the sorted
  default hides.
- **"Cartesian products need nested loops."** — meshgrid + ravel + stack;
  the loops exist only inside compiled code. For 3+ arrays, unpack the list:
  `np.meshgrid(*arrays, indexing='ij')`.
