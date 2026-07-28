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
>    positions via `t.nonzero(..., as_tuple=True)[0]` (+1 for the shift).
> 2. **Values** = the elements at the start positions: `a[starts]`.
> 3. **Counts** = differences between consecutive starts (with the tensor
>    length appended as the final fence):
>    `t.diff(t.cat([starts, t.tensor([a.numel()])]))`.

Where numpy had `np.r_[...]` as concatenation shorthand, torch has `t.cat`,
and `cat` takes TENSORS only — a bare `0` or `True` has to become
`t.zeros(1, dtype=t.int64)` or `t.tensor([True])` first. That is the one real
friction in this pattern; the rest is identical. The whole encoder is three
lines of slicing, nonzero, and diff — a model example of composing
primitives. Related one-liners: collapse consecutive duplicates =
`a[t.cat([t.tensor([True]), a[1:] != a[:-1]])]` (keep each run's first
element); decode an RLE = `t.repeat_interleave(values, counts)`.

**Cartesian products.** Every combination of one element per array:
`t.meshgrid(*arrays, indexing='ij')` builds one coordinate grid per input
(`'ij'` keeps the first array's axis FIRST — the ordering that makes output
rows lexicographic; torch REQUIRES you to say which, where numpy defaulted to
`'xy'`), then ravel each grid and stack as columns → shape
(∏ lengths, n_arrays).

**Row-level set operations.** `t.unique(z, dim=0)` handles distinct rows. For
"rows in both a and b" torch has no `intersect1d` at all — compose instead:
`ua = t.unique(a); ua[t.isin(ua, b)]` for values, or for rows, encode each
row as a single comparable scalar (base-K positional encoding for small
integer alphabets) and intersect those. Order-of-first-appearance variants
need work, because `t.unique` has **no `return_index`**: ask for
`return_inverse=True` instead and recover each value's first position by
scattering positions back with `reduce="amin"` (as in the unique KP).

## Worked example

Task: RLE-encode a vector, verify by decoding; build a cartesian product of
two arrays.

```python
import torch as t

a = t.tensor([1, 1, 2, 3, 3, 3])

# 1. Run starts: index 0 plus every "different from predecessor" position.
changes = t.nonzero(a[1:] != a[:-1], as_tuple=True)[0] + 1
starts = t.cat([t.zeros(1, dtype=t.int64), changes])
assert starts.tolist() == [0, 2, 3]

# 2-3. Values at the starts; counts as fenced differences.
values = a[starts]
counts = t.diff(t.cat([starts, t.tensor([a.numel()])]))
assert values.tolist() == [1, 2, 3]
assert counts.tolist() == [2, 1, 3]

# Round-trip check: repeat_interleave decodes RLE exactly.
assert t.repeat_interleave(values, counts).tolist() == a.tolist()

# Cartesian product of [1,2] x [10,20,30]: 6 rows, lexicographic.
arrays = [t.tensor([1, 2]), t.tensor([10, 20, 30])]
grids = t.meshgrid(*arrays, indexing='ij')
prod = t.stack([g.ravel() for g in grids], dim=1)
assert tuple(prod.shape) == (6, 2)
assert prod.tolist() == [[1, 10], [1, 20], [1, 30],
                         [2, 10], [2, 20], [2, 30]]
```

Why each step:

1. The `+1` on the nonzero result compensates for the slice shift
   (`a[1:] != a[:-1]` compares position i+1 with i) — narrate it or you'll
   drop it. The prepended 0 encodes "the first run starts at 0", which the
   comparison can't see, and it has to be `t.zeros(1, dtype=t.int64)` so
   `cat` sees a tensor of the index dtype.
2. The fence `t.cat([starts, t.tensor([a.numel()])])` turns "count = next
   start − this start" into a uniform diff, final run included — the same
   prepend/append move as the cumsum windows trick.
3. In the cartesian product, `indexing='ij'` is what makes the FIRST array
   vary slowest (lexicographic rows); the default 'xy' swaps the first two
   axes and scrambles the expected order.

## Faded practice

### q185
Run-length encoding as (values, counts).

```python starter
import torch as t

def solve(a):
    """(run values in order, run lengths)."""
    changes = t.nonzero(a[1:] != a[:-1], as_tuple=True)[0] + 1
    starts = t.cat([t.zeros(1, dtype=t.int64), changes])
    values = a[_____]
    counts = t.diff(t.cat([_____, t.tensor([a.numel()])]))
    return values, counts
```

```python solution
import torch as t

def solve(a):
    """(run values in order, run lengths)."""
    changes = t.nonzero(a[1:] != a[:-1], as_tuple=True)[0] + 1
    starts = t.cat([t.zeros(1, dtype=t.int64), changes])
    values = a[starts]
    counts = t.diff(t.cat([starts, t.tensor([a.numel()])]))
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

From the drill bank: q199 (rows common to two 2-D tensors — make rows
comparable units first), q205 (distinct rows in order of FIRST appearance —
`unique(dim=0, return_inverse=True)`, then recover first positions), q143
(collapse consecutive duplicates — the keep-run-starts mask).

## Misconceptions

- **"RLE needs itertools.groupby."** — The shifted-slice comparison +
  nonzero + diff pipeline is fully vectorized; groupby iterates in
  Python. Same information, orders of magnitude apart on long tensors.
- **"t.unique solves 'distinct rows in appearance order'."** — Bare unique
  SORTS (lexicographically with `dim=0`). Appearance order needs the inverse
  map and a first-position recovery — and torch gives you no `return_index`
  shortcut, so this is genuinely more work than it was in numpy.
- **"Cartesian products need nested loops."** — meshgrid + ravel + stack;
  the loops exist only inside compiled code. For 3+ arrays, unpack the list:
  `t.meshgrid(*arrays, indexing='ij')`.
