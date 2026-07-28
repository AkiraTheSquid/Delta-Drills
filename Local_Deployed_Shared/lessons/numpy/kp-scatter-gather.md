---
kc: numpy.scatter-gather
title: Scatter and grouped aggregation — bincount weights and add.at
supporting: [numpy.onehot-bincount, numpy.fancy-indexing]
new_syntax: []
faded: [132]
guided: [137]
independent: [131, 216, 165]
---

## Concept

Fancy indexing *gathers*: `x[idx]` pulls values out. The reverse — **push
values INTO positions, accumulating when positions repeat** — is scatter, and
it has a trap at its center:

> `out[idx] += x` does NOT accumulate repeats. It reads `out[idx]` once,
> adds, writes once — so a position listed twice receives only the LAST
> contribution, not the sum.

The correct tools, by situation:

- **`t.bincount(idx, weights=x)`** — THE grouped-sum workhorse. Entry k of
  the result is the sum of all `x[j]` whose `idx[j] == k` (dense over
  0..max(idx); `minlength=` pads it to a required size). Grouped MEANS are a
  ratio of two bincounts: `bincount(s, weights=d) / bincount(s)`. Grouped
  counts, sums, means — all label-indexed statistics reduce to this.
- **`out.index_add_(0, idx, x)`** — unbuffered in-place scatter-add along one
  dimension: every occurrence of a repeated index contributes. Use when the
  destination already exists (accumulate into a running buffer) rather than
  being built fresh. For a target indexed by a TUPLE of coordinate tensors —
  a 2-D grid — the general form is
  `out.index_put_((rows, cols), vals, accumulate=True)`, where the
  `accumulate=True` is doing all the work.
- **Other reductions** scatter through `scatter_reduce_`:
  `out.scatter_reduce_(0, idx, x, reduce="amax")` is scatter-max, and
  `"amin"`, `"sum"`, `"prod"`, `"mean"` are the rest. (NumPy spelled this
  family `np.add.at` / `np.maximum.at` — one `.at` per ufunc. Torch spells it
  as one method with a `reduce=` argument.)
- **The inverse direction** — expanding counts back into elements — is
  `t.repeat_interleave(t.arange(len(c)), c)`: value k appears c[k] times.
  (bincount ∘ this = identity.) Careful with the name: torch's `repeat`
  TILES the whole tensor, so the numpy `repeat` meaning lives on
  `repeat_interleave` — the two libraries swap these words.

Recognize the family by phrases like "grouped by label", "accumulate at
positions", "one entry per class": the answer is a bincount (fresh, dense
output) or an accumulating scatter (existing buffer).

## Worked example

Task: sum values into their target bins; grouped means by label; demonstrate
the `+=` trap.

```python
import torch as t

x = t.tensor([1.0, 2.0, 3.0])
i = t.tensor([0, 2, 0])              # two values target bin 0

# Grouped sum, dense over bins 0..max(i):
f = t.bincount(i, weights=x)
assert f.tolist() == [4.0, 0.0, 2.0]   # bin 0 got 1.0 AND 3.0

# THE TRAP: += with repeated indices drops contributions.
out = t.zeros(3)
out[i] += x                           # bin 0 written twice, keeps only ONE
assert out.tolist() == [3.0, 0.0, 2.0]     # WRONG total for bin 0!

# index_add_ accumulates every occurrence:
out2 = t.zeros(3)
out2.index_add_(0, i, x)
assert out2.tolist() == [4.0, 0.0, 2.0]    # matches bincount

# Grouped means: two bincounts, one division.
d = t.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
s = t.tensor([0, 1, 0, 1, 2, 2])
means = t.bincount(s, weights=d) / t.bincount(s)
assert means.tolist() == [2.0, 3.0, 5.5]
```

Why each step:

1. Running the broken `+=` right next to `index_add_` is the fastest way to
   make the buffered-write semantics stick — the wrong answer (3.0, not 4.0)
   is quiet, plausible, and version-independent.
2. bincount-with-weights vs index_add_: same accumulation, different setup.
   Fresh dense vector keyed by label → bincount. Existing tensor, arbitrary
   positions (possibly 2-D) → an accumulating scatter.
3. The grouped-means ratio works because both bincounts share the label
   layout — numerator and denominator are aligned by construction. (Empty
   groups would divide 0/0; drills either exclude them or expect the NaN.)

## Faded practice

### q132
Sum each value into its target bin: F[k] = Σ x[j] where i[j] == k.

```python starter
import torch as t

def solve(x, i):
    """Length max(i)+1 accumulation of x by target position."""
    return t.bincount(i, _____=x)
```

```python solution
import torch as t

def solve(x, i):
    """Length max(i)+1 accumulation of x by target position."""
    return t.bincount(i, weights=x)
```

## Guided practice

### q137
1. Mean per group label: a mean is a sum over a count — you can produce BOTH
   per label with one function called twice.
2. `t.bincount(s, weights=d)` gives grouped sums; the plain call gives
   group sizes.
3. Divide. Every label 0..max occurs at least once here, so no zero-division
   guard is needed.

## Independent practice

From the drill bank: q131 (increment z at idx once PER OCCURRENCE —
`index_add_`, or a bincount with `minlength` added to z), q216 (counts →
sorted elements: the bincount inverse via `repeat_interleave`), q165
(adjacency-matrix build from an edge list — a scatter of 1s into a 2-D
target, `a[rows, cols] = 1`, mind symmetric edges).

## Misconceptions

- **"`out[idx] += x` accumulates repeated indices."** — It's a gather-modify-
  scatter with ONE write per position: repeats keep only the last
  contribution. This is the single most famous silent bug in array
  programming, and torch inherits it exactly; the fixes are `index_add_`,
  `index_put_(..., accumulate=True)`, or bincount.
- **"bincount only counts."** — `weights=` turns it into grouped SUM;
  with a second plain call you get grouped mean. It's the general
  label-aggregation primitive, not just a histogram.
- **"Scatter needs a loop when indices repeat."** — That's exactly what the
  accumulating scatters exist for — unbuffered, every-occurrence
  application, with `scatter_reduce_`'s `reduce=` covering sum, max, min and
  the rest.
