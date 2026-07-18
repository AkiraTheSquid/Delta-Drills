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

- **`np.bincount(idx, weights=x)`** — THE grouped-sum workhorse. Entry k of
  the result is the sum of all `x[j]` whose `idx[j] == k` (dense over
  0..max(idx); `minlength=` pads it to a required size). Grouped MEANS are a
  ratio of two bincounts: `bincount(s, weights=d) / bincount(s)`. Grouped
  counts, sums, means — all label-indexed statistics reduce to this.
- **`np.add.at(out, idx, x)`** — unbuffered in-place scatter-add: every
  occurrence of a repeated index contributes. Use when the destination
  already exists (accumulate into a running buffer, 2-D targets with index
  tuples) rather than being built fresh. Any ufunc has `.at`:
  `np.maximum.at` does scatter-max.
- **The inverse direction** — expanding counts back into elements — is
  `np.repeat(np.arange(len(c)), c)`: value k appears c[k] times.
  (bincount ∘ this = identity.)

Recognize the family by phrases like "grouped by label", "accumulate at
positions", "one entry per class": the answer is a bincount (fresh, dense
output) or a ufunc-.at (existing buffer).

## Worked example

Task: sum values into their target bins; grouped means by label; demonstrate
the `+=` trap.

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])
i = np.array([0, 2, 0])              # two values target bin 0

# Grouped sum, dense over bins 0..max(i):
f = np.bincount(i, weights=x)
assert f.tolist() == [4.0, 0.0, 2.0]   # bin 0 got 1.0 AND 3.0

# THE TRAP: += with repeated indices drops contributions.
out = np.zeros(3)
out[i] += x                           # bin 0 written twice, keeps only ONE
assert out.tolist() == [3.0, 0.0, 2.0]     # WRONG total for bin 0!

# np.add.at accumulates every occurrence:
out2 = np.zeros(3)
np.add.at(out2, i, x)
assert out2.tolist() == [4.0, 0.0, 2.0]    # matches bincount

# Grouped means: two bincounts, one division.
d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
s = np.array([0, 1, 0, 1, 2, 2])
means = np.bincount(s, weights=d) / np.bincount(s)
assert means.tolist() == [2.0, 3.0, 5.5]
```

Why each step:

1. Running the broken `+=` right next to `add.at` is the fastest way to make
   the buffered-write semantics stick — the wrong answer (3.0, not 4.0) is
   quiet, plausible, and version-independent.
2. bincount-with-weights vs add.at: same accumulation, different setup.
   Fresh dense vector keyed by label → bincount. Existing array, arbitrary
   positions (possibly 2-D) → add.at.
3. The grouped-means ratio works because both bincounts share the label
   layout — numerator and denominator are aligned by construction. (Empty
   groups would divide 0/0; drills either exclude them or expect the NaN.)

## Faded practice

### q132
Sum each value into its target bin: F[k] = Σ x[j] where i[j] == k.

```python starter
import numpy as np

def solve(x, i):
    """Length max(i)+1 accumulation of x by target position."""
    return np.bincount(i, _____=x)
```

```python solution
import numpy as np

def solve(x, i):
    """Length max(i)+1 accumulation of x by target position."""
    return np.bincount(i, weights=x)
```

## Guided practice

### q137
1. Mean per group label: a mean is a sum over a count — you can produce BOTH
   per label with one function called twice.
2. `np.bincount(s, weights=d)` gives grouped sums; the plain call gives
   group sizes.
3. Divide. Every label 0..max occurs at least once here, so no zero-division
   guard is needed.

## Independent practice

From the drill bank: q131 (increment z at idx once PER OCCURRENCE — add.at,
or a bincount with `minlength` added to z), q216 (counts → sorted elements:
the bincount inverse via repeat), q165 (adjacency-matrix build from an edge
list — a scatter of 1s into a 2-D target: `np.add.at(A, (u, v), 1)`-style,
mind symmetric edges).

## Misconceptions

- **"`out[idx] += x` accumulates repeated indices."** — It's a gather-modify-
  scatter with ONE write per position: repeats keep only the last
  contribution. This is the single most famous silent bug in NumPy; the
  fixes are `np.add.at` or bincount.
- **"bincount only counts."** — `weights=` turns it into grouped SUM;
  with a second plain call you get grouped mean. It's the general
  label-aggregation primitive, not just a histogram.
- **"Scatter needs a loop when indices repeat."** — That's exactly what
  `ufunc.at` exists for — unbuffered, every-occurrence application, any
  ufunc (add, maximum, minimum…).
