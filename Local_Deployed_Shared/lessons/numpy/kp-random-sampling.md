---
kc: numpy.random-sampling
title: Sampling — bootstrap, choice, inverse-CDF
supporting: [numpy.random-generator, numpy.fancy-indexing, numpy.cumulative-diff, numpy.axis-reductions]
new_syntax: []
faded: [191]
guided: [200]
independent: [210]
---

## Concept

Statistical sampling in NumPy is the Generator KP plus three composition
patterns:

**Resampling = random indices + gather.** To draw from data, draw INDICES
and fancy-index: `x[rng.integers(0, x.size, shape)]`. Drawing a whole matrix
of indices at once — shape `(n_samples, x.size)` — yields every **bootstrap
resample as a row**; per-resample statistics are then one `axis=1`
reduction, and a **confidence interval** is a percentile pair over those
statistics: `np.percentile(means, [2.5, 97.5])` for the classic 95% CI.
(`rng.choice(x, size=..., replace=...)` packages the index-draw+gather;
`replace=False` gives distinct draws, as in "sample k rows without
replacement": `z[rng.choice(z.shape[0], k, replace=False)]`.)

**Inverse-CDF sampling = uniform + cumsum + argmax.** To sample a category
from a probability row p: draw u ~ Uniform[0,1), form the cumulative
distribution `cs = np.cumsum(p)`, and take the FIRST index where u < cs —
which is `(u < cs).argmax()` (argmax on booleans returns the first True).
Vectorized over many rows: u as a (rows, 1) column broadcasts against the
(rows, k) cumsum — one line samples every row's category simultaneously.

**Checking randomness claims = deterministic verification.** Tasks like
"which rows could be multinomial draws" have no randomness at all: verify
the defining properties (integer entries, non-negative, row sums equal n)
with masks and reductions.

The Generator discipline from np-1 carries over unchanged: when a drill
specifies `rng` and even the exact call to consume (as bootstrap tasks do —
"use EXACTLY rng.integers(...)"), follow it literally; the grader replays
the stream.

## Worked example

Task: bootstrap 95% CI of a mean; sample one category per probability row by
inverse-CDF.

```python
import numpy as np

x = np.arange(10, dtype=float)
rng = np.random.default_rng(0)

# Bootstrap: ALL resample indices in one draw — one row per resample.
idx = rng.integers(0, x.size, (1000, x.size))
resamples = x[idx]                       # gather: (1000, 10) of data values
means = resamples.mean(axis=1)           # one statistic per resample
lo, hi = np.percentile(means, [2.5, 97.5])
assert 2.0 < lo < hi < 7.0               # CI brackets the true mean 4.5
assert lo < x.mean() < hi

# Inverse-CDF, one category per row of a probability matrix.
p = np.array([[0.2, 0.8],
              [1.0, 0.0]])
u = rng.random((2, 1))                   # one uniform per row, as a column
cs = np.cumsum(p, axis=1)                # rows: [0.2, 1.0], [1.0, 1.0]
cats = (u < cs).argmax(axis=1)           # first cumsum bin exceeding u
assert cats[1] == 0                      # row 2 puts all mass on category 0
assert cats.shape == (2,)
```

Why each step:

1. The bootstrap's entire loop structure lives in the SHAPE of one index
   draw — (n_samples, n) means "1000 resamples of size n". Statistic and CI
   are then the axis machinery. No Python-level resampling loop exists.
2. In the inverse-CDF line, each piece is checkable: cs rows end at 1.0
   (valid distributions), `u < cs` is monotone per row so its first True is
   the sampled bin, and argmax finds first-True by the argmax-on-booleans
   property you met in np-3.
3. The seeded assertions (CI brackets 4.5; row of pure mass gives category
   0) show how to TEST stochastic code: assert properties that hold for
   every stream, not exact values — unless the stream itself is pinned, as
   graders do.

## Faded practice

### q191
Bootstrap 95% CI for the mean, consuming rng exactly as specified.

```python starter
import numpy as np

def solve(x, n_samples, rng):
    """(lo, hi) = 2.5th and 97.5th percentiles of bootstrap means."""
    idx = rng.integers(0, x.size, (n_samples, x.size))
    means = x[idx].mean(axis=_____)
    lo, hi = np.percentile(means, _____)
    return lo, hi
```

```python solution
import numpy as np

def solve(x, n_samples, rng):
    """(lo, hi) = 2.5th and 97.5th percentiles of bootstrap means."""
    idx = rng.integers(0, x.size, (n_samples, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return lo, hi
```

## Guided practice

### q200
1. One category per probability row via inverse-CDF — the three ingredients
   are a per-row uniform, a per-row cumulative sum, and a first-crossing
   search.
2. Draw u with shape (rows, 1) so it broadcasts against the (rows, k)
   cumsum.
3. `(u < cs).argmax(axis=1)` — why does argmax find the FIRST crossing?

## Independent practice

From the drill bank: q210 (keep the rows that could be multinomial(n) draws —
no sampling, just property verification with masks and row sums).

## Misconceptions

- **"Bootstrap = a for-loop of resamples."** — One (n_samples, n) index draw
  + gather + axis-1 statistic. The loop is the shape.
- **"Sampling a categorical needs np.random.choice per row."** — Choice
  doesn't vectorize over per-row probability vectors; the inverse-CDF
  broadcast (uniform column vs cumsum matrix) samples all rows in one
  expression.
- **"Verify random-looking data by re-simulating it."** — Tasks that ask
  "could this be a draw from X?" want the deterministic PROPERTIES of X
  checked (support, integrality, sums). Simulation can't prove membership;
  properties can.
