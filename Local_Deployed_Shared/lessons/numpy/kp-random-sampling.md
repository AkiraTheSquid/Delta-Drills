---
kc: numpy.random-sampling
title: Sampling — bootstrap, choice, inverse-CDF
supporting: [numpy.random-generator, numpy.fancy-indexing, numpy.cumulative-diff, numpy.axis-reductions]
new_syntax: [Tensor.argmax#dim, torch.quantile]
faded: [191]
guided: [200]
independent: [210, 91]
---

## Concept

Statistical sampling is the Generator KP plus three composition patterns:

**Resampling = random indices + gather.** To draw from data, draw INDICES
and fancy-index: `x[t.randint(0, x.numel(), shape, generator=rng)]`. Drawing
a whole matrix of indices at once — shape `(n_samples, x.numel())` — yields
every **bootstrap resample as a row**; per-resample statistics are then one
`dim=1` reduction, and a **confidence interval** is a quantile pair over
those statistics: `t.quantile(means, t.tensor([0.025, 0.975]))` for the
classic 95% CI. Note the scale — **torch quantiles run 0 to 1**, where
numpy's `percentile` ran 0 to 100; passing `[2.5, 97.5]` here is an error,
not a rescaling. The levels also have to be a TENSOR of the same dtype as
the data, not a plain list.

For "sample k rows without replacement", torch has no `choice`: draw a
permutation and slice it — `z[t.randperm(z.shape[0], generator=rng)[:k]]`.

**Inverse-CDF sampling = uniform + cumsum + argmax.** To sample a category
from a probability row p: draw u ~ Uniform[0,1) with
`t.rand(shape, generator=rng)`, form the cumulative distribution
`cs = t.cumsum(p, dim=1)`, and take the FIRST index where u < cs — which is
`(u < cs).to(t.int64).argmax(dim=1)`. The cast is not decoration: **torch's
argmax refuses a bool tensor**, where numpy quietly read it as 0/1. Cast to
int and the first-True property still holds. Vectorized over many rows: u as
a (rows, 1) column broadcasts against the (rows, k) cumsum — one line samples
every row's category simultaneously.

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
import torch as t

x = t.arange(10, dtype=t.float32)
rng = t.Generator().manual_seed(0)

# Bootstrap: ALL resample indices in one draw — one row per resample.
idx = t.randint(0, x.numel(), (1000, x.numel()), generator=rng)
resamples = x[idx]                       # gather: (1000, 10) of data values
means = resamples.mean(dim=1)            # one statistic per resample
lo, hi = t.quantile(means, t.tensor([0.025, 0.975]))   # 0-1 scale, a tensor
assert 2.0 < lo < hi < 7.0               # CI brackets the true mean 4.5
assert lo < x.mean() < hi

# Inverse-CDF, one category per row of a probability matrix.
p = t.tensor([[0.2, 0.8],
              [1.0, 0.0]])
u = t.rand((2, 1), generator=rng)        # one uniform per row, as a column
cs = t.cumsum(p, dim=1)                  # rows: [0.2, 1.0], [1.0, 1.0]
cats = (u < cs).to(t.int64).argmax(dim=1)  # first cumsum bin exceeding u
assert cats[1] == 0                      # row 2 puts all mass on category 0
assert tuple(cats.shape) == (2,)
print("1000 resample means, 95% CI: [", round(float(lo), 3), ",",
      round(float(hi), 3), "] around the true mean", x.mean().item())
print("cumulative probabilities per row:")
print(cs)
print("uniforms", u.squeeze(1), "-> categories", cats)
```

Why each step:

1. The bootstrap's entire loop structure lives in the SHAPE of one index
   draw — (n_samples, n) means "1000 resamples of size n". Statistic and CI
   are then the axis machinery. No Python-level resampling loop exists.
2. In the inverse-CDF line, each piece is checkable: cs rows end at 1.0
   (valid distributions), `u < cs` is monotone per row so its first True is
   the sampled bin, and argmax finds first-True by the argmax-on-0/1
   property you met in np-3 — after the cast torch requires.
3. The seeded assertions (CI brackets 4.5; row of pure mass gives category
   0) show how to TEST stochastic code: assert properties that hold for
   every stream, not exact values — unless the stream itself is pinned, as
   graders do.

## Faded practice

### q191
Bootstrap 95% CI for the mean, consuming rng exactly as specified.

```python starter
import torch as t

def solve(x, n_samples, rng):
    """(lo, hi) = 2.5% and 97.5% quantiles of bootstrap means."""
    idx = t.randint(0, x.numel(), (n_samples, x.numel()), generator=rng)
    means = x[idx].mean(dim=_____)
    qs = t.tensor(_____, dtype=means.dtype)
    lo, hi = t.quantile(means, qs)
    return lo, hi
```

```python solution
import torch as t

def solve(x, n_samples, rng):
    """(lo, hi) = 2.5% and 97.5% quantiles of bootstrap means."""
    idx = t.randint(0, x.numel(), (n_samples, x.numel()), generator=rng)
    means = x[idx].mean(dim=1)
    qs = t.tensor([0.025, 0.975], dtype=means.dtype)
    lo, hi = t.quantile(means, qs)
    return lo, hi
```

## Guided practice

### q200
1. One category per probability row via inverse-CDF — the three ingredients
   are a per-row uniform, a per-row cumulative sum, and a first-crossing
   search.
2. Draw u with shape (rows, 1) so it broadcasts against the (rows, k)
   cumsum.
3. `(u < cs).to(t.int64).argmax(dim=1)` — why does argmax find the FIRST
   crossing, and why does torch make you cast the mask first?

## Independent practice

From the drill bank: q210 (keep the rows that could be multinomial(n) draws —
no sampling, just property verification with masks and row sums).

Also from the bank: q91 (exactly p ones at distinct random positions,
placed with a Generator).

## Misconceptions

- **"Bootstrap = a for-loop of resamples."** — One (n_samples, n) index draw
  + gather + dim-1 statistic. The loop is the shape.
- **"Quantile levels are percentages."** — In torch they are fractions:
  `0.025`, not `2.5`. A level above 1 raises rather than rescaling, which is
  the one mercy here — the same mistake in the other direction (asking for
  `0.025` from numpy's `percentile`) returns a plausible wrong number.
- **"Sampling a categorical needs a `choice` call per row."** — torch has no
  `choice` at all, and it wouldn't vectorize over per-row probability
  vectors anyway; the inverse-CDF broadcast (uniform column vs cumsum
  matrix) samples all rows in one expression.
- **"Verify random-looking data by re-simulating it."** — Tasks that ask
  "could this be a draw from X?" want the deterministic PROPERTIES of X
  checked (support, integrality, sums). Simulation can't prove membership;
  properties can.
