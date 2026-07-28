---
kc: numpy.sliding-windows
title: Sliding windows and moving averages
supporting: [numpy.axis-reductions, numpy.slicing-views, numpy.cumulative-diff]
new_syntax: []
faded: [117]
guided: [166]
independent: [99, 175]
---

## Concept

"Every contiguous window of length w" — the substrate of moving averages,
local maxima, and convolutions — has one canonical constructor:

> **`x.unfold(dim, size, step)`**

For a length-n vector, `x.unfold(0, w, 1)` returns a **(n − w + 1, w) matrix
whose row i is `x[i : i + w]`** — every window, materialized as rows, WITHOUT
copying (it's a strided view into the original buffer; treat it as
read-only). Window count n − w + 1: one start position per element that still
has w−1 successors.

`unfold` takes the step as its third argument, so strided windows need no
extra slicing: `x.unfold(0, w, step)` is "windows starting every `step`
elements".

Once windows are rows, **window statistics are just dim-1 reductions**:

```python no-run
x.unfold(0, w, 1).mean(dim=1)    # moving average
x.unfold(0, w, 1).amax(dim=1)    # moving maximum
```

One alternative spelling earns its place:

- **cumsum trick for moving SUMS/averages**: a window sum is a difference of
  two running totals — `c[i+w] - c[i]` where c is the cumulative sum. O(n)
  with no (n, w) intermediate, the memory-friendly route when w is large.

Recognize the family by the phrase "every window / moving / rolling"; choose
the spelling by what's reduced (any statistic → unfold; sum/mean at scale →
cumsum).

## Worked example

Task: materialize all windows of length 3; compute the moving average two
ways and confirm they agree.

```python
import torch as t

x = t.arange(7, dtype=t.float32)            # 0 1 2 3 4 5 6

# 1. All windows as rows: shape (7-3+1, 3) = (5, 3).
wins = x.unfold(0, 3, 1)
assert tuple(wins.shape) == (5, 3)
assert wins[0].tolist() == [0.0, 1.0, 2.0]
assert wins[-1].tolist() == [4.0, 5.0, 6.0]

# 2a. Moving average = windows reduced along dim 1.
ma1 = wins.mean(dim=1)
assert ma1.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]

# 2b. cumsum spelling: window sum = difference of running totals.
c = t.cumsum(t.cat([t.zeros(1), x]), dim=0)   # c[i] = sum of first i elements
ma2 = (c[3:] - c[:-3]) / 3.0
assert t.allclose(ma2, ma1)
```

Why each step:

1. Checking the first and last rows of the window view fixes the boundary
   convention: the LAST window starts at n−w, so nothing hangs off the end —
   that's where the n−w+1 count comes from.
2. In the cumsum spelling, prepending a 0 makes the algebra uniform
   (`c[i+w] − c[i]` for every i, including i=0) — the standard trick for
   prefix-sum arithmetic. Note `t.cumsum` requires an explicit `dim`.
3. Two spellings, one answer: drills accept either; YOUR choice should follow
   constraints — arbitrary statistics need the window view, big-w sums want
   cumsum.

## Faded practice

### q117
The (n−w+1, w) matrix of all length-w windows.

```python starter
import torch as t

def solve(x, w):
    """Row i = x[i : i + w], every contiguous window."""
    return x.unfold(0, _____, 1)
```

```python solution
import torch as t

def solve(x, w):
    """Row i = x[i : i + w], every contiguous window."""
    return x.unfold(0, w, 1)
```

## Guided practice

### q166
1. Moving average over every window of n entries — the window view + mean
   works; the cumsum route avoids materializing windows.
2. cumsum: a window's sum is `c[i+n] − c[i]`. Handle the offset by prepending
   a zero, or slice-shift the cumsum directly.
3. Either implementation passes — write the one you can verify, check
   endpoints against a tiny example by hand.

## Independent practice

From the drill bank: q99 (window-3 moving average — either spelling),
q175 (windows with a STRIDE — `unfold`'s third argument, or build the
step-1 view and row-slice; output row count is a small formula worth
deriving first).

## Misconceptions

- **"Window tasks need an explicit Python loop over starts."** — `unfold`
  materializes every start position as a row in one call; reductions do the
  rest. The loop survives only in the O(n·w) mental model, not the code.
- **"unfold copies n·w elements."** — It's a strided VIEW: no copy,
  negligible memory. (Consequence: don't write into it; clone first if you
  must mutate.)
- **"Moving average output has length n."** — 'valid' windows only:
  n − w + 1. Padding to length n is a separate, explicit decision
  (`t.nn.functional.pad` first) — the drills here use the valid convention.
