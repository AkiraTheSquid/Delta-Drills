---
kc: numpy.nan-handling
title: NaN and Inf — detecting and repairing
supporting: [numpy.boolean-masking, numpy.where-select]
new_syntax: []
faded: [32]
guided: [18]
independent: [115, 142, 120, 170]
---

## Concept

**NaN ("not a number")** is the float value that marks missing or undefined
results (0/0, missing sensor readings). It has one property that breaks naive
code: **NaN compares unequal to everything, including itself.** `x == t.nan`
is ALWAYS False — so you cannot find NaNs with `==`.

The dedicated predicates are the way in — each returns a boolean mask:

- **`t.isnan(x)`** — True where the entry is NaN.
- **`t.isinf(x)`** — True where it is +Inf or −Inf (which are *not* NaN:
  they come from 1/0-style overflow and DO behave in comparisons).
- **`t.isfinite(x)`** — True where the entry is an ordinary number
  (not NaN, not ±Inf). Often the cleanest: "keep the good rows" is a
  condition on isfinite.

From the mask, the standard moves are the ones you already own:

- **Detect**: `t.isnan(x).any()` — is anything missing?
- **Repair**: masked assignment `x[t.isnan(x)] = fill` on a copy, or the
  packaged `t.nan_to_num(z, nan=0.0)` (which returns a new array and can
  also replace ±Inf via `posinf=`/`neginf=`).
- **Skip**: the `nan*` reduction family — `t.nansum`, `t.nanmean`,
  `t.nanmedian` — compute as if NaNs weren't there (a plain `mean` of data
  containing NaN is NaN, since NaN propagates through arithmetic).

On 2-D data, combining an isnan mask with `any(axis=...)` answers per-row /
per-column questions ("which columns contain a NaN?") — the axis mechanics
get full treatment in the broadcasting lesson, but the pattern is worth
seeing here.

## Worked example

Task: check a vector for missing values; repair them to 0.0 without touching
the input; take a NaN-ignoring mean.

```python
import torch as t

x = t.tensor([0.5, t.nan, 1.0])

# Detection MUST go through isnan — == can't see NaN, even against itself.
assert not (x == t.nan).any()           # always False everywhere: useless
assert t.isnan(x).any()                 # the real test
assert bool(t.isnan(x).any()) is True   # plain-bool contract at the boundary

# Repair, packaged: new array, NaNs -> 0.0, everything else unchanged.
fixed = t.nan_to_num(x, nan=0.0)
assert fixed.tolist() == [0.5, 0.0, 1.0]
assert t.isnan(x[1])                    # input untouched

# Repair, by hand — same result, and the form that generalizes to
# arbitrary fills (e.g. the mean of the good entries):
out = x.clone()
out[t.isnan(out)] = 0.0
assert out.tolist() == [0.5, 0.0, 1.0]

# NaN poisons plain reductions; the nan* family ignores it.
assert t.isnan(x.mean())                # NaN propagates
assert t.nanmean(x) == 0.75             # mean of 0.5 and 1.0
```

Why each step:

1. Running the broken `==` check next to `isnan` once is the fastest
   inoculation against the classic bug — the comparison silently returns all
   False rather than erroring.
2. `nan_to_num` vs manual masked assignment: the packaged call for standard
   fills, the manual pattern when the fill is computed (column means, medians
   — see the drills).
3. `x.mean()` coming out NaN is not an error, it's NaN doing its job of
   propagating; deciding between "propagate" and "ignore" (`nanmean`) is a
   data-semantics choice the task will state.

## Faded practice

### q32
Does the array contain any NaN? (plain Python bool)

```python starter
import torch as t

def solve(x):
    """True iff any entry of x is NaN."""
    return bool(t._____(x).any())
```

```python solution
import torch as t

def solve(x):
    """True iff any entry of x is NaN."""
    return bool(t.isnan(x).any())
```

## Guided practice

### q18
1. Every NaN becomes 0.0, everything else unchanged, input not modified —
   detection then repair.
2. There is a single packaged function whose keyword is literally `nan=`.
3. `t.nan_to_num(z, nan=0.0)` — or the copy + masked-assignment form; both
   satisfy "do not modify the input" (why?).

## Independent practice

From the drill bank: q115 (per-COLUMN flags for columns made ENTIRELY of NaN —
isnan mask + `all(axis=0)`), q142 (row indices containing any NaN or ±Inf —
think `isfinite`, `all(axis=1)`, invert, then nonzero).

Also from the bank: q120 (replace every NaN by its own COLUMN's mean of
the finite entries), q170 (the same fill written with keepdims so the
statistics broadcast — compare the two shapes).

## Misconceptions

- **"Find NaNs with `x == t.nan`."** — NaN ≠ NaN by IEEE definition, so that
  comparison is False everywhere. Only `t.isnan` detects them.
- **"Inf is a kind of NaN."** — Different animals: Inf is a well-ordered
  value (`t.inf > 1e300` is True) from overflow/division-by-zero; NaN is
  unordered missingness. `isnan`, `isinf`, `isfinite` slice the three cases
  cleanly.
- **"mean() skips missing values like pandas."** — PyTorch propagates: one NaN
  makes the whole mean NaN. Skipping is opt-in via `t.nanmean` and friends.
