---
kc: einsum.reductions
title: Sums as index removal
supporting: [einsum.notation-model, numpy.axis-reductions]
new_syntax: []
faded: [274, 280, 295, 282]
guided: [247]
independent: [289, 249, 261, 290, 302]
---

## Concept: One missing index sums one axis

Core rule: **a letter missing from output is summed over**.

For a matrix named `ij`, `'ij->j'` keeps `j` and deletes `i`. Each surviving
`j` therefore receives sum across `i`: one total per column. Conversely,
`'ij->i'` keeps `i` and sums across `j`: one total per row.

Read spec by asking which letters survive. Never memorize visual directions.

## Watch out

- **"`'ij->i'` sums the i axis."** — `i` survives. Missing `j` is summed.
- **"einsum only does products."** — One-input specs with missing output
  letters are plain sums. For example, `'ij->'` sums both axes.

## Worked example

Column sums keep `j`:

```python
import torch as t

a = t.tensor([[1, 2, 3],
              [10, 20, 30]])

column_sums = t.einsum('ij->j', a)
assert column_sums.tolist() == [11, 22, 33]
assert t.equal(column_sums, a.sum(dim=0))
```

`j` stays in output, so result has one value per column. Missing `i` is axis
being collapsed.

## Faded practice

### q274
Now reverse survivor: return one total per row.

```python starter
import torch as t

def solve(a):
    """Return one sum per row."""
    return t.einsum('_____', a)
```

```python solution
import torch as t

def solve(a):
    """Return one sum per row."""
    return t.einsum('ij->i', a)
```

## Concept: One deletion works same way in higher dimensions

Rank does not change rule. In `'btf->bf'`, `b` and `f` survive while `t`
disappears. Result keeps batch and feature positions, summing only time.

## Watch out

- **"A 3-D or 4-D tensor needs axis numbers."** — Same missing-letter rule
  works at any rank. Keep every axis you do not want summed.

## Worked example

Sum time from a `(batch, time, feature)` tensor:

```python
import torch as t

x = t.arange(12).reshape(2, 3, 2)
per_batch_feature = t.einsum('btf->bf', x)
assert t.equal(per_batch_feature, x.sum(dim=1))
assert per_batch_feature.shape == (2, 2)
```

Only `t` disappears, so only time is reduced.

## Faded practice

### q280
Given `(b, h, s, d)`, delete only head axis `h`.

```python starter
import torch as t

def solve(a):
    """Sum h; keep b, s, d."""
    return t.einsum('_____', a)
```

```python solution
import torch as t

def solve(a):
    """Sum h; keep b, s, d."""
    return t.einsum('bhsd->bsd', a)
```

## Concept: Several missing indices sum several axes

Multiple deletions perform multi-axis reduction in one spec. For video tensor
`btchw`, `'btchw->bc'` keeps batch and channel while summing time, height, and
width together.

## Watch out

- **"Only one missing letter can be summed."** — Every missing letter is
  summed. Three missing letters mean three collapsed axes.

## Worked example

Keep batch and channel; total time and spatial positions:

```python
import torch as t

video = t.arange(32).reshape(2, 2, 2, 2, 2)
per_batch_channel = t.einsum('btchw->bc', video)
assert t.equal(per_batch_channel, video.sum(dim=(1, 3, 4)))
assert per_batch_channel.shape == (2, 2)
```

`b` and `c` survive. Missing `t`, `h`, and `w` all collapse.

## Faded practice

### q295
Keep batch only; total channels and both spatial axes.

```python starter
import torch as t

def solve(x):
    """Return one total per batch item."""
    return t.einsum('_____', x)
```

```python solution
import torch as t

def solve(x):
    """Return one total per batch item."""
    return t.einsum('bchw->b', x)
```

## Concept: Mean equals einsum sum divided by collapsed length

Einsum sums; it does not divide. To compute mean, first delete desired index,
then divide by that axis's length. For `(b, t, d)`, `'btd->bd'` sums `t`; divide
by `data.shape[1]` because axis 1 is `t`.

## Watch out

- **"Einsum has special mean notation."** — No. Division stays outside spec.
- **"Any shape length works as denominator."** — Divide by length of exactly
  collapsed axis or product of lengths when several axes collapse.

## Worked example

Mean each matrix column by summing rows, then dividing by row count:

```python
import torch as t

a = t.tensor([[1.0, 3.0],
              [5.0, 7.0]])
column_means = t.einsum('ij->j', a) / a.shape[0]
assert t.allclose(column_means, a.mean(dim=0))
```

Missing `i` performs column sums; `a.shape[0]` converts those sums to means.

## Faded practice

### q282
Mean over middle axis `t`: choose sum spec and denominator.

```python starter
import torch as t

def solve(data):
    """Return mean over t from shape (b, t, d)."""
    return t.einsum('_____', data) / data.shape[_____]
```

```python solution
import torch as t

def solve(data):
    """Return mean over t from shape (b, t, d)."""
    return t.einsum('btd->bd', data) / data.shape[1]
```

## Guided practice

### q247
1. Column sums keep column index `j`.
2. Delete row index `i` from `'ij->?'`.
3. Check against `a.sum(axis=0)` mentally.

## Independent practice

q289 sums batch axis from `(b, t, d)` while keeping `(t, d)`.

Also from the bank: q249 (per-image sum of squares — the same tensor
twice, everything but the batch summed), q261 (the full contraction: every
index summed away to a scalar total), q290 (rank-GENERIC mean — sum every
axis, then divide by the element count), q302 (video (b, t, c, h, w) mean
over time and both spatial axes).

