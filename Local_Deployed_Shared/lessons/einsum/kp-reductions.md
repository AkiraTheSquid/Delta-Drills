---
kc: einsum.reductions
title: Sums as index removal
supporting: [einsum.notation-model, numpy.axis-reductions]
new_syntax: []
faded: [274, 280, 295, 282]
guided: [247]
independent: [289]
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
import numpy as np

a = np.array([[1, 2, 3],
              [10, 20, 30]])

column_sums = np.einsum('ij->j', a)
assert column_sums.tolist() == [11, 22, 33]
assert np.array_equal(column_sums, a.sum(axis=0))
```

`j` stays in output, so result has one value per column. Missing `i` is axis
being collapsed.

## Faded practice

### q274
Now reverse survivor: return one total per row.

```python starter
import numpy as np

def solve(a):
    """Return one sum per row."""
    return np.einsum('_____', a)
```

```python solution
import numpy as np

def solve(a):
    """Return one sum per row."""
    return np.einsum('ij->i', a)
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
import numpy as np

x = np.arange(12).reshape(2, 3, 2)
per_batch_feature = np.einsum('btf->bf', x)
assert np.array_equal(per_batch_feature, x.sum(axis=1))
assert per_batch_feature.shape == (2, 2)
```

Only `t` disappears, so only time is reduced.

## Faded practice

### q280
Given `(b, h, s, d)`, delete only head axis `h`.

```python starter
import numpy as np

def solve(a):
    """Sum h; keep b, s, d."""
    return np.einsum('_____', a)
```

```python solution
import numpy as np

def solve(a):
    """Sum h; keep b, s, d."""
    return np.einsum('bhsd->bsd', a)
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
import numpy as np

video = np.arange(32).reshape(2, 2, 2, 2, 2)
per_batch_channel = np.einsum('btchw->bc', video)
assert np.array_equal(per_batch_channel, video.sum(axis=(1, 3, 4)))
assert per_batch_channel.shape == (2, 2)
```

`b` and `c` survive. Missing `t`, `h`, and `w` all collapse.

## Faded practice

### q295
Keep batch only; total channels and both spatial axes.

```python starter
import numpy as np

def solve(x):
    """Return one total per batch item."""
    return np.einsum('_____', x)
```

```python solution
import numpy as np

def solve(x):
    """Return one total per batch item."""
    return np.einsum('bchw->b', x)
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
import numpy as np

a = np.array([[1.0, 3.0],
              [5.0, 7.0]])
column_means = np.einsum('ij->j', a) / a.shape[0]
assert np.allclose(column_means, a.mean(axis=0))
```

Missing `i` performs column sums; `a.shape[0]` converts those sums to means.

## Faded practice

### q282
Mean over middle axis `t`: choose sum spec and denominator.

```python starter
import numpy as np

def solve(data):
    """Return mean over t from shape (b, t, d)."""
    return np.einsum('_____', data) / data.shape[_____]
```

```python solution
import numpy as np

def solve(data):
    """Return mean over t from shape (b, t, d)."""
    return np.einsum('btd->bd', data) / data.shape[1]
```

## Guided practice

### q247
1. Column sums keep column index `j`.
2. Delete row index `i` from `'ij->?'`.
3. Check against `a.sum(axis=0)` mentally.

## Independent practice

q289 sums batch axis from `(b, t, d)` while keeping `(t, d)`.
