---
kc: einsum.reductions
title: Sums as index removal
supporting: [einsum.notation-model, numpy.axis-reductions]
new_syntax: []
faded: [274]
guided: [247]
independent: [289, 282, 295, 280]
---

## Concept

The core einsum rule — **a letter missing from the output is summed over** —
turns every axis reduction into a one-character deletion:

- `'ij->i'` — j is gone → sum over columns within each row: **row sums**.
- `'ij->j'` — i is gone → **column sums**.
- `'ij->'` — both gone → the grand total, as a scalar.

Compare with the `axis=` spelling you know: `x.sum(axis=1)` says which axis
*number* dies; `'ij->i'` says which *named* axes live. Same computation —
but the einsum form scales without renumbering. On a 5-D video tensor
`'btchw->bc'` reads directly as "keep batch and channel, sum time and both
spatial axes"; the axis-numbers version needs `axis=(1, 3, 4)` and a
comment.

Three practical notes:

1. **Multi-axis sums are just multiple deletions** — `'bhsd->bsd'` sums only
   h; `'bchw->b'` sums c, h, AND w at once.
2. **Order still matters on the right**: `'btd->td'` (sum batch) vs
   `'btd->dt'` (sum batch and transpose the survivors) — the output side
   does double duty.
3. **Means aren't native** — einsum only sums. A mean is a summed einsum
   divided by the collapsed axis length: `np.einsum('btd->bd', x) /
   x.shape[1]`. When several axes collapse, divide by the product of their
   lengths.

## Worked example

Task: row sums, column sums, and a "keep (b, c), sum everything else" on a
4-D tensor.

```python
import numpy as np

a = np.array([[1, 2, 3],
              [10, 20, 30]])

# 'ij->i': j deleted -> summed. One number per row.
rows = np.einsum('ij->i', a)
assert rows.tolist() == [6, 60]
assert np.array_equal(rows, a.sum(axis=1))    # same computation, named

# 'ij->j': i deleted. One number per column.
cols = np.einsum('ij->j', a)
assert cols.tolist() == [11, 22, 33]

# 4-D: keep batch, sum channel + spatial. Reads like the sentence.
x = np.arange(24).reshape(2, 3, 2, 2)
per_image = np.einsum('bchw->b', x)
assert per_image.tolist() == [66, 210]
assert np.array_equal(per_image, x.sum(axis=(1, 2, 3)))

# Mean over an axis = summed einsum / axis length.
data = np.arange(12, dtype=float).reshape(2, 3, 2)   # (b, t, d)
mean_t = np.einsum('btd->bd', data) / data.shape[1]
assert np.allclose(mean_t, data.mean(axis=1))
```

Why each step:

1. Placing the einsum and its `axis=` twin side by side (and asserting
   equality) grounds the new notation in the machinery you already trust —
   do this while learning, drop it once specs read fluently.
2. For `'bchw->b'`, the ritual: b kept; c, h, w missing → summed. The spec
   IS the sentence "total per image".
3. The mean pattern shows einsum's boundary honestly: no division inside the
   notation, so the denominator lives outside — and must be the length of
   exactly the summed axes.

## Faded practice

### q274
Row sums, spelled as index removal.

```python starter
import numpy as np

def solve(a):
    """Row sums: which letter disappears?"""
    return np.einsum('_____', a)
```

```python solution
import numpy as np

def solve(a):
    """Row sums: which letter disappears?"""
    return np.einsum('ij->i', a)
```

## Guided practice

### q247
1. Column sums: entry j of the result totals column j — which axis
   survives?
2. In `'ij->?'`, keeping j means deleting i.
3. `'ij->j'` — check against `a.sum(axis=0)` mentally: same collapse.

## Independent practice

From the drill bank: q289 (sum over the batch axis of (b, t, d)), q282 (MEAN
over the middle axis — sum via einsum, divide outside), q295 (per-sample
total of a 4-D batch), q280 (sum ONE axis of a 4-D tensor, keep the rest in
order).

## Misconceptions

- **"einsum can't do plain sums — it's for products."** — Single-input specs
  with dropped letters ARE sums (`'ij->'` is `a.sum()`). Products enter only
  with multiple inputs.
- **"'ij->i' sums rows... or is it columns?"** — Don't memorize directions;
  run the rule: j is DELETED, so for each surviving i you sum across j —
  across the columns, within a row. The deleted letter is the one you sum
  over, always.
- **"Means work by writing the axis twice or some trick."** — einsum has no
  division; a mean is einsum-sum ÷ collapsed length, explicitly. Forgetting
  the denominator (or using the wrong axis's length) is the common slip —
  q282's grader will tell you.
