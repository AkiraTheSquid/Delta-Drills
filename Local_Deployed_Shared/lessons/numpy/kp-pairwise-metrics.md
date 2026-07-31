---
kc: numpy.pairwise-metrics
title: Pairwise distances and similarities
supporting: [numpy.broadcasting-rules, numpy.axis-reductions, numpy.rescaling]
new_syntax: [torch.linalg.norm#dim, torch.linalg.norm#keepdim]
faded: [181]
guided: [127]
independent: [169, 128, 122, 193, 129]
---

## Concept

"The (n, m) matrix where entry [i, j] compares row i of X with row j of Y" —
distance matrices, similarity matrices, correlation matrices — is one
three-step recipe built entirely from broadcasting:

> **1. Insert axes so the rows meet in a new dimension.**
> `x[:, None, :]` is (n, 1, d); `y[None, :, :]` is (1, m, d).
> **2. Combine.** Subtracting gives `diff` of shape (n, m, d): slot [i, j]
> holds the elementwise difference of row i and row j.
> **3. Reduce the feature axis.** `(diff ** 2).sum(dim=-1)` collapses d,
> leaving the (n, m) table; `t.sqrt` finishes Euclidean distance.

The same skeleton with a different step 2/3 yields the whole family:

- **Euclidean distances**: subtract → square → sum → sqrt (above).
- **Cosine similarity**: normalize each ROW to unit length first (rescaling
  KP), then all pairwise DOT products — which for unit vectors is exactly
  `xn @ yn.T`. No 3-D intermediate needed: the matmul does the pairing.
- **Self-pairwise** (all points against themselves): use the same array
  twice — diagonal 0 for distance, 1 for cosine.
- **Correlation of columns**: `t.corrcoef(x.T)` — center + normalize + dot
  under the hood. Torch's `corrcoef` always treats ROWS as the variables and
  has no `rowvar` switch, so column-variables means transposing first; the
  transpose IS the switch.

Two practical notes. First, `a @ b.T` versus the 3-D broadcast: when the
combination is a *dot product*, matmul already computes all pairs — cheaper
than materializing (n, m, d). The subtract-based metrics genuinely need the
broadcast. Second, memory: (n, m, d) can be large; for big point sets the
expanded-square identity ‖x−y‖² = ‖x‖² + ‖y‖² − 2x·y computes distances from
matmuls alone (worth knowing it exists).

## Worked example

Task: all pairwise distances between two point sets; then cosine similarity
between rows of one matrix.

```python
import torch as t

x = t.tensor([[0.0, 0.0]])            # (1, 2): one point
y = t.tensor([[3.0, 4.0],
              [6.0, 8.0]])            # (2, 2): two points

# 1-2. Insert axes and subtract: (1,1,2) - (1,2,2) -> (1, 2, 2).
diff = x[:, None, :] - y[None, :, :]
assert tuple(diff.shape) == (1, 2, 2)

# 3. Square, reduce the last (feature) axis, root -> (1, 2) table.
d = t.sqrt((diff ** 2).sum(dim=-1))
assert d.tolist() == [[5.0, 10.0]]     # the 3-4-5 triangles

# Cosine similarity of rows with themselves: unit-normalize, then X @ X.T.
m = t.tensor([[1.0, 0.0],
              [1.0, 1.0]])
mn = m / t.linalg.norm(m, dim=1, keepdim=True)
cos = mn @ mn.T
assert t.allclose(t.diag(cos), t.ones(2))        # every row vs itself
assert t.isclose(cos[0, 1], 1.0 / t.sqrt(t.tensor(2.0)))  # 45 degrees apart
print(tuple(x.shape), "vs", tuple(y.shape), "-> diff", tuple(diff.shape),
      "-> distances", tuple(d.shape))
print("distance table", d)
print("cosine similarities")
print(cos)
```

Why each step:

1. Checking `diff.shape` before reducing is the habit that catches axis
   placement errors — the two `None`s must land in DIFFERENT slots, or the
   rows never pair up.
2. `axis=-1` (rather than 2) reduces "the feature axis" by name-from-the-end,
   so the same line works for self-pairwise, batched, or higher-rank
   variants.
3. In the cosine branch, notice what replaced the 3-D tensor: normalizing
   made the pairwise dot sufficient, and `@` computes every pair. Choosing
   the matmul shortcut when the metric is dot-shaped is the main efficiency
   decision in this family.

## Faded practice

### q181
Pairwise Euclidean distances between rows of x (n, d) and rows of y (m, d).

```python starter
import torch as t

def solve(x, y):
    """(n, m) distances: [i, j] = ||x[i] - y[j]||."""
    diff = x[:, _____, :] - y[_____, :, :]
    return t.sqrt((diff ** 2).sum(dim=_____))
```

```python solution
import torch as t

def solve(x, y):
    """(n, m) distances: [i, j] = ||x[i] - y[j]||."""
    diff = x[:, None, :] - y[None, :, :]
    return t.sqrt((diff ** 2).sum(dim=-1))
```

## Guided practice

### q127
1. Same recipe, one point set against ITSELF — what plays the roles of x
   and y?
2. The diagonal should come out 0 (each point vs itself) — a free
   correctness check.
3. `diff = z[:, None, :] - z[None, :, :]`, then square-sum-sqrt over the
   last axis.

## Independent practice

From the drill bank: q169 (cross cosine-similarity of two row sets —
normalize both, then one matmul), q128 (cosine similarity within one set),
q122 (Pearson correlation between COLUMNS — a packaged one-liner; find the
keyword that flips rows/columns), q193 (k nearest neighbors per point —
distance matrix, then per-row argsort, minding the self-distance).

Also from the bank: q129 (each point's TWO nearest neighbours, itself
excluded).

## Misconceptions

- **"Pairwise tables need double loops."** — Two `None`-insertions and a
  reduction replace both loops. The 3-D intermediate IS the pair table,
  before reduction.
- **"x[:, None] vs x[None, :] doesn't matter — broadcasting figures it
  out."** — It determines which array varies down rows vs across columns.
  Swap them and the table transposes; with different metrics for x and y
  roles, that's a wrong answer, not a style choice.
- **"Cosine similarity needs the 3-D broadcast too."** — It's all DOT
  products, so `xn @ yn.T` after row normalization does every pair with no
  (n, m, d) tensor. Reserve the broadcast for subtract-style metrics.
