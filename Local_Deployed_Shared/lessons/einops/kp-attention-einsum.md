---
kc: einops.attention-einsum
title: einsum at model rank — heads, positions and `...`
supporting: [einops.einsum, einops.dl-flatten-heads]
new_syntax: []
previews: []
concepts: [named-axes-at-rank, per-head-weights-and-ellipsis]
faded: [1555, 1556, 1557]
guided: []
independent: [1558, 1559, 1560]
integrated: [1561, 1562]
---

## Concept: named axes at model rank — the projection and the weighted sum

Every einsum in a transformer is one of the two-operand patterns you already
know, written at rank three or four with WORDS for names. Two of them do most
of the work in ARENA's chapter 1:

| pattern | what it is |
|---|---|
| `"b s d, d n -> b s n"` | a linear projection applied to every position of every sequence — the `d` axis is shared and summed, `b s` ride along |
| `"b s n, b s d -> b n d"` | a weighted sum of the value vectors: `s` is shared by both operands and missing from the right, so each head's weights collapse the positions |

Nothing new happens: a shared name multiplies, a name left off the right is
summed, and the survivors come out in the order you write them. What changes
at model rank is that the ORDER on the right is now a design decision. ARENA
keeps `batch posn nheads d_head` for every activation, and the next einsum in
the file assumes that layout.

## Watch out
- **Survivors in the wrong order** — `-> b n s` and `-> b s n` are both legal
  and contain the same numbers; only one lines up with the tensor you
  multiply next. Read the shape the task asks for left to right and write the
  right-hand side in that order.
- **Summing the wrong shared axis** — in `"b s n, b s d"` both `b` and `s`
  are shared. Keeping `s` and dropping `n` sums over the heads, not the
  positions. The axis you are averaging over is the one that must vanish.

## Worked example

We project a batch of sequences with a small weight, then use per-head weights to collapse the positions. Both are single einsum calls; only the names are longer.

```python
import torch as t
import einops

# (b, s, d): one sequence of 3 vectors
x = t.arange(12.0).reshape(1, 3, 4)
# (d, n): keep the first two features
w = t.eye(4)[:, :2]

proj = einops.einsum(x, w, "b s d, d n -> b s n")
print("proj", tuple(proj.shape), proj[0].tolist())
# Hidden checks
assert proj.shape == (1, 3, 2)
assert proj[0].tolist() == x[0, :, :2].tolist()
```

The same shapes, now with a head axis: `attn` holds one weight per position for each of two heads, and `s` is the axis that disappears.

```python
import torch as t
import einops

# (b, s, n): head 0 looks at position 0, head 1 at 1
attn = t.tensor([[[1.0, 0.0],
                  [0.0, 1.0],
                  [0.0, 0.0]]])
# (b, s, d)
v = t.arange(12.0).reshape(1, 3, 4)

z = einops.einsum(attn, v, "b s n, b s d -> b n d")
print("z", tuple(z.shape), z[0].tolist())
# Hidden checks
assert z.shape == (1, 2, 4)
assert z[0, 0].tolist() == v[0, 0].tolist()      # head 0 copied position 0
assert z[0, 1].tolist() == v[0, 1].tolist()      # head 1 copied position 1
```

## Faded practice

### q1555
Project every vector of a batch of sequences.

```python starter
import torch as t
import einops


def solve(x, w):
    """Project every position: (b, s, d) x (d, n) -> (b, s, n)."""
    return einops.einsum(t.tensor(x), t.tensor(w), "b s d, d n -> _____").tolist()
```

```python solution
import torch as t
import einops


def solve(x, w):
    """Project every position: (b, s, d) x (d, n) -> (b, s, n)."""
    return einops.einsum(t.tensor(x), t.tensor(w), "b s d, d n -> b s n").tolist()
```

### q1556
Collapse the positions with one weight per position per head.

```python starter
import torch as t
import einops


def solve(attn, v):
    """Weighted sum of values, one per head: (b, s, n), (b, s, d) -> (b, n, d)."""
    return einops.einsum(t.tensor(attn), t.tensor(v), "b s n, b s d -> _____").tolist()
```

```python solution
import torch as t
import einops


def solve(attn, v):
    """Weighted sum of values, one per head: (b, s, n), (b, s, d) -> (b, n, d)."""
    return einops.einsum(t.tensor(attn), t.tensor(v), "b s n, b s d -> b n d").tolist()
```

## Concept: one matrix per head, and `...` for the axes you never name

Attention does not use one projection matrix; it uses one PER HEAD, stored
as a single `(nheads, d_model, d_head)` tensor. einsum handles that with no
loop and no reshape: the weight simply has a third named axis, and it
survives on the right.

> `"batch posn d_model, nheads d_model d_head -> batch posn nheads d_head"`

`d_model` is shared and summed; `nheads` and `d_head` come from the weight;
`batch posn` come from the activation. Scores between queries and keys use
the same idea with a twist: the two operands each have a position axis, and
they must NOT be paired, so they get two different names:

> `"batch posn_q nheads d_head, batch posn_k nheads d_head -> batch nheads posn_q posn_k"`

Finally, `...` stands for "whatever leading axes there are". A pattern with
`...` on both sides of an operand and on the output works for a single
example, a batch, or a batch of batches, without rewriting it:

> `"... inst feats, inst hidden feats -> ... inst hidden"`

## Watch out
- **Same name for two axes that must range independently** — passing a
  tensor twice with `"i h f, i h f -> i f"` pairs each feature only with
  itself. The overlap matrix needs `f1` and `f2`.
- **einsum only contracts** — the `1/sqrt(d_head)` scaling, the mask and the
  softmax are ordinary tensor arithmetic applied AFTER the pattern.

## Worked example

Two heads, each with its own 3x2 matrix, project a single position. Then a weight tensor carrying an instance axis is applied under `...`.

```python
import torch as t
import einops

# (batch, posn, d_model)
x = t.ones(1, 1, 3)
# (nheads=2, d_model=3, d_head=2)
W = t.stack([t.eye(3)[:, :2], 2 * t.eye(3)[:, :2]])

q = einops.einsum(
    x, W,
    "batch posn d_model, nheads d_model d_head"
    " -> batch posn nheads d_head")
print("q", tuple(q.shape), q[0, 0].tolist())
# Hidden checks
assert q.shape == (1, 1, 2, 2)
assert q[0, 0, 1].tolist() == [2.0, 2.0]               # head 1 used its own doubled matrix
```

`...` absorbs the leading axes: the same call runs on a (2, 5, inst, feats) tensor and on a bare (inst, feats) one.

```python
import torch as t
import einops

# (inst, hidden, feats)
W = t.ones(3, 4, 2)
# (..., inst, feats) with two leading axes
big = t.ones(2, 5, 3, 2)
# no leading axes at all
small = t.ones(3, 2)

pattern = ("... inst feats, inst hidden feats"
           " -> ... inst hidden")
h_big = einops.einsum(big, W, pattern)
h_small = einops.einsum(small, W, pattern)
print(tuple(h_big.shape), tuple(h_small.shape))
# Hidden checks
assert h_big.shape == (2, 5, 3, 4)
assert h_small.shape == (3, 4)
```

## Faded practice

### q1557
One projection matrix per head, applied in a single call.

```python starter
import torch as t
import einops


def solve(x, W):
    """Per-head projection: every head gets its own matrix."""
    return einops.einsum(t.tensor(x), t.tensor(W), "batch posn d_model, _____ -> batch posn nheads d_head").tolist()
```

```python solution
import torch as t
import einops


def solve(x, W):
    """Per-head projection: every head gets its own matrix."""
    return einops.einsum(t.tensor(x), t.tensor(W), "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head").tolist()
```

## Solo practice

### q1558
Scaled attention scores: two position names, then divide by sqrt(d_head).

### q1559
A per-instance projection that works under any number of leading axes.

### q1560
Direct logit attribution: each head's output dotted with its position's correct-token direction.

## Integrated practice

### q1561
Queries for every component of a decomposed residual stream — pattern from the shapes alone.

### q1562
The feature-overlap matrix W^T W for every instance, with the same tensor passed twice.
