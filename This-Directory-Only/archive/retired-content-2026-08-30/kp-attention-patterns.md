---
kc: einsum.attention-patterns
title: Attention-shaped contractions
supporting: [einsum.batch-dims, einsum.broadcast-scaling]
new_syntax: []
faded: [299]
guided: [263]
independent: [254, 301, 305, 272, 288, 283]
---

## Concept

Transformer attention is two einsums with a softmax between them — and both
einsums are patterns you already own, at scale. The tensors:
queries/keys/values shaped (batch, heads, sequence, features) = `bhqd` /
`bhkd` / `bhkd`, where q and k index query/key positions and d is the
feature dimension.

**Scores: "every query against every key."**
`'bhqd,bhkd->bhqk'` — read it with the rules: b, h batch (everywhere);
d shared+dropped (dot product); q and k each private+kept (all pairs).
So within each (batch, head): the (q, k) table of query·key dots — a
batched pairwise-similarity matrix, precisely np-4's pairwise pattern in
einsum clothes. The single-head version drops the h: `'bqd,bkd->bqk'`.

**Mixing: "weighted average of values."**
`'bhqk,bhkd->bhqd'` — k shared+dropped: for each query, sum the value
vectors weighted by its attention row. Output back to (b, h, q, d).

Around these live the **projection** patterns — a weight matrix applied to
the last axis of a sequence tensor:

- `'btc,cf->btf'` (or `'bth,hd->btd'`) — shared weights across batch AND
  time: the batch-dims "shared operand" case with two carried letters.
- Stacked projections `'btd,nde->nbte'` — n projection matrices applied at
  once, n surviving as a new leading axis.
- Chains: `'bi,ih,ho->bo'` — a two-layer linear network in one spec (no
  nonlinearity, of course — einsum is linear algebra only).

Nothing here is new machinery. The exercise of this KP is READING these
five-letter specs fluently: batch letters ride, one letter contracts, the
rest position the output. When you can gloss `'bhqk,bhkd->bhqd'` as
"attention-weighted sum of values" at sight, this lesson has done its job.

## Worked example

Task: single-head attention scores, then the weighted value mix — on tiny
tensors where every number is checkable.

```python
import torch as t

# One batch item, sequences of 2 queries / 2 keys, feature dim 3.
q = t.tensor([[[1.0, 0.0, 0.0],
               [0.0, 1.0, 0.0]]])          # (b=1, q=2, d=3)
k = t.tensor([[[1.0, 0.0, 0.0],
               [1.0, 1.0, 0.0]]])          # (b=1, k=2, d=3)

# Scores: d contracts; q and k combine -> (1, 2, 2) table of dots.
scores = t.einsum('bqd,bkd->bqk', q, k)
assert scores.shape == (1, 2, 2)
assert scores[0].tolist() == [[1.0, 1.0],    # q0.k0, q0.k1
                              [0.0, 1.0]]    # q1.k0, q1.k1

# (Real attention: scale by 1/sqrt(d), softmax over k. Both live OUTSIDE
# einsum — they're not contractions.)
s = t.tensor([[[0.5, 0.5],
               [0.0, 1.0]]])               # a fake attention matrix (b,q,k)
v = t.tensor([[[10.0, 0.0],
               [0.0, 10.0]]])              # (b, k, d2=2)

# Mixing: k contracts -> each query gets its weighted sum of value rows.
out = t.einsum('bqk,bkd->bqd', s, v)
assert out[0, 0].tolist() == [5.0, 5.0]     # 0.5*v0 + 0.5*v1
assert out[0, 1].tolist() == [0.0, 10.0]    # 1.0*v1 — query 1 attends key 1

# Projection: one (d2, f) matrix shared across batch and sequence.
w = t.tensor([[1.0, 0.0, 0.0],
              [0.0, 1.0, 1.0]])            # (d2=2, f=3)
proj = t.einsum('bqd,df->bqf', out, w)
assert proj.shape == (1, 2, 3)
assert proj[0, 1].tolist() == [0.0, 10.0, 10.0]
print("bqd,bkd->bqk  scores", tuple(scores.shape))
print(scores[0])
print("bqk,bkd->bqd  mixed ", tuple(out.shape))
print(out[0])
print("bqd,df->bqf   proj  ", tuple(proj.shape))
print(proj[0])
```

Why each step:

1. Unit-vector queries make the score table transparent: q0 = e₁ dots both
   keys to 1; q1 = e₂ only overlaps the second key. Verify pairwise specs
   on bases first, random data second.
2. The mixing step's per-query reading ("query 1 puts all weight on key 1 →
   gets value row 1") is the semantic gloss of `'bqk,bkd->bqd'` — attach
   meanings to letters and the spec narrates itself.
3. The projection reuses the shared-operand batching: w carries no b or q,
   so one matrix serves every position. Full attention = these three
   contractions + softmax; you have now executed each part.

## Faded practice

### q299
Single-head attention scores: all query-key dots.

```python starter
import torch as t

def solve(q, k):
    """(b,q,d) x (b,k,d) -> (b,q,k): d contracts, q/k combine."""
    return t.einsum('_____', q, k)
```

```python solution
import torch as t

def solve(q, k):
    """(b,q,d) x (b,k,d) -> (b,q,k): d contracts, q/k combine."""
    return t.einsum('bqd,bkd->bqk', q, k)
```

## Guided practice

### q263
1. Same scores with a HEADS axis: (b,h,q,d) and (b,h,k,d) — which letters
   are along for the ride?
2. b and h batch; d contracts; q, k combine.
3. `'bhqd,bhkd->bhqk'` — the multi-head score tensor.

## Independent practice

From the drill bank: q254 (attention-weighted values, multi-head), q301
(pairwise similarities inside each (batch, head) — recognize it as the
score pattern), q305 (output projection of hidden states), q272 (sequence
projection), q288 (n stacked projection matrices at once — where does n go
in the output?), q283 (two-layer chain x @ w1 @ w2 in one spec).

## Misconceptions

- **"Attention einsums are special DL operations."** — They're the pairwise
  table (scores) and the weighted sum (mixing) with batch letters attached.
  If you can read 'i,j->ij' and 'ij,j->i', these are the same rules at
  rank 4.
- **"Softmax/scaling can be folded into the spec."** — einsum is
  multiply-and-sum only. Scale (÷√d) and softmax happen between the two
  contractions, in ordinary PyTorch.
- **"The letters q and k are special keywords."** — Still just names —
  but GOOD names: matching the letters to the domain (batch, head, query,
  key, depth) is what makes five-axis specs readable. Adopt the convention;
  don't imagine the engine sees it.
