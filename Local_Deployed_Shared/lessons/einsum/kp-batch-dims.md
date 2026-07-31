---
kc: einsum.batch-dims
title: Batch dimensions — carrying axes through
supporting: [einsum.matvec-matmul, einsum.outer-products]
new_syntax: []
faded: [268, 279, 310]
guided: [250]
independent: [265, 276, 297, 243, 287, 284]
---

## Concept: a batch letter runs the contraction per slice

Take a spec you know — matmul is `'ij,jk->ik'` — and prepend one letter to
every part: `'aij,ajk->aik'`. That `a` is a **batch axis**, and here is WHY it
batches:

`j` is shared and dropped, so it's the matmul's contracted axis; `i` and `k`
are its row and column. `a` is different — it appears in **both inputs AND the
output**, and it is never contracted. einsum can't merge anything across `a`,
so it simply *iterates* over it, running the whole `ij,jk->ik` matmul once at
each position of `a`. `a=0` pairs `x[0]` with `u[0]`, `a=1` pairs `x[1]` with
`u[1]`, and so on — independent products, stacked. No loop in your code; the
loop lives in compiled einsum.

## Worked example

```python
import torch as t

x = t.arange(12.0).reshape(2, 2, 3)     # (a=2, i=2, j=3)
u = t.arange(18.0).reshape(2, 3, 3)     # (a=2, j=3, k=3)

# 'a' everywhere -> run the matmul once per slice of a.
batched = t.einsum('aij,ajk->aik', x, u)
assert batched.shape == (2, 2, 3)
assert t.allclose(batched[0], x[0] @ u[0])   # slice 0 is its own matmul
assert t.allclose(batched[1], x[1] @ u[1])   # slice 1 is its own matmul
print("aij,ajk->aik :", tuple(x.shape), tuple(u.shape), "->",
      tuple(batched.shape))
print(batched)
```

Why: the per-slice asserts ARE the definition of "batched" — each `batched[k]`
equals `x[k] @ u[k]`. Write exactly that check whenever a batched spec feels
uncertain.

## Faded practice

### q268
Apply the *same* batching idea to matrix-vector products: a batch of matrices
`(b, i, j)` times a batch of vectors `(b, j)`, item n being `a[n] @ x[n]`.
(Start from matvec `'ij,j->i'` — where does the batch letter go?)

```python starter
import torch as t

def solve(a, x):
    """(b,i,j) x (b,j) -> (b,i): each matrix times ITS OWN vector."""
    return t.einsum('_____', a, x)
```

```python solution
import torch as t

def solve(a, x):
    """(b,i,j) x (b,j) -> (b,i): each matrix times ITS OWN vector."""
    return t.einsum('bij,bj->bi', a, x)
```

## Concept: omit the batch letter from one input — sharing

Now change one thing: leave the batch letter OFF one operand. `'bij,jk->bik'`
— the first factor has `b`, the second (`jk`) does not. WHY this shares:

The second operand has no `b`-axis, so it has nothing to vary as `b` moves —
einsum reuses that *same* matrix at every batch position. The base matmul
`ij,jk->ik` still runs per item, but the second factor is a single **constant**
applied to all of them. A missing batch letter means "broadcast this operand
across the batch" — no `t.tile`, no loop, just its absence.

## Worked example

```python
import torch as t

x = t.arange(12.0).reshape(2, 2, 3)     # (b=2, i=2, j=3): a batch
m = t.arange(9.0).reshape(3, 3)         # ONE (3,3) matrix — no batch axis

# 'm' has no b -> the same m multiplies every item of the batch.
shared = t.einsum('bij,jk->bik', x, m)
assert t.allclose(shared[0], x[0] @ m)
assert t.allclose(shared[1], x[1] @ m)   # same m both times
print("m has no b, so one matrix serves the whole batch:",
      tuple(m.shape), "->", tuple(shared.shape))
print(shared)
```

Why: notice what you did NOT write — no tiling of `m`, no broadcasting
gymnastics. Omitting `b` from the second operand IS the sharing.

## Faded practice

### q279
A linear layer without bias: a batch of vectors `x` of shape `(b, d)` times
ONE transformation matrix `w` of shape `(d, e)`, giving `(b, e)`. (Which
operand should be missing the batch letter?)

```python starter
import torch as t

def solve(x, w):
    """(b,d) batch of vectors, ONE (d,e) matrix w -> (b,e)."""
    return t.einsum('_____', x, w)
```

```python solution
import torch as t

def solve(x, w):
    """(b,d) batch of vectors, ONE (d,e) matrix w -> (b,e)."""
    return t.einsum('bd,de->be', x, w)
```

## Concept: drop the batch letter from the OUTPUT — summing over the batch

Third variation: keep the batch letter in both inputs, but leave it OUT of the
output. `'bi,bj->ij'` — `b` is in both inputs, absent from `ij`. WHY this sums:

Any letter dropped from the output is summed. `b` is shared across the inputs
(so the per-item outer products are formed), but dropping it from the output
collapses them: you get **Σ over the batch** of `outer(v_b, v_b)`, one `(i, j)`
matrix — not a `(b, i, j)` stack. Dropping the batch letter is how "sum over
the dataset" enters a spec.

## Worked example

```python
import torch as t

v = t.tensor([[1.0, 2.0],
              [3.0, 4.0]])               # (b=2, i=2): a batch of vectors

# 'b' in both inputs, absent from output -> outer products SUMMED over b.
gram = t.einsum('bi,bj->ij', v, v)
by_hand = t.outer(v[0], v[0]) + t.outer(v[1], v[1])
assert t.allclose(gram, by_hand)
print("b is in the inputs but not the output, so it is SUMMED away:")
print(gram)
print("outer(v0,v0) + outer(v1,v1) agrees:", bool(t.allclose(gram, by_hand)))
```

Why: two readings must agree — algebraically it's `Σ_b outer(v_b, v_b)`;
mechanically `b` is shared (multiplied along) and absent from the output
(summed). When both readings match, the spec is right.

## Faded practice

### q310
A batch of vectors `v` of shape `(b, d)` → the SINGLE `(d, d)` matrix that sums
every vector's self outer product. (The batch letter is in both inputs. Where
must it NOT appear, so that the per-item matrices collapse into one?)

```python starter
import torch as t

def solve(v):
    """(b,d) -> (d,d): sum over the batch of outer(v[n], v[n])."""
    return t.einsum('_____', v, v)
```

```python solution
import torch as t

def solve(v):
    """(b,d) -> (d,d): sum over the batch of outer(v[n], v[n])."""
    return t.einsum('bi,bj->ij', v, v)
```

## Guided practice

### q250
1. Every batch item needs its OWN similarity matrix, so the batch index
   survives to the output — it is not one of the summed indices.
2. You need two views of the same tensor: one indexed `b i d`, one `b j
   d`. Which single letter is missing from the output side?
3. `t.einsum('bid,bjd->bij', a, a)` — `d` is contracted, `b` is carried,
   `i` and `j` are the two vector slots.

## Independent practice

From the drill bank: q265 (batched matmul, the parallel case), q297 (a batch
of matrices times one constant matrix — the shared case), q276 (batched matmul
in the `bik,bkj->bij` naming), q243 (per-item outer products — batch letter
KEPT in the output), q287 (per-item SELF outer products), q284 (per-item
squared norm).

## Misconceptions

- **"Batching needs a loop or `vmap`."** — One letter, present
  everywhere, batches any contraction. The loop exists only in compiled code.
- **"All operands must mention every letter."** — An operand OMITTING the
  batch letter is broadcast across the batch — that's the shared case, not an
  error. Only the OUTPUT omitting it changes the math, by summing.
- **The three fates of a batch letter** — the whole content of batching, told
  apart by where the letter appears: in both inputs *and* the output
  (`bik,bkj->bij`) = **parallel**, independent per-item products; in one input
  only (`bij,jk->bik`) = **shared**, one constant applied to every item; in
  both inputs but dropped from the output (`bi,bj->ij`) = **summed** over the
  batch. One letter's placement is the difference between per-sample results
  and a dataset aggregate.
