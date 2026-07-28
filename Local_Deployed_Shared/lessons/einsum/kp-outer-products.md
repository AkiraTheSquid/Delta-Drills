---
kc: einsum.outer-products
title: Outer products — new axes from free indices
supporting: [einsum.notation-model, numpy.broadcasting-rules]
new_syntax: []
faded: [256]
guided: [278]
independent: [275, 292, 296]
---

## Concept

The dot product shared its letter. The **outer product** is the opposite
move: give each input its OWN letter and keep both —

- `'i,j->ij'` — no shared letters, nothing dropped. Every element of v1
  meets every element of v2 exactly once: entry [p, q] = v1[p] · v2[q].
  The (i, j) output is the multiplication table of the two vectors.

The general reading: **unshared kept letters multiply COMBINATORIALLY** —
the output ranges over all combinations, which is why the result's rank is
the sum of the input ranks. That single idea covers the family:

- **Self outer product**: pass the same vector twice — `'i,j->ij'` with
  (v, v) gives the symmetric all-pairs product matrix. (The two slots get
  different letters even though the DATA is the same tensor — letters label
  axes-of-slots, not variables.)
- **Tensor outer product of matrices**: `'pq,rs->prqs'` — four independent
  axes, kept in whatever output order the task demands. Axis order on the
  right is yours to choose; `(p, r, q, s)` interleaves the two inputs' axes.
- **Mixed forms**: share SOME letters, keep others —
  `'id,jd->ijd'` pairs the d axis elementwise (shared, kept!) while i and j
  combine combinatorially: entry [i, j] is the elementwise product of row i
  and row j. Shared-and-kept means "multiply along it but DON'T sum" — the
  third notational possibility, completing the set: shared+dropped = dot,
  unshared+kept = outer, shared+kept = elementwise.

Broadcasting connection: `'i,j->ij'` computes exactly
`v1[:, None] * v2[None, :]` — einsum is naming what the None-insertion
pattern built by hand in np-3.

## Worked example

Task: an outer product, its self- variant, and the shared-and-kept mixed
form.

```python
import torch as t

v1 = t.tensor([1.0, 2.0])
v2 = t.tensor([10.0, 20.0, 30.0])

# 'i,j->ij': independent letters, both kept -> all pairs.
outer = t.einsum('i,j->ij', v1, v2)
assert outer.shape == (2, 3)
assert outer.tolist() == [[10.0, 20.0, 30.0],
                          [20.0, 40.0, 60.0]]
# Same thing via broadcasting — einsum names this exact pattern:
assert t.equal(outer, v1[:, None] * v2[None, :])

# Self outer product: same array in both slots, different letters.
v = t.tensor([1.0, 2.0, 3.0])
self_outer = t.einsum('i,j->ij', v, v)
assert self_outer[1, 2] == 6.0                 # v[1] * v[2]
assert t.equal(self_outer, self_outer.T)  # symmetric by construction

# Shared AND kept: 'id,jd->ijd' — d pairs elementwise, i/j combine.
x = t.tensor([[1.0, 2.0],
              [3.0, 4.0]])                     # (t=2, d=2)
pairs = t.einsum('id,jd->ijd', x, x)
assert pairs.shape == (2, 2, 2)
assert pairs[0, 1].tolist() == [3.0, 8.0]      # row0 * row1, elementwise
```

Why each step:

1. The broadcasting equivalence is the deepest line here: outer products,
   None-insertion, and `'i,j->ij'` are one concept in three notations. If
   you can write any one, you can now write all three.
2. In the self-outer, note WHY two letters: each SLOT of the einsum gets its
   own axis names. Passing v twice with 'i,i->…' would instead pair the
   axes elementwise — a different (and here wrong) computation.
3. The mixed 'id,jd->ijd' is worth slow reading: d shared (multiply along
   it) and kept (don't sum) — so the elementwise products survive as the
   last axis. Dropping d instead ('id,jd->ij') would sum them — turning
   this into the pairwise-dots table. One letter's fate, two different
   drills.

## Faded practice

### q256
The (i, j) outer-product matrix of two vectors.

```python starter
import torch as t

def solve(v1, v2):
    """All pairwise products: independent letters, both kept."""
    return t.einsum('_____', v1, v2)
```

```python solution
import torch as t

def solve(v1, v2):
    """All pairwise products: independent letters, both kept."""
    return t.einsum('i,j->ij', v1, v2)
```

## Guided practice

### q278
1. All pairwise products of ONE vector's entries — an outer product of the
   vector with which second operand?
2. The same tensor can fill both slots; the spec doesn't change.
3. `t.einsum('i,j->ij', v, v)` — and the result should equal its own
   transpose (why?).

## Independent practice

From the drill bank: q275 (tensor product of two matrices with axis order
(p, r, q, s) — you control the output order), q292 (outer product plus a
scalar — einsum for the product, ordinary broadcasting for the +a),
q296 (the 'id,jd->ijd' mixed form on a sequence — elementwise products of
every row pair).

## Misconceptions

- **"Passing the same tensor twice needs the same letter."** — Letters name
  slot axes, not tensors. Self-outer is 'i,j->ij' with (v, v); using 'i,i'
  would elementwise-pair instead. Decide by the COMPUTATION, not the
  operand identity.
- **"Output rank = input rank."** — Unshared kept letters ADD ranks:
  vectors (1+1) → matrix, matrices (2+2) → 4-D tensor. If the output shape
  surprises you, count the distinct kept letters.
- **"Shared letters are always summed."** — Only if also dropped. Shared +
  KEPT = elementwise multiply along that axis, products retained
  ('id,jd->ijd'). The full decision table is: shared+dropped = contract,
  shared+kept = elementwise, unshared+kept = combinatorial.
