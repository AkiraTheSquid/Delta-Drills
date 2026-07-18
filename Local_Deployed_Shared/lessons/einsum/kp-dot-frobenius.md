---
kc: einsum.dot-frobenius
title: Dot and Frobenius inner products
supporting: [einsum.reductions, numpy.dot-matmul-patterns]
new_syntax: []
faded: [245]
guided: [267]
independent: [270, 308]
---

## Concept

Two inputs enter the spec, and one more rule activates:

> **A letter shared between two inputs pairs those axes elementwise —
> the operands are multiplied along it.** Combined with the deletion rule
> (missing from output → summed), a shared-then-dropped letter means
> "multiply corresponding entries, then add them up."

That sentence is the dot product — and its spec reads exactly that way:

- `'i,i->'` — both vectors' axes share `i`; `i` is absent from the output.
  Multiply pairwise, sum: the **dot product**, as a scalar.
- `'ij,ij->'` — same story with two letters: multiply corresponding entries
  of two same-shape matrices, sum everything — the **Frobenius inner
  product** (whose square root over one matrix with itself is the Frobenius
  norm you met in np-3).
- `'abc,abc->'` — identical shape, any rank: the pattern generalizes by
  adding letters, nothing else changes.

This KP is deliberately narrow — one new rule — because everything that
follows (matmul, batching, attention) is only ever combinations of the two
rules you now hold:

> **shared letter = multiply along it; dropped letter = sum over it.**

The multiply-then-reduce decomposition from np-3's dot-patterns KP
(`(a * b).sum()`) is exactly what these specs notate — einsum is that
pattern with names.

## Worked example

Task: a dot product, and a Frobenius inner product of two matrices — plus
the decomposition check.

```python
import numpy as np

v1 = np.array([1.0, 2.0, 3.0])
v2 = np.array([4.0, -5.0, 6.0])

# 'i,i->': shared i pairs the entries; missing i sums the products.
d = np.einsum('i,i->', v1, v2)
assert d == 12.0
assert d == np.dot(v1, v2) == (v1 * v2).sum()    # three spellings, one atom

# 'ij,ij->': the same, per-entry over a 2-D shape.
a = np.array([[1.0, 2.0],
              [3.0, 4.0]])
b = np.array([[5.0, 6.0],
              [7.0, 8.0]])
f = np.einsum('ij,ij->', a, b)
assert f == 70.0                                  # 5 + 12 + 21 + 32
assert f == (a * b).sum()

# Frobenius NORM of one matrix = sqrt of its inner product with itself.
assert np.isclose(np.sqrt(np.einsum('ij,ij->', a, a)),
                  np.linalg.norm(a))
```

Why each step:

1. The three-spellings assert (`einsum == dot == multiply+sum`) is the
   bridge between notations — when a spec confuses you, translate it back
   to multiply-then-reduce and it will parse.
2. Hand-computing one Frobenius term (1·5 = 5, 2·6 = 12, …) once makes
   "corresponding entries" concrete: `ij` on BOTH inputs means position
   [i, j] meets position [i, j] — no transpose, no cross terms.
3. The norm line ties this KP to np-3: new notation, previously-learned
   quantity. Expect drills to phrase it either way.

## Faded practice

### q245
The dot product of two vectors.

```python starter
import numpy as np

def solve(v1, v2):
    """Dot product: shared letter, empty output."""
    return np.einsum('_____', v1, v2)
```

```python solution
import numpy as np

def solve(v1, v2):
    """Dot product: shared letter, empty output."""
    return np.einsum('i,i->', v1, v2)
```

## Guided practice

### q267
1. Sum of products of corresponding entries of two SAME-SHAPE matrices —
   which letters are shared, and what's left in the output?
2. Two axes to pair: both get shared letters; the scalar answer means the
   output side is empty.
3. `'ij,ij->'` — the 2-D dot product, a.k.a. Frobenius inner product.

## Independent practice

From the drill bank: q270 (the same contraction at rank 3 — add a letter),
q308 (dot products of ADJACENT rows of a sequence — the spec is `'td,td->t'`
applied to two shifted slices of the same array; the slicing does the
"adjacent", einsum does the dots).

## Misconceptions

- **"'i,i->' errors because i is used twice."** — Sharing a letter ACROSS
  inputs is the multiplication mechanism, not a clash. (A letter repeated
  WITHIN one input — 'ii' — is a different, also-legal thing: the diagonal,
  coming two KPs from now.)
- **"The Frobenius product involves a transpose somewhere."** — No:
  positions meet their own coordinates ('ij,ij'). The transposed pairing
  'ij,ji' is a DIFFERENT operation (it computes tr(a@b) — later drill).
  Letter patterns are precise; read them, don't pattern-match on vibes.
- **"einsum with two inputs must produce an array."** — An empty output
  (`->`) is legal and yields a scalar. Dot products are the canonical case.
