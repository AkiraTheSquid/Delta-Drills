---
kc: einsum.dot-frobenius
title: Dot and Frobenius inner products
supporting: [einsum.reductions, numpy.dot-matmul-patterns]
new_syntax: []
concepts: [dot-shared-drop, frobenius-per-entry]
faded: [245, 267]
guided: [270]
independent: [308]
---

## Concept: the dot product 'i,i->'

So far each spec had ONE input. With two inputs, one new rule switches on:

> **A letter shared between two inputs pairs those axes elementwise** — the
> operands get multiplied along it.

Combine that with the rule you already know — a letter missing from the output
is summed away — and a letter that is *shared then dropped* means "multiply
corresponding entries, then add them up." That is exactly the dot product:

- `'i,i->'` — both vectors carry `i`, and `i` is absent from the output.
  So: pair entry `i` with entry `i`, multiply, sum. A single scalar falls out.

The empty right side (`->` with nothing after it) is what makes the answer a
scalar — every letter was summed, none survived.

This is the whole foundation: everything later (matvec, matmul, batching,
attention) is only ever combinations of these two rules — **shared letter =
multiply along it; dropped letter = sum over it.**

## Watch out

- **"`'i,i->'` is illegal because `i` is used twice."** — No. Sharing a letter
  ACROSS two inputs is the multiplication mechanism, not a name clash. (A letter
  repeated WITHIN one input, like `'ii'`, is a different legal thing — the
  diagonal — which comes later.)

## Worked example

```python
import numpy as np

v1 = np.array([1.0, 2.0, 3.0])
v2 = np.array([4.0, -5.0, 6.0])

# 'i,i->': shared i pairs the entries; the empty output sums the products.
d = np.einsum('i,i->', v1, v2)
assert d == 12.0                              # 1*4 + 2*-5 + 3*6
assert d == np.dot(v1, v2) == (v1 * v2).sum() # three spellings, one atom
```

Why: `1·4 + 2·(-5) + 3·6 = 4 - 10 + 18 = 12`. The three-spellings assert is the
bridge — when a spec confuses you, translate it back to "multiply the paired
axis, then sum" and it parses. `np.dot`, `(v1*v2).sum()`, and `'i,i->'` are the
same operation wearing three different clothes.

## Faded practice

### q245
The dot product of two vectors. There's no visible matrix or `.sum()` — just two
vectors and a scalar answer. Name the shared axis, then leave the output empty so
it collapses to a scalar.

```python starter
import numpy as np

def solve(v1, v2):
    """Dot product as a scalar: pair the shared axis, sum it away."""
    return np.einsum('_____', v1, v2)
```

```python solution
import numpy as np

def solve(v1, v2):
    """Dot product as a scalar: pair the shared axis, sum it away."""
    return np.einsum('i,i->', v1, v2)
```

## Concept: the Frobenius inner product 'ij,ij->'

The dot generalizes with no new rule — just more shared letters. Two matrices of
the SAME shape have two axes to pair, `i` and `j`:

- `'ij,ij->'` — position `[i, j]` on the first input meets its OWN position
  `[i, j]` on the second. Multiply every corresponding pair, sum them all. The
  result is a single scalar: the **Frobenius inner product**.

It's the dot product with one extra letter — the same "pair the shared axes,
drop them to sum" move, now over a 2-D grid instead of a line. (And its cousin:
dot a matrix with *itself* and square-root the result — `sqrt('ij,ij->')` — and
you have the Frobenius norm. Same spec, nothing new to learn.)

## Watch out

- **`'ij,ij'` is NOT `'ij,ji'`.** `'ij,ij->'` pairs each position with its own
  coordinates (no transpose). `'ij,ji->'` pairs `[i,j]` with `[j,i]` — a
  transposed pairing that computes `tr(a @ b)`, a completely different number.
  Read the letters exactly; don't pattern-match on "two matrices → one scalar."

## Worked example

```python
import numpy as np

a = np.array([[1.0, 2.0],
              [3.0, 4.0]])
b = np.array([[5.0, 6.0],
              [7.0, 8.0]])

# 'ij,ij->': position [i,j] meets [i,j] on both; multiply all, sum all.
f = np.einsum('ij,ij->', a, b)
assert f == 70.0                 # 1*5 + 2*6 + 3*7 + 4*8
assert f == (a * b).sum()        # the same multiply-then-reduce, named
```

Why: hand-compute one term — `1·5 = 5` at position [0,0], `2·6 = 12` at [0,1],
and so on: `5 + 12 + 21 + 32 = 70`. `ij` on BOTH inputs means [i,j] meets [i,j] —
no cross terms, no transpose. It is the dot rule with a second shared letter.

## Faded practice

### q267
The second matrix `b` is all ones. Multiplying by 1 changes nothing, so this
*looks* like "just add up every entry of `a`" — and it is. But get there through
the Frobenius spec: it's still the elementwise-multiply-and-sum of two matrices,
`a` paired position-for-position with a matrix of ones.

```python starter
import numpy as np

def solve(a, b):
    """Sum of a*b over every corresponding entry (b is all ones here)."""
    return np.einsum('_____', a, b)
```

```python solution
import numpy as np

def solve(a, b):
    """Sum of a*b over every corresponding entry (b is all ones here)."""
    return np.einsum('ij,ij->', a, b)
```

## Guided practice

### q270
1. Sum of products of corresponding elements of two SAME-SHAPE rank-3 arrays —
   how many axes need pairing now?
2. Three axes → three shared letters; the scalar answer means the output side is
   empty.
3. `'abc,abc->'` — the dot pattern generalizes by adding one letter per axis,
   nothing else changes.

## Independent practice

- q308: the dot product of ADJACENT rows of a sequence. The spec `'td,td->t'`
  keeps a letter (`t`) that survives — a first taste of the "batch letter" you'll
  meet in matvec/matmul. Here slicing makes the "adjacent" pairs; einsum does the
  per-row dots.
