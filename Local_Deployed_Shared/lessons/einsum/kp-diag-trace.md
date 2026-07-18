---
kc: einsum.diag-trace
title: Repeated indices on one operand — diagonal and trace
supporting: [einsum.notation-model, numpy.diag-triangles]
new_syntax: []
faded: [277, 257]
guided: [246]
independent: [266, 273, 298]
---

## Concept

One rule remains: **a letter repeated WITHIN a single operand walks the
diagonal** — it constrains those two axes to move together, selecting the
entries where their indices are equal:

- `'ii->i'` — both of the matrix's axes named i → only entries a[i, i]
  are visited; keeping i outputs them: the **diagonal**, as a vector.
- `'ii->'` — same selection, then i is dropped → summed: the **trace**.

Combined with everything else, this composes richly:

- **Batched diagonals/traces**: `'bii->bi'` (each matrix's diagonal),
  `'bii->b'` (each matrix's trace) — the batch letter rides along
  untouched.
- **Diagonal of a product, without the product**: the diagonal of a@b is
  Σₖ a[i,k]·b[k,i] — spec `'ik,ki->i'`: k contracts, and the OUTPUT
  constraint (only i survives, appearing in both inputs' outer positions)
  means only matching row/column pairs are computed. Cost O(n²) instead of
  the O(n³) full product.
- **Trace of a product**: drop the i too — `'ij,ji->'` computes tr(a@b)
  directly. (Note the letter pattern vs the Frobenius 'ij,ij->': transposed
  second operand. tr(a@b) = Frobenius(a, b.T) — the notation makes the
  identity visible.)

Full rule set, now complete — every einsum you will ever read is these
four:

> 1. letter dropped from output → **summed**
> 2. letter shared across operands → **multiplied along**
> 3. letter unshared and kept → **combinatorial (outer)**
> 4. letter repeated within one operand → **diagonal**

## Worked example

Task: diagonal and trace of a matrix; diagonal of a product without forming
the product.

```python
import numpy as np

a = np.array([[1.0, 2.0],
              [3.0, 4.0]])

# 'ii->i': both axes locked together -> visit a[0,0], a[1,1]; keep them.
diag = np.einsum('ii->i', a)
assert diag.tolist() == [1.0, 4.0]
assert np.array_equal(diag, np.diag(a))

# 'ii->': same walk, then sum -> the trace.
tr = np.einsum('ii->', a)
assert tr == 5.0
assert tr == np.trace(a)

# Diagonal of a @ b, computed directly: 'ik,ki->i'.
b = np.array([[5.0, 6.0],
              [7.0, 8.0]])
dprod = np.einsum('ik,ki->i', a, b)
assert np.array_equal(dprod, np.diag(a @ b))     # verified vs the slow way

# Trace of the product: drop i as well.
assert np.einsum('ij,ji->', a, b) == np.trace(a @ b)

# Batched: (batch, n, n) -> each matrix's trace, one letter more.
stack = np.stack([a, b])
assert np.einsum('bii->b', stack).tolist() == [5.0, 13.0]
```

Why each step:

1. `'ii->i'` vs `'ii->'` differ only in the output — the SELECTION (walk the
   diagonal) is rule 4; what happens to the selected values (keep vs sum) is
   the old rule 1. Factoring specs into selection+fate is how to read the
   exotic ones.
2. For `'ik,ki->i'`, trace entry 0: i=0 fixed, k ranges → Σₖ a[0,k]·b[k,0] —
   row 0 of a dotted with COLUMN 0 of b, which is precisely (a@b)[0,0].
   The spec computes only the n needed dots, not all n².
3. The batched line shows the compositionality payoff: no new function
   exists for "trace of each matrix in a stack" — but the spec is one
   letter away from the single-matrix version.

## Faded practice

### q277
The trace, via the repeated-index convention.

```python starter
import numpy as np

def solve(a):
    """Trace: walk the diagonal, sum it."""
    return np.einsum('_____', a)
```

```python solution
import numpy as np

def solve(a):
    """Trace: walk the diagonal, sum it."""
    return np.einsum('ii->', a)
```

### q257
The main diagonal, as a vector.

```python starter
import numpy as np

def solve(a):
    """Diagonal: walk it, keep it."""
    return np.einsum('_____', a)
```

```python solution
import numpy as np

def solve(a):
    """Diagonal: walk it, keep it."""
    return np.einsum('ii->i', a)
```

## Guided practice

### q246
1. The diagonal of a @ b WITHOUT computing the product — entry i is
   (row i of a) · (column i of b).
2. The contraction letter k is ordinary; the trick is that i appears in
   BOTH operands and survives — forcing the row/column indices to match.
3. `'ik,ki->i'` — check the letter positions against b's axes carefully
   (which axis of b is k, which is i?).

## Independent practice

From the drill bank: q266 (trace of each matrix in a batch), q273 (diagonal
of each matrix in a batch), q298 (tr(a@b) as one contraction — and compare
its spec with the Frobenius spec until the difference is obvious).

## Misconceptions

- **"'ii' is illegal / a typo."** — Within one operand it's the diagonal
  selector (rule 4). Across operands it's the pairing rule. Same letter,
  two well-defined meanings by placement.
- **"diag(a @ b) requires computing a @ b."** — `'ik,ki->i'` computes just
  the n diagonal dots: O(n²) vs O(n³). When a drill says "without the full
  product", this is what it's fishing for.
- **"'ij,ji->' and 'ij,ij->' are interchangeable."** — Transposed second
  operand: the first is tr(a@b), the second the Frobenius product Σ aᵢⱼbᵢⱼ.
  They agree only for symmetric b. Reading letter POSITIONS, not letter
  SETS, is the skill this lesson closes on.
