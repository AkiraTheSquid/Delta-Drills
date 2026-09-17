---
kc: einsum.diag-trace
title: Repeated indices on one operand — diagonal and trace
supporting: [einsum.notation-model, numpy.diag-triangles]
new_syntax: []
faded: [257, 277, 273, 266, 246, 298]
guided: [293]
independent: []
---

## Concept: 'ii->i' — a repeated index walks the diagonal

A letter repeated WITHIN a single operand makes those two axes move
**together** — einsum visits only the entries where their indices are equal.

`'ii->i'`: both of the matrix's axes are named `i`, so only entries `a[i, i]`
are visited; keeping `i` in the output emits them. That's the **diagonal**,
as a vector.

## Worked example

```python
import torch as t

a = t.tensor([[1.0, 2.0],
              [3.0, 4.0]])

# 'ii->i': both axes locked together -> visit a[0,0], a[1,1]; keep them.
diag = t.einsum('ii->i', a)
assert diag.tolist() == [1.0, 4.0]
assert t.equal(diag, t.diag(a))
print(a)
print("'ii->i' ->", diag, "| t.diag ->", t.diag(a))
```

Why: the repeated `i` is the whole trick — it selects the diagonal. Keeping
`i` on the output side means "emit what you selected".

## Faded practice

### q257
The main diagonal, as a vector.

```python starter
import torch as t

def solve(a):
    """Diagonal: walk it, keep it."""
    return t.einsum('_____', a)
```

```python solution
import torch as t

def solve(a):
    """Diagonal: walk it, keep it."""
    return t.einsum('ii->i', a)
```

## Concept: 'ii->' — drop the index to sum it (the trace)

Same selection as before — walk the diagonal — but now **drop** `i` from the
output. A letter dropped from the output is summed. So `'ii->'` selects the
diagonal, then sums it: the **trace**, a scalar.

Read every spec as selection + fate: `'ii->i'` and `'ii->'` select the same
entries; the only difference is keep (`->i`) vs sum (`->`).

## Worked example

```python
import torch as t

a = t.tensor([[1.0, 2.0],
              [3.0, 4.0]])

# 'ii->': same diagonal walk, then sum -> the trace.
tr = t.einsum('ii->', a)
assert tr == 5.0
assert tr == t.trace(a)
print("'ii->i' keeps the walk:", t.einsum('ii->i', a))
print("'ii->'  sums it:      ", tr, "| t.trace ->", t.trace(a))
```

Why: one character (`->i` vs `->`) flips "emit the diagonal" into "sum the
diagonal". That's the payoff of reading specs as selection + fate.

## Faded practice

### q277
The trace, via the repeated-index convention.

```python starter
import torch as t

def solve(a):
    """Trace: walk the diagonal, sum it."""
    return t.einsum('_____', a)
```

```python solution
import torch as t

def solve(a):
    """Trace: walk the diagonal, sum it."""
    return t.einsum('ii->', a)
```

## Concept: 'bii->bi' — a batch letter rides along

Add one more letter for a batch axis and the diagonal trick works per
matrix. For a stack of shape `(b, n, n)`, `'bii->bi'` extracts each matrix's
diagonal: the repeated `i` still walks each matrix's diagonal, and `b` is an
ordinary kept axis carried straight through.

No new function exists for "diagonal of every matrix in a stack" — the spec
is one letter away from the single-matrix version.

## Worked example

```python
import torch as t

stack = t.stack([t.diag(t.tensor([1.0, 2.0])),
                  t.diag(t.tensor([3.0, 4.0]))])   # shape (2, 2, 2)

# 'bii->bi': per-matrix diagonal; b rides along untouched.
diags = t.einsum('bii->bi', stack)
assert diags.tolist() == [[1.0, 2.0], [3.0, 4.0]]
print("stack", tuple(stack.shape), "-> 'bii->bi'", tuple(diags.shape))
print(diags)
```

Why: `b` is kept and unpaired, so it just indexes the batch; the `ii` does
the same diagonal work inside each slice.

## Faded practice

### q273
The diagonal of each matrix in a batch — shape (b, n).

```python starter
import torch as t

def solve(a):
    """Batched diagonal: (b, n, n) -> (b, n)."""
    return t.einsum('_____', a)
```

```python solution
import torch as t

def solve(a):
    """Batched diagonal: (b, n, n) -> (b, n)."""
    return t.einsum('bii->bi', a)
```

## Concept: 'bii->b' — batch trace (drop i, keep b)

Combine the two ideas: keep the batch letter, drop the diagonal letter.
`'bii->b'` sums each matrix's diagonal and keeps one scalar per batch entry —
the **trace of every matrix in the stack**.

## Worked example

```python
import torch as t

stack = t.stack([t.eye(2), 3 * t.eye(2)])   # traces 2 and 6

# 'bii->b': per-matrix trace; b kept, i dropped (summed).
traces = t.einsum('bii->b', stack)
assert traces.tolist() == [2.0, 6.0]
print("'bii->bi' keeps i:", t.einsum('bii->bi', stack).tolist())
print("'bii->b'  sums it:", traces)
```

Why: `b` kept + `i` dropped = "one summed diagonal per batch slice". Same
selection as `bii->bi`, different fate for `i`.

## Faded practice

### q266
The trace of each matrix in a batch — length b.

```python starter
import torch as t

def solve(a):
    """Batched trace: (b, n, n) -> (b,)."""
    return t.einsum('_____', a)
```

```python solution
import torch as t

def solve(a):
    """Batched trace: (b, n, n) -> (b,)."""
    return t.einsum('bii->b', a)
```

## Concept: 'ik,ki->i' — diagonal of a product, without the product

The diagonal of `a @ b` is `Σₖ a[i,k]·b[k,i]`. Write exactly that:
`'ik,ki->i'`. The shared `k` contracts (summed); `i` appears in BOTH operands'
outer positions and survives, forcing row `i` of `a` to pair with column `i`
of `b`. Only the `n` diagonal dots are computed — O(n²), not the O(n³) full
product.

## Worked example

```python
import torch as t

a = t.tensor([[1.0, 2.0],
              [3.0, 4.0]])
b = t.tensor([[5.0, 6.0],
              [7.0, 8.0]])

# 'ik,ki->i': entry i = row i of a . column i of b.
dprod = t.einsum('ik,ki->i', a, b)
assert t.equal(dprod, t.diag(a @ b))   # verified vs the slow way
print("a @ b in full — we only wanted its diagonal:")
print(a @ b)
print("'ik,ki->i' computes just that:", dprod)
```

Why: entry 0 is `Σₖ a[0,k]·b[k,0]` — row 0 of `a` dotted with column 0 of
`b`, i.e. `(a@b)[0,0]`. The spec computes only the n needed dots.

## Faded practice

### q246
Diagonal of a @ b, computed directly (no full product).

```python starter
import torch as t

def solve(a, b):
    """Diagonal of a @ b: entry i = row i of a . column i of b."""
    return t.einsum('_____', a, b)
```

```python solution
import torch as t

def solve(a, b):
    """Diagonal of a @ b: entry i = row i of a . column i of b."""
    return t.einsum('ik,ki->i', a, b)
```

## Concept: 'ij,ji->' — trace of a product

Drop the surviving `i` too and you sum the whole diagonal of the product:
`'ij,ji->'` computes `tr(a @ b)` directly. Note the letter positions:
`'ij,ji->'` transposes the second operand's axes relative to the Frobenius
spec `'ij,ij->'`. So `tr(a@b) = Frobenius(a, b.T)` — the notation makes the
identity visible. Read letter POSITIONS, not letter SETS.

## Worked example

```python
import torch as t

a = t.tensor([[1.0, 2.0],
              [3.0, 4.0]])
b = t.tensor([[5.0, 6.0],
              [7.0, 8.0]])

# 'ij,ji->': sum over the whole diagonal of a @ b.
assert t.einsum('ij,ji->', a, b) == t.trace(a @ b)
print("'ij,ji->' ->", t.einsum('ij,ji->', a, b).item(),
      "| t.trace(a @ b) ->", t.trace(a @ b).item())
```

Why: `'ij,ji->'` pairs `a[i,j]` with `b[j,i]` and sums everything — exactly
`Σᵢ (a@b)[i,i]`. The transposed second operand is what separates it from the
Frobenius product.

## Faded practice

### q298
Trace of a @ b as one contraction.

```python starter
import torch as t

def solve(a, b):
    """tr(a @ b) directly — one einsum, no full product."""
    return t.einsum('_____', a, b)
```

```python solution
import torch as t

def solve(a, b):
    """tr(a @ b) directly — one einsum, no full product."""
    return t.einsum('ij,ji->', a, b)
```

## Guided practice

### q293
1. Read the target carefully: a[k, 0, 0] fixes two indices at a CONSTANT.
   Einsum's subscripts can repeat an index or drop it, but they cannot pin
   one to zero.
2. So this one is not a contraction at all. Which plain indexing
   expression keeps the batch axis and takes position 0 of the other two?
3. `a[:, 0, 0]` — the drill is here to mark the boundary: repeated-letter
   tricks like `'bii->bi'` read the diagonal, not an arbitrary fixed
   position.

## Misconceptions

- **"'ii' is illegal / a typo."** — Within one operand it's the diagonal
  selector. Across operands it's the pairing rule. Same letter, two
  well-defined meanings by placement.
- **"diag(a @ b) requires computing a @ b."** — `'ik,ki->i'` computes just
  the n diagonal dots: O(n²) vs O(n³). When a drill says "without the full
  product", this is what it's fishing for.
- **"'ij,ji->' and 'ij,ij->' are interchangeable."** — Transposed second
  operand: the first is tr(a@b), the second the Frobenius product Σ aᵢⱼbᵢⱼ.
  They agree only for symmetric b. Read letter POSITIONS, not letter SETS.
