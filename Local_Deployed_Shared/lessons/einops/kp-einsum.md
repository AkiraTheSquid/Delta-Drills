---
kc: einops.einsum
title: einops.einsum: name the axes, drop the ones to sum
supporting: [einops.pattern-language, numpy.axis-reductions]
new_syntax: [einops.einsum]
previews: []
concepts: [one-operand, two-operands, repeated-names, batch-axes]
faded: [847, 848, 849, 850, 851, 852, 853, 854]
guided: []
independent: [855, 856, 857, 858, 859, 860, 864, 869, 874, 879, 884, 880]
integrated: [861, 862, 863, 865, 866, 867, 868, 870, 871, 872, 873, 875, 876, 877, 878, 881, 882, 883, 885, 886, 887, 888]
---

## Concept: one operand — name every axis, drop the ones to sum

`einops.einsum` is the einops spelling of Einstein summation: the tensors come
first, the pattern comes LAST, and the pattern names every axis of every
operand on the left of `->` and lists the axes you want to KEEP on the right.
Whatever name is missing from the right-hand side is summed away.

With a single operand that gives four classics in one notation (Rocktäschel
§2.1–2.4, `a = arange(6).reshape(2, 3)`):

| pattern | result | what it is |
|---|---|---|
| `"i j -> j i"` | shape (3, 2) | transpose — same names, new order |
| `"i j ->"` | `15` | sum of everything — no name kept |
| `"i j -> j"` | `[3, 5, 7]` | column sums — `i` summed, `j` kept |
| `"i j -> i"` | `[3, 12]` | row sums — `j` summed, `i` kept |

Two things carry over from `rearrange`: names are whole words separated by
spaces (`"batch seq -> seq batch"` is fine), and the pattern is the whole
derivation — no `dim=` to look up, no `.T`.

## Worked example

We take one 2×3 matrix and ask for its total, its column sums, its row sums and its transpose — four patterns, one operand — and check that the surviving names alone decide each shape.

```python
import torch as t
import einops

a = t.arange(6).reshape(2, 3)          # (i, j)

print("a =", a.tolist())
```

The four patterns below differ only in what survives on the right of `->`.

```python
import torch as t
import einops

whole = einops.einsum(a, "i j ->")       # nothing kept -> every element summed
cols = einops.einsum(a, "i j -> j")      # i vanishes -> one number per column
rows = einops.einsum(a, "i j -> i")      # j vanishes -> one number per row
flip = einops.einsum(a, "i j -> j i")    # both kept, reordered -> transpose

assert whole.item() == 15
assert cols.tolist() == [3, 5, 7]
assert rows.tolist() == [3, 12]
assert flip.shape == (3, 2)
print(cols, rows, flip.shape)
```

Why each step: the right-hand side is a LIST OF SURVIVORS. Read `"i j -> j"`
as "for each j, sum over i" — the summed axis is the one that is not there.

## Faded practice

### q847
Return one total per matrix column.

```python starter
import torch as t
import einops

def solve(mat):
    """One total per column."""
    a = t.tensor(mat)
    return einops._____(a, "_____").tolist()
```

```python solution
import torch as t
import einops

def solve(mat):
    """One total per column."""
    a = t.tensor(mat)
    return einops.einsum(a, "i j -> j").tolist()
```

### q848
Sum away the last axis only; the other two survive in order.

```python starter
import torch as t
import einops

def solve(x):
    """Sum away the last axis only."""
    a = t.tensor(x)
    return einops._____(a, "_____").tolist()
```

```python solution
import torch as t
import einops

def solve(x):
    """Sum away the last axis only."""
    a = t.tensor(x)
    return einops.einsum(a, "b i j -> b i").tolist()
```

## Concept: two operands — a shared name multiplies, then sums

With two tensors, separate their axis lists with a comma. A name that appears
in BOTH operands pairs the elements up: they are multiplied together, and if
the name is absent from the right-hand side the products are summed. That is
all matrix multiplication is (Rocktäschel §2.5–2.6):

- `"i j, j -> i"` — matrix × vector: `j` is shared and summed, `i` survives.
- `"i j, j k -> i k"` — matrix × matrix: `j` is the contracted middle axis.

For these matrix products, the shared name has the SAME length in both
operands, matching the `(m, n) @ (n, p)` rule. More generally, einsum also
allows a shared axis of length 1 to broadcast against the other operand.

## Worked example

We multiply a matrix by a vector, then by a second matrix, and check which shared name is summed in each case.

```python
import torch as t
import einops

a = t.arange(6).reshape(2, 3)      # (i, j)
v = t.arange(3)                    # (j,)
b = t.arange(15).reshape(3, 5)     # (j, k)

print("shapes:", tuple(a.shape), tuple(v.shape), tuple(b.shape))
```

`j` is the only name both operands share. It is absent from the right, so it is the axis that gets summed.

```python
import torch as t
import einops

mv = einops.einsum(a, v, "i j, j -> i")        # == a @ v
mm = einops.einsum(a, b, "i j, j k -> i k")    # == a @ b

assert mv.tolist() == [5, 14]
assert mm.shape == (2, 5) and mm[0].tolist() == [25, 28, 31, 34, 37]
print(mv, mm.shape)
```

Why each step: `j` is named in both operands and is missing on the right, so
for every `(i)` — or every `(i, k)` — the products over `j` are added up. The
surviving names are the output shape, in the order you wrote them.

## Faded practice

### q849
The vector sits on the LEFT of the matrix this time: `vec @ mat`.

```python starter
import torch as t
import einops

def solve(vec, mat):
    """Vector on the left of a matrix."""
    v, a = t.tensor(vec), t.tensor(mat)
    return einops._____(v, a, "_____").tolist()
```

```python solution
import torch as t
import einops

def solve(vec, mat):
    """Vector on the left of a matrix."""
    v, a = t.tensor(vec), t.tensor(mat)
    return einops.einsum(v, a, "i, i j -> j").tolist()
```

### q850
Compute `mat1 @ mat2.T` without transposing anything.

```python starter
import torch as t
import einops

def solve(mat1, mat2):
    """A times B-transpose."""
    a, b = t.tensor(mat1), t.tensor(mat2)
    return einops._____(a, b, "_____").tolist()
```

```python solution
import torch as t
import einops

def solve(mat1, mat2):
    """A times B-transpose."""
    a, b = t.tensor(mat1), t.tensor(mat2)
    return einops.einsum(a, b, "i k, j k -> i j").tolist()
```

## Concept: same name, kept or repeated — elementwise, dot, outer, trace

Three more moves fall out of the same two rules — shared names multiply,
missing names sum (Rocktäschel §2.7–2.9, and the trace):

- **Keep the shared name** and nothing is summed: `"i j, i j -> i j"` is the
  elementwise (Hadamard) product.
- **Share everything and keep nothing**: `"i, i ->"` is the dot product;
  `"i j, i j ->"` is the matrix inner product (sum of all pairwise products).
- **Share nothing**: `"i, j -> i j"` is the outer product — every `i` against
  every `j`, no summation because no name is missing.
- **Repeat a name INSIDE one operand**: `"i i ->"` walks the diagonal of a
  square matrix and sums it — the trace; `"i i -> i"` is the diagonal itself.

## Worked example

We compare corresponding-entry products, all-pairs products and diagonal selection on two small vectors and one square matrix.

```python
import torch as t
import einops

a = t.arange(3)                    # [0, 1, 2]
b = t.arange(3, 6)                 # [3, 4, 5]
m = t.tensor([[1, 2], [3, 4]])

print("a =", a.tolist(), "b =", b.tolist(), "m =", m.tolist())
```

Now the same two rules on data where the answer is easy to check by hand.

```python
import torch as t
import einops

dot = einops.einsum(a, b, "i, i ->")          # 0*3 + 1*4 + 2*5
outer = einops.einsum(a, b, "i, j -> i j")     # (3, 3): a[i] * b[j]
had = einops.einsum(m, m, "i j, i j -> i j")   # m * m
print("dot", dot.item(), "| outer row 1", outer[1].tolist(), "| hadamard", had.tolist())
```

Repeating a name INSIDE one operand is a third rule: it keeps only the entries whose two coordinates are equal (the diagonal); dropping that name then sums them.

```python
import torch as t
import einops

tr = einops.einsum(m, "i i ->")                # 1 + 4

assert dot.item() == 14
assert outer[1].tolist() == [3, 4, 5]
assert had.tolist() == [[1, 4], [9, 16]]
assert tr.item() == 5
print(dot, outer.shape, tr)
```

Why each step: shared names across operands pair the entries up and multiply
them; a name repeated INSIDE one operand keeps only the entries whose two
coordinates are equal; whatever is missing on the right is summed. The outer
product shares none and drops none, so it is pure multiplication.

## Faded practice

### q851
Return the main diagonal in order.

```python starter
import torch as t
import einops

def solve(mat):
    """The main diagonal."""
    a = t.tensor(mat)
    return einops._____(a, "_____").tolist()
```

```python solution
import torch as t
import einops

def solve(mat):
    """The main diagonal."""
    a = t.tensor(mat)
    return einops.einsum(a, "i i -> i").tolist()
```

### q852
Return the sum of all corresponding-entry products of two same-shape matrices.

```python starter
import torch as t
import einops

def solve(mat1, mat2):
    """Sum of all pairwise products of two matrices."""
    a, b = t.tensor(mat1), t.tensor(mat2)
    return einops._____(a, b, "_____").item()
```

```python solution
import torch as t
import einops

def solve(mat1, mat2):
    """Sum of all pairwise products of two matrices."""
    a, b = t.tensor(mat1), t.tensor(mat2)
    return einops.einsum(a, b, "i j, i j ->").item()
```

## Concept: a batch axis rides along

A name that appears in every operand AND on the right-hand side is neither
multiplied away nor summed: it is carried through, one independent
computation per index. That is what "batched" means (Rocktäschel §2.10):

- `"b i j, b j k -> b i k"` — batch matrix multiply: for each `b`, an ordinary
  `(i, j) @ (j, k)`.
- `"b i, b i -> b"` — one dot product per row of a batch.

Compare with `"i j, j k -> i k"`: adding `b` to every operand and to the
output is the whole change. No `torch.bmm`, no `unsqueeze`, no loop.

## Worked example

We multiply two matrices by two identities in one call — one product per batch entry — then take one dot product per row of a batch.

```python
import torch as t
import einops

x = t.arange(8).reshape(2, 2, 2)     # (b, i, j): two 2x2 matrices
y = t.tensor([[1, 0], [0, 1]]).expand(2, 2, 2)   # (b, j, k): two identities

bmm = einops.einsum(x, y, "b i j, b j k -> b i k")
assert bmm.tolist() == x.tolist()             # times the identity, per batch

print("bmm shape", tuple(bmm.shape), "== x:", bmm.tolist() == x.tolist())
```

The same carry-through works with one axis fewer: one dot product per row.

```python
import torch as t
import einops

rows = t.tensor([[1, 2], [3, 4]])
dots = einops.einsum(rows, rows, "b i, b i -> b")
assert dots.tolist() == [5, 25]               # 1*1+2*2, 3*3+4*4
print(bmm.shape, dots)
```

Why each step: `b` is on the right, so nothing is summed over it; `j` (or
`i` in the dot case) is missing from the right, so that is the contraction.

## Faded practice

### q853
Pair each matrix with the vector at the same batch position.

```python starter
import torch as t
import einops

def solve(mats, vecs):
    """One matrix-vector product per batch entry."""
    a, v = t.tensor(mats), t.tensor(vecs)
    return einops._____(a, v, "_____").tolist()
```

```python solution
import torch as t
import einops

def solve(mats, vecs):
    """One matrix-vector product per batch entry."""
    a, v = t.tensor(mats), t.tensor(vecs)
    return einops.einsum(a, v, "b i j, b j -> b i").tolist()
```

### q854
Return one outer-product matrix per pair of batch vectors.

```python starter
import torch as t
import einops

def solve(u, v):
    """One outer product per batch entry."""
    a, c = t.tensor(u), t.tensor(v)
    return einops._____(a, c, "_____").tolist()
```

```python solution
import torch as t
import einops

def solve(u, v):
    """One outer product per batch entry."""
    a, c = t.tensor(u), t.tensor(v)
    return einops.einsum(a, c, "b i, b j -> b i j").tolist()
```

## Solo practice

### q855
Sum across the first axis.

### q856
Gram matrix.

### q857
A three-operand contraction.

### q858
Row-wise dot products.

### q859
Diagonal of every matrix in a batch.

### q860
Scale each row by a weight.

### q864
The trace.

### q869
Matrix times vector.

### q874
Matrix product.

### q879
Dot product.

### q884
Outer product.

### q880
Squared norm.

## Integrated practice

### q861
Trace of a product without the product.

### q862
Per-batch Gram matrices.

### q863
Query-key scores.

### q865
Trace of the square, without the square.

### q866
One trace per matrix.

### q867
Sum of squared diagonal entries.

### q868
Scaled diagonal.

### q870
Vector times matrix.

### q871
Transpose times vector, no transpose.

### q872
One matrix, many vectors.

### q873
Many matrices, one vector.

### q875
A times B-transpose, no transpose.

### q876
A-transpose times B, no transpose.

### q877
Batched matrix product.

### q878
Three-matrix chain in one call.

### q881
One dot product per row.

### q882
Weighted dot product.

### q883
Sum of all pairwise products.

### q885
Transposed outer product.

### q886
One outer product per batch entry.

### q887
Three-way outer product.

### q888
Masked outer product.

## Misconceptions

- **"The pattern goes first, like `rearrange`."** — In `einops.einsum` the
  tensors come first and the pattern is the LAST argument.
- **"A name on the right is summed."** — Backwards: the right-hand side is the
  list of SURVIVORS. Whatever is missing there is what gets summed.
- **"Shared names need `@` or `.T` as well."** — The pattern IS the derivation:
  `"i k, j k -> i j"` is `A @ B.T`, transpose included, and a name repeated
  inside one operand (`"i i ->"`) walks the diagonal.
- **"A batch axis needs `bmm`."** — Put the same batch name in every operand
  and on the right; it rides along, one product per index.
