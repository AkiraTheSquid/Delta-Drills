---
kc: einops.einsum
title: einops.einsum: name the axes, drop the ones to sum
supporting: [einops.pattern-language, numpy.axis-reductions]
new_syntax: [einops.einsum]
previews: []
concepts: [one-operand, two-operands, repeated-names, batch-axes]
faded: [847, 848, 849, 850, 851, 852, 853, 854]
guided: []
independent: [855, 856, 857, 858, 859, 860]
integrated: [861, 862, 863, 864, 869, 874, 879, 884, 865, 866, 867, 868, 870, 871, 872, 873, 875, 876, 877, 878, 880, 881, 882, 883, 885, 886, 887, 888]
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

```python
import torch as t
import einops

a = t.arange(6).reshape(2, 3)          # (i, j)

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
Which axis disappears when you keep only `j`?

```python starter
import torch as t
import einops

def solve(mat):
    """Column sums via einsum."""
    a = t.tensor(mat)
    return einops._____(a, "_____").tolist()
```

```python solution
import torch as t
import einops

def solve(mat):
    """Column sums via einsum."""
    a = t.tensor(mat)
    return einops.einsum(a, "i j -> j").tolist()
```

### q848
Three names on the left; two survive. The one to drop is the last.

```python starter
import torch as t
import einops

def solve(x):
    """Sum the last axis of a 3-D tensor via einsum."""
    a = t.tensor(x)
    return einops._____(a, "_____").tolist()
```

```python solution
import torch as t
import einops

def solve(x):
    """Sum the last axis of a 3-D tensor via einsum."""
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

```python
import torch as t
import einops

a = t.arange(6).reshape(2, 3)      # (i, j)
v = t.arange(3)                    # (j,)
b = t.arange(15).reshape(3, 5)     # (j, k)

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
The shared axis is the vector's ONLY axis, and it is the matrix's first one this time.

```python starter
import torch as t
import einops

def solve(vec, mat):
    """Vector times matrix via einsum."""
    v, a = t.tensor(vec), t.tensor(mat)
    return einops._____(v, a, "_____").tolist()
```

```python solution
import torch as t
import einops

def solve(vec, mat):
    """Vector times matrix via einsum."""
    v, a = t.tensor(vec), t.tensor(mat)
    return einops.einsum(v, a, "i, i j -> j").tolist()
```

### q850
Both operands end in the same axis. Name it the same, and the transpose is free.

```python starter
import torch as t
import einops

def solve(mat1, mat2):
    """A times B-transpose via einsum."""
    a, b = t.tensor(mat1), t.tensor(mat2)
    return einops._____(a, b, "_____").tolist()
```

```python solution
import torch as t
import einops

def solve(mat1, mat2):
    """A times B-transpose via einsum."""
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

```python
import torch as t
import einops

a = t.arange(3)                    # [0, 1, 2]
b = t.arange(3, 6)                 # [3, 4, 5]
m = t.tensor([[1, 2], [3, 4]])

dot = einops.einsum(a, b, "i, i ->")          # 0*3 + 1*4 + 2*5
outer = einops.einsum(a, b, "i, j -> i j")     # (3, 3): a[i] * b[j]
had = einops.einsum(m, m, "i j, i j -> i j")   # m * m
tr = einops.einsum(m, "i i ->")                # 1 + 4

assert dot.item() == 14
assert outer[1].tolist() == [3, 4, 5]
assert had.tolist() == [[1, 4], [9, 16]]
assert tr.item() == 5
print(dot, outer.shape, tr)
```

Why each step: the SAME two rules decide every case — count which names are
shared (those multiply) and which are missing on the right (those sum). The
outer product shares none and drops none, so it is pure multiplication.

## Faded practice

### q851
The trace repeats a name and keeps nothing. Keep it instead.

```python starter
import torch as t
import einops

def solve(mat):
    """The diagonal via a repeated index."""
    a = t.tensor(mat)
    return einops._____(a, "_____").tolist()
```

```python solution
import torch as t
import einops

def solve(mat):
    """The diagonal via a repeated index."""
    a = t.tensor(mat)
    return einops.einsum(a, "i i -> i").tolist()
```

### q852
Both names shared, both dropped.

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

```python
import torch as t
import einops

x = t.arange(8).reshape(2, 2, 2)     # (b, i, j): two 2x2 matrices
y = t.tensor([[1, 0], [0, 1]]).expand(2, 2, 2)   # (b, j, k): two identities

bmm = einops.einsum(x, y, "b i j, b j k -> b i k")
assert bmm.tolist() == x.tolist()             # times the identity, per batch

rows = t.tensor([[1, 2], [3, 4]])
dots = einops.einsum(rows, rows, "b i, b i -> b")
assert dots.tolist() == [5, 25]               # 1*1+2*2, 3*3+4*4
print(bmm.shape, dots)
```

Why each step: `b` is on the right, so nothing is summed over it; `j` (or
`i` in the dot case) is missing from the right, so that is the contraction.

## Faded practice

### q853
Matrix–vector, with a `b` carried through both operands and the output.

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
Outer product shares no name — except the batch, which is carried through.

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
Sum across the batch axis.

### q856
Gram matrix A A^T.

### q857
A three-operand contraction.

### q858
Row-wise dot products.

### q859
Diagonals of a batch.

### q860
Row scaling as an einsum.

## Integrated practice

### q861
Trace of a product without the product.

### q862
Per-batch Gram matrices.

### q863
Query-key scores.

### q864
einsum_trace

### q869
einsum_mv

### q874
einsum_mm

### q879
einsum_inner

### q884
einsum_outer

### q865
trace of A squared

### q866
batch of traces

### q867
sum of squared diagonal

### q868
scaled diagonal

### q870
vector-matrix product

### q871
transpose-then-multiply

### q872
one matrix, many vectors

### q873
many matrices, one vector

### q875
A times B transpose

### q876
A transpose times B

### q877
batch matmul

### q878
three-matrix chain

### q880
squared norm

### q881
batched dot products

### q882
weighted dot product

### q883
matrix inner product

### q885
transposed outer product

### q886
batched outer products

### q887
three-way outer product

### q888
masked outer product

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
