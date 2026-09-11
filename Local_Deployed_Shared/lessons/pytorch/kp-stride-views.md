---
kc: cnn.stride-views
title: Strides and as_strided views
new_syntax: ['Tensor.stride', 'Tensor.as_strided']
concepts: [strides, build-then-compute]
supporting: ['numpy.views-and-copies', 'numpy.slicing-views', 'numpy.broadcasting-rules', 'numpy.axis-reductions', 'numpy.dot-matmul-patterns', 'python.defining-functions']
previews: []
faded: [1217, 1218, 1219, 1220]
guided: []
independent: [1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229]
integrated: [1230, 1231, 1232]
---

## Concept: An axis step is a storage step

A tensor's values live in one flat storage; its shape and strides say how to find them. Shape counts the positions along each axis, and `x.stride()` returns, per axis, how many storage slots to move when that axis's index goes up by one. A fresh `(2, 3)` matrix has strides `(3, 1)`: the next row is three slots on, the next column is one. Its transpose shares the same storage and simply swaps the strides to `(1, 3)` — no value moves.

`x.as_strided(shape, strides)` builds a new view on the same storage from a shape and strides you choose. The rule for choosing them is to derive every stride from the *source's* strides, never from an assumed contiguous layout, because the source may itself be a transposed or sliced view whose strides are not `(n, 1)`. A stride of zero is legal and useful: the index on that axis advances but the address does not, so one row is seen many times without being copied. Every index the new view can reach must land inside the source's storage; `as_strided` does not check for you.

```python
import torch as t
x=t.arange(6).reshape(2,3)
print(x.stride(), x.T.stride())
# Hidden checks
assert x.stride()==(3,1) and x.T.stride()==(1,3)
```

## Worked example

We build a view that shows a vector as two identical rows. The column stride is the vector's own stride, read from `v.stride(0)`; the row stride is zero, so both rows start at the same address.

```python
import torch as t
v=t.tensor([2.,5.,8.])
rows=v.as_strided((2,3),(0,v.stride(0)))
print(rows)
# Hidden checks
assert rows.tolist()==[[2.,5.,8.],[2.,5.,8.]]
```

Because the view shares storage, writing to the source shows up in every apparent copy at once — there is only one `5.` in memory.

```python
v[1]=-1.
print(rows)
# Hidden checks
assert rows.tolist()==[[2.,-1.,8.],[2.,-1.,8.]]
```

## Faded practice

### q1217
Return a transpose view using as_strided, shape (n,m). x: float (m,n). Inputs may be non-contiguous views with a storage offset.

```python starter
import torch as t

def solve(x):
    pass
```

```python solution
import torch as t

def solve(x):
    return x.as_strided((x.shape[1],x.shape[0]),(x.stride(1),x.stride(0)))
```

### q1218
Return the main diagonal as a view of length min(m,n), using as_strided. x: float (m,n). Inputs may be non-contiguous views with a storage offset.

```python starter
import torch as t

def solve(x):
    pass
```

```python solution
import torch as t

def solve(x):
    return x.as_strided((min(x.shape),),(x.stride(0)+x.stride(1),))
```

## Concept: Build the view, then compute

Many operations are "select some addresses, then reduce". The view does the selecting: a diagonal, for instance, advances one row *and* one column per step, so its single stride is the sum of the two source strides, and its length is the shorter side. The reduction then does the arithmetic on whatever the view exposes. Keeping the two steps separate means the view can be checked by printing it before any number is computed.

The pattern extends to products. A matrix-vector product uses the same vector against every row of the matrix; a zero-stride view of the vector with the matrix's shape expresses that reuse, and `(x * vv).sum(dim=1)` then multiplies and reduces along the feature axis. The reason to reduce over `dim=1` is that the feature axis is the one whose entries were paired; the row axis must survive as one result per row. Treat overlapping views as read-only: a write through the expanded vector would land at one address and change several apparent positions.

```python
import torch as t
x=t.arange(1.,10.).reshape(3,3).T
diag=x.as_strided((3,),(x.stride(0)+x.stride(1),))
print(diag, diag.sum())
# Hidden checks
assert diag.tolist()==[1.,5.,9.] and diag.sum().item()==15
```

## Worked example

We compute a matrix-vector product with no `@`. First the view: the vector expanded to the matrix's shape with a zero row stride, so every row of the view is the same vector.

```python
import torch as t
x=t.tensor([[1.,2.],[3.,4.]])
v=t.tensor([2.,-1.])
expanded=v.as_strided(x.shape,(0,v.stride(0)))
print(expanded)
# Hidden checks
assert expanded.tolist()==[[2.,-1.],[2.,-1.]]
```

Then the arithmetic: multiply elementwise and sum across the feature axis. The result matches `x @ v`, one number per row.

```python
print((x*expanded).sum(dim=1), x@v)
# Hidden checks
assert (x*expanded).sum(dim=1).tolist()==[0.,2.] and t.equal((x*expanded).sum(dim=1),x@v)
```

## Faded practice

### q1219
Return m copies of v as a view, shape (m,n), using as_strided. x: float (m,n); v: float (n,). Inputs may be non-contiguous views with a storage offset.

```python starter
import torch as t

def solve(x,v):
    pass
```

```python solution
import torch as t

def solve(x,v):
    return v.as_strided(x.shape,(0,v.stride(0)))
```

### q1220
Return x multiplied by v, shape (m,), using only as_strided, elementwise arithmetic and sum. x: float (m,n); v: float (n,). Inputs may be non-contiguous views with a storage offset.

```python starter
import torch as t

def solve(x,v):
    pass
```

```python solution
import torch as t

def solve(x,v):
    vv=v.as_strided(x.shape,(0,v.stride(0)))
    return (x*vv).sum(dim=1)
```

## Solo practice

### q1221
Return the diagonal one step above the main diagonal as a view of length min(m,n-1), using as_strided. x: float (m,n). Inputs may be non-contiguous views with a storage offset.

### q1222
Return the outer product of v with itself, shape (n,n), using only as_strided views (stride zero) and elementwise multiplication. v: float (n,). Inputs may be non-contiguous views with a storage offset.

### q1223
Return every other column of x as a view, starting at column zero; use as_strided. x: float (m,n). Inputs may be non-contiguous views with a storage offset.

### q1224
Return a view of shape (m,n,2) whose last axis repeats each source value twice; use as_strided. x: float (m,n). Inputs may be non-contiguous views with a storage offset.

### q1225
Return the sum of the main diagonal as a scalar tensor, using as_strided and sum. x: float (m,n). Inputs may be non-contiguous views with a storage offset.

### q1226
Return the row Gram matrix, shape (m,m), using only as_strided, elementwise arithmetic and sum; do not use matmul or einsum. x: float (m,n). Inputs may be non-contiguous views with a storage offset.

### q1227
Return sums of consecutive width-two windows from each row, shape (m,n-1). n is at least two. Allowed view operation: as_strided. x: float (m,n). Inputs may be non-contiguous views with a storage offset.

### q1228
Return the feature Gram matrix, shape (n,n), using only as_strided, elementwise arithmetic and sum; do not use matmul or einsum. x: float (m,n). Inputs may be non-contiguous views with a storage offset.

### q1229
Return scalar squared length of x multiplied by v, using only as_strided, elementwise arithmetic and sum; do not use matmul or einsum. x: float (m,n); v: float (n,). Inputs may be non-contiguous views with a storage offset.

## Integrated practice

### q1230
Return x times its transpose times x, shape (m,n). Allowed tensor operations: as_strided, elementwise arithmetic and sum; do not use matmul or einsum. x: float (m,n). Inputs may be non-contiguous views with a storage offset.

### q1231
Return matrix product a (m,k) times b (k,n), shape (m,n). Inputs are float matrices, possibly noncontiguous with nonzero storage offsets. Use as_strided, elementwise arithmetic and sum; no matmul or einsum. Inputs may be non-contiguous views with a storage offset.

### q1232
Return a pairwise squared-distance table between rows of x, shape (m,m). Allowed view operation: as_strided; do not use matmul, einsum or a prebuilt distance function. x: float (m,n). Inputs may be non-contiguous views with a storage offset.

## Misconceptions

- **A `(m, n)` tensor always has strides `(n, 1)`.** Only when contiguous; a transpose or slice has different strides, and a view built from `(n, 1)` reads the wrong values.
- **A stride of zero copies the data.** It re-reads one address; nothing is allocated.
- **`as_strided` checks that the view fits.** It does not; an out-of-range view reads garbage or crashes.
- **The diagonal stride is `n + 1`.** It is `stride(0) + stride(1)`, which is `n + 1` only for a contiguous matrix.
