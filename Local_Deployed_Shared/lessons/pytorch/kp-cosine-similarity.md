---
kc: tensor.cosine-similarity
title: Cosine similarity
new_syntax: []
concepts: [alignment, all-pairs]
supporting: ['tensor.row-normalization', 'numpy.dot-matmul-patterns', 'numpy.transpose-axes', 'numpy.argmin-argmax']
previews: []
faded: [1006, 1007, 1008, 1009]
guided: []
independent: [1010, 1011, 1012, 1013, 1014, 1015, 1074, 1075, 1076]
integrated: [1016, 1017, 1018]
---

## Concept: A dot product measures alignment

The dot product of two vectors adds up the products of matching coordinates. For unit vectors it is the cosine of the angle between them: `1` when they point the same way, `0` when they are perpendicular, `-1` when they oppose. So the procedure for comparing directions is: divide each vector by its length, then take the dot product. The result is the cosine similarity, and it lives in `[-1, 1]` whatever the original lengths were.

The reason to normalize first is that a raw dot product mixes two things — how aligned the vectors are and how long they are. Double one vector and the raw dot product doubles, though the direction did not change at all. In embeddings, search and attention, the question is "do these point the same way", and a long vector should not win merely for being long. Normalizing removes the length so only the angle remains.

```python
import torch as t
a=t.tensor([1.,0.])
b=t.tensor([0.,1.])
print(a@b, a@(-a))
# Hidden checks
assert float(a@b)==0 and float(a@(-a))==-1
```

## Worked example

We compare a vector with a stretched copy of itself. The raw dot product is large; after normalizing both, the similarity is exactly `1`, because they point the same way.

```python
import torch as t
a=t.tensor([3.,4.]); b=t.tensor([6.,8.])
print(a@b, (a/a.norm())@(b/b.norm()))
# Hidden checks
assert t.allclose((a/a.norm())@(b/b.norm()),t.tensor(1.))
```

A vector at right angles scores `0` regardless of its length — the sign of each coordinate product cancels the other. Predict this one before running.

```python
c=t.tensor([-8.,6.])
print((a/a.norm())@(c/c.norm()))
# Hidden checks
assert abs(float((a/a.norm())@(c/c.norm())))<1e-6
```

## Faded practice

### q1006
Return all row dot products, shape (m,n). x: queries (m,d); y: candidates (n,d). Float tensors with nonzero rows.

```python starter
import torch as t

def solve(x,y):
    pass
```

```python solution
import torch as t

def solve(x,y):
    return x@y.T
```

### q1007
Return lengths of every candidate row, shape (n,1). y: candidates (n,d). Float tensors with nonzero rows.

```python starter
import torch as t

def solve(y):
    pass
```

```python solution
import torch as t

def solve(y):
    return y.norm(dim=1,keepdim=True)
```

## Concept: Every row meets every other row

To compare `m` queries against `n` candidates you want an `(m, n)` table with the similarity of query `i` and candidate `j` at `[i, j]`. Give each collection its own axis: with unit rows `a` of shape `(m, d)` and `b` of shape `(n, d)`, the table is `a @ b.T`. The matrix product sums over the coordinate axis `d`, which is why that axis disappears, and it pairs every row of `a` with every column of `b.T` — that is, every row of `b`.

The reason for transposing `b` rather than `a` is a shape rule and a reading rule at once. `(m, d) @ (d, n)` is the only way the inner axes match, and the result then has queries down the rows and candidates across the columns, so `s[i]` is "everything about query `i`" and a `max(dim=1)` gives each query its best candidate. Swapping the inputs transposes the table, which is a quick sanity check when `m` and `n` differ.

```python
import torch as t
a=t.tensor([[1.,0.],[0.,1.]])
b=t.tensor([[1.,0.],[-1.,0.],[0.,1.]])
s=a@b.T
print(s)
# Hidden checks
assert s.tolist()==[[1.,-1.,0.],[0.,0.,1.]]
```

## Worked example

We score two unit queries against one unit candidate. With `m = 2` and `n = 1` the table is `(2, 1)`: one column, one similarity per query.

```python
import torch as t
a=t.tensor([[1.,0.],[0.,1.]])
b=t.tensor([[.6,.8]])
print(a@b.T)
# Hidden checks
assert t.allclose(a@b.T,t.tensor([[.6],[.8]]))
```

Swap the roles and the table is `(1, 2)`: the same two numbers laid out as a row, because now the single vector is the query. The values agree; only the orientation changes.

```python
print(b@a.T)
# Hidden checks
assert t.allclose(b@a.T,t.tensor([[.6,.8]]))
```

## Faded practice

### q1008
Return cosine similarities, shape (m,n). x: queries (m,d); y: candidates (n,d). Float tensors with nonzero rows.

```python starter
import torch as t

def solve(x,y):
    pass
```

```python solution
import torch as t

def solve(x,y):
    a=x/x.norm(dim=1,keepdim=True)
    b=y/y.norm(dim=1,keepdim=True)
    s=a@b.T
    return s
```

### q1009
Return the largest cosine similarity per query, shape (m,). x: queries (m,d); y: candidates (n,d). Float tensors with nonzero rows.

```python starter
import torch as t

def solve(x,y):
    pass
```

```python solution
import torch as t

def solve(x,y):
    a=x/x.norm(dim=1,keepdim=True)
    b=y/y.norm(dim=1,keepdim=True)
    s=a@b.T
    return s.max(dim=1)[0]
```

## Solo practice

### q1010
Return cosine similarity of each query to the first candidate, shape (m,). x: queries (m,d); y: candidates (n,d). Float tensors with nonzero rows.

### q1011
Return similarities for every candidate against every query, shape (n,m). x: queries (m,d); y: candidates (n,d). Float tensors with nonzero rows.

### q1012
Return nearest candidate indices by cosine similarity, shape (m,); ties choose first. x: queries (m,d); y: candidates (n,d). Float tensors with nonzero rows.

### q1013
Return each query’s mean similarity to the candidates, shape (m,). x: queries (m,d); y: candidates (n,d). Float tensors with nonzero rows.

### q1014
Return a Boolean matrix showing strictly opposing directions, shape (m,n). x: queries (m,d); y: candidates (n,d). Float tensors with nonzero rows.

### q1015
Return one minus cosine similarity for each pair, shape (m,n). x: queries (m,d); y: candidates (n,d). Float tensors with nonzero rows.

### q1074
Return each query’s best candidate similarity minus its mean candidate similarity, shape (m,). x: queries (m,d); y: candidates (n,d). Float tensors with nonzero rows.

### q1075
Return how many candidates point in a strictly positive direction relative to each query, shape (m,). x: queries (m,d); y: candidates (n,d). Float tensors with nonzero rows.

### q1076
Return similarities between every query and the unit direction of the sum of normalized candidate directions, shape (m,). Return zeros if candidate directions cancel. x: queries (m,d); y: candidates (n,d). Float tensors with nonzero rows.

## Integrated practice

### q1016
Return the original candidate vector closest in direction to each query, shape (m,d). x: queries (m,d); y: candidates (n,d). Float tensors with nonzero rows.

### q1017
Return Euclidean distances between normalized query and candidate directions, shape (m,n). x: queries (m,d); y: candidates (n,d). Float tensors with nonzero rows.

### q1018
Return each query’s similarity to the unit direction of the candidate centroid, shape (m,). Candidate mean is nonzero. x: queries (m,d); y: candidates (n,d). Float tensors with nonzero rows.

## Misconceptions

- **A larger dot product means more similar.** Only after normalizing; before, it also grows with length.
- **`a @ b` compares all pairs.** With two matrices of shape `(m, d)` and `(n, d)` it fails unless `m = d`; the candidate matrix must be transposed.
- **Normalizing the table normalizes the inputs.** Cosine similarity divides each *input row* by its length before the product; scaling the result afterwards is a different quantity.
- **The result can exceed `1`.** For unit rows it cannot; a value above `1` means something was not normalized.
