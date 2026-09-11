---
kc: tensor.row-normalization
title: Row lengths and unit rows
new_syntax: ['Tensor.norm', 'Tensor.norm#dim', 'Tensor.norm#keepdim', 'torch.where']
concepts: [row-length, unit-rows]
supporting: ['numpy.axis-reductions', 'numpy.broadcasting-rules', 'numpy.boolean-masking', 'numpy.argmin-argmax', 'numpy.constructors']
previews: []
faded: [993, 994, 995, 996]
guided: []
independent: [997, 998, 999, 1000, 1001, 1002, 1071, 1072, 1073]
integrated: [1003, 1004, 1005]
---

## Concept: Length belongs to a vector

A row of a matrix is a vector: its entries are coordinates, and together they fix both a direction and a length. The length is the Euclidean norm — square every coordinate, add them up, take the square root — and `x.norm(dim=1)` computes it for every row at once, reducing the coordinate axis and returning one number per row, shape `(m,)`. The general rule for any reduction applies: the axis you name is the one that disappears, so `dim=1` on an `(m, n)` matrix consumes the `n` coordinates and keeps the `m` rows.

The reason to compute the length separately from the direction is that most uses want to treat them differently — compare directions while ignoring how long the vectors are, or rescale every row to a chosen length. Squaring before adding is what makes the length ignore sign: a step of `-3` is as long as a step of `3`. Passing `keepdim=True` keeps the reduced axis as size one, so the result is `(m, 1)` instead of `(m,)`; that shape is the one that broadcasts back against the rows, which the next segment relies on.

```python
import torch as t
x=t.tensor([[6.,8.],[0.,3.]])
length=x.norm(dim=1, keepdim=True)
print(length)
# Hidden checks
assert length.tolist()==[[10.],[3.]]
```

## Worked example

We measure the rows of a matrix whose rows are the classic right-triangle sides, so the lengths are whole numbers. Predict them before running: `5, 12` and `8, 15`.

```python
import torch as t
x=t.tensor([[5.,12.],[8.,15.]])
print(x.norm(dim=1))
# Hidden checks
assert t.allclose(x.norm(dim=1),t.tensor([13.,17.]))
```

Now the same reduction over the other axis. `dim=0` consumes the rows and measures each *column* as a vector, so the result has one entry per column and the numbers are different.

```python
print(x.norm(dim=0))
# Hidden checks
assert t.allclose(x.norm(dim=0),t.tensor([9.434,19.209]),atol=1e-3)
```

## Faded practice

### q993
Return row lengths, shape (m,). x: float (m,n), every row nonzero.

```python starter
import torch as t

def solve(x):
    pass
```

```python solution
import torch as t

def solve(x):
    return x.norm(dim=1)
```

### q994
Return squared row lengths, shape (m, 1). x: float (m,n), every row nonzero.

```python starter
import torch as t

def solve(x):
    pass
```

```python solution
import torch as t

def solve(x):
    return (x*x).sum(dim=1,keepdim=True)
```

## Concept: Change length, preserve direction

Dividing a nonzero vector by its own length gives a unit vector: same direction, length exactly one. For a whole matrix the division is `x / x.norm(dim=1, keepdim=True)` — the `(m, 1)` lengths broadcast across each row, so every coordinate of row `i` is divided by the same number. The reason direction is preserved is that all coordinates are scaled together; the reason `keepdim=True` is required is that an `(m,)` length would try to line up with the columns, not the rows, and either fail or divide the wrong things. Multiply a unit row by a constant to give every row that length.

A zero row has no direction, and dividing it by its length divides by zero. When a task says zero rows must be preserved, replace the zero denominators before dividing: `t.where(condition, a, b)` picks `a` where the condition holds and `b` elsewhere, so `t.where(n==0, t.ones_like(n), n)` turns each zero length into a `1` and leaves the rest alone. The zero row is then divided by one and stays zero, with nothing special-cased outside the arithmetic.

```python
import torch as t
x=t.tensor([[6.,8.],[0.,3.]])
unit=x/x.norm(dim=1,keepdim=True)
print(unit)
# Hidden checks
assert t.allclose(unit,t.tensor([[.6,.8],[0.,1.]]))
```

## Worked example

We normalize a matrix in which one row is zero. First the lengths, with `keepdim=True` so they are `(m, 1)`; the zero row has length zero, which is the denominator we must not use.

```python
import torch as t
x=t.tensor([[0.,0.],[0.,-7.]])
n=x.norm(dim=1,keepdim=True)
print(n)
# Hidden checks
assert n.tolist()==[[0.],[7.]]
```

`where` swaps the zero for a one. Dividing then leaves the zero row at zero and turns the other into a unit vector pointing along negative `y`.

```python
unit=x/t.where(n==0,t.ones_like(n),n)
print(unit)
# Hidden checks
assert unit.tolist()==[[0.,0.],[0.,-1.]]
```

## Faded practice

### q995
Return unit rows, shape (m, n). x: float (m,n), every row nonzero.

```python starter
import torch as t

def solve(x):
    pass
```

```python solution
import torch as t

def solve(x):
    return x/x.norm(dim=1,keepdim=True)
```

### q996
Return each row rescaled to length three, shape (m, n). x: float (m,n), every row nonzero.

```python starter
import torch as t

def solve(x):
    pass
```

```python solution
import torch as t

def solve(x):
    return 3*x/x.norm(dim=1,keepdim=True)
```

## Solo practice

### q997
Return the distance of each row from the origin, shape (m,). x: float (m,n), every row nonzero.

### q998
Return one unit direction per row, shape (m, n). x: float (m,n), every row nonzero.

### q999
Return how much each row must be multiplied by to become unit length, shape (m,). x: float (m,n), every row nonzero.

### q1000
Return the unit direction of the sum of all rows, shape (n,); the sum is nonzero. x: float (m,n), every row nonzero.

### q1001
Return the unit direction of each row projected onto coordinates 1 through n-1, shape (m,n-1). x: float (m,n), every row nonzero.

### q1002
Return the row index with greatest distance from the origin, as a scalar tensor; ties choose first. x: float (m,n), every row nonzero.

### q1071
Shorten rows longer than one to unit length; preserve shorter rows, shape (m,n). x: float (m,n), every row nonzero.

### q1072
Return a matrix of row-length ratios: entry i,j is length of row i divided by length of row j, shape (m,m). x: float (m,n), every row nonzero.

### q1073
Return the component of each unit row direction on the first coordinate axis, shape (m,). x: float (m,n), every row nonzero.

## Integrated practice

### q1003
Return distances between every pair of rows, shape (m,m). x: float (m,n), every row nonzero.

### q1004
Return the unit direction from the first row to every other row, shape (m-1,n); duplicate rows stay zero. x: float (m,n), every row nonzero.

### q1005
Return the length of the average of unit row directions, as a scalar tensor. x: float (m,n), every row nonzero.

## Misconceptions

- **`norm()` with no `dim` gives row lengths.** It gives one number for the whole matrix; the axis must be named.
- **An `(m,)` length divides each row.** It lines up with the columns; use `keepdim=True` to get `(m, 1)`.
- **Normalizing changes the direction.** Every coordinate is scaled by the same factor, so only the length changes.
- **A zero row normalizes to zero on its own.** It divides by zero and becomes `nan`; guard the denominator with `where`.
