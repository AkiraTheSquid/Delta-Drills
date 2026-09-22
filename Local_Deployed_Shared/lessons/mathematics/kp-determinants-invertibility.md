---
kc: math.determinants-invertibility
kind: math
title: Determinants, independence, and unique solutions
supporting: [math.matrix-equations]
new_syntax: []
previews: []
concepts: []
faded: [50056, 50057]
guided: []
independent: [50058, 50059]
integrated: [50060, 50061]
---

## Concept

For a square matrix, the determinant is a scalar that detects whether its columns are independent. In two dimensions,
$$\det\begin{pmatrix}a&b\\c&d\end{pmatrix}=ad-bc.$$
Its absolute value is the area-scaling factor; a negative sign indicates reversed orientation, not failure. A nonzero determinant means the matrix is invertible: every right-hand side has exactly one solution.

Zero determinant means singular: columns are dependent. Some right-hand sides are unreachable (no solution); reachable right-hand sides have infinitely many solutions. It does not mean every system has no solution.

In 3D, absolute determinant measures volume scaling. For a triangular matrix, the determinant is the product of diagonal entries. In general, expand along the first row:
$$\det\begin{pmatrix}a&b&c\\d&e&f\\g&h&i\end{pmatrix}=a(ei-fh)-b(di-fg)+c(dh-eg).$$

These statements are exact. Numerical ray tracing uses a tolerance on $|\det M|$ to avoid unstable solves. That cutoff is an implementation convention, not the definition of singularity; determinant magnitude alone is not a scale-independent conditioning test.

## Worked example

Decide which of these systems have unique solutions.

**On paper.** First matrix:
$$\det M=\det\begin{pmatrix}2&1\\1&-1\end{pmatrix}=2\cdot(-1)-1\cdot1=-3.$$
Negative but nonzero, so every right-hand side has a unique solution.

Second matrix:
$$\det N=\det\begin{pmatrix}1&2\\2&4\end{pmatrix}=1\cdot4-2\cdot2=0.$$
Singular. Write out $N(u,v)=(3,6)$:
$$\begin{aligned}u+2v&=3\\2u+4v&=6\end{aligned}$$
The second row is twice the first, so there is really one equation: infinitely many solutions, such as $(3,0)$ and $(1,1)$. With right-hand side $(3,7)$ instead, doubling the first row gives $2u+4v=6$ while the second demands $7$: no solution.

A triangular matrix multiplies its diagonal:
$$\det\begin{pmatrix}2&5&1\\0&-1&4\\0&0&3\end{pmatrix}=2\cdot(-1)\cdot3=-6,$$
nonzero, so it is invertible too.

**In code.** $ad-bc$ read straight off the tensor entries. `t.linalg.det(regular)` gives the same value, up to floating-point rounding.

```python
import torch as t

regular = t.tensor([[2.0, 1.0],
                    [1.0, -1.0]])
singular = t.tensor([[1.0, 2.0],
                     [2.0, 4.0]])

det_regular = regular[0, 0] * regular[1, 1] - regular[0, 1] * regular[1, 0]
det_singular = singular[0, 0] * singular[1, 1] - singular[0, 1] * singular[1, 0]

print(det_regular)
print(det_singular)
# Hidden checks
assert det_regular.item() == -3.0
assert det_singular.item() == 0.0
assert abs(t.linalg.det(regular).item() + 3) < 1e-6
assert _delta_output == 'tensor(-3.)\ntensor(0.)\n'
```

## Faded practice

### q50056
Find the determinant of the matrix with rows (3, 1), (2, 4).

### q50057
A real square matrix has determinant −2. What follows?

## Solo practice

### q50058
The matrix has rows (1, 2), (2, 4), and b = (3, 6). How many solutions does Mx = b have?

### q50059
The matrix has rows (1, 2), (2, 4), and b = (3, 7). How many solutions does Mx = b have?

## Integrated practice

### q50060
Find the determinant of the matrix with rows (2, 1, 4), (0, −1, 3), (0, 0, 3).

### q50061
A 3-by-3 matrix has columns c1 = (1, 0, 0), c2 = (0, 1, 0), c3 = (1, 1, 0). What follows?
