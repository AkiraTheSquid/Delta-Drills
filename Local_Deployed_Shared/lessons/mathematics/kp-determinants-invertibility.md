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

In 3D, absolute determinant measures volume scaling. For a triangular matrix, the determinant is the product of diagonal entries. In general, expanding the first row of $\begin{pmatrix}a&b&c\\d&e&f\\g&h&i\end{pmatrix}$ gives $a(ei-fh)-b(di-fg)+c(dh-eg)$.

These statements are exact. Numerical ray tracing uses a tolerance on $|\det M|$ to avoid unstable solves. That cutoff is an implementation convention, not the definition of singularity; determinant magnitude alone is not a scale-independent conditioning test.

## Worked example

For $M=\begin{pmatrix}2&1\\1&-1\end{pmatrix}$, $\det M=-2-1=-3$. The determinant is negative but nonzero, so every right-hand side has a unique solution.

For $N=\begin{pmatrix}1&2\\2&4\end{pmatrix}$, $\det N=4-4=0$. The equation $N(u,v)=(3,6)$ reduces to $u+2v=3$, giving infinitely many solutions. But $N(u,v)=(3,7)$ is impossible: doubling the first equation would require $6=7$.

A 3D triangular matrix with diagonal $(2,-1,3)$ has determinant $-6$ and is also invertible.

```python
def det2(matrix):
    (a, b), (c, d) = matrix
    return a * d - b * c

regular = det2(((2, 1), (1, -1)))
singular = det2(((1, 2), (2, 4)))
print(regular, singular)
# Hidden checks
assert regular == -3
assert singular == 0
assert _delta_output == '-3 0\n'
```

ARENA 0.1 transfer: continue with [singular systems](?lesson=math.singular-systems). These are standalone mathematics problems; no PyTorch knowledge is required. Work the answer on paper, then select one choice in practice.

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
