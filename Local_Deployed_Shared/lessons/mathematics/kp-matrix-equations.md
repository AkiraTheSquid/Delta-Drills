---
kc: math.matrix-equations
kind: math
title: Matrix–vector products and solving linear systems
supporting: [math.linear-combinations]
new_syntax: []
previews: []
concepts: []
faded: [50050, 50051]
guided: []
independent: [50052, 50053, 50104, 50105, 50106, 50107, 50108, 50109]
integrated: [50054, 50055, 50110, 50111]
---

## Concept

An $m$-by-$n$ matrix has $m$ rows and $n$ columns. Multiplying it by an $n$-component column vector gives $m$ components. Each output component is a row's weighted sum; equivalently, $M\mathbf x$ combines the columns of $M$ using the entries of $\mathbf x$ as weights:
$$\begin{pmatrix}a&b\\c&d\end{pmatrix}\begin{pmatrix}x\\y\end{pmatrix}=\begin{pmatrix}ax+by\\cx+dy\end{pmatrix}=x\begin{pmatrix}a\\c\end{pmatrix}+y\begin{pmatrix}b\\d\end{pmatrix}.$$
This is not elementwise multiplication.

In $M\mathbf x=\mathbf b$, $\mathbf x$ holds unknown coefficients, $M$ holds their coefficients in each equation, and $\mathbf b$ holds the right-hand sides. Rows correspond to equations; columns correspond to unknowns. Keep the order of unknowns fixed. Moving a term across equality changes its sign.

Solving means finding coefficients that satisfy every equation simultaneously. Substitute a candidate back: the residual $M\mathbf x-\mathbf b$ must be zero for an exact solution. This checks the equations, not additional inequalities.

Use elimination or substitution on paper: add a multiple of one equation to another to cancel an unknown, then back-substitute. A square system need not have a unique solution; the determinant lesson explains when it does.

## Worked example

Solve
$$\begin{aligned}2u+v&=5\\u-v&=1\end{aligned}$$

**On paper.** Add the equations to cancel $v$: $3u=6$, so $u=2$. Back-substitute into the second: $2-v=1$, so $v=1$.

In matrix form, with unknown order $(u,v)$:
$$\begin{pmatrix}2&1\\1&-1\end{pmatrix}\begin{pmatrix}u\\v\end{pmatrix}=\begin{pmatrix}5\\1\end{pmatrix}.$$
Check by multiplying row by row:
$$\begin{pmatrix}2&1\\1&-1\end{pmatrix}\begin{pmatrix}2\\1\end{pmatrix}=\begin{pmatrix}2\cdot2+1\cdot1\\1\cdot2-1\cdot1\end{pmatrix}=\begin{pmatrix}5\\1\end{pmatrix},$$
so the residual is $(0,0)$.

For a 3D triangular system, solve the last row first and work upward:
$$\begin{aligned}u+2v&=7\\v+w&=5\\w&=2\end{aligned}\quad\Longrightarrow\quad w=2,\;v=3,\;u=1.$$
The principle is unchanged; there is one more coordinate equation.

**In code.** `t.linalg.solve` does the elimination; `@` is the matrix–vector product, so the residual is one line.

```python
import torch as t

matrix = t.tensor([[2.0, 1.0],
                   [1.0, -1.0]])
rhs = t.tensor([5.0, 1.0])

solution = t.linalg.solve(matrix, rhs)
residual = matrix @ solution - rhs

print(solution)
print(residual)
# Hidden checks
assert t.allclose(solution, t.tensor([2.0, 1.0]))
assert t.allclose(residual, t.zeros(2))
assert _delta_output == 'tensor([2., 1.])\ntensor([0., 0.])\n'
```

## Faded practice

### q50050
M has rows (1, 2) and (−1, 3). Find M(2, −1).

### q50051
For unknown order (u, v), which matrix encodes 3u − 2v = 7 and u + 4v = 5? Matrices are written as rows.

## Solo practice

### q50052
Solve u + v = 5 and 2u − v = 1. Choose (u, v).

### q50053
For M with rows (2, 1), (1, −1), candidate x = (1, 2), and b = (5, 1), find the residual Mx − b.

### q50104
M has rows (2, −1, 0) and (1, 3, 2). Find Mx for x = (1, 2, −1).

### q50105
Rewrite 2u = 3v + 4 and v − u = 1 as M(u, v) = (4, 1). Which M? Matrices are written as rows.

### q50106
M has shape (4, 3). For M @ x to work, what shape must x have, and what shape is the result?

### q50107
Solve 3u + 2v = 12 and u − 2v = −4. Choose (u, v).

### q50108
M = t.tensor([[1., 2.], [3., 4.]]), x = t.tensor([1., -1.]). What is M @ x?

### q50109
M has rows (1, 2) and (3, −1), b = (3, 3). Candidate x = (1, 1). What is the residual Mx − b?

## Integrated practice

### q50054
Solve u + 2v = 7, v + w = 5, and w = 2. Choose (u, v, w).

### q50055
A 3-by-2 matrix has two independent columns. Which statement about Mx = b is correct?

### q50110
Ray O + uD with O = (0, 0), D = (1, 1). Segment L1 + v(L2 − L1) with L1 = (2, 0), L2 = (0, 2). Setting them equal gives the system [D, L1 − L2](u, v) = L1 − O. Which (u, v) solves it?

### q50111
A holds one 2-by-2 system per ray: A.shape = (5, 2, 2), b.shape = (5, 2). What shape does t.linalg.solve(A, b) return?
