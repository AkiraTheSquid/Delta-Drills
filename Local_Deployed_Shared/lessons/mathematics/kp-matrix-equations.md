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
independent: [50052, 50053]
integrated: [50054, 50055]
---

## Concept

An $m$-by-$n$ matrix has $m$ rows and $n$ columns. Multiplying it by an $n$-component column vector gives $m$ components. Each output component is a row's weighted sum. Equivalently, $M\mathbf x$ combines the columns of $M$ using the entries of $\mathbf x$ as weights. This is not elementwise multiplication.

In $M\mathbf x=\mathbf b$, $\mathbf x$ holds unknown coefficients, $M$ holds their coefficients in each equation, and $\mathbf b$ holds the right-hand sides. Rows correspond to equations; columns correspond to unknowns. Keep the order of unknowns fixed. Moving a term across equality changes its sign.

Solving means finding coefficients that satisfy every equation simultaneously. Substitute a candidate back: the residual $M\mathbf x-\mathbf b$ must be zero for an exact solution. This checks the equations, not additional inequalities.

Use elimination or substitution on paper: add a multiple of one equation to another to cancel an unknown, then back-substitute. A square system need not have a unique solution; the determinant lesson explains when it does.

## Worked example

Solve $2u+v=5$ and $u-v=1$. Adding the equations cancels $v$, giving $3u=6$, hence $u=2$. Back-substitute: $2-v=1$, so $v=1$.

With unknown order $(u,v)$, the matrix is $\begin{pmatrix}2&1\\1&-1\end{pmatrix}$. Check both rows: $2(2)+1=5$ and $2-1=1$. The residual is $(0,0)$.

For a 3D triangular system, solve the last row first: $w=2$, $v+w=5$, $u+2v=7$ gives $w=2$, $v=3$, $u=1$. The principle is unchanged; there is one more coordinate equation.

```python
matrix, solution, rhs = ((2, 1), (1, -1)), (2, 1), (5, 1)
product = tuple(sum(a * x for a, x in zip(row, solution)) for row in matrix)
residual = tuple(a - b for a, b in zip(product, rhs))
print(product, residual)
# Hidden checks
assert product == rhs
assert residual == (0, 0)
assert _delta_output == '(5, 1) (0, 0)\n'
```

ARENA 0.1 transfer: continue with [intersection systems](?lesson=math.intersection-systems). These are standalone mathematics problems; no PyTorch knowledge is required. Work the answer on paper, then select one choice in practice.

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

## Integrated practice

### q50054
Solve u + 2v = 7, v + w = 5, and w = 2. Choose (u, v, w).

### q50055
A 3-by-2 matrix has two independent columns. Which statement about Mx = b is correct?
