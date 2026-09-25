---
kc: math.intersection-systems
kind: math
title: Turn an intersection into a linear system
supporting: [math.ray-geometry, math.matrix-equations]
new_syntax: []
previews: []
concepts: [columns]
faded: [50006, 50007]
guided: []
independent: [50008, 50009]
integrated: [50010, 50011, 50110]
---

## Concept: One coefficient per unknown

This lesson builds on matrix equations: products, coefficient order, elimination, and residuals. Here we turn geometry into those equations.

A matrix–vector product is a weighted sum of columns. If $M$ has columns $\mathbf c_1$ and $\mathbf c_2$, then
$$M\begin{pmatrix}u\\v\end{pmatrix}=u\,\mathbf c_1+v\,\mathbf c_2.$$
Each row is one coordinate equation; each column belongs to one unknown. Solving $M\mathbf x=\mathbf b$ finds weights $\mathbf x$ that produce $\mathbf b$.

For a ray $O+uD$ and segment $A+v(B-A)$, an intersection has both descriptions. Set them equal, subtract $O$, then move the segment displacement left:
$$O+uD=A+v(B-A)\quad\Longrightarrow\quad uD+v(A-B)=A-O.$$
Therefore
$$\underbrace{\begin{pmatrix}D & A-B\end{pmatrix}}_{M}\begin{pmatrix}u\\v\end{pmatrix}=\underbrace{A-O}_{\mathbf b}.$$
The segment column changes sign because we moved that term across the equality.

First solve the supporting lines. Then require $u\ge0$ and $0\le v\le1$. Solving alone does not enforce these inequalities. Substitute into both point formulas to check that their coordinates match. A small residual $M\mathbf x-\mathbf b$ checks the equations, but does not check membership.

A column combination with $u=3$, $v=2$:

```python
import torch as t

column_1 = t.tensor([2.0, 1.0])
column_2 = t.tensor([1.0, -3.0])
u = 3
v = 2

product = u * column_1 + v * column_2
print(product)

matrix = t.stack([column_1, column_2], dim=1)
weights = t.tensor([3.0, 2.0])
print(matrix @ weights)
# Hidden checks
assert product.tolist() == [8.0, -3.0]
assert (matrix @ weights).tolist() == [8.0, -3.0]
assert _delta_output == 'tensor([ 8., -3.])\ntensor([ 8., -3.])\n'
```

## Worked example

Does the ray from $O=(1,0)$ with direction $D=(2,1)$ cross the segment from $A=(5,-1)$ to $B=(5,5)$?

**On paper.** Build the columns and right-hand side:
$$A-B=\begin{pmatrix}0\\-6\end{pmatrix},\qquad A-O=\begin{pmatrix}4\\-1\end{pmatrix}.$$
So the system is
$$u\begin{pmatrix}2\\1\end{pmatrix}+v\begin{pmatrix}0\\-6\end{pmatrix}=\begin{pmatrix}4\\-1\end{pmatrix}\quad\Longrightarrow\quad\begin{aligned}2u&=4\\u-6v&=-1\end{aligned}$$
The first row gives $u=2$; then $2-6v=-1$, so $v=\tfrac12$. Both bounds hold: $u\ge0$ and $0\le v\le1$. Check that both descriptions reach one point:
$$O+2D=\begin{pmatrix}1\\0\end{pmatrix}+2\begin{pmatrix}2\\1\end{pmatrix}=\begin{pmatrix}5\\2\end{pmatrix},\qquad A+\tfrac12(B-A)=\begin{pmatrix}5\\-1\end{pmatrix}+\tfrac12\begin{pmatrix}0\\6\end{pmatrix}=\begin{pmatrix}5\\2\end{pmatrix}.$$

**In code.** Stack the columns, solve, then evaluate both descriptions at the solved parameters.

```python
import torch as t

o = t.tensor([1.0, 0.0])
d = t.tensor([2.0, 1.0])
a = t.tensor([5.0, -1.0])
b = t.tensor([5.0, 5.0])

matrix = t.stack([d, a - b], dim=1)
solution = t.linalg.solve(matrix, a - o)
u = solution[0]
v = solution[1]
print(solution)

ray_point = o + u * d
segment_point = a + v * (b - a)
print(ray_point)
print(segment_point)
# Hidden checks
assert t.allclose(solution, t.tensor([2.0, 0.5]))
assert t.allclose(ray_point, t.tensor([5.0, 2.0]))
assert t.allclose(segment_point, t.tensor([5.0, 2.0]))
assert _delta_output == 'tensor([2.0000, 0.5000])\ntensor([5., 2.])\ntensor([5., 2.])\n'
```

## Faded practice

### q50006
O = (2, 1), D = (1, 2), A = (6, 0), B = (6, 4). For unknowns [u, v], choose M in M[u, v] = A − O. Matrices are written as rows.

### q50007
O = (−2, 3), A = (4, 1). What is b in [D, A − B][u, v] = b?

## Solo practice

### q50008
O = (0, 1), D = (2, 0), A = (6, −1), B = (6, 3). Which [u, v] describes the crossing of O + uD and A + v(B − A)?

### q50009
Three nonsingular pairs solve to [u, v] = [0, 1], [2, −0.1], [−1, 0.5]. Which are ray–segment hits, with endpoints included?

## Integrated practice

### q50010
A nonsingular system solves to [u, v] = [2, 1.4], with residual M[u, v] − b = (0, 0). What follows?

### q50011
O = (1, 1), D = (2, −1), A = (5, −2), B = (5, 2). Does the ray hit the segment, and where?

### q50110
Ray O + uD with O = (0, 0), D = (1, 1). Segment L1 + v(L2 − L1) with L1 = (2, 0), L2 = (0, 2). Setting them equal gives the system [D, L1 − L2](u, v) = L1 − O. Which (u, v) solves it?
