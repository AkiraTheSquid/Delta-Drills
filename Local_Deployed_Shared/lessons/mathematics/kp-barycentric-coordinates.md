---
kc: math.barycentric-coordinates
kind: math
title: Triangle coordinates and ray–triangle systems
supporting: [math.intersection-systems, math.singular-systems, math.linear-combinations]
new_syntax: []
previews: []
concepts: [weights, three-unknowns]
faded: [50018, 50019, 50022, 50023]
guided: []
independent: [50020, 50024]
integrated: [50021, 50025]
---

## Concept: Three weights, one triangle

This lesson builds on linear combinations and coordinates. Here the vectors are triangle edges, and extra bounds restrict their coefficients.

For a nondegenerate triangle with vertices $A$, $B$, $C$, begin at $A$ and combine two edges. Expanding regroups the same point as a weighted sum of vertices:
$$P=A+u(B-A)+v(C-A)=(1-u-v)A+uB+vC.$$
These three vertex weights are barycentric coordinates. They add to one.

The point belongs to the filled triangle exactly when all three weights are nonnegative:
$$u\ge0,\qquad v\ge0,\qquad u+v\le1.$$
Independent bounds $u\le1$ and $v\le1$ admit the whole parallelogram. The sum bound cuts away the half beyond edge $BC$.

Boundaries count: $u=0$ is edge $AC$, $v=0$ is edge $AB$, and $u+v=1$ is edge $BC$, provided the other weights are nonnegative. Vertices are special boundary cases. In three dimensions these weights describe the triangle's plane, not every point in space.

With $u=v=0.6$ the weight on $A$ is $1-1.2=-0.2$:

```python
import torch as t

u = 0.6
v = 0.6

weights = t.tensor([1 - u - v, u, v])
inside = (weights >= 0).all()
print(weights)
print(inside)
# Hidden checks
assert not inside
assert _delta_output == 'tensor([-0.2000,  0.6000,  0.6000])\ntensor(False)\n'
```

## Worked example

Take $A=(2,0,0)$, $B=(2,6,0)$, $C=(2,0,3)$ and $u=v=\tfrac13$. Is the point inside, and where is it?

**On paper.** The weights are $(1-u-v,\,u,\,v)=(\tfrac13,\tfrac13,\tfrac13)$, all nonnegative, so the point is inside. Build it from the edges:
$$P=\begin{pmatrix}2\\0\\0\end{pmatrix}+\tfrac13\begin{pmatrix}0\\6\\0\end{pmatrix}+\tfrac13\begin{pmatrix}0\\0\\3\end{pmatrix}=\begin{pmatrix}2\\2\\1\end{pmatrix}.$$
Check with the vertex form: $\tfrac13(A+B+C)=\tfrac13(6,6,3)=(2,2,1)$.

**In code.** The same edge combination.

```python
import torch as t

a = t.tensor([2.0, 0.0, 0.0])
b = t.tensor([2.0, 6.0, 0.0])
c = t.tensor([2.0, 0.0, 3.0])
u = 1 / 3
v = 1 / 3

point = a + u * (b - a) + v * (c - a)
print(point)
# Hidden checks
assert t.allclose(point, t.tensor([2.0, 2.0, 1.0]))
assert _delta_output == 'tensor([2., 2., 1.])\n'
```

## Faded practice

### q50018
P = A + 0.2(B − A) + 0.3(C − A). What is the weight on A?

### q50019
For a nondegenerate triangle, u = 0.8, v = 0.4. Where is A + u(B − A) + v(C − A)?

## Concept: Match a ray point to a triangle point

Set the ray point equal to the triangle point. Move the ray displacement to the triangle side, then move the fixed vertex to the other side:
$$O+sD=A+u(B-A)+v(C-A)\quad\Longrightarrow\quad -sD+u(B-A)+v(C-A)=O-A.$$
With unknowns ordered $(s,u,v)$ this is one $3\times3$ system:
$$\begin{pmatrix}-D & B-A & C-A\end{pmatrix}\begin{pmatrix}s\\u\\v\end{pmatrix}=O-A.$$
Three spatial coordinates provide three equations. Changing only the first column's sign reverses $s$. Multiplying the entire equation by $-1$ is equivalent, provided both matrix and right-hand side change.

A nonsingular solve finds the crossing with the triangle's plane. Then check $s\ge0$ and all three triangle weights nonnegative. A plane crossing can still lie outside the triangle. Batching repeats this small system for every pair; an extra batch axis introduces no extra geometric unknown. Retain the singularity mask as in the segment case.

The columns for $D=(1,0,0)$, $A=(4,0,0)$, $B=(4,2,0)$, $C=(4,0,6)$. PyTorch prints a negated zero as `-0.`; it equals $0$:

```python
import torch as t

d = t.tensor([1.0, 0.0, 0.0])
a = t.tensor([4.0, 0.0, 0.0])
b = t.tensor([4.0, 2.0, 0.0])
c = t.tensor([4.0, 0.0, 6.0])

matrix = t.stack([-d, b - a, c - a], dim=1)
print(matrix)
# Hidden checks
assert matrix.tolist() == [[-1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 6.0]]
assert _delta_output == 'tensor([[-1.,  0.,  0.],\n        [-0.,  2.,  0.],\n        [-0.,  0.,  6.]])\n'
```

## Worked example

Does the ray from $O=(0,1,1)$ with $D=(2,0,0)$ hit the triangle $A=(6,0,0)$, $B=(6,4,0)$, $C=(6,0,4)$?

**On paper.** The columns and right-hand side:
$$-D=\begin{pmatrix}-2\\0\\0\end{pmatrix},\quad B-A=\begin{pmatrix}0\\4\\0\end{pmatrix},\quad C-A=\begin{pmatrix}0\\0\\4\end{pmatrix},\quad O-A=\begin{pmatrix}-6\\1\\1\end{pmatrix}.$$
The system is diagonal, so each row solves one unknown:
$$\begin{pmatrix}-2&0&0\\0&4&0\\0&0&4\end{pmatrix}\begin{pmatrix}s\\u\\v\end{pmatrix}=\begin{pmatrix}-6\\1\\1\end{pmatrix}\quad\Longrightarrow\quad s=3,\;u=\tfrac14,\;v=\tfrac14.$$
All checks pass: $s\ge0$, $u,v\ge0$, and $u+v=\tfrac12\le1$. The hit point is
$$O+3D=\begin{pmatrix}0\\1\\1\end{pmatrix}+3\begin{pmatrix}2\\0\\0\end{pmatrix}=\begin{pmatrix}6\\1\\1\end{pmatrix}.$$

**In code.** Stack the columns, solve for $(s,u,v)$, then the hit point and the four checks.

```python
import torch as t

o = t.tensor([0.0, 1.0, 1.0])
d = t.tensor([2.0, 0.0, 0.0])
a = t.tensor([6.0, 0.0, 0.0])
b = t.tensor([6.0, 4.0, 0.0])
c = t.tensor([6.0, 0.0, 4.0])

matrix = t.stack([-d, b - a, c - a], dim=1)
solution = t.linalg.solve(matrix, o - a)
s = solution[0]
u = solution[1]
v = solution[2]
print(solution)

point = o + s * d
hit = (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)
print(point)
print(hit)
# Hidden checks
assert t.allclose(solution, t.tensor([3.0, 0.25, 0.25]))
assert t.allclose(point, t.tensor([6.0, 1.0, 1.0]))
assert hit
assert _delta_output == 'tensor([3.0000, 0.2500, 0.2500])\ntensor([6., 1., 1.])\ntensor(True)\n'
```

## Faded practice

### q50022
With unknowns [s, u, v] and right-hand side O − A, which columns describe ray–triangle equality?

### q50023
A nonsingular ray–triangle system gives [s, u, v] = [2, 0.7, 0.6]. Does it hit the filled triangle?

## Solo practice

### q50020
Which triangle boundary contains u = 0.25, v = 0.75?

### q50024
A nonsingular ray–triangle system gives [s, u, v] = [0, 0, 1]. With boundaries included, what results?

## Integrated practice

### q50021
A = (1, 0, 0), B = (1, 8, 0), C = (1, 0, 4). Which point has u = 0.25, v = 0.5?

### q50025
O = (1, 1, 1), D = (2, 0, 0), A = (5, 0, 0), B = (5, 4, 0), C = (5, 0, 4). Which [s, u, v] is correct?
