---
kc: math.ray-geometry
kind: math
title: Points, directions, and finite segments
supporting: [math.vector-arithmetic]
new_syntax: []
previews: []
concepts: [affine]
faded: [50000, 50001]
guided: []
independent: [50002, 50003]
integrated: [50004, 50005]
---

## Concept: A position plus a displacement

This lesson builds on vector arithmetic: components, displacement, and scalar multiplication. Here those skills describe rays.

A point names a location; a vector names a change in location. Subtracting two points gives a displacement: from $A$ to $B$, move by $B-A$. Adding a displacement to a point gives another point.

A ray is every point
$$P(s)=O+sD,\qquad s\ge0,$$
where $O$ is its origin and $D$ a nonzero direction. $D$ is a displacement, even when ARENA stores it in the second row of a $(2,3)$ tensor. It is not an endpoint: given an endpoint $Q$, first compute $D=Q-O$. At $s=0$ you are at $O$; negative $s$ lies behind the ray.

A segment from $A$ to $B$ is
$$P(v)=(1-v)A+vB=A+v(B-A),\qquad 0\le v\le1.$$
The weights $1-v$ and $v$ add to one. The supporting line allows any real $v$; the segment does not.

A camera with $n\ge2$ pixels and y-limit $L$ at $x=1$ has equally spaced direction-y values
$$y_i=-L+i\cdot\frac{2L}{n-1},\qquad i=0,\dots,n-1.$$
There are $n-1$ gaps between $n$ endpoints. ARENA uses $D_x=1$, so $s=x$ when $O=0$.

The ray point $O+2D$ for $O=(2,-1)$, $D=(3,2)$. Tensor arithmetic acts on every component at once:

```python
import torch as t

origin = t.tensor([2.0, -1.0])
direction = t.tensor([3.0, 2.0])
s = 2

point = origin + s * direction
print(point)
# Hidden checks
assert point.tolist() == [8.0, 3.0]
assert _delta_output == 'tensor([8., 3.])\n'
```

## Worked example

Find the midpoint of the segment from $A=(-2,3)$ to $B=(6,7)$, and the point that a ray from $O=(2,-1)$ with direction $D=(3,2)$ reaches at $s=2$.

**On paper.** The midpoint is $v=\tfrac12$. Subtract, halve, add:
$$A+\tfrac12(B-A)=\begin{pmatrix}-2\\3\end{pmatrix}+\tfrac12\begin{pmatrix}6-(-2)\\7-3\end{pmatrix}=\begin{pmatrix}-2\\3\end{pmatrix}+\begin{pmatrix}4\\2\end{pmatrix}=\begin{pmatrix}2\\5\end{pmatrix}.$$
Check with the weighted form: $\tfrac12A+\tfrac12B=\tfrac12(4,10)=(2,5)$.

For the ray, scale the direction and add it to the origin:
$$O+2D=\begin{pmatrix}2\\-1\end{pmatrix}+2\begin{pmatrix}3\\2\end{pmatrix}=\begin{pmatrix}2+6\\-1+4\end{pmatrix}=\begin{pmatrix}8\\3\end{pmatrix}.$$

**In code.** The midpoint, as one tensor expression.

```python
import torch as t

a = t.tensor([-2.0, 3.0])
b = t.tensor([6.0, 7.0])
v = 0.5

point = a + v * (b - a)
print(point)
# Hidden checks
assert point.tolist() == [2.0, 5.0]
assert _delta_output == 'tensor([2., 5.])\n'
```

## Faded practice

### q50000
O = (3, −2), Q = (7, 4). A ray starts at O and passes through Q. Which D makes O + D = Q?

### q50001
P(s) = (2, 1) + s(−1, 2), s ≥ 0. Does (3, −1) belong to this ray?

## Solo practice

### q50002
A = (1, 2), B = (5, 6). What does v = 1.25 give in A + v(B − A)?

### q50003
A ray is written O + cD, where the scalar c is its parameter. The point P = O + 6D sits at c = 6. Rewrite the same ray as O + sD′ with a longer direction D′ = 3D (same O). For which value of s is O + sD′ = O + 6D?

## Integrated practice

### q50004
A five-pixel camera includes both y-limits −3 and 3 at x = 1. Which sequence contains the direction-y coordinates?

### q50005
A point has v = 0.2 from A to B. Swap the endpoints and write P = B + w(A − B). What is w?
