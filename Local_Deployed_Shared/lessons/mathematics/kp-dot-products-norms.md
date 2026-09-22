---
kc: math.dot-products-norms
kind: math
title: Dot products, lengths, and unit vectors
supporting: [math.vector-arithmetic]
new_syntax: []
previews: []
concepts: []
faded: [50038, 50039]
guided: []
independent: [50040, 50041]
integrated: [50042, 50043]
---

## Concept

The dot product pairs corresponding components, then adds:
$$\mathbf u\cdot\mathbf v=\sum_i u_i v_i.$$
The result is a scalar, not a vector. Dotting a vector with itself gives its squared Euclidean length, and the length (norm) is its square root:
$$\mathbf v\cdot\mathbf v=\sum_i v_i^2,\qquad \|\mathbf v\|=\sqrt{\mathbf v\cdot\mathbf v}.$$
Negative components still contribute positive squares.

For nonzero $\mathbf v$, the unit vector
$$\hat{\mathbf v}=\frac{\mathbf v}{\|\mathbf v\|}$$
has length one and the same direction. Zero cannot be normalized: its norm is zero. Scaling obeys $\|c\mathbf v\|=|c|\,\|\mathbf v\|$.

Two nonzero vectors with dot product zero are perpendicular. For core ray tracing, the essential use here is length: the distance between points is $\|B-A\|$, and displacement $sD$ has length $|s|\,\|D\|$. A parameter is a distance only when the direction has unit length (and the parameter is nonnegative).

## Worked example

Find the length and unit vector of $\mathbf v=(2,-3,6)$, then the distance from $A=(1,2)$ to $B=(4,6)$.

**On paper.** Dot $\mathbf v$ with itself, then take the square root:
$$\mathbf v\cdot\mathbf v=\begin{pmatrix}2\\-3\\6\end{pmatrix}\cdot\begin{pmatrix}2\\-3\\6\end{pmatrix}=4+9+36=49,\qquad \|\mathbf v\|=\sqrt{49}=7.$$
Divide every component by the length, not by its square:
$$\hat{\mathbf v}=\frac17\begin{pmatrix}2\\-3\\6\end{pmatrix}=\begin{pmatrix}2/7\\-3/7\\6/7\end{pmatrix}.$$
For the distance, subtract first, then measure:
$$B-A=\begin{pmatrix}4\\6\end{pmatrix}-\begin{pmatrix}1\\2\end{pmatrix}=\begin{pmatrix}3\\4\end{pmatrix},\qquad \|B-A\|=\sqrt{3^2+4^2}=5,$$
not $3+4=7$.

**In code.** `v @ v` is the dot product; `t.linalg.norm` does the square-root step for you. The unit vector's norm comes out as one.

```python
import torch as t

v = t.tensor([2.0, -3.0, 6.0])

squared_length = v @ v
length = t.sqrt(squared_length)
unit = v / length

print(squared_length)
print(length)
print(t.linalg.norm(unit))

a = t.tensor([1.0, 2.0])
b = t.tensor([4.0, 6.0])
distance = t.linalg.norm(b - a)
print(distance)
# Hidden checks
assert squared_length.item() == 49.0
assert length.item() == 7.0
assert abs(t.linalg.norm(unit).item() - 1) < 1e-6
assert distance.item() == 5.0
assert _delta_output == 'tensor(49.)\ntensor(7.)\ntensor(1.)\ntensor(5.)\n'
```

## Faded practice

### q50038
Find (1, −2, 3) · (4, 1, −1).

### q50039
Find the Euclidean norm of (−3, 4).

## Solo practice

### q50040
Which unit vector has the same direction as (0, −3, 4)?

### q50041
u = (2, 1), v = (1, −2). What does u · v = 0 imply?

## Integrated practice

### q50042
A = (−1, 2, 0), B = (1, −1, 6). What is their Euclidean distance?

### q50043
A nonzero vector D has norm 5. What is the norm of −2D?
