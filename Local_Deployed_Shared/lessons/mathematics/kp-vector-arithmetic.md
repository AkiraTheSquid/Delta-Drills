---
kc: math.vector-arithmetic
kind: math
title: Vectors: components, displacement, and scaling
supporting: []
new_syntax: []
previews: []
concepts: []
faded: [50032, 50033]
guided: []
independent: [50034, 50035, 50062, 50063, 50064, 50065, 50066, 50067]
integrated: [50036, 50037, 50069, 50070, 50071]
---

## Concept

A vector is an ordered list of components, written as a column:
$$\mathbf v=\begin{pmatrix}v_x\\v_y\\v_z\end{pmatrix}.$$
Add and subtract matching components; multiply every component by a scalar:
$$\begin{pmatrix}1\\2\\3\end{pmatrix}+\begin{pmatrix}4\\-1\\0\end{pmatrix}=\begin{pmatrix}5\\1\\3\end{pmatrix},\qquad -2\begin{pmatrix}1\\2\\3\end{pmatrix}=\begin{pmatrix}-2\\-4\\-6\end{pmatrix}.$$
A negative scale reverses direction; zero gives the zero vector.

A point is a location in a chosen coordinate frame. The displacement from point $A$ to point $B$ is $B-A$, not $A+B$. Adding a displacement to a point gives a point. Check the order with
$$A+(B-A)=B.$$

Translation changes point coordinates but not a displacement:
$$(B+T)-(A+T)=B-A.$$
Direction vectors are not endpoints. Their components describe how a point changes when a parameter increases by one.

## Worked example

Find the displacement from $A=(2,-1,3)$ to $B=(5,3,1)$, then the midpoint of $AB$.

**On paper.** Write both points as columns and subtract component by component:
$$B-A=\begin{pmatrix}5\\3\\1\end{pmatrix}-\begin{pmatrix}2\\-1\\3\end{pmatrix}=\begin{pmatrix}5-2\\3-(-1)\\1-3\end{pmatrix}=\begin{pmatrix}3\\4\\-2\end{pmatrix}.$$
The midpoint is $A$ plus half that displacement:
$$A+\tfrac12(B-A)=\begin{pmatrix}2\\-1\\3\end{pmatrix}+\begin{pmatrix}1.5\\2\\-1\end{pmatrix}=\begin{pmatrix}3.5\\1\\2\end{pmatrix}.$$
Reversing the subtraction gives $A-B=(-3,-4,2)$, the displacement back.

**In code.** The same two steps on tensors. Arithmetic on a tensor acts on every component at once, so no loop is needed. Predict the printed values before running it.

```python
import torch as t

a = t.tensor([2.0, -1.0, 3.0])
b = t.tensor([5.0, 3.0, 1.0])

displacement = b - a
midpoint = a + 0.5 * displacement

print(displacement)
print(midpoint)
# Hidden checks
assert displacement.tolist() == [3.0, 4.0, -2.0]
assert midpoint.tolist() == [3.5, 1.0, 2.0]
assert _delta_output == 'tensor([ 3.,  4., -2.])\ntensor([3.5000, 1.0000, 2.0000])\n'
```

## Faded practice

### q50032
For u = (2, −1, 3) and v = (−4, 5, 1), find u + v.

### q50033
Find −2(1, −3, 2).

## Solo practice

### q50034
A = (−1, 2, 4), B = (3, −1, 6). Find the displacement from A to B.

### q50035
Let p = (1, −2), u = (3, 1), v = (−1, 4). Find p + 2u − v.

### q50062
A = (4, −2, 7), B = (−1, 3, 2). Which point P satisfies P − A = 3(B − A)?

### q50063
Which scalar c makes c(−2, 3, −1) = (6, −9, 3)?

### q50064
Are u = (2, −4, 6) and w = (−3, 6, −9) parallel?

### q50065
u = (1, 2, −1), v = (3, 0, 2). Solve 2x + u = v for the vector x.

### q50066
A = (0, 3, −3), B = (6, 0, 3). Which point is one third of the way from A to B?

### q50067
A ray has origin O = (1, 1, 1) and direction D = (2, −1, 0). Which point does it reach at s = 0.5?

## Integrated practice

### q50036
Both points A and B are translated by T. Which expression still equals the original displacement B − A?

### q50037
A = (2, 0, −1), B = (6, 4, 3). Find A + (B − A)/4.

### q50069
A segment is stored as [[1, 2, 0], [5, −2, 4]]: row 0 is A, row 1 is B. Which point is three quarters of the way from A to B?

### q50070
ABCD is a parallelogram with its vertices in that order. A = (1, 0, 2), B = (4, 1, 2), D = (0, 3, 1). Find C.

### q50071
Two rays share origin O = (2, 1, 0), with directions D₁ = (1, 2, 0) and D₂ = (3, 6, 0). Compare the points O + 3D₁ and O + D₂.
