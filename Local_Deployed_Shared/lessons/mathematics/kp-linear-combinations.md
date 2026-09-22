---
kc: math.linear-combinations
kind: math
title: Linear combinations, span, and coordinates
supporting: [math.vector-arithmetic]
new_syntax: []
previews: []
concepts: []
faded: [50044, 50045]
guided: []
independent: [50046, 50047, 50088, 50089, 50090, 50091, 50092, 50093]
integrated: [50048, 50049, 50094, 50095]
---

## Concept

A linear combination $a\mathbf u+b\mathbf v$ scales vectors, then adds them. The span of $\mathbf u,\mathbf v$ is the set of all such combinations with real weights. Weights may be negative; span is not restricted to the region between the vectors.

Vectors are linearly independent if no nonzero choice of weights gives the zero vector. Equivalently, a target in their span has unique coordinates in those vectors. Dependent vectors provide redundant directions. For example, $(1,2)$ and $(2,4)$ span only a line; two independent vectors in $\mathbb R^2$ span the whole plane and form a basis.

A basis is an independent spanning set. Standard components are coordinates in the standard basis; changing the basis changes the weights, not the target vector. Two independent vectors in $\mathbb R^3$ span a plane through zero, not all of 3D space.

For geometry based at a point $A$, use
$$P=A+uE_1+vE_2.$$
The vector $P-A$ is in the span of the edge directions. Adding $A$ translates that plane. Triangle bounds on the weights come later; span alone imposes no bounds.

## Worked example

Express $P=(5,3)$ in the basis $E_1=(2,0)$, $E_2=(1,3)$.

**On paper.** Look for weights $u,v$ with $uE_1+vE_2=P$, and read off one equation per row:
$$u\begin{pmatrix}2\\0\end{pmatrix}+v\begin{pmatrix}1\\3\end{pmatrix}=\begin{pmatrix}5\\3\end{pmatrix}\quad\Longrightarrow\quad\begin{aligned}2u+v&=5\\3v&=3\end{aligned}$$
The second equation gives $v=1$. Substituting, $2u+1=5$, so $u=2$. Check:
$$2\begin{pmatrix}2\\0\end{pmatrix}+1\begin{pmatrix}1\\3\end{pmatrix}=\begin{pmatrix}4+1\\0+3\end{pmatrix}=\begin{pmatrix}5\\3\end{pmatrix}.$$
The standard coordinates $(5,3)$ and the basis coordinates $(2,1)$ describe the same vector in different bases.

In contrast, no combination of $(1,2)$ and $(2,4)$ gives $(0,1)$, because the second vector is twice the first:
$$a\begin{pmatrix}1\\2\end{pmatrix}+b\begin{pmatrix}2\\4\end{pmatrix}=(a+2b)\begin{pmatrix}1\\2\end{pmatrix}.$$
Every reachable vector has second component twice its first.

**In code.** Rebuild the target from its weights, then recover the weights by solving the system whose columns are the basis vectors.

```python
import torch as t

e1 = t.tensor([2.0, 0.0])
e2 = t.tensor([1.0, 3.0])
u = 2
v = 1

target = u * e1 + v * e2
print(target)

basis = t.stack([e1, e2], dim=1)
weights = t.linalg.solve(basis, target)
print(weights)
# Hidden checks
assert target.tolist() == [5.0, 3.0]
assert t.allclose(weights, t.tensor([2.0, 1.0]))
assert _delta_output == 'tensor([5., 3.])\ntensor([2., 1.])\n'
```

## Faded practice

### q50044
Find 2(1, 2) − (3, −1).

### q50045
Which target is in the span of (1, 2) and (2, 4)?

## Solo practice

### q50046
Let E1 = (2, 0), E2 = (1, 3). Which (u, v) gives uE1 + vE2 = (7, 3)?

### q50047
What is the span of (1, 0, 0) and (0, 1, 0) with unrestricted real weights?

### q50088
E1 = (1, 1), E2 = (1, −1). Which weights (u, v) give uE1 + vE2 = (5, 1)?

### q50089
E1 = (1, 0, 1), E2 = (0, 2, 1). Which weights (u, v) give uE1 + vE2 = (2, 4, 4)?

### q50090
E1 = (1, 0, 1), E2 = (0, 2, 1). Can uE1 + vE2 equal (1, 2, 4)?

### q50091
w = t.tensor([2., -1.]) and E = t.tensor([[1., 0.], [3., 1.]]) stores E1 and E2 as its rows. What is w @ E?

### q50092
E1 = (2, 1), E2 = (4, 2), E3 = (0, 1). Which pair does NOT span the whole plane?

### q50093
A = (0, 4), B = (8, 0). Which point is 0.25A + 0.75B?

## Integrated practice

### q50048
E1 = (1, 2), E2 = (2, 4). Target T = (3, 6) has which property?

### q50049
A = (1, −1, 2), E1 = (2, 0, 0), E2 = (0, 3, 0). Find A + E1/2 + E2/3.

### q50094
A = (1, 1), B = (5, 1), C = (1, 4). Which (u, v) gives A + u(B − A) + v(C − A) = (3, 2.5)?

### q50095
E1 = t.tensor([2., 1.]), E2 = t.tensor([3., 4.]), T = t.tensor([5., 5.]). What does t.linalg.solve(t.stack([E1, E2], dim=1), T) return?
