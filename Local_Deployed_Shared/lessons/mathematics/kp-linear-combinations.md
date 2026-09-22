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
independent: [50046, 50047]
integrated: [50048, 50049]
---

## Concept

A linear combination $a\mathbf u+b\mathbf v$ scales vectors, then adds them. The span of $\mathbf u,\mathbf v$ is the set of all such combinations with real weights. Weights may be negative; span is not restricted to the region between the vectors.

Vectors are linearly independent if no nonzero choice of weights gives the zero vector. Equivalently, a target in their span has unique coordinates in those vectors. Dependent vectors provide redundant directions. For example, $(1,2)$ and $(2,4)$ span only a line; two independent vectors in $\mathbb R^2$ span the whole plane and form a basis.

A basis is an independent spanning set. Standard components are coordinates in the standard basis; changing the basis changes the weights, not the target vector. Two independent vectors in $\mathbb R^3$ span a plane through zero, not all of 3D space.

For geometry based at a point $A$, use $P=A+uE_1+vE_2$. The vector $P-A$ is in the span of the edge directions. Adding $A$ translates that plane. Triangle bounds on the weights come later; span alone imposes no bounds.

## Worked example

Let $E_1=(2,0)$ and $E_2=(1,3)$. To express $P=(5,3)$, solve
$2u+v=5$ and $3v=3$. Thus $v=1$, $u=2$, and $P=2E_1+E_2$.

The standard coordinates $(5,3)$ and basis coordinates $(2,1)$ describe the same vector in different bases. In contrast, no combination of $(1,2)$ and $(2,4)$ gives $(0,1)$: every reachable vector has second component twice its first.

```python
e1, e2 = (2, 0), (1, 3)
u, v = 2, 1
target = tuple(u * x + v * y for x, y in zip(e1, e2))
print(target)
# Hidden checks
assert target == (5, 3)
assert _delta_output == '(5, 3)\n'
```

ARENA 0.1 transfer: continue with [barycentric coordinates](?lesson=math.barycentric-coordinates). These are standalone mathematics problems; no PyTorch knowledge is required. Work the answer on paper, then select one choice in practice.

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

## Integrated practice

### q50048
E1 = (1, 2), E2 = (2, 4). Target T = (3, 6) has which property?

### q50049
A = (1, −1, 2), E1 = (2, 0, 0), E2 = (0, 3, 0). Find A + E1/2 + E2/3.
