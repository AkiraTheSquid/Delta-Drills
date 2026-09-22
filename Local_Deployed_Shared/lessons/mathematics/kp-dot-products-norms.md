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
The result is a scalar, not a vector. In particular, $\mathbf v\cdot\mathbf v=\sum_i v_i^2$ is the squared Euclidean length. The length (norm) is $\|\mathbf v\|=\sqrt{\mathbf v\cdot\mathbf v}$. Negative components still contribute positive squares.

For nonzero $\mathbf v$, the unit vector $\hat{\mathbf v}=\mathbf v/\|\mathbf v\|$ has length one and the same direction. Zero cannot be normalized: its norm is zero. Scaling obeys $\|c\mathbf v\|=|c|\|\mathbf v\|$.

Two nonzero vectors with dot product zero are perpendicular. For core ray tracing, the essential use here is length: the distance between points is $\|B-A\|$, and displacement $sD$ has length $|s|\|D\|$. A parameter is a distance only when the direction has unit length (and the parameter is nonnegative).

## Worked example

For $\mathbf v=(2,-3,6)$, the squared length is $4+9+36=49$, so $\|\mathbf v\|=7$. The unit vector is $(2/7,-3/7,6/7)$, not $\mathbf v/49$.

For points $A=(1,2)$ and $B=(4,6)$, subtract first: $B-A=(3,4)$. The distance is $\sqrt{3^2+4^2}=5$, not $3+4=7$.

```python
from math import sqrt
v = (2, -3, 6)
length = sqrt(sum(x * x for x in v))
unit = tuple(x / length for x in v)
print(length, round(sum(x * x for x in unit), 12))
# Hidden checks
assert length == 7
assert abs(sum(x * x for x in unit) - 1) < 1e-12
assert _delta_output == '7.0 1.0\n'
```

ARENA 0.1 transfer: continue with [ray distance](?lesson=math.ray-distance). These are standalone mathematics problems; no PyTorch knowledge is required. Work the answer on paper, then select one choice in practice.

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
