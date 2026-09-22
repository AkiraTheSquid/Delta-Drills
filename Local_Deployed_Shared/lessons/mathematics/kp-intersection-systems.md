---
kc: math.intersection-systems
kind: math
title: Turn an intersection into a linear system
supporting: [math.ray-geometry]
new_syntax: []
previews: []
concepts: [columns]
faded: [50006, 50007]
guided: []
independent: [50008, 50009]
integrated: [50010, 50011]
---

## Concept: One coefficient per unknown

A matrix–vector product is a weighted sum of columns. If M has columns c₁ and c₂, then M[u, v] = u·c₁ + v·c₂. Each row is one coordinate equation; each column belongs to one unknown. Solving Mx = b finds weights x that produce b.

For a ray O + uD and segment A + v(B − A), an intersection has both descriptions. Subtract O, then move the segment displacement left:
uD + v(A − B) = A − O.
Therefore M = [D, A − B], x = [u, v], and b = A − O. The segment column changes sign because we moved that term across the equality.

First solve the supporting lines. Then require u ≥ 0 and 0 ≤ v ≤ 1. Solving alone does not enforce these inequalities. Substitute into both point formulas to check that their coordinates match. A small residual Mx − b checks the equations, but does not check membership.

```python
column_1, column_2, u, v = (2, 1), (1, -3), 3, 2
product = tuple(u * column_1[i] + v * column_2[i] for i in range(2))
print(product)
# Hidden checks
assert product == (8, -3)
assert _delta_output == '(8, -3)\n'
```

## Worked example

Let O = (1, 0), D = (2, 1), A = (5, −1), B = (5, 5). The x-equation is 2u = 4, so u = 2. The y-equation is u − 6v = −1, so v = 1/2. Both bounds hold. Both descriptions must reach (5, 2).

```python
o, d, a, b = (1, 0), (2, 1), (5, -1), (5, 5)
u, v = 2, 0.5
ray_point = tuple(o[i] + u * d[i] for i in range(2))
segment_point = tuple(a[i] + v * (b[i] - a[i]) for i in range(2))
print(ray_point, segment_point)
# Hidden checks
assert ray_point == segment_point == (5, 2)
assert _delta_output == '(5, 2) (5.0, 2.0)\n'
```

Return to [the connected PyTorch lesson](?lesson=raytracing.segment-intersection) to implement this reasoning. ARENA 0.1 transfer: `intersect_ray_1d`.

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
