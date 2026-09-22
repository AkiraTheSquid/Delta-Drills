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

First master [linear combinations and coordinates](?lesson=math.linear-combinations). Here the vectors are triangle edges, and extra bounds restrict their coefficients.

For a nondegenerate triangle with vertices A, B, C, begin at A and combine two edges: P = A + u(B − A) + v(C − A). Expanding gives P = (1 − u − v)A + uB + vC. These three vertex weights are barycentric coordinates. They add to one.

The point belongs to the filled triangle exactly when all three weights are nonnegative: u ≥ 0, v ≥ 0, u + v ≤ 1. Independent bounds u ≤ 1 and v ≤ 1 admit the whole parallelogram. The sum bound cuts away the half beyond edge BC.

Boundaries count: u = 0 is edge AC, v = 0 is edge AB, and u + v = 1 is edge BC, provided the other weights are nonnegative. Vertices are special boundary cases. In three dimensions these weights describe the triangle's plane, not every point in space.

```python
u, v = 0.6, 0.6
weights = (1 - u - v, u, v)
inside = all(weight >= 0 for weight in weights)
print(inside)
# Hidden checks
assert not inside
assert _delta_output == 'False\n'
```

## Worked example

A = (2, 0, 0), B = (2, 6, 0), C = (2, 0, 3). With u = v = 1/3, each vertex has weight 1/3. The point is inside. Compute its coordinates from the edges.

```python
a, b, c = (2, 0, 0), (2, 6, 0), (2, 0, 3)
u, v = 1/3, 1/3
point = tuple(a[i] + u*(b[i]-a[i]) + v*(c[i]-a[i]) for i in range(3))
print(point)
# Hidden checks
assert point == (2.0, 2.0, 1.0)
assert _delta_output == '(2.0, 2.0, 1.0)\n'
```

Return to [the connected PyTorch lesson](?lesson=raytracing.triangle-intersection) to implement this reasoning. ARENA 0.1 transfer: `triangle_ray_intersects`, `raytrace_triangle`.

## Faded practice

### q50018
P = A + 0.2(B − A) + 0.3(C − A). What is the weight on A?

### q50019
For a nondegenerate triangle, u = 0.8, v = 0.4. Where is A + u(B − A) + v(C − A)?

## Concept: Match a ray point to a triangle point

Set O + sD equal to A + u(B − A) + v(C − A). Move the ray displacement to the triangle side, then move the fixed vertex to the other side:
−sD + u(B − A) + v(C − A) = O − A.

With unknowns ordered [s, u, v], the columns are [−D, B − A, C − A]. Three spatial coordinates provide three equations. Changing only the first column's sign reverses s. Multiplying the entire equation by −1 is equivalent, provided both matrix and right-hand side change.

A nonsingular solve finds the crossing with the triangle's plane. Then check s ≥ 0 and all three triangle weights nonnegative. A plane crossing can still lie outside the triangle. Batching repeats this small system for every pair; an extra batch axis introduces no extra geometric unknown. Retain the singularity mask as in the segment case.

```python
d, a, b, c = (1, 0, 0), (4, 0, 0), (4, 2, 0), (4, 0, 6)
matrix = [[-d[i], b[i]-a[i], c[i]-a[i]] for i in range(3)]
print(matrix)
# Hidden checks
assert matrix == [[-1, 0, 0], [0, 2, 0], [0, 0, 6]]
assert _delta_output == '[[-1, 0, 0], [0, 2, 0], [0, 0, 6]]\n'
```

## Worked example

O = (0, 1, 1), D = (2, 0, 0), A = (6, 0, 0), B = (6, 4, 0), C = (6, 0, 4). The equations are −2s = −6, 4u = 1, 4v = 1. Thus s = 3, u = v = 1/4. The ray reaches (6, 1, 1) inside the triangle.

```python
s, u, v = 3, 0.25, 0.25
point = (0 + 2*s, 1, 1)
hit = s >= 0 and u >= 0 and v >= 0 and u + v <= 1
print(point, hit)
# Hidden checks
assert point == (6, 1, 1) and hit
assert _delta_output == '(6, 1, 1) True\n'
```

Return to [the connected PyTorch lesson](?lesson=raytracing.triangle-intersection) to implement this reasoning. ARENA 0.1 transfer: `triangle_ray_intersects`, `raytrace_triangle`.

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
