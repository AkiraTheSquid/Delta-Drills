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

First master [vector arithmetic](?lesson=math.vector-arithmetic): components, displacement, and scalar multiplication. This lesson applies those skills to rays.

A point names a location; a vector names a change in location. Subtracting two points gives a displacement: from A to B, move by B − A. Adding a displacement to a point gives another point.

A ray uses P(s) = O + sD, where O is its origin, D is a nonzero direction, and s ≥ 0. D is a displacement, even when ARENA stores it in the second row of a (2, 3) tensor. It is not an endpoint. Given an endpoint Q, first compute D = Q − O. At s = 0 you are at O; negative s lies behind the ray.

A segment uses P(v) = (1 − v)A + vB = A + v(B − A), with 0 ≤ v ≤ 1. The weights add to one. The supporting line allows any real v; the segment does not.

For n ≥ 2 camera pixels and y-limit L at x = 1, equally spaced direction-y values are −L + i·2L/(n−1). There are n−1 gaps between n endpoints. ARENA uses Dₓ = 1, so s = x when O = 0.

```python
origin, direction, s = (2, -1), (3, 2), 2
point = tuple(origin[i] + s * direction[i]
              for i in range(2))
print(point)
# Hidden checks
assert point == (8, 3)
assert _delta_output == '(8, 3)\n'
```

## Worked example

Find the midpoint from A = (−2, 3) to B = (6, 7). Half the displacement is (4, 2); adding it to A gives (2, 5). This arithmetic becomes a tensor expression in intersect_ray_1d.

```python
a, b, v = (-2, 3), (6, 7), 0.5
point = tuple(a[i] + v * (b[i] - a[i])
              for i in range(2))
print(point)
# Hidden checks
assert point == (2.0, 5.0)
assert _delta_output == '(2.0, 5.0)\n'
```

Return to [the connected PyTorch lesson](?lesson=raytracing.ray-parametrisation) to implement this reasoning. ARENA 0.1 transfer: `make_rays_1d`, `make_rays_2d`, `intersect_ray_1d`.

## Faded practice

### q50000
O = (3, −2), Q = (7, 4). A ray starts at O and passes through Q. Which D makes O + D = Q?

### q50001
P(s) = (2, 1) + s(−1, 2), s ≥ 0. Does (3, −1) belong to this ray?

## Solo practice

### q50002
A = (1, 2), B = (5, 6). What does v = 1.25 give in A + v(B − A)?

### q50003
A point is O + 6D. Replace D by D′ = 3D, leaving O fixed. Which parameter reaches the same point?

## Integrated practice

### q50004
A five-pixel camera includes both y-limits −3 and 3 at x = 1. Which sequence contains the direction-y coordinates?

### q50005
A point has v = 0.2 from A to B. Swap the endpoints and write P = B + w(A − B). What is w?
