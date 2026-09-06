---
kc: raytracing.ray-parametrisation
title: A ray is an origin and a direction
supporting: [numpy.constructors, numpy.broadcasting-rules, numpy.slicing-views, numpy.ranges]
new_syntax: []
previews: []
faded: [820, 821]
guided: []
independent: [822, 823, 824, 825, 826, 827]
integrated: [828, 829, 830]
---

## Concept: origin + u · direction, and the (2, 3) layout

A **ray** is half a line: it starts at a point and goes on forever in one
direction. ARENA writes every point on it as

> **P(u) = O + u · D**, with **u ≥ 0**

where **O** is the origin (a point), **D** the direction (a vector), and `u` a
scalar. `u = 0` is the origin, `u = 1` is one direction-length along, `u = 2`
twice that. Negative `u` would be *behind* the camera, so it is excluded.

In a tensor the ray is stored as a **(2, 3)** slab — row 0 is the origin, row
1 the direction, and the three columns are x, y, z:

```python
import torch as t

ray = t.tensor([[0.0, 0.0, 0.0],    # O
                [1.0, 0.5, 0.0]])   # D
origin, direction = ray[0], ray[1]
u = 2.0
print(origin + u * direction)       # the point at u = 2
```

Because the whole thing is arithmetic, **many `u` at once** is one broadcast:
a column of `u` values times the direction row gives one point per row.

```python
import torch as t

origin = t.tensor([0.0, 0.0, 0.0])
direction = t.tensor([1.0, 0.5, 0.0])
u = t.tensor([0.0, 1.0, 2.0])
points = origin + u[:, None] * direction    # (3,1) * (3,) → (3,3)
print(points)
```

Two facts you will use constantly in Chapter 0.1:

- The direction is **not unit length** and does not need to be. `D = (1, 0.5, 0)`
  is a perfectly good direction; `u` just measures in multiples of it.
- The same ray has many parametrisations — double `D` and halve every `u`
  and you get the same points. When ARENA asks "does the ray hit the
  segment", it solves for `u` and checks `u ≥ 0`, so **the sign of `u` is what
  carries the geometry**, not its size.

A family of rays all leaving the origin then differs only in the direction
row — which is exactly why `make_rays_1d` builds a stack of `(2, 3)` slabs and
only writes into row 1.

## Worked example

Where is the ray `O = (0, 0, 0)`, `D = (1, 2, 0)` when its x-coordinate is 3?
Solve `u · 1 = 3`, then read y = `u · 2`:

```python
import torch as t

ray = t.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
O, D = ray[0], ray[1]
u = (3.0 - O[0]) / D[0]
point = O + u * D
print(u, point)
assert point.tolist() == [3.0, 6.0, 0.0]
```

| ray `[O, D]` | `u` | `O + u·D` |
| --- | --- | --- |
| `[[0,0,0],[1,0.5,0]]` | 0 | `[0, 0, 0]` |
| `[[0,0,0],[1,0.5,0]]` | 2 | `[2, 1, 0]` |
| `[[1,1,0],[0,1,0]]` | 3 | `[1, 4, 0]` |

## Faded practice

### q820
Row 0, row 1, then the one line of arithmetic.

```python starter
def solve(ray, u):
    """The point at parameter u on a ray stored as [origin, direction]."""
    ray = t.tensor(ray)
    origin, direction = ray[_____], ray[_____]
    return (origin + _____ * direction).tolist()
```

```python solution
def solve(ray, u):
    """The point at parameter u on a ray stored as [origin, direction]."""
    ray = t.tensor(ray)
    origin, direction = ray[0], ray[1]
    return (origin + u * direction).tolist()
```

### q821
u needs a trailing axis so it lines up against the 3 coordinates.

```python starter
def solve(ray, us):
    """Many points on one ray, by broadcasting u down a column."""
    ray = t.tensor(ray)
    u = t.tensor(us)
    return (ray[0] + u[_____, _____] * ray[1]).tolist()
```

```python solution
def solve(ray, us):
    """Many points on one ray, by broadcasting u down a column."""
    ray = t.tensor(ray)
    u = t.tensor(us)
    return (ray[0] + u[:, None] * ray[1]).tolist()
```

## Solo practice

### q822
Solve O_x + u·D_x = x for u.

### q823
Solve for u from x, then read y.

### q824
Is p ahead of the camera on this ray?

### q825
Doubling D and halving u names the same point.

### q826
n integer steps along one ray.

### q827
One u, many rays.

## Integrated practice

### q828
Per-ray u from x, then per-ray y, vectorised.

### q829
Many x's → many u's → many points, one broadcast.

### q830
Which rays hit p ahead of the camera?

## Misconceptions

- **"The direction must be a unit vector."** No. `u` is measured in
  direction-lengths; any nonzero `D` is fine, and ARENA's rays are not
  normalised.
- **"Solving for the line is solving for the ray."** A line accepts any
  `u`; a ray needs `u ≥ 0`. Every "does this ray hit" test in 0.1 ends with
  that sign check.
- **"`rays[0]` is the origins."** For a stack of rays `rays[0]` is the whole
  first ray. The origins of every ray are `rays[:, 0]`.
