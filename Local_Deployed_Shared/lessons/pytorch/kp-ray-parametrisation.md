---
kc: raytracing.ray-parametrisation
title: A ray is an origin and a direction
supporting: [torch.constructors, torch.broadcasting-rules, torch.slicing-views, torch.ranges]
new_syntax: [syntax.unpack]
previews: []
faded: [820, 821]
guided: []
independent: [822, 823, 824, 825, 826, 827, 1604, 1605]
integrated: [828, 829, 830, 1606]
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
# the point at u = 2
print(origin + u * direction)
# Hidden checks
assert _delta_output == 'tensor([2., 1., 0.])\n'
```

`origin, direction = ray[0], ray[1]` unpacks a pair into two names.
The left and right sides have the same number of items, in the same order.

Because the whole thing is arithmetic, **many `u` at once** is one broadcast:
a column of `u` values times the direction row gives one point per row.

```python
import torch as t

origin = t.tensor([0.0, 0.0, 0.0])
direction = t.tensor([1.0, 0.5, 0.0])
u = t.tensor([0.0, 1.0, 2.0])
# (3,1) * (3,) → (3,3)
points = origin + u[:, None] * direction
print(points)
# Hidden checks
assert _delta_output == 'tensor([[0.0000, 0.0000, 0.0000],\n        [1.0000, 0.5000, 0.0000],\n        [2.0000, 1.0000, 0.0000]])\n'
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
# Hidden checks
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

One coordinate pins `u`. Rearranging `O_y + u·D_y = y` gives
`u = (y − O_y) / D_y`: subtract the origin's coordinate first, then divide
by the direction's. Putting `u` back into the formula is the check — the
y-coordinate of the point must be the `y` you started from. The drill asks
the same question of x: same two steps, the other coordinate index.

```python worked
import torch as t

ray = t.tensor([[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]])
for y in (1.0, 5.0):
    u = (y - ray[0, 1]) / ray[1, 1]
    print("y =", y, "-> u =", u.item(), "-> point",
          (ray[0] + u * ray[1]).tolist())
# Hidden checks
assert ((1.0 - ray[0, 1]) / ray[1, 1]).item() == 0.0
assert ((5.0 - ray[0, 1]) / ray[1, 1]).item() == 2.0
```

### q823
Solve for u from x, then read y.

### q824
Is p ahead of the camera on this ray?

### q825
Doubling D and halving u names the same point.

### q826
n integer steps along one ray.

Several `u` on one ray: make `u` a column so it lines up against the three
coordinates, and broadcasting adds the origin to every row. This example
picks its two `u` by hand; the drill wants `u = 0, 1, …, n−1`, and building
that list is the part left to you (`t.arange(n)` is the tool):

```python worked
import torch as t

ray = t.tensor([[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]])
us = t.tensor([0.5, 1.5])
print("u as a column:", tuple(us[:, None].shape))
points = ray[0] + us[:, None] * ray[1]
print(tuple(points.shape), points.tolist())
# Hidden checks
assert points.tolist() == [[1.5, 2.0, 2.5], [4.5, 4.0, 3.5]]
```

### q827
One u, many rays.

A stack of rays is `(n, 2, 3)`. The first index picks a ray; the second
picks a row inside it. Leave the first as `:` and you get the same row of
every ray at once — all the origins, or all the directions:

```python worked
import torch as t

rays = t.tensor([[[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]],
                 [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]])
print("rays[0] is one whole ray:",
      tuple(rays[0].shape))
print("rays[:, 0] is every origin:",
      rays[:, 0].tolist())
print("rays[:, 1] is every direction:",
      rays[:, 1].tolist())
# Hidden checks
assert rays[:, 0].tolist() == [[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]]
assert tuple(rays[0].shape) == (2, 3)
```

### q1604
One u, one point: unpack the two rows and apply the formula.

`u = 0` is the origin itself; every other `u` walks along the direction.
Unpack the two rows first and the formula is one line, the same line for
every `u`:

```python worked
import torch as t

ray = t.tensor([[2.0, 1.0, 0.0], [1.0, 3.0, 0.0]])
origin, direction = ray[0], ray[1]
for u in (0.0, 2.0):
    point = origin + u * direction
    print("u =", u, "->", point.tolist())
# Hidden checks
assert (origin + 0.0 * direction).tolist() == [2.0, 1.0, 0.0]
assert (origin + 2.0 * direction).tolist() == [4.0, 7.0, 0.0]
```

### q1605
Read u back off a known point — one coordinate is enough.

## Integrated practice

### q828
Per-ray u from x, then per-ray y, vectorised.

### q829
Many x's → many u's → many points, one broadcast.

### q830
Which rays hit p ahead of the camera?

### q1606
Every ray at the same u, then a test per row.

## Misconceptions

- **"The direction must be a unit vector."** No. `u` is measured in
  direction-lengths; any nonzero `D` is fine, and ARENA's rays are not
  normalised.
- **"Solving for the line is solving for the ray."** A line accepts any
  `u`; a ray needs `u ≥ 0`. Every "does this ray hit" test in 0.1 ends with
  that sign check.
- **"`rays[0]` is the origins."** For a stack of rays `rays[0]` is the whole
  first ray. The origins of every ray are `rays[:, 0]`.
