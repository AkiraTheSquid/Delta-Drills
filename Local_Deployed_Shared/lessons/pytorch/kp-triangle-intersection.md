---
kc: raytracing.triangle-intersection
title: Ray–triangle intersection
new_syntax: []
concepts: [barycentric, three-unknowns]
supporting: ['raytracing.segment-intersection', 'numpy.linalg-basics', 'numpy.broadcasting-rules']
previews: []
faded: [1137, 1138, 1139, 1140]
guided: []
independent: [1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149]
integrated: [1150, 1151, 1152]
---

## Concept: A triangle is spanned by two edge vectors

Starting at vertex `A`, the vectors `B − A` and `C − A` span the triangle's plane, and every point of the plane is `A + u·(B − A) + v·(C − A)` for some pair `(u, v)`. The point is inside the triangle exactly when `u ≥ 0`, `v ≥ 0` and `u + v ≤ 1`. The weight left over, `1 − u − v`, belongs to `A`; the three weights are the point's barycentric coordinates, each in `[0, 1]` for an interior point.

The reason the sum constraint is needed is that `u ≤ 1` and `v ≤ 1` on their own describe the *parallelogram* with corners `A`, `B`, `C` and `B + C − A` — twice the triangle, including the far corner past the edge `BC`. The line `u + v = 1` is that edge; requiring `u + v ≤ 1` keeps the half on `A`'s side. Endpoints are included, so a point exactly on an edge or at a vertex counts as inside. Stacking the two edge vectors as columns, `t.stack((b − a, c − a), dim=1)`, gives the `(3, 2)` matrix that maps `(u, v)` to a displacement from `A` — the same column-per-unknown rule as for segments.

```python
import torch as t
a=t.tensor([2.,0.,0.]); b=t.tensor([2.,4.,0.]); c=t.tensor([2.,0.,4.])
p=a+.25*(b-a)+.5*(c-a)
print(p)
# Hidden checks
assert p.tolist()==[2.,1.,2.]
```

## Worked example

We judge three `(u, v)` pairs. The first is well inside; the second has both coordinates below `1` but their sum exceeds `1`, so it lies in the parallelogram's far corner; the third sits exactly on vertex `C`.

```python
import torch as t
uv=t.tensor([[.2,.3],[.8,.7],[0.,1.]])
inside=(uv>=0).all(dim=1)&(uv.sum(dim=1)<=1)
print(inside)
# Hidden checks
assert inside.tolist()==[True,False,True]
```

Dropping the sum test admits the second pair. That is the parallelogram, not the triangle — a bug that renders every triangle as a quadrilateral.

```python
parallelogram=(uv>=0).all(dim=1)&(uv<=1).all(dim=1)
print(parallelogram)
# Hidden checks
assert parallelogram.tolist()==[True,True,True]
```

## Faded practice

### q1137
Return edge vectors B−A and C−A as columns, shape (3,2). tr: triangle (3,3), vertices A,B,C.

```python starter
import torch as t

def solve(tr):
    pass
```

```python solution
import torch as t

def solve(tr):
    return t.stack((tr[1]-tr[0],tr[2]-tr[0]),dim=1)
```

### q1138
Return the ray/triangle coefficient matrix, shape (3,3). r: ray (2,3) as [origin, direction]; tr: triangle (3,3), vertices A,B,C. Parallel or degenerate pairs are misses.

```python starter
import torch as t

def solve(r,tr):
    pass
```

```python solution
import torch as t

def solve(r,tr):
    o,d=r
    a,b,c=tr
    m=t.stack((-d,b-a,c-a),dim=1)
    return m
```

## Concept: Match a ray point to a triangle point

A ray point is `O + s·D`; a triangle point is `A + u·(B − A) + v·(C − A)`. Setting them equal and moving unknowns left gives `−s·D + u·(B − A) + v·(C − A) = O − A`: three equations in three unknowns, with matrix columns `−D`, `B − A`, `C − A` and right-hand side `O − A`. The solution `(s, u, v)` reads as: travel `s` along the ray, then triangle coordinates `(u, v)`. A hit needs all of `s ≥ 0` (in front of the origin), `u ≥ 0`, `v ≥ 0` and `u + v ≤ 1`.

The reason the first column is `−D` rather than `D` is the side of the equation it was moved from; use `D` and `s` comes out negated, so every forward hit looks like it is behind the camera. The reason for a validity mask, as with segments, is that a ray parallel to the triangle's plane, or a degenerate triangle, makes the matrix singular: test `det(m).abs() >= 1e-8`, substitute the `3 × 3` identity where it fails so a batched solve cannot raise, and mask those pairs out afterwards. Under this contract they are misses.

```python
import torch as t
m=t.tensor([[-1.,0.,0.],[0.,4.,0.],[0.,0.,4.]])
suv=t.linalg.solve(m,t.tensor([-2.,1.,2.]))
print(suv)
# Hidden checks
assert suv.tolist()==[2.,.25,.5]
```

## Worked example

We assemble the system for a ray from the origin along `+x` and a triangle standing in the plane `x = 3`. The first column is `−D`; the other two are the edges from `A`.

```python
import torch as t
o=t.tensor([0.,0.,0.]); d=t.tensor([1.,0.,0.])
a=t.tensor([3.,-1.,-1.]); b=t.tensor([3.,1.,-1.]); c=t.tensor([3.,-1.,1.])
m=t.stack((-d,b-a,c-a),dim=1)
print(m)
# Hidden checks
assert m.tolist()==[[-1.,0.,0.],[0.,2.,0.],[0.,0.,2.]]
```

Solving against `O − A` gives `s = 3` — three direction-lengths to reach the plane — and `(u, v) = (0.5, 0.5)`, the midpoint of edge `BC`, which is on the boundary and therefore a hit.

```python
suv=t.linalg.solve(m,o-a)
s,u,v=suv
print(suv, bool((s>=0)&(u>=0)&(v>=0)&(u+v<=1)))
# Hidden checks
assert suv.tolist()==[3.,.5,.5] and bool((s>=0)&(u>=0)&(v>=0)&(u+v<=1))
```

## Faded practice

### q1139
Return whether the ray intersects the triangle, as a scalar Boolean tensor. r: ray (2,3) as [origin, direction]; tr: triangle (3,3), vertices A,B,C. Parallel or degenerate pairs are misses.

```python starter
import torch as t

def solve(r,tr):
    pass
```

```python solution
import torch as t

def solve(r,tr):
    o,d=r
    a,b,c=tr
    m=t.stack((-d,b-a,c-a),dim=1)
    valid=t.linalg.det(m).abs()>=1e-8
    safe=t.where(valid,m,t.eye(3))
    suv=t.linalg.solve(safe,o-a)
    s,u,v=suv
    hit=valid&(s>=0)&(u>=0)&(v>=0)&(u+v<=1)
    return hit
```

### q1140
Return [s,u,v] for nonsingular plane intersections, else [0,0,0], shape (3,). r: ray (2,3) as [origin, direction]; tr: triangle (3,3), vertices A,B,C. Parallel or degenerate pairs are misses.

```python starter
import torch as t

def solve(r,tr):
    pass
```

```python solution
import torch as t

def solve(r,tr):
    o,d=r
    a,b,c=tr
    m=t.stack((-d,b-a,c-a),dim=1)
    valid=t.linalg.det(m).abs()>=1e-8
    safe=t.where(valid,m,t.eye(3))
    suv=t.linalg.solve(safe,o-a)
    return t.where(valid,suv,t.zeros(3))
```

## Solo practice

### q1141
Return whether the ray hits the triangle within one direction length (travel parameter s at most one), as a scalar Boolean tensor. r: ray (2,3) as [origin, direction]; tr: triangle (3,3), vertices A,B,C. Parallel or degenerate pairs are misses.

### q1142
Return whether the supporting line of the ray meets the triangle, with no forward-ray condition, as a scalar Boolean tensor. r: ray (2,3) as [origin, direction]; tr: triangle (3,3), vertices A,B,C. Parallel or degenerate pairs are misses.

### q1143
Return hit position in world coordinates, or zero vector on a miss, shape (3,). r: ray (2,3) as [origin, direction]; tr: triangle (3,3), vertices A,B,C. Parallel or degenerate pairs are misses.

### q1144
Return vertex weights [A,B,C] for a hit, or zero vector on a miss, shape (3,). r: ray (2,3) as [origin, direction]; tr: triangle (3,3), vertices A,B,C. Parallel or degenerate pairs are misses.

### q1145
Return forward travel parameter to a hit, else -1, as a scalar tensor. r: ray (2,3) as [origin, direction]; tr: triangle (3,3), vertices A,B,C. Parallel or degenerate pairs are misses.

### q1146
Return whether the ray hits strictly inside the triangle, excluding edges, as a scalar Boolean tensor. r: ray (2,3) as [origin, direction]; tr: triangle (3,3), vertices A,B,C. Parallel or degenerate pairs are misses.

### q1147
Return the distance between the hit point and first vertex, or -1 on a miss, as a scalar tensor. r: ray (2,3) as [origin, direction]; tr: triangle (3,3), vertices A,B,C. Parallel or degenerate pairs are misses.

### q1148
Return whether the ray hits the edge opposite A, including its endpoints, as a scalar Boolean tensor; use absolute tolerance 1e-6 for the edge. r: ray (2,3) as [origin, direction]; tr: triangle (3,3), vertices A,B,C. Parallel or degenerate pairs are misses.

### q1149
Return the centroid of the triangle relative to the ray origin, shape (3,). r: ray (2,3) as [origin, direction]; tr: triangle (3,3), vertices A,B,C. Parallel or degenerate pairs are misses.

## Integrated practice

### q1150
Return Euclidean travel distance to a hit, else -1, as a scalar tensor. r: ray (2,3) as [origin, direction]; tr: triangle (3,3), vertices A,B,C. Parallel or degenerate pairs are misses.

### q1151
Return interpolated vertex value at a hit when A,B,C carry values 2,5,11; return -1 on a miss, as a scalar tensor. r: ray (2,3) as [origin, direction]; tr: triangle (3,3), vertices A,B,C. Parallel or degenerate pairs are misses.

### q1152
Return the displacement from the triangle centroid to a hit, else zero, shape (3,). r: ray (2,3) as [origin, direction]; tr: triangle (3,3), vertices A,B,C. Parallel or degenerate pairs are misses.

## Misconceptions

- **`u ≤ 1` and `v ≤ 1` bound the triangle.** They bound the parallelogram; `u + v ≤ 1` is the third edge.
- **The first column is `D`.** It is `−D`; otherwise `s` flips sign and forward hits fail `s ≥ 0`.
- **A solvable system is a hit.** The plane was hit; the point must also satisfy `s ≥ 0` and the barycentric bounds.
- **Points on an edge are outside.** Bounds are inclusive; an edge point has one weight zero and still counts.
