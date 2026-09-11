---
kc: raytracing.segment-intersection
title: Ray–segment intersection
new_syntax: ['torch.linalg.det']
concepts: [two-descriptions, membership]
supporting: ['raytracing.ray-parametrisation', 'numpy.linalg-basics', 'numpy.stack-concat-interleave', 'numpy.constructors', 'numpy.boolean-masking', 'tensor.row-normalization', 'tensor.indexed-selection']
previews: []
faded: [1089, 1090, 1091, 1092]
guided: []
independent: [1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101]
integrated: [1102, 1103, 1104]
---

## Concept: Two descriptions of one point

A ray is every point `O + u·D` for `u ≥ 0`: an origin plus some multiple of a direction. A segment from `A` to `B` is every point `A + v·(B − A)` for `0 ≤ v ≤ 1`. If the ray and the segment meet, one point has both descriptions, so `O + u·D = A + v·(B − A)`. Move the unknowns to one side: `u·D + v·(A − B) = A − O`, a linear system in `u` and `v` — in the plane, two equations in two unknowns. Build the matrix with `D` as its first column and `A − B` as its second — `t.stack((d, a − b), dim=1)` — and solve it against the right-hand side `A − O`.

The reason each column is one unknown's direction is what a matrix–vector product means: the product `M @ [u, v]` is `u` times the first column plus `v` times the second, so the columns must be the vectors that `u` and `v` scale. Getting `A − B` rather than `B − A` in the second column is the sign that lets `v` be measured *from* `A`, so the membership test in the next segment reads as `0 ≤ v ≤ 1`. Solving the lines is only half the decision: the numbers `u` and `v` say where the infinite line through the ray and the infinite line through the segment cross, not yet whether the finite objects do.

```python
import torch as t
d=t.tensor([1.,0.]); a=t.tensor([3.,-1.]); b=t.tensor([3.,1.])
m=t.stack((d,a-b),dim=1)
uv=t.linalg.solve(m,a)
print(uv)
# Hidden checks
assert uv.tolist()==[3.,.5]
```

## Worked example

We set up the system for a ray from `(1, 1)` heading along `(1, 1)` and a vertical segment on `x = 4`. The first column is the ray direction; the second is `A − B`, pointing down the segment. Predict which entries are negative before running.

```python
import torch as t
o=t.tensor([1.,1.]); d=t.tensor([1.,1.])
a=t.tensor([4.,0.]); b=t.tensor([4.,6.])
m=t.stack((d,a-b),dim=1)
print(m)
# Hidden checks
assert m.tolist()==[[1.,0.],[1.,-6.]]
```

The right-hand side is `A − O`, not `A`: it is the displacement the two unknowns must jointly cover. Solving gives `u = 3` (three steps along the direction) and `v = 2/3` (two thirds of the way from `A` to `B`).

```python
uv=t.linalg.solve(m,a-o)
print(uv)
# Hidden checks
assert t.allclose(uv,t.tensor([3.,2/3]))
```

## Faded practice

### q1089
Return the two-column coefficient matrix, shape (2,2). r: ray (2,3) as [origin, direction]; s: segment (2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

```python starter
import torch as t

def solve(r,s):
    pass
```

```python solution
import torch as t

def solve(r,s):
    o,d=r[0,:2],r[1,:2]
    a,b=s[0,:2],s[1,:2]
    m=t.stack((d,a-b),dim=1)
    return m
```

### q1090
Return the displacement from ray origin to first segment endpoint, shape (2,). r: ray (2,3) as [origin, direction]; s: segment (2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

```python starter
import torch as t

def solve(r,s):
    pass
```

```python solution
import torch as t

def solve(r,s):
    o,d=r[0,:2],r[1,:2]
    a,b=s[0,:2],s[1,:2]
    v=a-o
    return v
```

## Concept: A solution must belong to both objects

The lines cross at `(u, v)`; the *ray* contains that point only if `u ≥ 0`, and the *segment* only if `0 ≤ v ≤ 1`, endpoints included. So a hit is `(u >= 0) & (v >= 0) & (v <= 1)`, three Boolean tests combined with `&`. Before any of that, the system must actually have one solution: if `D` and `A − B` are parallel, or either is zero, the matrix is singular and there is no unique crossing. `t.linalg.det(m)` is zero exactly then, so the validity test is `det.abs() >= 1e-8` — a tolerance, because floating-point parallel lines give a determinant near zero rather than exactly zero.

The reason to test validity before solving is that `solve` raises on a singular matrix, and in a batch one bad pair would abort every good one. The remedy is to substitute a harmless matrix — the identity — for the invalid pairs, solve everything, and then exclude the substituted answers with the original mask: `valid & hit`. The substituted solve produces numbers, but they are never evidence of a hit. Under this exercise's contract a parallel or degenerate pair is a miss, even where the lines coincide.

```python
import torch as t
m=t.tensor([[1.,-2.],[0.,0.]])
print(t.linalg.det(m))
# Hidden checks
assert float(t.linalg.det(m))==0
```

## Worked example

We judge three candidate `(u, v)` pairs at once, one per row. The first is on the ray and inside the segment; the second is behind the ray's origin; the third is past the segment's end.

```python
import torch as t
uv=t.tensor([[2.,.4],[-1.,.3],[3.,1.2]])
hit=(uv[:,0]>=0)&(uv[:,1]>=0)&(uv[:,1]<=1)
print(hit)
# Hidden checks
assert hit.tolist()==[True,False,False]
```

Now suppose the first pair came from a singular system that was replaced by the identity before solving. Its numbers look like a hit; the validity mask is what removes it.

```python
valid=t.tensor([False,True,True])
print(valid&hit)
# Hidden checks
assert (valid&hit).tolist()==[False,False,False]
```

## Faded practice

### q1091
Return whether the two supporting lines have a unique intersection, as a scalar Boolean tensor. r: ray (2,3) as [origin, direction]; s: segment (2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

```python starter
import torch as t

def solve(r,s):
    pass
```

```python solution
import torch as t

def solve(r,s):
    o,d=r[0,:2],r[1,:2]
    a,b=s[0,:2],s[1,:2]
    m=t.stack((d,a-b),dim=1)
    return t.linalg.det(m).abs()>=1e-8
```

### q1092
Return whether ray and segment intersect, as a scalar Boolean tensor. r: ray (2,3) as [origin, direction]; s: segment (2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

```python starter
import torch as t

def solve(r,s):
    pass
```

```python solution
import torch as t

def solve(r,s):
    o,d=r[0,:2],r[1,:2]
    a,b=s[0,:2],s[1,:2]
    m=t.stack((d,a-b),dim=1)
    v=a-o
    valid=t.linalg.det(m).abs()>=1e-8
    m=t.where(valid,m,t.eye(2))
    uv=t.linalg.solve(m,v)
    u,w=uv[0],uv[1]
    return valid & (u>=0) & (w>=0) & (w<=1)
```

## Solo practice

### q1093
Return the signed determinant of the intersection system, as a scalar tensor. r: ray (2,3) as [origin, direction]; s: segment (2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

### q1094
Return [ray parameter, segment parameter] for unique line intersections; otherwise [0,0], shape (2,). r: ray (2,3) as [origin, direction]; s: segment (2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

### q1095
Return whether the supporting lines intersect strictly behind the ray origin, as a scalar Boolean tensor. r: ray (2,3) as [origin, direction]; s: segment (2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

### q1096
Return whether the supporting lines meet within the segment's extent, whatever the ray's direction, as a scalar Boolean tensor. r: ray (2,3) as [origin, direction]; s: segment (2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

### q1097
Return the ray parameter u at the hit, or -1 on a miss, as a scalar tensor. r: ray (2,3) as [origin, direction]; s: segment (2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

### q1098
Return the xy hit point, or [0,0] on a miss, shape (2,). r: ray (2,3) as [origin, direction]; s: segment (2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

### q1099
Return ray parameter to the supporting-line intersection only when it lies behind the origin; return zero otherwise, as a scalar tensor. r: ray (2,3) as [origin, direction]; s: segment (2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

### q1100
Return distance from the first segment endpoint to the hit point, else -1, as a scalar tensor. r: ray (2,3) as [origin, direction]; s: segment (2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

### q1101
Return the segment midpoint relative to the ray origin, shape (2,). r: ray (2,3) as [origin, direction]; s: segment (2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

## Integrated practice

### q1102
Return Euclidean travel distance to a hit, or -1 on a miss, as a scalar float tensor. r: ray (2,3) as [origin, direction]; s: segment (2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

### q1103
Return the interpolation weights of the two segment endpoints at the hit, in endpoint order, or [0,0] on a miss, shape (2,). r: ray (2,3) as [origin, direction]; s: segment (2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

### q1104
Return whether the ray meets the segment strictly between its endpoints; exclude endpoint hits, as a scalar Boolean tensor. r: ray (2,3) as [origin, direction]; s: segment (2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

## Misconceptions

- **The columns are `D` and `B − A`.** With `B − A` the sign of `v` flips and the `0 ≤ v ≤ 1` test fails; the second column is `A − B`.
- **A solution means a hit.** It means the infinite lines cross; the ray needs `u ≥ 0` and the segment needs `v` in `[0, 1]`.
- **A singular pair can be skipped by catching the error.** In a batch the error aborts every pair; substitute the identity and mask afterwards.
- **`det == 0` is the right test.** Floating-point parallel lines give a tiny nonzero determinant; compare against a tolerance.
