---
kc: raytracing.mesh-visibility
title: Nearest hit in a mesh
new_syntax: []
concepts: [nearest-hit, depth-vs-distance]
supporting: ['raytracing.triangle-intersection', 'raytracing.batched-segments', 'raytracing.make-rays-2d', 'tensor.indexed-selection', 'numpy.dtype-astype']
previews: []
faded: [1153, 1154, 1155, 1156]
guided: []
independent: [1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165]
integrated: [1166, 1167, 1168]
---

## Concept: Visibility chooses the nearest of many hits

A ray through a mesh can intersect several triangles, and the one you *see* is the nearest: the valid, forward hit with the smallest travel `s`. So the batched pipeline keeps one `s` per ray–triangle pair, shape `(nr, nt)`, alongside the pair's hit verdict, and then reduces over the triangle axis with a minimum. To stop misses from winning, replace their `s` with infinity first — `t.where(hit, s, inf)` — so the minimum over a row is either the nearest real depth or `inf`, which means the ray saw only background.

The reason infinity is the right filler is that it is neutral for `min`, as `-inf` was for `max` in pooling: no finite depth loses to it, and a row with no hits reports `inf` rather than a spurious `0` or a stale value from a singular solve. The reason to be careful with `argmin` is that it always returns an index, even when every entry is `inf`; asking *which* triangle is visible therefore needs an explicit background check on the minimum value before trusting the index.

```python
import torch as t
s=t.tensor([[5.,2.,1.],[3.,4.,2.]])
hit=t.tensor([[True,True,False],[False,False,False]])
depth=t.where(hit,s,t.tensor(float("inf")))
print(depth.min(dim=1)[0])
# Hidden checks
assert depth.min(dim=1)[0].tolist()==[2.,float("inf")]
```

## Worked example

We find the visible triangle per ray from a depth table in which the second ray hits nothing. `min(dim=1)` returns both the value and the index; the index is meaningless where the value is infinite.

```python
import torch as t
d=t.tensor([[7.,3.],[float("inf"),float("inf")]])
value,index=d.min(dim=1)
print(value, index)
# Hidden checks
assert value.tolist()==[3.,float("inf")] and index[0].item()==1
```

`where` turns the background row's index into a sentinel `-1`, so a consumer can tell "triangle 0" from "nothing". Without it the second ray would claim to see triangle `0`.

```python
print(t.where(value<float("inf"),index,-1))
# Hidden checks
assert t.where(value<float("inf"),index,-1).tolist()==[1,-1]
```

## Faded practice

### q1153
Return hit verdicts for all pairs, shape (nr,nt). r: rays (nr,2,3) as [origin, direction]; tr: triangles (nt,3,3), vertices A,B,C. Singular pairs and hits behind the origin are misses.

```python starter
import torch as t

def solve(r,tr):
    pass
```

```python solution
import torch as t

def solve(r,tr):
    o=r[:,None,0,:]
    d=r[:,None,1,:]
    a=tr[None,:,0,:]
    b=tr[None,:,1,:]
    c=tr[None,:,2,:]
    m=t.stack((-d+t.zeros_like(a),b-a+t.zeros_like(o),c-a+t.zeros_like(o)),dim=-1)
    valid=t.linalg.det(m).abs()>=1e-8
    safe=t.where(valid[...,None,None],m,t.eye(3))
    suv=t.linalg.solve(safe,(o-a)[...,None])[...,0]
    s,u,v=suv[...,0],suv[...,1],suv[...,2]
    hit=valid&(s>=0)&(u>=0)&(v>=0)&(u+v<=1)
    return hit
```

### q1154
Return travel parameters for valid hits and infinity for misses, shape (nr,nt). r: rays (nr,2,3) as [origin, direction]; tr: triangles (nt,3,3), vertices A,B,C. Singular pairs and hits behind the origin are misses.

```python starter
import torch as t

def solve(r,tr):
    pass
```

```python solution
import torch as t

def solve(r,tr):
    o=r[:,None,0,:]
    d=r[:,None,1,:]
    a=tr[None,:,0,:]
    b=tr[None,:,1,:]
    c=tr[None,:,2,:]
    m=t.stack((-d+t.zeros_like(a),b-a+t.zeros_like(o),c-a+t.zeros_like(o)),dim=-1)
    valid=t.linalg.det(m).abs()>=1e-8
    safe=t.where(valid[...,None,None],m,t.eye(3))
    suv=t.linalg.solve(safe,(o-a)[...,None])[...,0]
    s,u,v=suv[...,0],suv[...,1],suv[...,2]
    hit=valid&(s>=0)&(u>=0)&(v>=0)&(u+v<=1)
    depth=t.where(hit,s,t.tensor(float("inf")))
    return depth
```

## Concept: Depth and physical distance are different

The travel parameter `s` in `O + s·D` counts *direction-lengths*, not metres: the Euclidean distance from the origin to the hit point is `s` times the length of `D`. The two coincide only when `D` is a unit vector. Under the camera convention where every direction has `Dx = 1`, `s` equals the travel along `x` — a useful depth for an image, but not a distance. Comparing `s` across rays with different `|D|` compares different units.

The reason to keep the distinction is what each quantity is for. Within one ray, `s` orders hits correctly whatever `|D|` is, so the nearest-hit reduction is safe. Across rays, or for lighting, shadows and fog, physical distance is needed, and that is `s * D.norm()`. Once the visible triangle is chosen, the hit point itself is `O + s·D`, and any attribute stored at the vertices can be interpolated with the barycentric weights from the same solve. Keep the ray and triangle axes distinct until the visibility decision is made; only then is there one `s` per ray to turn into a point.

```python
import torch as t
d=t.tensor([1.,2.,2.]); s=t.tensor(4.)
print(s*d.norm())
# Hidden checks
assert float(s*d.norm())==12.
```

## Worked example

We recover the hit point and the physical distance for a ray whose direction is not a unit vector. `|D| = 3`, so the distance is three times the travel parameter.

```python
import torch as t
o=t.tensor([1.,0.,0.]); d=t.tensor([1.,2.,2.]); s=4.
point=o+s*d
print(point)
# Hidden checks
assert point.tolist()==[5.,8.,8.]
```

The distance from the origin to that point, computed directly, agrees with `s * |D|`: the same `12`, arrived at two ways.

```python
print((point-o).norm(), s*d.norm())
# Hidden checks
assert float((point-o).norm())==12. and float(s*d.norm())==12.
```

## Faded practice

### q1155
Return nearest valid travel parameter per ray, or infinity, shape (nr,). r: rays (nr,2,3) as [origin, direction]; tr: triangles (nt,3,3), vertices A,B,C. Singular pairs and hits behind the origin are misses.

```python starter
import torch as t

def solve(r,tr):
    pass
```

```python solution
import torch as t

def solve(r,tr):
    o=r[:,None,0,:]
    d=r[:,None,1,:]
    a=tr[None,:,0,:]
    b=tr[None,:,1,:]
    c=tr[None,:,2,:]
    m=t.stack((-d+t.zeros_like(a),b-a+t.zeros_like(o),c-a+t.zeros_like(o)),dim=-1)
    valid=t.linalg.det(m).abs()>=1e-8
    safe=t.where(valid[...,None,None],m,t.eye(3))
    suv=t.linalg.solve(safe,(o-a)[...,None])[...,0]
    s,u,v=suv[...,0],suv[...,1],suv[...,2]
    hit=valid&(s>=0)&(u>=0)&(v>=0)&(u+v<=1)
    depth=t.where(hit,s,t.tensor(float("inf")))
    return depth.min(dim=1)[0]
```

### q1156
Return whether each ray has any visible surface, shape (nr,). r: rays (nr,2,3) as [origin, direction]; tr: triangles (nt,3,3), vertices A,B,C. Singular pairs and hits behind the origin are misses.

```python starter
import torch as t

def solve(r,tr):
    pass
```

```python solution
import torch as t

def solve(r,tr):
    o=r[:,None,0,:]
    d=r[:,None,1,:]
    a=tr[None,:,0,:]
    b=tr[None,:,1,:]
    c=tr[None,:,2,:]
    m=t.stack((-d+t.zeros_like(a),b-a+t.zeros_like(o),c-a+t.zeros_like(o)),dim=-1)
    valid=t.linalg.det(m).abs()>=1e-8
    safe=t.where(valid[...,None,None],m,t.eye(3))
    suv=t.linalg.solve(safe,(o-a)[...,None])[...,0]
    s,u,v=suv[...,0],suv[...,1],suv[...,2]
    hit=valid&(s>=0)&(u>=0)&(v>=0)&(u+v<=1)
    return hit.any(dim=1)
```

## Solo practice

### q1157
Return number of triangles hit by each ray, shape (nr,). r: rays (nr,2,3) as [origin, direction]; tr: triangles (nt,3,3), vertices A,B,C. Singular pairs and hits behind the origin are misses.

### q1158
Return the mean of the nearest travel parameters over the rays that see a surface, as a scalar tensor; return zero when no ray does. r: rays (nr,2,3) as [origin, direction]; tr: triangles (nt,3,3), vertices A,B,C. Singular pairs and hits behind the origin are misses.

### q1159
Return index of nearest triangle, or -1 on background, shape (nr,); ties choose first. r: rays (nr,2,3) as [origin, direction]; tr: triangles (nt,3,3), vertices A,B,C. Singular pairs and hits behind the origin are misses.

### q1160
Return the index of the triangle crossed by the most rays, counting every valid intersection, as a scalar integer tensor; ties choose first. r: rays (nr,2,3) as [origin, direction]; tr: triangles (nt,3,3), vertices A,B,C. Singular pairs and hits behind the origin are misses.

### q1161
Return indices of background rays, as a 1-D integer tensor. r: rays (nr,2,3) as [origin, direction]; tr: triangles (nt,3,3), vertices A,B,C. Singular pairs and hits behind the origin are misses.

### q1162
Return number of valid pair intersections strictly inside triangle edges, as a scalar integer tensor. r: rays (nr,2,3) as [origin, direction]; tr: triangles (nt,3,3), vertices A,B,C. Singular pairs and hits behind the origin are misses.

### q1163
Return fraction of rays seeing at least one triangle, as a scalar float tensor. r: rays (nr,2,3) as [origin, direction]; tr: triangles (nt,3,3), vertices A,B,C. Singular pairs and hits behind the origin are misses.

### q1164
Return whether each ray crosses more than one triangle, shape (nr,). r: rays (nr,2,3) as [origin, direction]; tr: triangles (nt,3,3), vertices A,B,C. Singular pairs and hits behind the origin are misses.

### q1165
Return the farthest valid travel parameter per ray, or -1 on a miss, shape (nr,). r: rays (nr,2,3) as [origin, direction]; tr: triangles (nt,3,3), vertices A,B,C. Singular pairs and hits behind the origin are misses.

## Integrated practice

### q1166
Return nearest hit positions in world coordinates, or zero vectors for background, shape (nr,3). r: rays (nr,2,3) as [origin, direction]; tr: triangles (nt,3,3), vertices A,B,C. Singular pairs and hits behind the origin are misses.

### q1167
Return nearest-hit Euclidean distance for each ray, or infinity for background, shape (nr,). r: rays (nr,2,3) as [origin, direction]; tr: triangles (nt,3,3), vertices A,B,C. Singular pairs and hits behind the origin are misses.

### q1168
Return how many pixels show each triangle as the nearest surface, shape (nt,), excluding background; ties choose first. r: rays (nr,2,3) as [origin, direction]; tr: triangles (nt,3,3), vertices A,B,C. Singular pairs and hits behind the origin are misses.

## Misconceptions

- **Misses can be filled with zero.** Zero wins every minimum; a miss must be `inf`.
- **`argmin` of an all-`inf` row is background.** It is a valid-looking index; check the minimum value first.
- **`s` is a distance.** It is a multiple of `D`; distance is `s * |D|`, equal only for unit directions.
- **Reduce over triangles before checking `s ≥ 0`.** A hit behind the camera would then be the nearest; mask per pair, then reduce.
