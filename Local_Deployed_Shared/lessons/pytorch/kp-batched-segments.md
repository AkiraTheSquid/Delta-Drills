---
kc: raytracing.batched-segments
title: Every ray against every segment
new_syntax: []
concepts: [pair-axes, judge-then-reduce]
supporting: ['raytracing.segment-intersection', 'numpy.broadcasting-rules', 'numpy.axis-reductions']
previews: []
faded: [1105, 1106, 1107, 1108]
guided: []
independent: [1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117]
integrated: [1118, 1119, 1120]
---

## Concept: Pair axes are independent

With `nr` rays and `ns` segments there are `nr × ns` candidate intersections, and the batched computation must keep both axes until each pair has its own verdict. The tool is broadcasting with inserted axes: give rays their own axis and segments their own, `o[:, None, :]` against `a[None, :, :]`, and every arithmetic step produces an `(nr, ns, …)` result in which position `[i, j]` is "ray `i` against segment `j`". The last axis still means coordinates.

The reason to insert axes explicitly, rather than relying on equal lengths, is that when `nr == ns` a plain `a − o` would silently pair ray `i` with segment `i` only — a zip, not a product — and produce a plausible-looking answer of the wrong shape. The reason the per-pair matrices are built by stacking on the *last* axis is that `torch.linalg` functions treat leading axes as a batch: an `(nr, ns, 2, 2)` tensor is `nr × ns` small matrices, and `det` or `solve` runs independently on each. Broadcasting a `(nr, 1, 2)` direction against a `(1, ns, 2)` edge needs one of them expanded to the full pair shape before stacking, because `stack` requires identical shapes; adding `t.zeros_like` of the other does that without a copy of meaning.

```python
import torch as t
o=t.tensor([[0.,0.],[1.,0.]])
a=t.tensor([[2.,1.],[3.,2.],[4.,3.]])
displacement=a[None,:,:]-o[:,None,:]
print(displacement)
# Hidden checks
assert displacement.shape==(2,3,2) and displacement[1,2].tolist()==[3.,3.]
```

## Worked example

We build a pair table for two "rays" and three "segments" reduced to single numbers, so the shape rule is visible on its own. Row `i` belongs to ray `i`; column `j` to segment `j`.

```python
import torch as t
a=t.tensor([1,2]); b=t.tensor([10,20,30])
table=a[:,None]+b[None,:]
print(table)
# Hidden checks
assert table.tolist()==[[11,21,31],[12,22,32]]
```

With equal counts the trap appears: `a + b` on two length-2 vectors is elementwise, a `(2,)` result pairing `0` with `0` and `1` with `1`. The inserted axes are what force the full product.

```python
c=t.tensor([10,20])
print(a+c, (a[:,None]+c[None,:]).shape)
# Hidden checks
assert (a+c).tolist()==[11,22] and (a[:,None]+c[None,:]).shape==(2,2)
```

## Faded practice

### q1105
Return displacement from every ray origin to every segment start, shape (nr,ns,2). r: rays (nr,2,3) as [origin, direction]; s: segments (ns,2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

```python starter
import torch as t

def solve(r,s):
    pass
```

```python solution
import torch as t

def solve(r,s):
    o=r[:,0,None,:2]
    a=s[None,:,0,:2]
    v=a-o
    return v
```

### q1106
Return determinants for every pair, shape (nr,ns). r: rays (nr,2,3) as [origin, direction]; s: segments (ns,2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

```python starter
import torch as t

def solve(r,s):
    pass
```

```python solution
import torch as t

def solve(r,s):
    o=r[:,0,None,:2]
    d=r[:,1,None,:2]
    a=s[None,:,0,:2]
    b=s[None,:,1,:2]
    d=d+t.zeros_like(a)
    e=(a-b)+t.zeros_like(o)
    m=t.stack((d,e),dim=-1)
    return t.linalg.det(m)
```

## Concept: Reduce only after judging each pair

Once every pair has a Boolean verdict, the hit table is `(nr, ns)`, and each question about the scene is a reduction along one axis. "Does ray `i` hit anything?" removes the segment axis: `hits.any(dim=1)`, shape `(nr,)`. "Was segment `j` seen by any ray?" removes the ray axis: `hits.any(dim=0)`, shape `(ns,)`. "How many segments does each ray hit?" is `hits.sum(dim=1)`. The axis you name is the one that disappears, and it must be the one you are asking *across*.

The reason to reduce last is that every ingredient of the verdict — the validity mask, `u ≥ 0`, `v` in `[0, 1]` — lives at the pair level; reducing any of them early, or reducing the wrong axis, produces a plausible image with wrong pixels. A placeholder solve for a singular pair must be masked out at the pair level before `any` ever sees it, and `u ≥ 0` must be applied per pair, or a segment behind the camera lights up a pixel.

```python
import torch as t
hits=t.tensor([[False,True,False],[False,False,False]])
print(hits.any(dim=1),hits.any(dim=0))
# Hidden checks
assert hits.any(1).tolist()==[True,False] and hits.any(0).tolist()==[False,True,False]
```

## Worked example

We reduce a `(3, 2)` hit table both ways. Three rays, two segments: the first ray hits one segment, the second hits both, the third hits none.

```python
import torch as t
hits=t.tensor([[True,False],[True,True],[False,False]])
print(hits.sum(dim=1))
# Hidden checks
assert hits.sum(1).tolist()==[1,2,0]
```

Reducing over the ray axis instead answers a question about segments: how many rays saw each one. The shape changes from `(3,)` to `(2,)`, which is the quickest check that the right axis went away.

```python
print(hits.sum(dim=0), hits.sum(dim=0).shape)
# Hidden checks
assert hits.sum(0).tolist()==[2,1]
```

## Faded practice

### q1107
Return hit verdicts for every pair, shape (nr,ns). r: rays (nr,2,3) as [origin, direction]; s: segments (ns,2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

```python starter
import torch as t

def solve(r,s):
    pass
```

```python solution
import torch as t

def solve(r,s):
    o=r[:,0,None,:2]
    d=r[:,1,None,:2]
    a=s[None,:,0,:2]
    b=s[None,:,1,:2]
    d=d+t.zeros_like(a)
    e=(a-b)+t.zeros_like(o)
    m=t.stack((d,e),dim=-1)
    v=a-o
    valid=t.linalg.det(m).abs()>=1e-8
    safe=t.where(valid[...,None,None],m,t.eye(2))
    uv=t.linalg.solve(safe,v[...,None])[...,0]
    u,w=uv[...,0],uv[...,1]
    hit=valid&(u>=0)&(w>=0)&(w<=1)
    return hit
```

### q1108
Return whether each ray hits anything, shape (nr,). r: rays (nr,2,3) as [origin, direction]; s: segments (ns,2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

```python starter
import torch as t

def solve(r,s):
    pass
```

```python solution
import torch as t

def solve(r,s):
    o=r[:,0,None,:2]
    d=r[:,1,None,:2]
    a=s[None,:,0,:2]
    b=s[None,:,1,:2]
    d=d+t.zeros_like(a)
    e=(a-b)+t.zeros_like(o)
    m=t.stack((d,e),dim=-1)
    v=a-o
    valid=t.linalg.det(m).abs()>=1e-8
    safe=t.where(valid[...,None,None],m,t.eye(2))
    uv=t.linalg.solve(safe,v[...,None])[...,0]
    u,w=uv[...,0],uv[...,1]
    hit=valid&(u>=0)&(w>=0)&(w<=1)
    return hit.any(dim=1)
```

## Solo practice

### q1109
Return the ray parameter u of the supporting-line intersection for every pair, or -1 for a singular pair, shape (nr,ns). r: rays (nr,2,3) as [origin, direction]; s: segments (ns,2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

### q1110
Return how many segments each ray hits, shape (nr,). r: rays (nr,2,3) as [origin, direction]; s: segments (ns,2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

### q1111
Return whether every ray hits at least one segment, as a scalar Boolean tensor. r: rays (nr,2,3) as [origin, direction]; s: segments (ns,2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

### q1112
Return whether each ray hits every segment, shape (nr,). r: rays (nr,2,3) as [origin, direction]; s: segments (ns,2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

### q1113
Return indices of rays that miss every segment, as a 1-D integer tensor. r: rays (nr,2,3) as [origin, direction]; s: segments (ns,2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

### q1114
Return total number of valid ray–segment intersections, as a scalar integer tensor. r: rays (nr,2,3) as [origin, direction]; s: segments (ns,2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

### q1115
Return number of rays hitting at least one segment, as a scalar integer tensor. r: rays (nr,2,3) as [origin, direction]; s: segments (ns,2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

### q1116
Return whether each ray hits exactly one segment, shape (nr,). r: rays (nr,2,3) as [origin, direction]; s: segments (ns,2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

### q1117
Return number of segment pairs whose line intersection is behind the ray origin, per ray, shape (nr,). r: rays (nr,2,3) as [origin, direction]; s: segments (ns,2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

## Integrated practice

### q1118
Return the smallest forward ray parameter among hit segments, or infinity on a miss, shape (nr,). r: rays (nr,2,3) as [origin, direction]; s: segments (ns,2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

### q1119
Return counts of rays hitting each segment strictly inside its endpoints, shape (ns,). r: rays (nr,2,3) as [origin, direction]; s: segments (ns,2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

### q1120
Return nearest-hit Euclidean travel distances, or infinity for misses, shape (nr,). r: rays (nr,2,3) as [origin, direction]; s: segments (ns,2,3) as [start, end]. Float tensors in the xy plane; a singular pair is a miss.

## Misconceptions

- **Equal counts let you skip the inserted axes.** Then `a − o` zips ray `i` with segment `i`; the product needs `[:, None]` and `[None, :]`.
- **`stack` broadcasts its inputs.** It requires identical shapes; expand the `(nr, 1, 2)` operand to `(nr, ns, 2)` first.
- **Reduce as soon as you can.** Every mask is per pair; reduce once, at the end, along the axis the question names.
- **`any(dim=0)` asks about rays.** It removes the ray axis and answers per segment; per-ray questions reduce `dim=1`.
