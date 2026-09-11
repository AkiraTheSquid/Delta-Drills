---
kc: raytracing.make-rays-2d
title: A 2-D fan of camera rays
new_syntax: []
concepts: [pixel-grid, pixel-direction]
supporting: ['raytracing.make-rays-1d', 'numpy.broadcasting-rules', 'numpy.reshape-flatten', 'torch.slice-assignment', 'numpy.stack-concat-interleave', 'tensor.row-normalization']
previews: []
faded: [1121, 1122, 1123, 1124]
guided: []
independent: [1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133]
integrated: [1134, 1135, 1136]
---

## Concept: Pixels form a product of two axes

An image of `ny × nz` pixels is stored as a flat list of `ny * nz` rays, and the flat order must be decided before anything is built: here `z` changes fastest, so the list walks across one row of `z` values, then moves to the next `y`. The pixel coordinates along each axis are `t.linspace(-limit, limit, n)` — `n` samples from `−limit` to `+limit` inclusive, with the single-pixel case sitting at `−limit`. The grid is the product of the two vectors, made by broadcasting `y[:, None]` against `z[None, :]` to `(ny, nz)`, stacking the two coordinates on a last axis to `(ny, nz, 2)`, and reshaping to `(ny * nz, 2)`.

The reason the product order matters is that `reshape(-1, 2)` reads the `(ny, nz)` grid row by row, so the axis you put first is the one that changes slowest. Put `z` first and every pixel lands in the wrong place — a transposed image that still has the right shape. The reason to test on a `2 × 3` grid rather than `3 × 3` is the same: a square image hides the swap, a rectangular one makes it a shape error.

```python
import torch as t
y=t.tensor([-3.,3.]); z=t.tensor([-1.,0.,1.])
yy=y[:,None]+t.zeros_like(z)[None,:]
zz=t.zeros_like(y)[:,None]+z[None,:]
print(t.stack((yy,zz),dim=-1).reshape(-1,2))
# Hidden checks
assert t.stack((yy,zz),dim=-1).reshape(-1,2).tolist()==[[-3.,-1.],[-3.,0.],[-3.,1.],[3.,-1.],[3.,0.],[3.,1.]]
```

## Worked example

We look at how a flat index maps back to a pixel. A `2 × 3` grid numbered in flat order shows `z` changing fastest: the first row holds `0, 1, 2`, the second `3, 4, 5`.

```python
import torch as t
image=t.arange(6).reshape(2,3)
print(image)
# Hidden checks
assert image.tolist()==[[0,1,2],[3,4,5]]
```

`linspace` samples the coordinates. With four samples over `[-1, 1]` the spacing is two thirds, and both ends are included; with one sample the result is just `-limit`.

```python
print(t.linspace(-1.,1.,4), t.linspace(-1.,1.,1))
# Hidden checks
assert t.allclose(t.linspace(-1.,1.,4),t.tensor([-1.,-1/3,1/3,1.])) and t.linspace(-1.,1.,1).tolist()==[-1.]
```

## Faded practice

### q1121
Return y pixel coordinates as a vector, shape (ny,). ny: pixels along y; yl: half-width along y. Samples run inclusively from -limit to +limit (a single pixel sits at -limit).

```python starter
import torch as t

def solve(ny,yl):
    pass
```

```python solution
import torch as t

def solve(ny,yl):
    return t.linspace(-yl,yl,ny)
```

### q1122
Return all pixel yz coordinates, shape (ny*nz,2). ny: pixels along y; nz: pixels along z; yl: half-width along y; zl: half-width along z. Each axis is sampled inclusively from -limit to +limit (a single pixel sits at -limit); flatten with z changing fastest.

```python starter
import torch as t

def solve(ny,nz,yl,zl):
    pass
```

```python solution
import torch as t

def solve(ny,nz,yl,zl):
    y=t.linspace(-yl,yl,ny)
    z=t.linspace(-zl,zl,nz)
    yy=y[:,None]+t.zeros_like(z)[None,:]
    zz=t.zeros_like(y)[:,None]+z[None,:]
    return t.stack((yy,zz),dim=-1).reshape(-1,2)
```

## Concept: A pixel names a direction through the image plane

The camera sits at the origin and looks along `+x`; the image plane is `x = 1`. A pixel at `(y, z)` on that plane is the point `(1, y, z)`, and the ray through it from the origin has direction `(1, y, z)` — the point itself, because the origin is zero. Each stored ray is `[origin, direction]`, shape `(2, 3)`, so the whole fan is `(ny * nz, 2, 3)`: allocate zeros, set `[:, 1, 0]` to `1`, and write the flattened `y` and `z` grids into `[:, 1, 1]` and `[:, 1, 2]`.

The reason direction equals the pixel point only for a camera at the origin is that a direction is a difference: `pixel − origin`. Move the camera to `O` while the image plane stays where it is in the world and the direction becomes `pixel − O`; move the camera *and* its plane together and the direction is unchanged. The stored ray keeps both origin and direction so that later code can evaluate `O + u·D` without knowing where the camera was.

```python
import torch as t
pixels=t.tensor([[1.,-.5,-1.],[1.,-.5,1.]])
r=t.zeros(2,2,3)
r[:,1]=pixels
print(r)
# Hidden checks
assert r[:,0].eq(0).all() and r[:,1].tolist()==pixels.tolist()
```

## Worked example

We move the camera to `(0, 2, 0)` while the image plane stays at `x = 1`. The direction to a pixel is the pixel minus the new origin, and it is no longer equal to the pixel's coordinates.

```python
import torch as t
o=t.tensor([0.,2.,0.]); pixel=t.tensor([1.,3.,1.])
d=pixel-o
print(d)
# Hidden checks
assert d.tolist()==[1.,1.,1.]
```

Walking `u = 2` along that ray from the origin lands at `O + 2·D`; at `u = 1` it passes exactly through the pixel, which is the check that the direction was computed the right way round.

```python
print(o+1*d, o+2*d)
# Hidden checks
assert (o+1*d).tolist()==pixel.tolist() and (o+2*d).tolist()==[2.,4.,2.]
```

## Faded practice

### q1123
Return camera rays from the origin through plane x=1, shape (ny*nz,2,3). ny: pixels along y; nz: pixels along z; yl: half-width along y; zl: half-width along z. Each axis is sampled inclusively from -limit to +limit (a single pixel sits at -limit); flatten with z changing fastest.

```python starter
import torch as t

def solve(ny,nz,yl,zl):
    pass
```

```python solution
import torch as t

def solve(ny,nz,yl,zl):
    y=t.linspace(-yl,yl,ny)
    z=t.linspace(-zl,zl,nz)
    yy=y[:,None]+t.zeros_like(z)[None,:]
    zz=t.zeros_like(y)[:,None]+z[None,:]
    r=t.zeros(ny*nz,2,3)
    r[:,1,0]=1
    r[:,1,1]=yy.reshape(-1)
    r[:,1,2]=zz.reshape(-1)
    return r
```

### q1124
Return each camera ray’s direction, shape (ny*nz,3). ny: pixels along y; nz: pixels along z; yl: half-width along y; zl: half-width along z. Each axis is sampled inclusively from -limit to +limit (a single pixel sits at -limit); flatten with z changing fastest.

```python starter
import torch as t

def solve(ny,nz,yl,zl):
    pass
```

```python solution
import torch as t

def solve(ny,nz,yl,zl):
    y=t.linspace(-yl,yl,ny)
    z=t.linspace(-zl,zl,nz)
    yy=y[:,None]+t.zeros_like(z)[None,:]
    zz=t.zeros_like(y)[:,None]+z[None,:]
    r=t.zeros(ny*nz,2,3)
    r[:,1,0]=1
    r[:,1,1]=yy.reshape(-1)
    r[:,1,2]=zz.reshape(-1)
    return r[:,1]
```

## Solo practice

### q1125
Return the yz coordinates as an image grid, shape (ny,nz,2). ny: pixels along y; nz: pixels along z; yl: half-width along y; zl: half-width along z. Each axis is sampled inclusively from -limit to +limit (a single pixel sits at -limit); flatten with z changing fastest.

### q1126
Return only the rays through pixels with nonnegative y coordinate, from the origin through plane x=1, shape (k,2,3), in image order. ny: pixels along y; nz: pixels along z; yl: half-width along y; zl: half-width along z. Each axis is sampled inclusively from -limit to +limit (a single pixel sits at -limit); flatten with z changing fastest.

### q1127
Return points reached by these rays at parameter u=2, shape (ny*nz,3). ny: pixels along y; nz: pixels along z; yl: half-width along y; zl: half-width along z. Each axis is sampled inclusively from -limit to +limit (a single pixel sits at -limit); flatten with z changing fastest.

### q1128
Return only rays through the first z-column of the image, shape (ny,2,3). ny: pixels along y; nz: pixels along z; yl: half-width along y; zl: half-width along z. Each axis is sampled inclusively from -limit to +limit (a single pixel sits at -limit); flatten with z changing fastest.

### q1129
Return rays from camera (0,1,0) through the fixed plane x=1, shape (ny*nz,2,3). ny: pixels along y; nz: pixels along z; yl: half-width along y; zl: half-width along z. Each axis is sampled inclusively from -limit to +limit (a single pixel sits at -limit); flatten with z changing fastest.

### q1130
Return directions through plane x=2 using the same yz pixel coordinates, shape (ny*nz,3). ny: pixels along y; nz: pixels along z; yl: half-width along y; zl: half-width along z. Each axis is sampled inclusively from -limit to +limit (a single pixel sits at -limit); flatten with z changing fastest.

### q1131
Return each pixel’s squared distance from the centre of plane x=1, as a flattened vector (ny*nz,). ny: pixels along y; nz: pixels along z; yl: half-width along y; zl: half-width along z. Each axis is sampled inclusively from -limit to +limit (a single pixel sits at -limit); flatten with z changing fastest.

### q1132
Return the yz coordinates of the final pixel in each y-row, shape (ny,2). ny: pixels along y; nz: pixels along z; yl: half-width along y; zl: half-width along z. Each axis is sampled inclusively from -limit to +limit (a single pixel sits at -limit); flatten with z changing fastest.

### q1133
Return directions from camera (0,0,1) to pixels on fixed plane x=1, shape (ny*nz,3). ny: pixels along y; nz: pixels along z; yl: half-width along y; zl: half-width along z. Each axis is sampled inclusively from -limit to +limit (a single pixel sits at -limit); flatten with z changing fastest.

## Integrated practice

### q1134
Return rays from the origin with unit-length directions through plane x=1, shape (ny*nz,2,3). ny: pixels along y; nz: pixels along z; yl: half-width along y; zl: half-width along z. Each axis is sampled inclusively from -limit to +limit (a single pixel sits at -limit); flatten with z changing fastest.

### q1135
Return rays through plane x=1 with the image reversed along z within each y row, shape (ny*nz,2,3). ny: pixels along y; nz: pixels along z; yl: half-width along y; zl: half-width along z. Each axis is sampled inclusively from -limit to +limit (a single pixel sits at -limit); flatten with z changing fastest.

### q1136
Return rays from camera (-1,1,0) through fixed plane x=1 with unit-length directions, shape (ny*nz,2,3). ny: pixels along y; nz: pixels along z; yl: half-width along y; zl: half-width along z. Each axis is sampled inclusively from -limit to +limit (a single pixel sits at -limit); flatten with z changing fastest.

## Misconceptions

- **The flat order does not matter.** It decides which pixel each ray is; `z` fastest means `y` is the outer axis of the grid.
- **`linspace` excludes the endpoint like `arange`.** It includes both ends, and one sample sits at the lower limit.
- **Direction equals the pixel point.** Only for a camera at the origin; in general it is pixel minus origin.
- **A square test grid is sufficient.** It hides a swapped axis order; use `ny ≠ nz`.
