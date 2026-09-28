---
kc: raytracing.make-rays-2d
title: A 2-D fan of camera rays
new_syntax: []
concepts: [pixel-grid, pixel-direction]
supporting: ['raytracing.make-rays-1d', 'torch.broadcasting-rules', 'torch.reshape-flatten', 'torch.slice-assignment', 'torch.stack-concat-interleave', 'tensor.row-normalization']
previews: []
faded: [1121, 1122, 1123, 1124]
guided: []
independent: [1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133]
integrated: [1134, 1135, 1136, 1710, 1711, 1712, 1713]
---

## Concept: Pixels form a product of two axes

An image of `ny × nz` pixels is stored as a flat list of `ny * nz` rays, and the flat order must be decided before anything is built: here `z` changes fastest, so the list walks across one row of `z` values, then moves to the next `y`. The pixel coordinates along each axis are `t.linspace(-limit, limit, n)` — `n` samples from `−limit` to `+limit` inclusive, with the single-pixel case sitting at `−limit`. The grid is the product of the two vectors, made by broadcasting `y[:, None]` against `z[None, :]` to `(ny, nz)`, stacking the two coordinates on a last axis to `(ny, nz, 2)`, and reshaping to `(ny * nz, 2)`.

The reason the product order matters is that `reshape(-1, 2)` reads the `(ny, nz)` grid row by row, so the axis you put first is the one that changes slowest. Put `z` first and every pixel lands in the wrong place — a transposed image that still has the right shape. The reason to test on a `2 × 3` grid rather than `3 × 3` is the same: a square image hides the swap, a rectangular one shows it in the coordinate sequence — the flat count `ny * nz` is the same either way, so the shape alone never tells.

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

The camera sits at the point `(0, 0, 0)` and looks along `+x`. In front of it is the image plane, the flat sheet `x = 1`. A pixel with coordinates `(y, z)` on that sheet is the 3-D point `(1, y, z)`: the `x` is always `1`, and `y`, `z` come from the pixel grid.

```python
import torch as t
pixel=t.tensor([1.,-1.,2.])
print(pixel)
# Hidden checks
assert pixel.tolist()==[1.,-1.,2.]
```

A ray needs a direction, and a direction is a difference: where you are going minus where you start, `pixel − origin`. Below, `origin` is a variable holding the camera's position. Here that position is `(0, 0, 0)`, so subtracting it changes nothing and the direction comes out equal to the pixel.

```python
origin=t.zeros(3)
print(origin, pixel-origin)
# Hidden checks
assert origin.tolist()==[0.,0.,0.] and (pixel-origin).tolist()==pixel.tolist()
```

One ray is stored as two rows, `[origin, direction]`, so its shape is `(2, 3)`: row 0 is where the ray starts, row 1 is the way it points.

```python
ray=t.stack((origin,pixel-origin))
print(ray, ray.shape)
# Hidden checks
assert ray.shape==(2,3) and ray[0].tolist()==[0.,0.,0.] and ray[1].tolist()==[1.,-1.,2.]
```

The whole fan is one such ray per pixel, shape `(ny * nz, 2, 3)`. Start from zeros: every row 0 is already the origin, and every direction still needs filling. Index `[:, 1]` picks the direction row of every ray at once.

```python
rays=t.zeros(6,2,3)
print(rays.shape, rays[:,1])
# Hidden checks
assert rays.shape==(6,2,3) and rays.eq(0).all()
```

Every pixel sits on the plane `x = 1`, so the `x` part of every direction is `1`. That is component `0` of the direction row, `[:, 1, 0]`.

```python
rays[:,1,0]=1
print(rays[:,1])
# Hidden checks
assert rays[:,1,0].eq(1).all() and rays[:,1,1:].eq(0).all()
```

The `y` and `z` parts come from the pixel grid built in the first segment, flattened with `reshape(-1)` into one value per pixel in the same `z`-fastest order as the rays.

```python
y=t.tensor([-1.,1.]); z=t.tensor([-2.,0.,2.])
yy=y[:,None]+t.zeros_like(z)[None,:]
zz=t.zeros_like(y)[:,None]+z[None,:]
print(yy.reshape(-1), zz.reshape(-1))
# Hidden checks
assert yy.reshape(-1).tolist()==[-1.,-1.,-1.,1.,1.,1.] and zz.reshape(-1).tolist()==[-2.,0.,2.,-2.,0.,2.]
```

Write them into components `1` and `2` of the direction row, and each ray now points through its own pixel.

```python
rays[:,1,1]=yy.reshape(-1)
rays[:,1,2]=zz.reshape(-1)
print(rays)
# Hidden checks
assert rays[:,0].eq(0).all() and rays[3,1].tolist()==[1.,1.,-2.] and rays[2,1].tolist()==[1.,-1.,2.]
```

## Worked example

Now move the camera: `origin` becomes `(0, 2, 0)` while the image plane stays at `x = 1`. The subtraction `pixel − origin` now does real work, and the direction is no longer the pixel's coordinates.

```python
import torch as t
origin=t.tensor([0.,2.,0.])
pixel=t.tensor([1.,3.,1.])
d=pixel-origin
print(d)
# Hidden checks
assert d.tolist()==[1.,1.,1.]
```

The ray keeps both rows so later code can find any point on it as `origin + u·d` without knowing where the camera was. At `u = 1` it passes exactly through the pixel, which checks that the direction was computed the right way round; `u = 2` is twice as far along.

```python
print(origin+1*d, origin+2*d)
# Hidden checks
assert (origin+1*d).tolist()==pixel.tolist() and (origin+2*d).tolist()==[2.,4.,2.]
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

### q1710
Return the flat pixel index whose camera ray (from the origin through the plane x=1) points closest to the +x axis, i.e. has the largest cosine with (1,0,0), as an int64 scalar; ties choose the earlier pixel. ny, nz: pixels along y and z; yl, zl: half-widths along y and z. Each axis is sampled at equally spaced points from −limit to +limit inclusive (a single pixel sits at −limit), and z changes fastest in the flat order.

### q1711
Return camera rays from the origin through the image plane x=f with UNIT-LENGTH directions, shape (ny*nz,2,3): each direction is the pixel point (f, y, z) scaled to length 1. ny, nz: pixels along y and z; yl, zl: half-widths along y and z; f: the plane's distance along x, positive. Each axis is sampled at equally spaced points from −limit to +limit inclusive (a single pixel sits at −limit), and z changes fastest in the flat order.

### q1712
Return rays that leave from a vertical bar of cameras instead of one point, shape (ny*nz,2,3): the ray for pixel (y, z) starts at origin (0, y, 0) and has direction (1, 0, z). ny, nz: pixels along y and z; yl, zl: half-widths along y and z. Each axis is sampled at equally spaced points from −limit to +limit inclusive (a single pixel sits at −limit), and z changes fastest in the flat order.

### q1713
Return the flat indices of the pixels on the image border — any pixel in the first or last row, or in the first or last column — in increasing flat order, shape (2·ny + 2·nz − 4,), int64. ny: pixels along y (rows); nz: pixels along z (columns), both ≥ 2; z changes fastest in the flat order.

## Misconceptions

- **The flat order does not matter.** It decides which pixel each ray is; `z` fastest means `y` is the outer axis of the grid.
- **`linspace` excludes the endpoint like `arange`.** It includes both ends, and one sample sits at the lower limit.
- **Direction equals the pixel point.** Only for a camera at the origin; in general it is pixel minus origin.
- **A square test grid is sufficient.** It hides a swapped axis order; use `ny ≠ nz`.
