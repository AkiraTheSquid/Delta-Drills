---
kc: raytracing.make-rays-1d
title: make_rays_1d: a fan of rays as one (n, 2, 3) tensor
supporting: [torch.out-argument, torch.slice-assignment, raytracing.ray-parametrisation, numpy.ranges, numpy.constructors]
new_syntax: []
previews: []
faded: [831, 832]
guided: []
independent: [833, 834, 835, 836, 837, 838]
integrated: [839, 840, 841]
---

## Concept: zeros, then write the two direction columns

ARENA's first exercise, `make_rays_1d(num_pixels, y_limit)`, asks for a
**fan** of rays: all from the origin, all with direction x = 1, and direction
y spread evenly from `-y_limit` to `+y_limit`. The result is one tensor of
shape **`(num_pixels, 2, 3)`** — one `(2, 3)` ray slab per pixel.

Everything you need is the three pages before this one, in order:

1. **Canvas.** `t.zeros((num_pixels, 2, 3))` — every origin is already
   `(0, 0, 0)` and every direction z is already `0`. Two columns remain.
2. **Direction x.** Every ray's `[1, 0]` slot is `1`: a scalar broadcast
   through a slice, `rays[:, 1, 0] = 1`.
3. **Direction y.** `num_pixels` values from `-y_limit` to `y_limit`, written
   into the `[:, 1, 1]` column — `t.linspace(..., out=rays[:, 1, 1])`, or the
   same with assignment.

```python
import torch as t

def make_rays_1d(num_pixels, y_limit):
    rays = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    rays[:, 1, 0] = 1
    return rays

print(make_rays_1d(3, 1.0))
```

Read the output as three slabs. Row 0 of each is the origin; row 1 is
`(1, y, 0)` with `y` walking `-1, 0, 1`. Nothing else was touched.

The habit to take from this: **allocate the final shape once, then write the
parts that vary**. Building rows in a loop and stacking them gives the same
numbers and is the thing you will unlearn in the next chapter.

## Worked example

The same construction with y spread over `[-2, 2]` in 5 steps, checked
against the two facts that define it:

```python
import torch as t

rays = t.zeros((5, 2, 3))
rays[:, 1, 0] = 1
rays[:, 1, 1] = t.linspace(-2.0, 2.0, 5)
print(rays[:, 1])                          # every direction row
assert (rays[:, 0] == 0).all()             # all origins at 0
assert rays[:, 1, 1].tolist() == [-2.0, -1.0, 0.0, 1.0, 2.0]
```

| `num_pixels, y_limit` | direction rows `rays[:, 1]` |
| --- | --- |
| `2, 1.0` | `[[1, -1, 0], [1, 1, 0]]` |
| `3, 3.0` | `[[1, -3, 0], [1, 0, 0], [1, 3, 0]]` |

## Faded practice

### q831
Which row is the direction, which column is x, and the two ends of the fan.

```python starter
def solve(num_pixels, y_limit):
    """The fan of rays: zeros, then two direction columns."""
    rays = t.zeros((num_pixels, 2, 3))
    rays[_____, _____, _____] = 1
    t.linspace(_____, _____, num_pixels, out=rays[:, 1, 1])
    return rays.tolist()
```

```python solution
def solve(num_pixels, y_limit):
    """The fan of rays: zeros, then two direction columns."""
    rays = t.zeros((num_pixels, 2, 3))
    rays[:, 1, 0] = 1
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    return rays.tolist()
```

### q832
Same slots as before; only the spelling of the write changes.

```python starter
def solve(num_pixels, y_limit):
    """make_rays_1d with the y column assigned."""
    rays = t.zeros((num_pixels, 2, 3))
    rays[:, 1, 0] = 1
    rays[_____, _____, _____] = t.linspace(-y_limit, y_limit, num_pixels)
    return rays.tolist()
```

```python solution
def solve(num_pixels, y_limit):
    """make_rays_1d with the y column assigned."""
    rays = t.zeros((num_pixels, 2, 3))
    rays[:, 1, 0] = 1
    rays[:, 1, 1] = t.linspace(-y_limit, y_limit, num_pixels)
    return rays.tolist()
```

## Solo practice

### q833
The direction rows of the fan.

### q834
make_rays_1d rotated into the x–z plane.

### q835
The fan, moved to a shared non-zero origin.

### q836
Three checks on a freshly built fan.

### q837
The same fan with a different x speed.

### q838
Where the fan is at a given x.

## Integrated practice

### q839
make_rays_2d: a grid of directions in one canvas.

### q840
The fan against a vertical line.

### q841
Reparametrising the fan without moving it.

## Misconceptions

- **"Build each ray, then stack."** It gives the same numbers, but the
  exercise is teaching the allocate-then-write habit that every later
  chapter uses. One canvas, two column writes.
- **"Row 0 / row 1 are x / y."** Rows are origin and direction; the *columns*
  are x, y, z. `rays[:, 1, 1]` is direction-y for every ray.
- **"`linspace(0, y_limit)`."** The fan is symmetric: from `-y_limit` to
  `+y_limit`.
