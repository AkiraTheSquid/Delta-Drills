---
kc: numpy.geometry-transforms
title: Geometry — coordinates and transforms
supporting: [numpy.dot-matmul-patterns, numpy.slicing-views, numpy.tile-repeat-meshgrid]
new_syntax: []
faded: [66, 182]
guided: []
independent: [105, 201]
---

## Concept: points live in rows — cartesian to polar

Geometric computation over point sets is column bookkeeping plus the linear
algebra you already have. The convention: **points live in rows.** An (n, 2)
array is n points; `z[:, 0]` is all the x's, `z[:, 1]` all the y's. Formulas
written on one point vectorize over the column vectors automatically:

- **Cartesian → polar**: `r = np.sqrt(x**2 + y**2)`,
  `theta = np.arctan2(y, x)`. Note `arctan2(y, x)` takes y FIRST and handles
  all four quadrants (plain `arctan(y/x)` loses quadrant information and
  divides by zero on the y-axis).

## Worked example

```python
import numpy as np

z = np.array([[1.0, 0.0],
              [0.0, 2.0]])

# Column bookkeeping: x's and y's as vectors, then per-point formulas.
x, y = z[:, 0], z[:, 1]
r = np.sqrt(x ** 2 + y ** 2)
theta = np.arctan2(y, x)                 # y FIRST — full-quadrant angle
assert r.tolist() == [1.0, 2.0]
assert np.allclose(theta, [0.0, np.pi / 2])   # +x axis; +y axis
```

Why: unpacking `x, y = z[:, 0], z[:, 1]` FIRST, then writing the scalar
formula, is the pattern: derive on one point, run on all. The arctan2
argument order (y, x) is checked by the two axis points — get it backwards
and the asserts catch it immediately.

## Faded practice

### q66
Cartesian (n, 2) points → polar radii and angles.

```python starter
import numpy as np

def solve(z):
    """(radii, angles) of the points — angles via the quadrant-aware arctan."""
    x, y = z[:, 0], z[:, 1]
    r = np.sqrt(x ** 2 + y ** 2)
    t = np._____(y, x)
    return r, t
```

```python solution
import numpy as np

def solve(z):
    """(radii, angles) of the points — angles via the quadrant-aware arctan."""
    x, y = z[:, 0], z[:, 1]
    r = np.sqrt(x ** 2 + y ** 2)
    t = np.arctan2(y, x)
    return r, t
```

## Concept: transforms are matmuls — homogeneous coordinates

A linear transform T applied to all points at once is `pts @ T.T` — each row
(point) gets dotted with each row of T. No per-point loop.

For transforms that include TRANSLATION, the standard trick is
**homogeneous coordinates**: append a 1 to every point
(`np.c_[pts, np.ones(len(pts))]` — `np.c_` is column-wise concatenation),
multiply by the 3×3 homogeneous matrix, then **de-homogenize** by dividing
the first two columns by the third: `out[:, :2] / out[:, 2:3]`. Keeping the
divisor as a (n, 1) slice — `2:3`, not `2` — preserves the column shape so
the division broadcasts per row. Affine transforms leave the third column at
1 and the division is a formality; projective ones genuinely need it.

(Fields over grids — e.g. a Gaussian bump on meshgrid coordinates — are this
KP's formulas evaluated on coordinate matrices; see independent practice.)

## Worked example

```python
import numpy as np

# Homogeneous transform: translate by (+2, -1).
t = np.array([[1.0, 0.0, 2.0],
              [0.0, 1.0, -1.0],
              [0.0, 0.0, 1.0]])
pts = np.array([[0.0, 0.0],
                [1.0, 1.0]])

h = np.c_[pts, np.ones(len(pts))]        # append the 1s column: (n, 3)
out = h @ t.T                            # transform all points at once
result = out[:, :2] / out[:, 2:3]        # de-homogenize (w column is 1 here)
assert result.tolist() == [[2.0, -1.0],
                           [3.0, 0.0]]
```

Why: `h @ t.T` rather than looping `t @ p` per point — points-as-rows means
the matrix arrives transposed. Verify with the origin: it must land exactly
on the translation column (2, −1). The `2:3` slice keeps a (n, 1) column so
the division broadcasts row-wise.

## Faded practice

### q182
Apply a 3×3 homogeneous transform to (n, 2) points.

```python starter
import numpy as np

def solve(pts, t):
    """Transformed points: append 1s, multiply, de-homogenize."""
    h = np.c_[pts, np.ones(len(pts))]
    out = h @ _____
    return out[:, :2] / out[:, 2:3]
```

```python solution
import numpy as np

def solve(pts, t):
    """Transformed points: append 1s, multiply, de-homogenize."""
    h = np.c_[pts, np.ones(len(pts))]
    out = h @ t.T
    return out[:, :2] / out[:, 2:3]
```

## Independent practice

From the drill bank: q105 (Gaussian bump over an n×n grid — linspace +
meshgrid coordinates, then the exp formula), q201 (distance from every point
to every LINE through two given points — the cross-product distance formula,
vectorized over both sets; derive the scalar formula first, then broadcast).

## Misconceptions

- **"Angle = arctan(y/x)."** — Loses the quadrant (arctan can't tell
  (1,1) from (−1,−1)) and explodes at x=0. `np.arctan2(y, x)` — y first —
  is the vectorized, quadrant-correct form.
- **"Translation can ride in a 2×2 matrix."** — Linear maps fix the origin;
  translation doesn't. Hence homogeneous coordinates: the appended 1 gives
  the matrix a column to put the offset in.
- **"Transform points one at a time: t @ p."** — Row-stacked points
  transform en masse as `pts @ t.T`. Same math, one matmul, and the shape
  (n, 2 or 3) documents itself.
