---
kc: math.ray-distance
kind: math
title: Nearest hit, depth, and actual distance
supporting: [math.barycentric-coordinates, math.dot-products-norms]
new_syntax: []
previews: []
concepts: [distance]
faded: [50026, 50027]
guided: []
independent: [50028, 50029, 50096, 50097, 50099, 50100, 50101]
integrated: [50030, 50031, 50103]
---

## Concept: Compare hits along the same ray

This lesson builds on vector norms and unit vectors. Here those lengths distinguish geometric distance from a ray parameter.

A mesh contains many triangles, so a ray may have several valid forward intersections. Visibility selects the nearest. For a fixed nonzero direction $D$, a forward point $O+sD$ is at Euclidean distance
$$\|(O+sD)-O\|=s\,\|D\|=s\sqrt{D_x^2+D_y^2+D_z^2}$$
from $O$. The factor $\|D\|$ is fixed along that ray, so minimizing valid $s$ also minimizes distance.

The parameter $s$ is not generally a distance: doubling $D$ halves the $s$ needed to reach the same point. Comparing raw $s$ across differently scaled rays can mislead. The x displacement is $sD_x$; the world x-coordinate is $O_x+sD_x$. Neither is generally Euclidean distance. For ARENA's camera $O=0$ and $D_x=1$, $s$ equals x-depth; off-axis rays still have $\|D\|>1$.

Before taking a minimum, replace every invalid or backward hit by $+\infty$. A miss must not become zero, which would beat every positive hit. If all pairs miss, the minimum remains $+\infty$, an explicit no-hit result. This order also prevents a negative $s$ behind the origin from hiding a valid surface in front.

Three candidate hits, one backward and one invalid:

```python
import torch as t

s = t.tensor([4.0, -2.0, 1.0])
valid = t.tensor([True, True, False])

forward = valid & (s >= 0)
candidates = t.where(forward, s, t.inf)
nearest = candidates.min()
print(candidates)
print(nearest)
# Hidden checks
assert nearest.item() == 4.0
assert _delta_output == 'tensor([4., inf, inf])\ntensor(4.)\n'
```

## Worked example

A ray with direction $D=(1,2,2)$ hits a surface at $s=4$. How deep is the hit in $x$, and how far has the ray travelled?

**On paper.** The direction's length:
$$\|D\|=\sqrt{1^2+2^2+2^2}=\sqrt9=3.$$
The displacement to the hit and its length:
$$sD=4\begin{pmatrix}1\\2\\2\end{pmatrix}=\begin{pmatrix}4\\8\\8\end{pmatrix},\qquad \|sD\|=\sqrt{16+64+64}=\sqrt{144}=12=s\,\|D\|.$$
The x displacement is $4$ and the Euclidean travel distance is $12$: two measurements of the same hit.

**In code.** Depth and distance from $s$ and $D$.

```python
import torch as t

direction = t.tensor([1.0, 2.0, 2.0])
s = 4

length = t.linalg.norm(direction)
depth = s * direction[0]
distance = s * length
print(length)
print(depth)
print(distance)
# Hidden checks
assert length.item() == 3.0
assert depth.item() == 4.0
assert distance.item() == 12.0
assert _delta_output == 'tensor(3.)\ntensor(4.)\ntensor(12.)\n'
```

## Faded practice

### q50026
One ray has candidate s = [−1, 5, 2, 0.5]. The first three pairs are nonsingular and inside their triangles; the last is outside its triangle. Which hit is visible?

### q50027
D = (2, 3, 6), s = 2. What is the Euclidean distance from O to O + sD?

## Solo practice

### q50028
O = (3, 0, 0), D = (2, 0, 0), s = 4. Which pair is (x displacement, world x-coordinate)?

### q50029
A ray misses every triangle. Miss candidates become +∞ before taking the minimum. What results?

### q50096
One ray has candidate hits s = [3, −1, 0.5, 6] with valid = [True, True, False, True]. Which s is the visible hit?

### q50097
O = (0, 0, 0), D = (1, 2, 2). A surface is hit at s = 3. How far is the hit point from O?

### q50099
A camera ray from O = (0, 0, 0) has direction D = (1, 0.6, 0.8) and hits at s = 5. How far is the hit point from O?

### q50100
Two rays share the origin O. Ray A has D = (1, 0, 0) and hits at s = 4. Ray B has D = (1, 1, 0) and hits at s = 3. Which hit point is farther from O?

### q50101
O = (0, 0, 0), D = (0, 3, 4). A hit is at s = 2. Using the unit direction D̂ = D / ‖D‖ instead, what parameter reaches the same point?

## Integrated practice

### q50030
Ray A has ‖D_A‖ = 10, s_A = 1. Ray B has ‖D_B‖ = 1, s_B = 3. Which hit is closer to its own origin?

### q50031
D = (1, 0, 0) hits at s = 8. Replace D with D′ = (4, 0, 0), keeping O and the surface fixed. What are the new parameter and distance?

### q50103
Three rays share O = (0, 0, 0). Ray A: D = (1, 0, 0), hit at s = 6. Ray B: D = (1, 1, 1), hit at s = 4. Ray C: D = (2, 0, 0), hit at s = 3.5. Order the hit points from nearest to farthest.
