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
independent: [50028, 50029]
integrated: [50030, 50031]
---

## Concept: Compare hits along the same ray

First master [dot products, norms, and unit vectors](?lesson=math.dot-products-norms). Here those lengths distinguish geometric distance from a ray parameter.

A mesh contains many triangles, so a ray may have several valid forward intersections. Visibility selects the nearest. For a fixed nonzero direction D, a forward point O + sD is Euclidean distance s‖D‖ from O, where ‖D‖ = √(Dₓ² + Dᵧ² + D_z²). The factor ‖D‖ is fixed along that ray, so minimizing valid s also minimizes distance.

The parameter s is not generally a distance: doubling D halves the s needed to reach the same point. Comparing raw s across differently scaled rays can mislead. The x displacement is sDₓ; the world x-coordinate is Oₓ + sDₓ. Neither is generally Euclidean distance. For ARENA's camera O = 0 and Dₓ = 1, s equals x-depth; off-axis rays still have ‖D‖ > 1.

Before taking a minimum, replace every invalid or backward hit by +∞. A miss must not become zero, which would beat every positive hit. If all pairs miss, the minimum remains +∞, an explicit no-hit result. This order also prevents a negative s behind the origin from hiding a valid surface in front.

```python
s = [4.0, -2.0, 1.0]
valid = [True, True, False]
candidates = [x if ok and x >= 0 else float('inf')
              for x, ok in zip(s, valid)]
nearest = min(candidates)
print(nearest)
# Hidden checks
assert nearest == 4.0
assert _delta_output == '4.0\n'
```

## Worked example

A ray has D = (1, 2, 2) and hits at s = 4. Its direction length is 3. Its x displacement is 4 and its Euclidean travel distance is 12: two measurements of the same hit.

```python
from math import sqrt
direction, s = (1, 2, 2), 4
length = sqrt(sum(component**2
                  for component in direction))
depth, distance = s * direction[0], s * length
print(depth, distance)
# Hidden checks
assert depth == 4 and distance == 12.0
assert _delta_output == '4 12.0\n'
```

Return to [the connected PyTorch lesson](?lesson=raytracing.mesh-visibility) to implement this reasoning. ARENA 0.1 transfer: `raytrace_mesh`.

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

## Integrated practice

### q50030
Ray A has ‖D_A‖ = 10, s_A = 1. Ray B has ‖D_B‖ = 1, s_B = 3. Which hit is closer to its own origin?

### q50031
D = (1, 0, 0) hits at s = 8. Replace D with D′ = (4, 0, 0), keeping O and the surface fixed. What are the new parameter and distance?
