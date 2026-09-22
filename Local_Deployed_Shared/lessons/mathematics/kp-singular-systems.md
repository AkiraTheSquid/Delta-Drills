---
kc: math.singular-systems
kind: math
title: Unique crossings and batched validity
supporting: [math.intersection-systems, math.determinants-invertibility]
new_syntax: []
previews: []
concepts: [validity]
faded: [50012, 50013]
guided: []
independent: [50014, 50015]
integrated: [50016, 50017]
---

## Concept: A solver result needs a validity mask

First master [determinants and invertibility](?lesson=math.determinants-invertibility), including the difference between no solutions and infinitely many. Here we apply that distinction to intersection handling.

For a 2 × 2 matrix with rows [a, b] and [c, d], det(M) = ad − bc. Its absolute value measures the area scaling of the column directions. A nonzero determinant means the columns are independent, so every right-hand side has one solution.

A zero determinant means the columns collapse onto a line or point. The system can have no solution or infinitely many, depending on b. It does not mean geometric objects never overlap. ARENA's simplified contract treats singular pairs as misses, including coincident lines and degenerate objects.

The batched implementation declares a pair invalid when |det(M)| < 10⁻⁸. This is a course convention, not a scale-invariant accuracy test: rescaling columns rescales the determinant. Substitute an identity matrix for invalid pairs before solving so one singular pair cannot abort the batch. Keep the original validity mask: a solution of the replacement matrix is not evidence of a hit.

For each ray, combine validity with membership for every segment pair, then reduce over segments using “any.” Reducing over rays answers a different question.

```python
a, b, c, d = 1, 2, 3, 6
determinant = a * d - b * c
print(determinant)
# Hidden checks
assert determinant == 0
assert _delta_output == '0\n'
```

## Worked example

Two rays each have three candidate crossings. A hit needs both solver validity and geometric membership. The first ray's only apparent hit came from a replaced matrix; the second has a real hit.

```python
valid = [[False, True, True], [True, True, False]]
within_bounds = [[True, False, False], [False, True, True]]
hits = [any(v and h for v, h in zip(vrow, hrow))
        for vrow, hrow in zip(valid, within_bounds)]
print(hits)
# Hidden checks
assert hits == [False, True]
assert _delta_output == '[False, True]\n'
```

Return to [the connected PyTorch lesson](?lesson=raytracing.batched-segments) to implement this reasoning. ARENA 0.1 transfer: `intersect_ray_1d`, `intersect_rays_1d`, `raytrace_triangle`, `raytrace_mesh`.

## Faded practice

### q50012
A matrix has rows [[2, 4], [1, 2]]. Which statement is justified?

### q50013
A ray and a segment overlap on the same line. Their matrix is singular. Under ARENA's singular-pair-as-miss contract, what should the function report?

## Solo practice

### q50014
Invalid means |det(M)| < 10⁻⁸. Which listed pair is valid under that rule?

### q50015
An invalid pair is replaced with the identity. The replacement solution is [u, v] = [2, 0.5]. How should the pair contribute?

## Integrated practice

### q50016
A pair-hit table has rows = rays, columns = segments: [[False, True], [False, False], [True, False]]. Which output says whether each ray hits any segment?

### q50017
Multiply both columns of a 2 × 2 matrix by 100. Its determinant was 10⁻¹⁰. What does this show about a fixed 10⁻⁸ cutoff?
