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

This lesson builds on determinants and invertibility, including the difference between no solutions and infinitely many. Here we apply that distinction to intersection handling.

For a $2\times2$ matrix,
$$\det\begin{pmatrix}a&b\\c&d\end{pmatrix}=ad-bc.$$
Its absolute value measures the area scaling of the column directions. A nonzero determinant means the columns are independent, so every right-hand side has one solution.

A zero determinant means the columns collapse onto a line or point. The system can have no solution or infinitely many, depending on $\mathbf b$. It does not mean geometric objects never overlap. ARENA's simplified contract treats singular pairs as misses, including coincident lines and degenerate objects.

The batched implementation declares a pair invalid when
$$|\det M|<10^{-8}.$$
This is a course convention, not a scale-invariant accuracy test: rescaling columns rescales the determinant. Substitute an identity matrix for invalid pairs before solving so one singular pair cannot abort the batch. Keep the original validity mask: a solution of the replacement matrix is not evidence of a hit.

For each ray, combine validity with membership for every segment pair, then reduce over segments using “any.” Reducing over rays answers a different question.

The determinant of rows $(1,2)$ and $(3,6)$:

```python
import torch as t

matrix = t.tensor([[1.0, 2.0],
                   [3.0, 6.0]])

determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
print(determinant)
# Hidden checks
assert determinant.item() == 0.0
assert _delta_output == 'tensor(0.)\n'
```

## Worked example

First, one parallel pair on paper; then a batch of pairs.

**On paper.** The ray from $O=(0,0)$ with $D=(1,1)$ against the segment from $A=(2,0)$ to $B=(4,2)$:
$$M=\begin{pmatrix}D & A-B\end{pmatrix}=\begin{pmatrix}1&-2\\1&-2\end{pmatrix},\qquad \det M=1\cdot(-2)-(-2)\cdot1=0.$$
Singular: the ray and the segment are parallel. Writing out the rows with $\mathbf b=A-O=(2,0)$ shows why there is no crossing:
$$\begin{aligned}u-2v&=2\\u-2v&=0\end{aligned}$$
One quantity cannot equal both $2$ and $0$. The pair is marked invalid, so it is a miss.

Now two rays, three segments each. Rows are rays, columns are segments; a hit needs validity AND membership:
$$\text{valid}=\begin{pmatrix}0&1&1\\1&1&0\end{pmatrix},\quad \text{inside}=\begin{pmatrix}1&0&0\\0&1&1\end{pmatrix},\quad \text{valid}\wedge\text{inside}=\begin{pmatrix}0&0&0\\0&1&0\end{pmatrix}.$$
Reduce each row with "any": the first ray misses (its only in-bounds pair came from a replaced matrix), the second hits.

**In code.** `&` combines the masks; `.any(dim=1)` reduces over segments, one answer per ray.

```python
import torch as t

valid = t.tensor([[False, True, True],
                  [True, True, False]])
within_bounds = t.tensor([[True, False, False],
                          [False, True, True]])

hit_pairs = valid & within_bounds
hits = hit_pairs.any(dim=1)
print(hit_pairs)
print(hits)
# Hidden checks
assert hits.tolist() == [False, True]
assert _delta_output == 'tensor([[False, False, False],\n        [False,  True, False]])\ntensor([False,  True])\n'
```

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
