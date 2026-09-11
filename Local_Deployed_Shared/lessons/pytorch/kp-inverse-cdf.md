---
kc: tensor.inverse-cdf
title: Sampling by inverse CDF
new_syntax: ['Tensor.cumsum']
concepts: [interval-partition, boundary-count]
supporting: ['numpy.axis-reductions', 'numpy.broadcasting-rules', 'numpy.random-threading', 'tensor.indexed-selection', 'numpy.argmin-argmax', 'numpy.dtype-astype', 'tensor.stable-probabilities']
previews: []
faded: [1058, 1059, 1060, 1061]
guided: []
independent: [1062, 1063, 1064, 1065, 1066, 1067, 1086, 1087, 1088]
integrated: [1068, 1069, 1070]
---

## Concept: Probabilities partition the unit interval

To sample from a categorical distribution `p` of `k` classes, picture a dart landing uniformly between `0` and `1`, and give each class a slice of that interval whose width is its probability. Class `0` owns `[0, p[0])`, class `1` owns `[p[0], p[0]+p[1])`, and so on; the right endpoint of class `j`'s slice is the sum of the first `j+1` probabilities. `p.cumsum(dim=0)` produces exactly those running totals — entry `j` is `p[0] + … + p[j]` — so the endpoints of every slice come from one call, and the last endpoint is `1` because the probabilities sum to one.

The reason this works is that a uniform dart lands in a slice with probability equal to the slice's width, and the widths were chosen to be the class probabilities. The reason the map from dart to class is kept separate from generating the dart is testability: a random generator cannot be checked against a fixed answer, but "which slice contains `0.3`" can. A zero-probability class gets a slice of width zero — two consecutive endpoints coincide — and must never be selected, even by a dart landing exactly on that shared boundary. The slices include their left endpoint so that `0.0` lands in class `0` and a dart equal to an endpoint moves on to the next class.

```python
import torch as t
p=t.tensor([.2,.5,.3])
ends=p.cumsum(dim=0)
print(ends)
# Hidden checks
assert t.allclose(ends,t.tensor([.2,.7,1.]))
```

## Worked example

We lay out the slices for a distribution in which the first class is impossible. Its right endpoint equals its left one, so the slice has no width; the remaining two classes share the interval `0.4 : 0.6`.

```python
import torch as t
p=t.tensor([0.,.4,.6])
ends=p.cumsum(dim=0)
print(ends)
# Hidden checks
assert t.allclose(ends,t.tensor([0.,.4,1.]))
```

Cumulative sums are order-dependent: the same probabilities in another order give different endpoints, because each class's slice begins where the previous one ended. Predict the endpoints for the reversed vector before running.

```python
print(t.tensor([.6,.4,0.]).cumsum(dim=0))
# Hidden checks
assert t.allclose(t.tensor([.6,.4,0.]).cumsum(dim=0),t.tensor([.6,1.,1.]))
```

## Faded practice

### q1058
Return the cumulative interval right endpoints, shape (k,). p: class probabilities (k,), nonnegative, summing to one.

```python starter
import torch as t

def solve(p):
    pass
```

```python solution
import torch as t

def solve(p):
    return p.cumsum(dim=0)
```

### q1059
Return, for each draw, whether it lies at or beyond each interval's right endpoint, shape (n,k) Boolean. p: class probabilities (k,), nonnegative, summing to one; u: uniform draws (n,) in [0,1). A draw selects the class whose probability interval contains it; intervals include their left endpoint.

```python starter
import torch as t

def solve(p,u):
    pass
```

```python solution
import torch as t

def solve(p,u):
    return u[:,None]>=p.cumsum(dim=0)
```

## Concept: A draw's class is the number of endpoints it has passed

With the endpoints in hand, the class of a draw `u` is the index of the first slice whose right endpoint is strictly greater than `u` — equivalently, the number of endpoints that are less than or equal to `u`. For many draws at once, compare every draw against every endpoint: `u[:, None] >= ends` broadcasts an `(n, 1)` column of draws against the `(k,)` endpoints to make an `(n, k)` Boolean table, and summing each row over `dim=1` counts the endpoints passed, which is the class index.

The reason `>=` and not `>` is the left-endpoint convention: a draw equal to an endpoint has left the slice that ends there, so the endpoint counts as passed. The reason a zero-width slice is never chosen falls out of the same count: its endpoint coincides with the previous one, so any draw that passes one passes both and lands at least one class further on. The comparison-and-count form has no loop and no search; it is the whole inverse CDF in one broadcast.

```python
import torch as t
p=t.tensor([.2,.5,.3]); u=t.tensor([.1,.2,.8])
idx=(u[:,None]>=p.cumsum(dim=0)).sum(dim=1)
print(idx)
# Hidden checks
assert idx.tolist()==[0,1,2]
```

## Worked example

Two draws against a distribution whose middle class has all the mass. Both `0.0` and `0.8` must land on class `1`: the first because class `0` has no width, the second because it is below the endpoint `1.0`. Look at the Boolean table first.

```python
import torch as t
p=t.tensor([0.,1.,0.]); u=t.tensor([0.,.8])
table=u[:,None]>=p.cumsum(dim=0)
print(table)
# Hidden checks
assert table.tolist()==[[True,False,False],[True,False,False]]
```

Each row has exactly one `True` — the zero-width endpoint at `0.0` — so the count is `1` for both draws, and the impossible classes are never chosen.

```python
print(table.sum(dim=1))
# Hidden checks
assert table.sum(dim=1).tolist()==[1,1]
```

## Faded practice

### q1060
Return sampled class indices, shape (n,). p: class probabilities (k,), nonnegative, summing to one; u: uniform draws (n,) in [0,1). A draw selects the class whose probability interval contains it; intervals include their left endpoint.

```python starter
import torch as t

def solve(p,u):
    pass
```

```python solution
import torch as t

def solve(p,u):
    return (u[:,None]>=p.cumsum(dim=0)).sum(dim=1)
```

### q1061
Return the probability of the class selected by each draw, shape (n,). p: class probabilities (k,), nonnegative, summing to one; u: uniform draws (n,) in [0,1). A draw selects the class whose probability interval contains it; intervals include their left endpoint.

```python starter
import torch as t

def solve(p,u):
    pass
```

```python solution
import torch as t

def solve(p,u):
    idx=(u[:,None]>=p.cumsum(dim=0)).sum(dim=1)
    return p[idx]
```

## Solo practice

### q1062
Return the spread of the sampled class indices: largest minus smallest, as a scalar integer tensor. p: class probabilities (k,), nonnegative, summing to one; u: uniform draws (n,) in [0,1). A draw selects the class whose probability interval contains it; intervals include their left endpoint.

### q1063
Return the interval left endpoints, shape (k,). p: class probabilities (k,), nonnegative, summing to one.

### q1064
Return whether each draw selects the most probable class, shape (n,); ties choose first. p: class probabilities (k,), nonnegative, summing to one; u: uniform draws (n,) in [0,1). A draw selects the class whose probability interval contains it; intervals include their left endpoint.

### q1065
Return counts of sampled classes, shape (k,), including classes never sampled. p: class probabilities (k,), nonnegative, summing to one; u: uniform draws (n,) in [0,1). A draw selects the class whose probability interval contains it; intervals include their left endpoint.

### q1066
Return each draw’s position within its selected interval, expressed as a fraction from zero to one, shape (n,). p: class probabilities (k,), nonnegative, summing to one; u: uniform draws (n,) in [0,1). A draw selects the class whose probability interval contains it; intervals include their left endpoint.

### q1067
Return the sampled frequencies minus the requested probabilities, shape (k,). p: class probabilities (k,), nonnegative, summing to one; u: uniform draws (n,) in [0,1). A draw selects the class whose probability interval contains it; intervals include their left endpoint.

### q1086
Return midpoint of the interval containing each draw, shape (n,). p: class probabilities (k,), nonnegative, summing to one; u: uniform draws (n,) in [0,1). A draw selects the class whose probability interval contains it; intervals include their left endpoint.

### q1087
Return negative log probability of each sampled class, shape (n,). p: class probabilities (k,), nonnegative, summing to one; u: uniform draws (n,) in [0,1). A draw selects the class whose probability interval contains it; intervals include their left endpoint.

### q1088
Return empirical frequency of sampling a class strictly above the distribution’s expected class index, as a scalar tensor. p: class probabilities (k,), nonnegative, summing to one; u: uniform draws (n,) in [0,1). A draw selects the class whose probability interval contains it; intervals include their left endpoint.

## Integrated practice

### q1068
Return counts of consecutive sampled-class transitions, shape (k,k); row is previous class, column is next. p: class probabilities (k,), nonnegative, summing to one; u: uniform draws (n,) in [0,1). A draw selects the class whose probability interval contains it; intervals include their left endpoint.

### q1069
Return total probability mass of classes sampled at least once, as a scalar tensor. p: class probabilities (k,), nonnegative, summing to one; u: uniform draws (n,) in [0,1). A draw selects the class whose probability interval contains it; intervals include their left endpoint.

### q1070
Return expected class index under the empirical sample distribution minus its expectation under p, as a scalar tensor. p: class probabilities (k,), nonnegative, summing to one; u: uniform draws (n,) in [0,1). A draw selects the class whose probability interval contains it; intervals include their left endpoint.

## Misconceptions

- **The class is the endpoint nearest to the draw.** It is the count of endpoints at or below the draw; nearness would pick the wrong side of a boundary.
- **A zero-probability class can be hit by a boundary draw.** Its endpoint coincides with the previous one, so the count skips it.
- **`u >= ends` compares each draw with its own endpoint.** It needs `u[:, None]` to compare every draw with every endpoint; without the new axis the shapes do not line up.
- **The cumulative sum is order-free.** Reordering the probabilities moves every slice; the endpoints are running totals, not sorted values.
