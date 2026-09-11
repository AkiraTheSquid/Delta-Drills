---
kc: tensor.stable-probabilities
title: Stable softmax and log-softmax
new_syntax: ['Tensor.exp', 'Tensor.log']
concepts: [shift-invariance, log-space]
supporting: ['numpy.axis-reductions', 'numpy.elementwise-ufuncs', 'numpy.broadcasting-rules', 'numpy.ranges']
previews: []
faded: [1032, 1033, 1034, 1035]
guided: []
independent: [1036, 1037, 1038, 1039, 1040, 1041, 1080, 1081, 1082]
integrated: [1042, 1043, 1044]
---

## Concept: Only score differences matter

A classifier's raw outputs are logits: real-valued scores, one per class, that can be negative or huge. Softmax turns a row of scores into probabilities in two steps: exponentiate every score, so all are positive, then divide each by the row's total, so they sum to one. `x.exp()` applies e-to-the-power elementwise; the row sum is `sum(dim=1, keepdim=True)` so it broadcasts back against the row.

The procedure has a third step that the definition does not mention but every implementation needs: subtract the row's maximum from every score first. The reason is overflow. `exp(1000)` is larger than a float can hold and becomes infinity, and infinity divided by infinity is not a number. But softmax is shift-invariant — adding the same constant to every score in a row multiplies every exponential by the same factor, which cancels in the ratio — so subtracting the maximum changes nothing mathematically while guaranteeing that the largest exponential is `exp(0) = 1` and every other is between `0` and `1`. Scores are then read as "how far below the best", which is the only thing softmax ever depended on.

```python
import torch as t
x=t.tensor([[1.,2.,3.]])
z=x-x.max(dim=1,keepdim=True)[0]
e=z.exp()
p=e/e.sum(dim=1,keepdim=True)
print(p)
# Hidden checks
assert t.allclose(p,t.softmax(x,dim=1))
```

## Worked example

We give two rows of scores: one moderate, one enormous. Without the shift the second row would overflow; with it, both rows are safe and the second comes out as an even split, because its scores are equal.

```python
import torch as t
x=t.tensor([[2.,0.],[5000.,5000.]])
z=x-x.max(dim=1,keepdim=True)[0]
print(z)
# Hidden checks
assert z.tolist()==[[0.,-2.],[0.,0.]]
```

After the shift the largest entry of each row is zero, so `exp` gives `1` there and less elsewhere. Normalizing each row by its own sum produces the probabilities; predict the second row before running.

```python
e=z.exp()
p=e/e.sum(dim=1,keepdim=True)
print(p)
# Hidden checks
assert t.allclose(p,t.softmax(x,dim=1)) and p[1].tolist()==[.5,.5]
```

## Faded practice

### q1032
Return row scores relative to each row’s maximum, shape (b,c). x: logits (b,c), finite float.

```python starter
import torch as t

def solve(x):
    pass
```

```python solution
import torch as t

def solve(x):
    return x-x.max(dim=1,keepdim=True)[0]
```

### q1033
Return positive unnormalized weights after removing the largest row score, shape (b,c). x: logits (b,c), finite float.

```python starter
import torch as t

def solve(x):
    pass
```

```python solution
import torch as t

def solve(x):
    z=x-x.max(dim=1,keepdim=True)[0]
    return z.exp()
```

## Concept: Keep tiny probabilities in log space

A probability of `exp(-1000)` underflows to exactly zero in floating point, and `log(0)` is minus infinity — so "take softmax, then log" destroys the very numbers a loss function needs. The remedy is to never form the probability: compute the log-probability directly as score minus log-normalizer, where the log-normalizer of a row is `log(sum(exp(x)))`, the log-sum-exp. `x.log()` is the natural logarithm, elementwise.

The log-sum-exp itself needs the same shift trick: with `m` the row maximum, `log(sum(exp(x))) = m + log(sum(exp(x - m)))`. The reason this is exact is that factoring `exp(m)` out of the sum and taking its log gives back `m`. The reason it is stable is that the sum inside now has a largest term of `1`, so it can neither overflow nor become zero, and its log is finite. Log-softmax is then `x - logsumexp(x)`, and a class whose score is a thousand below the best gets log-probability about `-1000` — a finite, usable number, where the probability route would have given `log(0)`.

```python
import torch as t
x=t.tensor([[0.,-1000.]])
m=x.max(dim=1,keepdim=True)[0]
logp=x-m-(x-m).exp().sum(dim=1,keepdim=True).log()
print(logp)
# Hidden checks
assert logp.tolist()==[[0.,-1000.]]
```

## Worked example

We compute the log-normalizer of two rows the stable way and check it against the library. The first row has ordinary scores; the second has scores near `900`, where `exp` alone would overflow.

```python
import torch as t
x=t.tensor([[1.,2.],[900.,901.]])
m=x.max(dim=1)[0]
inner=(x-m[:,None]).exp().sum(dim=1)
print(inner)
# Hidden checks
assert t.allclose(inner,t.tensor([1.3679,1.3679]),atol=1e-3)
```

The inner sums are identical because both rows have the same *differences* — `0` and `-1` — and only differences survive the shift. Adding the maximum back gives each row its own log-normalizer.

```python
lse=m+inner.log()
print(lse)
# Hidden checks
assert t.allclose(lse,t.logsumexp(x,dim=1))
```

## Faded practice

### q1034
Return normalized class probabilities, shape (b,c). x: logits (b,c), finite float.

```python starter
import torch as t

def solve(x):
    pass
```

```python solution
import torch as t

def solve(x):
    z=x-x.max(dim=1,keepdim=True)[0]
    e=z.exp()
    p=e/e.sum(dim=1,keepdim=True)
    return p
```

### q1035
Return row log-normalizers, shape (b,). x: logits (b,c), finite float.

```python starter
import torch as t

def solve(x):
    pass
```

```python solution
import torch as t

def solve(x):
    m=x.max(dim=1)[0]
    return m+(x-m[:,None]).exp().sum(dim=1).log()
```

## Solo practice

### q1036
Return the ratio of each row's largest class probability to its smallest, shape (b,). x: logits (b,c), finite float.

### q1037
Return log probabilities without built-in log-normalization functions, shape (b,c). x: logits (b,c), finite float.

### q1038
Return each row’s largest class probability, shape (b,). x: logits (b,c), finite float.

### q1039
Return the probability assigned to the first class, shape (b,). x: logits (b,c), finite float.

### q1040
Return the entropy of each row’s distribution, shape (b,); entropy is minus the sum of probability times log probability. x: logits (b,c), finite float.

### q1041
Return the log of the mean exponential score per row, shape (b,). x: logits (b,c), finite float.

### q1080
Return the expected zero-based class index for each distribution, shape (b,). x: logits (b,c), finite float.

### q1081
Return squared distance of each class distribution from uniform, shape (b,). x: logits (b,c), finite float.

### q1082
Return effective number of equally likely classes, defined as the exponential of entropy, shape (b,). x: logits (b,c), finite float.

## Integrated practice

### q1042
Return the class probabilities after halving every score difference, shape (b,c). x: logits (b,c), finite float.

### q1043
Return the mean class distribution across the batch, shape (c,). x: logits (b,c), finite float.

### q1044
Return how much probability each row assigns to scores strictly below that row’s average, shape (b,). x: logits (b,c), finite float.

## Misconceptions

- **Subtracting the maximum changes the probabilities.** It multiplies numerator and denominator by the same factor; the ratio is unchanged.
- **Log-softmax is `log(softmax(x))`.** Mathematically yes, numerically no: the probability underflows to zero first. Compute score minus log-sum-exp instead.
- **The row maximum must be squeezed before subtracting.** `max(dim=1, keepdim=True)[0]` is `(b, 1)` and broadcasts across the row; the squeezed `(b,)` form does not line up.
- **`exp` of a large score is just a large number.** Past about `88` for float32 it is infinity, and every later step is `nan`.
