---
kc: tensor.classifier-evaluation
title: Accuracy and cross-entropy
new_syntax: []
concepts: [accuracy, cross-entropy]
supporting: ['tensor.stable-probabilities', 'tensor.indexed-selection', 'numpy.argmin-argmax', 'numpy.boolean-masking', 'numpy.dtype-astype']
previews: []
faded: [1045, 1046, 1047, 1048]
guided: []
independent: [1049, 1050, 1051, 1052, 1053, 1054, 1083, 1084, 1085]
integrated: [1055, 1056, 1057]
---

## Concept: Predictions and labels meet example by example

A classifier emits one row of scores per example, one score per class, shape `(b, c)`. Its prediction for an example is the class with the largest score in that row, `x.argmax(dim=1)`, which reduces the class axis and returns one index per example, shape `(b,)`. The labels `y` are also `(b,)` integers, so `pred == y` compares each example with its own label and gives a Boolean per example. Accuracy is the fraction of `True`: convert to float and take the mean.

The reason everything is done per example first is that accuracy counts examples, not scores. Averaging the scores, or comparing the whole score matrix with the labels, answers a different question. The reason to convert Booleans to float before `mean` is that a Boolean tensor has no mean; `to(t.float32)` turns `True` into `1.` and `False` into `0.`, so the mean is the fraction correct. When two classes tie for the largest score, `argmax` returns the first, which is a convention to know about rather than a fact about the model.

```python
import torch as t
x=t.tensor([[2.,5.],[4.,1.]])
y=t.tensor([1,1])
pred=x.argmax(dim=1)
print(pred, pred==y)
# Hidden checks
assert pred.tolist()==[1,0] and (pred==y).tolist()==[True,False]
```

## Worked example

Three examples, three classes. We take the prediction per row, compare with the labels, and turn the matches into an accuracy. The middle example is wrong — its largest score is in column 2 but its label is 0.

```python
import torch as t
x=t.tensor([[9.,1.,0.],[1.,2.,7.],[0.,6.,5.]])
y=t.tensor([0,0,1])
hits=x.argmax(dim=1)==y
print(hits)
# Hidden checks
assert hits.tolist()==[True,False,True]
```

Two of three matches is an accuracy of two thirds. The float conversion is what makes `mean` legal on the Boolean result.

```python
accuracy=hits.to(t.float32).mean()
print(accuracy)
# Hidden checks
assert t.allclose(accuracy,t.tensor(2/3))
```

## Faded practice

### q1045
Return predicted class indices, shape (b,); ties choose first. x: logits (b,c), finite float.

```python starter
import torch as t

def solve(x):
    pass
```

```python solution
import torch as t

def solve(x):
    return x.argmax(dim=1)
```

### q1046
Return which examples were classified correctly, shape (b,). x: logits (b,c), finite float; y: true class index per example (b,).

```python starter
import torch as t

def solve(x,y):
    pass
```

```python solution
import torch as t

def solve(x,y):
    return x.argmax(dim=1)==y
```

## Concept: Loss asks how much belief reached the true class

Accuracy is blind to confidence: a prediction that was barely right and one that was certain count the same. Cross-entropy measures the probability the model assigned to the *true* class, on a log scale, negated: `-log p[true]`. A confident correct prediction has `p[true]` near one and loss near zero; a confident wrong one has `p[true]` near zero and a large loss. The computation is log-softmax of each row (score minus the row's log-sum-exp, with the maximum subtracted first for stability), then pick the entry at the true class with paired indexing `l[t.arange(b), y]`, then negate.

The reason to stay in log space is the one from the stable-probabilities lesson: a tiny probability underflows to zero and its log becomes infinite, while the log-probability itself is a finite, ordinary number. The reason to keep one loss per example until the end is that the reduction is a separate decision — the mean over a batch for training, or a sum divided by the total example count when combining batches of unequal size, because averaging per-batch averages would let a small batch count as much as a large one.

```python
import torch as t
x=t.tensor([[0.,0.],[0.,-9.]])
y=t.tensor([0,1])
z=x-x.max(dim=1,keepdim=True)[0]
l=z-z.exp().sum(dim=1,keepdim=True).log()
loss=-l[t.arange(2),y]
print(loss)
# Hidden checks
assert t.allclose(loss,t.nn.functional.cross_entropy(x,y,reduction="none"))
```

## Worked example

We compute the loss for two examples whose true class is `0` in both cases. In the first row class `0` has the higher score, in the second it has the lower, so the second loss must be larger. Start with the log-probabilities.

```python
import torch as t
x=t.tensor([[3.,1.],[1.,3.]])
y=t.tensor([0,0])
z=x-x.max(dim=1,keepdim=True)[0]
l=z-z.exp().sum(dim=1,keepdim=True).log()
print(l)
# Hidden checks
assert t.allclose(l,t.log_softmax(x,dim=1))
```

Paired indexing pulls out each example's own true-class entry; negating turns log-probabilities into losses. The two rows are mirror images, so the two losses are `-log(p)` and `-log(1-p)` for the same `p`.

```python
loss=-l[t.arange(2),y]
print(loss)
# Hidden checks
assert t.allclose(loss,t.tensor([.1269,2.1269]),atol=1e-3)
```

## Faded practice

### q1047
Return accuracy as a scalar tensor. x: logits (b,c), finite float; y: true class index per example (b,).

```python starter
import torch as t

def solve(x,y):
    pass
```

```python solution
import torch as t

def solve(x,y):
    return (x.argmax(dim=1)==y).to(t.float32).mean()
```

### q1048
Return per-example cross entropy, shape (b,), without a built-in loss function. x: logits (b,c), finite float; y: true class index per example (b,).

```python starter
import torch as t

def solve(x,y):
    pass
```

```python solution
import torch as t

def solve(x,y):
    z=x-x.max(dim=1,keepdim=True)[0]
    l=z-z.exp().sum(dim=1,keepdim=True).log()
    loss=-l[t.arange(len(y)),y]
    return loss
```

## Solo practice

### q1049
Return the number of correct predictions, as a scalar tensor. x: logits (b,c), finite float; y: true class index per example (b,).

### q1050
Return the mean negative log probability of the true class, as a scalar tensor, without a built-in loss function. x: logits (b,c), finite float; y: true class index per example (b,).

### q1051
Return log probability of the true class for each example, shape (b,). x: logits (b,c), finite float; y: true class index per example (b,).

### q1052
Return how far each true-class score falls below its row maximum, shape (b,). x: logits (b,c), finite float; y: true class index per example (b,).

### q1053
Return the batch indices of all misclassified examples, as a 1-D integer tensor. x: logits (b,c), finite float; y: true class index per example (b,).

### q1054
Return probabilities assigned to the true classes, shape (b,). x: logits (b,c), finite float; y: true class index per example (b,).

### q1083
Return numbers of examples belonging to each true class, shape (c,). x: logits (b,c), finite float; y: true class index per example (b,).

### q1084
Return summed cross entropy on correctly classified examples, as a scalar tensor; zero if none. x: logits (b,c), finite float; y: true class index per example (b,).

### q1085
Return the cross entropy above the uniform-prediction loss, one value per example, shape (b,). x: logits (b,c), finite float; y: true class index per example (b,).

## Integrated practice

### q1055
Return the summed loss on misclassified examples, as a scalar tensor; return zero when none were missed. x: logits (b,c), finite float; y: true class index per example (b,).

### q1056
Return accuracy weighted by prediction confidence: sum maximum class probabilities on correct predictions divided by their sum on all examples. x: logits (b,c), finite float; y: true class index per example (b,).

### q1057
Return the index of the example with largest cross entropy, as a scalar integer tensor; ties choose first. x: logits (b,c), finite float; y: true class index per example (b,).

## Misconceptions

- **Accuracy is the mean of the scores.** It is the mean of per-example correctness; scores never enter except through `argmax`.
- **A Boolean tensor can be averaged directly.** It must be converted to float first.
- **Cross-entropy is `-log` of the predicted class's probability.** It is `-log` of the *true* class's probability; a wrong prediction is penalized through the belief it failed to give the label.
- **Combine batches by averaging their accuracies.** Sum the correct counts and divide by the total number of examples, or small batches are over-weighted.
