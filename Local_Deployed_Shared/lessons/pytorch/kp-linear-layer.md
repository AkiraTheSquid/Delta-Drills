---
kc: cnn.linear-layer
title: The linear layer
new_syntax: []
concepts: [weight-rows, leading-axes]
supporting: ['cnn.module-state', 'numpy.dot-matmul-patterns', 'numpy.transpose-axes', 'numpy.broadcasting-rules', 'numpy.axis-reductions', 'numpy.argmin-argmax', 'numpy.reshape-flatten']
previews: []
faded: [1185, 1186, 1187, 1188]
guided: []
independent: [1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197]
integrated: [1198, 1199, 1200]
---

## Concept: Each output owns a weight row

A linear layer turns `in_features` numbers into `out_features` numbers. It stores one weight row per output, so the weight has shape `(out_features, in_features)`, and output `j` is the dot product of row `j` with the input, plus a bias `b[j]`. For a batch `x` of shape `(batch, in_features)` the whole thing is `x @ w.T + b`: transposing makes the shapes `(batch, in) @ (in, out)`, so the feature axes meet and the result is `(batch, out)`.

The reason the weight is stored as `(out, in)` rather than `(in, out)` is convention — PyTorch's `nn.Linear` does it, and every checkpoint you load will — and the transpose in the forward is the price. The reason the bias is `(out_features,)` and simply added is broadcasting: it lines up with the last axis of the product, so each output column gets its own offset on every example. Test shapes with a non-square weight; a square one lets an accidental missing transpose run silently with the wrong numbers.

```python
import torch as t
x=t.tensor([[1.,2.]])
w=t.tensor([[2.,0.],[0.,3.],[1.,-1.]])
b=t.tensor([1.,-1.,0.])
y=x@w.T+b
print(y)
# Hidden checks
assert y.tolist()==[[3.,5.,-1.]]
```

## Worked example

We run a layer with three inputs and two outputs on a batch of two examples. Row `0` of the weight sums the inputs, row `1` takes the first minus the last, and the bias shifts each output. Predict the `(2, 2)` result before running.

```python
import torch as t
x=t.tensor([[1.,1.,1.],[2.,0.,-2.]])
w=t.tensor([[1.,1.,1.],[1.,0.,-1.]])
b=t.tensor([0.,10.])
y=x@w.T+b
print(y)
# Hidden checks
assert y.tolist()==[[3.,10.],[0.,14.]]
```

Drop the transpose and the shapes do not meet: `(2, 3) @ (2, 3)` has no shared inner axis. Catching this at the shape level is the point of a non-square weight.

```python
try:
    x@w
except RuntimeError as e:
    print("shape mismatch:", type(e).__name__)
# Hidden checks
assert (x@w.T).shape==(2,2)
```

## Faded practice

### q1185
Return the affine output, shape (...,out_features). x: float (...,in_features); w: float (out_features,in_features); b: float (out_features,).

```python starter
import torch as t

def solve(x,w,b):
    pass
```

```python solution
import torch as t

def solve(x,w,b):
    y=x@w.T+b
    return y
```

### q1186
Return the affine output with zero bias, shape (...,out_features). x: float (...,in_features); w: float (out_features,in_features).

```python starter
import torch as t

def solve(x,w):
    pass
```

```python solution
import torch as t

def solve(x,w):
    return x@w.T
```

## Concept: Leading axes survive; only the last axis is transformed

The layer treats the last axis as features and every axis before it as "which example". A `(batch, seq, in_features)` input — a batch of token sequences, say — goes through the same `x @ w.T + b` and comes out `(batch, seq, out_features)`: the matrix product only touches the last axis of `x`, and the bias broadcasts along it. The general rule is that a linear layer never needs to know how many leading axes there are.

The reason this matters is that it is the same weights for every example and every position. Batching does not make a weight per example; every row of every leading axis is evidence about one shared map, which is what makes the parameters learnable from many examples at once. The weight is the layer's state, created once and reused on every call; `nn.Linear` registers it as a parameter and, when `bias=False`, stores no bias at all rather than a zero one — absent state, not a broadcast constant.

```python
import torch as t
x=t.ones(2,3,4); w=t.ones(5,4)
print((x@w.T).shape)
# Hidden checks
assert (x@w.T).shape==(2,3,5)
```

## Worked example

We push a single vector and a batch of vectors through the same layer and check that the batch result is just the vector results stacked. First the batch: two examples, two inputs, one output.

```python
import torch as t
x=t.tensor([[1.,0.],[0.,1.]])
w=t.tensor([[2.,3.]]); b=t.tensor([4.])
print(x@w.T+b)
# Hidden checks
assert (x@w.T+b).tolist()==[[6.],[7.]]
```

Now one example on its own, as a plain `(2,)` vector. The product is `(1,)` — the leading batch axis was never required — and it equals the first row of the batched answer.

```python
print(x[0]@w.T+b)
# Hidden checks
assert (x[0]@w.T+b).tolist()==[6.]
```

## Faded practice

### q1187
Return positive parts of affine outputs, shape (...,out_features). x: float (...,in_features); w: float (out_features,in_features); b: float (out_features,).

```python starter
import torch as t

def solve(x,w,b):
    pass
```

```python solution
import torch as t

def solve(x,w,b):
    y=x@w.T+b
    return y.clamp(min=0)
```

### q1188
Return the affine output averaged across output features, shape (...). x: float (...,in_features); w: float (out_features,in_features); b: float (out_features,).

```python starter
import torch as t

def solve(x,w,b):
    pass
```

```python solution
import torch as t

def solve(x,w,b):
    y=x@w.T+b
    return y.mean(dim=-1)
```

## Solo practice

### q1189
Return the affine output of x after centring each example: subtract every example's own mean over in_features first, shape (...,out_features). x: float (...,in_features); w: float (out_features,in_features); b: float (out_features,).

### q1190
Return the affine outputs centred to zero mean within each example, shape (...,out_features). x: float (...,in_features); w: float (out_features,in_features); b: float (out_features,).

### q1191
Return index of the largest affine output for every example, shape (...); ties choose first. x: float (...,in_features); w: float (out_features,in_features); b: float (out_features,).

### q1192
Return the affine output with all leading example axes combined, shape (number_of_examples,out_features). x: float (...,in_features); w: float (out_features,in_features); b: float (out_features,).

### q1193
Return the mean affine output across all examples and positions, shape (out_features,). x: float (...,in_features); w: float (out_features,in_features); b: float (out_features,).

### q1194
Return the tied-weight autoencoder output, shape (...,in_features). x: float (...,in_features); w: float (out_features,in_features); b: float (out_features,).

### q1195
Return total squared affine activation per example, shape (...). x: float (...,in_features); w: float (out_features,in_features); b: float (out_features,).

### q1196
Return the residual tied-weight autoencoder output: reconstruction plus original input, shape (...,in_features). x: float (...,in_features); w: float (out_features,in_features); b: float (out_features,).

### q1197
Return the difference between affine outputs on x and on zero input, shape (...,out_features). x: float (...,in_features); w: float (out_features,in_features); b: float (out_features,).

## Integrated practice

### q1198
Return a new learned affine module initialized from float weight w (out,in) and bias b (out,). Its forward accepts (...,in) and returns (...,out); register parameters as weight and bias.

### q1199
Return an affine-then-ReLU module using supplied float weight w (out,in), bias b (out,). Register weight and bias; preserve every leading input axis.

### q1200
Return a learned affine module without bias: register weight from w (out,in) as its only parameter; forward maps (...,in) to (...,out).

## Misconceptions

- **The weight is `(in, out)` so that `x @ w` works.** PyTorch stores `(out, in)`; the transpose belongs in the forward.
- **The bias has one entry per example.** It has one per output feature and broadcasts over every leading axis.
- **A batch needs a loop over examples.** One matrix product handles every example and every leading position with the same weights.
- **`bias=False` means a bias of zeros.** It means no bias tensor exists; the state is absent, not constant.
