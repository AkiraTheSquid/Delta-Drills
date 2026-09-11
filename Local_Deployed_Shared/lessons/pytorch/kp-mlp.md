---
kc: cnn.mlp
title: A two-layer MLP for images
new_syntax: []
concepts: [flatten-per-image, hidden-nonlinearity]
supporting: ['cnn.linear-layer', 'numpy.reshape-flatten', 'tensor.classifier-evaluation']
previews: []
faded: [1201, 1202, 1203, 1204]
guided: []
independent: [1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213]
integrated: [1214, 1215, 1216]
---

## Concept: Flatten each image, not the batch

An MLP reads a vector of features per example, but an image arrives as `(channels, height, width)`. The first step is to flatten each image into one vector while keeping examples apart: `x.reshape(x.shape[0], -1)` keeps the batch axis and merges everything after it into a single feature axis of length `channels * height * width`. The `-1` asks reshape to work out that length. The first weight matrix then has shape `(hidden, pixels)`, with `pixels` equal to the number of values in one image, and the hidden pre-activations are `flat @ w1.T + b1`, shape `(batch, hidden)`.

The reason to keep axis `0` is that the batch is not part of any image; flattening it too would glue neighbouring examples into one long vector and make the layer's input width depend on the batch size. The reason the flattening order does not matter for correctness is that the first layer learns a weight per feature position — whatever order the pixels arrive in, it is the same order every time.

```python
import torch as t
x=t.arange(12.).reshape(2,2,3)
flat=x.reshape(x.shape[0],-1)
print(flat)
# Hidden checks
assert flat.tolist()==[[0.,1.,2.,3.,4.,5.],[6.,7.,8.,9.,10.,11.]]
```

## Worked example

We flatten a batch of three one-channel `2×2` images and feed them to a first layer with five hidden units. Each image has four values, so `pixels = 4` and the weight is `(5, 4)`. Predict the flattened shape before running.

```python
import torch as t
x=t.ones(3,1,2,2); w1=t.ones(5,4)
flat=x.reshape(3,-1)
print(flat.shape)
# Hidden checks
assert flat.shape==(3,4)
```

The product keeps the batch of three and produces five hidden values per image. Every input was `1` and every weight was `1`, so each hidden value is the sum of four ones.

```python
h=flat@w1.T
print(h.shape, h[0])
# Hidden checks
assert h.shape==(3,5) and h[0].tolist()==[4.,4.,4.,4.,4.]
```

## Faded practice

### q1201
Return image features, shape (b,pixels). x: images (b,...), float.

```python starter
import torch as t

def solve(x):
    pass
```

```python solution
import torch as t

def solve(x):
    return x.reshape(x.shape[0],-1)
```

### q1202
Return hidden pre-activation values, shape (b,hidden). x: images (b,...), float; w1: (hidden,pixels); b1: (hidden,). pixels is the number of values in one image.

```python starter
import torch as t

def solve(x,w1,b1):
    pass
```

```python solution
import torch as t

def solve(x,w1,b1):
    return x.reshape(x.shape[0],-1)@w1.T+b1
```

## Concept: A nonlinearity between the layers makes a new representation

Two linear layers in a row, `w2 @ (w1 @ x)`, are together just one linear map, `(w2 @ w1) @ x`; stacking them adds parameters but no expressive power. An activation between them changes that. ReLU, `clamp(min=0)`, zeroes every negative hidden value, so different inputs switch different hidden units on and off and the second layer sees a different linear map in different regions of input space. The full forward is: flatten, first affine, ReLU, second affine.

The reason the last layer has no activation is that its outputs are logits — unnormalized class scores that the loss function consumes directly. A ReLU there would forbid negative scores and a softmax there would double-normalize once the loss applied its own. So the hidden layer gets the nonlinearity and the output layer stays affine. Inspect the hidden activations when debugging: a unit that is zero for every example is a dead unit, and the shape `(batch, hidden)` tells you at once whether the flattening went right.

```python
import torch as t
x=t.tensor([[-1.,2.],[3.,-2.]])
w=t.tensor([[1.,1.],[-1.,1.]])
h=(x@w.T).clamp(min=0)
print(h)
# Hidden checks
assert h.tolist()==[[1.,3.],[1.,0.]]
```

## Worked example

We finish the network on those hidden activations: a second layer with two outputs turns `(batch, hidden)` into `(batch, classes)`. Note that the zeroed hidden unit in the second row contributes nothing, whichever weight multiplies it.

```python
import torch as t
h=t.tensor([[1.,3.],[1.,0.]])
w2=t.tensor([[1.,-1.],[2.,1.]])
logits=h@w2.T
print(logits)
# Hidden checks
assert logits.tolist()==[[-2.,5.],[1.,2.]]
```

Now the same head applied to the hidden values *before* ReLU, to see what the nonlinearity changed. The second row differs, because its negative hidden unit now leaks through.

```python
raw=t.tensor([[1.,3.],[1.,-5.]])
print(raw@w2.T)
# Hidden checks
assert (raw@w2.T).tolist()==[[-2.,5.],[6.,-3.]]
```

## Faded practice

### q1203
Return hidden activations after the positive-part nonlinearity, shape (b,hidden). x: images (b,...), float; w1: (hidden,pixels); b1: (hidden,). pixels is the number of values in one image.

```python starter
import torch as t

def solve(x,w1,b1):
    pass
```

```python solution
import torch as t

def solve(x,w1,b1):
    flat=x.reshape(x.shape[0],-1)
    h=(flat@w1.T+b1).clamp(min=0)
    return h
```

### q1204
Return classifier logits, shape (b,classes). x: images (b,...), float; w1: (hidden,pixels); b1: (hidden,); w2: (classes,hidden); b2: (classes,). pixels is the number of values in one image.

```python starter
import torch as t

def solve(x,w1,b1,w2,b2):
    pass
```

```python solution
import torch as t

def solve(x,w1,b1,w2,b2):
    flat=x.reshape(x.shape[0],-1)
    h=(flat@w1.T+b1).clamp(min=0)
    y=h@w2.T+b2
    return y
```

## Solo practice

### q1205
Return the logits of the two-layer image classifier averaged over the batch, shape (classes,). x: images (b,...), float; w1: (hidden,pixels); b1: (hidden,); w2: (classes,hidden); b2: (classes,). pixels is the number of values in one image.

### q1206
Return predicted class indices, shape (b,); ties choose first. x: images (b,...), float; w1: (hidden,pixels); b1: (hidden,); w2: (classes,hidden); b2: (classes,). pixels is the number of values in one image.

### q1207
Return mean hidden activation over the batch, shape (hidden,). x: images (b,...), float; w1: (hidden,pixels); b1: (hidden,). pixels is the number of values in one image.

### q1208
Return number of active hidden features per example, shape (b,); positive values are active. x: images (b,...), float; w1: (hidden,pixels); b1: (hidden,). pixels is the number of values in one image.

### q1209
Return logits with the first hidden feature silenced, shape (b,classes). x: images (b,...), float; w1: (hidden,pixels); b1: (hidden,); w2: (classes,hidden); b2: (classes,). pixels is the number of values in one image.

### q1210
Return class probabilities, shape (b,classes). x: images (b,...), float; w1: (hidden,pixels); b1: (hidden,); w2: (classes,hidden); b2: (classes,). pixels is the number of values in one image.

### q1211
Return the change in logits caused by removing the hidden nonlinearity, shape (b,classes): linear-only minus normal model. x: images (b,...), float; w1: (hidden,pixels); b1: (hidden,); w2: (classes,hidden); b2: (classes,). pixels is the number of values in one image.

### q1212
Return the first hidden feature’s contribution to every logit, shape (b,classes), excluding output bias. x: images (b,...), float; w1: (hidden,pixels); b1: (hidden,); w2: (classes,hidden). pixels is the number of values in one image.

### q1213
Return logits relative to each example’s first class score, shape (b,classes). x: images (b,...), float; w1: (hidden,pixels); b1: (hidden,); w2: (classes,hidden); b2: (classes,). pixels is the number of values in one image.

## Integrated practice

### q1214
Build a learned two-layer image classifier from w1 (hidden,pixels), b1 (hidden,), w2 (classes,hidden), b2 (classes,). Return a module mapping images (b,...) to logits (b,classes), with ReLU only between layers. Register parameters as w1,b1,w2,b2.

### q1215
Return a learned image classifier with two affine layers and ReLU between them, initialized from w1 (hidden,pixels), b1 (hidden,), w2 (classes,hidden), b2 (classes,). Images have shape (batch,...); when pixels equals classes, logits also include the original image values in pixel order. Register w1,b1,w2,b2.

### q1216
Build the two-layer image classifier as a module with two child modules named first and second, each registering its own weight and bias (from w1,b1 and w2,b2); forward flattens images (b,...), applies first, ReLU, then second, returning logits (b,classes). w1: (hidden,pixels); b1: (hidden,); w2: (classes,hidden); b2: (classes,). pixels is the number of values in one image.

## Misconceptions

- **`x.reshape(-1)` flattens the images.** It flattens the whole batch into one vector; the batch axis must be kept.
- **Two linear layers learn more than one.** Without a nonlinearity between them they compose into a single linear map.
- **The output layer needs ReLU or softmax.** Logits are consumed raw by the loss; an activation there changes the model.
- **`pixels` is `height * width`.** It is every value in one image, channels included.
