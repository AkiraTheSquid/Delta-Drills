---
kc: cnn.batch-normalization
title: Batch normalization
new_syntax: ['syntax.with', 'torch.no_grad', 'Tensor.training']
concepts: [channel-statistics, running-statistics, train-eval]
supporting: ['cnn.module-state', 'numpy.axis-reductions', 'numpy.elementwise-ufuncs', 'python.control-flow', 'numpy.stack-concat-interleave']
previews: []
faded: [1297, 1298, 1299, 1300]
guided: []
independent: [1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309]
integrated: [1310, 1311, 1312]
---

## Concept: One distribution per channel

BatchNorm treats each channel as its own distribution. For a batch `x` of shape `(batch, channels, height, width)`, the statistics of channel `c` are taken over every example and every pixel — axes `0`, `2` and `3` — while channels are never mixed. The procedure is: compute the per-channel mean, subtract it, compute the per-channel variance of the centred values, divide by `sqrt(var + eps)`, then apply a learned scale `g` and shift `b` per channel. Every per-channel vector has shape `(channels,)` and must be broadcast back as `[None, :, None, None]` to line up with the channel axis.

The reason the reduction spans batch AND pixels is that a channel is one feature detector; its output at any position on any image is a sample from the same distribution. The variance is the *population* variance — the plain mean of squared deviations, with no `n-1` correction — because the batch is not a sample of some larger truth here; it is the thing being normalized. `eps` sits inside the square root so that a constant channel, whose variance is exactly zero, divides by `sqrt(eps)` rather than by zero and comes out as all zeros, which then equals the shift `b`.

```python
import torch as t
x=t.tensor([[[[1.,3.]],[[2.,6.]]]])
mean=x.mean(dim=(0,2,3),keepdim=True)
var=((x-mean)**2).mean(dim=(0,2,3),keepdim=True)
z=(x-mean)/t.sqrt(var+1e-5)
print(z)
# Hidden checks
assert t.allclose(z,t.tensor([[[[-1.,1.]],[[-1.,1.]]]]),atol=1e-3)
```

## Worked example

We normalize a batch of two images with one channel and two pixels each, so the channel's four values are `5, 7, 1, 3`. Their mean is `4`, and the squared deviations are `1, 9, 9, 1`, so the population variance is `5`. Predict the sign of each normalized value before running.

```python
import torch as t
x=t.tensor([[[[5.,7.]]],[[[1.,3.]]]])
mean=x.mean(dim=(0,2,3))
var=((x-mean[None,:,None,None])**2).mean(dim=(0,2,3))
print(mean, var)
# Hidden checks
assert mean.tolist()==[4.] and var.tolist()==[5.]
```

Dividing by `sqrt(var + eps)` leaves values roughly between `-1.4` and `1.4`; the learned scale and shift then move them wherever the next layer wants. With `g = 2` and `b = 1` the pixel that was at the mean becomes exactly `1`.

```python
g=t.tensor([2.]); b=t.tensor([1.]); eps=1e-5
z=(x-mean[None,:,None,None])/t.sqrt(var[None,:,None,None]+eps)
y=z*g[None,:,None,None]+b[None,:,None,None]
print(y)
# Hidden checks
assert t.allclose(y[0,0,0],t.tensor([1.894,3.683]),atol=1e-2)
```

## Faded practice

### q1297
Return per-channel batch means, shape (channels,). x: float (batch,channels,height,width).

```python starter
import torch as t

def solve(x):
    pass
```

```python solution
import torch as t

def solve(x):
    mean=x.mean(dim=(0,2,3))
    return mean
```

### q1298
Return training-mode normalized and affine-transformed x. x: float (batch,channels,height,width); g: scale (channels,); b: shift (channels,); eps > 0. Variance is the population variance.

```python starter
import torch as t

def solve(x,g,b,eps):
    pass
```

```python solution
import torch as t

def solve(x,g,b,eps):
    mean=x.mean(dim=(0,2,3))
    var=((x-mean[None,:,None,None])**2).mean(dim=(0,2,3))
    z=(x-mean[None,:,None,None])/t.sqrt(var[None,:,None,None]+eps)
    y=z*g[None,:,None,None]+b[None,:,None,None]
    return y
```

## Concept: Running statistics remember past batches

At evaluation time there may be no batch to take statistics from — a single image, say — so BatchNorm keeps a *running* mean and variance per channel and updates them during training. The update is an exponential moving average: `running = (1 - momentum) * running + momentum * batch_stat`. With `momentum = 0.1` each new batch nudges the estimate a tenth of the way toward itself; with `momentum = 1` the estimate is simply the last batch.

The update is bookkeeping, not computation the network learns from, so it is done in place on the buffer and wrapped in `with t.no_grad():`. `with` opens a block in which a context is active and closes it when the block ends; `t.no_grad()` is the context in which autograd stops recording. The reason it is needed is that the batch mean was computed from `x`, which may carry a gradient graph; writing it into the buffer without `no_grad` would splice that graph into the buffer's history, and the next training step would try to backpropagate into the previous batch. Wrapping only the update, not the normalization, keeps the gradient path through the output intact.

```python
import torch as t
running=t.tensor([2.,6.])
batch_mean=t.tensor([4.,2.])
momentum=.25
with t.no_grad():
    running.copy_((1-momentum)*running+momentum*batch_mean)
print(running)
# Hidden checks
assert running.tolist()==[2.5,5.]
```

## Worked example

We watch the running mean converge toward a repeated batch. The buffer starts at `0`, the batch mean is `8`, and `momentum = 0.5` moves halfway each step. After one update the estimate is `4`; predict the second.

```python
import torch as t
running=t.tensor([0.])
batch_mean=t.tensor([8.])
momentum=.5
with t.no_grad():
    running.copy_((1-momentum)*running+momentum*batch_mean)
print(running)
# Hidden checks
assert running.tolist()==[4.]
```

The second step starts from the updated value, not the original: `(1-0.5)*4 + 0.5*8 = 6`. Each application closes half the remaining gap, which is why the estimate never overshoots and never quite arrives.

```python
with t.no_grad():
    running.copy_((1-momentum)*running+momentum*batch_mean)
print(running)
# Hidden checks
assert running.tolist()==[6.]
```

## Faded practice

### q1299
Return the updated running mean after this batch. x: float (batch,channels,height,width); rm: running mean (channels,); momentum in (0,1]. Variance is the population variance.

```python starter
import torch as t

def solve(x,rm,momentum):
    pass
```

```python solution
import torch as t

def solve(x,rm,momentum):
    return (1-momentum)*rm+momentum*x.mean(dim=(0,2,3))
```

## Concept: Training uses the batch; evaluation uses the memory

A module carries a flag, `self.training`, that `m.train()` sets to `True` and `m.eval()` sets to `False`. BatchNorm's forward branches on it: in training mode it normalizes with the current batch's statistics and updates the running buffers; in evaluation mode it normalizes with the running buffers and changes nothing. The learned scale and shift are applied identically in both modes.

The reason for two behaviours is what each mode is for. Training wants the actual distribution of the batch, so the network learns against what it will see. Evaluation wants each prediction to depend only on its own input: if the batch statistics were used, the output for one image would change depending on which other images happened to share its batch, and a model served one request at a time would divide by a variance of zero. So evaluation uses the statistics remembered from training, and the buffers are frozen — which is also why evaluation must not update them.

```python
import torch as t
m=t.nn.Module()
m.eval()
if m.training:
    label="batch statistics"
else:
    label="running statistics"
print(label)
# Hidden checks
assert label=="running statistics"
```

## Worked example

We normalize the same one-channel batch in both modes and compare. In evaluation mode the stored mean `1` and variance `4` are used, so a pixel at `1` maps to zero and a pixel at `3` maps to `2 / sqrt(4 + eps)`, just under `1`.

```python
import torch as t
x=t.tensor([[[[1.,3.]]]])
rm=t.tensor([1.]); rv=t.tensor([4.]); eps=1e-5
z_eval=(x-rm[None,:,None,None])/t.sqrt(rv[None,:,None,None]+eps)
print(z_eval)
# Hidden checks
assert t.allclose(z_eval,t.tensor([[[[0.,1.]]]]),atol=1e-3)
```

In training mode the batch's own statistics apply — mean `2`, variance `1` — so the same two pixels land symmetrically at `-1` and `1`. Same input, different output: that is the flag doing its job.

```python
mean=x.mean(dim=(0,2,3)); var=((x-mean[None,:,None,None])**2).mean(dim=(0,2,3))
z_train=(x-mean[None,:,None,None])/t.sqrt(var[None,:,None,None]+eps)
print(z_train)
# Hidden checks
assert t.allclose(z_train,t.tensor([[[[-1.,1.]]]]),atol=1e-3)
```

## Faded practice

### q1300
Return evaluation-mode normalized and affine-transformed x. x: float (batch,channels,height,width); g: scale (channels,); b: shift (channels,); rm: running mean (channels,); rv: running variance (channels,); eps > 0. Variance is the population variance.

```python starter
import torch as t

def solve(x,g,b,rm,rv,eps):
    pass
```

```python solution
import torch as t

def solve(x,g,b,rm,rv,eps):
    mean=rm
    var=rv
    z=(x-mean[None,:,None,None])/t.sqrt(var[None,:,None,None]+eps)
    y=z*g[None,:,None,None]+b[None,:,None,None]
    return y
```

## Solo practice

### q1301
Return the training-mode BatchNorm output followed by ReLU, preserving the shape of x. x: float (batch,channels,height,width); g: scale (channels,); b: shift (channels,); eps > 0. Variance is the population variance.

### q1302
Return the evaluation-mode BatchNorm output averaged over both spatial axes, shape (batch,channels). x: float (batch,channels,height,width); g: scale (channels,); b: shift (channels,); rm: running mean (channels,); rv: running variance (channels,); eps > 0. Variance is the population variance.

### q1303
Return population variance of each input channel, shape (channels,). x: float (batch,channels,height,width).

### q1304
Return the updated running variance after this batch. x: float (batch,channels,height,width); rv: running variance (channels,); momentum in (0,1]. Variance is the population variance.

### q1305
Return training output minus evaluation output for the same input. x: float (batch,channels,height,width); g: scale (channels,); b: shift (channels,); rm: running mean (channels,); rv: running variance (channels,); eps > 0. Variance is the population variance.

### q1306
Return the running mean after observing this same batch twice. Each update uses the result of the previous update. x: float (batch,channels,height,width); rm: running mean (channels,); momentum in (0,1]. Variance is the population variance.

### q1307
Return the effective scale per channel for evaluation: the multiplier that converts each input value into output before adding a constant offset. g: scale (channels,); rv: running variance (channels,); eps > 0.

### q1308
Return the effective offset per channel for evaluation, so output equals input times effective scale plus this offset. g: scale (channels,); b: shift (channels,); rm: running mean (channels,); rv: running variance (channels,); eps > 0.

### q1309
Return training-mode channel means and population variances after the affine transform, shape (2,channels), means first. x: float (batch,channels,height,width); g: scale (channels,); b: shift (channels,); eps > 0. Variance is the population variance.

## Integrated practice

### q1310
Implement a stateful BatchNorm module. Its training forward must update the running statistics once per call; evaluation must leave every buffer unchanged. g: scale (channels,); b: shift (channels,); rm: running mean (channels,); rv: running variance (channels,); eps > 0; momentum in (0,1]. Return a new training-mode module whose parameters are weight and bias and whose buffers are running_mean, running_var and num_batches_tracked; normalize with the population variance and use it for the running update too.

### q1311
Implement a stateful BatchNorm module whose forward also applies ReLU after the affine step, as a ResNet block does. g: scale (channels,); b: shift (channels,); rm: running mean (channels,); rv: running variance (channels,); eps > 0; momentum in (0,1]. Return a new training-mode module whose parameters are weight and bias and whose buffers are running_mean, running_var and num_batches_tracked; normalize with the population variance and use it for the running update too.

### q1312
Implement a stateful BatchNorm module whose forward ends with global average pooling, returning shape (batch,channels). g: scale (channels,); b: shift (channels,); rm: running mean (channels,); rv: running variance (channels,); eps > 0; momentum in (0,1]. Return a new training-mode module whose parameters are weight and bias and whose buffers are running_mean, running_var and num_batches_tracked; normalize with the population variance and use it for the running update too.

## Misconceptions

- **Statistics are per image.** They are per channel, pooled over every example in the batch and every pixel; a per-image mean would erase the difference between a bright image and a dark one.
- **Variance uses `n-1`.** BatchNorm uses the population variance; `x.var()`'s default correction gives a different number.
- **Evaluation mode still updates the buffers.** It reads them and leaves them alone; only training mode writes.
- **The running update is part of the gradient path.** It is bookkeeping done under `no_grad`; the normalized output, not the buffer write, is what gradients flow through.
