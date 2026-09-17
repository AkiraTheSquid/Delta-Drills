---
kc: cnn.training-loop
title: The training loop
new_syntax: ['Tensor.backward', 'Tensor.grad', 'torch.optim.Adam', 'Tensor.step', 'Tensor.zero_grad', 'torch.nn.functional.cross_entropy', 'Tensor.eval', 'Tensor.train']
concepts: [backward-fills-grad, optimizer-step, zero-grad-and-loss, train-eval-modes]
supporting: ['cnn.mlp', 'tensor.classifier-evaluation', 'python.control-flow', 'cnn.batch-normalization', 'cnn.feature-extraction']
previews: []
faded: [1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448]
guided: []
independent: [1449, 1450, 1451, 1452, 1453, 1454, 425]
integrated: [1455, 1456, 1457]
---

## Concept: Backward fills every parameter's gradient

A loss is a single number computed from the model's output. Calling `backward()` on it walks the computation back to every parameter that took part and writes the derivative of the loss with respect to that parameter into the parameter's `grad` attribute — a tensor of the same shape, `None` before the first backward. One step of gradient descent then moves every parameter a little way against its gradient: `p -= lr * p.grad`, done under `no_grad` because the update itself must not be recorded as part of any computation.

The reason the gradient has the parameter's shape is that it holds one derivative per weight value. The reason the update must be wrapped in `no_grad` is that a parameter is a leaf that requires a gradient; changing it in place while recording is an error.

```python
import torch as t
lin=t.nn.Linear(2,1,bias=False)
with t.no_grad():
    lin.weight.copy_(t.tensor([[1.,2.]]))
x=t.tensor([[1.,1.]])
loss=(lin(x)**2).mean()
loss.backward()
print(lin.weight.grad)
with t.no_grad():
    lin.weight-=0.1*lin.weight.grad
print(lin.weight)
# Hidden checks
assert lin.weight.grad.tolist()==[[6.,6.]] and (lin.weight-t.tensor([[0.4,1.4]])).abs().max().item()<1e-6
```

## Worked example

The output is `1*1 + 2*1 = 3`, the loss is `3² = 9`, and the derivative of `out²` with respect to each weight is `2 * out * x = 6`. Predict the loss after the step before running: the new output is `0.4 + 1.4 = 1.8`, so the loss should be `3.24`.

```python
print(((lin(x)**2).mean()).item())
# Hidden checks
assert abs(((lin(x)**2).mean()).item()-3.24)<1e-5
```

## Faded practice

### q1441
Given a module `m`, an input batch `x`, a `target` tensor of the same shape as the module's output and a learning rate `lr`, take ONE step of plain gradient descent on the mean squared error between output and target: compute the loss, run backward, and subtract `lr` times each parameter's gradient from that parameter in place without recording the update. Return `m`.

```python starter
import torch as t

def solve(m,x,target,lr):
    loss=((m(x)-target)**2).mean()
    loss._____()
    with t.no_grad():
        for p in m.parameters():
            p-=lr*p._____
    return m
```

```python solution
import torch as t

def solve(m,x,target,lr):
    loss=((m(x)-target)**2).mean()
    loss.backward()
    with t.no_grad():
        for p in m.parameters():
            p-=lr*p.grad
    return m
```

### q1442
Given a module `m`, an input batch `x` and a `target` tensor of the output's shape, compute the mean squared error between output and target, run backward, and return the gradient tensor of the module's FIRST parameter (in registration order).

```python starter
import torch as t

def solve(m,x,target):
    loss=((m(x)-target)**2).mean()
    loss._____()
    return list(m.parameters())[0]._____
```

```python solution
import torch as t

def solve(m,x,target):
    loss=((m(x)-target)**2).mean()
    loss.backward()
    return list(m.parameters())[0].grad
```

## Concept: An optimizer owns the parameters and applies the step

Writing the update by hand works but scales badly and misses the better update rules. An optimizer object is constructed with the parameters it may change and a learning rate; after `backward()` has filled the gradients, its `step()` applies the rule to every one of those parameters at once. Adam is the rule ARENA uses throughout: it scales each parameter's step by a running estimate of its gradient size, so on the very first step every value with a nonzero gradient moves by almost exactly the learning rate (the step is `lr` times `g / (|g| + eps)`), against the sign of its gradient; a zero gradient moves nothing.

The reason to hand the optimizer the parameters up front is that it keeps per-parameter running statistics between steps; the reason the first Adam step is `±lr` is that those statistics start empty, so the normalised gradient is just its sign.

```python
import torch as t
lin=t.nn.Linear(2,1,bias=False)
with t.no_grad():
    lin.weight.copy_(t.tensor([[1.,2.]]))
opt=t.optim.Adam(lin.parameters(),lr=0.1)
x=t.tensor([[1.,1.]])
loss=(lin(x)**2).mean()
loss.backward()
opt.step()
print(lin.weight)
# Hidden checks
assert (lin.weight-t.tensor([[0.9,1.9]])).abs().max().item()<1e-4
```

## Worked example

The gradient is `[6, 6]` as before, but Adam's first step ignores its size: both weights move by the learning rate `0.1`. Predict the loss after the step before running: output `0.9 + 1.9 = 2.8`, loss `7.84`.

```python
print(((lin(x)**2).mean()).item())
# Hidden checks
assert abs(((lin(x)**2).mean()).item()-7.84)<1e-3
```

## Faded practice

### q1443
Given a module `m`, an input batch `x`, a `target` tensor of the output's shape and a learning rate `lr`, build an Adam optimizer over the module's parameters with that learning rate, compute the mean squared error between output and target, run backward, and apply ONE optimizer step. Return `m`.

```python starter
import torch as t

def solve(m,x,target,lr):
    opt=t.optim._____(m.parameters(),lr=lr)
    loss=((m(x)-target)**2).mean()
    loss._____()
    opt._____()
    return m
```

```python solution
import torch as t

def solve(m,x,target,lr):
    opt=t.optim.Adam(m.parameters(),lr=lr)
    loss=((m(x)-target)**2).mean()
    loss.backward()
    opt.step()
    return m
```

### q1444
Given `m`, `x`, a `target` of the output's shape and a learning rate `lr`, return a tuple of two Python floats: the mean squared error before an Adam step and the mean squared error after it (recompute the output after the step).

```python starter
import torch as t

def solve(m,x,target,lr):
    opt=t.optim._____(m.parameters(),lr=lr)
    loss=((m(x)-target)**2).mean()
    loss._____()
    opt._____()
    after=((m(x)-target)**2).mean()
    return (loss.item(),after.item())
```

```python solution
import torch as t

def solve(m,x,target,lr):
    opt=t.optim.Adam(m.parameters(),lr=lr)
    loss=((m(x)-target)**2).mean()
    loss.backward()
    opt.step()
    after=((m(x)-target)**2).mean()
    return (loss.item(),after.item())
```

## Concept: Clear the gradients each step, and use the classification loss

`backward()` *adds* to `grad`; it never overwrites. Two backward passes on the same batch without clearing leave twice the gradient, so a loop must call the optimizer's `zero_grad()` once per step — after the step, or before the backward, either is fine as long as it happens every iteration. For classification the loss is cross-entropy: given a `(batch, classes)` matrix of logits and a `(batch,)` tensor of integer labels, it computes the log-sum-exp of each row minus the logit of the true class and averages over the batch — the same formula as the evaluation lesson, in one library call.

```python
import torch as t
lin=t.nn.Linear(2,1,bias=False)
with t.no_grad():
    lin.weight.copy_(t.tensor([[1.,2.]]))
opt=t.optim.Adam(lin.parameters(),lr=0.1)
x=t.tensor([[1.,1.]])
(lin(x)**2).mean().backward()
(lin(x)**2).mean().backward()
print(lin.weight.grad)
opt.zero_grad()
print(lin.weight.grad)
# Hidden checks
assert lin.weight.grad is None or lin.weight.grad.abs().sum().item()==0
```

## Worked example

Logits `[1, 2, 3]` with true class `2`: the loss is `log(e¹ + e² + e³) - 3 ≈ 0.4076`. Predict which of the two batches has the larger loss before running: the second row's true class is the smallest logit.

```python
logits=t.tensor([[1.,2.,3.],[1.,2.,3.]])
y=t.tensor([2,0])
print(t.nn.functional.cross_entropy(logits[:1],y[:1]).item())
print(t.nn.functional.cross_entropy(logits,y).item())
# Hidden checks
assert abs(t.nn.functional.cross_entropy(logits[:1],y[:1]).item()-0.4076)<1e-3
assert abs(t.nn.functional.cross_entropy(logits,y).item()-(0.4076+2.4076)/2)<1e-3
```

The batch loss is the mean of the two rows, `(0.4076 + 2.4076) / 2`.

## Faded practice

### q1445
Given a classifier `m`, a batch `x`, integer labels `y`, a learning rate `lr` and a step count `k`, build an Adam optimizer and take `k` steps on the cross-entropy loss between the logits and the labels, clearing the gradients every step. Return the list of `k` loss values as Python floats, in order.

```python starter
import torch as t

def solve(m,x,y,lr,k):
    opt=t.optim._____(m.parameters(),lr=lr)
    losses=[]
    for step in [0]*k:
        loss=t.nn.functional._____(m(x),y)
        loss._____()
        opt._____()
        opt._____()
        losses.append(loss.item())
    return losses
```

```python solution
import torch as t

def solve(m,x,y,lr,k):
    opt=t.optim.Adam(m.parameters(),lr=lr)
    losses=[]
    for step in [0]*k:
        loss=t.nn.functional.cross_entropy(m(x),y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    return losses
```

### q1446
Given a classifier `m`, a batch `x` and integer labels `y`, compute the cross-entropy loss and run backward TWICE (recomputing the loss the second time) without clearing anything in between, and return the resulting gradient tensor of the module's first parameter.

```python starter
import torch as t

def solve(m,x,y):
    t.nn.functional._____(m(x),y)._____()
    t.nn.functional._____(m(x),y)._____()
    return list(m.parameters())[0]._____
```

```python solution
import torch as t

def solve(m,x,y):
    t.nn.functional.cross_entropy(m(x),y).backward()
    t.nn.functional.cross_entropy(m(x),y).backward()
    return list(m.parameters())[0].grad
```

## Concept: Train mode for the update, eval mode for the measurement

A module has a mode flag. In *train* mode a batch norm normalises with the current batch's statistics and updates its running averages; in *eval* mode it uses the stored running averages and changes nothing. `train()` and `eval()` set the flag on the module and every child. The loop therefore switches to train mode before the epoch's updates and to eval mode before measuring accuracy, and the measurement runs under `no_grad` because no update follows it.

```python
import torch as t
net=t.nn.Sequential(t.nn.Conv2d(1,2,kernel_size=3,padding=1,bias=False),t.nn.BatchNorm2d(2),t.nn.ReLU(),t.nn.Flatten(),t.nn.Linear(32,3))
def evaluate(m,test_batches):
    m.eval()
    correct=0
    total=0
    with t.no_grad():
        for x,y in test_batches:
            correct+=(m(x).argmax(dim=1)==y).sum().item()
            total+=len(y)
    return correct/total
x=t.linspace(-1.,1.,2*16).reshape(2,1,4,4)
print(net.training, net.train()(x).sum().item(), net.eval()(x).sum().item(), net.training)
# Hidden checks
assert net.training is False
```

## Worked example

The full loop, one epoch at a time: train mode, then for every training batch compute logits, loss, backward, step, zero the gradients and record the loss; then eval mode and one accuracy number for the epoch. Predict how many losses and how many accuracies two epochs over three training batches produce before running.

```python
def train(m,train_batches,test_batches,lr,epochs):
    opt=t.optim.Adam(m.parameters(),lr=lr)
    loss_list=[]
    acc_list=[]
    for epoch in [0]*epochs:
        m.train()
        for x,y in train_batches:
            loss=t.nn.functional.cross_entropy(m(x),y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            loss_list.append(loss.item())
        acc_list.append(evaluate(m,test_batches))
    return loss_list,acc_list
te=[(t.linspace(0.,1.,4*16).reshape(4,1,4,4),t.tensor([0,1,2,0]))]
print(evaluate(net,te))
# Hidden checks
assert 0<=evaluate(net,te)<=1
```

The untrained accuracy is whatever the random head happens to guess. Now three training batches, two epochs:

```python
tr=[(t.linspace(-1.,1.,4*16).reshape(4,1,4,4)*(k+1)/3,(t.arange(4)+k)%3) for k in [0,1,2]]
losses,accs=train(net,tr,te,0.05,2)
print(len(losses), len(accs), accs)
# Hidden checks
assert len(losses)==6 and len(accs)==2
```

Six losses (one per batch per epoch) and two accuracies (one per epoch).

## Faded practice

### q1447
Given a classifier `m` and `test_batches`, a list of `(x, y)` pairs, put the module in evaluation mode and, without recording gradients, count how many predictions (largest logit per example) equal their label across every batch. Return the accuracy `correct / total` as a Python float.

```python starter
import torch as t

def solve(m,test_batches):
    m._____()
    correct=0
    total=0
    with t.no_grad():
        for x,y in test_batches:
            correct+=(m(x).argmax(dim=1)==y).sum().item()
            total+=len(y)
    return correct/total
```

```python solution
import torch as t

def solve(m,test_batches):
    m.eval()
    correct=0
    total=0
    with t.no_grad():
        for x,y in test_batches:
            correct+=(m(x).argmax(dim=1)==y).sum().item()
            total+=len(y)
    return correct/total
```

### q1448
Train the classifier `m` for `epochs` epochs and return its batch losses and per-epoch accuracies. Given lists of `(x, y)` pairs `train_batches` and `test_batches` (logits of shape `(B, C)`, labels of shape `(B,)`) and a learning rate `lr`, build an Adam optimizer over the module's parameters and, for each epoch: switch to train mode; for every training batch compute the cross-entropy loss of the logits against the labels, run backward, step, clear the gradients and record the loss as a float; then switch to evaluation mode and, without recording gradients, record the accuracy over all test batches. Return the tuple `(loss_list, acc_list)`.

```python starter
import torch as t

def solve(m,train_batches,test_batches,lr,epochs):
    opt=t.optim._____(m.parameters(),lr=lr)
    loss_list=[]
    acc_list=[]
    for epoch in [0]*epochs:
        m._____()
        for x,y in train_batches:
            loss=t.nn.functional._____(m(x),y)
            loss._____()
            opt._____()
            opt._____()
            loss_list.append(loss.item())
        m._____()
        correct=0
        total=0
        with t.no_grad():
            for x,y in test_batches:
                correct+=(m(x).argmax(dim=1)==y).sum().item()
                total+=len(y)
        acc_list.append(correct/total)
    return (loss_list,acc_list)
```

```python solution
import torch as t

def solve(m,train_batches,test_batches,lr,epochs):
    opt=t.optim.Adam(m.parameters(),lr=lr)
    loss_list=[]
    acc_list=[]
    for epoch in [0]*epochs:
        m.train()
        for x,y in train_batches:
            loss=t.nn.functional.cross_entropy(m(x),y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            loss_list.append(loss.item())
        m.eval()
        correct=0
        total=0
        with t.no_grad():
            for x,y in test_batches:
                correct+=(m(x).argmax(dim=1)==y).sum().item()
                total+=len(y)
        acc_list.append(correct/total)
    return (loss_list,acc_list)
```

## Solo practice

### q1449
Without any optimizer object, take `k` steps of plain gradient descent (learning rate `lr`) on the mean squared error between the output of `m` on `x` and `target`, clearing the module's gradients after each in-place update. Return `m`.

### q1450
Run ONE epoch of training in train mode: an Adam optimizer over the parameters of `m`, and for each `(x, y)` in `train_batches` the cross-entropy loss, backward, step and clearing of the gradients. Return the list of per-batch losses as Python floats.

### q1451
Given a classifier `m`, a batch `x` and integer labels `y`, run backward on the cross-entropy loss once and return a list with the SUM of each parameter's gradient as a Python float, one entry per parameter in registration order.

### q1452
Given a classifier `m` and `test_batches`, a list of `(x, y)` pairs, return a list with the accuracy of each batch on its own as a Python float, measured in evaluation mode without recording gradients.

### q1453
Given a module `m` containing a batch norm and a batch `x`, return a tuple of two tensors computed without recording gradients: the output in train mode, then the output in evaluation mode.

### q1454
Given a classifier `m` in evaluation mode, a batch `x` and integer labels `y`, return the cross-entropy loss of the logits against the labels as a Python float, computed without recording gradients.

### q425
One update of a scalar model by hand: the squared-error slope, then the descent step, as arithmetic.

## Integrated practice

### q1455
Train only the last child of the classifier `m` and return its batch losses and per-epoch accuracies. Given `m` (children registered in order), freeze every parameter and unfreeze only the LAST child, build an Adam optimizer over just the parameters whose gradient flag is set, and train for `epochs` epochs exactly as in the lesson (train mode, cross-entropy, backward, step, clear, record the float loss per batch; then evaluation-mode accuracy over the test batches per epoch, without recording gradients). Return `(loss_list, acc_list)`.

### q1456
Train WITHOUT an optimizer object: for each epoch, in train mode, take a plain gradient-descent step (learning rate `lr`, in place, without recording) on every parameter after the cross-entropy backward of each training batch, clearing the module's gradients after each update; then in evaluation mode, without recording gradients, measure accuracy over the test batches. Return the list of per-epoch accuracies as Python floats.

### q1457
Train exactly as in the lesson (Adam over all parameters, train mode, cross-entropy, backward, step, clear; evaluation-mode accuracy per epoch without recording gradients) and return a report tuple: the list of per-epoch accuracies, the index of the epoch with the highest accuracy (the first one on ties, as a Python int), and the final batch loss as a Python float.

## Misconceptions

- **Backward overwrites the gradient.** It accumulates; without a zero-grad every step, step `n` uses the sum of `n` gradients.
- **The optimizer computes the gradient.** Backward does; the optimizer only applies the update rule to the gradients it finds.
- **Adam's first step is proportional to the gradient.** Every value with a nonzero gradient moves by almost exactly the learning rate whatever the gradient's size; the running statistics only shape later steps.
- **Eval mode disables gradients.** It only changes how batch norm and similar layers behave; `no_grad` is a separate switch.
