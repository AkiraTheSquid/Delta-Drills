---
kc: cnn.forward-hooks
title: Forward hooks
new_syntax: ['Tensor.register_forward_hook', 'Tensor.remove', 'Tensor.apply', 'torch.isnan']
concepts: [hook-signature-and-handle, apply-recursively]
supporting: ['cnn.sequential', 'cnn.module-state', 'python.control-flow']
previews: []
faded: [1458, 1459, 1460, 1461]
guided: []
independent: [1462, 1463, 1464, 1465, 1466, 1467]
integrated: [1468, 1469, 1470]
---

## Concept: A forward hook runs after a module's forward, and its handle removes it

A *forward hook* is a function you attach to a module; every time the module finishes its forward pass, the hook is called with three arguments — the module, the tuple of inputs it received, and the output it produced — and it can inspect any of them. Registering returns a *handle*; calling the handle's `remove()` detaches the hook again. Hooks stack: register two and both fire, and a hook you forget to remove keeps firing on every later call.

If the hook returns a value, that value **replaces** the module's output for the caller — the way to patch a layer without editing its forward. A hook that only records things returns nothing, and the output passes through unchanged.

```python
import torch as t
net=t.nn.Sequential(t.nn.Linear(4,3),t.nn.ReLU(),t.nn.Linear(3,2))
shapes=[]
def record(module,inputs,output):
    shapes.append(tuple(output.shape))
handles=[child.register_forward_hook(record) for child in net]
net(t.zeros(5,4))
print(shapes)
for h in handles:
    h.remove()
net(t.zeros(5,4))
print(len(shapes))
# Hidden checks
assert shapes==[(5,3),(5,3),(5,2)] and len(shapes)==3
```

## Worked example

Attach a hook that doubles the output of the ReLU. Predict, before running, how the final output compares with and without the hook: the second linear layer sees doubled activations, so its bias-free part doubles.

```python
with t.no_grad():
    for p in net.parameters():
        p.copy_(t.linspace(-1.,1.,p.numel()).reshape(p.shape))
x=t.ones(1,4)
before=net(x)
def double(module,inputs,output):
    return output*2
h=net[1].register_forward_hook(double)
during=net(x)
print(before, during)
# Hidden checks
assert (during-before).abs().max().item()>0
```

The hook's returned tensor became the ReLU's output, so the final logits changed. Remove the handle and the chain is back to normal:

```python
h.remove()
print(net(x))
# Hidden checks
assert (net(x)-before).abs().max().item()==0
```

The output equals the first print: once the handle is removed, the hook is gone.

## Faded practice

### q1458
Given a Sequential chain `m` and an input `x`, attach a forward hook to EACH direct child that appends the child's output shape (a tuple of Python ints) to a list; run the chain on `x` once; detach every hook through its handle; run the chain once more; and return the list — one entry per child.

```python starter
import torch as t

def solve(m,x):
    shapes=[]
    def record(module,inputs,output):
        shapes.append(tuple(output.shape))
    handles=[child._____(record) for child in m]
    m(x)
    for h in handles:
        h._____()
    m(x)
    return shapes
```

```python solution
import torch as t

def solve(m,x):
    shapes=[]
    def record(module,inputs,output):
        shapes.append(tuple(output.shape))
    handles=[child.register_forward_hook(record) for child in m]
    m(x)
    for h in handles:
        h.remove()
    m(x)
    return shapes
```

### q1459
Given a Sequential chain `m`, an input `x` and an index `k`, capture the OUTPUT tensor of the `k`-th direct child (position `k` in the chain) with a forward hook while running the chain on `x`, detach the hook, and return the captured tensor.

```python starter
import torch as t

def solve(m,x,k):
    captured=[]
    def grab(module,inputs,output):
        captured.append(output)
    h=list(m)[k]._____(grab)
    m(x)
    h._____()
    return captured[0]
```

```python solution
import torch as t

def solve(m,x,k):
    captured=[]
    def grab(module,inputs,output):
        captured.append(output)
    h=list(m)[k].register_forward_hook(grab)
    m(x)
    h.remove()
    return captured[0]
```

## Concept: `apply` visits every module in the tree, including the root

Attaching a hook to each child by hand misses nested children — a block inside a group inside a network. A module's `apply(fn)` calls `fn` on every module in the tree — each child's own subtree is finished before that child, siblings go in registration order, and the module itself comes last — so one small function can register a hook on all of them at once. The standard use is a NaN detector: a hook that checks its output with `isnan` (elementwise, `True` where a value is not-a-number) and records the module when any entry is NaN. NaN usually propagates, so modules downstream of the culprit tend to be flagged too; the first record is the earliest module whose OUTPUT held a NaN, and comparing its input tells you whether it produced the NaN or merely passed one on.

```python
import torch as t
class NanModule(t.nn.Module):
    def forward(self,x):
        return x*float("nan")
net=t.nn.Sequential(t.nn.Identity(),t.nn.Sequential(NanModule(),t.nn.Identity()),t.nn.Identity())
count=[0]
def tally(module):
    count[0]+=1
net.apply(tally)
print(count[0])
print(t.isnan(t.tensor([1.,float("nan"),3.])))
# Hidden checks
assert count[0]==6 and t.isnan(t.tensor([1.,float("nan"),3.])).tolist()==[False,True,False]
```

## Worked example

Six modules: three top-level children, the two inside the nested chain, and the root. Now the detector. Predict how many modules flag NaN before running: the NaN module, the identity after it inside the nested chain, the nested chain itself, the outer identity after it, and the root — five.

```python
flagged=[]
def check(module,inputs,output):
    if t.isnan(output).any():
        flagged.append(module)
handles=[]
def add_hook(module):
    handles.append(module.register_forward_hook(check))
net.apply(add_hook)
net(t.ones(3))
print(len(flagged), type(flagged[0]) is NanModule)
# Hidden checks
assert len(flagged)==5 and type(flagged[0]) is NanModule
```

Five, and the first flagged module is the source. Keeping the handles in a list is what lets you remove every hook afterwards — a second run then records nothing new:

```python
for h in handles:
    h.remove()
net(t.ones(3))
print(len(flagged))
# Hidden checks
assert len(flagged)==5
```

## Faded practice

### q1460
Given a module `m`, return the number of modules `apply` visits — every module in the tree, nested ones included, plus `m` itself — as a Python int.

```python starter
import torch as t

def solve(m):
    count=[0]
    def tally(module):
        count[0]+=1
    m._____(tally)
    return count[0]
```

```python solution
import torch as t

def solve(m):
    count=[0]
    def tally(module):
        count[0]+=1
    m.apply(tally)
    return count[0]
```

### q1461
Given a module `m` and an input `x`, use `apply` to attach to EVERY module in the tree a forward hook that records the module when any entry of its OUTPUT is not-a-number; run the model on `x`; return how many modules were recorded, as a Python int.

```python starter
import torch as t

def solve(m,x):
    flagged=[]
    def check(module,inputs,output):
        if t._____(output).any():
            flagged.append(module)
    def add_hook(module):
        module._____(check)
    m._____(add_hook)
    m(x)
    return len(flagged)
```

```python solution
import torch as t

def solve(m,x):
    flagged=[]
    def check(module,inputs,output):
        if t.isnan(output).any():
            flagged.append(module)
    def add_hook(module):
        module.register_forward_hook(check)
    m.apply(add_hook)
    m(x)
    return len(flagged)
```

## Solo practice

### q1462
Given a Sequential chain `m` and an input `x`, return a list with one `(min, max)` tuple of Python floats per direct child, the smallest and largest value of that child's output, captured with forward hooks that are detached before returning.

### q1463
Given a Sequential chain `m`, an input `x` and an index `k`, attach a forward hook to the `k`-th direct child that appends its output to a list, run the chain on `x`, detach the hook through its handle, run the chain on `x` again, and return a tuple: the first captured tensor and the number of captures (a Python int).

### q1464
Return the output of the Sequential chain `m` on `x` while the direct child at position `k` has its output rescaled. Attach to that child a forward hook that replaces the child's output with the output multiplied by the scalar `c`; run the chain on `x`; detach the hook; return the tensor the chain produced while the hook was attached. Later children see the rescaled tensor, so the result is not simply the plain output times `c`.

### q1465
Given a module `m` and an input `x`, use `apply` to attach a forward hook to EVERY module in the tree that appends the output shape (a tuple of Python ints) to a list; run the model on `x`; return the list in the order the hooks fired.

### q1466
Given a Sequential chain `m` and an input `x`, return a list of Python booleans, one per direct child, saying whether that child's OUTPUT contains any not-a-number entry, captured with forward hooks that are detached before returning.

### q1467
Given a module `m`, return the number of modules strictly BELOW it in the tree (every nested module, at any depth, but not `m` itself), as a Python int, using `apply`.

## Integrated practice

### q1468
Build a removable NaN detector. Given a module `m` and an input `x`: use `apply` to attach to every module in the tree a forward hook that records the module when any entry of its output is not-a-number, keeping every handle in a list; run the model on `x`; detach every hook through its handle; run the model on `x` again; return a tuple `(count, first_is_source)`: the number of records as a Python int, and a Python boolean saying whether the first recorded module is the source (its output has NaN while its input does not). With no record at all return `(0, False)`.

### q1469
Given a module `m` and an input `x`, use `apply` to attach to every module in the tree a forward hook that records the mean of the module's output as a Python float, keeping the handles; run the model on `x`; detach every hook; return the list of means in firing order (nested modules before their parents, the root last).

### q1470
Given a module `m`, an input `x`, the registered `name` of one of its direct children and a scalar `c`: look the child up by name through the children mapping, attach a forward hook that replaces that child's output with the output multiplied by `c`, run the model on `x`, detach the hook, run the model again, and return a tuple of two tensors — the output with the hook attached and the output after it was removed.

## Misconceptions

- **A hook changes the output only if it edits it in place.** A returned value replaces the output; a hook that returns nothing leaves it unchanged.
- **Hooks vanish after one call.** They fire on every forward until their handle is removed.
- **`apply` visits only the direct children.** It walks the whole tree, children before parents, and ends with the module itself.
- **Only the NaN source is flagged.** NaN usually propagates, so downstream modules are flagged too; the first record is the first module whose output held a NaN, which is the source only if its input was clean.
