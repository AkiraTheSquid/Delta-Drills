---
kc: numpy.random-samplers
title: Drawing random numbers
supporting: [numpy.constructors, numpy.ndarray-model]
new_syntax: [torch.rand, torch.randn, torch.randint, torch.randperm]
previews: []
faded: [42, 676, 677]
guided: []
independent: [678, 679, 680, 681, 682, 683]
integrated: [684, 685, 686]
---

## Concept

Four functions cover almost all the random tensors you will ever need. They are
constructors like `t.zeros` and `t.ones` — you say what SHAPE you want and get a
tensor of that shape back — except the numbers are drawn rather than fixed.

```python
import torch as t

print("rand    ", t.rand(3))            # uniform floats in [0, 1)
print("randn   ", t.randn(3))           # standard normal: mean 0, spread 1
print("randint ", t.randint(0, 10, (3,)))   # whole numbers in [0, 10)
print("randperm", t.randperm(6))        # 0..5 shuffled into a random order
```

The four differ in **what they draw from**, not in how you call them:

| Call | What comes out |
|---|---|
| `t.rand(shape)` | floats spread evenly across `[0, 1)` — 0 possible, 1 never |
| `t.randn(shape)` | floats from the standard normal — negatives are normal here |
| `t.randint(low, high, shape)` | whole numbers in `[low, high)` — `high` excluded |
| `t.randperm(n)` | every number `0..n-1` exactly once, in a random order |

`rand` and `randn` take the shape the way the constructors do — loose integers or
a tuple, whichever reads better:

```python
import torch as t

print(t.rand(2, 3).shape)      # loose integers
print(t.rand((2, 3)).shape)    # the same shape as a tuple
print(t.randn(4).shape)        # 1-D, four numbers
```

`randint` is the odd one out: its shape argument comes **third and must be a
tuple**, because the first two slots are already spoken for by the range.

```python
import torch as t

x = t.randint(0, 10, (5,))
print(x, x.dtype)              # note the dtype: whole numbers, not floats
print(t.randint(0, 10, (2, 3)))
```

`randperm` is not a draw from a distribution at all — it is a **shuffling**.
Every number from `0` to `n-1` comes out exactly once, which is what makes it
the tool for putting things in a random order rather than picking random things.

```python
import torch as t

p = t.randperm(5)
print("a permutation:", p)
print("sorted back  :", sorted(p.tolist()))
assert sorted(p.tolist()) == [0, 1, 2, 3, 4]
```

That last assert is the whole difference between `randperm` and `randint`:
`randint` draws **with replacement**, so it repeats values and skips others.

```python
import torch as t

print("randint may repeat:", t.randint(0, 5, (5,)))
print("randperm never does:", t.randperm(5))
```

Because a draw is different every time you run it, the things you can *assert*
about a random tensor are its structure and its bounds — shape, dtype, and the
range the values fall in — not the values themselves. That is what the drills
below ask for.

## Watch out

- **`t.randint`'s `high` is EXCLUSIVE** — `t.randint(0, 10, (5,))` never
  produces a 10. Same convention as `range` and as slicing.
- **`t.randint` wants a SHAPE, not a count** — the third argument is `(n,)`,
  with the trailing comma, not `n`. `(n, 1)` is a column, which is a different
  tensor.
- **`t.rand` is not `t.randn`** — one is uniform on `[0, 1)`, the other is the
  standard normal and goes negative. They have the same shape and the same
  dtype, so the shape cannot tell you which one you called.
- **`t.randperm(n)` takes a COUNT, not a shape** — it is always 1-D, and it is a
  permutation of `0..n-1`, never of your own values.

## Worked example

Task: draw with each of the four samplers and report the facts about the result
that are true on every run.

```python
import torch as t

# Uniform: shape you asked for, floats, every value in [0, 1).
u = t.rand(2, 3)
assert tuple(u.shape) == (2, 3)
assert str(u.dtype) == "torch.float32"
assert bool((u >= 0).all())
assert bool((u < 1).all())
print("uniform:", u)

# Normal: same shape rules, same dtype, but NOT bounded to [0, 1).
g = t.randn(2, 3)
assert tuple(g.shape) == (2, 3)
assert str(g.dtype) == "torch.float32"
print("normal :", g)

# Ints: the range comes first, the SHAPE third, and the dtype is integral.
i = t.randint(0, 10, (5,))
assert tuple(i.shape) == (5,)
assert str(i.dtype) == "torch.int64"
assert bool((i >= 0).all())
assert bool((i < 10).all())
print("ints   :", i)

# Permutation: a count, not a shape — and every value appears exactly once.
p = t.randperm(5)
assert tuple(p.shape) == (5,)
assert sorted(p.tolist()) == [0, 1, 2, 3, 4]
print("perm   :", p)
```

Why each step:

1. Asserting shape and dtype rather than values is the only thing that can be
   true on every run — and it is what the graders below check.
2. The bound checks are what distinguish `rand` from `randn` in code: only the
   uniform one is trapped in `[0, 1)`.
3. `sorted(p.tolist())` is the definition of a permutation, written out. If it
   ever came back with a repeat, `randperm` would not be doing its job.

## Faded practice

### q42
A uniform random tensor of a given 3-D shape.

```python starter
import torch as t

def solve(shape):
    """Return a tensor of the given shape, uniform in [0, 1)."""
    return t._____(shape)
```

```python solution
import torch as t

def solve(shape):
    """Return a tensor of the given shape, uniform in [0, 1)."""
    return t.rand(shape)
```

### q676
The shape, dtype and rank of a standard-normal draw.

```python starter
import torch as t

def solve(rows, cols):
    """Return (shape, dtype name, ndim) of a standard-normal draw."""
    x = t._____(rows, cols)
    return (tuple(x.shape), str(x.dtype), x.ndim)
```

```python solution
import torch as t

def solve(rows, cols):
    """Return (shape, dtype name, ndim) of a standard-normal draw."""
    x = t.randn(rows, cols)
    return (tuple(x.shape), str(x.dtype), x.ndim)
```

### q677
Random integers, and a check that they landed inside the half-open range.

```python starter
import torch as t

def solve(low, high, n):
    """Return (shape, every value >= low?, every value < high?) for n random ints."""
    x = t._____(low, high, (n,))
    return (tuple(x.shape), bool((x >= low).all()), bool((x < high).all()))
```

```python solution
import torch as t

def solve(low, high, n):
    """Return (shape, every value >= low?, every value < high?) for n random ints."""
    x = t.randint(low, high, (n,))
    return (tuple(x.shape), bool((x >= low).all()), bool((x < high).all()))
```

## Solo practice

### q678
The shape, bounds and dtype of a uniform draw.

### q679
A random permutation, and the values sorted back into order.

### q680
Shuffle a tensor's rows with a random permutation.

### q681
A 2-D draw of random integers, with the bounds checked.

### q682
A uniform draw whose shape arrives as a tuple.

### q683
A 2-D normal draw, reported by shape, size and row length.

## Integrated practice

### q684
All four samplers at once, reported by dtype and by the sorted permutation.

### q685
A random permutation MATRIX: the identity with its rows shuffled.

### q686
Random integers reported by shape, count, range and dtype together.
