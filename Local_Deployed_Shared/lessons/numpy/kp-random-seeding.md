---
kc: numpy.random-seeding
title: Seeding your own reproducible stream
supporting: [numpy.random-samplers, numpy.constructors, numpy.ndarray-model]
new_syntax: [torch.Generator, Tensor.manual_seed, torch.rand#generator, torch.randn#generator, torch.randint#generator, torch.randperm#generator]
previews: []
faded: [696, 697]
guided: []
independent: [694, 698, 699, 700, 701, 702, 703]
integrated: [704, 705, 706]
---

## Concept

So far the generator has always arrived from somewhere else. Making one is two
calls, and they are almost always written as one line:

```python
import torch as t

rng = t.Generator().manual_seed(0)
print(rng)
print(t.rand(3, generator=rng))
```

`t.Generator()` builds an empty stream; `.manual_seed(seed)` fixes where that
stream starts and **returns the generator**, which is why the two chain onto one
line. An unseeded `t.Generator()` still works — it just starts somewhere nobody
chose, so nothing about it can be reproduced.

Every sampler takes the generator the same way — as a keyword argument named
`generator`, always last:

```python
import torch as t

rng = t.Generator().manual_seed(0)
print("rand    ", t.rand(3, generator=rng))
print("randn   ", t.randn(3, generator=rng))
print("randint ", t.randint(0, 10, (3,), generator=rng))
print("randperm", t.randperm(4, generator=rng))
```

Note the shape of the API. In NumPy you call **methods on** the generator —
`rng.random(5)`. In PyTorch the generator is an **argument** to an ordinary
function — `t.rand(5, generator=rng)`. Same idea, inverted spelling, and it is
the thing most often got wrong coming from NumPy. `randint` keeps its own
argument order: the shape stays third and `generator=` comes after it.

The seed is not a source of randomness. It is a **name for a whole sequence**.
Two generators given the same seed are not merely similar; they are the same
stream, and they stay in step for as long as you draw from them equally:

```python
import torch as t

a = t.Generator().manual_seed(42)
b = t.Generator().manual_seed(42)
print("a:", t.rand(3, generator=a))
print("b:", t.rand(3, generator=b))
```

Draws consume that sequence, so position matters as much as the seed. After the
draws above, `a` and `b` are three numbers in — and a fresh generator on the
same seed is back at the start:

```python
import torch as t

a = t.Generator().manual_seed(42)
first = t.rand(3, generator=a)
second = t.rand(3, generator=a)
fresh = t.rand(3, generator=t.Generator().manual_seed(42))

print("first :", first)
print("second:", second)
print("fresh :", fresh)
assert first.tolist() != second.tolist()   # the stream moved on
assert fresh.tolist() == first.tolist()    # the seed replayed it
```

That pair of asserts is the entire contract. It is also why an *extra* draw
slipped in anywhere shifts every value after it: you have not changed the
sequence, you have changed your position in it.

Reseeding an existing generator rewinds it to the same starting point:

```python
import torch as t

rng = t.Generator().manual_seed(5)
before = t.rand(2, generator=rng)
rng.manual_seed(5)                      # back to the beginning of the stream
after = t.rand(2, generator=rng)
print("before:", before)
print("after :", after)
assert before.tolist() == after.tolist()
```

Do that to a generator **you** made. Never to one you were handed — that is the
caller's position in their stream, and rewinding it is the bug the previous
lesson warned about.

You will also meet the **global** API in the wild: `t.manual_seed(0)` followed by
plain `t.rand(shape)` with no generator. It seeds one shared, process-level
stream. It works, and plenty of code uses it, but any function anywhere can draw
from it and move everyone else along. For anything you need to reproduce, prefer
an explicit `Generator`: no hidden global state, and no interference between
distant pieces of code.

## Watch out

- **`t.Generator()` on its own is not reproducible.** Without `.manual_seed(...)`
  it starts from a state nobody chose. Seeding is the whole point.
- **`.manual_seed` returns the generator**, which is what makes
  `t.Generator().manual_seed(0)` one expression. It also mutates in place, so
  `rng.manual_seed(0)` on an existing generator rewinds it.
- **A seed names a SEQUENCE, not a number.** Same seed and same draw order gives
  the same values; change either and everything after the change moves.
- **Never reseed a generator you were handed.** Your own: fine. The caller's:
  that is their stream position, and rewinding it corrupts their run.
- **`t.manual_seed(0)` and `t.Generator().manual_seed(0)` are different things.**
  The first seeds the shared global stream; the second seeds a private one.
- **`generator=` is a KEYWORD argument.** It never rides along positionally:
  `t.rand(n, rng)` is not the same call and will not do what you want.

## Worked example

Task: build a seeded stream, show that the same seed replays it exactly, and
show that consuming the stream moves you along it.

```python
import torch as t

# One expression: build the stream, then fix where it starts.
rng = t.Generator().manual_seed(42)

first = t.rand(3, generator=rng)
print("first draw :", first)

# The SAME generator again: the stream has advanced past the first three.
second = t.rand(3, generator=rng)
print("second draw:", second)
assert first.tolist() != second.tolist()

# A NEW generator on the same seed starts the same sequence over.
replay = t.rand(3, generator=t.Generator().manual_seed(42))
assert replay.tolist() == first.tolist()
print("replayed   :", replay)

# Reseeding in place does the same thing to a generator you own.
rng.manual_seed(42)
rewound = t.rand(3, generator=rng)
assert rewound.tolist() == first.tolist()
print("rewound    :", rewound)

# A different seed is a different sequence entirely.
other = t.rand(3, generator=t.Generator().manual_seed(43))
assert other.tolist() != first.tolist()
print("seed 43    :", other)
```

Why each step:

1. Building and seeding in one line is the idiom; the assert on `second` is what
   "each draw consumes the stream" means, written as code.
2. `replay` and `rewound` are the same claim reached two ways — a fresh
   generator on the seed, and rewinding the one you have. Both land at the
   start of the same sequence.
3. The last block is the check people forget: reproducibility is only useful if
   *different* seeds really do give different runs.

## Faded practice

### q696
Build your own seeded generator and take the first draw off it.

```python starter
import torch as t

def solve(seed, n):
    """Seed a new generator and return its first n uniform floats."""
    rng = t._________()._________(seed)
    return t.rand(n, generator=rng).tolist()
```

```python solution
import torch as t

def solve(seed, n):
    """Seed a new generator and return its first n uniform floats."""
    rng = t.Generator().manual_seed(seed)
    return t.rand(n, generator=rng).tolist()
```

### q697
Two generators on one seed, and the proof they agree.

```python starter
import torch as t

def solve(seed, n):
    """Two generators, one seed: return (draw a, draw b, are they identical?)."""
    rng_a = t._________()._________(seed)
    rng_b = t._________()._________(seed)
    a = t.rand(n, generator=rng_a)
    b = t.rand(n, generator=rng_b)
    return (a.tolist(), b.tolist(), a.tolist() == b.tolist())
```

```python solution
import torch as t

def solve(seed, n):
    """Two generators, one seed: return (draw a, draw b, are they identical?)."""
    rng_a = t.Generator().manual_seed(seed)
    rng_b = t.Generator().manual_seed(seed)
    a = t.rand(n, generator=rng_a)
    b = t.rand(n, generator=rng_b)
    return (a.tolist(), b.tolist(), a.tolist() == b.tolist())
```

## Solo practice

### q698
Show that a seed replays: draw, reseed from scratch, draw again.

### q699
Two different seeds, and the proof the runs differ.

### q700
Two consecutive draws off one seeded stream.

### q701
Reseed a generator mid-stream and show it restarted.

### q702
Random integers off a stream you seeded yourself.

### q703
A permutation off a seeded stream, with its shape and dtype.

### q694
A normal draw off a generator, reported with its shape and dtype.

## Integrated practice

### q704
One seeded stream feeding all four samplers, in order.

### q705
A seed fixes a whole sequence, not just the first draw.

### q706
A reproducible row shuffle driven entirely by a seed.
