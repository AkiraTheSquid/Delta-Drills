---
kc: numpy.random-generator
title: Random numbers with a Generator
supporting: [numpy.constructors]
new_syntax: []
faded: [8]
guided: [42]
independent: [104]
---

## Concept

Reproducible randomness in PyTorch goes through a **`Generator`** object — you
make one, then hand it to the sampling functions:

```python no-run
rng = t.Generator().manual_seed(seed)   # fixing the seed makes runs reproducible
t.rand(5, generator=rng)                # 5 uniform floats in [0, 1)
t.rand((3, 3), generator=rng)           # any shape, same convention as constructors
t.randint(0, 10, (4,), generator=rng)   # random ints in [0, 10)
t.randn((2, 2), generator=rng)          # standard-normal draws
t.randperm(n, generator=rng)            # a random shuffling of 0..n-1
```

```python
import torch as t

rng = t.Generator().manual_seed(0)
print("rand    ", t.rand(3, generator=rng))
print("randint ", t.randint(0, 10, (5,), generator=rng))
print("randn   ", t.randn(3, generator=rng))
print("randperm", t.randperm(6, generator=rng))
```

Note the shape of the API: unlike NumPy, where you call *methods on* the
generator (`rng.random(5)`), in PyTorch the generator is an **argument** to a
normal function (`t.rand(5, generator=rng)`). Same idea, inverted spelling.

The key mental model: a generator is a **deterministic stream**. Seeded with
the same value, it produces the same sequence forever; each draw consumes the
next chunk of the stream. That's why reproducible code (and graders, and
experiments) *passes the rng in as an argument* instead of creating one
internally: whoever owns the stream controls reproducibility. When a function
receives `rng`, use it directly — creating a new generator or reseeding inside
breaks the caller's stream.

```python
a = t.Generator().manual_seed(42)
b = t.Generator().manual_seed(42)
print("a:", t.rand(3, generator=a))
print("b:", t.rand(3, generator=b))

# a has now advanced three draws; b is replayed from the start.
print("a again:", t.rand(3, generator=a))
assert not t.equal(t.rand(3, generator=a), t.rand(3, generator=b))
```

The second line of that output is the whole reproducibility contract, and the
last assert is why an *extra* draw slipped in anywhere shifts every value
after it.

You will also see the **global** API in the wild (`t.manual_seed(0)`, then
plain `t.rand(shape)` with no generator) — a shared, process-level stream. It
still works and some bank drills use it, but for anything you need to
reproduce, prefer the explicit `Generator`: no hidden global state, no spooky
interference between distant modules.

A pattern this unlocks immediately: **random structure via deterministic
building blocks** — e.g. a random permutation *matrix* is just the identity
matrix with its rows shuffled: `t.eye(n)[t.randperm(n, generator=rng)]`.

```python
rng = t.Generator().manual_seed(1)
order = t.randperm(4, generator=rng)
perm = t.eye(4)[order]
print("order", order)
print(perm)

# Every row and every column still has exactly one 1 — that is what makes it
# a permutation matrix rather than just a random 0/1 grid.
assert t.equal(perm.sum(dim=0), t.ones(4))
assert t.equal(perm.sum(dim=1), t.ones(4))
```

## Worked example

Task: given a seeded generator, draw floats reproducibly; show that the same
seed replays the same stream and that consuming draws advances it.

```python
import torch as t

# Two generators, same seed -> identical streams.
rng_a = t.Generator().manual_seed(42)
rng_b = t.Generator().manual_seed(42)

first_a = t.rand(3, generator=rng_a)
first_b = t.rand(3, generator=rng_b)
assert t.equal(first_a, first_b)          # same stream position, same values
assert first_a.shape == (3,)
assert bool(((0 <= first_a) & (first_a < 1)).all())   # uniform in [0, 1)

# Each draw CONSUMES stream: the next request continues where we left off.
second_a = t.rand(3, generator=rng_a)
assert not t.equal(first_a, second_a)

# A function that receives a generator must use it as given —
# this is what "use the next n draws from rng" means in the drills.
def noisy_zeros(rng, n):
    return t.zeros(n) + t.rand(n, generator=rng)

out1 = noisy_zeros(t.Generator().manual_seed(7), 4)
out2 = noisy_zeros(t.Generator().manual_seed(7), 4)
assert t.equal(out1, out2)                # reproducible: caller owns the seed
```

Why each step:

1. Seeding twice with 42 and comparing draws makes "deterministic stream"
   concrete — randomness here is repeatable on demand.
2. The consumed-stream check explains grader behavior: a drill that hands you
   `rng` has pre-computed what the *next* draws will be. Reseeding or making
   your own generator inside `solve` produces different numbers and fails.
3. `noisy_zeros` is the shape of every rng-taking function you'll write:
   thread the generator through, never create one mid-function.

## Faded practice

### q8
The next n uniform floats from a generator you are handed.

```python starter
import torch as t

def solve(rng, n):
    """Return the next n uniform [0,1) floats from rng's stream."""
    return t.rand(n, _____=rng)
```

```python solution
import torch as t

def solve(rng, n):
    """Return the next n uniform [0,1) floats from rng's stream."""
    return t.rand(n, generator=rng)
```

## Guided practice

### q42
1. A random tensor of a given 3-D shape, uniform in [0, 1) — this drill uses
   the global stream, so no generator argument is needed.
2. The sampling functions share the shape convention with constructors: pass
   the whole tuple.
3. `t.rand(shape)` — the grader checks shape, dtype, range, and that values
   vary (so no constant tensors).

## Independent practice

From the drill bank: q104 (random permutation MATRIX — combine `t.eye` with
`t.randperm` as row selector).

## Misconceptions

- **"I should reseed inside my function for safety."** — The opposite: a
  function that receives `rng` must use the caller's stream. Reseeding or
  creating a new generator inside breaks reproducibility (and fails graders
  that check the exact stream).
- **"`rng.rand(3)` — the generator has the sampling methods."** — It does not.
  The generator is passed *to* `t.rand(...)` as `generator=`. Its own methods
  are about seed and state.
- **"`t.rand(3)` might include 1.0."** — The interval is half-open
  [0, 1): 0 is possible, 1 is not. Scale/shift for other ranges.
- **"Same code, same randomness."** — Only with the same SEED and the same
  DRAW ORDER. Any extra draw in between shifts everything after it; that's
  what "each draw consumes the stream" means.
