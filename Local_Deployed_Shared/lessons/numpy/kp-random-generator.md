---
kc: numpy.random-generator
title: Random numbers with Generator
supporting: [numpy.constructors]
new_syntax: []
faded: [8]
guided: [42]
independent: [104]
---

## Concept

Modern NumPy randomness goes through a **`Generator`** object — you make one,
then ask *it* for numbers:

```python no-run
rng = np.random.default_rng(seed)   # seed optional; fixing it makes runs reproducible
rng.random(5)          # 5 uniform floats in [0, 1)
rng.random((3, 3))     # any shape, same convention as constructors
rng.integers(0, 10, size=4)   # random ints in [0, 10)
rng.normal(size=(2, 2))       # standard-normal draws
rng.permutation(n)     # a random shuffling of 0..n-1
rng.choice(x, size=k)  # sample from an existing array
```

The key mental model: a generator is a **deterministic stream**. Seeded with
the same value, it produces the same sequence forever; each draw consumes the
next chunk of the stream. That's why reproducible code (and graders, and
experiments) *passes the rng in as an argument* instead of creating one
internally: whoever owns the stream controls reproducibility. When a function
receives `rng`, use it directly — creating a new generator or reseeding inside
breaks the caller's stream.

You will also see the older global-state API in the wild (`np.random.seed`,
`np.random.random`) — a shared, module-level stream. It still works and some
bank drills use it, but for new code prefer the explicit `Generator`: no
hidden global state, no spooky interference between distant modules.

A pattern this unlocks immediately: **random structure via deterministic
building blocks** — e.g. a random permutation *matrix* is just the identity
matrix with its rows shuffled: `np.eye(n)[rng.permutation(n)]`.

## Worked example

Task: given a seeded generator, draw floats reproducibly; show that the same
seed replays the same stream and that consuming draws advances it.

```python
import numpy as np

# Two generators, same seed -> identical streams.
rng_a = np.random.default_rng(42)
rng_b = np.random.default_rng(42)

first_a = rng_a.random(3)
first_b = rng_b.random(3)
assert np.array_equal(first_a, first_b)   # same stream position, same values
assert first_a.shape == (3,)
assert ((0 <= first_a) & (first_a < 1)).all()   # uniform in [0, 1)

# Each draw CONSUMES stream: the next request continues where we left off.
second_a = rng_a.random(3)
assert not np.array_equal(first_a, second_a)

# A function that receives a generator must use it as given —
# this is what "use the next n draws from rng" means in the drills.
def noisy_zeros(rng, n):
    return np.zeros(n) + rng.random(n)

out1 = noisy_zeros(np.random.default_rng(7), 4)
out2 = noisy_zeros(np.random.default_rng(7), 4)
assert np.array_equal(out1, out2)          # reproducible: caller owns the seed
```

Why each step:

1. Seeding twice with 42 and comparing draws makes "deterministic stream"
   concrete — randomness in NumPy is repeatable on demand.
2. The consumed-stream check explains grader behavior: a drill that hands you
   `rng` has pre-computed what the *next* draws will be. Reseeding or making
   your own generator inside `solve` produces different numbers and fails.
3. `noisy_zeros` is the shape of every rng-taking function you'll write:
   thread the generator through, never create one mid-function.

## Faded practice

### q8
The next n uniform floats from a generator you are handed.

```python starter
import numpy as np

def solve(rng, n):
    """Return the next n uniform [0,1) floats from rng's stream."""
    return rng._____(n)
```

```python solution
import numpy as np

def solve(rng, n):
    """Return the next n uniform [0,1) floats from rng's stream."""
    return rng.random(n)
```

## Guided practice

### q42
1. A random array of a given 3-D shape, uniform in [0, 1) — this drill uses
   the older module-level API (`np.random.*`).
2. Both APIs share the shape convention with constructors: pass the whole
   tuple.
3. `np.random.random(shape)` — the grader checks shape, dtype, range, and that
   values vary (so no constant arrays).

## Independent practice

From the drill bank: q104 (random permutation MATRIX — combine `np.eye` with
`rng.permutation` as row selector).

## Misconceptions

- **"I should reseed inside my function for safety."** — The opposite: a
  function that receives `rng` must use the caller's stream. Reseeding or
  creating a new generator inside breaks reproducibility (and fails graders
  that check the exact stream).
- **"`rng.random(3)` might include 1.0."** — The interval is half-open
  [0, 1): 0 is possible, 1 is not. Scale/shift for other ranges.
- **"Same code, same randomness."** — Only with the same SEED and the same
  DRAW ORDER. Any extra draw in between shifts everything after it; that's
  what "each draw consumes the stream" means.
