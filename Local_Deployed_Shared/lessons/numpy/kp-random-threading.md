---
kc: numpy.random-threading
title: Using a stream you were handed
supporting: [numpy.random-seeding, numpy.random-samplers, numpy.ndarray-model]
new_syntax: []
previews: []
faded: [8, 687]
guided: []
independent: [688, 689, 690, 691, 692, 693]
integrated: [104, 695, 707]
---

## Concept

You can now build a stream and thread it into any sampler. This concept is the
other half, and it is the half that appears in real code far more often: a
function is **handed** an `rng` and has to use *that one*.

Nothing new to spell here — every symbol below came from the previous two
concepts. What is new is a rule:

> Use the generator you were given. Do not make one. Do not reseed it.

```python
import torch as t

def good(rng, n):
    return t.rand(n, generator=rng)      # threads the caller's stream

def ignores(rng, n):
    return t.rand(n)                     # silently uses the GLOBAL stream

def rewinds(rng, n):
    rng.manual_seed(0)                   # silently rewinds the CALLER's stream
    return t.rand(n, generator=rng)

shared = t.Generator().manual_seed(7)
print("threaded:", good(shared, 3))
print("the caller's stream has now advanced by three draws")
```

Both wrong versions **run fine and return plausible numbers**. Nothing raises.
That is what makes this worth its own concept: the failure has no symptom at
the call site, only wrong values somewhere downstream.

Why it matters is ownership. A generator is a deterministic stream, and each
draw consumes the next piece of it, so whoever holds the generator controls
what every function that receives it will see:

```python
import torch as t

def draw(rng, n):
    return t.rand(n, generator=rng)

# The caller seeds once and calls twice. The two results differ because the
# stream MOVED — not because anything is random about `draw` itself.
rng = t.Generator().manual_seed(0)
first = draw(rng, 3)
second = draw(rng, 3)
print("first :", first)
print("second:", second)
assert first.tolist() != second.tolist()

# And because `draw` threaded the stream instead of making its own, the caller
# can reproduce the whole run from the seed alone.
replay = draw(t.Generator().manual_seed(0), 3)
assert replay.tolist() == first.tolist()
print("replay:", replay)
```

That last assert is the contract a grader checks. A drill that hands you `rng`
has already computed what your draw must be; reseeding or building your own
produces different numbers and fails, with nothing in the error message about
generators.

The generator only reaches the sampler. Everything you build from the draw is
ordinary tensor work:

```python
import torch as t

rng = t.Generator().manual_seed(1)
order = t.randperm(4, generator=rng)   # random part: needs the stream
perm = t.eye(4)[order]                 # ordinary part: does not
print("order:", order)
print(perm)
```

## Watch out

- **Dropping `generator=` is silent.** `t.rand(n)` runs fine and returns
  plausible numbers — from the wrong stream. Nothing errors; the grader just
  reports different values.
- **Do not reseed a generator you were handed.** The caller owns the stream;
  reseeding it inside your function silently rewinds their sequence.
- **Making your own generator "just to be safe" is the unsafe move.** It is the
  one thing guaranteed to produce numbers the caller cannot reproduce.

## Worked example

Task: draw from a generator you were handed, show that the stream advances, and
show that threading is what makes the result the caller's rather than yours.

```python
import torch as t

def next_uniforms(rng, n):
    """The next n uniform floats from rng's stream — the caller's stream."""
    return t.rand(n, generator=rng)

# The caller owns the seed, so the caller can predict what comes out.
rng = t.Generator().manual_seed(42)
first = next_uniforms(rng, 3)
print("first draw :", first)

# The same generator, a second call: the stream has MOVED ON.
second = next_uniforms(rng, 3)
print("second draw:", second)
assert first.tolist() != second.tolist()

# Every sampler takes the same keyword, off the same stream, in call order.
ints = t.randint(0, 10, (3,), generator=rng)
perm = t.randperm(4, generator=rng)
print("ints:", ints, " perm:", perm)

# And threading is what makes it reproducible for the caller: a fresh generator
# on the same seed replays the first draw exactly.
replay = next_uniforms(t.Generator().manual_seed(42), 3)
assert replay.tolist() == first.tolist()
print("replayed   :", replay)
```

Why each step:

1. `next_uniforms` is the shape of every rng-taking function you will write:
   the generator comes in, goes straight to the sampler, and nothing else
   touches it.
2. The second call proves the stream is consumed, which is why a drill that
   hands you `rng` has already computed what your draw must be.
3. The replay at the end is the payoff: because the function threaded the
   generator instead of making one, the caller can reproduce the result.

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

### q687
The same move on the normal sampler, returned as a plain list.

```python starter
import torch as t

def solve(rng, n):
    """Return the next n standard-normal floats from rng's stream."""
    return t.randn(n, _____=rng).tolist()
```

```python solution
import torch as t

def solve(rng, n):
    """Return the next n standard-normal floats from rng's stream."""
    return t.randn(n, generator=rng).tolist()
```

## Solo practice

### q688
Random integers in a given range, off the stream you were handed.

### q689
A permutation of 0..n-1, off the stream you were handed.

### q690
Two consecutive draws from one stream, and the proof they differ.

### q691
The next n uniform floats, leaving the stream otherwise undisturbed.

### q692
Noise added to a row of zeros, drawn from the caller's stream.

### q693
Shuffle a tensor's rows with a permutation drawn from the caller's stream.

## Integrated practice

### q104
A random permutation MATRIX built from a generator you are handed.

### q695
All four samplers off the SAME stream, in a fixed order.

### q707
A permutation matrix plus the order that produced it, checked for validity.
