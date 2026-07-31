---
kc: einops.channel-groups-temporal
title: Channel groups and temporal windows
supporting: [einops.dl-flatten-heads, einops.pooling]
new_syntax: []
faded: [362]
guided: [375]
independent: [369, 372, 378, 397, 316, 359]
---

## Concept

The capstone patterns: channel axes and time axes that carry HIDDEN
STRUCTURE, unpacked and repacked with everything this course built. Two
families:

**Grouped channels.** A channel axis of size g·c often means "g groups of c
channels" — and the single load-bearing question is **which index is slow**:

- **Group SLOWEST** (`(g c)`): channels are g0c0, g0c1, …, g1c0… — whole
  groups in blocks. Splitting groups out: `'b (g c) h w -> g b c h w', g=G`.
- **Group FASTEST** (`(c g)`): channels INTERLEAVE — g0c0, g1c0, g2c0,
  g0c1… Splitting the same tensor: `'b (c g) h w -> g b c h w', g=G`.

Identical shapes, opposite layouts. The drills alternate between the two
precisely to force the read: find the task's phrase ("group index
slowest" / "channels interleave") and place the factor accordingly.
Group operations then compose: swap two group factors
(`'b (g1 g2 c) h w -> b (g2 g1 c) h w'`), transpose groups against
within-group channels (`(g c) -> (c g)`) — all one-pattern moves.

**Temporal windows.** Sequences (b, c, t) pool and chunk exactly like
spatial axes: average non-overlapping PAIRS of time steps —
`reduce('b c (t two) -> b c t', 'mean', two=2)`; length-w windows —
`two→w`. Multi-head sequence packing, chunked attention, and space-time
tensors are these same factored axes on the time dimension.

Nothing in this KP is new grammar — it is fluency under pressure: rank-4/5
tensors, two conventions in play, and the pattern as the single place the
truth lives. This is the level ARENA's einops exercises (and real model
surgery) operate at.

## Worked example

Task: split interleaved channel groups; swap group blocks; average-pool
time pairs.

```python
import torch as t
import einops

# 6 channels = 3 channels x 2 groups, GROUP INDEX FASTEST (interleaved):
# channel order is g0c0, g1c0, g0c1, g1c1, g0c2, g1c2.
x = t.arange(12.0).reshape(1, 6, 1, 2)          # (b, 6, h, w)

split = einops.rearrange(x, 'b (c g) h w -> g b c h w', g=2)
assert split.shape == (2, 1, 3, 1, 2)
# Group 0 must hold original channels 0, 2, 4 (every second one):
assert t.equal(split[0, 0, :, 0, 0], x[0, ::2, 0, 0])

# Same data read as GROUP SLOWEST would give a different (wrong here) split:
wrong = einops.rearrange(x, 'b (g c) h w -> g b c h w', g=2)
assert not t.equal(split, wrong)          # conventions matter!

# Temporal pooling: average adjacent pairs of time steps.
seq = t.arange(8.0).reshape(1, 1, 8)            # (b, c, t=8)
halved = einops.reduce(seq, 'b c (t two) -> b c t', 'mean', two=2)
assert halved[0, 0].tolist() == [0.5, 2.5, 4.5, 6.5]
print("channels        ", x[0, :, 0, 0])
print("'(c g)' group 0 ", split[0, 0, :, 0, 0], " <- every second channel")
print("'(g c)' group 0 ", wrong[0, 0, :, 0, 0], " <- a contiguous half")
print("seq", seq[0, 0], "-> pairwise mean", halved[0, 0])
```

Why each step:

1. The interleaved split's check (`group 0 == channels 0, 2, 4`) is the
   ground truth for "(c g) with g fast": stride-2 selection. If the task
   had said "group index slowest", group 0 would be channels 0, 1, 2 —
   the `wrong` variant. One assert distinguishes them.
2. Running BOTH readings on the same data — and seeing them differ — is
   the inoculation this KP exists for: shape-compatible ≠ correct, and
   only the task's layout sentence decides.
3. The temporal pooling is spatial pooling with the axis renamed — placed
   here to make the transfer explicit: axes are axes; the pattern language
   doesn't care whether they're pixels or time steps.

## Faded practice

### q362
Split out interleaved channel groups (group index FASTEST).

```python starter
import torch as t
import einops

def solve(x, split):
    """(b, c*split, h, w), groups interleaved -> (split, b, c, h, w)."""
    return einops.rearrange(x, 'b (_____) h w -> split b c h w', split=split)
```

```python solution
import torch as t
import einops

def solve(x, split):
    """(b, c*split, h, w), groups interleaved -> (split, b, c, h, w)."""
    return einops.rearrange(x, 'b (c split) h w -> split b c h w', split=split)
```

## Guided practice

### q375
1. (b, c, 2t) downsampled by averaging non-overlapping PAIRS — temporal
   pooling with window 2.
2. The time axis factors as (t × 2); the window name reduces away.
3. `einops.reduce(x_seq, 'b c (t two) -> b c t', 'mean', two=2)`.

## Independent practice

From the drill bank: q369 (swap two group factors g1, g2 within the channel
axis — both stay packed), q372 (transpose groups against within-group
channels: (g c) → (c g)), q378 (average-pool time with window length w),
q397 (spatial axes leading, batch and channels merged with batch slowest —
a pure but rank-heavy rearrange).

Also from the bank: q316 (channel axis interleaves `coord` groups of k,
group index SLOWEST — split the group out and move it to the front), q359
(same split, phrased for unpacking: `part1, part2 = solve(x, 2)`).

## Misconceptions

- **"(g c) and (c g) differ only in variable naming."** — They are
  opposite memory layouts: blocks vs interleave. Same shape, different
  tensor. The task's "slowest/fastest/interleaved" sentence is the spec —
  translate it to paren order before writing anything.
- **"Group tricks need index arithmetic over channel numbers."** — Every
  grouped-channel task in the bank is one rearrange with the right
  factoring. Hand-computed channel indices are the sign you're fighting
  the notation.
- **"Time axes are special."** — To the pattern language a time axis is an
  axis. Pooling, chunking, windowing transfer verbatim from the spatial
  versions — reuse the pattern, rename the letters.
