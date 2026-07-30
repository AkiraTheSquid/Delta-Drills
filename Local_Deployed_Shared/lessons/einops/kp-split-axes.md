---
kc: einops.split-axes
title: Splitting axes with named factors
supporting: [einops.merge-axes]
new_syntax: []
faded: [390]
guided: [315]
independent: [337, 320, 393, 331]
---

## Concept

Parentheses on the **input** side SPLIT an axis into factors — the exact
inverse of merging:

> `'c (h w) -> c h w', h=h`
> — the length-h·w axis is declared to be h blocks of w, and unpacked.

The new ingredients:

1. **You must tell einops the factor sizes it can't infer.** An axis of
   length 12 could be (h=2, w=6), (3, 4), (4, 3)… — so sizes arrive as
   keyword arguments. Give any factors such that the rest are forced —
   for a two-way split, ONE keyword suffices (`h=3` fixes w = 12/3).
2. **Order inside the parens declares how the axis was PACKED** — left
   slow, right fast, same convention as merging. `(h w)` says "this axis
   is h blocks, each of length w". Splitting with the wrong order doesn't
   error (sizes may still divide) — it unpacks garbage. The task's
   description of how the data was laid out ("row-major tiles", "group
   index slowest/fastest") is the ground truth for the order.
3. **Split and merge combine in one pattern** — the signature einops move.
   `'(b w) ... -> b ...'` unpacks; `'... -> ... (h p)'` repacks;
   `'(h w) p1 p2 c -> (h p1) (w p2) c'` does both at once (that one
   reassembles an image from its tile stack — split the tile index into
   grid coordinates, then merge each with its within-tile axis).

Reading a split-merge pattern: first find every parenthesized group on the
left (what gets unpacked, and in what packing order), then on the right
(what gets packed). The names in the middle just carry through.

## Worked example

Task: restore a flattened image given its height; split a sequence into
chunks; reassemble tiles into an image.

```python
import torch as t
import einops

# Round-trip: flatten (merge), then restore (split) with one known factor.
img = t.arange(12).reshape(3, 2, 2)                  # (c, h, w)
flat = einops.rearrange(img, 'c h w -> c (h w)')      # (3, 4)
back = einops.rearrange(flat, 'c (h w) -> c h w', h=2)
assert t.equal(back, img)                       # perfect inverse

# Split a sequence into p-token segments: t = n segments of length p.
seq = t.arange(24).reshape(2, 6, 2)                  # (b, t=6, d)
chunks = einops.rearrange(seq, 'b (n p) d -> b n p d', p=3)
assert chunks.shape == (2, 2, 3, 2)
assert t.equal(chunks[0, 0], seq[0, :3])       # first 3 tokens

# Split AND merge at once: tile stack -> image.
# 6 tiles of shape (2, 2), listed row-major from a (2x3)-tile image.
tiles = t.arange(24).reshape(6, 2, 2)                # ((h w), p1, p2)
image = einops.rearrange(tiles, '(h w) p1 p2 -> (h p1) (w p2)', h=2)
assert image.shape == (4, 6)                          # (2*2, 3*2)
# Tile 0 occupies the top-left 2x2 block:
assert image[:2, :2].tolist() == tiles[0].tolist()
```

Why each step:

1. The round-trip (`back == img`) is the defining property of split-as-
   inverse-of-merge, and doubles as your self-test recipe: whenever a split
   pattern feels shaky, merge it back and compare.
2. In the chunking, `p=3` (not n) is given — either works, and choosing the
   one the task names ("segments of length p") keeps the code aligned with
   the prose.
3. The tile reassembly deserves slow reading: `(h w)` unpacks the tile
   index into grid row/column (row-major, hence h slow); then `(h p1)`
   merges grid-row with within-tile-row. Two coordinate systems zipped
   together — one pattern, no loops, no arithmetic on indices.

## Faded practice

### q390
Restore (c, h·w) to (c, h, w), given h.

```python starter
import torch as t
import einops

def solve(flat, h):
    """Undo the per-channel flatten: declare how the axis was packed."""
    return einops.rearrange(flat, '_____', h=h)
```

```python solution
import torch as t
import einops

def solve(flat, h):
    """Undo the per-channel flatten: declare how the axis was packed."""
    return einops.rearrange(flat, 'c (h w) -> c h w', h=h)
```

## Guided practice

### q315
1. (b, t, d) with t divisible by p → (b, t/p, p, d): the time axis splits
   into (segments × length-p).
2. Which factor does the task hand you? Pass it as the keyword.
3. `'b (n p) d -> b n p d', p=p` — n is inferred.

## Independent practice

From the drill bank: q337 (split b images back OUT of a side-by-side strip
(h, b·w, c) — which packing order was used?), q320 (halve each image's
height into two batch entries — split then merge into b), q393 (split an
even channel axis into pairs, pair-member axis to the front).

Also from the bank: q331 (slice to the even-indexed images, then tile them
into an r-row grid).

## Misconceptions

- **"einops can infer both factors of a split."** — It can infer ONE
  (total ÷ known); the rest are yours to supply as keywords. No keyword,
  no split — the error message will list what's missing.
- **"Wrong paren order in a split will error."** — Only if the sizes fail
  to divide. `(h w)` vs `(w h)` with square-ish factors both "work" and one
  is silently scrambled. The packing order comes from how the data was
  BUILT — reread the task's layout description, then round-trip-test.
- **"Tile reassembly needs index arithmetic."** — The split-merge pattern
  `('(h w) p1 p2 -> (h p1) (w p2)')` IS the index arithmetic, stated
  declaratively. If you're computing offsets by hand around einops, the
  pattern can probably absorb the work.
