---
kc: einops.pattern-language
title: The einops pattern language — naming and permuting axes
supporting: [numpy.reshape-flatten, einsum.notation-model]
new_syntax: [einops-pattern-string]
faded: [345]
guided: [388]
independent: [335, 330, 379]
---

## Concept

`einops.rearrange(tensor, 'PATTERN')` is reshape/transpose with the axes
spelled out in words. The pattern is two axis lists around an arrow:

> `'b h w c -> b c h w'`
> — left side: name each input axis, in order. Right side: the same names,
> in the output's order.

Unlike einsum's one-letter subscripts, einops names are whole
space-separated WORDS (`batch`, `h`, `nh`), and — the big semantic
difference — **every name on the left must appear on the right** (rearrange
never sums; reducing is a different function, later KP). What rearrange
does is exactly what the name-shuffle says:

- `'b h w c -> b c h w'` — channels-last to channels-first: axis `c` moves
  to position 1, values untouched.
- `'b t d -> t b d'` — batch-first to time-first.
- `'h w -> w h'` — a 2-D transpose.

Why this beats `x.permute(0, 3, 1, 2)`: the pattern is
self-verifying documentation. It states what each axis MEANS, the library
checks that the input really has 4 axes, and six months later the intent is
still legible. In deep-learning code, layout bugs (bhwc vs bchw) are among
the most common and least visible — naming the axes at every hop is the
antidote, which is why the ARENA curriculum drills einops before touching
models.

Reading discipline (same ritual as einsum): identify each name's position
on the left (what it is) and on the right (where it goes). If a name
appears exactly once per side, the operation is a pure permutation — data
moves, nothing merges, splits, or disappears. Merging and splitting add
parentheses to this grammar — next two KPs.

## Worked example

Task: channels-last batch → channels-first, and batch-first sequence →
time-first — with one-element verification.

```python
import torch as t
import einops

arr = t.arange(24).reshape(2, 2, 2, 3)      # (b, h, w, c) channels-LAST

# Name the four axes; emit them with c pulled to the front block.
first = einops.rearrange(arr, 'b h w c -> b c h w')
assert first.shape == (2, 3, 2, 2)
# Track one element: input (b=1, h=0, w=1, c=2) must land at (1, 2, 0, 1).
assert arr[1, 0, 1, 2] == first[1, 2, 0, 1]

# Sequence layout swap: batch-first -> time-first.
seq = t.arange(12).reshape(2, 3, 2)         # (b, t, d)
tfirst = einops.rearrange(seq, 'b t d -> t b d')
assert tfirst.shape == (3, 2, 2)
assert seq[1, 2, 0] == tfirst[2, 1, 0]

# The pattern is checked against reality: wrong axis count = loud error.
try:
    einops.rearrange(seq, 'b h w c -> b c h w')   # 3-D data, 4-name pattern
    raised = False
except Exception:
    raised = True
assert raised
```

Why each step:

1. The tracked element (`arr[1,0,1,2] == first[1,2,0,1]`) is the same
   verification you used for einsum relayouts — indices permute exactly as
   the names did. One element proves the whole mapping.
2. Note what the names buy in the sequence example: 'b t d -> t b d' READS
   as "time first"; the transpose-tuple spelling `(1, 0, 2)` says the same
   thing to the machine and nothing to the reader.
3. The deliberate error shows einops as a shape CHECKER: patterns carry
   expectations, and mismatches fail at the call — not three functions
   later. This is a feature to lean on, not an annoyance.

## Faded practice

### q345
Channels-last batch to channels-first.

```python starter
import torch as t
import einops

def solve(arr):
    """(b, h, w, c) -> (b, c, h, w)."""
    return einops.rearrange(arr, '_____')
```

```python solution
import torch as t
import einops

def solve(arr):
    """(b, h, w, c) -> (b, c, h, w)."""
    return einops.rearrange(arr, 'b h w c -> b c h w')
```

## Guided practice

### q388
1. (b, t, d) to time-first (t, b, d) — name the three axes, reorder two of
   them.
2. All names appear on both sides — a pure permutation.
3. `'b t d -> t b d'`.

## Independent practice

From the drill bank: q335 (channels-last image → channels-first — one image,
no batch axis), q330 (move the batch axis to the END), q379 (swap height and
width within each image of a batch).

## Misconceptions

- **"einops names are single characters like einsum."** — They're
  space-separated words: `'batch height width channels -> ...'` is legal and
  sometimes clearest. The space is the separator; 'bhwc' would be ONE axis
  named bhwc.
- **"rearrange can drop an axis I don't need."** — Every input name must
  appear in the output; rearrange is lossless by design. Dropping = summing
  or selecting, which are reduce (later KP) or plain indexing.
- **"It's just transpose with extra steps."** — It's transpose PLUS shape
  verification PLUS documentation. The pattern fails loudly when the input
  doesn't match the declared layout — the check you didn't know you needed
  until a bhwc/bchw bug eats an afternoon.
