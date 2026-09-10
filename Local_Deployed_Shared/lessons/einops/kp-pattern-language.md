---
kc: einops.pattern-language
title: The einops pattern language — naming and permuting axes
supporting: [numpy.reshape-flatten]
new_syntax: [einops.rearrange]
faded: [345, 941, 942, 943, 944]
guided: [388]
independent: [335, 330, 379, 319, 327, 344, 913, 914, 915, 916]
integrated: [945, 946, 947]
---

## Concept

`einops.rearrange(tensor, 'PATTERN')` is reshape/transpose with the axes
spelled out in words. The pattern is two axis lists around an arrow:

> `'b h w c -> b c h w'`
> — left side: name each input axis, in order. Right side: the same names,
> in the output's order.

einops names are whole
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

Reading discipline: identify each name's position
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
# Track one element: input (b=1, h=0, w=1, c=2) must land at (1, 2, 0, 1).

print("channels-last", tuple(arr.shape), "-> channels-first", tuple(first.shape))
# Hidden checks
assert first.shape == (2, 3, 2, 2)
assert arr[1, 0, 1, 2] == first[1, 2, 0, 1]
```

The image example moved the channel axis; the same grammar moves the time axis of a sequence in front of its batch axis.

```python
import torch as t
import einops

# Sequence layout swap: batch-first -> time-first.
seq = t.arange(12).reshape(2, 3, 2)         # (b, t, d)
tfirst = einops.rearrange(seq, 'b t d -> t b d')

print("batch-first", tuple(seq.shape), "-> time-first", tuple(tfirst.shape))
# Hidden checks
assert tfirst.shape == (3, 2, 2)
assert seq[1, 2, 0] == tfirst[2, 1, 0]
```

Finally, the pattern is an EXPECTATION about the input, and einops enforces it.

```python
import torch as t
import einops

# The pattern is checked against reality: wrong axis count = loud error.
try:
    einops.rearrange(seq, 'b h w c -> b c h w')   # 3-D data, 4-name pattern
    raised = False
except Exception as err:
    raised = True
    print("4-name pattern on 3-D data ->", type(err).__name__)

print("element (1,0,1,2) moved to (1,2,0,1):",
      arr[1, 0, 1, 2].item(), "==", first[1, 2, 0, 1].item())
# Hidden checks
assert raised
```

Why each step:

1. The tracked element (`arr[1,0,1,2] == first[1,2,0,1]`) is the same
   verification that works for every relayout — indices permute exactly as
   the names did. One element is a spot check of the mapping; a full
   `t.equal` against `arr.permute(0, 3, 1, 2)` is the proof.
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
    return einops._____(arr, '_____')
```

```python solution
import torch as t
import einops

def solve(arr):
    """(b, h, w, c) -> (b, c, h, w)."""
    return einops.rearrange(arr, 'b h w c -> b c h w')
```

### q941
The same idea on the smallest surface there is: a 2-D table, axes named in
words.

```python starter
import torch as t
import einops


def solve(table):
    """Exchange the two axes."""
    return einops._____(t.tensor(table), '_____').tolist()
```

```python solution
import torch as t
import einops


def solve(table):
    """Exchange the two axes."""
    return einops.rearrange(t.tensor(table), 'rows cols -> cols rows').tolist()
```

### q942
Three axes now, and the two that move are the OUTER and the INNER one.

```python starter
import torch as t
import einops


def solve(vol):
    """Outer and inner axes trade places."""
    return einops._____(t.tensor(vol), '_____').tolist()
```

```python solution
import torch as t
import einops


def solve(vol):
    """Outer and inner axes trade places."""
    return einops.rearrange(t.tensor(vol), 'd h w -> w h d').tolist()
```

### q943
Four axes, and the pair that moves is in the middle — the outermost and the
innermost stay exactly where they are.

```python starter
import torch as t
import einops


def solve(x):
    """Exchange the two middle axes."""
    return einops._____(
        t.tensor(x), '_____'
    ).tolist()
```

```python solution
import torch as t
import einops


def solve(x):
    """Exchange the two middle axes."""
    return einops.rearrange(
        t.tensor(x), 'batch heads seq dim -> batch seq heads dim'
    ).tolist()
```

### q944
Every axis moves. Read the input list right to left to write the output list.

```python starter
import torch as t
import einops


def solve(x):
    """Reverse the axis order."""
    return einops._____(t.tensor(x), '_____').tolist()
```

```python solution
import torch as t
import einops


def solve(x):
    """Reverse the axis order."""
    return einops.rearrange(t.tensor(x), 'a b c d -> d c b a').tolist()
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

Also from the bank: q319 (channels-first batch to channels-last — the
single most common layout flip), q327 ((h, w, c) to (h, c, w) — colour
lands between height and width), q344 (transpose H and W WITHIN each
channel).

## Integrated practice

### q945
The relaid batch and the shape that relaying it produced — the sizes travel with
their names.

### q946
The whole batch time-first, plus where one named element ended up.

### q947
Two hops in a row: each pattern names the layout it is HANDED, not the one the
data started in.

## Misconceptions

- **"einops names are single characters."** — They're
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
