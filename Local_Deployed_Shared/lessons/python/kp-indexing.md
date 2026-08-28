---
kc: python.indexing
title: Indexing — pulling one item out, counting from zero
supporting: [python.lists-and-tuples]
new_syntax: [syntax.index, syntax.negative-index, syntax.nested-index]
faded: [586, 587]
guided: []
independent: [588, 589]
integrated: [590, 591]
---

## Concept

Square brackets after a container mean **"give me the item at this position"**.
Positions are counted from **zero**, so the first item is at 0 and the last is at
`len(...) - 1`:

```python
letters = ["a", "b", "c"]

print(letters[0])
print(letters[1])
print(letters[2])
print("length is", len(letters), "so the last position is", len(letters) - 1)
```

Counting from zero is the single most common source of early off-by-one errors,
and it is worth saying out loud: `letters[1]` is the SECOND item.

Because "the last one" is needed constantly, negative positions count backwards
from the end. `-1` is the last item, `-2` the one before it:

```python
letters = ["a", "b", "c", "d"]

print(letters[-1])
print(letters[-2])
```

Negative indexing is not just shorter than `letters[len(letters) - 1]` — it is
safer, because there is no length to get wrong.

Reaching into nested data is **one index per level**, read left to right. The
first bracket picks the row; the second picks inside that row:

```python
rows = [[1, 2, 3], [4, 5, 6]]

print(rows[0])
print(rows[0][2])
print(rows[1][0])
```

`rows[0]` is itself a list, so `rows[0][2]` is "the third item of the first
row". The two indices are not interchangeable: `rows[2][0]` asks for a third row
that does not exist.

The two styles combine, which is how you name a corner of a table without
measuring anything:

```python
rows = [[1, 2, 3], [4, 5, 6]]

print("top-left    :", rows[0][0])
print("top-right   :", rows[0][-1])
print("bottom-right:", rows[-1][-1])
```

Asking for a position that does not exist stops the program with an
`IndexError` rather than guessing:

```python
letters = ["a", "b"]
try:
    print(letters[5])
except IndexError as exc:
    print("IndexError:", exc)
```

## Watch out

- **The first item is at 0** — so the last valid position is one LESS than the
  length. `items[len(items)]` is always one past the end.
- **`-1` is the last item, not "one before the start"** — the negative side
  starts at `-1`, there is no `-0`.
- **The first index is the row** — `rows[i][j]` picks row `i`, then item `j`
  inside it. Swapping them reads a different value, or fails.

## Worked example

One row of a table, pulled out and then read from both ends.

```python
rows = [[10, 20, 30], [40, 50]]

row = rows[1]

print("the row itself :", row)
print("how many items :", len(row))
print("its first item :", row[0])
print("its last item  :", row[-1])
print("not the same as the last ROW:", rows[-1])
```

Why each step:

1. `rows[1]` names the second row, so the reads below are all one level down and
   do not need a second bracket every time.
2. `row[0]` and `row[-1]` are the two ends of THAT row.
3. The last line is the trap in one line: `rows[-1]` is the last row of the
   whole table, which is a different depth from `row[-1]`.

## Faded practice

### q586
The first item lives at position zero.

```python starter
def solve(items):
    """First item — position zero."""
    return items[_____]
```

```python solution
def solve(items):
    """First item — position zero."""
    return items[0]
```

### q587
The last item, counted back from the end rather than measured.

```python starter
def solve(items):
    """Last item — count back from the end."""
    return items[_____]
```

```python solution
def solve(items):
    """Last item — count back from the end."""
    return items[-1]
```

## Solo practice

### q588
One index per level: pick the row, then pick inside it.

The order of the two brackets is the whole answer here — the first one always
chooses which inner list you are reading:

```python worked
rows = [["a", "b"], ["c", "d"], ["e", "f"]]

print("row 1        :", rows[1])
print("row 1, item 0:", rows[1][0])
print("row 0, item 1:", rows[0][1])
```

### q589
Both ends of the same list, using a negative position for the back two.

## Integrated practice

### q590
Pull one row out, then measure it and read its last item.

### q591
Two opposite corners of a table, plus the shape they sit in.

## Misconceptions

- **"`items[1]` is the first item."** — It is the second. Counting starts at 0.
- **"To get the last item I need the length."** — `items[-1]` does it with no
  measurement, and cannot go one past the end.
- **"`rows[i][j]` and `rows[j][i]` are the same."** — Only on a square table, and
  only by coincidence. The first index always picks the row.
