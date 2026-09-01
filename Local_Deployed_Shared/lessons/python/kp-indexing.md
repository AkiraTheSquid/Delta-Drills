---
kc: python.indexing
title: Indexing — pulling one item out, counting from zero
supporting: [python.lists-and-tuples]
new_syntax: [syntax.index, syntax.negative-index, syntax.nested-index]
faded: [586, 587, 762, 763, 764, 765, 766, 767, 768, 769]
guided: []
independent: [588, 589, 770, 771, 772, 773, 774, 775, 776, 777]
integrated: [590, 591, 778, 779]
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

### q762
Counting starts at zero — the second item is at position 1.

```python starter
def solve(items):
    """items[1] is the SECOND item."""
    return (items[_____], items[_____])
```

```python solution
def solve(items):
    """items[1] is the SECOND item."""
    return (items[1], items[2])
```

### q763
The last valid position is one LESS than the length.

```python starter
def solve(items):
    """The last position is len minus one."""
    last = items[len(items) - _____]
    return last
```

```python solution
def solve(items):
    """The last position is len minus one."""
    last = items[len(items) - 1]
    return last
```

### q764
Read one end from zero, the other from minus one, and compare.

```python starter
def solve(items):
    """Do the two ends match?"""
    return items[_____] == items[_____]
```

```python solution
def solve(items):
    """Do the two ends match?"""
    return items[0] == items[-1]
```

### q765
The first bracket picks the row; the second reads inside it.

```python starter
def solve(rows):
    """Pick the row, then read inside it."""
    row = rows[_____]
    return row[_____]
```

```python solution
def solve(rows):
    """Pick the row, then read inside it."""
    row = rows[0]
    return row[-1]
```

### q766
Bottom-left: the last row first, then its first item.

```python starter
def solve(rows):
    """Last row first, then its first item."""
    return rows[_____][_____]
```

```python solution
def solve(rows):
    """Last row first, then its first item."""
    return rows[-1][0]
```

### q767
Tuples index exactly like lists — and stay tuples.

```python starter
def solve(point):
    """Tuples index exactly like lists."""
    return (point[_____], point[_____], type(point).__name__)
```

```python solution
def solve(point):
    """Tuples index exactly like lists."""
    return (point[0], point[1], type(point).__name__)
```

### q768
One step further back than len minus one.

```python starter
def solve(items):
    """One step further back than len minus one."""
    mid = len(items) - _____
    return items[mid]
```

```python solution
def solve(items):
    """One step further back than len minus one."""
    mid = len(items) - 2
    return items[mid]
```

### q769
-1 and len-1 are two roads to the same item.

```python starter
def solve(items):
    """-1 and len-1 name the same position."""
    return (items[_____], items[len(items) - 1])
```

```python solution
def solve(items):
    """-1 and len-1 name the same position."""
    return (items[-1], items[len(items) - 1])
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

### q770
Same column, opposite rows — the outer bracket is the one that changes.

### q771
One step in from each end.

### q772
One pick, then both ends of what was picked.

### q773
Swapping the brackets reads a different cell.

### q774
Row zero, last item — and what kind of value lives there.

### q775
Flip a pair by indexing, then index what you built.

### q776
A position and the one before it — the arithmetic happens inside the brackets.

### q777
Three levels deep: three brackets each way.

## Integrated practice

### q590
Pull one row out, then measure it and read its last item.

### q591
Two opposite corners of a table, plus the shape they sit in.

### q778
Corners, count, and whether the edges are the same row.

### q779
Pick, read, and report where the value came from.

## Misconceptions

- **"`items[1]` is the first item."** — It is the second. Counting starts at 0.
- **"To get the last item I need the length."** — `items[-1]` does it with no
  measurement, and cannot go one past the end.
- **"`rows[i][j]` and `rows[j][i]` are the same."** — Only on a square table, and
  only by coincidence. The first index always picks the row.
