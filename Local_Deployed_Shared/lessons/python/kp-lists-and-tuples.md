---
kc: python.lists-and-tuples
title: Lists and tuples — holding more than one value
supporting: [python.values-and-names, python.types-and-conversion]
new_syntax: [builtin.len, builtin.list, builtin.tuple, syntax.list-literal, syntax.tuple]
faded: [580, 581]
guided: []
independent: [582, 583]
integrated: [584, 585]
---

## Concept

A name holds one value — but that value can itself be a **container** holding
many.

A **list** is written with square brackets, and holds its items in order:

```python
scores = [10, 20, 30]
print(scores)
print(len(scores))
```

`len(...)` asks the container how many items it holds. Asking is better than
remembering: the answer stays right when the list changes.

A list can hold values of different types, and its items can themselves be
lists. A list of lists is how a table (rows of equal length) is written before
any library gets involved:

```python
mixed = [1, "two", 3.0]
rows = [[1, 2, 3], [4, 5, 6]]

print(mixed)
print(rows)
print(len(rows), "rows of", len(rows[0]))
```

A **tuple** is written with parentheses instead. It holds items in order in
exactly the same way — the difference is that a tuple cannot be changed after it
is built:

```python
point = (3, 4)
print(point, len(point))
```

That is why a tuple is the usual way to hand back several answers at once: it is
a fixed group of values, not a collection you are still working on. Use a list
when the contents will grow or change, a tuple when they are one finished
answer.

The two convert into each other, and converting always produces a NEW container:

```python
values = [1, 2, 3]
as_tuple = tuple(values)
back = list(as_tuple)

print(as_tuple)
print(back)
print("same length:", len(values) == len(back))
```

A one-item tuple needs a trailing comma — `(5)` is just the number 5 with
brackets around it, while `(5,)` is a tuple holding it:

```python
print(type((5)).__name__)
print(type((5,)).__name__)
```

## Watch out

- **Brackets decide the type** — `[a, b]` is a list, `(a, b)` is a tuple. They
  hold the same values and are still different types.
- **`len` counts the OUTER level only** — `len([[1, 2, 3]])` is `1`: one inner
  list. What is inside it is a second question.
- **`(5)` is not a tuple** — a one-item tuple is `(5,)`.

## Worked example

A table as a list of lists, measured at both levels, then frozen into a tuple.

```python
rows = [[1, 2, 3], [4, 5, 6]]

n_rows = len(rows)
n_cols = len(rows[0])

print("rows:", n_rows)
print("columns:", n_cols)

shape = (n_rows, n_cols)
print("as one finished answer:", shape, type(shape).__name__)
```

Why each step:

1. `len(rows)` counts the inner lists — the rows of the table.
2. `len(rows[0])` reaches into the first row and counts what IT holds. Two
   levels, two separate measurements.
3. The pair is packed into a tuple because it is one finished answer with two
   parts, which is exactly the job a tuple is for.

## Faded practice

### q580
Build the list, then ask it how long it is.

```python starter
def solve(a, b, c):
    """Build the list, then measure it."""
    items = [a, b, c]
    return (items, _____(items))
```

```python solution
def solve(a, b, c):
    """Build the list, then measure it."""
    items = [a, b, c]
    return (items, len(items))
```

### q581
Two values in a tuple — parentheses, not brackets.

```python starter
def solve(a, b):
    """A tuple of the two values."""
    return _____a, b_____
```

```python solution
def solve(a, b):
    """A tuple of the two values."""
    return (a, b)
```

## Solo practice

### q582
A list of inner lists, measured at both levels.

The two measurements are different questions asked at different depths, and
mixing them up is the most common early mistake with nested data:

```python worked
rows = [["a", "b", "c"], ["d", "e", "f"]]

print("how many rows:", len(rows))
print("how wide is a row:", len(rows[0]))
print("the first row itself:", rows[0])
```

### q583
A round trip: list to tuple and back, with the lengths compared.

## Integrated practice

### q584
The same three values in both containers, reported with their sizes.

### q585
Count the rows, measure every one of them, and say whether they agree.

## Misconceptions

- **"A tuple is just a list that is written differently."** — It is a different
  type, and it cannot be changed once built. A grader comparing a list against a
  tuple reports them as unequal.
- **"`len` gives the total number of values."** — It gives the number of items at
  the top level. For a list of lists that is the number of rows.
- **"Converting a list to a tuple changes the list."** — It builds a new
  container. The original list is still there, unchanged.
