---
kc: python.calling-functions
title: Calling a function — arguments in, one value out
supporting: [python.types-and-conversion, python.lists-and-tuples]
new_syntax: [builtin.max, builtin.min, builtin.round, builtin.sorted, builtin.sum, syntax.call, syntax.keyword-argument]
faded: [592, 593, 780, 781, 782, 783, 784, 785, 786, 787]
guided: []
independent: [594, 595, 788, 789, 790, 791, 792, 793, 794, 795]
integrated: [596, 597, 796, 797]
---

## Concept

A **function** is a piece of work that has been given a name. **Calling** it is
writing that name followed by parentheses — and the parentheses are what make
the work happen:

```python
items = [10, 20, 30]

print(len(items))
print(len)
```

The second line prints something like `<built-in function len>`. Without
parentheses you are talking ABOUT the function; with them you are asking it to
run. Almost every "why did I get `<function ...>` back?" is this.

Values handed to a call are its **arguments**, separated by commas. Order
matters, because each position means something different to the function:

```python
print(round(3.14159, 2))
print(round(3.14159, 4))
```

`round` takes the number first and how many decimal places second. Swapping them
asks a different question — and usually an impossible one.

Every call produces **one value**, which you can name, print, or feed straight
into another call:

```python
numbers = [4, 1, 7]

print(sum(numbers))
print(max(numbers))
print(min(numbers))
print(sum(numbers) / len(numbers))
```

That last line is a call inside an expression built from two other calls: each
one is worked out first, and the results are then divided.

Some arguments are settings rather than data, and those are passed **by name**
so the call still reads as English. A named argument is written
`name=value` and can be left out entirely, in which case the function uses its
own default:

```python
items = [3, 1, 2]

print(sorted(items))
print(sorted(items, reverse=True))
```

`sorted` returns a NEW list and leaves the original alone — which you can check,
and which is why the name `sorted` is worth trusting:

```python
items = [3, 1, 2]
ordered = sorted(items)

print(ordered)
print(items)
```

## Watch out

- **No parentheses, no work** — `len` is the function, `len(items)` is the
  number.
- **Position is meaning** — `round(x, 2)` and `round(2, x)` are different
  questions.
- **A keyword argument must be named exactly** — `reverse=True` changes the
  direction; `True` on its own in that slot means something else entirely.

## Worked example

Three calls over one list, then a fourth built from two of them.

```python
readings = [4, 9, 2, 9]

count = len(readings)
total = sum(readings)
biggest = max(readings)

print("count  :", count)
print("total  :", total)
print("biggest:", biggest)
print("mean   :", round(total / count, 2))
```

Why each step:

1. `len`, `sum` and `max` each take the same list and each hand back one value.
   Naming those values is what lets the last line read as a formula.
2. The mean is `total / count` — two names, no recomputation.
3. `round(..., 2)` wraps that expression: the division happens first, and the
   rounded result is what gets printed.

## Faded practice

### q592
Ask the built-in how many items the list holds.

```python starter
def solve(items):
    """Call len on the list."""
    return _____(items)
```

```python solution
def solve(items):
    """Call len on the list."""
    return len(items)
```

### q593
Two arguments, in the order the function expects them.

```python starter
def solve(x, places):
    """round takes the value first, then how many places."""
    return round(x, _____)
```

```python solution
def solve(x, places):
    """round takes the value first, then how many places."""
    return round(x, places)
```

### q780
The parentheses are what make the work happen.

```python starter
def solve(numbers):
    """One call, one value back."""
    return _____(numbers)
```

```python solution
def solve(numbers):
    """One call, one value back."""
    return sum(numbers)
```

### q781
The two extremes, biggest first.

```python starter
def solve(numbers):
    """The two extremes, biggest first."""
    return (_____(numbers), _____(numbers))
```

```python solution
def solve(numbers):
    """The two extremes, biggest first."""
    return (max(numbers), min(numbers))
```

### q782
A new, ordered list comes back from one call.

```python starter
def solve(items):
    """sorted returns a new, ordered list."""
    return _____(items)
```

```python solution
def solve(items):
    """sorted returns a new, ordered list."""
    return sorted(items)
```

### q783
The direction travels as a named argument.

```python starter
def solve(items, flag):
    """The direction travels as a named argument."""
    return sorted(items, _____=flag)
```

```python solution
def solve(items, flag):
    """The direction travels as a named argument."""
    return sorted(items, reverse=flag)
```

### q784
The distance between the two extremes.

```python starter
def solve(items):
    """The distance between the two extremes."""
    return _____(items) - _____(items)
```

```python solution
def solve(items):
    """The distance between the two extremes."""
    return max(items) - min(items)
```

### q785
The total over the count — each call runs first, then the division.

```python starter
def solve(numbers):
    """sum over len — each call runs first, then the division."""
    return sum(numbers) / _____(numbers)
```

```python solution
def solve(numbers):
    """sum over len — each call runs first, then the division."""
    return sum(numbers) / len(numbers)
```

### q786
The function is a value; its answer is another.

```python starter
def solve(items):
    """The function is a value; its answer is another."""
    return (type(_____).__name__, type(_____).__name__)
```

```python solution
def solve(items):
    """The function is a value; its answer is another."""
    return (type(len).__name__, type(len(items)).__name__)
```

### q787
Three calls, one line: the mean, trimmed.

```python starter
def solve(numbers, places):
    """Three calls, one line: the mean, trimmed."""
    return _____(sum(numbers) / len(numbers), places)
```

```python solution
def solve(numbers, places):
    """Three calls, one line: the mean, trimmed."""
    return round(sum(numbers) / len(numbers), places)
```

## Solo practice

### q594
Three built-in calls over one list, reported together.

Each call is independent and each hands back exactly one value, so they can be
built into a tuple in whatever order the question asks for:

```python worked
numbers = [8, 3, 5]

print("sum:", sum(numbers))
print("max:", max(numbers))
print("min:", min(numbers))
print("as one answer:", (sum(numbers), max(numbers), min(numbers)))
```

### q595
Largest first — the direction is a named setting, not a different function.

### q788
A new list comes back; the original is untouched — return both and prove it.

### q789
Ascending and descending, from two calls of the same function.

### q790
Equal after trimming is a different question from equal.

### q791
A zero span means every value is the same.

### q792
min and max take bare arguments too — order a pair without a list.

### q793
Two sums over the edge rows, added.

### q794
Two roads to the smallest value.

### q795
round(x) is an int; round(x, 0) is still a float.

## Integrated practice

### q596
Count, average and round, with each call feeding the next.

### q597
One sorted call, read from the front, plus the span between the extremes.

### q796
The list in order, and four one-call facts about it.

### q797
Round the mean, and say whether rounding changed it.

## Misconceptions

- **"`len` gives the length."** — `len(items)` gives the length. `len` on its own
  is the function object, and returning it is the most common way a drill fails
  with no error message.
- **"Arguments can go in any order."** — Only keyword arguments can. Positional
  arguments mean whatever their position means.
- **"`sorted` sorts the list."** — It returns a new one. The original is
  unchanged, which is exactly why you have to keep the result.
