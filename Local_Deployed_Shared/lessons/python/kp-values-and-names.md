---
kc: python.values-and-names
title: Values and names — what = really does
supporting: []
new_syntax: [builtin.print, builtin.type, syntax.assign, syntax.tuple]
faded: [568, 569, 718, 719, 720, 721, 722, 723, 724, 725]
guided: []
independent: [570, 571, 708, 709, 710, 711, 726, 727, 728, 729]
integrated: [572, 573, 712, 730]
---

## Concept

Everything in this course is built out of two moves: producing a **value**, and
giving that value a **name** so you can talk about it again later.

A value is a thing — the number `5`, the text `"hi"`, a list of numbers. An
**expression** is anything that produces one: `2 + 3` is an expression whose
value is `5`.

`print(...)` shows you a value. It is how a program says something out loud, and
almost every cell below ends with one.

```python
print(2 + 3)
print("two" + "three")
```

The `=` sign is **not** a claim that two things are equal. It is an instruction:
*work out the value on the right, then attach the name on the left to it.* The
name is a label, and the value is what the label is stuck to.

```python
total = 2 + 3
print(total)
```

Read `total = 2 + 3` as "let `total` be 5 from now on". Once a name exists you
can use it anywhere the value would work, including in building the next one:

```python
price = 4
quantity = 3
subtotal = price * quantity
print(subtotal)
```

Because the right-hand side is worked out **first**, a name can be built out of
its own old value. This trips people up until they read it in the right order:

```python
n = 10
n = n + 3
print(n)
```

The second line does not say "n equals n plus 3", which would be nonsense. It
says: take what `n` is now (10), add 3 (13), then re-attach the name `n` to
that. The old value is simply let go.

A name holds one value at a time, but you can hand several back together by
putting them in **parentheses**, separated by commas. That is a *tuple*, and it
is how a piece of code gives more than one answer at once.

```python
pair = (1, 2)
print(pair)
```

Every value has a **kind** — a whole number, a decimal, a piece of text, a
tuple. `type(...)` reports it, and the answer is about the value, not about the
name stuck to it.

```python
count = 5
label = "five"
print(type(count))
print(type(label))
```

Re-attaching a name to a different value can change the kind as well as the
value, because the kind was never the name's to begin with.

```python
n = 5
print(type(n))
n = (5, 5)
print(type(n))
```

## Watch out

- **`=` is not a comparison** — `x = 5` sets a name; `x == 5` asks a question and
  produces `True` or `False`. Almost every early error message about assignment
  is one of these two written where the other belonged.
- **A quoted name is text, not the value** — `total` is the name; `"total"` is a
  five-letter string that has nothing to do with it.
- **Re-assigning does not change the old value** — it moves the label. Anything
  else already holding that value keeps it.

## Worked example

Two values, named, then combined and handed back as a pair.

```python
a = 7
b = 2

total = a + b
difference = a - b

print("total is", total)
print("difference is", difference)
print("both together:", (total, difference))
```

Why each step:

1. `a` and `b` are names for the two inputs, so the arithmetic below reads as
   words rather than as bare numbers.
2. `total` and `difference` name the two results. Naming a result is what lets
   the next line use it without recomputing it.
3. The final `(total, difference)` is a tuple: one value that carries both
   answers, which is exactly what a function returns when it has two things to
   say.

## Faded practice

### q568
Give the sum a name, then hand the name back.

```python starter
def solve(a, b):
    """Store the sum under a name, then return the name."""
    _____ = a + b
    return total
```

```python solution
def solve(a, b):
    """Store the sum under a name, then return the name."""
    total = a + b
    return total
```

### q569
Swap two values by naming each one first.

```python starter
def solve(x, y):
    """Swap, via two names."""
    first = _____
    second = x
    return (first, second)
```

```python solution
def solve(x, y):
    """Swap, via two names."""
    first = y
    second = x
    return (first, second)
```

### q718
Name the first result, then build the second out of that name.

```python starter
def solve(price, quantity, fee):
    """Name the first result, then read that name to build the second."""
    subtotal = price * quantity
    total = _____ + fee
    return total
```

```python solution
def solve(price, quantity, fee):
    """Name the first result, then read that name to build the second."""
    subtotal = price * quantity
    total = subtotal + fee
    return total
```

### q719
Build the new value out of the name's own old value.

```python starter
def solve(n):
    """The right-hand side runs first, reading the OLD value."""
    n = _____ + 5
    return n
```

```python solution
def solve(n):
    """The right-hand side runs first, reading the OLD value."""
    n = n + 5
    return n
```

### q720
Put both inputs and their sum into one tuple.

```python starter
def solve(first, second):
    """Three values, one tuple."""
    s = first + second
    p = (_____, _____, s)
    return p
```

```python solution
def solve(first, second):
    """Three values, one tuple."""
    s = first + second
    p = (first, second, s)
    return p
```

### q721
Ask what KIND of value arrived, and hand the answer back.

```python starter
def solve(value):
    """type(...) reports the kind of the VALUE."""
    kind = _____(value)
    return kind
```

```python solution
def solve(value):
    """type(...) reports the kind of the VALUE."""
    kind = type(value)
    return kind
```

### q722
Stick a second label on the same value, then read both.

```python starter
def solve(base):
    """Two labels stuck to one value."""
    left = base
    right = _____
    return (left, right, left + right)
```

```python solution
def solve(base):
    """Two labels stuck to one value."""
    left = base
    right = base
    return (left, right, left + right)
```

### q723
Re-attach the name to its own square.

```python starter
def solve(a):
    """Re-attach the name to its own square."""
    x = a
    x = _____ * _____
    return x
```

```python solution
def solve(a):
    """Re-attach the name to its own square."""
    x = a
    x = x * x
    return x
```

### q724
Build a new string, name it, and report its kind.

```python starter
def solve(word):
    """A built string is a value like any other: name it, ask its kind."""
    greeting = word + _____
    kind = type(greeting)
    return (greeting, kind)
```

```python solution
def solve(word):
    """A built string is a value like any other: name it, ask its kind."""
    greeting = word + "!"
    kind = type(greeting)
    return (greeting, kind)
```

### q725
Two named results, handed back as one pair.

```python starter
def solve(a, b):
    """Two named results, packed as one pair."""
    diff = a - b
    prod = _____ * _____
    both = (diff, prod)
    return both
```

```python solution
def solve(a, b):
    """Two named results, packed as one pair."""
    diff = a - b
    prod = a * b
    both = (diff, prod)
    return both
```

## Solo practice

### q570
Compute the total once, name it, and build the doubled value out of that name.

A name exists so the value behind it is computed once and read many times.
Reaching for the formula a second time is the habit this drill is built to
break:

```python worked
price = 6
quantity = 2

total = price * quantity
doubled = total + total

print("computed once:", total)
print("reused, not recomputed:", doubled)
```

### q571
Add 3 to the same name twice, and report where it ends up.

### q708
Name each argument, then report what kind of value one of them is.

### q709
Re-attach one name twice and report the kind before and after.

### q710
Two names, one value: move one label and show the other did not follow.

### q711
Rotate three values by one place, keeping the one that would be overwritten.

### q726
A running total: name the first sum, read it to build the grand total.

### q727
A quoted word is a value too: pair a text label with a computed number.

### q728
Powers by reuse: square the square, never multiply four times.

### q729
Re-attach a name to a tuple built from its own old value, and report the new kind.

## Integrated practice

### q572
Three named results — a sum, a difference, and a swapped pair — returned together.

### q573
A chain: each step is built from the name the step before it wrote.

### q712
Name a result, build a second value out of it, then re-attach the first name —
and show that the value built earlier kept what it was given.

### q730
Swap by naming, join by reading, and report the kind + made — for numbers and for text.

## Misconceptions

- **"`x = x + 1` is a contradiction."** — It is an instruction, not a claim. The
  right-hand side is worked out with the OLD value, and the name is then
  re-attached to the result.
- **"Printing a value and returning it are the same thing."** — `print` shows
  something to a human; `return` hands a value back to whatever called the
  function. A drill that prints instead of returning gives the grader nothing.
- **"A name IS the value."** — A name is a label. Two names can be stuck to the
  same value, and re-labelling one leaves the other where it was.
