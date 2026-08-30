---
kc: python.values-and-names
title: Values and names — what = really does
supporting: []
new_syntax: [builtin.print, builtin.type, syntax.assign, syntax.tuple]
faded: [568, 569]
guided: []
independent: [570, 571]
integrated: [572, 573]
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

## Integrated practice

### q572
Three named results — a sum, a difference, and a swapped pair — returned together.

### q573
A chain: each step is built from the name the step before it wrote.

## Misconceptions

- **"`x = x + 1` is a contradiction."** — It is an instruction, not a claim. The
  right-hand side is worked out with the OLD value, and the name is then
  re-attached to the result.
- **"Printing a value and returning it are the same thing."** — `print` shows
  something to a human; `return` hands a value back to whatever called the
  function. A drill that prints instead of returning gives the grader nothing.
- **"A name IS the value."** — A name is a label. Two names can be stuck to the
  same value, and re-labelling one leaves the other where it was.
