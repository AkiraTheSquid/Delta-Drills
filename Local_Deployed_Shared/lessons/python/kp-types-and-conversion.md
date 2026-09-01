---
kc: python.types-and-conversion
title: The everyday types, and converting between them
supporting: [python.values-and-names]
new_syntax: [builtin.bool, builtin.float, builtin.int, builtin.round, builtin.str, python.type-name, syntax.equality]
faded: [574, 575, 731, 732, 733, 734, 735, 736, 737, 738]
guided: []
independent: [576, 577, 713, 714, 715, 716, 739, 740, 741, 742]
integrated: [578, 579, 717, 743]
---

## Concept

Every value has a **type**, and the type decides what the value can do. Four of
them carry almost all of the early work:

| Type | What it holds | Written as |
|---|---|---|
| `int` | a whole number | `5`, `-2`, `0` |
| `float` | a number with a fractional part | `2.5`, `-0.75`, `4.0` |
| `str` | text | `"hi"`, `'42'` |
| `bool` | a yes/no answer | `True`, `False` |

`type(value)` hands back the type itself. Its `__name__` is the readable word,
which is usually what you actually want to look at:

```python
print(type(5).__name__)
print(type(2.5).__name__)
print(type("42").__name__)
print(type(True).__name__)
```

Notice that `42` and `"42"` are **not** the same value. One is a number; the
other is two characters that happen to look like one. The difference shows up
the moment you use `+`, because `+` does a different job for each type:

```python
print(4 + 2)
print("4" + "2")
```

That is not a quirk — it is the type deciding the meaning. Numbers add; text
joins end to end.

Asking whether they are **equal** gives the same answer for the same reason.
`==` compares values, and a value made of characters is never equal to a value
made of digits — no matter how alike they look on the page:

```python
print("42" == 42)
print(42 == 42)
```

Comparing does not convert. If you want the question answered about the
*number*, you have to do the converting yourself first:

```python
print(int("42") == 42)
```

To move between the two you **convert**, by calling the type's own name as a
function. Each conversion produces a NEW value and leaves the original alone:

```python
text = "42"
number = int(text)

print(number + 8)
print(text)
```

`float(...)` and `str(...)` work the same way, and going number → text →
number again gets you back where you started:

```python
print(float("2.5"))
print(str(7) + "!")
print(int(str(7)))
```

Two conversions that look similar and are not: `int(...)` throws the fractional
part **away**, while `round(...)` goes to the nearest whole number.

```python
print(int(3.9))
print(round(3.9))
print(int(-2.7), round(-2.7))
```

`bool(...)` is a conversion too, and the one that surprises people. It asks a
single question of a value — *is there anything here?* — and almost everything
answers yes. Zero answers no, and so does empty text:

```python
print(bool(7), bool(0))
print(bool("hi"), bool(""))
print(bool(0.0), bool(-3))
```

The trap is the text `"0"`. It is one character long, so there **is** something
there and it converts to `True`. The zero only shows up once the digits have
been read as a number:

```python
print(bool("0"))
print(bool(int("0")))
```

## Watch out

- **`int()` truncates, it does not round** — `int(3.9)` is `3`. If you wanted
  `4`, `round` is the call you meant.
- **`"5"` is not `5`** — a string of digits stays text until something converts
  it, and comparing the two gives `False`.
- **`bool` is its own type** — `True` behaves like `1` in arithmetic, but
  `type(True).__name__` is `"bool"`, not `"int"`.

## Worked example

One string, read three different ways, with the type named at each step.

```python
text = "12"

as_int = int(text)
as_float = float(text)
back_to_text = str(as_int)

print(as_int, type(as_int).__name__)
print(as_float, type(as_float).__name__)
print(back_to_text, type(back_to_text).__name__)
print("the original is untouched:", text, type(text).__name__)
```

Why each step:

1. `int(text)` reads the digits as a whole number. The string itself is not
   changed — conversion produces a new value.
2. `float(text)` reads the same digits as a number with a fractional part. `12`
   and `12.0` are equal in value and different in type.
3. `str(as_int)` goes back the other way, which is how a number gets glued into
   a message.

## Faded practice

### q574
Report the name of a value's type.

```python starter
def solve(value):
    """The name of the value's type."""
    return type(value)._____
```

```python solution
def solve(value):
    """The name of the value's type."""
    return type(value).__name__
```

### q575
Text in, arithmetic out — the digits have to become a number first.

```python starter
def solve(digits):
    """Text in, number out."""
    return _____(digits) + 8
```

```python solution
def solve(digits):
    """Text in, number out."""
    return int(digits) + 8
```

### q731
Convert first — then * means arithmetic, not repetition.

```python starter
def solve(text):
    """Convert first — then * means arithmetic."""
    return _____(text) * 2
```

```python solution
def solve(text):
    """Convert first — then * means arithmetic."""
    return float(text) * 2
```

### q732
A number cannot be glued onto text until it IS text.

```python starter
def solve(count):
    """str() makes a number gluable."""
    label = _____(count)
    return label + " items"
```

```python solution
def solve(count):
    """str() makes a number gluable."""
    label = str(count)
    return label + " items"
```

### q733
Its truth, and the name of its kind.

```python starter
def solve(value):
    """Its truth, and the name of its kind."""
    return (_____(value), type(value).__name__)
```

```python solution
def solve(value):
    """Its truth, and the name of its kind."""
    return (bool(value), type(value).__name__)
```

### q734
To the nearest whole number — not truncated.

```python starter
def solve(x):
    """round() goes to the nearest whole number."""
    return _____(x)
```

```python solution
def solve(x):
    """round() goes to the nearest whole number."""
    return round(x)
```

### q735
Put both values on the same side of the type line, then compare.

```python starter
def solve(text, number):
    """Convert to the SAME type, then compare."""
    return text == _____(number)
```

```python solution
def solve(text, number):
    """Convert to the SAME type, then compare."""
    return text == str(number)
```

### q736
Truncate both — drop the fraction, do not round — then multiply.

```python starter
def solve(x, y):
    """int() drops the fraction; it does not round."""
    return _____(x) * _____(y)
```

```python solution
def solve(x, y):
    """int() drops the fraction; it does not round."""
    return int(x) * int(y)
```

### q737
Text to number to text: the kind changes twice, the digits never do.

```python starter
def solve(digits):
    """Text to number to text: the kind changes, the digits do not."""
    n = _____(digits)
    back = _____(n)
    return (n, back)
```

```python solution
def solve(digits):
    """Text to number to text: the kind changes, the digits do not."""
    n = int(digits)
    back = str(n)
    return (n, back)
```

### q738
The truth of the NUMBER, not of the text holding it.

```python starter
def solve(text):
    """bool of the NUMBER, not of the text holding it."""
    return bool(_____(text))
```

```python solution
def solve(text):
    """bool of the NUMBER, not of the text holding it."""
    return bool(int(text))
```

## Solo practice

### q576
One string of digits, returned as an int, as a float, and as text again.

Each conversion is its own call, and the original never changes — which is why
all three can exist at the same time:

```python worked
text = "9"

print(int(text), type(int(text)).__name__)
print(float(text), type(float(text)).__name__)
print(str(int(text)), type(str(int(text))).__name__)
```

### q577
Truncating and rounding, side by side, on the same number.

### q713
Ask two values whether they count as a yes.

### q714
Round a number that arrives as text — which it cannot be until it is converted.

### q715
The same digits compared as text and as a number.

### q716
Ask before converting, and ask after, on a piece of text holding `0`.

### q739
Convert two digit strings, add as numbers, and carry the sum back as text too.

### q740
One float, three readings: truncated, rounded, and whether they agree.

### q741
Two values can share a truth without being equal.

### q742
Each conversion has a kind of its own — name both.

## Integrated practice

### q578
Name the type, show the text, and decide whether the value counts as a number.

### q579
The same `+` doing both of its jobs, in one function.

### q717
Text in — round it, name the type of what you get, say whether it counts as a
yes, and whether writing it back out spells what you started with.

### q743
Parse a quantity, rebuild it into a label, and report its truth and the label's kind.

## Misconceptions

- **"Converting changes the value."** — It produces a new one. `int(text)` hands
  back a number and leaves `text` exactly as it was.
- **"int() rounds."** — It truncates toward zero: `int(3.9)` is `3` and
  `int(-2.7)` is `-2`.
- **"If it looks like a number it is one."** — `"42"` looks like a number to a
  human and is text to Python. Only a conversion makes it arithmetic-ready.
