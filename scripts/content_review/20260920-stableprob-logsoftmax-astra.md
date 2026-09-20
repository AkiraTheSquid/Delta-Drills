## 1. Drills — per-id findings

**1728: major.** Grading works; prompt and faded scaffolding fail rubric.

Standard: [CONTENT_RUBRIC.md](/home/stellar-thread/Applications/Delta-Drills-Local/scripts/CONTENT_RUBRIC.md). Reviewed only bank row 1728. Other IDs below refer exclusively to excerpts on the supplied lesson page.

### D1 — `GIVE_PROMPT` · major

**Quote:** “computed as its score minus the row's log-sum-exp — never by forming the probability and taking its log.”

**Issue:** Supplies computational route. Numerical requirement should describe required behavior.

**Replacement:** “Return each class’s natural-log softmax probability, preserving finite results even when the corresponding probability would round to zero.”

Repeated method disclosure in lesson excerpt **103**; grouped below.

### D2 — `GIVE_STARTER` · major

**Quote:**

```python
"""Log-softmax: score minus the row's log-sum-exp, never through a probability."""
```

**Issue:** Starter docstring repeats solution strategy.

**Replacement:**

```python
"""Return the natural-log class probabilities for each row."""
```

### D3 — `GIVE_STARTER`, `GIVE_RUNG` · major

**Quotes:** Frontmatter assigns `1728` to `faded`; starter body ends with:

```python
pass
```

**Issue:** Empty implementation requires writing solution from scratch. Faded requires scaffold with blanks. Same defect appears in lesson starters **1032, 1033, 1034, 1035, 1728**.

**Replacement instruction:** “Complete the blanks to return each row’s natural-log class probabilities.”

Concrete scaffold for current row-based contract:

```python
def solve(x):
    z = x - x.max(dim=1, keepdim=True)[0]
    return z - z.___().sum(dim=1, keepdim=True).___()
```

Both declared new symbols, `Tensor.exp` and `Tensor.log`, remain blank.

### D4 — `STMT_TERMS` · minor

**Quotes:** “logits”; “the row's log-sum-exp”.

**Issue:** Standalone statement leaves specialist vocabulary undefined; logarithm base also unstated.

**Replacement:** “Each row contains unnormalized class scores, called logits; its softmax probabilities are proportional to e raised to those scores and sum to one, and log-probability means their natural logarithm.”

Remove “log-sum-exp” from prompt when applying D1.

### Passing checks

**1728:** `STMT_TASK`, `STMT_INPUT`, `STMT_OUTPUT`, `STMT_SELF_CONTAINED`, `STMT_CASES`, `STMT_NEAR_MISS`, `STMT_LENGTH`, `GIVE_TESTS`, `CORR_RUNS`, `CORR_DECISIVE`, `CORR_TYPES`.

Executed in memory using backend comparison harness and backend Python environment:

| Submission | Case 1 | Case 2 | Case 3 | Case 4 |
|---|---|---|---|---|
| Supplied answer | Pass | Pass | Pass | Pass |
| No-op: return input | Pass | Fail | Fail | Fail |
| Constant: first expected tensor | Pass | Fail | Fail | Fail |
| Near miss: softmax, then log | Fail | Pass | Pass | Pass |

Cases cover underflow, unequal scores, ties, multiple rows, single-row edge. Expected outputs differ. Backend converts tensors before shape-aware numerical comparison.

**Whole-drill passes:** none.

## 2. Lesson pages — per page, per rubric code

### [kp-stable-probabilities.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/pytorch/kp-stable-probabilities.md) — major

### `GIVE_PROMPT` · major — practice excerpts

**Affected IDs:** **1728, 103**. Same pattern: prompt supplies stabilization method.

1728: D1 above.

**103 quote:** “Softmax of a logit vector that survives entries like 1000 — shift before exponentiating.”

**Replacement:** “Return the softmax probabilities of a one-dimensional floating-point PyTorch tensor of length `n`, preserving its shape and remaining numerically stable for scores such as 1000.”

### `GIVE_STARTER`, `GIVE_RUNG` · major — Faded practice

**Affected IDs:** **1032, 1033, 1034, 1035, 1728**.

Shared exact starter fragment:

```python
def solve(x):
```

Each ends with `pass`; none contains implementation blanks. **D3 covers this repeated defect.** 1728 additionally contains D2’s strategy-bearing docstring.

**Replacement instruction:** “Complete the blanks in the supplied scaffold to produce the requested tensor.”

Requires actual scaffold changes; sentence alone cannot fix rung mismatch.

### L1 — `EXPL_CORRECT` · major — Keep tiny probabilities in log space

**Quote:** “the sum inside now has a largest term of `1`, so it can neither overflow nor become zero, and its log is finite.”

**Issue:** Bounding individual terms does not bound their sum within dtype capacity. Executed counterexample: summing shifted exponentials for **65,536 equal float16 scores** produces `inf`.

**Replacement:** “For a nonempty row, shifted exponentials lie between zero and one and include at least one one, so their sum is positive and at most the class count; the accumulation dtype must still represent that sum.”

### L2 — `EXPL_CORRECT` · major — Misconceptions

**Quote:** “Past about `88` for float32 it is infinity, and every later step is `nan`.”

**Issue:** “Every later step” is false. Executed float32 input `[[90., 0.]]`: exponentials `[[inf, 1.]]`; normalized result `[[nan, 0.]]`.

**Replacement:** “Large float32 scores can overflow during exponentiation; normalization can then produce NaN from infinity divided by infinity, or zero from a finite weight divided by infinity.”

### L3 — `EXPL_CORRECT` · major — first Worked example

**Quote:** “After the shift the largest entry of each row is zero, so `exp` gives `1` there and less elsewhere.”

**Issue:** Tied maxima all produce one. Current example’s second row explicitly contains a tie.

**Replacement:** “After shifting, every entry tied for the row maximum is zero and exponentiates to one; entries below the maximum exponentiate to values below one.”

### L4 — `EXPL_CORRECT` · major — positive-weight claims

**Affected locations:** first Concept; lesson excerpt **1033**.

**Quotes:**

- “exponentiate every score, so all are positive”
- “Return positive unnormalized weights after removing the largest row score, shape (b,c).”

**Issue:** Mathematical positivity becomes an unconditional floating-point claim. Supplied q1033 solution returns `[[1., 0.]]` for `[[0., -1000.]]`.

**Concept replacement:** “Exponentials are strictly positive mathematically, but sufficiently small values round to zero in floating-point arithmetic.”

**1033 replacement:** “Return a floating-point tensor of shape `(b, c)` containing nonnegative class weights proportional to exponentials of the row scores, with each row’s largest weight equal to one.”

### L5 — `EXPL_CORRECT`, `EXPL_WHY` · major — log-softmax procedure

**Quotes:**

- “Log-softmax is then `x - logsumexp(x)`”
- “Compute score minus log-sum-exp instead.”

**Issue:** Identity is mathematically correct; presented numerical procedure can lose the normalization correction. Executed float32 input `[[1e20, 1e20]]`: literal subtraction yields `[[0., 0.]]`, while supplied centered implementation yields approximately `[[-0.693147, -0.693147]]`.

Code already avoids this failure; prose never explains why it keeps subtraction centered.

**Replacement:** “For log-softmax, keep scores centered: with `z = x - m`, compute `z - log(sum(exp(z)))`; this avoids losing the small normalization correction when a large maximum is added back and then subtracted.”

### L6 — `EXPL_TRANSFER` · major — q1728 after second Concept

**Concept quote:**

```python
logp=x-m-(x-m).exp().sum(dim=1,keepdim=True).log()
```

**1728 solution quote:**

```python
return x-m-(x-m).exp().sum(dim=1,keepdim=True).log()
```

**Issue:** Concept supplies exact return expression. Drill keeps same row/class arrangement; changing values and batch size does not require adapting that expression.

**Replacement prompt:** “Return natural-log class probabilities for each column of `x`, a floating-point PyTorch tensor of shape `(c, b)` whose rows represent classes and columns represent examples; return a tensor of shape `(c, b)`.”

This transfer revision requires corresponding scaffold, answer, and case changes.

### Lesson passing checks

Both concept segments pass `EXPL_GENERAL_FIRST` and `EXPL_LENGTH`: approximately **184** and **166** prose words.

Both worked examples pass `EXPL_INTRO`. Example blocks pass `EXPL_INTERLEAVE` and `EXPL_PRINTS`. All **six explanatory Python blocks** execute successfully with their assertions.

Prerequisite ordering and glossary compliance remain unverified; no unsupported `EXPL_PREREQ` or `EXPL_TERMS` finding assigned.

## 3. Systemic patterns

| Pattern | Rubric codes | Affected scope |
|---|---|---|
| Prompts disclose numerical method instead of specifying behavior | `GIVE_PROMPT` | 1728; lesson excerpt 103 |
| Faded ownership paired with empty implementation | `GIVE_STARTER`, `GIVE_RUNG` | Lesson starters 1032–1035, 1728 |
| Strategy repeated across prompt, docstring, concept solution | `GIVE_PROMPT`, `GIVE_STARTER`, `EXPL_TRANSFER` | 1728 |
| Exact mathematical properties presented as unconditional floating-point guarantees | `EXPL_CORRECT` | Positive weights, bounded shifted sum, overflow propagation |
| Implementation handles cancellation better than explanation teaches | `EXPL_CORRECT`, `EXPL_WHY` | Log-space Concept; Misconceptions |

Grading defects were **not** demonstrated: current answer passes, listed near miss fails, constant/no-op submissions fail, numerical comparison handles tensors correctly.

## 4. Top 10 fixes with rewrites

Ranked fixes below reference findings above; repeated occurrences remain grouped.

| Rank | Item / code | Replacement wording |
|---|---|---|
| **1** | Faded scaffold — D3, `GIVE_STARTER` / `GIVE_RUNG` | “Complete the blanks in the supplied scaffold to produce the requested tensor.” Supply actual blanks for all five affected starters. |
| **2** | 1728 prompt disclosure — D1, `GIVE_PROMPT` | “Return each class’s natural-log softmax probability, preserving finite results even when the corresponding probability would round to zero.” |
| **3** | 1728 starter disclosure — D2, `GIVE_STARTER` | “Return the natural-log class probabilities for each row.” |
| **4** | 1728 transcription — L6, `EXPL_TRANSFER` | “Return natural-log class probabilities for each column of `x`, shape `(c, b)`, where rows represent classes and columns represent examples.” |
| **5** | Cancellation explanation — L5, `EXPL_CORRECT` / `EXPL_WHY` | “Keep the scores centered throughout log-softmax so a large common offset cannot erase the normalization correction.” |
| **6** | Shifted-sum guarantee — L1, `EXPL_CORRECT` | “The shifted sum lies between one and the class count; its accumulation dtype must still represent that sum.” |
| **7** | Positive-weight guarantee — L4, `EXPL_CORRECT` | “Exponential weights are positive mathematically but may round to zero in floating-point arithmetic.” |
| **8** | Overflow consequences — L2, `EXPL_CORRECT` | “Overflowed normalization can produce NaNs at overflowing entries and zeros at finite entries.” |
| **9** | Tied maxima — L3, `EXPL_CORRECT` | “Every maximum, including tied maxima, exponentiates to one after shifting.” |
| **10** | Standalone terminology — D4, `STMT_TERMS` | “Logits are unnormalized class scores; log-probability means the natural logarithm of the corresponding softmax probability.” |

**Combined prompt for the transfer revision:**

> Return the natural-log softmax class probabilities for each example as a floating-point PyTorch tensor of shape `(c, b)`. Input `x` has shape `(c, b)`, with rows representing classes and columns representing examples; its entries are unnormalized class scores, and softmax probabilities are proportional to e raised to those scores and sum to one within each example. Preserve finite log-probabilities even when the corresponding probabilities would round to zero.
>
> - `b ≥ 1`, `c ≥ 1`.
> - Input scores are finite.
> - Complete the supplied blanks.

No edits performed.

Git: MODERATE — repo=Delta-Drills-Local branch=main; staged=0, unstaged=11, untracked=1, conflicts=0. Existing changes preserved; final status matches baseline.