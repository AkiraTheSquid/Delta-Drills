# Delta Drills content review

Standard: [CONTENT_RUBRIC.md](/home/stellar-thread/Applications/Delta-Drills-Local/scripts/CONTENT_RUBRIC.md). Scope: supplied lesson file; question-bank rows **1363–1373 only**. No edits.

**Bank verdicts: 6 major, 1 minor, 4 pass. Lesson: major.** Executed supplied answers against all **35 cases** in backend Python: **35/35 pass**. Both worked-example fences pass. Constant, no-op, reconstructed near-miss solutions each fail at least one case per drill.

## 1. Drills — per-id findings

Verdicts below concern bank records plus example overlap. Separate defects in lesson practice directions appear in §2.

| ID | Rung | Verdict | Findings |
|---|---|---|---|
| 1363 | Faded | **major** | D1: missing faded scaffold |
| 1364 | Faded | **major** | D1: missing faded scaffold |
| 1365 | Faded | **major** | D1: missing faded scaffold |
| 1366 | Faded | **major** | D1: missing faded scaffold |
| 1368 | Independent | **minor** | D4: difficulty below assigned rung |
| 1369 | Independent | **major** | D2: lesson reproduces drill; D3: near miss lacks input identification |
| 1370 | Independent | **major** | D5: no edge case |

**Pass: 1367, 1371, 1372, 1373.** Lesson directions associated with 1371–1372 still fail `GIVE_PROMPT`; see L4.

Across all 11: task, input rank/layout, output type/shape stated; prompts self-contained; no duplicate moves demonstrated within reviewed rungs. All case sets have differing expected outputs. Supplied answers satisfy `CORR_RUNS`; mutation checks satisfy `CORR_DECISIVE`.

### D1 — Faded starters remove scaffold instead of fading syntax

**Codes:** `GIVE_RUNG` — **major**; `GIVE_STARTER` — **major**.  
**IDs:** 1363, 1364, 1365, 1366. Related lesson starters: 220, 135.

**Exact quotes:**

- Lesson ownership: `faded: [220, 1363, 1364, 135, 1365, 1366]`
- Bank starter, all four reviewed faded drills: `return None`
- Lesson starters, 1363–1364: `return x._____(dim=_____)`
- Lesson starters, 1365–1366: `return x._____(dim=_____, keepdim=_____)`
- Lesson starters, 220 and 135: `return x.sum(dim=_____)`

Bank records provide from-scratch bodies despite faded ownership. Lesson alternatives preserve scaffold but expose declared syntax: `dim`, `keepdim`, sometimes `sum`.

**Replacement sentence:**

> Complete the blanks so the function returns the tensor described in the problem.

**Required starter replacements:**

```python
# 220, 135, 1363, 1364
return x._____(_____=_____)

# 1365, 1366
return x._____(_____=_____, _____=_____)
```

Use corresponding scaffold in both sources; preserve goal-based docstrings.

### D2 — Worked example reproduces q1369’s input and result

**Code:** `GIVE_EXAMPLE` — **major**.  
**ID:** 1369.

**Exact quotes:**

- Lesson and first test input:  
  `x = t.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])`
- Lesson: `rm = x.mean(dim=1, keepdim=True)`
- Lesson: `centered = x - rm`
- Test expectation: `[[-1.0, 0.0, 1.0], [-10.0, 0.0, 10.0]]`

Executed lesson produces precisely that expected output. Drill’s example can be copied from instruction.

**Replacement example sentence:**

> For `x = [[2.0, 8.0], [3.0, 7.0]]`, return `[[-3.0, 3.0], [-2.0, 2.0]]`.

Replace duplicated literals consistently in starter run, answer run, first case. Prompt itself already describes goal adequately.

### D3 — Near miss belongs to an unidentified second input

**Code:** `STMT_NEAR_MISS` — **minor**.  
**ID:** 1369.

**Exact quotes:**

- `"output": "[[-1.0, -1.0], [4.0, -2.0]]"`
- `"call": "solve(x).tolist()"`
- `"why": "Without keepdim the row means are a (r,) vector, which broadcasts against the COLUMNS: on a square matrix this runs without error and subtracts the wrong row's mean from every entry."`

Recorded result matches **case 2**, using `[[1.0, 3.0], [6.0, 2.0]]`. Entry does not identify that input. On starter’s rectangular input, reconstructed mistake raises `RuntimeError`.

**Replacement `why` sentence:**

> For `x = [[1.0, 3.0], [6.0, 2.0]]`, dropping the singleton dimension produces `[[-1.0, -1.0], [4.0, -2.0]]` instead of `[[-1.0, 1.0], [2.0, -2.0]]`, because the row means align with columns.

### D4 — Independent drill requires fewer decisions than faded sibling

**Codes:** `DIFF_RUNG` — **minor**; `DIFF_LABEL` — **note**.  
**ID:** 1368.

**Exact quotes:**

- Answer: `return imgs.mean(dim=(1, 2))`
- Label: `"difficulty_label": "medium"`
- Faded q1366 answer: `return x.mean(dim=(-2, -1), keepdim=True)`
- q1366 label: `"difficulty_label": "easy"`

Q1368 requires one obvious reduction over image dimensions. Faded q1366 adds preservation of reduced dimensions. Ladder therefore fails stated decision-count ordering.

**Replacement prompt sentence, retaining task but moving to Faded/easy:**

> Return a 1-D tensor of shape `(b,)` containing the mean pixel value of each image in the float tensor `imgs`, whose shape is `(b, h, w)`.

Provide fully blanked scaffold after reassignment. Wording alone cannot fix rung.

### D5 — Three ordinary cases; no edge case

**Code:** `STMT_CASES` — **major**.  
**ID:** 1370.

**Exact quotes — complete input assignments:**

```python
x = t.tensor([[1.0, 9.0], [5.0, 0.0]])
x = t.tensor([[3.0, 1.0, 1.0], [3.0, 1.0, 1.0]])
x = t.tensor([[0.0, 0.0, 7.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
```

Shapes: `(2, 2)`, `(2, 3)`, `(3, 3)`. No singleton dimension; all inputs non-negative. Existing cases reject stated near miss, so `CORR_DECISIVE` passes independently.

**Replacement/additional example sentences:**

> For `x = [[-3.0], [-2.0]]`, return `-5.0`, because the only column totals `-5.0`.

> For `x = [[-4.0, -1.0, -7.0]]`, return `-1.0`.

Add corresponding graded cases.

## 2. Lesson pages — per page, per rubric code

Page: [kp-axis-reductions.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/numpy/kp-axis-reductions.md).

| Segment / section | Verdict | Codes |
|---|---|---|
| `Concept: axis= — the axis you name disappears` | **major** | `EXPL_INTRO`, `EXPL_INTERLEAVE`, `EXPL_TRANSFER` |
| `Concept: tuples of axes, and keepdims` | **major** | Same; q1369 overlap under D2 |
| Faded practice | **major** | `GIVE_STARTER` / D1; `GIVE_PROMPT` / L4 |
| Solo / Integrated practice directions | **major** | `GIVE_PROMPT` / L4 |
| Guided practice; Misconceptions | **pass within supplied text** | No additional demonstrated defect |

Concept prose: approximately **161** and **114** whitespace-delimited words; both pass `EXPL_LENGTH`. Both explain general rules before worked examples: `EXPL_GENERAL_FIRST` passes. Shape/broadcast reasoning supports `EXPL_WHY`; asserting blocks print, satisfying `EXPL_PRINTS`.

Both worked fences execute correctly. Backend accepts illustrated `axis` and `keepdims` aliases; **no `EXPL_CORRECT` syntax finding**. Earlier-KP symbol coverage and runtime example scheduling were not established from supplied files.

### L1 — Both worked examples lack prose introductions

**Code:** `EXPL_INTRO` — **major**.  
**Segments:** both.

**Exact quote — shared opening:**

````markdown
## Worked example

```python
import torch as t
````

Heading immediately enters code. Comments inside fence do not supply required opening prose.

**Replacement introductory sentences:**

- First segment:

  > This example compares column totals with row means in a `(2, 3)` matrix, showing how each choice determines the output shape.

- Second segment:

  > This example first totals each trailing two-dimensional slice of a four-dimensional tensor, then shows why row means need a singleton dimension when subtracting them from a matrix.

### L2 — Both worked examples exceed 16-line block limit

**Code:** `EXPL_INTERLEAVE` — **major**.  
**Segments:** both.

**Exact boundary quotes:**

| Segment | Block starts | Block ends | Lines, including blanks |
|---|---|---|---:|
| First | `import torch as t` | `assert row_means.tolist() == [2.0, 20.0]` | 18 |
| Second | `import torch as t` | `assert centered[0].tolist() == [-1.0, 0.0, 1.0]` | 22 |

Internal comments do not create prose between fences.

**Replacement bridge sentences:**

- Split first block after column-total calculation:

  > Each column now has one total because its entries were combined across rows; next, combine entries across columns to obtain one mean per row.

- Split second block after tuple reduction:

  > The totals have shape `(2, 3)` because each trailing `(2, 2)` slice contributes one number; next, keep each row mean at shape `(2, 1)` so subtraction matches rows.

Keep resulting fences at most 16 lines; synchronize wording if implementing L3’s changed example axes.

### L3 — Immediate faded tasks repeat worked-example moves

**Code:** `EXPL_TRANSFER` — **major**.  
**Lesson slots:** q220, q135. Their bank rows were not inspected.

**Exact quotes:**

| Slot | Worked example | Faded task / solution |
|---|---|---|
| q220 | `col_sums = x.sum(dim=0)` | `One sum per column.`; `return x.sum(dim=0)` |
| q135 | `totals = batch.sum(dim=(-2, -1))` | `Per-slice totals of a 4-D batch: collapse the LAST two axes in one call.`; `return x.sum(dim=(-2, -1))` |

Same rank, axis choice, reduction. Page does not require transfer beyond applying same move to another tensor.

**Replacement teaching sentences, preserving drill goals:**

- First worked example:

  > We will calculate the total and mean of each row, leaving one result per row.

  Change example calculations to row reductions; q220 then requires different axis.

- Second worked example’s tuple portion:

  > We will total corresponding entries across the two leading dimensions of a four-dimensional tensor, leaving the trailing dimensions intact.

  Demonstrate leading-axis reduction; q135 then requires trailing-axis transfer.

### L4 — Practice directions disclose choices being assessed

**Code:** `GIVE_PROMPT` — **major**.  
**Lesson slots:** 108, 135, 505, 1363, 1364, 1371, 1372.

Grouped once below. Findings concern lesson directions, not unseen bank rows or deliberately marked solution fences.

| Slot | Exact quote | Replacement sentence |
|---|---|---|
| q108 | `` `t.quantile` takes `dim=` `` | “Return one quantile value for each row.” |
| q135 | `collapse the LAST two axes in one call.` | “Return a tensor of shape `(a, b)` containing the total of each `(c, d)` slice of the input tensor.” |
| q505 | `Make every row sum to 1 — this is what keepdim was for.` | “Return a tensor in which each entry is divided by its row’s original total.” |
| q1363 | `the one to cross out is the LAST one` | “Return a tensor of shape `(n, r)` containing every row’s total in each input matrix.” |
| q1364 | `collapse the LEADING axis` | “Return one `(r, c)` tensor containing the average value at each corresponding position across the `n` matrices.” |
| q1371 | `the two keepdims do different jobs.` | “Return the average share of each column after each row’s entries have been expressed as fractions of that row’s total.” |
| q1372 | `A tuple of axes AND keepdim, in the one call` | “Return the images with each pixel divided by the total of its own image.” |

## 3. Systemic patterns

- **Scaffold mismatch — `GIVE_RUNG`, `GIVE_STARTER`:** bank supplies `return None`; lesson supplies partially exposed syntax. D1 groups all affected starters.
- **Instruction leaking into assessment — `GIVE_PROMPT`, `GIVE_EXAMPLE`:** lesson directions name axis choices/keywords; q1369 additionally reuses exact worked input and output. D2/L4.
- **Example structure repeats — `EXPL_INTRO`, `EXPL_INTERLEAVE`, `EXPL_TRANSFER`:** both segments enter oversized fences directly, then pair with matching faded moves. L1–L3.
- **Case correctness exceeds case completeness — `STMT_CASES`, `STMT_NEAR_MISS`:** 35 expected results correct; q1370 still lacks edge coverage; q1369’s near miss needs explicit input. D3/D5.
- **Difficulty follows labeling more than decisions — `DIFF_RUNG`, `DIFF_LABEL`:** q1368 asks less than faded q1366. D4.

## 4. Top 10 fixes with rewrites

Ordered by instructional impact. Examples below describe input/output behavior; they do not prescribe implementation. For faded slots, retain scaffold without an accompanying worked solution.

### 1. q1369 — Remove copied lesson example

**`GIVE_EXAMPLE` — major; `STMT_NEAR_MISS` — minor.**

> Return a tensor of shape `(r, c)` obtained by subtracting each row’s mean from every value in that row of the float tensor `x`, also shaped `(r, c)`.

**Example:** `[[2.0, 8.0], [3.0, 7.0]] → [[-3.0, 3.0], [-2.0, 2.0]]`.

Replace duplicated case and example literals; identify near-miss input explicitly.

### 2. Lesson: tuples / keepdims segment — Teach transfer before q135

**`EXPL_INTRO`, `EXPL_INTERLEAVE`, `EXPL_TRANSFER` — major.**

Introduce example in prose; demonstrate reduction across leading dimensions; split tuple reduction from singleton-dimension example.

**Replacement practice prompt for q135 slot:**

> Return a tensor of shape `(a, b)` containing the total of each `(c, d)` slice of the numeric tensor `x`, whose shape is `(a, b, c, d)`.

No axis values or call-count instruction.

### 3. Lesson: single-axis segment — Teach transfer before q220

**`EXPL_INTRO`, `EXPL_INTERLEAVE`, `EXPL_TRANSFER` — major.**

Demonstrate row totals and row means, with prose between short blocks.

**Replacement practice prompt for q220 slot:**

> Return a 1-D tensor of shape `(c,)` containing the total of each column of the numeric tensor `x`, whose shape is `(r, c)`.

Column task now changes axis from worked example.

### 4. q1363 — Restore faded scaffold; remove axis answer

**`GIVE_RUNG`, `GIVE_STARTER`, `GIVE_PROMPT` — major.**

> Return a tensor of shape `(n, r)` containing the total of every row in the float tensor `x`, whose shape is `(n, r, c)`. Entry `[i, j]` must contain the total of row `j` in matrix `i`.

Use fully blanked one-keyword scaffold from D1.

### 5. q1364 — Restore faded scaffold; remove leading-axis instruction

**`GIVE_RUNG`, `GIVE_STARTER`, `GIVE_PROMPT` — major.**

> Return a tensor of shape `(r, c)` containing the average value at each corresponding position across the `n` matrices in the float tensor `x`, whose shape is `(n, r, c)`.

Use fully blanked one-keyword scaffold.

### 6. q1365 — Preserve aid without revealing syntax

**`GIVE_RUNG`, `GIVE_STARTER` — major.**

> Return the mean of each column of the float tensor `x`, whose shape is `(r, c)`, as a tensor of shape `(1, c)`.

Use fully blanked two-keyword scaffold. Existing prompt’s mathematical contract already sound.

### 7. q1366 — Preserve aid without revealing syntax

**`GIVE_RUNG`, `GIVE_STARTER` — major.**

> Return a tensor of shape `(n, 1, 1)` containing the mean value of each `(h, w)` slice of the float tensor `x`, whose shape is `(n, h, w)`. Entry `[i, 0, 0]` must contain the mean of slice `i`.

Use fully blanked two-keyword scaffold.

### 8. q1370 — Add singleton, negative cases

**`STMT_CASES` — major.**

> Return the largest column total of the float tensor `x`, whose shape is `(r, c)`, as a Python float.

**Constraints:** `r ≥ 1`, `c ≥ 1`; finite entries.

**Examples:** `[[-3.0], [-2.0]] → -5.0`; `[[-4.0, -1.0, -7.0]] → -1.0`.

Add both as graded cases; retain existing cases.

### 9. q1372 lesson direction — Remove tuple/keyword recipe

**`GIVE_PROMPT` — major. Bank prompt passes.**

> Return a tensor of shape `(b, h, w)` in which every pixel of the float tensor `imgs`, also shaped `(b, h, w)`, is divided by the total of its own image.

**Constraints:** positive dimensions; non-negative pixels; positive total for each image.

Remove lesson’s “tuple of axes AND keepdim” direction.

### 10. q1371 lesson direction — Describe shares without retained-axis recipe

**`GIVE_PROMPT` — major. Bank prompt passes.**

> Return a tensor of shape `(1, c)` containing the average share of each column across the rows of the float tensor `counts`, whose shape is `(r, c)`. An entry’s share is its value divided by its own row’s total.

**Constraints:** positive dimensions; non-negative entries; positive total for each row.

Remove lesson’s “two keepdims” direction.

---

Git: MODERATE — repo=Delta-Drills-Local branch=main; staged=0, unstaged=12, untracked=6, conflicts=0. Unchanged from baseline; unrelated changes preserved.