## 1. Drills — per-id findings

Reviewed **1654–1689** plus **nine lesson pages owning those ids**. Concurrent changes incorporated; resolved findings excluded. No files written.

Verification: **141/141 supplied cases pass** in backend Python environment; **56/56 executable lesson blocks pass** with lesson output capture supplied. Additional executed near misses expose grading gaps in **eight drills**.

**Verdicts: all 36 major.** Every opening violates the rubric’s literal `STMT_TASK` requirement. **Whole-item passes: none.** Twenty-two drills have no additional finding: **1654–1657, 1659, 1662, 1664, 1666–1670, 1672–1676, 1678–1680, 1687–1688.**

### D1. Input metadata precedes task — `STMT_TASK`, major

Affected: **1654–1689**. Each opening describes inputs instead of stating the function’s goal.

Below: exact opening excerpts → replacement opening sentences. Retain existing input details as short constraints; remove duplicated task wording.

| Id | Exact opening excerpt | Replacement sentence |
|---|---|---|
| 1654 | “`z` is a 1-D int tensor of even length.” | Return the values of `z` with its two halves exchanged, together with its unchanged original values, as a tuple of lists. |
| 1655 | “`z` is a 2-D int tensor with at least 3 rows and 3 columns.” | Return a copy of `z` with its interior rotated 180 degrees and its border unchanged, together with the original, as a tuple of nested lists. |
| 1656 | “`z` is a 2-D int tensor.” | Return a copy of `z` with rows numbered 0, 2, 4, and so on reversed, together with the unchanged original, as a tuple of nested lists. |
| 1657 | “`x` is a 1-D int tensor of length `n`” | Return the values remaining after removing `k` entries from each end of `x`, in reverse order, as a list. |
| 1658 | “`x` is a 1-D float tensor.” | Return the first index whose value is farthest from the mean of `x`. |
| 1659 | “`x` is a 1-D float tensor with at least one nonzero entry.” | Return the first index with greatest absolute value and that value’s sign, as a tuple of Python integers. |
| 1660 | “`x` is a 1-D float tensor and `v` a float.” | Return the indices nearest to and farthest from `v`, choosing the first index for each tie. |
| 1661 | “`x` is a 1-D float tensor with at least 2 entries” | Return a copy of `x` with its first greatest-magnitude entry replaced by the mean of the remaining entries, together with the original, as a tuple of lists. |
| 1662 | “`A` is a square float tensor of shape `(n, n)`” | Return the Python float equal to the sum of `x[i] * A[i, j] * x[j]` over all valid pairs `i, j`. |
| 1663 | “`A` is an invertible float tensor of shape `(n, n)`” | Return the solution of `A x = b` as a list, together with a Boolean indicating whether its residual satisfies the stated tolerance. |
| 1664 | “`A` and `B` are invertible float tensors of shape `(n, n)`” | Return the vector `y` for which applying `B` and then `A` produces `c`, as a list of floats. |
| 1665 | “`A` and `B` are float tensors of the same square shape `(n, n)`.” | Return the matrix product, the position-by-position product, and whether their top-left entries are exactly equal. |
| 1666 | “`a` and `b` are 1-D int tensors of the same length `n`.” | Return a list containing each entry of `a`, the corresponding entry of `b`, and a zero, in that order for every position. |
| 1667 | “`a` is a 2-D int tensor of shape `(n, m)`” | Return the rows of `a`, followed by `sep` as one row, followed by the rows of `b`, as a nested list. |
| 1668 | “`a` and `b` are 2-D int tensors of the same shape `(n, m)`.” | Return a nested list with `a` in the top-left and bottom-right blocks and `b` in the other two blocks. |
| 1669 | “`a` and `b` are 1-D int tensors of the same length `n`.” | Return both a flat list containing `a` followed by `b` and a nested list containing them as two rows. |
| 1670 | “`x` is a 1-D int tensor and `p ≠ q` are ints.” | Return a copy with every occurrence of `p` exchanged with `q`, together with the unchanged original, as a tuple of lists. |
| 1671 | “`x` is a 1-D int tensor and `v` an int.” | Return a list with every occurrence of `v` or `-v` replaced by zero, together with the number of original entries matching neither value. |
| 1672 | “`x` is a 1-D int tensor.” | Return a copy with even values doubled and odd values negated, together with the unchanged original, as a tuple of lists. |
| 1673 | “`x` is a 1-D int tensor with at least 2 entries.” | Return a copy with each entry equal to its original left neighbor replaced by zero, together with the unchanged original, as a tuple of lists. |
| 1674 | “`n` is an odd int ≥ 3.” | Return `n` floats that rise evenly from zero to one at the middle position and fall evenly back to zero. |
| 1675 | “`m` is a positive int.” | Return a nested list of shape `(2, m)` containing consecutive floating-point values from zero, read across the first row and then the second. |
| 1676 | “`n, m` are ints of at least 2” | Return a zero grid with an evenly increasing column and a counting row, giving the row priority at their intersection, together with the intersection value. |
| 1677 | “`x` is a 1-D float tensor of length `n`” | Overwrite the last `k` entries of `x` with evenly spaced values descending from one to zero, then return its updated values as a list. |
| 1678 | “`n, m` are ints and `h ≤ n`, `w ≤ m` are block sizes” | Return a zero grid with a constant top-left block and repeated copies of `v` in the bottom-right block, giving the bottom-right block priority wherever they overlap. |
| 1679 | “`n` is an int, `m` an even int” | Return a zero grid containing `v` in the even-numbered columns of row `r1` and its reversed values in the odd-numbered columns of row `r2`. |
| 1680 | “`src` is a 2-D float tensor of shape `(n, m)`” | Return `src` surrounded by a one-cell zero margin, with the four outer corners set to `k`, as a nested list. |
| 1681 | “`z` is a 2-D float tensor of shape `(n, m)`” | Modify `z` by filling its last `k` columns with `s`, then replacing its last column with its first column’s current values. |
| 1682 | “`x` is a float tensor of shape `(m, n)` with no all-zero row.” | Return the first row with greatest Euclidean length, rescaled to length one, as a list of floats. |
| 1683 | “`x` is a float tensor of shape `(m, n)` with no all-zero row” | Return every row rescaled to Euclidean length `L` while preserving its direction, as a nested list. |
| 1684 | “`x` is a float tensor of shape `(m, n)`.” | Return a Boolean list identifying rows whose Euclidean length differs from one by less than `1e-6`, together with their count. |
| 1685 | “`x` is a float tensor of shape `(m, n)`” | Return a nested list with every nonzero row rescaled to Euclidean length one and every zero row preserved. |
| 1686 | “`x` is a float tensor of shape `(b, c)`” | Return each row’s requested value minus that row’s mean, as a list of floats. |
| 1687 | “`x` is a float tensor of shape `(b, c)`” | Return each row’s two requested values in request order, together with their first-minus-second differences, as a tuple of lists. |
| 1688 | “`x` is a float tensor of shape `(b, c)`” | Return a copy with each row’s requested entry exchanged with its first entry, together with the unchanged original, as a tuple of nested lists. |
| 1689 | “`x` is a float tensor of shape `(b, c)`” | Return one value per column, using that column’s requested row index, as a list of floats. |

### D2. Incorrect implementations pass every stored case — `CORR_DECISIVE`, major

Affected: **1661, 1665, 1671, 1677, 1681, 1682, 1684, 1685**. Each near miss below was executed; each passes **all four** existing cases.

Prompt rewrites alone cannot repair these findings. Add the specified grading checks.

| Id | Exact quote | Accepted near miss | Replacement sentence and decisive check |
|---|---|---|---|
| 1661 | “largest absolute value (first on ties)” | Selects the **last** tied maximum. Existing tied input `[4.0, 4.0]` produces the same result either way. | **Rewrite:** “If several entries share the greatest absolute value, replace only the one with the smallest index.” **Add:** `[-4, 4, 1] → ([2.5, 4, 1], [-4, 4, 1])`. |
| 1665 | “exactly the same value in row 0, column 0” | Uses approximate equality for `same_corner`. Existing cases cannot distinguish it from exact equality. | **Rewrite:** “Set `same_corner` to `True` only when the two computed values are exactly equal.” **Add:** `A=[[1,1],[0,1]]`, `B=[[1,0],[2**-20,1]]`; `same_corner=False`, although approximate equality reports `True`. |
| 1671 | “`x` must not change.” | Replaces `out = x.clone()` with `out = x`, mutating the caller’s input. | **Rewrite:** “Produce the returned list without changing any entry of the supplied tensor `x`.” **Check:** save `x` before the call and assert its values remain identical afterward. |
| 1677 | “Overwrite the LAST `k` entries of `x` in place” | Clones `x`, updates the clone, returns its values; caller’s `x` remains untouched. | **Rewrite:** “Update the supplied tensor itself and return its updated values as a list.” **Check:** assert `x.tolist() == result` after the call; the existing first fixture distinguishes the clone-only implementation. |
| 1681 | “In place: set every entry in the last `k` columns of `z` to `s`” | Clones `z` before both writes. All returned values remain correct. | **Rewrite:** “Perform both updates on the supplied tensor `z`, then return its final values as a nested list.” **Check:** assert `z.tolist() == result` after the call. |
| 1682 | “the row of `x` with the greatest length” | Chooses the row with the largest absolute **coordinate**, then normalizes that row correctly. | **Rewrite:** “Choose the row with greatest Euclidean length, defined as the square root of the sum of its squared coordinates.” **Add:** `[[5,0],[4,4]] → [0.70710678,0.70710678]`. |
| 1684 | “within `1e-6` of `1.0`” | Tests whether the length is exactly one. | **Rewrite:** “A row qualifies when the absolute difference between its Euclidean length and one is strictly less than `1e-6`.” **Add:** `[[1.0000005,0],[1.000002,0]] → ([True,False],1)`. |
| 1685 | “every nonzero row scaled to length 1” | Replaces every denominator below one with one, leaving short nonzero rows unnormalized. | **Rewrite:** “Rescale every nonzero row to length one, including rows whose original length is less than one.” **Add:** `[[0.3,0.4],[0,0]] → [[0.6,0.8],[0,0]]`. |

### D3. Undefined empty-input domains — `STMT_INPUT`, major

Affected: **1658, 1660, 1665, 1682**.

| Id | Exact quote | Defect | Replacement sentence |
|---|---|---|---|
| 1658 | “`x` is a 1-D float tensor.” | Empty vector satisfies the stated input description; no requested index exists. Reference answer raises. | “`x` is a nonempty 1-D float tensor.” |
| 1660 | “`x` is a 1-D float tensor and `v` a float.” | Same empty-vector problem. | “`x` is a nonempty 1-D float tensor and `v` is a float.” |
| 1665 | “the same square shape `(n, n)`” | `n=0` permits no top-left entry. | “`A` and `B` are float tensors of shape `(n, n)`, where `n ≥ 1`.” |
| 1682 | “shape `(m, n)` with no all-zero row” | Zero rows satisfy “no all-zero row” vacuously, but no winning row exists. | “`x` is a float tensor of shape `(m, n)`, where `m ≥ 1`, `n ≥ 1`, and every row contains a nonzero entry.” |

### D4. q1671’s count contradicts its zero case — `STMT_OUTPUT`, major

**Quote:** “the int number of entries that were NOT changed.”

For `x=[0,1,0]`, `v=0`, **no numerical value changes**, yet expected `kept` is **1**. Grading counts unmatched entries, not unchanged values.

**Replacement:** “Return `(out, kept)`, where `out` is the list after replacing every occurrence of `v` or `-v` with zero, and `kept` counts original entries equal to neither value.”

Also **`STMT_NEAR_MISS`, minor**:

**Quote:** “No entry equals both `v` and `-v` at once”.

False when `v=0`.

**Replacement:** “For nonzero `v`, no entry equals both values, so an AND condition misses every match; an OR condition also handles `v=0`, when both comparisons select the zeros.”

### D5. Undefined numerical terms — `STMT_TERMS`, minor

| Ids | Exact quote | Replacement sentence |
|---|---|---|
| 1663 | “within floating-point tolerance” | “Set `ok` to `True` when every residual component satisfies `abs((A x)[i] - b[i]) ≤ 1e-8 + 1e-5 * abs(b[i])`.” |
| 1682–1685 | “length” | “A row’s length means its Euclidean length: the square root of the sum of its squared coordinates.” |

### D6. Rung does not increase decisions — `DIFF_RUNG`, minor

Affected: **1663, 1683, 1686**.

| Id | Exact lesson evidence | Defect | Replacement sentence |
|---|---|---|---|
| 1663 | Solo q512: “Solve a @ x = b, then verify a @ x really does reproduce b.” | Integrated drill asks for the same solve-and-check decisions. | “Solo practice: return the solution of `A x = b` and whether it satisfies the stated residual tolerance.” |
| 1683 | Faded q996: “Return each row rescaled to length three” | Replacing fixed `3` with parameter `L` adds no substantive decision. | “Solo practice: return every row rescaled to the supplied positive length `L`, preserving its direction.” |
| 1686 | Faded q1022: “Return requested values minus each row’s mean” | Integrated task repeats the same selection and subtraction. | “Solo practice: return each requested value minus its row’s mean.” |

### D7. q1689 uses one KP idea — `GIVE_RUNG`, major; `DIFF_RUNG`, minor

**Quote:** “Return a plain list of `c` floats: for each column `j`, the value of `x` at row `ids[j]`, column `j`.”

This is paired indexing with axes exchanged. It provides useful transfer, but no second indexed-selection idea.

**Replacement:** “Solo practice: return one requested value per column, preserving column order.”

Move its ownership to Solo, or add another substantive decision before retaining Integrated.

**Passing checks, compact:** `CORR_RUNS` **1654–1689** on supplied cases; observed return structures compatible with expectations; all case sets have differing expected outputs. Backend comparison already applies floating-point tolerance, so long decimal literals alone are **not** `CORR_TYPES` findings.

## 2. Lesson pages — per page, per rubric code

All nine pages: **major**. Shared defects appear once in Section 3, with affected ids, exact quotes, and replacements.

| Page | Findings by rubric code and segment |
|---|---|
| [tensors/kp-slicing-views.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/tensors/kp-slicing-views.md) | **`EXPL_TRANSFER`, major:** “Write through a slice” / q231; “Copy before changing values” / q74 — S4. **`EXPL_LENGTH`, minor:** all six Concept segments — S5. |
| [tensors/kp-argmin-argmax.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/tensors/kp-argmin-argmax.md) | **`EXPL_INTRO`, major:** all three worked examples — S1. **`GIVE_STARTER`, major:** q219, q98 — S2. **`EXPL_TRANSFER`, major:** q38, q219, q98 — S4. **`EXPL_LENGTH`, minor:** “index as a handle for surgery”; “closest-to-target” — S5. **`EXPL_WHY`, major:** distance transformation, detailed below. **`EXPL_CORRECT`, major:** sorting claim, below. |
| [tensors/kp-linalg-basics.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/tensors/kp-linalg-basics.md) | **`EXPL_INTRO`, major:** both worked examples — S1. **`EXPL_TRANSFER`, major:** q239, q107 — S4. **`EXPL_CORRECT`, major:** residual check described as detecting ill-conditioning, below. **`DIFF_RUNG`, minor:** q1663 repeats Solo q512 — D6. |
| [tensors/kp-stack-concat-interleave.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/tensors/kp-stack-concat-interleave.md) | **`GIVE_STARTER`, major:** q974, q84, q976 — S2. **`EXPL_TRANSFER`, major:** q974, q975, q84, q976 — S4. **`EXPL_CORRECT`, major:** universal storage-layout claim, below. |
| [tensors/kp-boolean-masking.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/tensors/kp-boolean-masking.md) | **`EXPL_INTRO`, major:** all three worked examples — S1. **`EXPL_TRANSFER`, major:** q236, q52, q12 — S4. **`EXPL_CORRECT`, major:** nonexistent `.copy()` method; overgeneralized mask-output shape, below. **`EXPL_LENGTH`, minor:** “using a mask — count and filter” — S5. |
| [pytorch/kp-out-argument.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/pytorch/kp-out-argument.md) | **`EXPL_TRANSFER`, major:** q798 repeats the worked column fill — S4. **`EXPL_LENGTH`, minor:** sole Concept segment — S5. |
| [pytorch/kp-slice-assignment.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/pytorch/kp-slice-assignment.md) | **`GIVE_STARTER`, major:** q810 names its blanked method — S2. **`EXPL_CORRECT`, major:** rebinding described as allocation, below. |
| [pytorch/kp-row-normalization.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/pytorch/kp-row-normalization.md) | **`GIVE_RUNG`, major:** Faded q993–q996 have bare `pass` starters — S3. **`EXPL_TRANSFER`, major:** q993, q995 — S4. **`DIFF_RUNG`, minor:** Integrated q1683 repeats Faded q996’s decisions — D6. |
| [pytorch/kp-indexed-selection.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/pytorch/kp-indexed-selection.md) | **`GIVE_RUNG`, major:** Faded q1019–q1022 have bare `pass` starters — S3; Integrated q1689 uses one idea — D7. **`EXPL_TRANSFER`, major:** q1020, q1019 — S4. **`DIFF_RUNG`, minor:** q1686 repeats Faded q1022 — D6; q1689 — D7. |

**Argmin/argmax — “closest-to-target”: `EXPL_WHY`, major.**

**Quote:** “build the quantity you want minimized, then ask where its minimum sits.”

The page presents `abs(z - target)` without explaining why signed subtraction alone fails.

**Replacement:** “Subtracting the target gives signed offsets, absolute value turns those offsets into distances on either side of the target, and the smallest distance identifies the nearest entry; use that index on the original vector when the requested output is its value.”

**Argmin/argmax — worked explanation and Misconceptions: `EXPL_CORRECT`, major.**

**Quote:** “sorting is O(n log n) and loses positions”.

Sorting can retain original indices.

**Replacement:** “Sorting can retain original indices, but it orders every entry; finding one nearest position only requires selecting the smallest distance.”

**Linear algebra — solver worked example: `EXPL_CORRECT`, major.**

**Quote:** “catches both wrong answers and ill-conditioned systems.”

A small residual does not establish that a system is well-conditioned.

**Replacement:** “Substituting the solution checks its residual; an ill-conditioned system can still have a small residual while its solution remains sensitive to small input errors.”

**Stacking — “Pair and Unroll”: `EXPL_CORRECT`, major.**

**Quote:** “PyTorch stores and reads tensors in row-major order”.

Not universal: views can have different strides. Logical flattening order should be distinguished from physical storage.

**Replacement:** “Flattening reads the paired tensor in logical row order, visiting both values in row `i` before row `i+1`; tensor views can have different physical storage strides.”

**Boolean masking — “combining masks and masked assignment”: `EXPL_CORRECT`, major.**

**Quote:** “`.copy()` first”.

`torch.Tensor.copy()` does not exist; executing it raises `AttributeError`.

**Replacement:** “When the input must remain unchanged, create an independent tensor with `.clone()` before assigning through its mask.”

**Boolean masking — “count and filter” and Misconceptions: `EXPL_CORRECT`, major.**

**Quote:** “`x[mask]` returns a 1-D array”.

True for a full-shape mask selecting individual entries; false for a row mask selecting whole rows. The page’s own q85 uses row filtering.

**Replacement:** “A mask with the same shape as `x` selects individual entries into a 1-D copy in their original order, while a 1-D mask over the first axis selects whole rows and preserves each row’s columns; count the selected positions separately when the task also asks how many matched.”

**Slice assignment — Misconceptions: `EXPL_CORRECT`, major.**

**Quote:** “Rebinding (`x = ...`) is what makes new tensors.”

Rebinding can merely create another reference: `x = y`.

**Replacement:** “Assignment binds a name to its right-hand-side object; `x = y` shares the existing tensor, while `x = y.clone()` binds `x` to an independent copy.”

## 3. Systemic patterns

### S1. Worked examples start with code — `EXPL_INTRO`, major

Affected: **three pages, eight segments**.

**Exact structural quote:** `## Worked example` followed immediately by `` ```python ``. No introductory prose.

Insert the following replacement introductions:

| Page / segment | Replacement sentence |
|---|---|
| Argmin/argmax — index, not value | Find the first minimum in a vector containing a repeated minimum, then compare its index with the value stored there. |
| Argmin/argmax — index as handle | Replace the first maximum in a copy and inspect both tensors to verify that the input survives. |
| Argmin/argmax — closest-to-target | Compare each value’s distance from a target, then use the smallest distance’s position to recover the original value. |
| Boolean masking — comparison is mask | Build two masks over the same vector to show how each comparison produces one Boolean per entry. |
| Boolean masking — count and filter | Use one mask to count matching entries and collect their values in their original order. |
| Boolean masking — combining masks | Negate values inside an open interval while preserving the original tensor. |
| Linear algebra — two multiplications | Apply elementwise multiplication and matrix multiplication to the same operands, then compare their values and shapes. |
| Linear algebra — solve | Solve a diagonal linear system, then substitute the result to check that it reproduces the right-hand side. |

### S2. Faded starters reveal declared new syntax — `GIVE_STARTER`, major

Affected lesson excerpts: **q219, q98, q974, q84, q976, q810**.

| Page / id | Exact quote | Replacement sentence and scaffold change |
|---|---|---|
| Argmin/argmax / q219 | `result[t.argmax(x)] = 0.0` | **Sentence:** “Return a copy with its first largest entry replaced by zero.” Keep the prerequisite copy operation visible; change the selection line to `result[t._____(x)] = 0.0`. |
| Argmin/argmax / q98 | `return int(t.argmin(t._____(z - target)))` | **Sentence:** “Return the first index nearest to the target.” Blank both new calls: `return int(t._____(t._____(z - target)))`. |
| Stacking / q974 | `return t._____([a, b], dim=0)` | **Sentence:** “Return the two matrices joined top to bottom.” Blank the declared axis keyword and its value: `return t._____([a, b], _____=_____)`. |
| Stacking / q84 | `return t.stack([a, b], dim=0)._____(dim=0)` | **Sentence:** “Return the position-by-position average of the two matrices.” Blank the stack call and axis choices instead of leaving the KP’s new operation complete. |
| Stacking / q976 | `return t.column_stack((a, b))._____()` | **Sentence:** “Return alternating values from the two vectors, beginning with `a`.” Use `return t._____((a, b))._____()` so both declared new operations require recall. |
| Slice assignment / q810 | `"""One row overwritten with copy_."""` | **Sentence:** “Replace the selected row with the supplied row values.” Remove `copy_` from the starter docstring; retain its existing method blank. |

### S3. Faded starters provide no scaffold — `GIVE_RUNG`, major

Affected: **row-normalization q993–q996; indexed-selection q1019–q1022**.

**Exact quote in all eight starters:** `pass`.

These are from-scratch tasks labeled Faded.

| Page / ids | Replacement sentence | Concrete scaffold |
|---|---|---|
| Row normalization / q993 | Complete the scaffold to return one length for each requested vector. | `return x._____(_____=_____)` |
| Row normalization / q994 | Complete the reduction to return squared row lengths with the reduced axis retained. | `return (x * x)._____(_____=_____, _____=_____)` |
| Row normalization / q995 | Complete the scaffold to preserve direction while making each requested vector unit length. | `return x / x._____(_____=_____, _____=_____)` |
| Row normalization / q996 | Complete the scaffold to give every row length three. | `return 3 * x / x._____(_____=_____, _____=_____)` |
| Indexed selection / q1020 | Complete the two index slots to return one requested entry per vector. | `return x[_____, _____]` |
| Indexed selection / q1022 | Complete the selection, then subtract each row’s mean. | `return x[_____, _____] - x.mean(dim=1)` |
| Indexed selection / q1019 | Complete the method and axis choice to return the requested values with the required singleton axis. | `return x._____(_____, _____)` |
| Indexed selection / q1021 | Complete the selection before reducing its requested values to a scalar. | `return x._____(_____, ids[:, None]).sum()` |

Axis choices must match any transfer changes in S4.

### S4. Faded practice repeats the demonstrated move — `EXPL_TRANSFER`, major

Affected: **19 lesson excerpts**. Parameterizing example literals does not create the required transfer.

Replacement prompts below change the selection criterion, axis, input organization, or application. Corresponding starters and cases must change with them.

| Page / id | Exact practice quote | Replacement sentence |
|---|---|---|
| Argmin/argmax / q38 | “Index of the smallest element” | Return the first index containing the largest value. |
| Argmin/argmax / q219 | “Largest entry replaced with 0” | Return a copy whose first smallest entry has been replaced by the original largest value. |
| Argmin/argmax / q98 | “The INDEX of the entry closest to target” | Return the first index whose value is farthest from the target. |
| Boolean masking / q236 | “Boolean array marking entries strictly greater than a threshold.” | Return a Boolean tensor marking entries unequal to the supplied value. |
| Boolean masking / q52 | “Number of True entries in a boolean array” | Return the number of False entries as a Python integer. |
| Boolean masking / q12 | “Entries strictly between 3 and 8 negated, input untouched.” | Return a copy with values outside the closed interval `[lo, hi]` negated and values inside it unchanged. |
| Linear algebra / q239 | “Matrix product of shapes (m, k) and (k, n).” | Return the length-`m` vector obtained by applying an `(m, k)` matrix to a length-`k` vector. |
| Linear algebra / q107 | “Solve the linear system a @ x = b” | Return the solutions for two right-hand sides supplied as the columns of one matrix. |
| Stacking / q974 | “the row counts add up” | Remove the middle row of an odd-height matrix and return the remaining rows in their original order. |
| Stacking / q975 | “The same kind of seam along the other axis” | Return the two matrices side by side with the supplied separator column between them. |
| Stacking / q84 | “Elementwise average of two same-shape arrays” | Return the position-by-position average of the first and last matrices in a three-dimensional tensor. |
| Stacking / q976 | “Pair-and-unroll” | Return alternating entries from columns 1 and 0 of a two-column matrix, beginning with column 1. |
| Out argument / q798 | “n evenly spaced values from lo to hi, written into column col.” | Fill every second position of the selected column with evenly spaced values from `lo` to `hi`, leaving the remaining canvas entries zero. |
| Row normalization / q993 | “Return row lengths” | Return one Euclidean length per column. |
| Row normalization / q995 | “Return unit rows” | Return each column rescaled to length one; every input column contains a nonzero entry. |
| Indexed selection / q1020 | “Return requested values as a vector, shape (b,).” | Return one requested row entry per column, preserving column order. |
| Indexed selection / q1019 | “Return requested value per row, shape (b,1).” | Return one requested row entry per column as a tensor of shape `(1, c)`. |
| Slicing / q231 | “Fill the selected range in place.” | Fill a matrix’s interior with the supplied value while preserving its border. |
| Slicing / q74 | “Replace every step-th value in a copy; preserve the input.” | Return a copy with every second column replaced by the supplied value, preserving the input. |

### S5. Concept prose misses length bounds — `EXPL_LENGTH`, minor

Counts below use whitespace-separated prose, excluding fenced code. Required range: **80–250 words**.

| Page / segment | Count | Exact quote | Replacement |
|---|---:|---|---|
| Slicing — Choose a stretch | 65 | “Predict the three numbers below, then run.” | Before running the slice, identify the first selected position and the excluded boundary, then count the positions between them to predict both the selected values and the result’s length. |
| Slicing — Select rows or columns | 52 | “A colon on its own means “all.” First, look at the matrix:” | Choose each axis independently: an integer selects one position and removes that axis, while a slice selects a range and keeps it, so one column can appear either as a flat vector or as a two-dimensional result with one column. |
| Slicing — Reverse along an axis | 50 | “Predict which value moves to the front:” | For each named axis, the last position becomes the first and all other axes keep their order, so reversing a vector changes entry order, reversing matrix columns mirrors each row, and reversing matrix rows moves whole rows without reversing their entries. |
| Slicing — Quarter-turn | 47 | “Start with this small rectangle:” | Follow one corner and one edge through the turn, then predict the new row count, column count, and top row; a rectangular input swaps its height and width, while a square input keeps its shape despite moving its values. |
| Slicing — Write through a slice | 53 | “First, select the middle two values:” | Track which original index each view position reaches before assigning through it, because a write changes that shared location and becomes visible through every other view, while values outside the selected region keep their previous contents. |
| Slicing — Copy before changing | 46 | “We will mark every second value.” | Create the independent copy before any assignment, then select the positions within that copy; this order preserves the original values for later reads and lets you verify separately that the selected positions changed, the unselected positions survived, and the input stayed intact. |
| Argmin/argmax — surgery | 58 | “It reads cleanest and never backfires.” | Keeping the source separate also matters when a later computation needs the value being replaced, since reading that value from the edited result would use new data and change the meaning of the next step. |
| Argmin/argmax — closest-to-target | 63 | “the *transformed* array chooses the index” | Use the expanded distance explanation under `EXPL_WHY` in Section 2 to replace this closing explanation. |
| Boolean masking — count and filter | 65 | “`x[mask]` returns a 1-D array” | Use the full-shape-mask versus row-mask replacement in Section 2; it supplies the missing distinction and sufficient explanatory prose. |
| Out argument — sole Concept | 261 | “The target’s” is rendered in source as “The target's **shape must already match**” | Replace the mismatch paragraph with: “Before either constructor writes into a canvas view, match its shape and slot count to the generated result; with the mismatched slice below, PyTorch warns and resizes the view, causing values to land outside the intended column.” |

Across the report, the recurring failures are **metadata-first prompts (`STMT_TASK`)**, **cases that miss a required distinction (`CORR_DECISIVE`)**, **miscalibrated Faded assistance (`GIVE_STARTER`, `GIVE_RUNG`)**, and **example transcription (`EXPL_TRANSFER`)**. Passing reference answers does not resolve those defects.

## 4. Top 10 fixes with rewrites

Ranking prioritizes demonstrated false acceptance, then incorrect teaching and answer leakage. Grading repairs from D2 remain necessary alongside these prompts.

1. **q1671 — `STMT_OUTPUT`, `CORR_DECISIVE`; major.** Count definition and input preservation both fail.

   > Return `(out, kept)` for a 1-D integer tensor `x` and integer `v`: `out` is a Python list with every occurrence of `v` or `-v` replaced by zero, and `kept` counts original entries equal to neither value. Leave `x` unchanged.

   Decisive example: `[0,1,0]`, `v=0` → `([0,1,0],1)`; separately verify unchanged input.

2. **q1677 — `CORR_DECISIVE`; major.** Copy-only implementation passes an in-place task.

   > Overwrite the last `k` entries of the supplied 1-D float tensor `x` with evenly spaced values descending from `1.0` to `0.0`, preserving its earlier entries. Return the updated tensor’s values as a Python list; when `k=1`, the replacement value is `1.0`.
   >
   > Constraint: integer `k` satisfies `1 ≤ k ≤ len(x)`.

   Grade caller-visible mutation.

3. **q1681 — `CORR_DECISIVE`; major.** Same missing mutation check.

   > Modify the supplied float tensor `z` of shape `(n, m)` by setting its last `k` columns to `s`, then replacing its last column with the first column’s values after that update. Return its final values as a nested Python list of shape `(n, m)`.
   >
   > Constraints: `s` is a float; integer `k` satisfies `1 ≤ k ≤ m`.

   Grade caller-visible mutation, including `k=m`.

4. **q1661 — `CORR_DECISIVE`; major.** Current tie case cannot detect choosing the wrong occurrence.

   > Return `(out, original)` as two Python lists for a 1-D float tensor `x`: in `out`, replace the first entry with greatest absolute value by the mean of all other entries. Preserve every other value and leave `x` unchanged.
   >
   > Constraints: at least two entries; every entry lies between `-1000` and `1000`.

   Add `[-4,4,1] → ([2.5,4,1],[-4,4,1])`.

5. **q1665 — `CORR_DECISIVE`; major.** Exact equality needs a near-equal counterexample.

   > Return `(mat, elem, same_corner)` for float tensors `A` and `B` of shape `(n, n)`: `mat` and `elem` are nested lists containing their matrix product and position-by-position product. Set `same_corner` to `True` exactly when the two computed top-left values are equal.
   >
   > Constraint: `n ≥ 1`.

   Add the `2**-20` difference case from D2.

6. **q1682 — `CORR_DECISIVE`; major.** Largest-coordinate selection passes.

   > Return the first row of `x` with greatest Euclidean length, rescaled to length one, as a Python list of `n` floats. Euclidean length is the square root of the sum of squared coordinates.
   >
   > Constraints: `x` is a float tensor of shape `(m, n)`; `m,n ≥ 1`; every row contains a nonzero entry.

   Add `[[5,0],[4,4]] → approximately [0.70710678,0.70710678]`.

7. **q1684 — `CORR_DECISIVE`; major.** Exact comparison passes a tolerance task.

   > Return `(is_unit, count)` for a float tensor `x` of shape `(m, n)`: `is_unit` is a list of `m` Booleans, and `count` is the number that are `True`. A row qualifies when its Euclidean length—the square root of its squared-coordinate sum—differs from `1.0` by strictly less than `1e-6`.

   Add rows immediately inside and outside the tolerance.

8. **q1685 — `CORR_DECISIVE`; major.** No short nonzero row tests normalization below length one.

   > Return a nested Python list of shape `(m, n)` with every nonzero row of float tensor `x` rescaled to Euclidean length one and every all-zero row unchanged. Euclidean length is the square root of the sum of squared coordinates; do not divide by zero.

   Add `[[0.3,0.4],[0,0]] → [[0.6,0.8],[0,0]]`.

9. **Boolean-masking lesson — `EXPL_CORRECT`, `EXPL_TRANSFER`; major.** Incorrect copy API plus literal repetition of the worked interval task.

   Teaching replacement:

   > Create an independent tensor with `.clone()` before updating selected positions when the input must survive.

   Replacement Faded prompt:

   > Return a new 1-D integer tensor with values outside the closed interval `[lo, hi]` negated and all other values unchanged. Preserve the input tensor.
   >
   > Constraints: `lo` and `hi` are integers with `lo ≤ hi`.

   Example mapping for grading: `[-3,0,2,5]`, interval `[0,2]` → `[3,0,2,-5]`.

10. **Stacking lesson — `GIVE_STARTER`, `EXPL_TRANSFER`; major.** q84 supplies the new operation while repeating the demonstrated average.

    Replacement Faded prompt:

    > Return the position-by-position average of the first and last matrices in float tensor `x`, as a float tensor of shape `(r, c)`.
    >
    > Constraints: `x` has shape `(k, r, c)`; `k ≥ 2`.

    Blank the new operations and axis choices. Grading example: `[[[2,4]],[[90,90]],[[6,8]]] → [[4,6]]`; the middle matrix must not contribute.

Git: MODERATE — repo=Delta-Drills-Local branch=main; staged=0, unstaged=31, untracked=2, conflicts=0. Mixed scopes; no changes made by this review.