## 1. Drills — per-id findings

Standard: [CONTENT_RUBRIC.md](/home/stellar-thread/Applications/Delta-Drills-Local/scripts/CONTENT_RUBRIC.md). **Seven grouped findings: six major, one minor.**

| Drill | Verdict | Findings |
|---|---|---|
| 1729 | **major** | D1: input types missing; D2: faded scaffold missing |
| 1730 | **major** | D1, D2; D3: prompt reveals method; D4: starter reveals method/axis; D5: truncated near miss |

**Passing checks — 1729, 1730:** `STMT_TASK`, `STMT_OUTPUT`, `STMT_SELF_CONTAINED`, `STMT_TERMS`, `STMT_CASES`, `STMT_LENGTH`, `STMT_VOICE`, `GIVE_EXAMPLE`, `GIVE_TESTS`, `CORR_RUNS`, `CORR_DECISIVE`, `CORR_TYPES`.

- Both answers pass **4/4 cases** in backend virtualenv.
- Each case set covers `(nr, ns)` values `(2,3)`, `(1,1)`, `(3,2)`, `(2,2)`; four distinct expected outputs.
- Constant first-case output fails three cases; returning either input unchanged fails all four.
- Reconstructed near misses fail three cases each.
- Backend comparator converts tensors, checks shapes, compares floats with tolerance.
- **1729 additionally passes:** `GIVE_PROMPT`, `STMT_NEAR_MISS`.
- `CORR_VISUAL`: inapplicable; both artifacts marked `stdout`, submissions use function mode.

### D1 — `STMT_INPUT` · major · IDs 1729, 1730

**Exact quote, both prompts:**

> `r`: rays `(nr, 2, 3)` as `[origin, direction]`; `s`: segments `(ns, 2, 3)` as `[start, end]`.

Shapes/layout supplied; argument types absent. “Floating-point PyTorch tensor” describes **return value**. Untyped starters supply no missing input contract.

**Replacement:**

> Inputs `r` and `s` are floating-point PyTorch tensors of shapes `(nr, 2, 3)` and `(ns, 2, 3)`, containing `[origin, direction]` and `[start, end]`, respectively; their coordinate order is `(x, y, z)`.

### D2 — `GIVE_RUNG` · major · IDs 1729, 1730

**Exact ownership quote:**

> faded: [1105, 1106, 1107, 1108, 1729, 1730]

**Exact starter statement, both drills:**

```python
    pass
```

Both require writing entire implementation. Rubric defines faded practice as **scaffold with blanks**.

Same defect appears in lesson’s starters for **1105, 1106, 1107, 1108**. Those occurrences belong to page review; their bank rows were not read or graded.

**Replacement instruction:**

> Complete the blanks in `solve(r, s)` to return the requested tensor.

Concrete replacement scaffolds for reviewed drills:

**1729**

```python
import torch as t

def solve(r, s):
    x = r[:, 0, :2]
    y = s[:, 1, :2] - s[:, 0, :2]
    return _____
```

**1730**

```python
import torch as t

def solve(r, s):
    x = r[:, 1, :2]
    y = s[:, 0, :2] - s[:, 1, :2]
    x = _____
    y = _____
    return _____
```

Known input extraction remains; pair-axis construction remains learner’s decision.

### D3 — `GIVE_PROMPT` · major · ID 1730

**Exact quote:**

> Both columns must be brought to the full pair shape before they are stacked.

Prescribes assembly sequence and stacking operation. Output contract already specifies required columns and pair table.

**Replacement:**

> Within each matrix, rows correspond to x and y coordinates; columns contain the ray direction and segment `start − end`, respectively.

### D4 — `GIVE_STARTER` · major · ID 1730

**Exact quote:**

> """The per-pair 2x2 matrix [direction, start - end], stacked on the last axis."""

“Stacked on the last axis” supplies operation and axis choice.

**Replacement docstring:**

```python
"""Return the required 2×2 matrix for every ray–segment pair."""
```

### D5 — `STMT_NEAR_MISS` · minor · ID 1730

**Exact `wrong_examples.output`:**

> raises RuntimeError: stack expects each tensor to be equal size, but got [2, 1, 2] at entry 0 and [1,

Near miss genuinely raises; recorded output truncates second shape. Execution reproduced complete error.

**Replacement output:**

```text
raises RuntimeError: stack expects each tensor to be equal size, but got [2, 1, 2] at entry 0 and [1, 3, 2] at entry 1
```

**Replacement explanation sentence:**

> For the first input, the operands have shapes `(2, 1, 2)` and `(1, 3, 2)`; stacking fails because those shapes differ.

## 2. Lesson pages — per page, per rubric code

Page: [Every ray against every segment](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/pytorch/kp-batched-segments.md). **Verdict: major.**

| Segment | Explanation verdict | Practice findings |
|---|---|---|
| Pair axes are independent | **major:** L1, L2 | D1–D4 mirrored here; D2 also affects 1105, 1106 |
| Reduce only after judging each pair | **pass** | D2 affects 1107, 1108 |

### L1 — `EXPL_CORRECT` · major · Pair axes + Misconceptions

**Exact concept quote:**

> Broadcasting a `(nr, 1, 2)` direction against a `(1, ns, 2)` edge needs one of them expanded to the full pair shape before stacking, because `stack` requires identical shapes; adding `t.zeros_like` of the other does that without a copy of meaning.

**Exact repeated misconception correction:**

> It requires identical shapes; expand the `(nr, 1, 2)` operand to `(nr, ns, 2)` first.

Expanding only one operand generally fails. With `nr=2`, `ns=3`, expanding direction leaves shapes `(2,3,2)` and `(1,3,2)`; execution confirms stacking still raises.

**Replacement concept sentence:**

> Before stacking, both operands must have shape `(nr, ns, 2)`: extend the direction across segments and the edge across rays, because `stack` requires identical input shapes.

**Replacement misconception sentence:**

> Both operands must already have shape `(nr, ns, 2)`; expanding only the direction leaves the edge with an unmatched ray axis.

### L2 — `EXPL_WHY` · major · Pair axes are independent

**Exact quote:**

> The reason the per-pair matrices are built by stacking on the *last* axis is that `torch.linalg` functions treat leading axes as a batch: an `(nr, ns, 2, 2)` tensor is `nr × ns` small matrices, and `det` or `solve` runs independently on each.

Explains batch placement, but fails to justify **last-axis choice**. Stacking on either final or penultimate axis produces `(nr, ns, 2, 2)`; only one places supplied vectors in columns.

Executed counterexample:

- Last axis: `[[2, 0], [1, -2]]`.
- Penultimate axis: `[[2, 1], [0, -2]]`.
- Identical shapes; different coefficient matrices.

**Replacement:**

> Stack on the last axis so the existing coordinate axis indexes matrix rows and the new axis indexes columns, giving `[[D_x, A_x − B_x], [D_y, A_y − B_y]]` for each pair; the leading `(nr, ns)` axes identify independent systems.

### Shared page findings

- **`STMT_INPUT` · major:** D1 repeated in page prompts 1729, 1730.
- **`GIVE_RUNG` · major:** D2 across faded starters 1105, 1106, 1107, 1108, 1729, 1730.
- **`GIVE_PROMPT`, `GIVE_STARTER` · major:** D3/D4 repeated in page’s 1730 prompt/docstring.

Quotes and replacements remain grouped above.

### Passing page checks

- **Both segments:** `EXPL_GENERAL_FIRST`, `EXPL_INTRO`, `EXPL_INTERLEAVE`, `EXPL_PRINTS`, `EXPL_LENGTH`.
- Concept prose: **210 words**, **164 words**.
- All **six explanatory Python fences** execute successfully in page order; assertions pass.
- **Selected drills:** no `EXPL_PREREQ` gap found. Page/supporting lessons show required indexing, broadcasting, constructors, stacking.
- **1729, 1730:** `EXPL_TRANSFER` passes. Segment-edge replication and constructing matrix columns require adapting scalar/displacement pair-table examples.
- **Reduction explanation:** axes, counts, masks, stated example values agree.

## 3. Systemic patterns

- **Ownership and aid disagree — `GIVE_RUNG`.** Every faded starter shown on this page ends with `pass`. Grouped defect spans six IDs.
- **Method leaks concentrate in 1730 — `GIVE_PROMPT`, `GIVE_STARTER`.** Prompt supplies assembly sequence; docstring supplies operation and axis.
- **Shared input boilerplate omits types — `STMT_INPUT`.** Both revised prompts specify result type but leave argument types implicit.
- **Correct solutions coexist with faulty teaching — `EXPL_CORRECT`, `EXPL_WHY`.** Reference code expands both operands correctly; prose teaches expanding one and inadequately explains column orientation.
- **Near-miss evidence loses information — `STMT_NEAR_MISS`.** 1730’s real failure is stored as an incomplete exception.
- **Grading evidence is strong — `STMT_CASES`, `CORR_DECISIVE`, `CORR_TYPES`.** Shape checks prevent 1729’s missing-ray-axis near miss from passing through broadcasting.

## 4. Top 10 fixes with rewrites

Ten ranked edit targets below derive from seven grouped findings.

| Rank | Target | Code · severity | Replacement |
|---|---|---|---|
| 1 | 1730 starter docstring | `GIVE_STARTER` · major | “Return the required 2×2 matrix for every ray–segment pair.” |
| 2 | 1730 assembly instruction | `GIVE_PROMPT` · major | “Within each matrix, rows correspond to x and y coordinates; columns contain the ray direction and segment `start − end`, respectively.” |
| 3 | Pair-axes expansion rule | `EXPL_CORRECT` · major | “Before stacking, both operands must have shape `(nr, ns, 2)`: extend the direction across segments and the edge across rays.” |
| 4 | Misconceptions expansion correction | `EXPL_CORRECT` · major | “Expanding only the direction leaves the edge with an unmatched ray axis; both operands must have shape `(nr, ns, 2)`.” |
| 5 | 1729 faded starter | `GIVE_RUNG` · major | “Complete the blanks to return each segment’s xy edge once per ray.” Supply D2 scaffold. |
| 6 | 1730 faded starter | `GIVE_RUNG` · major | “Complete the blanks to return the required matrix for every ray–segment pair.” Supply D2 scaffold. |
| 7 | 1729 input contract | `STMT_INPUT` · major | “Inputs `r` and `s` are floating-point PyTorch tensors with shapes `(nr, 2, 3)` and `(ns, 2, 3)`; coordinates follow `(x, y, z)` order.” |
| 8 | 1730 input contract | `STMT_INPUT` · major | “Inputs `r` and `s` are floating-point PyTorch tensors containing `[origin, direction]` and `[start, end]`, with shapes `(nr, 2, 3)` and `(ns, 2, 3)`.” |
| 9 | Pair-axes column rationale | `EXPL_WHY` · major | “The new last axis indexes columns, while the existing coordinate axis indexes rows; this places each direction vector in a matrix column.” |
| 10 | 1730 near-miss record | `STMT_NEAR_MISS` · minor | “For the first input, stacking operands shaped `(2, 1, 2)` and `(1, 3, 2)` raises `RuntimeError` because their shapes differ.” Store complete error from D5. |

### LeetCode-quality prompt: 1729

Return a floating-point PyTorch tensor `edges` of shape `(nr, ns, 2)` where `edges[i, j]` contains segment `j`’s xy displacement from its start to its end. Include one row for every ray, even though segment edges do not depend on rays.

- `r`: floating-point PyTorch tensor `(nr, 2, 3)`, containing `[origin, direction]`.
- `s`: floating-point PyTorch tensor `(ns, 2, 3)`, containing `[start, end]`.
- Coordinates follow `(x, y, z)` order; ignore z.
- Proposed constraints: `nr, ns ≥ 1`; finite values; matching input dtype/device.

**Example — tensor values shown as nested lists:**

```text
r = [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
     [[2.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]
s = [[[1.0, 1.0, 0.0], [4.0, 3.0, 0.0]]]

Output = [[[3.0, 2.0]],
          [[3.0, 2.0]]]
```

The segment’s xy displacement is `(3, 2)`, appearing once for each ray.

### LeetCode-quality prompt: 1730

Return a floating-point PyTorch tensor `matrices` of shape `(nr, ns, 2, 2)`, containing one matrix for every ray–segment pair. Matrix `matrices[i, j]` has ray `i`’s xy direction as its first column and segment `j`’s xy `start − end` vector as its second column.

- `r`: floating-point PyTorch tensor `(nr, 2, 3)`, containing `[origin, direction]`.
- `s`: floating-point PyTorch tensor `(ns, 2, 3)`, containing `[start, end]`.
- Coordinates follow `(x, y, z)` order; ignore z.
- Proposed constraints: `nr, ns ≥ 1`; finite values; matching input dtype/device.

**Example — tensor values shown as nested lists:**

```text
r = [[[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]]]
s = [[[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]]

Output = [[[[2.0,  0.0],
            [1.0, -2.0]]]]
```

The columns are `(2, 1)` and `(0, -2)`.

Review made no edits. `CLAUDE.md` became dirty after baseline.

Git: MODERATE — repo=Delta-Drills-Local branch=main; staged=0, unstaged=13, untracked=3, conflicts=0. Mixed scopes; unrelated changes preserved.