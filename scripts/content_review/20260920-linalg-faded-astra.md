# Delta Drills content review

Standard: [CONTENT_RUBRIC.md](/home/stellar-thread/Applications/Delta-Drills-Local/scripts/CONTENT_RUBRIC.md). Scope: drills **1722–1727**, both concept pages in [kp-linalg-basics.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/tensors/kp-linalg-basics.md).

**Verdict: all six drills major; both lesson pages major.** Canonical answers passed **24/24 cases** using local backend PyTorch 2.12.0 and backend comparison logic. **7/7 lesson fences passed**, including hidden checks. Findings below concern rubric defects despite executable answers. No edits or filesystem writes performed.

## 1. Drills — per-id findings

Shared findings appear once; table maps them to each drill.

| ID | Verdict | Applicable findings |
|---|---|---|
| 1722 | **major** | D1: statement contract; L3: example-input reuse |
| 1723 | **major** | D1: statement contract; D2: computation bypass; D5: near-miss explanation; L3: example reuse; L5: transfer |
| 1724 | **major** | D1: statement contract; D3: starter giveaway; L3: example-input reuse |
| 1725 | **major** | D1: statement contract; L3: example-input reuse |
| 1726 | **major** | D1: statement contract; D4: missing edge case; L3: example-input reuse; L5: transfer |
| 1727 | **major** | D1: statement contract; D2: computation bypass |

**Compact passes**

- **1722–1727:** `CORR_RUNS`, `CORR_TYPES`, `STMT_SELF_CONTAINED`, `STMT_LENGTH`, `GIVE_PROMPT`, `GIVE_TESTS`, `DIFF_LABEL`.
- **1722, 1724, 1725, 1726:** `CORR_DECISIVE`—tested constant returns, input passthroughs, recorded near misses each fail at least one case.
- **1722, 1723, 1724, 1725, 1727:** `STMT_CASES`—varying outputs plus size-1 axes.
- **1722, 1724–1727:** `STMT_NEAR_MISS`—reconstructed first-case mistakes reproduce recorded behavior and differ from correct results.
- **1722, 1723, 1725–1727:** authored faded scaffolds conceal target operator/method.

All six use `submission_mode: "function"`; `expected_artifact_type: "stdout"` does not override function grading. JSON `pass` starters alone do not establish a faded-rung violation: runtime supports replacement starters.

### D1 — Delayed task; implicit container types

**Codes/severity:** `STMT_TASK` **major**, `STMT_INPUT` **major** — **1722–1727**. Additionally, `STMT_OUTPUT` **major** — **1722, 1724, 1725**.

Every first sentence describes inputs rather than stating function’s goal. “Float matrix/vector” specifies numeric content and rank, but leaves Python container type unstated; starters have no argument annotations. Three prompts also omit return container type.

Exact opening quotes and replacement sentences:

| ID | Exact quote | Replacement |
|---|---|---|
| 1722 | “`a` is a float matrix of shape `(m, k)` and `v` a float vector of shape `(k,)`.” | “Return the matrix–vector product of floating-point PyTorch tensors `a`, shape `(m, k)`, and `v`, shape `(k,)`, as a PyTorch tensor of shape `(m,)`.” |
| 1723 | “`a` has shape `(m, k)` and `b` has shape `(k, n)`, both float.” | “Return the shape of the matrix product of floating-point PyTorch tensors `a`, shape `(m, k)`, and `b`, shape `(k, n)`, as a plain Python tuple of two integers.” |
| 1724 | “`a` has shape `(m, k)`, `b` has shape `(k, n)` and `c` has shape `(n, p)`, all float.” | “Return the matrix product of floating-point PyTorch tensors `a`, `b`, and `c`, in that order, as a PyTorch tensor of shape `(m, p)`; their shapes are `(m, k)`, `(k, n)`, and `(n, p)`.” |
| 1725 | “`a` is an invertible float matrix of shape `(n, n)` and `b` a float matrix of shape `(n, r)`: `r` right-hand sides side by side.” | “Return a floating-point PyTorch tensor `x` of shape `(n, r)` satisfying `a @ x = b`, where `a` and `b` are floating-point PyTorch tensors of shapes `(n, n)` and `(n, r)`, and `a` is invertible.” |
| 1726 | “An invertible float matrix `a` of shape `(n, n)` was applied to an unknown vector `x`, producing `y = a @ x` of shape `(n,)`.” | “Recover the vector `x` satisfying `a @ x = y` and return its entries as a Python list of `n` floats, given floating-point PyTorch tensors `a`, shape `(n, n)`, and `y`, shape `(n,)`, with `a` invertible.” |
| 1727 | “`mats` is a batch of invertible float matrices, shape `(batch, n, n)`, and `b` has shape `(batch, n)`: one right-hand side per matrix.” | “Return the solution batch’s shape as a plain Python tuple of two integers, given floating-point PyTorch tensors `mats`, shape `(batch, n, n)`, and `b`, shape `(batch, n)`, with every matrix invertible.” |

Exact return phrases triggering `STMT_OUTPUT`:

- **1722:** “Return the matrix–vector product, shape `(m,)`”
- **1724:** “Return the chained matrix product of the three, shape `(m, p)`”
- **1725:** “Return the `(n, r)` matrix `x`”

Replacements above explicitly require PyTorch tensors. Shape-only contracts still need D2’s substantive fix.

### D2 — Required computation can be skipped

**Code/severity:** `CORR_DECISIVE` **major** — **1723, 1727**.

| ID | Exact quote | Executed bypass | Result |
|---|---|---|---|
| 1723 | “read it off the product, do not hand-build it.” | `return (a.shape[0], b.shape[1])` | **4/4 pass**, without computing product |
| 1727 | “Solve every system in ONE call and return the SHAPE of the result” | `return tuple(b.shape)` | **4/4 pass**, without solving any system |

q1727 directly fails rubric’s no-op requirement. q1723’s forbidden hand-built answer remains indistinguishable from canonical output. Additional shape-only cases cannot establish whether either computation occurred.

**Replacement sentences:**

- **1723:** “Return the matrix product of `a` and `b` together with its shape, as `(product_tensor, shape_tuple)`.”
- **1727:** “Return a floating-point PyTorch tensor `x` of shape `(batch, n)` such that `mats[i] @ x[i] = b[i]` for every batch index `i`.”

**Required companion change:** grade computed values; update canonical answers and expected results alongside prompts.

### D3 — Faded starter exposes missing operator

**Code/severity:** `GIVE_STARTER` **major** — **1724**.

**Exact quote:**

```python
return a @ b _____ c
```

Page declares `syntax.matmul` as new syntax. First operator supplies answer to second blank.

**Replacement sentence:** “Fill in the blanks so the function returns the chained matrix product.”

**Replacement scaffold:**

```python
return a _____ b _____ c
```

### D4 — No edge case

**Code/severity:** `STMT_CASES` **major** — **1726**.

**Exact quotes:** all four right-hand-side fixtures:

```python
y=t.tensor([2.,4.])
y=t.tensor([3.,1.])
y=t.tensor([5.,1.])
y=t.tensor([7.,3.])
```

Every coefficient matrix is `(2, 2)`; every vector has length two. No size-1 case.

**Replacement/addition sentence:** “For `a = torch.tensor([[4.0]])` and `y = torch.tensor([10.0])`, return `[2.5]`.”

Add corresponding graded case; preserve existing non-diagonal cases.

### D5 — Near-miss explanation misstates broadcasting

**Code/severity:** `STMT_NEAR_MISS` **minor** — **1723**.

**Exact quote:** “`*` is elementwise, so it needs matching shapes”

Unequal shapes can support elementwise multiplication through broadcasting. This drill’s fourth case, `(4, 1)` and `(1, 5)`, demonstrates that: elementwise near miss executes and produces expected shape.

**Replacement sentence:** “Elementwise multiplication requires broadcast-compatible shapes; `(2, 3)` and `(3, 4)` are incompatible, while matrix multiplication matches the first matrix’s columns with the second matrix’s rows.”

## 2. Lesson pages — per page, per rubric code

Pages identified by segment heading, as requested by rubric.

| Page | Verdict | Findings | Compact passes |
|---|---|---|---|
| **Concept: two multiplications — \* vs @** | **major** | `EXPL_INTRO` L1; `EXPL_INTERLEAVE` L2; `GIVE_EXAMPLE` L3; `EXPL_TERMS` L4; `EXPL_TRANSFER` L5; `GIVE_STARTER` D3 | `EXPL_GENERAL_FIRST`, `EXPL_CORRECT`, `EXPL_PRINTS`, `EXPL_LENGTH` |
| **Concept: t.linalg.solve — never build the inverse** | **major** | `EXPL_INTRO` L1; `EXPL_INTERLEAVE` L2; `GIVE_EXAMPLE` L3; `EXPL_TRANSFER` L5; `EXPL_CORRECT` L6 | `EXPL_GENERAL_FIRST`, `EXPL_PRINTS`, `EXPL_LENGTH`; all executable fences pass |

D1–D5 also apply wherever corresponding drill text/scaffolds appear in lesson. Repeated findings not duplicated.

### L1 — Worked examples lack prose introductions

**Code/severity:** `EXPL_INTRO` **major** — **both pages**.

**Exact quote:** both introductions consist of heading immediately followed by code:

````markdown
## Worked example

```python
import torch as t
````

Code comments do not satisfy required opening prose. Mechanical INTRO check flags both.

**Replacement introductions:**

- **Multiplication:** “This example compares entrywise multiplication with matrix multiplication on the same operands, then checks how the input shapes determine the product’s shape.”
- **Solver:** “This example recovers an unknown vector from two linear equations, then substitutes the result back into those equations to check it.”

### L2 — Worked examples lack interleaving

**Code/severity:** `EXPL_INTERLEAVE` **major** — **both pages**.

**Exact quotes identifying unsplit steps:**

- **Multiplication:** “`# Elementwise vs matrix product — same operands, different operations:`” and “`# Shape rule: (2,3) @ (3,2) -> (2,2); the inner 3s must match and vanish.`”
- **Solver:** “`# Solve a @ x = b_vec — NOT by computing an inverse.`” and “`# Verification pattern: substitute back, compare with float tolerance.`”

Each example puts distinct teaching steps inside one fence. Mechanical INTERLEAVE check flags both single-block examples. Multiplication fence has 22 source lines; solver fence has 14.

**Replacement prose at new block boundaries:**

- **Multiplication:** “The two products differ because matrix multiplication adds contributions across a shared dimension; next, check which dimensions remain in the result.”
- **Solver:** “Substitute the recovered vector into the original equations; compare the reconstructed right-hand side with the given one using a tolerance because floating-point calculations can round.”

Split before shape demonstration and before verification respectively.

### L3 — Examples reuse graded input literals

**Code/severity:** `GIVE_EXAMPLE` **major** — **1722–1726**, across both pages.

| Page / affected IDs | Exact lesson quote | Graded overlap |
|---|---|---|
| Multiplication: **1722, 1724** | `a = t.tensor([[1.0, 2.0],` followed by `[3.0, 4.0]])` | Same `a` values as both drills’ first cases |
| Multiplication: **1723** | `print((t.ones((2, 3)) @ t.ones((3, 4))).shape)` | Exact operands and `(2, 4)` result from first graded case |
| Solver: **1725, 1726** | `a_sys = t.tensor([[2.0, 0.0],` followed by `[0.0, 4.0]])` | Same coefficient matrix as both drills’ first cases |
| Solver: **1725** | `a_sys = t.tensor([[3.0, 1.0],` followed by `[1.0, 2.0]])` | Same coefficient matrix as third graded case |

For 1722/1724/1725/1726, overlap is partial fixture reuse; not claiming complete input/output duplication. Rubric explicitly prohibits building examples from drill input literals.

**Replacement sentences and matching example data:**

- **Multiplication:** “Compare the two products using `a = [[2.0, 1.0], [0.0, 3.0]]` and `b = [[4.0, 0.0], [1.0, 2.0]]`; the matrix product is `[[9.0, 2.0], [3.0, 6.0]]`.”
- **Shape example:** “A matrix with shape `(2, 4)` multiplied by one with shape `(4, 3)` produces shape `(2, 3)`.”
- **Solver concept:** “For coefficient matrix `[[5.0, 2.0], [1.0, 3.0]]` and right-hand side `[1.0, -5.0]`, the solution is `[1.0, -2.0]`.”
- **Solver worked example:** “Recover the vector from coefficient matrix `[[4.0, 1.0], [1.0, 3.0]]` and right-hand side `[15.0, 1.0]`; the solution is `[4.0, -1.0]`.”

Update associated code, prints, assertions together.

### L4 — Terms precede owning lessons

**Code/severity:** `EXPL_TERMS` **minor** — **multiplication page**.

**Exact quotes:**

- “shapes must match (or broadcast).”
- “entry `[i, j]` is the dot product of row i of `a` with column j of `b`.”

Glossary assigns broadcasting to `torch.broadcasting-rules`, dot product to `torch.dot-matmul-patterns`; neither precedes this KP through its prerequisite chain. Dot-product KP explicitly depends on this one.

**Replacement sentences:**

- “For this comparison, use matrices with equal shapes; elementwise multiplication pairs entries at the same row and column.”
- “To obtain entry `[i, j]`, multiply corresponding entries of row `i` of `a` and column `j` of `b`, then add those products.”

### L5 — Faded drills repeat demonstrated decisions

**Code/severity:** `EXPL_TRANSFER` **major** — **1723, 1726**.

| Page / ID | Exact quote | Transfer defect | Replacement task sentence |
|---|---|---|---|
| Multiplication / **1723** | `return tuple((a _____ b).shape)` | Earlier example already multiplies matrices and reads resulting shape; blank repeats same decision. | “Given a vector `v` of shape `(k,)` and a matrix `b` of shape `(k, n)`, return their row-vector–matrix product and its shape.” |
| Solver / **1726** | `return t.linalg._____(a, y).tolist()` | Same matrix/vector solve as example; added list conversion already supplied by scaffold. | “Recover the row vector `x` satisfying `x @ a = y`, and return its entries as a Python list of floats.” |

These are transfer redesigns: change operand orientation, then synchronize starters, answers, cases. Merely changing numbers would leave defect intact.

### L6 — Residual check falsely presented as condition diagnosis

**Code/severity:** `EXPL_CORRECT` **major** — **solver page**.

**Exact quote:** “Why: `solve` + `allclose` verification — the pair costs one line and catches both wrong answers and ill-conditioned systems.”

A small residual does not diagnose conditioning or guarantee accurate recovered coordinates.

**Executed counterexample:** coefficient matrix `diag(1, 1e-12)`, right-hand side `[1, 1e-12]`. Condition number approximately `1e12`; true solution `[1, 1]`. Default `allclose` accepts residual from incorrect candidate `[1, 2]`.

**Replacement sentence:** “Substitution checks whether the candidate approximately satisfies the equations; it does not diagnose ill-conditioning, where small changes in the inputs can cause large changes in the solution.”

## 3. Systemic patterns

- **Statement template defers goal and omits container contracts.** `STMT_TASK`, `STMT_INPUT`, `STMT_OUTPUT` → D1, all six drills.
- **Shape-only outputs cannot verify requested numerical work.** `CORR_DECISIVE` → D2, 1723/1727. More shape fixtures cannot close this gap.
- **Worked examples share presentation defects.** `EXPL_INTRO`, `EXPL_INTERLEAVE` → L1/L2, both pages.
- **Example/drill separation is insufficient.** `GIVE_EXAMPLE`, `EXPL_TRANSFER` → L3/L5. Fresh literals prevent reuse; changed decisions establish transfer. Both checks matter.
- **Passing canonical answers does not establish content quality.** `CORR_RUNS` passes throughout; `CORR_DECISIVE`, `STMT_CASES`, `GIVE_STARTER`, `EXPL_CORRECT` still identify concrete failures.

## 4. Top 10 fixes with rewrites

Ranks prioritize invalid learning evidence and factual errors, then statement quality. **#1, #4, #7 change contracts and require matching starter/answer/test updates.**

### 1. q1727 — Grade solutions, not input shape

**`CORR_DECISIVE` — major; D2.**

> Return a floating-point PyTorch tensor `x` of shape `(batch, n)` satisfying `mats[i] @ x[i] = b[i]` for every batch index `i`. Inputs `mats` and `b` are floating-point PyTorch tensors of shapes `(batch, n, n)` and `(batch, n)`; every matrix is invertible.

Example: matrices `[[[2, 0], [0, 4]], [[1, 0], [0, 1]]]`, right-hand sides `[[2, 4], [5, 6]]` → tensor `[[1, 1], [5, 6]]`.

### 2. q1724 — Conceal both operators

**`GIVE_STARTER`, `STMT_TASK`, `STMT_INPUT`, `STMT_OUTPUT` — major; D1/D3.**

> Return the matrix product of floating-point PyTorch tensors `a`, `b`, and `c`, in that order. Their shapes are `(m, k)`, `(k, n)`, and `(n, p)`; return a PyTorch tensor of shape `(m, p)`.

Example: `a = [[1, 2]]`, `b = [[1, 0], [0, 1]]`, `c = [[3], [4]]` → tensor `[[11]]`.

Scaffold: `return a _____ b _____ c`.

### 3. Solver page — Correct verification claim

**`EXPL_CORRECT` — major; L6.**

> Substitution checks whether the candidate approximately satisfies the equations. It does not diagnose ill-conditioning, where small changes in the inputs can cause large changes in the solution.

Keep residual check; remove claim that it detects conditioning.

### 4. q1723 — Require values plus transferable shape reasoning

**`CORR_DECISIVE`, `EXPL_TRANSFER` — major; D2/L5.**

> Return the product of a row vector `v` and a matrix `b`, together with the product’s shape. Inputs are floating-point PyTorch tensors of shapes `(k,)` and `(k, n)`; return `(result, shape)`, where `result` is a PyTorch tensor of shape `(n,)` and `shape` is a plain Python tuple.

Example: `v = [2, 1]`, `b = [[1, 3], [4, 2]]` → `(tensor([6, 8]), (2,))`.

### 5. Both worked examples — Introduce, then separate steps

**`EXPL_INTRO`, `EXPL_INTERLEAVE` — major; L1/L2.**

Multiplication opening:

> This example compares entrywise multiplication with matrix multiplication on the same operands.

Before separate shape block:

> Matrix multiplication removes the shared inner dimension; use the two outer dimensions to predict the result’s shape.

Solver opening:

> Recover the unknown vector, then check it against the original equations.

Before separate verification block:

> Substitute the recovered vector into the equations and compare the reconstructed right-hand side using a floating-point tolerance.

### 6. Both lesson pages — Replace reused fixtures

**`GIVE_EXAMPLE` — major; L3.**

Multiplication example:

> With `a = [[2, 1], [0, 3]]` and `b = [[4, 0], [1, 2]]`, entrywise multiplication produces `[[8, 0], [0, 6]]`, while matrix multiplication produces `[[9, 2], [3, 6]]`.

Solver example:

> The matrix `[[4, 1], [1, 3]]` maps `[4, -1]` to `[15, 1]`; recover the unknown vector from that matrix and right-hand side.

Use floating-point tensors; update all associated checks.

### 7. q1726 — Add transfer and singleton coverage

**`EXPL_TRANSFER`, `STMT_CASES`, `STMT_TASK`, `STMT_INPUT` — major; D1/D4/L5.**

> Recover the row vector `x` satisfying `x @ a = y`, and return its entries as a Python list of `n` floats. Inputs `a` and `y` are floating-point PyTorch tensors of shapes `(n, n)` and `(n,)`, and `a` is invertible.

Example: `a = [[1, 2], [0, 1]]`, `y = [2, 7]` → `[2.0, 3.0]`.

Required edge case: `a = [[4]]`, `y = [10]` → `[2.5]`.

### 8. q1722 — State concrete input/output contract

**`STMT_TASK`, `STMT_INPUT`, `STMT_OUTPUT` — major; D1.**

> Return the matrix–vector product of floating-point PyTorch tensors `a`, shape `(m, k)`, and `v`, shape `(k,)`, as a PyTorch tensor of shape `(m,)`. Output entry `i` is the sum of corresponding entry products from row `i` of `a` and `v`.

Example: `a = [[2, 0, 1]]`, `v = [1, 5, -1]` → tensor `[1]`.

### 9. q1725 — Define columns as separate systems

**`STMT_TASK`, `STMT_INPUT`, `STMT_OUTPUT` — major; D1.**

> Return a floating-point PyTorch tensor `x` of shape `(n, r)` satisfying `a @ x = b`. Inputs `a` and `b` are floating-point PyTorch tensors of shapes `(n, n)` and `(n, r)`, with `a` invertible; each column of `b` defines a separate system, and the corresponding column of `x` contains its solution.

Example: `a = [[1, 1], [0, 1]]`, `b = [[3, 5], [1, 2]]` → tensor `[[2, 3], [1, 2]]`.

### 10. Multiplication explanations — Define operations accurately

**`EXPL_TERMS`, `STMT_NEAR_MISS` — minor; L4/D5.**

Lesson replacement:

> To calculate one matrix-product entry, multiply matching entries from the relevant row and column, then add the products.

q1723 near-miss replacement:

> Elementwise multiplication requires broadcast-compatible shapes; `(2, 3)` and `(3, 4)` are incompatible, while matrix multiplication matches the first matrix’s columns with the second matrix’s rows.

---

Git: MODERATE — repo=Delta-Drills-Local branch=main; staged=0, unstaged=19, untracked=6, conflicts=0. Baseline unstaged=16; three additional tracked paths became dirty during review. No changes made by this review.