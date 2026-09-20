Reviewed against [CONTENT_RUBRIC.md](/home/stellar-thread/Applications/Delta-Drills-Local/scripts/CONTENT_RUBRIC.md) only. No edits.

**Verdicts: 27 major, 3 minor, 2 pass drills; eight major lesson pages.** All 128 stored cases pass reference answers. Additional permitted-input probes expose failures; seven incorrect variants also pass every stored case. All 48 lesson example blocks execute successfully.

Verification used backend venv, read-only execution, exact shape/dtype checks, floating comparison tolerance `rtol=1e-5, atol=1e-6`. Global prerequisite order and out-of-range drill test cases were not inspected.

Git: MODERATE — repo=Delta-Drills-Local branch=main, ahead 1; staged=0, unstaged=18, untracked=1, conflicts=0. Existing content changes preserved; concurrent unrelated work committed during review.

## 1. Drills — per-id findings

**Pass:** 1692, 1713.

Repeated defects grouped below. D-numbers identify findings; rubric codes determine severity.

| ID | Verdict | Findings |
|---|---|---|
| 1690 | minor | `DIFF_RUNG` D6 |
| 1691 | major | `CORR_RUNS` D1; `STMT_CASES` D3; `DIFF_RUNG` D6 |
| 1693 | major | `CORR_RUNS` D1 |
| 1694 | major | `STMT_OUTPUT` D4; `GIVE_EXAMPLE` D5 |
| 1695 | major | `STMT_CASES` D3; `STMT_OUTPUT` D4; `GIVE_EXAMPLE` D5 |
| 1696 | major | `CORR_RUNS` D1; `GIVE_EXAMPLE` D5 |
| 1697 | major | `STMT_OUTPUT` D4; `GIVE_EXAMPLE` D5 |
| 1698 | major | `CORR_RUNS` D11; `STMT_LENGTH` D7 |
| 1699 | major | `CORR_RUNS` D11 |
| 1700 | major | `CORR_RUNS` D11 |
| 1701 | major | `CORR_RUNS` D11; `STMT_LENGTH` D7 |
| 1702 | major | `CORR_RUNS` D2; `STMT_CASES` D3; `GIVE_EXAMPLE` D5 |
| 1703 | major | `CORR_RUNS` D2 |
| 1704 | major | `CORR_RUNS` D2; `GIVE_EXAMPLE` D5 |
| 1705 | major | `CORR_RUNS` D1, D2; `GIVE_EXAMPLE` D5 |
| 1706 | major | `CORR_RUNS` D2 |
| 1707 | major | `CORR_RUNS` D2; `STMT_CASES` D3 |
| 1708 | major | `CORR_RUNS` D2; `STMT_TERMS` D9 |
| 1709 | major | `CORR_RUNS` D2 |
| 1710 | major | `CORR_RUNS` D11; `STMT_LENGTH` D7; `STMT_TERMS` D8 |
| 1711 | minor | `DIFF_RUNG` D6; `STMT_TERMS` D8 |
| 1712 | minor | `STMT_TERMS` D8 |
| 1714 | major | `CORR_RUNS` D2; `STMT_CASES` D3; `GIVE_EXAMPLE` D5; `DIFF_RUNG` D6; `STMT_TERMS` D10 |
| 1715 | major | `CORR_RUNS` D2; `STMT_CASES` D3; `GIVE_EXAMPLE` D5 |
| 1716 | major | `CORR_RUNS` D2; `GIVE_EXAMPLE` D5; `STMT_TERMS` D9 |
| 1717 | major | `CORR_RUNS` D2; `GIVE_EXAMPLE` D5; `STMT_TERMS` D9 |
| 1718 | major | `CORR_RUNS` D2; `STMT_OUTPUT` D4 |
| 1719 | major | `CORR_RUNS` D2; `STMT_CASES` D3; `STMT_TERMS` D9 |
| 1720 | major | `CORR_RUNS` D2; `STMT_TERMS` D9 |
| 1721 | major | `CORR_RUNS` D2; `STMT_LENGTH` D7 |

### D1 — Reference answers fail permitted inputs

**`CORR_RUNS` — major.** Stored cases pass; these additional inputs satisfy current statements. Replacement sentences clarify required behavior; corresponding answer repairs remain necessary.

| ID | Exact quote | Executed failure | Replacement sentence; required repair |
|---|---|---|---|
| 1691 | `return -(l.exp()*l).sum(dim=1)` | Float32 `x=[[3e38,-3e38]]` → `[nan]`; entropy should round to `[0]`. Subtraction overflows; subsequent `0 * -inf` produces NaN. | “Return each row’s softmax entropy in natural-log units, treating a zero-probability contribution as zero and keeping the result finite for finite logits.” Widen vulnerable arithmetic; handle zero contributions. |
| 1693 | `s=x/temp.reshape(-1,1)` | `x=[[1e38,1e38]]`, `temp=[0.1]` → `[[nan,nan]]`; expected `[[0.5,0.5]]`. | “Equal scores must receive equal probabilities at every allowed positive temperature, including when directly dividing the scores would overflow.” Stabilize before overflow occurs. |
| 1696 | `floor=x.min(dim=1,keepdim=True)[0]-1` | `x=[[1e8,1e8]]`, `y=[1]` → `0`; expected `1`. Float32 subtraction leaves `floor` unchanged, allowing class 0 to win twice. | “The two selected classes must have distinct indices; resolve equal scores by increasing class index.” Exclude the first winner reliably instead of assuming subtracting one lowers a float. |
| 1705 | `f=((o-a)@e)/(e@e)` | `r=[[0,0,0],[1,0,0]]`, `s=[[2,1,0],[2,1,0]]` → `[nan,nan]`; expected `[2,1]`. A collapsed segment is not excluded by the statement. | “If both segment endpoints coincide, return that endpoint as the nearest point on a miss.” Handle zero segment length before division. |

### D2 — Direction magnitude incorrectly changes intersection membership

**`CORR_RUNS` — major.**

**IDs:** 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721.

**Exact shared quote:**

```python
valid=t.linalg.det(m).abs()>=1e-8
```

Absolute determinant magnitude depends on direction length. Statements allow non-unit directions and exclude singular pairs; these probes are nonsingular.

- **Segments:** `O=(0,0,0)`, `D=(1e-10,0.5e-10,0)`, endpoints `(1,-1,0)` and `(1,1,0)`. Actual hit: `(1,0.5)`. All eight answers reject the hit. For example, 1702 returns `[0,0]`; 1703 returns `-1`; 1705 returns `[1,0]`.
- **Triangles:** `O=(0,0,0)`, `D=(1e-10,0,0)`, vertices `(1,-1,-1)`, `(1,1,-1)`, `(1,-1,1)`. Actual hit: `(1,0,0)`, weights `[0,0.5,0.5]`. All relevant answers report misses. For 1719, adding the corresponding triangle at `x=2` produces an actual parameter gap of `1e10`, but the answer returns infinity.

**Replacement sentence:**

> “A positive rescaling of a nonzero ray direction must preserve which objects it intersects; parallel or degenerate pairs count as misses.”

**Repair:** Make degeneracy handling independent of direction magnitude. State any remaining numerical tolerance in each standalone contract; grade scaled-direction cases.

### D3 — Cases accept demonstrably wrong implementations

**`STMT_CASES` — major.** Each variant below was executed and passed **all four** stored cases for its drill.

| IDs | Exact quote from existing data | Wrong implementation accepted | Replacement case sentence |
|---|---|---|---|
| 1691 | `x=t.tensor([[10.,0.]])` | Ordinary softmax followed by `p.log()` passes; no case exercises probability underflow. | “For logits `[[1200,0]]`, return entropy `[0]`, not NaN.” |
| 1695 | `x=t.tensor([[2.,0.,0.],[2.,0.,0.]]); y=t.tensor([0,0])` | Return the first loss in each group instead of its mean. The only multi-example group has identical losses. | “For scores `[[0,0],[3,0],[0,2]]` and labels `[0,0,0]`, return approximately `[0.370867,2.126928]`.” |
| 1702, 1707 | 1702: `t.tensor([0.0, 0.0], dtype=t.float32)`; 1707: `t.tensor([[0.0, 0.0]], dtype=t.float32)` | Return zero on every miss instead of the ray’s own origin. Every tested miss has zero origin. | “A ray starting at `(5,5,0)` along `(1,0,0)` misses the segment from `(0,0,0)` to `(0,2,0)`; return `[5,5]`, or `[[5,5]]` for the batched task.” |
| 1714 | `t.tensor([0.0, 0.5, 0.5], dtype=t.float32)` and `t.tensor([0.5, 0.25, 0.25], dtype=t.float32)` | Swap B’s and C’s weights. Those weights are equal in every case. | “For a ray from the origin along `(4,2,1)` and triangle `[(4,0,0),(4,4,0),(4,0,4)]`, return `[0.25,0.5,0.25]`.” |
| 1715 | `t.tensor(1.4142135381698608, dtype=t.float32)` | Always return distance to vertex A. A is nearest or tied for nearest in every hit case. | “For a ray from the origin along `(4,3,0)` and triangle `[(4,0,0),(4,4,0),(4,0,4)]`, return `1`, because B is nearest.” |
| 1719 | `t.tensor([2.0], dtype=t.float32)` and `t.tensor([2.0, float('inf')], dtype=t.float32)` | Replace every finite gap with literal `2`. Every tested finite gap equals two. | “For a unit `+x` ray hitting triangles at parameters `2` and `7`, return `[5]`.” |

Existing `wrong_examples` outputs all differ from their corresponding first expected outputs. The defect is additional plausible mistakes surviving the case sets.

### D4 — Tie policy omitted where it changes returned values

**`STMT_OUTPUT` — major.**

| IDs | Exact quote | Ambiguity | Replacement sentence |
|---|---|---|---|
| 1694, 1695, 1697 | 1694: “were predicted as k”; 1695: “the model classified correctly”; 1697: “whose predicted class is j” | Reference answers choose the earliest maximum, but these standalone prompts omit that policy. Drill 1695 actually grades tied scores in its second case. | “The predicted class is the index of the largest score in the row; ties choose the smallest class index.” |
| 1718 | “the NEAREST hit of at least one ray” | Coincident triangles are equally nearest. Reference answer marks only the earliest triangle visible; marking both also follows the current wording. Executed duplicate-triangle probe returns `[True,False]`. | “Each ray selects one nearest triangle; equal hit parameters choose the smallest triangle index.” |

### D5 — Lesson examples reuse drill inputs

**`GIVE_EXAMPLE` — major.**

**IDs:** 1694, 1695, 1696, 1697, 1702, 1704, 1705, 1714, 1715, 1716, 1717.

These are literal input reuse in the supplied lesson source, not an assertion about runtime example scheduling.

| Lesson / affected IDs | Exact lesson quote reused by drills | Replacement example sentence |
|---|---|---|
| Classifier evaluation: 1694, 1696, 1697 | `x=t.tensor([[9.,1.,0.],[1.,2.,7.],[0.,6.,5.]])` and `y=t.tensor([0,0,1])` | “For scores `[[4,1,0],[0,2,6],[0,5,2],[3,1,0]]` and labels `[0,1,1,2]`, correctness is `[True,False,True,False]`, giving accuracy `1/2`.” |
| Classifier evaluation: 1695 | `x=t.tensor([[3.,1.],[1.,3.]])` and `y=t.tensor([0,0])` | “For scores `[[0,1.5],[1.5,0]]` and labels `[1,1]`, the per-example losses are approximately `[0.201413,1.701413]`.” |
| Segment intersection: 1702, 1704, 1705 | `o=t.tensor([1.,1.]); d=t.tensor([1.,1.])` and `a=t.tensor([4.,0.]); b=t.tensor([4.,6.])` | “A ray from `(2,-1)` along `(1,2)` meets the segment from `(5,0)` to `(5,10)` at `(5,5)`, with ray parameter `3` and segment fraction `1/2`.” |
| Triangle intersection: 1714–1717 | `o=t.tensor([0.,0.,0.]); d=t.tensor([1.,0.,0.])` and `a=t.tensor([3.,-1.,-1.]); b=t.tensor([3.,1.,-1.]); c=t.tensor([3.,-1.,1.])` | “A ray from `(1,0,0)` along `(2,0,0)` meets the triangle `[(5,-2,-1),(5,2,-1),(5,-2,3)]` at parameter `2`, with triangle coordinates `(0.5,0.25)`.” |

Segment drill inputs merely append the zero z-coordinate to the same worked geometry. Drill 1695 additionally reproduces the lesson’s two loss values as its expected grouped means.

### D6 — Integrated tasks repeat lower-rung decisions

**`DIFF_RUNG` — minor.** Cross-rung duplication; **not** `DIFF_DUP`, whose rubric scope is same-rung duplication.

| ID | Exact quote | Lower-rung equivalent | Replacement sentence, placed on Solo |
|---|---|---|---|
| 1690 | “Return the probability each row assigns to its own predicted class” | q1038: “Return each row’s largest class probability” | “Return a float tensor of shape `(b,)` containing each row’s largest softmax probability.” |
| 1691 | “Return the entropy of each row's class distribution” | q1040 already asks for row entropy; requiring log space changes the method, not the requested result. | “Return each row’s softmax entropy as a float tensor of shape `(b,)`, including rows with probabilities that underflow to zero.” |
| 1711 | “Return camera rays from the origin through the image plane x=f” | q1123 builds the same rays at `x=1`; q1130 already changes the plane to `x=2`. Parameterizing the constant adds little decision-making. | “Return rays from the origin through equally spaced pixels on `x=f`, stored as `[origin,direction]` with shape `(ny*nz,2,3)`.” |
| 1714 | “Return the barycentric weights of the three vertices at the hit” | q1144: “Return vertex weights [A,B,C] for a hit” | “Return the hit’s vertex weights in `[A,B,C]` order as a float tensor of shape `(3,)`, or zeros on a miss.” |

Retaining Integrated requires a genuinely additional decision, beyond rewording these tasks.

### D7 — Four-sentence prompts exceed stated limit

**`STMT_LENGTH` — minor. IDs:** 1698, 1701, 1710, 1721.

| ID | Exact quote identifying a separate sentence | Replacement |
|---|---|---|
| 1698 | “Every query has a unique nearest candidate.” | “Return an integer tensor of shape `(n,)` counting the queries assigned to each candidate by maximum cosine similarity, assuming every query has a unique winner.” |
| 1701 | “Every query has a unique nearest candidate; among the rest, ties choose the earlier candidate.” | “Return each query’s second-nearest candidate index by cosine similarity, assuming a unique nearest candidate and resolving remaining ties by smallest index.” |
| 1710 | “Rays go from the origin through the plane x=1.” | “Return the smallest flat pixel index whose ray from the origin through `x=1` has maximum cosine similarity with the positive x-axis.” |
| 1721 | “Directions need not be unit length.” | “Return a float tensor of shape `(nr,)` containing `1/(1+distance)` for each ray’s nearest hit, using physical Euclidean distance even for non-unit directions, and zero for misses.” |

Move argument descriptions and constraints into short lists.

### D8 — Inclusive sampling does not specify spacing

**`STMT_TERMS` — minor. IDs:** 1710, 1711, 1712.

**Exact quote:**

> “Each axis is sampled inclusively from −limit to +limit (a single pixel sits at −limit)”

Endpoints are specified; uniform spacing is not. Reference answers require uniform spacing.

**Replacement sentence:**

> “Sample each axis at equally spaced coordinates from its negative half-width to its positive half-width, including both endpoints; a one-pixel axis uses the negative endpoint.”

### D9 — Travel parameters lack standalone definitions

**`STMT_TERMS` — minor.**

| IDs | Exact quote | Replacement sentence |
|---|---|---|
| 1708 | “smallest ray parameter u” | “The ray parameter `u` locates the point `origin + u·direction`; it measures Euclidean distance only when the direction has unit length.” |
| 1716, 1717 | “the travel parameter s” | “The travel parameter `s` locates the point `origin + s·direction`; `s≥0` is on the forward ray.” |
| 1719 | “second s minus nearest s” | “For each ray, define hit depth `s` by `hit_point = origin + s·direction`, and return the second-smallest valid depth minus the smallest.” |
| 1720 | “the smallest s among hits” | “Hit depth `s` is the nonnegative parameter in `hit_point = origin + s·direction`.” |

### D10 — Naming vertex weights does not define barycentric coordinates

**`STMT_TERMS` — minor. ID:** 1714.

**Exact quote:**

> “the barycentric weights of the three vertices at the hit, [weight of A, weight of B, weight of C]”

**Replacement sentence:**

> “Return weights `[wA,wB,wC]` satisfying `hit_point = wA·A + wB·B + wC·C` and `wA+wB+wC=1`, in vertex order.”

This defines the requested quantity without supplying the solution formula.

### D11 — Norm overflow corrupts similarity and camera results

**`CORR_RUNS` — major. IDs:** 1698, 1699, 1700, 1701, 1710.

**Exact shared quotes:** `a=x/x.norm(dim=1,keepdim=True)`, `b=y/y.norm(dim=1,keepdim=True)`; 1710: `cos=d[:,0]/d.norm(dim=1)`.

Current prompts supply no magnitude restrictions excluding these finite inputs.

| ID | Executed input | Actual → required |
|---|---|---|
| 1698 | `x=[[1e20,0]]`, `y=[[1,0],[0,1]]` | `[1,1]` → `[1,0]` |
| 1699 | `y=[[1e20,0],[1e20,0]]` | `-1` → `1` |
| 1700 | `x=[[1e20,0]]`, `y=[[0,1],[1e20,0]]` | `[False]` → `[True]` |
| 1701 | `x=[[1e20,0]]`, `y=[[1,0],[0,1],[-1,0]]` | `[0]` → `[1]` |
| 1710 | `ny=3,nz=2,yl=1e20,zl=1e20` | `0` → `2` |

**Replacement constraint sentence:**

> “Cosine similarities and distances must remain valid for finite nonzero vectors, including coordinates around `1e20`.”

**Repair:** Use range-safe norm and distance calculations. If a narrower numerical domain is intended, declare and test that domain instead of accepting unrestricted floats.

## 2. Lesson pages — per page, per rubric code

All eight pages pass the inspected `EXPL_GENERAL_FIRST`, `EXPL_INTRO`, and `EXPL_PRINTS` checks. Concept prose spans **152–243 words**, satisfying `EXPL_LENGTH`. All 48 ordinary example fences execute and fit the 16-line limit.

### Shared finding: Faded starters contain no scaffold

**`GIVE_RUNG` — major.**

**Exact quotes:** `## Faded practice` followed by starter bodies containing only `pass`.

| Page | Verdict | Affected Faded IDs |
|---|---|---|
| [Stable probabilities](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/pytorch/kp-stable-probabilities.md) | major | 1032, 1033, 1034, 1035 |
| [Classifier evaluation](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/pytorch/kp-classifier-evaluation.md) | major | 1045, 1046, 1047, 1048 |
| [Cosine similarity](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/pytorch/kp-cosine-similarity.md) | major | 1006, 1007, 1008, 1009 |
| [Segment intersection](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/pytorch/kp-segment-intersection.md) | major | 1089, 1090, 1091, 1092 |
| [Batched segments](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/pytorch/kp-batched-segments.md) | major | 1105, 1106, 1107, 1108 |
| [Make rays 2-D](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/pytorch/kp-make-rays-2d.md) | major | 1121, 1122, 1123, 1124 |
| [Triangle intersection](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/pytorch/kp-triangle-intersection.md) | major | 1137, 1138, 1139, 1140 |
| [Mesh visibility](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/pytorch/kp-mesh-visibility.md) | major | 1153, 1154, 1155, 1156 |

An empty function is from-scratch practice. It does not satisfy the rubric’s “scaffold with blanks.”

**Replacement learner sentence:**

> “Complete the blanks in the supplied scaffold to produce the requested result.”

**Structural repair:** Supply partial implementations retaining learned setup and blanking the tested operations. In stable probabilities, blank every occurrence of the declared new syntax, `exp` and `log`; for example, a faded exponential step can retain `return z.__()`.

### Stable probabilities

**`GIVE_PROMPT` — major — Solo practice, q103.**

**Exact quote:**

> “Softmax of a logit vector that survives entries like 1000 — shift before exponentiating.”

“Shift before exponentiating” supplies the central implementation decision.

**Replacement:**

> “Return a float probability tensor of shape `(n,)` proportional to the exponentials of finite input scores `x`, keeping the result finite even when scores are near `1000`.”

**`DIFF_RUNG` — minor — Integrated practice:** q1690 and q1691 repeat Solo tasks; quotes and replacements in D6.

### Classifier evaluation

**`GIVE_EXAMPLE` — major — both Worked example segments.** Reused score/label literals affect 1694–1697; exact quotes and replacement examples in D5.

**`STMT_OUTPUT` — major — Integrated practice.** q1694, q1695, q1697 omit the prediction tie policy required by their answers; quotes and replacement sentence in D4.

No additional confirmed explanation defect beyond the shared Faded mismatch.

### Cosine similarity

**`EXPL_CORRECT` — major — Misconceptions: wrong matrix compatibility condition.**

**Exact quote:**

> “With two matrices of shape `(m, d)` and `(n, d)` it fails unless `m = d`; the candidate matrix must be transposed.”

For `a @ b`, the matching dimensions are `d` and `n`, not `m` and `d`.

**Replacement:**

> “For matrices shaped `(m,d)` and `(n,d)`, `a @ b` is dimensionally valid only when `d=n`; comparing every query with every candidate requires the candidate rows to become columns.”

**`EXPL_CORRECT` — major — Misconceptions: floating-point overstatement.**

**Exact quote:**

> “For unit rows it cannot; a value above `1` means something was not normalized.”

Executed counterexample: normalize float32 `[3,3,3]`, then dot it with itself → `1.000000238418579`.

**Replacement:**

> “Exact cosine similarity lies in `[-1,1]`, but floating-point normalization and dot products can exceed the endpoints slightly through rounding.”

### Segment intersection

**`EXPL_CORRECT` — major — “A solution must belong to both objects” and Misconceptions.**

**Exact quotes:**

> “the validity test is `det.abs() >= 1e-8`”

> “Floating-point parallel lines give a tiny nonzero determinant; compare against a tolerance.”

An absolute cutoff is scale-dependent, as D2 demonstrates. Exactly parallel floating-point vectors can also produce exactly zero.

**Replacement:**

> “Rounding can perturb a determinant near zero, but any numerical parallelism test must account for vector scale because multiplying a ray direction by a positive factor does not change the ray.”

**`GIVE_EXAMPLE` — major — first Worked example.** Same geometry appears in 1702, 1704, 1705; quotes and replacement in D5.

### Batched segments

**`EXPL_INTERLEAVE` — major — Faded practice solutions q1107, q1108.**

Both solution fences contain **17 physical lines**. Ordinary explanation fences pass.

**Exact excerpts:**

```python
v=a-o
```

```python
uv=t.linalg.solve(safe,v[...,None])[...,0]
```

**Replacement explanatory sentence:**

> “Each right-hand side is the displacement from that ray’s origin to that segment’s start.”

**Concrete repair:** Remove the standalone `v=a-o` line and use:

```python
uv=t.linalg.solve(safe,(a-o)[...,None])[...,0]
```

That reduces each fence to 16 lines without changing the function.

The embedded intersection solutions also use the scale-dependent validity expression discussed under D2.

### Make rays 2-D

**`EXPL_CORRECT` — major — “Pixels form a product of two axes.”**

**Exact quote:**

> “a square image hides the swap, a rectangular one makes it a shape error.”

After flattening, both axis orders still contain `ny*nz` pixels. A rectangular grid does not necessarily cause a shape error.

**Replacement:**

> “A rectangular grid makes swapped axis order easier to detect from the coordinate sequence, even though the flattened result still has `ny*nz` entries.”

**`STMT_SELF_CONTAINED` — major — Solo practice q1124, q1127, q1128.**

These page-owned statements omit camera geometry needed when served alone.

| ID | Exact quote | Replacement sentence |
|---|---|---|
| 1124 | “Return each camera ray’s direction, shape (ny*nz,3).” | “Return directions from the origin through pixels `(1,y,z)` on the plane `x=1`, as a float tensor of shape `(ny*nz,3)`.” |
| 1127 | “Return points reached by these rays at parameter u=2, shape (ny*nz,3).” | “For rays starting at the origin and passing through pixels `(1,y,z)` on `x=1`, return their positions at ray parameter `u=2`, shape `(ny*nz,3)`.” |
| 1128 | “Return only rays through the first z-column of the image, shape (ny,2,3).” | “Return rays from the origin through the first z-column of pixels on `x=1`, stored as `[origin,direction]` with shape `(ny,2,3)`.” |

Keep the existing sampling and order constraints after these replacements.

**`DIFF_RUNG` — minor:** q1711, D6.  
**`STMT_TERMS` — minor:** inclusive sampling lacks “equally spaced,” D8.

### Triangle intersection

**`EXPL_CORRECT` — major — first Worked example, transition to parallelogram.**

**Exact quote:**

> “Dropping the sum test admits the second pair. That is the parallelogram, not the triangle”

Dropping `u+v≤1` while retaining only nonnegativity admits an unbounded region. The next code block additionally introduces separate upper bounds.

**Replacement:**

> “Replacing `u+v≤1` with the separate bounds `u≤1` and `v≤1`, while retaining nonnegativity, admits the parallelogram’s far corner.”

**`GIVE_EXAMPLE` — major — second Worked example.** Reused ray/triangle literals affect 1714–1717; D5.

**`DIFF_RUNG` — minor — Integrated practice.** q1714 repeats Solo q1144; D6.

### Mesh visibility

**`STMT_OUTPUT` — major — Integrated practice, q1718.**

**Exact quote:**

> “the NEAREST hit of at least one ray”

The page specifies tie-breaking for q1159 and q1168, but omits it from q1718. A standalone drill cannot inherit the convention silently.

**Replacement:**

> “A triangle is visible if at least one ray selects it as its nearest hit, with equal hit parameters resolved by the smallest triangle index.”

The page’s embedded intersection solutions also repeat D2’s absolute determinant cutoff. Shared Faded mismatch applies to both segments.

## 3. Systemic patterns

- **`CORR_RUNS` — major:** Numerical domain exceeds reference implementations. Finite logits overflow intermediate arithmetic; finite nonzero vectors overflow norms; direction scaling changes geometric hit decisions. D1, D2, D11 provide executed witnesses and replacements.
- **`STMT_CASES` — major:** Convenient fixtures conceal required distinctions. Equal B/C weights hide vertex swaps; vertex A always suffices; zero-origin misses hide wrong fallback behavior; all finite depth gaps equal two. Seven wrong implementations pass all their stored cases.
- **`GIVE_RUNG` — major:** All 32 page-authored Faded starters are empty bodies. Rung labels promise assistance the source does not supply.
- **`GIVE_EXAMPLE` — major:** Eleven target drills reuse inputs from preceding lesson demonstrations. Classifier 1695 also reproduces the demonstrated loss values.
- **`STMT_OUTPUT` — major; `STMT_TERMS` — minor:** Standalone contracts rely on unstated conventions: tie winners, parameter meaning, uniform pixel spacing, barycentric interpretation.
- **`DIFF_RUNG` — minor:** Four Integrated tasks substantially repeat lower-rung work.
- **`EXPL_CORRECT` — major:** Main worked calculations execute, but surrounding prose includes false claims about matrix compatibility, floating-point cosine bounds, flattened shape errors, and parallelogram constraints.

No `DIFF_DUP` finding asserted from cross-rung similarity. No `CORR_VISUAL` finding: all target rows declare `expected_artifact_type: stdout`.

## 4. Top 10 fixes with rewrites

Ranked primarily by incorrect results and accepted wrong solutions. Cases below are grading examples, not worked implementation walkthroughs.

### 1. Drill 1705 — collapsed segments and scale-dependent misses

**Codes:** `CORR_RUNS` — major; D1, D2.

> Return the `(x,y)` coordinates where the forward ray intersects the closed segment; if there is no hit, return the segment point nearest to the ray’s origin. Return a float tensor of shape `(2,)`.

- `r`: float tensor `(2,3)`, storing `[origin,direction]`; `s`: float tensor `(2,3)`, storing `[start,end]`; all z-coordinates are zero.
- Ray points are `origin + u·direction`, with `u≥0`; direction is nonzero and need not have unit length.
- Parallel or degenerate pairs count as misses; a collapsed segment’s nearest point is its sole endpoint.

**Grade:** `r=[[0,0,0],[1,0,0]]`, `s=[[2,1,0],[2,1,0]]` → `[2,1]`.

### 2. Drill 1696 — top two can select the same class twice

**Code:** `CORR_RUNS` — major; D1.

> Return the fraction of examples whose true class appears among their two highest-scoring distinct class indices. Resolve equal scores by increasing class index and return a scalar float tensor.

- `x`: finite float tensor `(b,c)`, with `b≥1`, `c≥2`.
- `y`: integer tensor `(b,)`, with labels in `[0,c)`.
- Large equal scores are valid.

**Grade:** `x=[[100000000,100000000]]`, `y=[1]` → `1.0`.

### 3. Drill 1693 — temperature scaling overflows before stabilization

**Code:** `CORR_RUNS` — major; D1.

> Return each row’s temperature-scaled class probabilities as a float tensor of shape `(b,c)`, where class weights are proportional to `exp(score/temperature)`. Every row must sum to one and contain finite values, including rows with large equal scores.

- `x`: finite float tensor `(b,c)`, `b,c≥1`.
- `temp`: finite positive float tensor `(b,)`; entry `i` applies to row `i`.

**Grade:** `x=[[1e38,1e38]]`, `temp=[0.1]` → `[[0.5,0.5]]`.

### 4. Drill 1691 — entropy produces NaN and cases miss instability

**Codes:** `CORR_RUNS`, `STMT_CASES` — major; D1, D3.

> Return the entropy of each row’s softmax distribution as a float tensor of shape `(b,)`. Entropy is `−Σ p·ln(p)`, with zero-probability contributions defined as zero; results must remain finite for finite input logits.

- `x`: finite float tensor `(b,c)`, `b,c≥1`.
- Use natural-log units.

**Grade:** `x=[[3e38,-3e38]]` → `[0]`; also grade `[[1200,0]]` to reject naive probability-then-log arithmetic.

### 5. Drill 1699 — identical directions produce similarity −1

**Code:** `CORR_RUNS` — major; D11.

> Return the mean cosine similarity across all ordered pairs of different row indices in `y`, as a scalar float tensor. Rows with identical values still form a valid pair when their indices differ.

- `y`: finite float tensor `(n,d)`, `n≥2`, `d≥1`; every row is nonzero.
- Cosine similarity is the dot product divided by the product of the two Euclidean lengths.
- Large finite coordinates are valid.

**Grade:** `y=[[1e20,0],[1e20,0]]` → `1.0`.

### 6. Drill 1714 — swapped vertex weights pass grading

**Code:** `STMT_CASES` — major; D3.

> Return the hit’s barycentric weights `[wA,wB,wC]` as a float tensor of shape `(3,)`, satisfying `hit_point=wA·A+wB·B+wC·C` and `wA+wB+wC=1`. Return `[0,0,0]` on a miss.

- `r`: float tensor `(2,3)`, `[origin,direction]`; `tr`: float tensor `(3,3)`, vertices `[A,B,C]`.
- Ray parameters are nonnegative; triangle edges and vertices count as hits.
- Parallel or degenerate pairs are misses; nonzero direction magnitude does not change membership.

**Grade:** `r=[[0,0,0],[4,2,1]]`, `tr=[[4,0,0],[4,4,0],[4,0,4]]` → `[0.25,0.5,0.25]`.

Place on Solo unless another decision is added.

### 7. Drill 1715 — distance to A alone passes grading

**Code:** `STMT_CASES` — major; D3.

> Return the smallest Euclidean distance from the ray’s triangle hit point to any of the three vertices, as a scalar float tensor. Return `−1` if the ray misses.

- `r`: float tensor `(2,3)`, `[origin,direction]`; `tr`: float tensor `(3,3)`, vertices `[A,B,C]`.
- The forward ray includes its origin; triangle edges and vertices count as hits.
- Parallel or degenerate pairs are misses.

**Grade:** `r=[[0,0,0],[4,3,0]]`, `tr=[[4,0,0],[4,4,0],[4,0,4]]` → `1.0`; B is nearest.

### 8. Drill 1719 — every finite expected gap equals two

**Code:** `STMT_CASES` — major; D3.

> Return each ray’s second-smallest valid triangle-hit parameter minus its smallest, as a float tensor of shape `(nr,)`. Return positive infinity for fewer than two hit triangles; two different triangles at the same depth produce a gap of zero.

- `r`: float tensor `(nr,2,3)`, `[origin,direction]`; `tr`: float tensor `(nt,3,3)`.
- Hit parameter `s` satisfies `hit_point=origin+s·direction`, with `s≥0`.
- Triangle boundaries count; parallel or degenerate pairs are misses.

**Grade:** A unit `+x` ray from the origin hits triangles at `s=2` and `s=7` → `[5]`; add an equal-depth case → `[0]`.

### 9. Drill 1695 — first loss per group passes as a mean

**Codes:** `STMT_CASES`, `STMT_OUTPUT` — major; D3, D4.

> Return a float tensor `[mean_correct_loss,mean_wrong_loss]`, averaging per-example cross entropy separately over correctly and incorrectly classified examples. An empty group contributes zero; prediction uses the largest score, with ties choosing the smallest class index.

- `x`: finite float tensor `(b,c)`, `b,c≥1`.
- `y`: integer tensor `(b,)`, labels in `[0,c)`.
- Per-example loss is the negative natural logarithm of the true class’s softmax probability.

**Grade:** `x=[[0,0],[3,0],[0,2]]`, `y=[0,0,0]` → approximately `[0.370867,2.126928]`.

### 10. Drill 1718 — coincident nearest triangles have unspecified winners

**Code:** `STMT_OUTPUT` — major; D4.

> Return a Boolean tensor of shape `(nt,)` indicating which triangles are selected as the nearest hit by at least one ray. Each ray selects at most one triangle; equal hit parameters choose the smallest triangle index, and a ray with no hit selects none.

- `r`: float tensor `(nr,2,3)`, `[origin,direction]`; `tr`: float tensor `(nt,3,3)`.
- Hit parameter `s` satisfies `hit_point=origin+s·direction`, with `s≥0`.
- Triangle boundaries count; parallel or degenerate pairs are misses; directions need not have unit length.

**Grade:** One ray hitting two identical triangles → `[True,False]`.