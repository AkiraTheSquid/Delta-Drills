# Vocab Calibration Report — Part1 Ray Tracing (all 10 exercises)

**Date:** 2026-05-18
**Pass:** part1-pilot (full tagging of `chapter0_fundamentals/part1_ray_tracing/0_1_0` … `0_1_9`)
**Total tagged:** 10/10 exercises in `part1_ray_tracing`
**Vocab size:** 61 atoms in `vocab/atoms.json` (36 part2 + 25 part1)

---

## Headline stats

| Metric | Target | Part1 (n=10) | Part2 combined (n=24) | Verdict |
|---|---|---|---|---|
| Atoms per exercise (mean) | 4-7 | **7.90** | 4.88 | above band; driven by 4 dense composing exercises (5/6/8/9) |
| Atoms per exercise (range) | n/a | 3-14 | 1-9 | wider variance; rotation_matrix at floor, lighting at ceiling |
| Total atom tags | n/a | **79** | 117 | |
| False positives total | n/a (lower better) | **2** | 20 | seed predictions extremely accurate |
| Proposed new atoms | decaying | **0** | 4 (part2 batch1 only) | full vocab steady-state |
| Recurring atom fraction (≥2 of 10) | n/a | 17 of 27 = **63%** | 65% | matches part2 convergence |

**Per-exercise atom counts:** 0:4, 1:7, 2:10, 3:5, 4:6, 5:8, 6:10, 7:3, 8:12, 9:14
**Role breakdown:** 67 core, 12 incidental (15% incidental).

The mean of 7.90 lands above the 4-7 target band — but this is genuine pedagogical structure, not over-tagging:

- **3-5 atoms** (0, 3, 7): single-concept (construct-and-fill, einops-flatten, rotation literal)
- **6-8 atoms** (1, 4, 5): introduce one new core atom on top of a stable batch-geometry spine
- **10-14 atoms** (2, 6, 8, 9): compose the full vocab. 8 = 6+device (inheritance), 9 = 8+lighting (genuine new atoms)

Part1 has fewer trivial exercises than part2 (no 1-2 atom utility exercises like ReLU/pad) — every exercise builds on the geometric pipeline, so atom counts compound.

---

## Proposed new atoms

**Zero new atoms surfaced from part1 tagging.** All 10 exercises fit cleanly into the existing 27-atom part1 vocab (25 part1-specific + reused `einsum-contraction`, `broadcasting-rules`).

One **candidate atom** flagged in `0_1_9.json` notes but NOT promoted:

- `tensor-min-with-indices` — `Tensor.min(dim)` returning both vals and idx, distinct from `einops.reduce(..., 'min')` which is values-only. Load-bearing in 0_1_9 (the indices feed `index-by-tensor`). Will check chapter1 (attention argmax, top-k) for recurrence before promoting.

The seed vocab was correctly sized for part1.

---

## False-positive analysis

**2 false positives total** (0.2 per exercise — way down from part2's 0.83). The seed's `expected exercises` column was almost perfectly accurate.

| FP | Atom | Why removed |
|---|---|---|
| 0_1_7 | `stack-vs-cat` | Solution uses `t.tensor([[...]])` literal instead of `t.stack`; stylistic alternative |
| 0_1_9 | `einops-reduce-min` | Predicted from 0_1_6 pattern, but 0_1_9 switched to `s.min(dim=-1)` to get gather indices — semantically distinct op |

Note these are both "predicted-but-replaced-with-equivalent" rather than topic-only mentions. The systematic seed-overtag pattern that plagued part2 (broadcasting-rules predicted from topic-mentions, conv atoms from instantiation) is absent here — part1 atoms map cleanly to code constructs.

---

## Seed prediction accuracy (per-atom)

Comparing the seed's `expected exercises` column to actual tagged exercises:

| Verdict | Count | Atoms |
|---|---|---|
| **Exact MATCH** | **20 of 27** | most singletons + spine atoms |
| Extra exercise added | 2 | `slice-view-mutation` (+1,2 — view-form ubiquitous in early solves); `boolean-mask-combine` (+4 — triangle inside-test had 4 predicates) |
| Missing exercise(s) | 4 | `broadcasting-rules` (-5,6,8,9 — broadcasting handled by einops.repeat in solutions, not implicit); `einops-rearrange-flatten` (-8,9 — only fires in 0_1_3 and 0_1_7 driver); `einops-reduce-min` (-9 — switched to .min(dim=-1)); `unbind-tuple-unpack` (-9 — used indexed slicing instead); `ray-parametric-form` (-7,8,9 — geometry implicit, not reasoning-about) |
| Mixed | 1 | `ray-parametric-form` (+3, -7,8,9) |

**Headline:** seed accuracy of **20/27 exact + 3/27 partial = 85%** is far above part2's batch1 figure. The hand-drafted part1 vocab benefited from a full notebook read before drafting — the part2 seed was extrapolated from topic-mentions.

Predicted 86 total atom-tags, actual 79 → seed slightly **over**-predicted (+8.9%). Compatible with the "predict-broadcasting-when-broadcast-shows-up" pattern, mostly corrected here.

---

## Atom-usage frequencies — the spine of part1

Top atoms (≥5 exercises):

| atom | count | domain | exercises |
|---|---|---|---|
| `ray-parametric-form` | 7 | geometry | 0,1,2,3,4,5,6 |
| `stack-vs-cat` | 7 | tensor-algebra | 1,2,4,5,6,8,9 |
| `linalg-solve-batched` | 7 | linear-algebra | 1,2,4,5,6,8,9 |
| `einops-repeat-broadcast` | 6 | einops-patterns | 2,3,5,6,8,9 |
| `boolean-mask-combine` | 6 | boolean-masking | 2,4,5,6,8,9 |
| `unbind-tuple-unpack` | 5 | tensor-algebra | 1,4,5,6,8 |
| `singular-matrix-mask-trick` | 5 | linear-algebra | 2,5,6,8,9 |
| `triangle-barycentric` | 5 | geometry | 4,5,6,8,9 |

**Spine 1 — the batched-solve quintet:** `linalg-solve-batched` + `stack-vs-cat` + `singular-matrix-mask-trick` + `boolean-mask-combine` + `triangle-barycentric` (5+) co-fire in 5/6/8/9. This is the densest co-occurrence pattern in part1 — the "batched 3x3 geometric solve" idiom.

**Spine 2 — einops batch broadcasting:** `einops-repeat-broadcast` recurs in 6/10 exercises; `einops-rearrange-flatten` only fires in 0_1_3 (and 0_1_7 driver). The "outer product via repeat" pattern is far more ubiquitous than the `(y z)` fold/unfold.

**Spine 3 — ray-parametric-form:** technically the most-tagged (7/10) but with most tags marked `incidental` — it's a conceptual prerequisite that underpins every exercise but is rarely the *coding* focus after 0_1_1. This validates seed Q3's instinct to consider marking it `drillable: false` and routing to docs/SR.

**Singletons (1 exercise):** `linspace-out-param` (0), `try-except-solve` (1), `any-reduce-axis` (2), `rotation-matrix-3d` (7), `cross-product-normal`/`vector-normalize-keepdim`/`einsum-contraction`/`where-clip-negative`/`index-by-tensor` (all in 9), `broadcasting-rules` (2). 10 of 27 atoms are singletons; most are predicted to recur in chapter1 (einsum, normalize, where, gather all reappear in attention).

---

## Domain breakdown (where the work actually is)

| domain | atom-tags | % | comment |
|---|---|---|---|
| geometry | 15 | 19% | ray-parametric-form 7 + triangle-barycentric 5 + others |
| linear-algebra | 15 | 19% | linalg-solve 7 + singular-trick 5 + others |
| tensor-algebra | 14 | 18% | stack 7 + unbind 5 + einsum/broadcast |
| boolean-masking | 11 | 14% | mask-combine 6 + inf-mask 3 + any-reduce + where-clip |
| einops-patterns | 10 | 13% | repeat 6 + rearrange 2 + reduce 2 |
| tensor-construction | 8 | 10% | zeros 2 + slice-view 4 + linspace + index-by-tensor |
| device-perf | 6 | 8% | only fires in 7/8/9 |

Geometry + linear-algebra + tensor-algebra = 56% of all atom tags. Part1 is fundamentally a "batched 3x3 linear-algebra over geometric primitives" course, with einops as the connective tissue.

---

## Delta Drills coverage

Across 79 atom-tags:

| DD code | tags | % | seed prediction (vocab-level) |
|---|---|---|---|
| DD-N (no drill coverage) | 32 | **41%** | 44% — close match |
| DD-? (partial / uncertain) | 27 | **34%** | 26% — close match |
| DD-Y (covered) | 20 | **25%** | 30% — slight under-realization |

**Headline: 75% of part1 atom-tags point to DD-N or DD-? (no/uncertain drill coverage).** Healthier than part2's 84%, but the gap is still real.

Per-atom-tag breakdown highlights the drill investment opportunities:

- **`stack-vs-cat` (DD-?, 7 tags)** — the single highest-leverage DD-? atom in part1
- **`linalg-solve-batched` (DD-?, 7 tags)** — second highest; batched-linalg drill format is unexplored
- **`unbind-tuple-unpack` (DD-?, 5 tags)** — easy drill ("pick the unbind axis to get this shape")
- **`triangle-barycentric` (DD-N, 5 tags)** — geometry-specific, not a clean array-op drill
- **`singular-matrix-mask-trick` (DD-N, 5 tags)** — the single most "trick-flavored" atom; ideal for a compound drill

vs part2 calibration:
- Part1's DD-Y share (25%) is **higher** than part2's batch-combined (16%) → einops/numpy drills naturally cover more part1 content
- Part1's DD-N share (41%) is **dramatically lower** than part2's 50% (and 64% vocab-level) → no Module-mechanics/training-infra cluster pulling the average up
- The DD-? cluster is the obvious extension target for both parts: `linalg-solve-batched` (part1) and `as-strided-windowing` (part2)

---

## Comparison to part2 calibration

| Dimension | Part1 (n=10) | Part2 (n=24) | Read |
|---|---|---|---|
| Mean atoms/exercise | 7.90 | 4.88 | Part1 exercises compose more atoms (every exercise builds on prior) |
| FP rate | 0.20 | 0.83 | Part1 seed was hand-drafted from full notebook reads; part2 was extrapolated |
| New atoms surfaced | 0 | 4 (batch1 only; 0 in batch2) | both vocabs at steady-state |
| DD-Y % of tags | 25% | 16% | part1 better-covered by existing einops drills |
| DD-N % of tags | 41% | 50% | both have substantial gaps |
| Recurring atom % | 63% | 65% | comparable convergence |
| Singleton fraction | 10/27 = 37% | 14/40 = 35% | comparable |

Part1 is a "denser, more focused, better-seed" cohort. Part2 will pull this average down once chapter1 expands the corpus.

---

## Recommendations

### 1. Accept the part1 vocab as-is — no merges, splits, renames, or drops.

The user pre-locked all defaults in the vocab review; the actual tagging supports this. Specifically:

- **Q1 (singular-matrix-mask-trick granularity):** keep composite. Fires as a unit in 5 exercises; splitting would create 3 sub-atoms each at 5 hits — bad signal/noise.
- **Q2 (stack-vs-cat split):** keep joint. Every exercise uses `stack`, never `cat`, but the *contrast* IS the conceptual content — and a dedicated tutorial aside in 0_1_1 teaches both sides.
- **Q3 (ray-parametric-form drillable):** **flag for future demotion.** 7-tag count is mostly incidental (5 of 7 incidental). Strong signal for `drillable: false` + docs/SR route, exactly as Q3 anticipated.
- **Q4 (triangle-barycentric split):** keep composite. Confirmed — the three sub-pieces fire together in all 5 exercises.
- **Q5 (tensor-reshape-view vs einops-rearrange-flatten):** keep separate. The (y z) grammar is the load-bearing skill in 0_1_3; vanilla `.view()` doesn't appear anywhere in part1.
- **Q6 (jaxtyping/imshow/etc):** confirmed cut. Zero false-positive temptation to tag any of these.
- **Q7 (cross-chapter reuse):** flag for chapter1 — `singular-matrix-mask-trick`, `einops-repeat-broadcast`, `einsum-contraction`, `vector-normalize-keepdim`, `cross-product-normal` (no, this one's geometry-bound), `where-clip-negative` are all high-probability transformer-block recurrences.

### 2. Edge candidates for the prereq DAG.

Based on co-occurrence and exercise order:

- `ray-parametric-form` → `segment-line-intersect-2d` → `triangle-barycentric` (the geometric ladder; 1→2D→3D)
- `linalg-solve-batched` (scalar in 0_1_1) → `singular-matrix-mask-trick` (batched in 0_1_2) — the "scalar try/except → batched mask-trick" transition is the single sharpest learning step in part1
- `linalg-solve-batched` + `stack-vs-cat` + `unbind-tuple-unpack` → tightly co-fire 5+ times → "the linear-system idiom" composite node
- `einops-repeat-broadcast` + `einops-rearrange-flatten` → distinct but both prerequisites for `einops-reduce-min`
- `tensor-zeros-init` + `linspace-out-param` + `slice-view-mutation` → tightly co-fire in 0_1_0 → "preallocate-and-fill" composite node
- `tensor-to-device` + `device-consistent-construct` → always co-fire (0_1_8, 0_1_9) → "GPU-port a function" composite node
- `triangle-barycentric` + `inf-masking` + `einops-reduce-min` → unique to 0_1_6 → "min-over-batch with dropout" composite

### 3. New candidate atom to track (not promoting yet).

`tensor-min-with-indices` — `t.Tensor.min(dim)` returning `(vals, idx)`, used in 0_1_9 to feed the per-ray gather. Distinct enough from `einops-reduce-min` (which is values-only) that splitting may be warranted if it recurs in chapter1.

### 4. Stop tagging here; next blockers are decisions, not data.

- Decide ray-parametric-form's `drillable` status (Q3).
- Resolve the DD-? cluster (extend drill set to batched-linalg-solve drills + as-strided drills from part2).
- Run chapter1 (transformer-interp) tagging to test cross-chapter reuse hypotheses.

---

## "All defaults" review check

**The user's all-defaults lock held up.** No vocab decisions emerged from tagging that would have flagged for human review:

- No atom was over-tagged in a way that suggested it should have been split
- No two atoms collapsed in usage in a way that suggested merging
- No atom was systematically absent in a way that suggested it should have been cut
- The 2 false positives (`stack-vs-cat` in 0_1_7, `einops-reduce-min` in 0_1_9) are both "predicted-but-replaced-with-equivalent", not "wrongly predicted at all"

The only forward-looking observation worth flagging is Q3's `ray-parametric-form` `drillable` question — but that's already documented in the seed markdown as a deferred decision, not a new finding.

---

## What changed from part2 calibration

- **Mean atoms/exercise:** 4.88 → 7.90 (part1 exercises compound; every late exercise builds on prior; no trivial utility exercises)
- **FP rate:** 0.83 → 0.20 (hand-drafted-from-source seed vs extrapolated-from-topics seed)
- **New atoms:** 4 (part2 batch1) → 0 (part1) (part1 seed was already at steady-state by design)
- **DD-Y share of tags:** 16% → 25% (part1 is einops-natural; part2 is Module-mechanics-heavy)
- **DD-N share of tags:** 50% → 41% (no training-infra/Module-mechanics cluster)
- **Singleton fraction:** comparable (35% vs 37%)

Part1 is the "well-behaved" half of chapter0 — single coherent application domain (geometry), tight conceptual ladder, accurate hand-drafted seed.
