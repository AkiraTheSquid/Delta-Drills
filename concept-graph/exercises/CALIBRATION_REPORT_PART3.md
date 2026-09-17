# Vocab Calibration Report — Part3 Optimization (all 12 exercises)

**Date:** 2026-05-18
**Pass:** part3-pilot (full tagging of `chapter0_fundamentals/part3_optimization/0_3_0` … `0_3_11`)
**Total tagged:** 12/12 exercises in `part3_optimization`
**Vocab size:** 123 atoms in `vocab/atoms.json` (36 part2 + 25 part1 + 62 part3)

---

## Headline stats

| Metric | Target | Part3 (n=12) | Part1 (n=10) | Part2 combined (n=24) | Verdict |
|---|---|---|---|---|---|
| Atoms per exercise (mean) | 4-7 | **13.17** | 7.90 | 4.88 | well above band; predicted in seed (11.3) but landed even higher |
| Atoms per exercise (range) | n/a | 4-32 | 3-14 | 1-9 | widest variance yet; 0_3_11 is the densest exercise across all of chapter 0 |
| Total atom tags | n/a | **158** | 79 | 117 | |
| False positives total | n/a (lower better) | **7** | 2 | 20 | 0.58 per exercise — middling |
| Proposed new atoms | decaying | **0** | 0 | 4 (part2 batch1 only) | full vocab steady-state |
| Recurring atom fraction (≥2 of 12) | n/a | 45 of 68 = **66%** | 63% | 65% | stable convergence |

**Per-exercise atom counts:** 0:6, 1:12, 2:14, 3:16, 4:15, 5:7, 6:14, 7:17, 8:8, 9:4, 10:13, 11:32
**Role breakdown:** 155 core, 3 incidental (1.9% incidental — much lower than parts 1/2; explained below).
**Unique atoms used:** 68 of 123 vocab entries (55%) — the part3 sub-vocab covers ~all of part3 work plus a handful of cross-part reuses.

The mean of 13.17 lands far above the 4-7 target band and above seed prediction of 11.3 — this is *real* density, not over-tagging:

- **4-8 atoms** (0, 5, 8, 9): single-concept exercises (SGD-on-tensor harness, optimizer-comparison harness, sweep config, broadcast protocol)
- **12-17 atoms** (1, 2, 3, 4, 6, 7, 10): full optimizer-impl spine (9 plumbing atoms + 2-4 math) OR full training-loop spine (~10 atoms) OR distributed-primitive cluster (~8 atoms)
- **32 atoms** (11 only): the chapter capstone — composes optimizer-state + training-loop + wandb + every distributed primitive + every DDP-integration atom + both NEW instrumentation atoms

Part3 has zero trivial 1-2 atom utility exercises, and three exercises compose ≥15 atoms — that's why the mean climbs.

---

## Proposed new atoms

**Zero new atoms surfaced from part3 tagging.** All 12 exercises fit cleanly into the existing 62-atom part3 sub-vocab.

The 7 NEW atoms added during seed drafting (optimizer-repr-string, pseudocode-to-code-translate, loss-item-scalar-extract, dataloader-pin-memory-workers, dataclasses-replace-args, time-stage-instrumentation, model-save-state-dict) all earned ≥1 tag during extraction, validating their inclusion:

| atom | tags | exercises |
|---|---|---|
| `pseudocode-to-code-translate` | 4 | 1, 2, 3, 4 |
| `optimizer-repr-string` | 4 | 1, 2, 3, 4 |
| `dataloader-pin-memory-workers` | 1 | 11 |
| `dataclasses-replace-args` | 1 | 8 |
| `loss-item-scalar-extract` | 2 | 7, 11 |
| `time-stage-instrumentation` | 1 | 11 |
| `model-save-state-dict` | 1 | 11 |

The seed-drafted vocab was correctly sized for part3 — no further additions needed.

---

## False-positive analysis

**7 false positives total** (0.58 per exercise — higher than part1's 0.20, lower than part2's 0.83). Pattern is meaningful:

| FP | Atom | Why removed |
|---|---|---|
| 0_3_4 | `weight-decay-l2-add` | The defining point of AdamW is that it does NOT use L2-style g+=lambda*theta; replaced with `weight-decay-decoupled`. Seed predicted from the SGD/RMSprop/Adam pattern but the seed itself flagged this as a contrast (Q3). |
| 0_3_4 | `conditional-hparam-branch` | Solution applies `theta *= 1 - lr*lmda` unconditionally (lmda=0 makes it a no-op); SGD/RMSprop/Adam solutions all use `if lmda != 0:` shortcut, AdamW does not. |
| 0_3_6 | `optimizer-repr-string` | Template/solution for param-groups SGD rewrite does not define `__repr__` — focus is on init/zero_grad/step structure only. |
| 0_3_6 | `pseudocode-to-code-translate` | Exercise is structural refactor of existing SGD, not pseudocode→code. The math is unchanged from 0_3_1. |
| 0_3_7 | `freeze-requires-grad` | Seed flagged this as speculative tag for the implied `ResNetFinetuner` parent class (R5.7); we don't see that parent in the cells, so removed for conservative-to-visible-code stance. |
| 0_3_7 | `dataloader-batching` | DataLoader is constructed in the parent ResNetFinetuner (not visible); the subclass inherits self.train_loader without recreating it. |
| 0_3_11 | `trainer-subclass-extend` | Markdown explicitly says "We recommend not using inheritance for this" — solution builds DistResNetTrainer as a sibling class (no super() calls). |

**Three patterns:**
1. **AdamW-specific cuts (2 FPs):** seed over-predicted by analogy from Adam; AdamW's whole point is *contrast*
2. **Refactor exercises (2 FPs in 0_3_6):** when an exercise restructures rather than re-implements, meta atoms (pseudocode-to-code, repr) drop out
3. **Cross-cell inheritance (3 FPs in 0_3_7, 0_3_11):** seed extrapolated from textual descriptions ("inherits from ResNetFinetuner"); tagging policy is "only tag what's in the visible cells"

Note: the AdamW FPs were *predicted* by the seed review (Q3 explicitly framed weight-decay-decoupled vs weight-decay-l2-add as "the conceptual content of the exercise"). The seed correctly anticipated this would be a place where the spine atom doesn't carry forward.

---

## Seed prediction accuracy

Predicted **~135 atom-tags across 12 exercises**, actual **158**. Seed under-predicted by ~17%.

Sources of under-prediction:
- **Exercise 11** predicted 25 atoms, actual 32 — wandb-init-run, wandb-log-step, wandb-finish, tqdm-postfix-metrics, validation-no-grad, train-eval-mode-branch, tensor-to-device all under-predicted (seed treated these as "inherited from 0_3_7" but the sibling class re-implements them)
- **Exercise 10** predicted 10, actual 13 — driver block (run_simple_model) contains 5 more DDP-integration atoms than the seed accounted for (init-process-group, mp-spawn-workers, per-rank-cuda-device, tensor-to-device, nn-parameter-wrap)
- **Exercise 7** predicted 16, actual 17 — backward-on-scalar-loss snuck in (the spine atom is in the training_step body)

No exercise was significantly over-predicted. Seed's per-exercise predictions were all within 3 of actual except for 11 (+7) and 10 (+3).

---

## Atom-usage frequencies — the spine of part3

Top atoms (≥4 exercises):

| atom | count | domain | exercises |
|---|---|---|---|
| `param-grad-access` | 6 | optimizer-state-plumbing | 1,2,3,4,6,11 |
| `buffer-copy_-inplace` | 6 | optimizer-state-plumbing | 1,2,3,4,6,9 |
| `optimizer-init-params-list` | 5 | optimizer-state-plumbing | 1,2,3,4,6 |
| `optimizer-state-tensor-buffers` | 5 | optimizer-state-plumbing | 1,2,3,4,6 |
| `zero-grad-set-none` | 5 | optimizer-state-plumbing | 1,2,3,4,6 |
| `inplace-param-update` | 5 | optimizer-state-plumbing | 1,2,3,4,6 |
| `inference-mode-step` | 5 | optimizer-state-plumbing | 1,2,3,4,6 |
| `backward-on-scalar-loss` | 5 | optim-loop-harness | 0,5,7,10,11 |
| `training-step-cycle` | 5 | training-infra | 0,5,7,10,11 |
| `weight-decay-l2-add` | 4 | optimizer-math | 1,2,3,6 |
| `conditional-hparam-branch` | 4 | optimizer-math | 1,2,3,6 |
| `pseudocode-to-code-translate` | 4 | optimizer-math | 1,2,3,4 |
| `optimizer-repr-string` | 4 | optimizer-state-plumbing | 1,2,3,4 |

**Spine 1 — optimizer-impl septet:** the 7 plumbing atoms (`optimizer-init-params-list` / `optimizer-state-tensor-buffers` / `zero-grad-set-none` / `param-grad-access` / `inplace-param-update` / `buffer-copy_-inplace` / `inference-mode-step`) all co-fire in 5 exercises (1, 2, 3, 4, 6). The single tightest co-occurrence cluster in any part. This IS "implementing an optimizer".

**Spine 2 — pseudocode-translation quartet:** `weight-decay-l2-add` + `conditional-hparam-branch` + `pseudocode-to-code-translate` + `optimizer-repr-string` co-fire in (1, 2, 3) with `weight-decay-l2-add`+`conditional-hparam-branch` also picking up exercise 6 (param-groups SGD rewrite). The optimizer-math sub-spine.

**Spine 3 — optim-loop harness:** `backward-on-scalar-loss` + `training-step-cycle` recur in 5 exercises (0, 5, 7, 10, 11). The "fwd→loss→back→step→zero" body is part3's most cross-cutting non-optimizer concept.

**Singletons (1 exercise):** 23 of 68 atoms (34%). Of those: all 4 parameter-group atoms (6 only), 5 of 9 wandb atoms (8 only), `weight-decay-decoupled` (4 only), 4 distributed/DDP atoms in exercise 11 (broadcast-initial-weights, all-reduce-grad-sync, distributed-sampler-shard, all-reduce-eval-metrics), the two NEW instrumentation atoms (11 only), and dataloader-pin-memory-workers (11 only).

The high singleton fraction (34%) is *structural* — each of (6 = param groups, 8 = sweeps, 9/10 = collectives, 11 = full DDP) is the only exercise of its sub-domain. Identical pattern to part1 (37%) and part2 (35%).

---

## Domain breakdown (where the work actually is)

| domain | atom-tags | % | comment |
|---|---|---|---|
| optimizer-state-plumbing | 43 | 27% | the spine — 7 atoms × 5-6 exercises each |
| optimizer-math | 26 | 16% | weight-decay + momentum + EMA + bias-correction + sqrt-eps |
| wandb | 16 | 10% | init/log/watch/finish + sweep cluster |
| training-scaffolding | 15 | 10% | dataclass-args, trainer-class, ce-loss, argmax-accuracy, tqdm, loss-item, pin-memory |
| optim-loop-harness | 14 | 9% | backward, detach-clone, t-stack, optimizer-loop-on-tensor, optimizer-class-dispatch |
| ddp-integration | 14 | 9% | init-process-group, mp-spawn, per-rank-device, broadcast-init, all-reduce-grad-sync, sampler-shard, rank0-only, all-reduce-eval, time-stage, model-save |
| distributed-primitives | 12 | 8% | rank-world-size, send-recv, broadcast-fanout, reduce-gather, all-reduce-compose, reduce-mean |
| training-infra | 8 | 5% | training-step-cycle, validation-no-grad, dataloader-batching, train-eval-mode-branch |
| parameter-groups | 4 | 3% | all 4 atoms, all in exercise 6 |
| device-perf | 3 | 2% | tensor-to-device (cross-part reuse) |

Optimizer-state + optimizer-math = 43% of all tags. Part3 is fundamentally a **"build your own optimizer"** course, with wandb/training/distributed as the surrounding scaffolding.

---

## Delta Drills coverage

Across 158 atom-tags:

| DD code | tags | % | seed prediction (vocab-level) |
|---|---|---|---|
| DD-N (no drill coverage) | 123 | **77.8%** | 86% — close match |
| DD-? (partial / uncertain) | 35 | **22.2%** | 14% — over-realized; DD-? cluster fired more than expected |
| DD-Y (covered) | 0 | **0%** | 0% — exact match |

**Headline: 100% of part3 atom-tags point to DD-N or DD-?, with ZERO DD-Y coverage.** This is the heaviest gap of any part — predicted in the task brief and seed markdown.

Per-atom-tag breakdown of DD-? highlights (these are drill investment opportunities):

- **`buffer-copy_-inplace` (DD-?, 6 tags)** — highest-leverage DD-? atom in part3; semantically distinct from generic inplace ops (storage-shared mutation)
- **`optimizer-state-tensor-buffers` (DD-?, 5 tags)** — zeros_like-per-param pattern
- **`inplace-param-update` (DD-?, 5 tags)** — the autograd-preserving in-place `-=`
- **`weight-decay-l2-add` (DD-?, 4 tags)** — pure tensor op, drill candidate
- **`momentum-buffer-update` (DD-?, 3 tags)** — EMA-style update
- **`ema-second-moment` (DD-?, 3 tags)** — g.pow(2) EMA
- **`ema-first-moment` (DD-?, 2 tags)** — g EMA
- **`sqrt-eps-stabilize` (DD-?, 3 tags)** — denominator stabilization

The optimizer-math cluster (5 atoms, 13 tags total) is the obvious drill-extension target: each is a 1-2 line tensor op that composes the canonical "given m, v, g, beta, write the AdamW step" drill.

vs part1/part2 calibrations:
- Part3's DD-Y share (0%) is dramatically lower than part1 (25%) or part2 (16%) — confirms the part3 vocab cannot be drilled by current array-op format
- Part3's DD-N share (78%) is highest of any part (part1: 41%, part2: 50%)
- Per the user's note: "separate planned pipeline for math practice" makes the DD-N skew non-blocking

---

## Comparison to part1/part2 calibrations

| Dimension | Part1 (n=10) | Part2 (n=24) | **Part3 (n=12)** | Read |
|---|---|---|---|---|
| Mean atoms/exercise | 7.90 | 4.88 | **13.17** | monotonic climb — exercises get more compositional each part |
| Max single-exercise atom count | 14 (0_1_9) | 9 | **32 (0_3_11)** | capstone exercise is in part3 |
| FP rate | 0.20 | 0.83 | **0.58** | middling — seed under-predicted some inheritance & contrast cases |
| New atoms surfaced | 0 | 4 (batch1 only) | **0** | both vocabs at steady-state |
| DD-Y % of tags | 25% | 16% | **0%** | monotonic collapse — part3 vocab is fundamentally non-array-op |
| DD-N % of tags | 41% | 50% | **78%** | monotonic climb |
| Recurring atom % | 63% | 65% | **66%** | comparable convergence |
| Singleton fraction | 37% | 35% | **34%** | comparable |
| Incidental tag % | 15% | n/a | **1.9%** | much lower — part3 atoms are all coding-focal, no "geometric prerequisite" atoms like ray-parametric-form |

The incidental fraction collapse (15% → 1.9%) is the most interesting structural shift: part1 had `ray-parametric-form` as a 7-tag "incidental backbone" concept; part3 has *no* analogous conceptual scaffold. Every part3 atom is something the learner explicitly writes.

---

## Recommendations

### 1. Accept the part3 vocab as-is — no merges, splits, renames, or drops.

The 7 NEW atoms from the seed-drafting pass all earned ≥1 tag. The pre-seeded 55 atoms covered the rest cleanly. Specifically:

- **Q1 (optimizer-state cluster granularity, 7 atoms):** the 7 atoms **all** co-fire in 5 exercises as a unit. Each independently caught a real failure mode (e.g. `optimizer-init-params-list` and `buffer-copy_-inplace` are subtle), but the perfect co-fire pattern means a composite `optimizer-step-plumbing` *would* preserve the signal. **Defer to user; current granular split works fine for extraction.**
- **Q2 (`momentum-buffer-update` vs `ema-first-moment`):** they truly don't co-fire — momentum in 1/2/6, first-moment in 3/4. Keep separate; the pedagogical distinction holds.
- **Q3 (weight-decay-l2 vs decoupled):** confirmed by 0_3_4 false positive — these are *contrast* atoms, not merge candidates. Keep separate.
- **Q4 (parameter-group atoms):** all 4 fired together in exercise 6 only. Could collapse to one composite `param-groups` atom with no signal loss — but keeping granular preserves the 4-failure-mode mapping.
- **Q5 (trainer-class-skeleton vs training-step-cycle):** confirmed distinct: trainer-class-skeleton fires in 7/11, training-step-cycle in 0/5/7/10/11 (and is also the inner body of 7/11). They co-fire when both present but training-step-cycle has wider reach.
- **Q6 (wandb cluster):** all 4 lifecycle atoms (init/log/watch/finish) fired in 7 and partial overlap in 8/11. Could merge `wandb-init-run` + `wandb-finish` into `wandb-run-lifecycle`, but tests would lose granularity for the "I forgot wandb.finish" failure mode. **Keep current.**
- **Q7 (broadcast-source-fanout vs dist-send-recv-pair):** confirmed distinct levels — send/recv fires in 9/10 (low-level), broadcast-fanout fires in 9/11 (high-level). Both load-bearing.
- **Q8 (cross-part reuses):** `nn-parameter-wrap` fired in 0_3_10 as predicted. `freeze-requires-grad` did NOT fire (FP in 0_3_7); the speculative tag was correctly identified as a removal candidate.
- **Q9 (cuts):** confirmed; none of the cut atoms snuck back in.
- **Q10 (`trainer-subclass-extend` drillable):** fired in 0_3_7 (super().pre_training_setup) but NOT in 0_3_11 (markdown explicitly says don't inherit). Becomes a 1-tag atom — flag for `drillable: false` along with `trainer-class-skeleton`.

### 2. Edge candidates for the prereq DAG.

Based on co-occurrence and exercise order:

- The optimizer-state septet (7 atoms) → fire as a unit in exercises 1-6 → "the optimizer plumbing idiom" composite node
- `pseudocode-to-code-translate` → unique to 1-4 (optimizer impl) → strong candidate for a META atom feeding all four optimizer drills
- `weight-decay-l2-add` → `weight-decay-decoupled` — contrast pair, the *step* between Adam (3) and AdamW (4)
- `ema-second-moment` (RMSprop+Adam+AdamW) → `ema-first-moment` (Adam+AdamW) → `bias-correction-divide` (Adam+AdamW) — the "adaptive optimizer ladder"
- `dist-send-recv-pair` → `broadcast-source-fanout` → `reduce-gather-sum` → `all-reduce-compose` → `all-reduce-grad-sync` — the collective-comm ladder, perfectly linear across exercises 9 → 10 → 11
- `wandb-init-run` + `wandb-log-step` + `wandb-finish` → tightly co-fire in 7/11 → "wandb run lifecycle" composite
- `sweep-config-dict` + `sweep-hparam-distribution` + `sweep-agent-launch` + `wandb-config-into-args` → only in 8 → "wandb sweep" composite

### 3. NEW candidate atom to track (not promoting yet).

None. The seed pass already added the 7 atoms that surfaced from full notebook reads.

### 4. Stop tagging here; next blockers are decisions, not data.

- Decide `trainer-class-skeleton` and `trainer-subclass-extend` `drillable` status (matches part1's Q3 instinct for `ray-parametric-form`).
- Resolve the DD-? cluster: extend drill set to "fused EMA-style updates" + "weight decay variants" + "buffer-copy_-inplace" semantics drills.
- Build the planned separate math-practice pipeline to cover the optimizer-math + protocol atoms that the array-op format can't reach.

---

## "All defaults" review check

**The user's all-defaults lock held up.** No vocab decisions emerged from tagging that would have flagged for human review:

- Q3 (weight-decay-l2 vs decoupled) was explicitly anticipated and 0_3_4 confirmed the seed's contrast framing was right
- Q8 (speculative cross-part reuse `freeze-requires-grad`) was anticipated as low-cost; tagging confirmed it as removable
- Q10 (`trainer-subclass-extend` drillable) had hard data added (1-tag count) — re-flag for next pass but not blocking
- No atom over-tagged in a way suggesting it should split
- No atom systematically absent in a way suggesting it should be cut
- All 7 NEW atoms earned ≥1 tag

The 7 false positives are concentrated in 3 contexts (AdamW contrast, refactor exercises, cross-cell inheritance) — all forward-predictable patterns, not vocab-design failures.

---

## Validator status

`python3 concept-graph/scripts/validate.py` exits with code 1 — but **all part3 errors are clean**: every atom referenced in the 12 new `0_3_*.json` files resolves to a known vocab entry. The validator's 4 PROPOSED NEW ATOMS warnings are all from pre-existing part2 files (`0_2_1.json, 0_2_4.json, 0_2_5.json, 0_2_18.json, 0_2_19.json, 0_2_22.json`) referencing `conv-kernel-shape`, `functional-module-wrap`, `kaiming-uniform-init`, `register-buffer` — these are part2-pipeline artifacts not introduced by this pass.

---

## Questions logged to REVIEW_QUEUE.md during tagging

Zero new questions logged. All decisions were either:
- Already covered by Q1-Q10 in the vocab-gen review queue
- Default-resolvable (e.g. "exercise atom count exceeded prediction" → tag what's there)
- Confirmable false positives (logged in `atoms_in_seed_but_not_actually_present` per-exercise rather than as queue questions)

---

## What changed from part1/part2 calibrations

- **Mean atoms/exercise:** 4.88 → 7.90 → 13.17 (monotonic compositional climb)
- **Max single-exercise count:** 9 → 14 → 32 (capstone exercise lives in part3)
- **FP rate:** 0.83 → 0.20 → 0.58 (regressed from part1 — seed under-predicted inheritance and refactor cases)
- **New atoms surfaced post-seed:** 4 → 0 → 0 (vocab-design has converged for chapter 0)
- **DD-Y share of tags:** 25% → 16% → 0% (numpy/einops drills don't cover part3 at all)
- **DD-N share of tags:** 41% → 50% → 78% (part3 is purely Python/protocol/algorithmic)
- **Incidental tag share:** 15% → n/a → 1.9% (no conceptual-scaffold atoms in part3)

Part3 is the **"protocols and procedures"** half of chapter 0 — three distinct sub-cohorts (optimizer-impl, ML ops, distributed) with no shared mathematical primitive. The connective tissue is the training loop itself.
