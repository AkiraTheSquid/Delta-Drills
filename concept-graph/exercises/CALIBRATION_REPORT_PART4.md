# Vocab Calibration Report — Part4 Backprop (all 21 exercises)

**Date:** 2026-05-18
**Pass:** part4-pilot (full tagging of `chapter0_fundamentals/part4_backprop/0_4_0` … `0_4_20`)
**Total tagged:** 21/21 exercises in `part4_backprop`
**Vocab size:** 177 atoms in `vocab/atoms.json` (36 part2 + 25 part1 + 62 part3 + 54 part4)

---

## Headline stats

| Metric | Target | Part4 (n=21) | Part3 (n=12) | Part1 (n=10) | Part2 (n=24) | Verdict |
|---|---|---|---|---|---|---|
| Atoms per exercise (mean) | 4-7 | **6.48** | 13.17 | 7.90 | 4.88 | back in band — many per-op exercises pull mean down |
| Atoms per exercise (range) | n/a | 2-14 | 4-32 | 3-14 | 1-9 | wide variance; 0_4_14 (sum) is densest non-capstone |
| Total atom tags | n/a | **136** | 158 | 79 | 117 | |
| False positives total | n/a (lower better) | **6** | 7 | 2 | 20 | 0.29 per exercise — matches part1 |
| Proposed new atoms | decaying | **0** | 0 | 0 | 4 (part2 batch1) | full vocab steady-state |
| Recurring atom fraction (≥2 of 21) | n/a | 29 of 63 = **46%** | 66% | 63% | 65% | lower — part4 has many singleton per-op atoms by design |

**Per-exercise atom counts:** 0:3, 1:3, 2:7, 3:6, 4:2, 5:6, 6:8, 7:12, 8:4, 9:13, 10:5, 11:6, 12:5, 13:5, 14:14, 15:8, 16:2, 17:6, 18:9, 19:3, 20:9
**Role breakdown:** 131 core, 5 incidental (3.7% incidental — comparable to part3, far below part1's 15%).
**Unique atoms used:** 63 of 177 vocab entries (36%) — the part4 sub-vocab is well-scoped.

The mean of 6.48 falls back into the 4-7 target band, BELOW part3's 13.17 — explained by part4's high count of trivial per-op exercises (0,1,4,10,11,12,13,16, all ≤6 atoms). The §6 capstone exercises (7,9,14,18,20) compose 9-14 atoms each:

- **2-5 atoms** (0,1,4,8,10,11,12,13,16,19): per-op derivative exercises + toposort + cross_entropy composition — single-concept by design
- **6-9 atoms** (2,3,5,6,15,17,18,20): per-op with broadcasting OR custom-Tensor-wrapper construction OR custom Linear/SGD
- **12-14 atoms** (7,9,14): the three "spine" exercises — wrap_forward_fn, backprop, sum-and-elementwise-cluster

---

## Proposed new atoms

**Zero new atoms surfaced from part4 tagging.** All 21 exercises fit cleanly into the existing 54-atom part4 sub-vocab plus 9 cross-part reuses.

The 54 NEW part4 atoms added during seed drafting all earned ≥1 tag during extraction, validating their inclusion. No atom was unused. Most-used part4 atoms:

| atom | tags | exercises |
|---|---|---|
| `backward-fn-signature` | 10 | 0,2,3,10,11,12,13,14,15,17 |
| `register-back-fn-after-wrap` | 9 | 4,10,11,12,13,14,15,17 (note: 4 has it as the *introduction* exercise) |
| `wrap-forward-fn-generic` | 8 | 7,10,11,12,13,14,15,17 |
| `tensor-wraps-ndarray` | 6 | 5,6,7,9,18,19 |
| `chain-rule-elementwise` | 5 | 0,2,10,11,15 |
| `unbroadcast-pattern` | 4 | 1,2,14,15 |
| `kwargs-pass-through-recipe` | 4 | 7,12,13,14 |
| `arg-position-back-functions` | 4 | 2,15,17 (multi-arg backwards) |
| `grad-tracking-global-toggle` | 4 | 5,6,7,20 |
| `recipe-dataclass` | 4 | 5,6,7,9 |
| `parents-dict-by-argidx` | 4 | 5,6,7,9 |

Cross-part reuses fired:
- `broadcasting-rules` (DD-Y, 4 tags) — in 1, 2, 14, 15. As predicted.
- `nn-parameter-wrap` (DD-N, 1 tag) — 18 only. As predicted.
- `nn-module-subclass` (DD-N, 1 tag) — 18 only. As predicted.
- `relu-elementwise-max` (DD-Y, 1 tag) — 16 only. As predicted.
- `module-extra-repr` (DD-N, 1 tag) — 18 only. As predicted (was in original 36-atom vocab, fires here too).
- `matmul-2d` (DD-Y, 1 tag) — 17 only. Cross-part reuse from part1 vocab.
- `optimizer-init-params-list` / `zero-grad-set-none` / `inplace-param-update` (DD-N, 1 tag each) — 20 only. Part3 optimizer-state-plumbing atoms reused in the minimal SGD exercise.

The seed-drafted vocab was correctly sized for part4 — no further additions needed.

---

## False-positive analysis

**6 false positives total** (0.29 per exercise — comparable to part1's 0.20, much lower than part2's 0.83 or part3's 0.58). The FPs concentrate in two systematic patterns:

| FP | Atom | Why removed |
|---|---|---|
| 0_4_18 | `reshape-back` | Seed predicted reshape-back for 18 because MLP.forward does x.reshape((x.shape[0], 28*28)). But the actual coding task is the Linear class only; the MLP class is shown as a downstream demo, no reshape_back code is written here (that was 0_4_12). |
| 0_4_19 | `exp-back` | exp_back is REGISTERED in 0_4_11. Exercise 19's cross_entropy invokes .exp() via autograd dispatch but writes no new exp_back code. |
| 0_4_19 | `reshape-back` | cross_entropy doesn't call reshape. The implicit reshape in MLP usage is upstream; not in the cross_entropy fn body. |
| 0_4_19 | `sum-back-expand-broadcast` | sum_back is REGISTERED in 0_4_14. Exercise 19's .sum(-1, keepdim=True) is autograd-dispatched; no new sum_back code written here. |
| 0_4_19 | `getitem-back-add-at` | getitem_back is REGISTERED in 0_4_14. Exercise 19's [arange, true_labels] indexing is autograd-dispatched; no new getitem_back code written here. |
| 0_4_20 | `leaf-tensor-condition` | Seed predicted this for SGD; but SGD just reads p.grad (set by backprop in 0_4_9). The is_leaf logic itself lives in the Tensor class / backprop loop, not in the SGD code. |

**Two patterns:**
1. **Composition-without-implementation (5 of 6 FPs, all in 0_4_19):** when an exercise INVOKES a registered op via autograd dispatch rather than implementing it from scratch, the §2 per-op-derivative atom should NOT fire. Tagging policy: §2 atoms tag at REGISTRATION exercise only. This is a meaningful pattern, logged as Q14 in REVIEW_QUEUE for review.
2. **Out-of-scope predicted atom (1 FP in 0_4_18, 1 FP in 0_4_20):** seed over-predicted by analogy from related concepts (reshape used in MLP downstream demo; leaf condition consumed by SGD but defined elsewhere).

The composition-without-implementation pattern (Q14) is the major part4-specific tagging policy: a §2 atom (e.g. `exp-back`) is a one-line gradient identity that fires only when the USER writes that identity. Once registered, downstream callers don't re-fire the atom — they just compose via the framework. This is the SAME policy as part2's BatchNorm exercise (the user writes the BN code; downstream usage of batchnorm in ResNet doesn't re-tag `batchnorm-running-stats`).

---

## Seed prediction accuracy

Predicted **~160 atom-tags across 21 exercises** (seed markdown), actual **136**. Seed over-predicted by ~15%.

Sources of over-prediction:
- **Exercise 19** predicted 11 atoms, actual 3 (8 fewer). All cuts are §2 composition FPs (exp/reshape/sum/getitem/log/multiply backs that the seed predicted to re-fire here).
- **Exercise 18** predicted 11 atoms, actual 9. Cut: reshape-back (MLP demo only); kept the rest.
- **Exercise 0** predicted 4, actual 3. Cut: grad-expressed-in-out (which fires in 0_4_11/exp instead — log's derivative is expressed in x, not out, and the markdown contrasts the two).
- **Exercise 4** predicted 4, actual 2. The §5 dispatcher atoms (wrap_forward_fn, unbox-args, box-array, kwargs-pass-through) all fire starting at 0_4_5 and culminate at 0_4_7, NOT at 0_4_4 which is just the lookup-table.
- **Exercise 16** predicted 5, actual 2. ReLU is just `maximum(x, 0.0)` — a 1-line composition; the wrap/register atoms don't fire because no new op is registered (relu is just a function, not a registered numpy primitive).

No exercise was significantly under-predicted; seed's per-exercise predictions were within 3 of actual for every exercise except 19 (+8 over) and 16 (+3 over).

Predicted top-atom ranking was accurate:
- `backward-fn-signature` predicted 11×, actual 10× (rank 1 confirmed)
- `register-back-fn-after-wrap` predicted 10×, actual 9× (rank 2 confirmed)
- `wrap-forward-fn-generic` predicted 8×, actual 8× (rank 3 exact)
- `tensor-wraps-ndarray` predicted 7×, actual 6× (rank 4 close)

---

## Atom-usage frequencies — the spine of part4

Top atoms (≥4 exercises):

| atom | count | domain | exercises |
|---|---|---|---|
| `backward-fn-signature` | 10 | backward-rule-pattern | 0,2,3,10,11,12,13,14,15,17 |
| `register-back-fn-after-wrap` | 9 | fwd-back-dispatcher | 4,10,11,12,13,14,15,17 |
| `wrap-forward-fn-generic` | 8 | fwd-back-dispatcher | 7,10,11,12,13,14,15,17 |
| `tensor-wraps-ndarray` | 6 | autograd-object | 5,6,7,9,18,19 |
| `chain-rule-elementwise` | 5 | backward-rule-pattern | 0,2,10,11,15 |
| `unbroadcast-pattern` | 4 | broadcast-in-backward | 1,2,14,15 |
| `broadcasting-rules` | 4 | tensor-algebra | 1,2,14,15 |
| `kwargs-pass-through-recipe` | 4 | fwd-back-dispatcher | 7,12,13,14 |
| `arg-position-back-functions` | 4 | backward-rule-pattern | 2,15,17 |
| `grad-tracking-global-toggle` | 4 | autograd-object | 5,6,7,20 |
| `recipe-dataclass` | 4 | autograd-object | 5,6,7,9 |
| `parents-dict-by-argidx` | 4 | autograd-object | 5,6,7,9 |

**Spine 1 — per-op backward signature (10×):** `backward-fn-signature` is the META atom of the whole part — co-fires with every per-op derivative exercise. Equivalent role to part3's `pseudocode-to-code-translate`.

**Spine 2 — wrap-and-register idiom (9× + 8×):** `register-back-fn-after-wrap` + `wrap-forward-fn-generic` co-fire perfectly in the §5 dispatcher cluster (4,7,10..17). This is the "extend the framework" 2-step idiom that runs at every new-op exercise. Tightest co-occurrence cluster in part4.

**Spine 3 — Recipe construction quartet (4× each, identical exercise sets):** `recipe-dataclass` + `parents-dict-by-argidx` co-fire perfectly in (5,6,7,9). `grad-tracking-global-toggle` fires in those plus 20 (SGD's NoGrad). The autograd-object cluster.

**Spine 4 — unbroadcast cohort:** `unbroadcast-pattern` + `broadcasting-rules` co-fire perfectly in (1,2,14,15). The broadcast-in-backward sub-spine — every binary backward depends on this.

**Singletons (1 exercise):** 34 of 63 atoms (54%) — higher than parts 1/2/3 (~35%). Structural reason: the §2 per-op-derivative cohort is 11 atoms × 1 eponymous exercise + 0-2 composers; each per-op atom is a 1-2-tag atom by design. The other singletons cluster in §6 capstone (5: dfs/cycle/backprop-loop/dispatch/back-fn-call) and §7 layers (5: linear/kaiming/parameter-wrap/module-base/sgd-vanilla).

The 54% singleton fraction is unusual but matches the seed prediction (45% predicted; 54% actual) — the part4 vocab structurally encodes one atom per learnable gradient identity, and most gradient identities show up in exactly one exercise.

---

## Domain breakdown (where the work actually is)

| domain | atom-tags | % | comment |
|---|---|---|---|
| fwd-back-dispatcher | 30 | 22% | the §5 spine — wrap_forward_fn + add_back_func + is_differentiable + kwargs + unbox/box |
| autograd-object | 24 | 18% | Tensor/Recipe/parents/grad-tracking/leaf — the data model |
| backward-rule-pattern | 20 | 15% | the §1 meta atoms (signature, chain rule, arg-position, grad-in-out) |
| backward-rules-per-op | 14 | 10% | the §2 atoms: log/multiply/negative/exp/reshape/permute/sum/max/matmul/add-sub-div/getitem |
| toposort-and-backprop | 13 | 10% | DFS + backprop loop body — §6 |
| custom-autograd-layers | 10 | 7% | §7 — Linear/Kaiming/Parameter-wrap/Module-base/CE/SGD/NoGrad |
| broadcast-in-backward | 8 | 6% | §3 — unbroadcast + duality + coerce-float |
| tensor-algebra | 5 | 4% | broadcasting-rules + matmul-2d cross-part reuses |
| misc-safety-nondiff | 5 | 4% | non-diff-fn-wrap, inplace-warning, rmul-dispatch |
| pytorch-module-mechanics | 3 | 2% | nn-module-subclass, nn-parameter-wrap, module-extra-repr (cross-part) |
| optimizer-state-plumbing | 3 | 2% | optimizer-init-params-list, zero-grad-set-none, inplace-param-update (cross-part from part3) |
| activation | 1 | 1% | relu-elementwise-max (cross-part) |

§5 dispatcher (22%) + §4 autograd-object (18%) + §1 backward-rule-pattern (15%) = 55% of all tags. Part4 is fundamentally a **"build your own autograd framework"** course, with the per-op gradient identities (§2, 10%) as the surface-level coding tasks.

---

## Delta Drills coverage

Across 136 atom-tags:

| DD code | tags | % | seed prediction (vocab-level) |
|---|---|---|---|
| DD-N (no drill coverage) | 129 | **94.9%** | 100% — close match |
| DD-Y (covered) | 6 | **4.4%** | 0% — slightly over-realized (broadcasting-rules + relu + matmul-2d reuse) |
| DD-? (partial / uncertain) | 1 | **0.7%** | 0% — minor |

**Headline: 95% of part4 atom-tags point to DD-N coverage, with only 5% in DD-Y/DD-?.** Slightly worse gap than part3 (78% DD-N) — predicted in the seed markdown.

The 6 DD-Y tags all come from cross-part reuses (`broadcasting-rules` 4×, `relu-elementwise-max` 1×, `matmul-2d` 1×). The part4-native vocab has zero DD-Y atoms — every backprop primitive (signature, chain-rule, unbroadcast, every per-op derivative, every dispatcher atom, every backprop-loop atom) is DD-N.

vs part1/part2/part3 calibrations:
- Part4's DD-Y share (4%) is the lowest of any part (part1: 25%, part2: 16%, part3: 0%, part4: 4%) — slightly higher than part3 only because of the 4 broadcasting-rules cross-part reuses
- Part4's DD-N share (95%) is the highest of any part (part1: 41%, part2: 50%, part3: 78%, part4: 95%)
- Per the user's note (REVIEW_QUEUE.md): "separate planned pipeline for math practice" makes the DD-N skew non-blocking. The §2 per-op derivative atoms (`log-back`, `exp-back`, `multiply-back`, `matmul-back-transpose-pair`, etc.) are exactly the format of planned math-practice drills.

The bright spot, predicted in seed markdown: every §2 atom is a 1-line gradient identity (e.g. `grad_out / x`, `grad_out @ y.T`, `out * grad_out`). The §2 cohort is the highest-leverage drill investment opportunity in chapter 0.

---

## Comparison to part1/part2/part3 calibrations

| Dimension | Part1 (n=10) | Part2 (n=24) | Part3 (n=12) | **Part4 (n=21)** | Read |
|---|---|---|---|---|---|
| Mean atoms/exercise | 7.90 | 4.88 | 13.17 | **6.48** | back in band — part4's per-op exercises pull mean down |
| Max single-exercise atom count | 14 (0_1_9) | 9 | 32 (0_3_11) | **14 (0_4_14 sum, 0_4_9 backprop)** | matches part1's max; capstone still in part3 |
| FP rate | 0.20 | 0.83 | 0.58 | **0.29** | matches part1 — seed predictions were accurate |
| New atoms surfaced | 0 | 4 (batch1 only) | 0 | **0** | vocabs at steady-state |
| DD-Y % of tags | 25% | 16% | 0% | **4%** | monotonic collapse with a small part4 bump from cross-part reuse |
| DD-N % of tags | 41% | 50% | 78% | **95%** | monotonic climb |
| Recurring atom % | 63% | 65% | 66% | **46%** | structurally lower — per-op atoms are by design 1-2 tag |
| Incidental tag % | 15% | n/a | 1.9% | **3.7%** | comparable to part3 |
| Singleton fraction | 37% | 35% | 34% | **54%** | structurally higher — see "atom-usage frequencies" |

The singleton fraction jump (34% → 54%) is the most interesting structural shift, mirroring part4's design: **one atom per gradient identity** means each per-op exercise's atom is, by definition, a 1-tag atom (only fires at its eponymous exercise). The §2 atoms (11) + §6 backprop-loop atoms (5) + §7 layer atoms (5) account for all 21 of the "extra" singletons vs part3.

---

## Recommendations

### 1. Accept the part4 vocab as-is — no merges, splits, renames, or drops.

The 54 NEW atoms from the seed-drafting pass all earned ≥1 tag during extraction. The pre-seeded 9 atoms (4 named in seed Q12 + 5 surprises) covered the cross-part reuse cleanly. Specifically:

- **Q1 (`chain-rule-elementwise` vs `grad-expressed-in-out`):** they truly don't co-fire — chain-rule fires in 5 elementwise exercises; grad-in-out fires only in 0_4_11 (exp). Keep separate — the distinction is real.
- **Q2 (§2 11-atom granularity):** all 11 atoms fired at their eponymous exercise; most as singletons or 2-tag (composer in 0_4_3 manual-chain). Keeping granular preserved the per-formula failure signal. Confirmed correct.
- **Q3 (`unbroadcast-pattern` vs `sum-back-expand-broadcast`):** they co-fire in 0_4_14 only; otherwise unbroadcast in 1/2/15, sum-back in 14 alone. Distinct firing patterns; keep separate.
- **Q4 (`tensor-wraps-ndarray` vs `recipe-dataclass`):** confirmed co-fire in 5,6,7,9 — exact overlap. Could merge with no signal loss, but the Recipe-vs-Tensor distinction is independently bug-bearing. Keep separate.
- **Q5 (`box-array-to-tensor-with-recipe` + `requires-grad-propagation` + `parents-dict-by-argidx`):** all 3 co-fire in 5,6,7. Could merge into one `forward-fn-output-construction` composite, but each is independently a failure mode. Keep granular.
- **Q6 (§6 toposort+backprop 10-atom granularity):** all 6 backprop-loop-body atoms (`backprop-pop-outgrad-loop`, `dispatch-back-fn-from-recipe`, `back-fn-call-with-recipe-args`, `grads-dict-accumulate-parents`, `leaf-tensor-condition`, `grad-accumulate-on-leaf`) fire in 0_4_9 alone. Confirmed singleton-cluster; granular split mirrors the 6 help-dropdowns in 0_4_9.
- **Q7 (`add-sub-div-back-lambdas` bundling):** fired once in 0_4_14 as a unit (6 lambdas in one cell). Bundling correctly captures the "bulk-register" idiom.
- **Q8 (`getitem-back-add-at` promotion):** fired in 0_4_14 only; did NOT co-fire with cross-entropy in 0_4_19 (filed as FP). Keep under §2 as planned.
- **Q9 (`logsumexp-cross-entropy` vs `arange-fancy-index-cross-entropy`):** both fired in 0_4_19 only, as the two halves of the one-line solution. Could merge but cross-chapter reuse argument holds (log-sum-exp recurs in chapter 1 attention).
- **Q10 (§7 trainable-layers 5-atom cluster):** all 5 fired in 0_4_18 + 1 reused in 0_4_20. Could collapse but each is independently a failure mode (Parameter wrap, kaiming, etc.). Keep granular.
- **Q11 (cuts):** confirmed; none of the cut atoms snuck back in (no `np-add-at-scatter`, no `directional-derivative-end-grad`, no `mnist-mlp-architecture`, no `numerical-stability-logsumexp-subtract-max`).
- **Q12 (cross-part reuses):** all 4 fired as predicted. Plus 5 bonus cross-part reuses fired: `module-extra-repr` (1×), `matmul-2d` (1×), `optimizer-init-params-list` / `zero-grad-set-none` / `inplace-param-update` (1× each in 0_4_20 SGD).
- **Q13 (`backward-fn-signature` as META atom):** fired 10× (rank 1). Confirmed as the load-bearing META atom of the part; do NOT mark `drillable: false` — the signature convention IS the load-bearing convention.

### 2. Edge candidates for the prereq DAG.

Based on co-occurrence and exercise order:

- The §2 11-atom cohort → strict topological order driven by exercise sequence: `log-back` → `multiply-back` → `negative-back` → `exp-back` → `reshape-back` → `permute-back-argsort` → `sum-back-expand-broadcast` → `max-back-tied-half` → `matmul-back-transpose-pair`. Each unlocks the next.
- `unbroadcast-pattern` → `multiply-back`, `max-back-tied-half`, `add-sub-div-back-lambdas` (the unbroadcast-dependency triangle)
- `backward-func-lookup` (0_4_4) → `wrap-forward-fn-generic` (0_4_7) → `register-back-fn-after-wrap` → every per-op exercise from 0_4_10 onwards (the framework-extension chain)
- `tensor-wraps-ndarray` + `recipe-dataclass` + `parents-dict-by-argidx` → all 3 prereqs of `box-array-to-tensor-with-recipe`
- `dfs-three-set-toposort` + `cycle-detection-temp-set` + `get-children-callable-param` (0_4_8) → `sorted-computational-graph` → `backprop-pop-outgrad-loop` (0_4_9) — the toposort-to-backprop chain, perfectly linear across 2 exercises
- `linear-affine-on-custom-tensor` + `kaiming-uniform-sf-init` + `parameter-wrap-around-tensor` → `nn-module-subclass` (R9.3) prereq — the custom Linear stack
- `sgd-vanilla-from-scratch` → prereqs `parameter-subclass-of-tensor` + `no-grad-context-mgr-update` + `grad-accumulate-on-leaf` + `inplace-param-update` (the SGD-on-custom-autograd quintet)

### 3. NEW candidate atom to track (not promoting yet).

None. The seed pass added the right atoms.

### 4. Stop tagging here; next blockers are decisions, not data.

- Decide policy on §2 atoms re-firing at composer exercises (Q14 below). Current policy: §2 atoms fire at REGISTRATION only.
- Build the planned separate math-practice pipeline to cover the 11 §2 per-op derivative atoms + the unbroadcast/sum-back/matmul-back identities — these are 1-line tensor identities that exactly match the planned drill format.
- The §5 dispatcher cluster (wrap_forward_fn + add_back_func + BackwardFuncLookup) is the densest framework-construction cluster in chapter 0 — needs a "fill-in-the-framework" drill format that the current array-op drills can't cover.

---

## "All defaults" review check

**The user's all-defaults lock held up.** No vocab decisions emerged from tagging that would have flagged for human review:

- All 13 Q1-Q13 questions from the vocab-gen REVIEW_QUEUE held up under tagging
- 5 of 6 false positives are the same systematic pattern (Q14, logged) — composition-without-implementation
- The 6th FP (`leaf-tensor-condition` in 0_4_20) is a one-off; doesn't merit vocab change
- All 54 NEW part4 atoms earned ≥1 tag
- All 9 cross-part reuses (4 predicted + 5 bonus) fired as expected

---

## Validator status

`python3 concept-graph/scripts/validate.py` exits with code 1 — but **all part4 atoms resolve cleanly**: every atom referenced in the 21 new `0_4_*.json` files maps to a known vocab entry. The validator's 4 PROPOSED NEW ATOMS warnings are all from pre-existing part2 files (`0_2_1.json, 0_2_4.json, 0_2_5.json, 0_2_18.json, 0_2_19.json, 0_2_22.json`) referencing `conv-kernel-shape`, `functional-module-wrap`, `kaiming-uniform-init`, `register-buffer` — these are part2-pipeline artifacts not introduced by this pass (identical to part3 calibration's note).

---

## Questions logged to REVIEW_QUEUE.md during tagging

One new question logged (Q14): policy on §2 per-op atoms re-firing at composer exercises. Pattern: 5 of 6 part4 FPs were §2 atoms predicted to fire at downstream callers (0_4_19 cross_entropy). Decision taken: §2 atoms tag at REGISTRATION exercise only. Logged as default-resolved.

---

## What changed from part1/part2/part3 calibrations

- **Mean atoms/exercise:** 7.90 → 4.88 → 13.17 → **6.48** (non-monotonic — part4's per-op exercises drag mean back down)
- **Max single-exercise count:** 14 → 9 → 32 → **14** (back to part1 level; capstone still in part3)
- **FP rate:** 0.20 → 0.83 → 0.58 → **0.29** (back to part1 level — seed predictions were tight)
- **New atoms surfaced post-seed:** 0 → 4 → 0 → **0** (vocab-design has converged for chapter 0)
- **DD-Y share of tags:** 25% → 16% → 0% → **4%** (small bump from broadcasting-rules cross-part reuse)
- **DD-N share of tags:** 41% → 50% → 78% → **95%** (monotonic climb; part4 is purest "autograd framework" content)
- **Singleton fraction:** 37% → 35% → 34% → **54%** (jump — structural, per-op vocab design)

Part4 is the **"build your own autograd framework"** half of chapter 0 — four distinct sub-cohorts (per-op derivatives, autograd object, dispatcher, toposort/loop), with the unifying conceptual bridge being the `(grad_out, out, x, ...) -> grad_in` backward fn signature. Combined with part3 (protocols and procedures), they fill the algorithmic/framework-construction half of fundamentals that the numpy/einops drills can't cover.
