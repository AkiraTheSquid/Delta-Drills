# Vocab Calibration Report — Part5 VAEs & GANs (all 12 exercises)

**Date:** 2026-05-18
**Pass:** part5-pilot (full tagging of `chapter0_fundamentals/part5_vaes_and_gans/0_5_0` … `0_5_11`)
**Total tagged:** 12/12 exercises in `part5_vaes_and_gans`
**Vocab size:** 210 atoms in `vocab/atoms.json` (36 part2 + 25 part1 + 62 part3 + 54 part4 + 33 part5)

---

## Headline stats

| Metric | Target | Part5 (n=12) | Part4 (n=21) | Part3 (n=12) | Part1 (n=10) | Verdict |
|---|---|---|---|---|---|---|
| Atoms per exercise (mean) | 4-7 | **8.83** | 6.48 | 13.17 | 7.90 | above band — VAE/GAN exercises are architecture+training composites |
| Atoms per exercise (range) | n/a | 3-18 | 2-14 | 4-32 | 3-14 | 0_5_7 (GAN training) is densest |
| Total atom tags | n/a | **106** | 136 | 158 | 79 | |
| False positives total | n/a (lower better) | **4** | 6 | 7 | 2 | 0.33 per exercise — matches part4 (0.29) |
| Proposed new atoms | decaying | **0** | 0 | 0 | 0 | full vocab steady-state |
| Recurring atom fraction (≥2 of 12) | n/a | 30 of 55 = **55%** | 46% | 66% | 63% | rebound — cross-part reuses fire across multiple part5 exercises |

**Per-exercise atom counts:** 0:6, 1:13, 2:9, 3:16, 4:3, 5:13, 6:3, 7:18, 8:4, 9:8, 10:8, 11:5
**Role breakdown:** 90 core, 16 incidental (15% incidental — back to part1 levels; cross-part reuses dominate the incidental count).
**Unique atoms used:** 55 of 210 vocab entries (26%) — part5 sub-vocab is well-scoped.

The mean of 8.83 is ABOVE the 4-7 target band — driven by training-loop exercises 1, 3, 7 (13/16/18 atoms each, each bundling architecture+loss+training-cycle+wandb cluster) and the densely-composed GAN architecture exercise 5 (13 atoms). The 4 transposed-conv exercises (8-11) sit at 4-8 atoms each, in band.

- **3-5 atoms** (0_5_4 modules, 0_5_6 init, 0_5_8 convT-minimal, 0_5_11 convT-module): single-concept exercises
- **6-9 atoms** (0_5_0 AE-arch, 0_5_2 VAE-arch, 0_5_9 convT-1d-full, 0_5_10 convT-2d): architecture-only or single-loop mechanics
- **13-18 atoms** (0_5_1 AE-train, 0_5_3 VAE-train, 0_5_5 GAN-arch, 0_5_7 GAN-train): full training-loop exercises bundling 3-4 cohorts at once

---

## Proposed new atoms

**Zero new atoms surfaced from part5 tagging.** All 12 exercises fit cleanly into the existing 33-atom part5 sub-vocab plus 23 cross-part reuses.

Of the 33 NEW part5 atoms added during seed drafting:
- **32 earned ≥1 tag** during extraction (validating their inclusion)
- **1 went unused:** `model-train-eval-toggle-around-sample` — seed predicted firing in 0_5_7 GAN training, but the actual log_samples doesn't toggle .eval()/.train() explicitly (the @t.inference_mode() decorator handles the no-grad semantics). Filed as FP in 0_5_7.json. The atom is now a 0-tag singleton — candidate for cut in a future vocab-trim pass, but kept as it captures a real PyTorch pattern that may fire in chapter 1/2.

Most-used part5-native atoms:

| atom | tags | exercises |
|---|---|---|
| `convT-kernel-axis-swap` | 4 | 8, 9, 10, 11 |
| `rearrange-as-sequential-layer` | 3 | 0, 2, 5 |
| `no-relu-on-final-layer` | 3 | 0, 2, 5 |
| `log-samples-eval-callback` | 3 | 1, 3, 7 |
| `encoder-decoder-symmetric` | 3 | 0, 2, 5 |
| `convT-as-flipped-padded-conv` | 3 | 8, 9, 10 |
| `holdout-data-one-per-class` | 2 | 1, 3 |
| `dcgan-wrapper-netG-netD` | 2 | 5, 7 |
| `padding-amount-formula-convT` | 2 | 9, 10 |
| `fractional-stride-zero-insertion` | 2 | 9, 10 |
| `randn-like-noise-source` | 2 | 2, 7 |
| `mse-reconstruction-loss` | 2 | 1, 3 |
| `bottleneck-latent-projection` | 2 | 0, 2 |

Cross-part reuses fired (23 atoms):
- **Explicit seed-listed reuses (8 of 8 fired as predicted):** `conv-padding-zero` (3×), `conv-windowing-1d` (2×), `conv-windowing-2d` (1×), `tensor-zeros-init` (2×), `slice-view-mutation` (2×), `training-step-cycle` (3×), `dataclass-training-args` (3×), `wandb-log-step` (3×)
- **Incidental reuses (15 fired):** `nn-module-subclass` (5×), `module-composition` (3×), `nn-parameter-wrap` (1×), `module-extra-repr` (2×), `conv-output-shape` (1×), `batchnorm-affine-params` (1×), `relu-elementwise-max` (1×), `tqdm-postfix-metrics` (3×), `wandb-init-run` (3×), `wandb-watch-model` (2×), `wandb-finish` (3×), `dataloader-batching` (3×), `loss-item-scalar-extract` (1×), `inference-mode-step` (3×), `tensor-to-device` (3×)

All 13 incidental atoms predicted in the seed markdown fired; plus 2 extras (`tensor-to-device`, `module-composition`) that fired but weren't in the seed's incidental list.

The seed-drafted vocab was correctly sized for part5 — no further additions needed.

---

## False-positive analysis

**4 false positives total** (0.33 per exercise — comparable to part4's 0.29, far below part2's 0.83). All 4 FPs concentrate in the same systematic pattern (composition-without-implementation, identical to part4's Q14):

| FP | Atom | Why removed |
|---|---|---|
| 0_5_3 | `mu-logsigma-encoder-head` | Seed predicted firing here, but VAE is constructed in 0_5_2; this exercise only calls self.model(img) which returns (img_reconstructed, mu, logsigma). Head implementation lives in 0_5_2. |
| 0_5_3 | `reparameterization-trick` | Same — the `z = mu + sigma * t.randn_like(sigma)` line is inside VAE.sample_latent_vector defined in 0_5_2. 0_5_3 only consumes (mu, logsigma) outputs to compute KL. |
| 0_5_3 | `randn-like-noise-source` | The randn_like call lives in VAE.sample_latent_vector (0_5_2). 0_5_3 has no random sampling. |
| 0_5_7 | `model-train-eval-toggle-around-sample` | Seed predicted netG.eval()/.train() around log_samples for GANs, but the solution doesn't toggle — model is constructed with .train() in __init__ and @t.inference_mode() handles sampling semantics without explicit eval(). |

**Two patterns:**
1. **Composition-without-implementation (3 of 4 FPs, all in 0_5_3):** when an exercise INVOKES a module whose internals were registered upstream, the architecture atoms of that module should NOT re-fire. Same policy as part4 Q14: tag at the IMPLEMENTATION exercise only. The VAE training loop (0_5_3) doesn't re-tag mu-logsigma/reparam/randn-like — those atoms tag at 0_5_2 only.
2. **Behavioral atom not actually exercised (1 FP in 0_5_7):** seed over-predicted by analogy from the general "wrap-sample-in-eval-mode" PyTorch idiom. The actual solution sidesteps this via @t.inference_mode().

---

## Seed prediction accuracy

Predicted **~100 atom-tags across 12 exercises** (seed markdown), actual **106**. Seed under-predicted by ~6%.

Sources of mis-prediction:
- **Exercise 1** predicted 7, actual 13 (+6). Seed missed the wandb cluster (init/watch/finish) and the incidental tqdm/inference-mode/dataloader/tensor-to-device atoms that fire in the training loop. Pattern: training-loop exercises always pull in the full wandb+tqdm cluster.
- **Exercise 3** predicted 11, actual 16 (+5). Same wandb cluster + tqdm + inference-mode + loss-item-extract; plus 3 §2 atom FPs (mu-logsigma, reparam, randn-like) cut.
- **Exercise 7** predicted 13, actual 18 (+5). Same wandb cluster + tqdm + inference-mode; plus 1 FP cut (model-train-eval-toggle).
- **Exercise 0** predicted 6, actual 6 (exact).
- **Exercise 2** predicted 9, actual 9 (exact).
- **Exercise 5** predicted 13, actual 13 (exact).

Pattern: seed under-predicted the 3 training-loop exercises by exactly the size of the wandb+tqdm+inference-mode cluster (~5 atoms each). Logged as Q19. The architecture-only and convT-mechanics exercises were predicted exactly.

Predicted top-atom ranking was accurate:
- `convT-kernel-axis-swap` predicted 4×, actual 4× (exact)
- `convT-as-flipped-padded-conv` predicted 3×, actual 3× (exact)
- `encoder-decoder-symmetric` predicted 3×, actual 3× (exact)
- `rearrange-as-sequential-layer` predicted 3×, actual 3× (exact)
- `randn-like-noise-source` predicted 3×, actual 2× (off by one — VAE training FP)

---

## Atom-usage frequencies — the spine of part5

Top atoms (≥3 exercises):

| atom | count | domain | exercises |
|---|---|---|---|
| `nn-module-subclass` | 5 | pytorch-module-mechanics | 0,2,5,11 (+0_5_4 trio) |
| `convT-kernel-axis-swap` | 4 | transposed-conv-mechanics | 8,9,10,11 |
| `rearrange-as-sequential-layer` | 3 | autoencoder-architecture | 0,2,5 |
| `no-relu-on-final-layer` | 3 | autoencoder-architecture | 0,2,5 |
| `encoder-decoder-symmetric` | 3 | autoencoder-architecture | 0,2,5 |
| `convT-as-flipped-padded-conv` | 3 | transposed-conv-mechanics | 8,9,10 |
| `log-samples-eval-callback` | 3 | generative-training-scaffolding | 1,3,7 |
| `training-step-cycle` (R) | 3 | training-loop-spine | 1,3,7 |
| `dataclass-training-args` (R) | 3 | training-loop-spine | 1,3,7 |
| `wandb-log-step` (R) | 3 | wandb-cluster | 1,3,7 |
| `wandb-init-run` (R) | 3 | wandb-cluster | 1,3,7 |
| `wandb-finish` (R) | 3 | wandb-cluster | 1,3,7 |
| `tqdm-postfix-metrics` (R) | 3 | wandb-cluster | 1,3,7 |
| `inference-mode-step` (R) | 3 | training-loop-spine | 1,3,7 |
| `dataloader-batching` (R) | 3 | training-loop-spine | 1,3,7 |
| `tensor-to-device` (R) | 3 | training-loop-spine | 1,3,7 |
| `conv-padding-zero` (R) | 3 | convolution-primitives | 8,9,10 |
| `module-composition` (R) | 3 | pytorch-module-mechanics | 0,2,5 |

**Spine 1 — AE→VAE→GAN architecture bridge (3× each for 3 atoms):** `encoder-decoder-symmetric` + `rearrange-as-sequential-layer` + `no-relu-on-final-layer` co-fire perfectly in (0, 2, 5). This is the cross-architecture META cluster — predicted exactly. Confirms `encoder-decoder-symmetric` as the part5 equivalent of part4's `backward-fn-signature`.

**Spine 2 — transposed-conv mechanics (3-4× each):** `convT-kernel-axis-swap` (4×, all 4 convT exercises) + `convT-as-flipped-padded-conv` (3×, 8-10) + `conv-padding-zero` (3×, 8-10). The convT cohort is structurally clean — every exercise pulls in the same primitives plus exercise-specific transformations.

**Spine 3 — training-loop trio (3× for 7 atoms):** `training-step-cycle` + `dataclass-training-args` + `wandb-init-run` + `wandb-log-step` + `wandb-finish` + `tqdm-postfix-metrics` + `inference-mode-step` + `dataloader-batching` + `tensor-to-device` ALL co-fire perfectly in (1, 3, 7). The "9 atoms that come for free with every training-loop exercise" cluster — the wandb+tqdm+training-step-cycle bundle.

**Spine 4 — GAN-training cohort (singletons in 0_5_7):** all 6 §4 atoms fire only in exercise 7 (BCE-log, generator-fool-D, detach, two-opt, noise-batch, clip-grad). Self-contained algorithm cohort, exactly as seed predicted. Equivalent role to part4's exercise 9 (backprop loop) and part3's exercise 11 (DDP-ResNet).

**Singletons (1 exercise):** 25 of 55 atoms (45%) — between part4 (54%) and part3 (34%). The §4 GAN-training cohort + §5 DCGAN-init cohort + §6 fractional-stride/padding-formula contribute most singletons.

---

## Domain breakdown (where the work actually is)

| domain | atom-tags | % | comment |
|---|---|---|---|
| training-loop-spine + wandb-cluster | 30 | 28% | the 9-atom trio firing in 1,3,7 |
| autoencoder-architecture (§1) | 11 | 10% | 4 atoms × 2-3 exercises each |
| transposed-conv-mechanics (§6) | 16 | 15% | convT-as-flipped-padded, axis-swap, fractional-stride, padding-formula, init |
| GAN-architecture (§3) | 9 | 8% | 6 atoms × 1-2 exercises (mostly singletons in 0_5_5, dcgan-wrapper bridges to 0_5_7) |
| GAN-training (§4) | 7 | 7% | all 6 atoms fire only in 0_5_7 (+ randn-like-noise-source reuse) |
| VAE-math (§2) | 5 | 5% | mu-logsigma + reparam + KL + ELBO + randn-like-noise |
| generative-training-scaffolding (§7) | 5 | 5% | log-samples + holdout-data + mse-recon (each 2-3×) |
| DCGAN-init (§5) | 3 | 3% | 3 atoms × 1 exercise (singleton cohort) |
| pytorch-module-mechanics (cross-part) | 11 | 10% | nn-module-subclass, module-composition, module-extra-repr, nn-parameter-wrap |
| convolution-primitives (cross-part) | 6 | 6% | conv-padding-zero, conv-windowing-1d, conv-windowing-2d |
| tensor-primitives (cross-part) | 4 | 4% | tensor-zeros-init, slice-view-mutation |
| activation/batchnorm (cross-part) | 2 | 2% | batchnorm-affine-params, relu-elementwise-max, conv-output-shape |

Training-loop+wandb (28%) + transposed-conv (15%) + pytorch-module-mechanics (10%) + AE-architecture (10%) = 63% of all tags. Part5 is **two intersecting halves**: (1) "build architectures and training loops for generative models" with heavy cross-part reuse from parts 2/3, and (2) "implement transposed convolution from scratch" — a clean 4-exercise §6 cohort tied to the AE/VAE/GAN architectures.

---

## Delta Drills coverage

Across 106 atom-tags:

| DD code | tags | % | seed prediction (vocab-level) |
|---|---|---|---|
| DD-N (no drill coverage) | 92 | **86.8%** | 87% — exact match |
| DD-Y (covered) | 6 | **5.7%** | 0% native, ~6% with reuses — match |
| DD-? (partial / uncertain) | 8 | **7.5%** | 13% — slightly lower |

**Headline: 87% of part5 atom-tags point to DD-N coverage, with 13% in DD-Y/DD-?.** Comparable to part4's 95% DD-N (slightly better) — cross-part reuses bring in some DD-Y coverage (tensor-zeros-init 2×, slice-view-mutation 2×, relu-elementwise-max 1×, batchnorm-affine-params 1×).

The 6 DD-Y tags all come from cross-part reuses (`tensor-zeros-init` 2×, `slice-view-mutation` 2×, `relu-elementwise-max` 1×, `batchnorm-affine-params` 1×). The 8 DD-? tags come from `randn-like-noise-source` (2×), `mse-reconstruction-loss` (2×), `clip-grad-norm-pre-step` (1×), `inference-mode-step` (3×) — 1-line idioms that could be drilled but no existing drill covers them.

The part5-native vocab has zero DD-Y atoms — every architecture/VAE-math/GAN-training/convT-mechanics atom is DD-N. As predicted in the seed markdown, the bright spot for the planned math-practice pipeline is the §2 VAE math atoms (reparam, KL, ELBO) + §4 GAN training atoms (BCE-log, generator-non-saturating) + §6 convT mechanics atoms (fractional-stride, padding-formula) — all are 1-line tensor identities.

vs part1/part2/part3/part4 calibrations:
- Part5's DD-Y share (6%) is comparable to part3 (0%) and part4 (4%)
- Part5's DD-N share (87%) is lower than part4 (95%) but higher than part3 (78%) — the cross-part conv-primitive + tensor-primitive reuses brought DD-Y up
- Per the user's note (REVIEW_QUEUE.md): "separate planned pipeline for math practice" makes the DD-N skew non-blocking

---

## Comparison to part1/part2/part3/part4 calibrations

| Dimension | Part1 (n=10) | Part2 (n=24) | Part3 (n=12) | Part4 (n=21) | **Part5 (n=12)** | Read |
|---|---|---|---|---|---|---|
| Mean atoms/exercise | 7.90 | 4.88 | 13.17 | 6.48 | **8.83** | second-highest — VAE/GAN architecture+training are composites |
| Max single-exercise atom count | 14 (0_1_9) | 9 | 32 (0_3_11) | 14 (0_4_14) | **18 (0_5_7)** | new chapter-0 third-place; capstone still in part3 |
| FP rate | 0.20 | 0.83 | 0.58 | 0.29 | **0.33** | matches part4 — seed predictions were tight |
| New atoms surfaced | 0 | 4 (batch1 only) | 0 | 0 | **0** | vocab-design has fully converged for chapter 0 |
| DD-Y % of tags | 25% | 16% | 0% | 4% | **6%** | part4 bump continues; cross-part reuse keeps the tail |
| DD-N % of tags | 41% | 50% | 78% | 95% | **87%** | slight drop from part4 — cross-part conv-primitive + tensor-primitive reuses help |
| Recurring atom % | 63% | 65% | 66% | 46% | **55%** | between part3/part4 — the training-loop trio creates 9 recurring atoms |
| Incidental tag % | 15% | n/a | 1.9% | 3.7% | **15%** | back to part1 level — cross-part reuses dominate incidentals |
| Singleton fraction | 37% | 35% | 34% | 54% | **45%** | between part3 and part4 — §4 GAN-training cohort + §5 DCGAN-init + §3 GAN-arch contribute most singletons |

The incidental-tag % jumping back to 15% is the most interesting shift — every part5 training-loop exercise pulls in 5-7 cross-part incidental atoms (the wandb+tqdm+dataloader+inference-mode+tensor-to-device cluster from parts 2/3). Part5 is the **most-cross-part-reusing chapter so far**: 23 of 55 unique atoms used (42%) are cross-part reuses, compared to part4's ~17% reuse share.

---

## Recommendations

### 1. Accept the part5 vocab as-is — no merges, splits, renames, or drops.

The 33 NEW atoms from the seed-drafting pass: 32 of 33 earned ≥1 tag. The one unused atom (`model-train-eval-toggle-around-sample`) is a real PyTorch pattern not exercised here but plausibly fires in chapter 1/2; keep with a 0-tag-part5 note. Specifically:

- **Q1 (`encoder-decoder-symmetric` as META):** confirmed to fire in 3 exercises across 3 architectures (AE, VAE, GAN-generator). Treated as normal atom; the symmetry IS load-bearing as predicted.
- **Q2 (`mu-logsigma-encoder-head` rearrange split):** confirmed single-atom is correct — the Rearrange("b (n d) -> n b d", n=2) trick is the canonical impl and the doubled-Linear+Rearrange is one failure mode.
- **Q3 (`reparameterization-trick` t.exp split):** confirmed single-atom; the t.exp(logsigma) is part of the one-liner, no separate failure mode.
- **Q4 (`kl-divergence-gaussian-closed-form` single identity):** confirmed; whole expression is one line.
- **Q5 (`bce-log-loss-real-fake` vs `generator-loss-fool-discriminator`):** both fired in 0_5_7 as predicted; the non-saturating distinction is independently learnable. Keep separate.
- **Q6 (`detach-stop-gradient-trick` META):** fired in 0_5_7 only. Keep as normal §4 atom — the singleton-cluster structure of §4 makes META promotion redundant.
- **Q7 (§3 GAN architecture 6-atom granularity):** all 6 fired in 0_5_5, plus `dcgan-wrapper-netG-netD` fired again in 0_5_7. Granular split preserved per-block failure modes. Keep 6.
- **Q8 (§5 DCGAN init 2-atom split):** both fired in 0_5_6; Conv/Linear init vs BN init are conceptually different and independently buggy. Keep separate.
- **Q9 (§6 6.1+6.2 collapse):** both fired in 0_5_8/9/10 (6.1: 3×, 6.2: 4× — 6.2 also fires in 0_5_11 as incidental). They are independent failure modes (spatial vs channel-axis). Keep separate.
- **Q10 (`convT-init-uniform-by-kernel` vs `kaiming-uniform-sf-init`):** convT-init fired only in 0_5_11; kaiming-uniform-sf-init did NOT fire here. The denominator distinction (out-channels*kH*kW vs in-features) holds. Keep separate.
- **Q11 (cross-part reuses):** all 8 explicit reuses fired as predicted (3 × 3-tag, 1 × 2-tag, 4 × 1-2-tag). 15 incidental reuses fired (vs 13 predicted) — the extras are tensor-to-device (3×) and module-composition (3×) which transferred from part2 into the training-loop exercises.
- **Q12 (cuts):** all 8 cut atoms stayed cut. No `bce-loss-call` (solution uses explicit -log() form), no `transposed-conv-windowed-einsum` (R8.2/R8.3 covered it), no `kaiming-uniform-init-on-convT` (DCGAN uses normal_(0, 0.02)), etc.
- **Q13 (cross-part audit forward look):** confirmed §6 transposed-conv atoms only fire in part5 (not in later chapters' exercises that don't exist yet in this repo). §4 GAN-training cohort is part5-unique. DCGAN-init scheme is part5-unique. To re-audit when chapter 1/2 lands.

### 2. Edge candidates for the prereq DAG.

Based on co-occurrence and exercise order:

- The §1 AE-architecture 3-atom cluster → strict topological order: 0_5_0 introduces all 4 atoms; 0_5_2 (VAE) reuses 3 of them + 3 §2 atoms; 0_5_5 (GAN) reuses 3 of them + 6 §3 atoms.
- The §2 VAE-math 5-atom cluster → strict order: mu-logsigma-encoder-head + reparam-trick + randn-like-noise-source all introduce in 0_5_2; kl-divergence-gaussian-closed-form + elbo-loss-sum-with-beta introduce in 0_5_3.
- The §3 GAN-architecture 6-atom cluster → all introduce in 0_5_5; dcgan-wrapper-netG-netD recurs in 0_5_7.
- The §4 GAN-training 6-atom cluster → all introduce in 0_5_7. Clean singleton-cluster.
- The §5 DCGAN-init 3-atom cluster → all introduce in 0_5_6. Clean singleton-cluster.
- The §6 transposed-conv 5-atom cluster → 0_5_8 introduces convT-as-flipped-padded-conv + convT-kernel-axis-swap; 0_5_9 adds fractional-stride-zero-insertion + padding-amount-formula-convT; 0_5_10 generalizes both to 2D; 0_5_11 adds convT-init-uniform-by-kernel.
- The §7 generative-scaffolding 4-atom cluster → mse-reconstruction-loss + log-samples-eval-callback + holdout-data-one-per-class introduce in 0_5_1; reuse in 0_5_3 + 0_5_7 (log-samples only). model-train-eval-toggle-around-sample did NOT fire — orphan node in DAG.

### 3. NEW candidate atom to track (not promoting yet).

None. The seed pass added the right atoms. The 1 unused atom (`model-train-eval-toggle-around-sample`) is a candidate for future cut OR retain-for-chapter-1.

### 4. Stop tagging here; next blockers are decisions, not data.

- Decide whether to cut `model-train-eval-toggle-around-sample` (0-tag in part5; may fire in chapters 1/2). Default: retain.
- Build the planned separate math-practice pipeline to cover the §2 VAE math + §4 GAN training + §6 convT mechanics atoms — these are 1-line tensor identities that exactly match the planned drill format.
- The §3 GAN-architecture 6-atom cohort (all in 0_5_5) is a candidate for an "architecture-modification drill format" (start with a Conv2d block, modify into a ConvTranspose2d block; start with a generator, swap in a discriminator block) that the current array-op drills can't cover.

---

## "All defaults" review check

**The user's all-defaults lock held up.** No vocab decisions emerged from tagging that would have flagged for human review:

- All 13 Q1-Q13 questions from the vocab-gen REVIEW_QUEUE held up under tagging
- 3 of 4 false positives are the same systematic pattern (Q14 from part4, re-fired here as Q19/Q20) — composition-without-implementation
- The 4th FP (`model-train-eval-toggle-around-sample` in 0_5_7) is a one-off; doesn't merit vocab change
- 32 of 33 NEW part5 atoms earned ≥1 tag
- All 8 explicit cross-part reuses fired as expected; 15 incidental reuses fired (vs 13 predicted)

---

## Validator status

`python3 concept-graph/scripts/validate.py` exits with code 1 — but **all part5 atoms resolve cleanly**: every atom referenced in the 12 new `0_5_*.json` files maps to a known vocab entry. The validator's 4 PROPOSED NEW ATOMS warnings are all from pre-existing part2 files (`0_2_1.json, 0_2_4.json, 0_2_5.json, 0_2_18.json, 0_2_19.json, 0_2_22.json`) referencing `conv-kernel-shape`, `functional-module-wrap`, `kaiming-uniform-init`, `register-buffer` — identical to part3 and part4 calibration notes; not introduced by this pass.

---

## Questions logged to REVIEW_QUEUE.md during tagging

Two new questions logged (Q19, Q20):

**Q19: Seed under-predicted training-loop exercises by ~5 atoms each** (vs part4's pattern of over-predicting). The wandb+tqdm+inference-mode+dataloader+tensor-to-device cluster transfers fully into every training-loop exercise; seed should bake this in as a default reusable cluster for any future training-loop ARENA exercises.

**Q20: 3 §2 VAE-math atoms re-firing predicted in 0_5_3 (training loop) but actually fire only at 0_5_2 (VAE construction)**. Same composition-without-implementation pattern as part4 Q14 — tag at IMPLEMENTATION exercise only. Confirms the policy across parts.

---

## What changed from part1/part2/part3/part4 calibrations

- **Mean atoms/exercise:** 7.90 → 4.88 → 13.17 → 6.48 → **8.83** (rebound — training-loop exercises bundle architecture+loss+training-cycle+wandb)
- **Max single-exercise count:** 14 → 9 → 32 → 14 → **18** (new chapter-0 third-place — 0_5_7 GAN training)
- **FP rate:** 0.20 → 0.83 → 0.58 → 0.29 → **0.33** (matches part4 level)
- **New atoms surfaced post-seed:** 0 → 4 → 0 → 0 → **0** (vocab-design has fully converged for chapter 0)
- **DD-Y share of tags:** 25% → 16% → 0% → 4% → **6%** (small bump from convT cross-part reuses)
- **DD-N share of tags:** 41% → 50% → 78% → 95% → **87%** (drop from part4 — cross-part conv/tensor primitives help)
- **Singleton fraction:** 37% → 35% → 34% → 54% → **45%** (between part3 and part4 — §4+§5+§3 contribute most singletons)
- **Cross-part reuse %:** n/a → n/a → ~12% → ~17% → **42%** of unique atoms — part5 is the most-cross-part-reusing chapter

Part5 is the **"generative-architecture-meets-transposed-conv"** half of chapter 0 — three distinct sub-cohorts (AE/VAE architecture, GAN architecture+training, transposed-conv mechanics), with the unifying conceptual bridge being `encoder-decoder-symmetric` (the AE↔VAE↔GAN-generator architectural skeleton). Combined with part4 (autograd framework) and part3 (training protocols), the three close out the algorithmic+composition+framework half of fundamentals. The §4 GAN-training cohort (6 singletons in 0_5_7) is structurally equivalent to part4's exercise 9 backprop and part3's exercise 11 DDP-ResNet — one exercise concentrates an entire algorithm.
