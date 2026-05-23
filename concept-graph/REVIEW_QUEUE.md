# Concept Graph — Review Queue

Questions accumulated during the pipeline that would normally have blocked for user review.
User said (2026-05-18): "just keep going... I want everything done by the time I get back, so don't ask any more questions that stop the pipeline for now."

Each entry: source (part + stage), question, default-taken, why-default-might-be-wrong.

---

## Part 3 / vocab-gen (62 atoms added — atoms.json 61→123)

Full draft: `/home/stellar-thread/Documents/coding-ideas/arena_part3_seed_atoms.md`

**Note on `*(reused)*` labels in markdown** — agent labeled 55 of 62 atoms as `*(reused)*` claiming they were pre-seeded into atoms.json. This is wrong; all 62 were added this run. When reviewing, treat the whole markdown as fresh.

### Q1. Optimizer-state cluster granularity (atoms 1.1-1.7)
- 7 atoms or merge to one composite `optimizer-step-plumbing`?
- **Default**: keep 7 granular. Each is independently the cause of "test fails for non-obvious reason"; they co-fire in every optimizer exercise but pedagogically distinct.

### Q2. `momentum-buffer-update` (2.3) vs `ema-first-moment` (2.4)
- Mathematically same update (`x ← αx + βg`), differs only in whether α+β=1.
- **Default**: keep separate. The RMSprop/Adam pedagogy hammers the distinction.

### Q3. `weight-decay-l2-add` (2.1) vs `weight-decay-decoupled` (2.2)
- Same op, different placement in update.
- **Default**: keep separate. AdamW exercise frames this as THE conceptual content.

### Q4. Parameter-group atoms (4.1-4.4)
- All 4 fire only in exercise 6. Collapse to one `param-groups`?
- **Default**: keep 4 granular. They're independent failure modes.

### Q5. `trainer-class-skeleton` (5.2) vs `training-step-cycle` (R5.1)
- Class architecture vs loop body. Co-fire but not redundant.
- **Default**: keep separate.

### Q6. WandB cluster (6.1-6.9, 9 atoms)
- Merge literal init+finish (6.1+6.4) into `wandb-run-lifecycle`?
- **Default**: keep 9 atoms.

### Q7. `broadcast-source-fanout` (7.3) vs `dist-send-recv-pair` (7.2)
- 7.3 implemented as pattern over 7.2. Kill 7.2 as too low-level?
- **Default**: keep both.

### Q8. Cross-part reuses R5.6, R5.7
- `nn-parameter-wrap` only in one toy class; `freeze-requires-grad` speculative (not in cells we have).
- **Default**: keep both reuses (low cost; refines if proven wrong).

### Q9. Cut atoms list — confirm cuts
- `pytorch-optim-sgd-call`, `pathological-curve-loss-fn`, `bivariate-gaussian-loss`, `rosenbrock-banana-loss`, `dataclasses-replace`, `torch-no-grad-vs-inference-mode`, `nccl-vs-gloo-backend`, `wandb-login-api-key`.
- **Default**: cuts stand.

### Q10. `trainer-subclass-extend` (5.3) — drillable: false?
- Conceptual scaffold, not coding focus. Part3's analog of part1's `ray-parametric-form`.
- **Default**: flag for `drillable: false` (user added separate math-practice pipeline, so this is fine to defer).

---

## Note on DD coverage going forward

User confirmed (2026-05-18) they have a **separate pipeline planned for math practice**. So DD-N percentages don't need to drive vocab-trimming decisions — just tag accurately and let the parallel pipeline cover what numpy/einops drills can't.

---

## Part 3 / tagging

Full calibration: `concept-graph/exercises/CALIBRATION_REPORT_PART3.md`.

12/12 exercises tagged. 158 atom-tags. 0 new atoms proposed. 7 seed false positives. Validator: all part3 atoms resolve cleanly (pre-existing part2 errors unrelated).

No questions blocked the pipeline during tagging — all decisions were either pre-resolved by Q1-Q10 in the vocab-gen queue or fell into default-resolvable patterns. Forward-looking observations below.

### Q11. `weight-decay-l2-add` and `conditional-hparam-branch` did NOT fire in 0_3_4 (AdamW)
- Seed predicted both for AdamW by analogy from Adam, but AdamW replaces L2 with decoupled WD and the solution drops the conditional shortcut (multiplies by 1 unconditionally).
- **Default**: filed as false positives in 0_3_4.json's `atoms_in_seed_but_not_actually_present`. No vocab change — Q3 already framed weight-decay-l2 vs decoupled as contrast atoms.

### Q12. `optimizer-repr-string` and `pseudocode-to-code-translate` did NOT fire in 0_3_6 (param-groups SGD rewrite)
- 0_3_6 is a structural refactor, not a from-scratch impl. No __repr__ in the template, no LaTeX pseudocode to translate.
- **Default**: filed as false positives. Pattern: "meta" atoms drop out of refactor exercises. Worth noting for any future refactor-style ARENA exercises.

### Q13. `freeze-requires-grad` and `dataloader-batching` did NOT fire in 0_3_7
- Seed flagged both as speculative — they live in the implied ResNetFinetuner parent class which is in solutions.py and not in the visible cells.
- **Default**: filed as false positives. Tagging policy stays "tag what's in the visible cells", not "tag what the markdown describes". Confirms Q8's conservative read.

### Q14. `trainer-subclass-extend` did NOT fire in 0_3_11
- Markdown explicitly says "We recommend not using inheritance for this" for DistResNetTrainer. The solution builds it as a sibling of WandbResNetFinetuner, not a subclass.
- **Default**: filed as false positive. The atom is now a 1-tag singleton (only fires in 0_3_7). Reinforces Q10's flag for `drillable: false` on the trainer-class-skeleton family.

### Q15. Seed under-predicted exercise 11 by 7 atoms
- Predicted 25, actual 32. Sources: wandb cluster (init/log/finish) + training-loop spine (validation-no-grad, train-eval-mode-branch, tqdm-postfix-metrics) + tensor-to-device — all treated as "inherited from 0_3_7" but the sibling class re-implements them.
- **Default**: tag what's in the visible cells. No vocab change. Lesson for next part: seed should not assume inheritance reduces tag count if the exercise explicitly avoids inheritance.

### Q16. Seed under-predicted exercise 10 by 3 atoms
- The run_simple_model driver block IS in 0_3_10's cells (verified). Seed treated it as belonging to 0_3_11. The 5 DDP-integration atoms (init-process-group, mp-spawn, per-rank-cuda, tensor-to-device, nn-parameter-wrap) plus 3 incidentals (all-reduce-grad-sync, training-step-cycle, backward-on-scalar-loss) fire here for the first time.
- **Default**: tagged the driver block as part of 0_3_10. Most driver-block atoms marked `incidental` because the exercise's coding focus is reduce/all_reduce.

### Q17. Cross-part reuse audit (forward look)
- All 4 cross-part reuses that fired in part3: `tensor-to-device` (5 tags), `nn-parameter-wrap` (1 tag), `training-step-cycle` (5 tags), `dataloader-batching` (1 tag). Plus 3 from the R5 sub-table not yet covered: `validation-no-grad` (2 tags), `train-eval-mode-branch` (2 tags), `cross-entropy-classification-loss` (2 tags) — all confirmed cross-part recurrences.
- **Default**: no vocab change. Reinforces that the part2 training-infra atoms transferred correctly to part3.

---

## Part 4 / vocab-gen (54 atoms added — atoms.json 123→177)

Full draft: `/home/stellar-thread/Documents/coding-ideas/arena_part4_seed_atoms.md`

Predicted **58 atoms used by part 4** (54 new + 4 reused from existing vocab: `broadcasting-rules`, `nn-parameter-wrap`, `nn-module-subclass`, `relu-elementwise-max`). Slightly above the 35-55 target band; the over-count is concentrated structurally in §2 (11 per-op derivative atoms) and §6 (10 backprop-loop atoms).

### Q1. `chain-rule-elementwise` (1.2) vs `grad-expressed-in-out` (1.3)
- Both are "meta" atoms about the per-op cohort. Merge into `elementwise-grad-rule` with `variant: x | out`?
- **Default**: keep separate. The choice of expressing grad in `out` vs `x` is independently learnable (governs recompute cost).

### Q2. §2 per-op-derivative cohort granularity (11 atoms: log/multiply/negative/exp/reshape/permute/sum/max/matmul/add-sub-div/getitem)
- All co-fire with their eponymous exercise (mostly 1-tag singletons) plus 1-2 composer exercises.
- Could collapse to a single `per-op-backward` composite, losing per-formula failure signal.
- **Default**: keep 11 granular. The split mirrors the per-exercise structure 1:1 and is the exact format of the planned math-practice drills.

### Q3. `unbroadcast-pattern` (3.1) vs `sum-back-expand-broadcast` (2.7)
- 2.7 is `unbroadcast` applied to the `sum` case (plus `expand_dims` for `keepdim=False`). 3.1 is the general pattern.
- Atom 3.2 (`sum-and-broadcast-duality`) names the relationship.
- **Default**: keep separate. The `expand_dims`-then-`broadcast_to` shape of `sum_back` is its own pedagogically-named idiom.

### Q4. `tensor-wraps-ndarray` (4.1) vs `recipe-dataclass` (4.2)
- Co-fire in every Tensor exercise. Merge into `tensor-recipe-object`?
- **Default**: keep separate. Recipe is independently the source of "why does my backward fn see the wrong args" bugs.

### Q5. `box-array-to-tensor-with-recipe` (5.6) vs `requires-grad-propagation` (4.3) vs `parents-dict-by-argidx` (4.5)
- Three atoms that co-fire perfectly in the `wrap_forward_fn` body.
- **Default**: keep granular. Each is independently a failure mode (forgetting `is_differentiable` gate, building wrong parent dict, etc.).

### Q6. §6 toposort+backprop atom granularity (10 atoms)
- Exercise 8: 3 atoms. Exercise 9: ~6 atoms (the loop body + dispatch + accumulate).
- Could collapse `backprop-pop-outgrad-loop` + `dispatch-back-fn-from-recipe` + `back-fn-call-with-recipe-args` + `grads-dict-accumulate-parents` into one `backprop-loop-body` composite.
- **Default**: keep 10 granular. The exercise 9 help-dropdowns each address one of these failure modes — independently named in ARENA pedagogy.

### Q7. `add-sub-div-back-lambdas` (2.10) — split into 3 atoms?
- Bundled because they're registered together in the same cell and tested as a unit; but `true_divide`'s `-x/y^2` derivative is conceptually distinct.
- **Default**: keep bundled. The "bulk-register lambdas with unbroadcast" idiom is itself the atom.

### Q8. `getitem-back-add-at` (2.11) — promote to its own subsection?
- Indexing is mathematically distinct from elementwise ops (scatter vs pointwise); `np.add.at` is a non-obvious primitive.
- **Default**: keep under §2; promote IF tagging shows it co-fires with cross-entropy (2.11 + 7.7 may be tightly coupled).

### Q9. `logsumexp-cross-entropy` (7.6) vs `arange-fancy-index-cross-entropy` (7.7)
- Co-fire only in exercise 19; compose the same one-line solution.
- **Default**: keep separate. The log-sum-exp trick recurs in chapter 1 attention; fancy-index recurs in any classification head.

### Q10. §7 trainable-layers atoms (7.1-7.5) — collapse?
- Five atoms about "build Linear on custom autograd".
- **Default**: keep 5 granular. Each is independently a failure mode (forgetting `Parameter` wrap, wrong fan-in scale, wrong transpose, etc.).

### Q11. Cut atoms — confirm cuts
- `np-add-at-scatter` (absorbed by 2.11)
- `directional-derivative-end-grad` (captured by 6.9 `end-grad-default-ones-like`)
- `module-extra-repr` (already in vocab from part2; would be singleton here)
- `mnist-mlp-architecture` (the MLP class is just composed Linears + ReLUs; not a part4-specific skill)
- `mnist-training-loop` (not exercised — markdown only)
- `numerical-stability-logsumexp-subtract-max` (mentioned as bonus only in 0_4_19; not load-bearing)
- `is-leaf-flag` (subsumed by 4.6 `leaf-tensor-condition`)
- **Default**: cuts stand.

### Q12. Cross-part reuses — confirm 4 atoms
- `broadcasting-rules` (heavy reuse: unbroadcast, multiply_back, sum_back, max_back)
- `nn-parameter-wrap` (custom Parameter wraps the custom Tensor; same semantic role)
- `nn-module-subclass` (custom Module mirrors torch.nn.Module pattern)
- `relu-elementwise-max` (relu defined as `maximum(x, 0.0)` over custom Tensor)
- **Default**: reuse all 4. No new "custom-tensor-flavored" duplicates needed.

### Q13. `backward-fn-signature` (1.1) is the under-named meta atom
- Fires in 11 of 21 exercises — equivalent role to part3's `pseudocode-to-code-translate`.
- **Default**: keep as a META atom (not `drillable: false`; the signature `(grad_out, out, x, ...) -> grad_in` IS the load-bearing convention).

---

## Part 4 / tagging

Full calibration: `concept-graph/exercises/CALIBRATION_REPORT_PART4.md`.

21/21 exercises tagged. 136 atom-tags. 0 new atoms proposed. 6 seed false positives. Validator: all part4 atoms resolve cleanly (pre-existing part2 errors unrelated, same as part3 calibration).

No questions blocked the pipeline during tagging — all decisions were either pre-resolved by Q1-Q13 in the vocab-gen queue, or fell into default-resolvable patterns. One new question (Q14) logged with default-taken.

### Q14. §2 per-op atoms re-firing at composer exercises (5 of 6 part4 FPs)
- 0_4_19 (cross_entropy) composes logprobs = logits - logits.exp().sum(-1, keepdim=True).log() and -logprobs[arange(0, B), true_labels]. Seed predicted that `exp-back`, `reshape-back`, `sum-back-expand-broadcast`, `getitem-back-add-at` would all re-fire here.
- These atoms were REGISTERED at their eponymous exercises (0_4_11, 0_4_12, 0_4_14, 0_4_14). 0_4_19's code does not implement any backward fn — it composes via the autograd framework.
- **Default**: §2 atoms tag at REGISTRATION exercise only. Filed all 4 as false positives in 0_4_19.json's `atoms_in_seed_but_not_actually_present` with the same reason. Same policy as part2/3: tag what's in the cells, not what propagates through the autograd graph.
- Why default might be wrong: if user wants the SR system to surface "you struggled with the gradient flow through exp() during cross_entropy", then re-firing at composers would be the right call. But that's a CHAIN-OF-CONCEPTS view, distinct from this vocab's per-atom granularity.

### Q15. `chain-rule-elementwise` and `grad-expressed-in-out` cleanly separated
- chain-rule fires in 5 exercises (0,2,10,11,15) — every elementwise per-op exercise
- grad-in-out fires in 0_4_11 (exp) only — confirmed as a clean contrast with 0_4_0 (log uses x, not out)
- **Default**: Q1 vocab decision validated. Keep separate.

### Q16. `register-back-fn-after-wrap` fires in 0_4_4 (BackwardFuncLookup intro)
- 0_4_4 is the lookup-table exercise; the test cell registers 3 funcs via `BACK_FUNCS.add_back_func(np.log, 0, log_back)` etc.
- Counted this as the introduction; subsequent firings (0_4_10..17) are USE rather than INTRO.
- **Default**: tag in 0_4_4 + all subsequent register cells (8 tags total). The atom's signature is the (forward_fn, argnum, back_fn) call — present in 0_4_4.

### Q17. Cross-part reuse audit (forward look)
- All 4 predicted cross-part reuses (`broadcasting-rules`, `nn-parameter-wrap`, `nn-module-subclass`, `relu-elementwise-max`) fired exactly where expected.
- 5 bonus cross-part reuses fired: `module-extra-repr` (1×, 0_4_18), `matmul-2d` (1×, 0_4_17), `optimizer-init-params-list` / `zero-grad-set-none` / `inplace-param-update` (1× each, 0_4_20 SGD).
- `freeze-requires-grad` did NOT fire (correctly — not part of part4 exercises).
- **Default**: no vocab change. Reinforces that part2/part3 reuses transferred correctly into part4.

### Q18. Seed over-predicted by ~15% (160 → 136 actual)
- Mostly concentrated in 0_4_19 (composition-without-implementation, see Q14) and 0_4_16 (relu is 1-line composition; doesn't register a new numpy primitive so wrap/register atoms don't fire).
- **Default**: tag what's in the visible cells. Pattern: composition exercises (especially 1-line wrappers like relu and the cross-entropy log-sum-exp expression) under-fire compared to from-scratch-implementation exercises.

---

## Part 5 / vocab-gen (33 atoms added — atoms.json 177→210)

Full draft: `/home/stellar-thread/Documents/coding-ideas/arena_part5_seed_atoms.md`

Predicted **41 atoms used by part 5** (33 new + 8 reused from existing vocab: `pad1d/pad2d` via `conv-padding-zero`, `conv-windowing-1d`, `conv-windowing-2d`, `tensor-zeros-init`, `slice-view-mutation`, `training-step-cycle`, `dataclass-training-args`, `wandb-log-step`). Slightly above the 25-40 target band; over-count concentrated in §3 (6 GAN architecture atoms) and §4 (6 GAN training atoms — the full alternating-D/G algorithm).

### Q1. `encoder-decoder-symmetric` (1.1) — promote to META atom?
- Fires in 3 exercises across 3 architectures (AE, VAE, GAN-generator). Analogous to part4's `backward-fn-signature`.
- **Default**: keep as a normal atom. The symmetry pattern IS load-bearing (forgetting to mirror dilates the bug surface); not just a conceptual unifier.

### Q2. `mu-logsigma-encoder-head` (2.1) — split out the `Rearrange("b (n d) -> n b d")` trick?
- "Double the Linear output then rearrange into 2-stack" is one specific implementation; could promote to `paired-output-via-doubled-linear`.
- **Default**: keep merged. The rearrange is the canonical impl; atom name already implies the mechanism.

### Q3. `reparameterization-trick` (2.2) — split out `t.exp(logsigma)` recovery?
- Math is `z = mu + sigma * eps`; impl is `z = mu + t.exp(logsigma) * t.randn_like(...)`. The exp-of-logsigma step is conceptually distinct (where the log parametrization "pays off" in stability).
- **Default**: keep merged. The atom IS the one-liner.

### Q4. `kl-divergence-gaussian-closed-form` (2.3) — keep as a single identity?
- Could split into `gaussian-prior-kl-formula` + `mean-reduction-over-batch` + `beta-weighting`.
- **Default**: keep bundled. The whole expression is exactly one line; splitting would lose the "this is the canonical Gaussian-vs-standard-normal KL" semantics.

### Q5. `bce-log-loss-real-fake` (4.1) vs `generator-loss-fool-discriminator` (4.2) — merge?
- Two halves of the same minimax game.
- **Default**: keep separate. The generator's loss has the non-obvious "non-saturating" choice (use `-log(D(G))` not `-log(1-D(G))`); that choice is independently learnable.

### Q6. `detach-stop-gradient-trick` (4.3) — promote to a top-level §4 META?
- THE most-named GAN-training idiom.
- **Default**: leave it as a normal §4 atom. Promotion adds no extraction signal; the atom's importance is already captured by its expected re-firing pattern.

### Q7. §3 GAN architecture cohort (6 atoms) — collapse?
- `generator-project-and-reshape` + `convtranspose-bn-activation-block` + `discriminator-classifier-head` + `conv-leakyrelu-block-discriminator` + `dcgan-wrapper-netG-netD` + `channel-list-reverse-build` all fire only in exercise 5. Could collapse to one `dcgan-block-composition`.
- **Default**: keep 6 granular. Each is independently a failure mode (forgetting BN in some blocks, wrong activation on last layer, etc.). Same as part3/part4 cohort-collapse default.

### Q8. §5 DCGAN init cohort — collapse 5.1 + 5.2?
- Both fire in exercise 6 only and are described as a unit in the DCGAN paper.
- **Default**: keep separate. 5.1 (Conv/Linear weights from N(0, 0.02)) and 5.2 (BN: weight from N(1, 0.02), bias = 0) ARE conceptually different inits; either being wrong is independently a "model won't train" bug.

### Q9. §6 transposed-conv cohort — collapse 6.1 + 6.2?
- Both describe transformations applied to enable using `conv_minimal` for transposed conv.
- **Default**: keep separate. 6.1 is the spatial transformation (pad + flip); 6.2 is the channel-axis convention (in/out swap). Different failure modes.

### Q10. `convT-init-uniform-by-kernel` (6.5) vs part4's `kaiming-uniform-sf-init` — same atom?
- Both use uniform `[-sf, sf]` with a fan-based scaling factor. Different denominator (`out_channels * kH * kW` here vs `in_features` in part4).
- **Default**: keep separate. The denominator IS the load-bearing distinction (fan-out vs fan-in conventions). Cross-part audit note: consider renaming `kaiming-uniform-sf-init` to `linear-fan-in-uniform-init` for cleaner contrast — deferred.

### Q11. Cross-part reuse confirmations (§8 explicit + ~13 incidental)
- Explicit reuses (8): `pad1d/pad2d`, `conv-windowing-1d`, `conv-windowing-2d`, `tensor-zeros-init`, `slice-view-mutation`, `training-step-cycle`, `dataclass-training-args`, `wandb-log-step`.
- Incidental reuses (~13): `nn-module-subclass`, `module-composition`, `nn-parameter-wrap`, `conv-output-shape`, `batchnorm-running-stats`, `batchnorm-affine-params`, `relu-elementwise-max`, `tqdm-postfix-metrics`, `wandb-init-run`, `wandb-watch-model`, `wandb-finish`, `dataloader-batching`, `loss-item-scalar-extract`, `inference-mode-step`.
- **Default**: split is OK — explicit ones get markdown callout; incidental ones fire silently. Same split as part4's §9 vs incidental.

### Q12. Cut atoms — confirm cuts
- `bce-loss-call` (solution uses explicit `-log()` form, not `nn.BCELoss`)
- `transposed-conv-windowed-einsum` (impl path goes through `conv1d_minimal`/`conv2d_minimal`, so R8.2/R8.3 cover it)
- `kaiming-uniform-init-on-convT` (DCGAN uses `normal_(0, 0.02)` not Kaiming)
- `latent-space-interpolation` (visualization-only, no test cell)
- `celeba-dataset-loading` (data-loading boilerplate)
- `normalize-output-uint8-for-wandb-image` (single-purpose viz scaling)
- `mnist-vs-celeb-hparam-presets` (configuration, not a coding skill)
- `inplace-relu-flag` (not used in this part)
- **Default**: cuts stand.

### Q13. Cross-part reuse audit (forward look)
- §6 transposed-conv atoms unlikely to recur outside chapter 0 bonus (transformers don't use convT).
- VAE math atoms (2.1-2.4) won't recur in chapter 0; reparam trick recurs anywhere with stochastic latents (RL value functions, mixture models).
- GAN training cohort (§4) essentially unique to part5.
- DCGAN init scheme (§5) might recur in chapter 2 RL exercises that build DCGAN-style policies.
- **Default**: pre-flagged for cross-chapter convergence test.

---

## Part 5 / tagging

Full calibration: `concept-graph/exercises/CALIBRATION_REPORT_PART5.md`.

12/12 exercises tagged. 106 atom-tags. 0 new atoms proposed. 4 seed false positives. Validator: all part5 atoms resolve cleanly (pre-existing part2 errors unrelated, same as part3/part4 calibrations).

No questions blocked the pipeline during tagging — all decisions were either pre-resolved by Q1-Q13 in the part5 vocab-gen queue or fell into default-resolvable patterns. Two new questions (Q19, Q20) logged with defaults-taken.

### Q19. Seed under-predicted training-loop exercises by ~5 atoms each
- 0_5_1 (AE train) predicted 7, actual 13 (+6); 0_5_3 (VAE train) predicted 11, actual 16 (+5); 0_5_7 (GAN train) predicted 13, actual 18 (+5).
- Each undercount is the wandb+tqdm+inference-mode+dataloader+tensor-to-device cluster (wandb-init-run, wandb-watch-model, wandb-finish, tqdm-postfix-metrics, inference-mode-step, dataloader-batching, tensor-to-device — 5-7 cross-part incidental atoms) that transfers fully into every training-loop exercise.
- **Default**: tag what's in the visible cells. No vocab change — these are all already in atoms.json. Lesson for next part: any future ARENA training-loop exercise should bake in this 5-7 atom cluster as a default reusable prediction. Logged for chapter 1 transformer-training exercise prep.

### Q20. §2 VAE-math atoms re-firing predicted in 0_5_3 (training loop) but actually fire only at 0_5_2 (VAE construction)
- Seed predicted `mu-logsigma-encoder-head`, `reparameterization-trick`, `randn-like-noise-source` would re-fire in 0_5_3 (VAE training loop). All 3 are FPs because the VAE training loop only calls `self.model(img)` which returns `(img_reconstructed, mu, logsigma)` — the head/reparam/randn-like code lives in `VAE.sample_latent_vector` defined in 0_5_2.
- **Default**: §2 atoms tag at IMPLEMENTATION exercise only (0_5_2). Same policy as part4 Q14: composition-without-implementation should not re-fire the architecture atoms. Filed all 3 as FPs in 0_5_3.json's `atoms_in_seed_but_not_actually_present`.
- Why default might be wrong: if user wants the SR system to surface "you struggled with the reparam trick when computing KL loss", then re-firing at the consumer would be the right call. But that's the CHAIN-OF-CONCEPTS view, distinct from this vocab's per-atom granularity. Same as part4 Q14.

### Q21. `model-train-eval-toggle-around-sample` did NOT fire in 0_5_7
- Seed predicted netG.eval()/.train() around log_samples for GANs, but the actual solution doesn't toggle — model constructed with `.train()` in __init__ and `@t.inference_mode()` handles the no-grad/dropout semantics for sampling without explicit eval().
- **Default**: filed as FP in 0_5_7.json. The atom is now a 0-tag singleton in part5 (no exercise uses it). Kept in vocab as a candidate for chapter 1/2 transformer/RL exercises where the eval/train toggle may appear explicitly.

### Q22. Cross-part reuse audit (forward look) — part5 is the most-cross-part-reusing chapter
- 23 of 55 unique atoms used in part5 (42%) are cross-part reuses, vs part4's ~17%, part3's ~12%. The training-loop trio (1, 3, 7) pulls in the full wandb+tqdm+training-step-cycle bundle from parts 2/3, plus the convT exercises (8, 9, 10) reuse conv-padding-zero / conv-windowing-1d / conv-windowing-2d / tensor-zeros-init / slice-view-mutation from parts 1/2.
- **Default**: no vocab change. Reinforces that the part2/3 training-infra and part1/2 conv-primitive atoms transferred correctly into part5. The bridge atom `encoder-decoder-symmetric` (3×, 0/2/5) bridges autoencoder/VAE/GAN architecture and is the structural equivalent of part4's `backward-fn-signature`.
