# 3-way concept-edges comparison

## Per-source summary

| source | total | is-a | refines | uses | part-of | alternative-to |
|---|--:|--:|--:|--:|--:|--:|
| mine | 170 | 17 | 14 | 105 | 28 | 6 |
| opus | 359 | 12 | 12 | 240 | 93 | 2 |
| codex | 380 | 21 | 20 | 239 | 97 | 3 |

## Pairwise agreement

| pair | full-triple agree | pair-only (diff kind) | only A | only B | direction-flips |
|---|--:|--:|--:|--:|--:|
| mine ↔ opus | 57 | 15 | 92 | 274 | 6 |
| mine ↔ codex | 44 | 17 | 100 | 310 | 9 |
| opus ↔ codex | 87 | 31 | 212 | 240 | 22 |

## 3-way overlap

- **Edges all 3 agree on (same `(from, to)`):** 44
- **Edges all 3 agree on AND same `kind`:** 31
- **Edges with 2-of-3 support:** 119
- **Only in mine:** 81
- **Only in opus:** 206
- **Only in codex:** 245

## All-3 consensus edges (same pair, same kind)
These are the high-confidence relations — promote to `accepted` without review.

- `add-sub-div-back-lambdas` → `backward-fn-signature` [is-a]
- `as-strided-windowing` → `contiguous-layout` [uses]
- `backward-on-scalar-loss` → `training-step-cycle` [part-of]
- `conv-windowing-2d` → `conv-windowing-1d` [refines]
- `einops-repeat-broadcast` → `broadcasting-rules` [uses]
- `encoder-decoder-symmetric` → `module-composition` [uses]
- `exp-back` → `backward-fn-signature` [is-a]
- `generator-loss-fool-discriminator` → `two-optimizers-alternating-step` [part-of]
- `getitem-back-add-at` → `backward-fn-signature` [is-a]
- `kl-divergence-gaussian-closed-form` → `elbo-loss-sum-with-beta` [part-of]
- `kwargs-pass-through-recipe` → `recipe-dataclass` [part-of]
- `log-back` → `backward-fn-signature` [is-a]
- `matmul-back-transpose-pair` → `backward-fn-signature` [is-a]
- `matvec` → `matmul-2d` [refines]
- `max-back-tied-half` → `backward-fn-signature` [is-a]
- `module-composition` → `nn-module-subclass` [part-of]
- `mse-reconstruction-loss` → `elbo-loss-sum-with-beta` [part-of]
- `multiply-back` → `backward-fn-signature` [is-a]
- `negative-back` → `backward-fn-signature` [is-a]
- `nn-parameter-wrap` → `nn-module-subclass` [part-of]
- `parameter-subclass-of-tensor` → `tensor-wraps-ndarray` [is-a]
- `parents-dict-by-argidx` → `recipe-dataclass` [part-of]
- `permute-back-argsort` → `backward-fn-signature` [is-a]
- `reshape-back` → `backward-fn-signature` [is-a]
- `singular-matrix-mask-trick` → `linalg-solve-batched` [uses]
- `sqrt-eps-stabilize` → `ema-second-moment` [uses]
- `sum-back-expand-broadcast` → `backward-fn-signature` [is-a]
- `triangle-barycentric` → `linalg-solve-batched` [uses]
- `triangle-barycentric` → `ray-parametric-form` [uses]
- `unbroadcast-pattern` → `broadcasting-rules` [uses]
- `weight-decay-decoupled` → `weight-decay-l2-add` [refines]

## Pair-agreement but kind disagreement

- `arg-position-back-functions` → `backward-fn-signature` — mine=uses, opus=part-of, codex=part-of
- `box-array-to-tensor-with-recipe` → `wrap-forward-fn-generic` — mine=uses, opus=part-of, codex=part-of
- `clip-grad-norm-pre-step` → `two-optimizers-alternating-step` — mine=uses, opus=part-of, codex=part-of
- `conv-windowing-1d` → `as-strided-windowing` — mine=uses, opus=uses, codex=refines
- `convT-kernel-axis-swap` → `convT-as-flipped-padded-conv` — mine=part-of, opus=part-of, codex=uses
- `end-grad-default-ones-like` → `backprop-pop-outgrad-loop` — mine=uses, opus=part-of, codex=part-of
- `fractional-stride-zero-insertion` → `convT-as-flipped-padded-conv` — mine=part-of, opus=part-of, codex=refines
- `functional-module-wrap` → `nn-module-subclass` — mine=refines, opus=part-of, codex=is-a
- `grads-dict-accumulate-parents` → `backprop-pop-outgrad-loop` — mine=uses, opus=part-of, codex=part-of
- `kwargs-pass-through-recipe` → `recipe-dataclass` — mine=part-of, opus=part-of,uses, codex=part-of
- `non-diff-fn-wrap` → `wrap-forward-fn-generic` — mine=alternative-to, opus=uses, codex=part-of
- `parameter-wrap-around-tensor` → `parameter-subclass-of-tensor` — mine=uses, opus=uses, codex=refines
- `trainer-subclass-extend` → `trainer-class-skeleton` — mine=refines, opus=refines,uses, codex=is-a
- `unbox-args-tensor-to-array` → `wrap-forward-fn-generic` — mine=uses, opus=part-of, codex=part-of

## Pairwise direction-flips (both have the link, disagree on direction)

### mine ↔ opus flips (6)

- `all-reduce-compose → all-reduce-grad-sync` (mine) vs `all-reduce-grad-sync → all-reduce-compose` (opus)
- `coerce-float-arg-to-array → multiply-back` (mine) vs `multiply-back → coerce-float-arg-to-array` (opus)
- `dfs-three-set-toposort → sorted-computational-graph` (mine) vs `sorted-computational-graph → dfs-three-set-toposort` (opus)
- `maxpool-reduce → avgpool-reduce` (mine) vs `avgpool-reduce → maxpool-reduce` (opus)
- `rank-world-size-args → init-process-group-nccl` (mine) vs `init-process-group-nccl → rank-world-size-args` (opus)
- `sum-and-broadcast-duality → unbroadcast-pattern` (mine) vs `unbroadcast-pattern → sum-and-broadcast-duality` (opus)

### mine ↔ codex flips (9)

- `as-strided-windowing → stride-zero-broadcast` (mine) vs `stride-zero-broadcast → as-strided-windowing` (codex)
- `backprop-pop-outgrad-loop → sorted-computational-graph` (mine) vs `sorted-computational-graph → backprop-pop-outgrad-loop` (codex)
- `dfs-three-set-toposort → sorted-computational-graph` (mine) vs `sorted-computational-graph → dfs-three-set-toposort` (codex)
- `inference-mode-step → validation-no-grad` (mine) vs `validation-no-grad → inference-mode-step` (codex)
- `linspace-out-param → slice-view-mutation` (mine) vs `slice-view-mutation → linspace-out-param` (codex)
- `maxpool-reduce → avgpool-reduce` (mine) vs `avgpool-reduce → maxpool-reduce` (codex)
- `mp-spawn-workers → per-rank-cuda-device` (mine) vs `per-rank-cuda-device → mp-spawn-workers` (codex)
- `wrap-forward-fn-generic → recipe-dataclass` (mine) vs `recipe-dataclass → wrap-forward-fn-generic` (codex)
- `zero-grad-set-none → param-grad-access` (mine) vs `param-grad-access → zero-grad-set-none` (codex)

### opus ↔ codex flips (22)

- `backprop-pop-outgrad-loop → grad-accumulate-on-leaf` (opus) vs `grad-accumulate-on-leaf → backprop-pop-outgrad-loop` (codex)
- `bn-weight-bias-init-pattern → module-modules-iter-isinstance-dispatch` (opus) vs `module-modules-iter-isinstance-dispatch → bn-weight-bias-init-pattern` (codex)
- `conv-kernel-shape → conv-output-shape` (opus) vs `conv-output-shape → conv-kernel-shape` (codex)
- `conv-stride-downsample → conv-output-shape` (opus) vs `conv-output-shape → conv-stride-downsample` (codex)
- `conv-windowing-1d → conv-kernel-shape` (opus) vs `conv-kernel-shape → conv-windowing-1d` (codex)
- `conv-windowing-1d → conv-padding-zero` (opus) vs `conv-padding-zero → conv-windowing-1d` (codex)
- `conv-windowing-2d → conv-kernel-shape` (opus) vs `conv-kernel-shape → conv-windowing-2d` (codex)
- `conv-windowing-2d → conv-output-shape` (opus) vs `conv-output-shape → conv-windowing-2d` (codex)
- `conv-windowing-2d → conv-padding-zero` (opus) vs `conv-padding-zero → conv-windowing-2d` (codex)
- `conv-windowing-2d → conv-stride-downsample` (opus) vs `conv-stride-downsample → conv-windowing-2d` (codex)
- `conv-windowing-2d → einsum-contraction` (opus) vs `einsum-contraction → conv-windowing-2d` (codex)
- `dcgan-normal-init-002 → module-modules-iter-isinstance-dispatch` (opus) vs `module-modules-iter-isinstance-dispatch → dcgan-normal-init-002` (codex)
- `dcgan-wrapper-netG-netD → module-composition` (opus) vs `module-composition → dcgan-wrapper-netG-netD` (codex)
- `ema-second-moment → ema-first-moment` (opus) vs `ema-first-moment → ema-second-moment` (codex)
- `mp-spawn-workers → init-process-group-nccl` (opus) vs `init-process-group-nccl → mp-spawn-workers` (codex)
- `pseudocode-to-code-translate → ema-second-moment` (opus) vs `ema-second-moment → pseudocode-to-code-translate` (codex)
- `pseudocode-to-code-translate → sqrt-eps-stabilize` (opus) vs `sqrt-eps-stabilize → pseudocode-to-code-translate` (codex)
- `requires-grad-propagation → grad-tracking-global-toggle` (opus) vs `grad-tracking-global-toggle → requires-grad-propagation` (codex)
- `tqdm-postfix-metrics → loss-item-scalar-extract` (opus) vs `loss-item-scalar-extract → tqdm-postfix-metrics` (codex)
- `wandb-log-step → loss-item-scalar-extract` (opus) vs `loss-item-scalar-extract → wandb-log-step` (codex)
- `wrap-forward-fn-generic → tensor-wraps-ndarray` (opus) vs `tensor-wraps-ndarray → wrap-forward-fn-generic` (codex)
- `zero-grad-set-none → param-grad-access` (opus) vs `param-grad-access → zero-grad-set-none` (codex)

