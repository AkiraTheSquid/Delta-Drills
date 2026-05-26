// Auto-generated bridge — DO NOT EDIT BY HAND.
// Regenerate via: arena-procedural-drills/scripts/gen_prereq_subtopics.py
// Source of truth: arena-procedural-drills/ notebook metadata (walked at build time).
//
// Lists the prereq subtopic keys introduced by Colab procedural drills.
// These keys have NO flashcards in the regular bank — the drill beacon is
// the only way to bump their EWMA. atom_readiness.js consumes this to map
// drill-atom IDs to their direct subtopic for Case 0 resolution.

window.PREREQ_SUBTOPICS = {
  "schema_version": 1,
  "generated_at": "2026-05-25",
  "source": "arena-procedural-drills/ notebook metadata (canonical walk)",
  "description": "Prereq subtopic keys introduced by Colab procedural drills. These keys are NOT in the regular question bank (zero flashcards) \u2014 drills are the only practice surface. Backend questions.py + frontend atom_readiness.js both consume this file to know about them.",
  "atom_to_subtopic": {
    "1x1-conv-channel-reshape": "CNN: 1x1 conv channel-reshape",
    "add-sub-div-back-lambdas": "Backprop: add/sub/div back as lambdas",
    "all-reduce-compose": "Distributed: all_reduce composition",
    "all-reduce-eval-metrics": "Distributed: all_reduce eval metrics",
    "all-reduce-grad-sync": "Distributed: all_reduce grad sync",
    "any-reduce-axis": "Numpy: any() reduce along axis",
    "arange-fancy-index-cross-entropy": "Loss: arange fancy-index cross-entropy",
    "arg-position-back-functions": "Backprop: Arg-position back funcs",
    "argmax-accuracy-eval": "Eval: argmax accuracy",
    "as-strided-noncontig-source": "Numpy: Applied patterns and advanced",
    "as-strided-windowing": "PyTorch: as_strided windowing",
    "avgpool-reduce": "CNN: AvgPool as reduce",
    "back-fn-call-with-recipe-args": "Backprop: back fn call with recipe args",
    "backprop-pop-outgrad-loop": "Backprop: backprop pop-outgrad loop",
    "backward-fn-signature": "Backprop: backward fn signature",
    "backward-func-lookup": "Backprop: BackwardFuncLookup",
    "backward-on-scalar-loss": "PyTorch: backward()",
    "batchnorm-affine-params": "CNN: BatchNorm affine params",
    "batchnorm-running-stats": "CNN: BatchNorm running stats",
    "bce-log-loss-real-fake": "GAN: BCE log loss real/fake",
    "bias-correction-divide": "Optimizer: Adam bias-correction divide",
    "block-group-stack": "CNN: BlockGroup stack",
    "bn-weight-bias-init-pattern": "GAN: BN weight=1 bias=0 init",
    "boolean-mask-combine": "Numpy: Boolean mask combine",
    "boolean-mask-identity-replace": "Numpy: Indexing and selection",
    "bottleneck-latent-projection": "Generative: Bottleneck latent projection",
    "box-array-to-tensor-with-recipe": "Backprop: Box array as Tensor + recipe",
    "broadcast-initial-weights": "Distributed: broadcast initial weights",
    "broadcast-source-fanout": "Generative: Broadcast source fan-out",
    "broadcasting-rules": "Numpy: Vectorization and broadcasting",
    "buffer-copy_-inplace": "PyTorch: in-place buffer copy",
    "chain-rule-elementwise": "Backprop: Elementwise chain rule",
    "channel-list-reverse-build": "GAN: channel-list reverse build",
    "clip-grad-norm-pre-step": "Optimizer: clip_grad_norm pre-step",
    "coerce-float-arg-to-array": "Backprop: Coerce float arg to array",
    "conditional-hparam-branch": "PyTorch: Conditional hparam branch",
    "contiguous-layout": "PyTorch: Contiguous layout",
    "conv-channel-sum": "CNN: Channel-axis sum semantics",
    "conv-kernel-shape": "CNN: Kernel shape (OC, IC, KH, KW)",
    "conv-leakyrelu-block-discriminator": "GAN: Conv+LeakyReLU discriminator block",
    "conv-output-shape": "CNN: Conv output shape",
    "conv-padding-zero": "CNN: Conv zero padding",
    "conv-stride-downsample": "CNN: Stride downsample arithmetic",
    "conv-windowing-1d": "CNN: 1-D conv windowing",
    "conv-windowing-2d": "CNN: 2-D conv windowing",
    "convT-as-flipped-padded-conv": "CNN: ConvT as flipped padded conv",
    "convT-kernel-axis-swap": "CNN: ConvT kernel axis swap",
    "convtranspose-bn-activation-block": "GAN: ConvT+BN+Activation block",
    "cross-entropy-classification-loss": "Loss: Cross-entropy classification",
    "cross-product-normal": "Geometry: Cross-product surface normal",
    "cuda-empty-cache": "PyTorch: torch.cuda.empty_cache",
    "cycle-detection-temp-set": "Backprop: cycle detection via temp set",
    "dataclass-training-args": "Config: @dataclass training args",
    "dataclasses-replace-args": "Config: dataclasses.replace args",
    "dataloader-batching": "PyTorch: DataLoader batching",
    "dataloader-pin-memory-workers": "PyTorch: DataLoader pin_memory + workers",
    "dcgan-normal-init-002": "GAN: DCGAN normal init 0.02",
    "dcgan-wrapper-netG-netD": "Generative: DCGAN netG+netD wrapper",
    "detach-clone-snapshot": "PyTorch: detach + clone snapshot",
    "detach-stop-gradient-trick": "GAN: detach stop-gradient trick",
    "device-consistent-construct": "PyTorch: Device-consistent tensor construction",
    "dfs-three-set-toposort": "Backprop: DFS three-set toposort",
    "diagonal-via-strides": "Numpy: Diagonal via strides",
    "discriminator-classifier-head": "GAN: Discriminator classifier head",
    "dispatch-back-fn-from-recipe": "Backprop: dispatch back fn from recipe",
    "dist-send-recv-pair": "Distributed: dist.send/recv pair",
    "distributed-sampler-shard": "Distributed: DistributedSampler shard",
    "einops-einsum": "Einops: Deep Learning",
    "einops-rearrange": "Einops: Rearrange",
    "einops-rearrange-flatten": "Einops: Rearrange-as-flatten",
    "einops-reduce": "Einops: Reduce",
    "einops-reduce-min": "Einops: Reduce with min",
    "einops-repeat": "Einops: Repeat",
    "einops-repeat-broadcast": "Einops: Repeat-as-broadcast",
    "einsum-contraction": "Einsum: Index contraction semantics",
    "elbo-loss-sum-with-beta": "VAE: ELBO loss sum with beta",
    "ema-first-moment": "Optimizer: Adam EMA first moment",
    "ema-second-moment": "Optimizer: Adam EMA second moment",
    "encoder-decoder-symmetric": "CNN: Encoder-decoder symmetric layout",
    "end-grad-default-ones-like": "Backprop: end-grad ones_like default",
    "examples-seen-step-axis": "Trainer: examples-seen step axis",
    "exp-back": "Backprop: exp_back",
    "fractional-stride-zero-insertion": "CNN: ConvT fractional-stride zero insertion",
    "freeze-requires-grad": "PyTorch: freeze via requires_grad=False",
    "functional-module-wrap": "PyTorch: functional module wrap",
    "generator-loss-fool-discriminator": "GAN: Generator loss to fool D",
    "generator-project-and-reshape": "GAN: Generator project + reshape",
    "get-children-callable-param": "Backprop: get_children callable param",
    "getitem-back-add-at": "Backprop: getitem_back via add-at",
    "grad-accumulate-on-leaf": "Backprop: Grad accumulate on leaf",
    "grad-expressed-in-out": "Backprop: grad expressed in out",
    "grad-tracking-global-toggle": "Backprop: Grad-tracking toggle",
    "grads-dict-accumulate-parents": "Backprop: grads dict accumulate parents",
    "holdout-data-one-per-class": "Generative: Hold-out one-per-class data",
    "hparam-precedence-merge": "Config: hparam precedence merge",
    "index-by-tensor": "PyTorch: index by tensor",
    "inf-masking": "Numpy: Inf-fill masking trick",
    "inference-mode-step": "PyTorch: Inference mode step",
    "init-process-group-nccl": "Distributed: init_process_group nccl",
    "inplace-op-unsafe-warning": "Backprop: In-place op unsafe warning",
    "inplace-param-update": "PyTorch: In-place param update",
    "is-differentiable-flag": "Backprop: is_differentiable flag",
    "kaiming-uniform-init": "Init: Kaiming uniform",
    "kaiming-uniform-sf-init": "Init: Kaiming uniform SF init",
    "kl-divergence-gaussian-closed-form": "VAE: KL divergence Gaussian closed-form",
    "kwargs-pass-through-recipe": "Backprop: Kwargs pass-through",
    "leaf-tensor-condition": "Backprop: leaf tensor condition",
    "linalg-solve-batched": "PyTorch: Batched linalg.solve",
    "linear-affine-on-custom-tensor": "Backprop: Linear affine on custom Tensor",
    "linspace-out-param": "PyTorch: linspace out= param",
    "log-back": "Backprop: log_back",
    "log-samples-eval-callback": "Logging: log-samples eval callback",
    "logsumexp-cross-entropy": "Loss: logsumexp cross-entropy",
    "loss-item-scalar-extract": "PyTorch: loss.item() scalar extract",
    "manual-chain-forward-and-back": "Backprop: manual chain forward-and-back",
    "matmul-2d": "Numpy: matmul 2-D",
    "matmul-back-transpose-pair": "Backprop: matmul_back transpose pair",
    "matvec": "PyTorch: matrix-vector product",
    "max-back-tied-half": "Backprop: max_back with tied half-mass",
    "maxpool-reduce": "CNN: MaxPool as reduce",
    "model-save-state-dict": "Distributed: model save state_dict rank-0",
    "model-train-eval-toggle-around-sample": "GAN: model.train/eval toggle around sample",
    "module-base-class-custom": "Backprop: Module base class custom",
    "module-composition": "PyTorch: Module composition",
    "module-extra-repr": "PyTorch: Module __repr__",
    "module-modules-iter-isinstance-dispatch": "GAN: model.modules() isinstance dispatch",
    "momentum-buffer-update": "Optimizer: Momentum buffer",
    "mp-spawn-workers": "Distributed: mp.spawn workers",
    "mse-reconstruction-loss": "Generative: MSE reconstruction loss",
    "mu-logsigma-encoder-head": "VAE: mu+logsigma encoder head",
    "multiply-back": "Backprop: multiply_back",
    "negative-back": "Backprop: negative_back",
    "nested-param-group-loop": "Config: nested param-group loop",
    "nn-module-subclass": "PyTorch: nn.Module subclassing",
    "nn-parameter-wrap": "PyTorch: nn.Parameter",
    "no-grad-context-mgr-update": "Backprop: no_grad ctx-mgr update",
    "no-relu-on-final-layer": "CNN: No-ReLU on final layer",
    "noise-batch-from-latent": "GAN: Noise batch from latent_dim",
    "non-diff-fn-wrap": "Backprop: non-differentiable fn wrap",
    "optimizer-class-dispatch": "Config: Optimizer class dispatch",
    "optimizer-init-params-list": "PyTorch: Optimizer init",
    "optimizer-loop-on-tensor": "Optimizer: optimizer.step loop over params",
    "optimizer-repr-string": "Optimizer: __repr__ string",
    "optimizer-state-tensor-buffers": "Optimizer: Per-param state buffers",
    "padding-amount-formula-convT": "CNN: ConvT padding amount formula",
    "param-grad-access": "PyTorch: param.grad access",
    "param-group-dict-list": "Config: param-group dict list",
    "parameter-subclass-of-tensor": "Backprop: Parameter subclasses Tensor",
    "parameter-wrap-around-tensor": "Backprop: Parameter wrap around Tensor",
    "params-iterable-vs-groups": "Config: params iterable vs groups",
    "parents-dict-by-argidx": "Backprop: Parents dict by argidx",
    "per-rank-cuda-device": "Distributed: per-rank cuda device",
    "permute-back-argsort": "Backprop: permute_back via argsort",
    "randn-like-noise-source": "Generative: randn-like noise source",
    "rank-world-size-args": "Distributed: rank/world_size args",
    "rank0-only-side-effects": "Distributed: rank-0-only side effects",
    "ray-parametric-form": "Geometry: Ray parametric form",
    "rearrange-as-sequential-layer": "Einops: Rearrange as nn.Sequential layer",
    "recipe-dataclass": "Backprop: Recipe dataclass",
    "reduce-gather-sum": "Distributed: reduce.gather + sum",
    "reduce-op-mean-divide": "Distributed: reduce-op mean divide",
    "register-back-fn-after-wrap": "Backprop: register back fn",
    "register-buffer": "PyTorch: register_buffer",
    "relu-elementwise-max": "CNN: ReLU as elementwise max",
    "reparameterization-trick": "VAE: Reparameterization trick",
    "replace-final-head": "Transfer: Replace final head",
    "requires-grad-leaf-assert": "Generative: requires_grad leaf assert",
    "requires-grad-propagation": "Backprop: requires_grad propagation",
    "reshape-back": "Backprop: reshape_back",
    "residual-skip-add": "CNN: Residual skip-connection add",
    "resnet-stem": "CNN: ResNet stem block",
    "rmul-scalar-tensor-mix": "PyTorch: __rmul__ scalar/tensor mix",
    "rotation-matrix-3d": "Geometry: Rotation matrix 3-D (full)",
    "rotation-matrix-3d-y-axis": "Numpy: Applied patterns and advanced",
    "segment-line-intersect-2d": "Geometry: Segment-line intersect 2-D",
    "sgd-vanilla-from-scratch": "Optimizer: SGD vanilla from scratch",
    "singular-matrix-mask-trick": "Numpy: Singular matrix mask trick",
    "slice-view-mutation": "PyTorch: Slice view mutation",
    "sorted-computational-graph": "Backprop: Sorted computation graph",
    "sqrt-eps-stabilize": "Numerical: sqrt-eps stabilization",
    "stack-vs-cat": "PyTorch: stack vs cat",
    "state-dict-load": "Transfer: state_dict load",
    "step-counter-increment": "Trainer: step counter increment",
    "stride-zero-broadcast": "PyTorch: Zero-stride broadcasting",
    "sum-and-broadcast-duality": "Backprop: sum/broadcast duality",
    "sum-back-expand-broadcast": "Backprop: sum_back via expand_broadcast",
    "sweep-config-dict": "Config: wandb sweep config dict",
    "sweep-hparam-distribution": "Config: sweep hparam distribution",
    "t-stack-trajectory": "Generative: torch.stack trajectory",
    "tensor-item-scalar": "Numpy: Core array literacy",
    "tensor-reshape-view": "PyTorch: reshape vs view",
    "tensor-to-device": "PyTorch: tensor.to(device)",
    "tensor-unbind": "Numpy: Indexing and selection",
    "tensor-wraps-ndarray": "PyTorch: tensor from ndarray",
    "tensor-zeros-init": "Numpy: Core array literacy",
    "time-stage-instrumentation": "Logging: time-stage instrumentation",
    "topk-predictions": "Eval: topk predictions",
    "tqdm-postfix-metrics": "Logging: tqdm postfix metrics",
    "train-eval-mode-branch": "PyTorch: train/eval mode",
    "trainer-class-skeleton": "Trainer: Trainer class skeleton",
    "trainer-subclass-extend": "Trainer: subclass extend pattern",
    "training-step-cycle": "PyTorch: Training step cycle",
    "triangle-barycentric": "Geometry: Barycentric coords",
    "try-except-solve": "LinAlg: try/except solve",
    "two-optimizers-alternating-step": "GAN: Two-optimizers alternating step",
    "unbind-tuple-unpack": "PyTorch: Unbind tuple-unpack",
    "unbox-args-tensor-to-array": "Backprop: Unbox Tensor args to array",
    "unbroadcast-pattern": "Backprop: Unbroadcast pattern",
    "validation-no-grad": "PyTorch: no_grad validation",
    "vector-normalize-keepdim": "PyTorch: vector normalize keepdim",
    "wandb-config-into-args": "Logging: wandb.config into args",
    "wandb-finish": "Logging: wandb.finish",
    "wandb-init-run": "Logging: wandb.init run",
    "wandb-log-step": "Logging: wandb.log step",
    "wandb-watch-model": "Logging: wandb.watch model",
    "weight-decay-decoupled": "Optimizer: decoupled weight decay (AdamW)",
    "weight-decay-l2-add": "Optimizer: Weight decay L2",
    "where-clip-negative": "PyTorch: where to clip negative",
    "wrap-forward-fn-generic": "Backprop: wrap forward fn",
    "zero-grad-set-none": "PyTorch: zero_grad"
  },
  "subtopic_to_atoms": {
    "Backprop: Arg-position back funcs": [
      "arg-position-back-functions"
    ],
    "Backprop: BackwardFuncLookup": [
      "backward-func-lookup"
    ],
    "Backprop: Box array as Tensor + recipe": [
      "box-array-to-tensor-with-recipe"
    ],
    "Backprop: Coerce float arg to array": [
      "coerce-float-arg-to-array"
    ],
    "Backprop: DFS three-set toposort": [
      "dfs-three-set-toposort"
    ],
    "Backprop: Elementwise chain rule": [
      "chain-rule-elementwise"
    ],
    "Backprop: Grad accumulate on leaf": [
      "grad-accumulate-on-leaf"
    ],
    "Backprop: Grad-tracking toggle": [
      "grad-tracking-global-toggle"
    ],
    "Backprop: In-place op unsafe warning": [
      "inplace-op-unsafe-warning"
    ],
    "Backprop: Kwargs pass-through": [
      "kwargs-pass-through-recipe"
    ],
    "Backprop: Linear affine on custom Tensor": [
      "linear-affine-on-custom-tensor"
    ],
    "Backprop: Module base class custom": [
      "module-base-class-custom"
    ],
    "Backprop: Parameter subclasses Tensor": [
      "parameter-subclass-of-tensor"
    ],
    "Backprop: Parameter wrap around Tensor": [
      "parameter-wrap-around-tensor"
    ],
    "Backprop: Parents dict by argidx": [
      "parents-dict-by-argidx"
    ],
    "Backprop: Recipe dataclass": [
      "recipe-dataclass"
    ],
    "Backprop: Sorted computation graph": [
      "sorted-computational-graph"
    ],
    "Backprop: Unbox Tensor args to array": [
      "unbox-args-tensor-to-array"
    ],
    "Backprop: Unbroadcast pattern": [
      "unbroadcast-pattern"
    ],
    "Backprop: add/sub/div back as lambdas": [
      "add-sub-div-back-lambdas"
    ],
    "Backprop: back fn call with recipe args": [
      "back-fn-call-with-recipe-args"
    ],
    "Backprop: backprop pop-outgrad loop": [
      "backprop-pop-outgrad-loop"
    ],
    "Backprop: backward fn signature": [
      "backward-fn-signature"
    ],
    "Backprop: cycle detection via temp set": [
      "cycle-detection-temp-set"
    ],
    "Backprop: dispatch back fn from recipe": [
      "dispatch-back-fn-from-recipe"
    ],
    "Backprop: end-grad ones_like default": [
      "end-grad-default-ones-like"
    ],
    "Backprop: exp_back": [
      "exp-back"
    ],
    "Backprop: get_children callable param": [
      "get-children-callable-param"
    ],
    "Backprop: getitem_back via add-at": [
      "getitem-back-add-at"
    ],
    "Backprop: grad expressed in out": [
      "grad-expressed-in-out"
    ],
    "Backprop: grads dict accumulate parents": [
      "grads-dict-accumulate-parents"
    ],
    "Backprop: is_differentiable flag": [
      "is-differentiable-flag"
    ],
    "Backprop: leaf tensor condition": [
      "leaf-tensor-condition"
    ],
    "Backprop: log_back": [
      "log-back"
    ],
    "Backprop: manual chain forward-and-back": [
      "manual-chain-forward-and-back"
    ],
    "Backprop: matmul_back transpose pair": [
      "matmul-back-transpose-pair"
    ],
    "Backprop: max_back with tied half-mass": [
      "max-back-tied-half"
    ],
    "Backprop: multiply_back": [
      "multiply-back"
    ],
    "Backprop: negative_back": [
      "negative-back"
    ],
    "Backprop: no_grad ctx-mgr update": [
      "no-grad-context-mgr-update"
    ],
    "Backprop: non-differentiable fn wrap": [
      "non-diff-fn-wrap"
    ],
    "Backprop: permute_back via argsort": [
      "permute-back-argsort"
    ],
    "Backprop: register back fn": [
      "register-back-fn-after-wrap"
    ],
    "Backprop: requires_grad propagation": [
      "requires-grad-propagation"
    ],
    "Backprop: reshape_back": [
      "reshape-back"
    ],
    "Backprop: sum/broadcast duality": [
      "sum-and-broadcast-duality"
    ],
    "Backprop: sum_back via expand_broadcast": [
      "sum-back-expand-broadcast"
    ],
    "Backprop: wrap forward fn": [
      "wrap-forward-fn-generic"
    ],
    "CNN: 1-D conv windowing": [
      "conv-windowing-1d"
    ],
    "CNN: 1x1 conv channel-reshape": [
      "1x1-conv-channel-reshape"
    ],
    "CNN: 2-D conv windowing": [
      "conv-windowing-2d"
    ],
    "CNN: AvgPool as reduce": [
      "avgpool-reduce"
    ],
    "CNN: BatchNorm affine params": [
      "batchnorm-affine-params"
    ],
    "CNN: BatchNorm running stats": [
      "batchnorm-running-stats"
    ],
    "CNN: BlockGroup stack": [
      "block-group-stack"
    ],
    "CNN: Channel-axis sum semantics": [
      "conv-channel-sum"
    ],
    "CNN: Conv output shape": [
      "conv-output-shape"
    ],
    "CNN: Conv zero padding": [
      "conv-padding-zero"
    ],
    "CNN: ConvT as flipped padded conv": [
      "convT-as-flipped-padded-conv"
    ],
    "CNN: ConvT fractional-stride zero insertion": [
      "fractional-stride-zero-insertion"
    ],
    "CNN: ConvT kernel axis swap": [
      "convT-kernel-axis-swap"
    ],
    "CNN: ConvT padding amount formula": [
      "padding-amount-formula-convT"
    ],
    "CNN: Encoder-decoder symmetric layout": [
      "encoder-decoder-symmetric"
    ],
    "CNN: Kernel shape (OC, IC, KH, KW)": [
      "conv-kernel-shape"
    ],
    "CNN: MaxPool as reduce": [
      "maxpool-reduce"
    ],
    "CNN: No-ReLU on final layer": [
      "no-relu-on-final-layer"
    ],
    "CNN: ReLU as elementwise max": [
      "relu-elementwise-max"
    ],
    "CNN: ResNet stem block": [
      "resnet-stem"
    ],
    "CNN: Residual skip-connection add": [
      "residual-skip-add"
    ],
    "CNN: Stride downsample arithmetic": [
      "conv-stride-downsample"
    ],
    "Config: @dataclass training args": [
      "dataclass-training-args"
    ],
    "Config: Optimizer class dispatch": [
      "optimizer-class-dispatch"
    ],
    "Config: dataclasses.replace args": [
      "dataclasses-replace-args"
    ],
    "Config: hparam precedence merge": [
      "hparam-precedence-merge"
    ],
    "Config: nested param-group loop": [
      "nested-param-group-loop"
    ],
    "Config: param-group dict list": [
      "param-group-dict-list"
    ],
    "Config: params iterable vs groups": [
      "params-iterable-vs-groups"
    ],
    "Config: sweep hparam distribution": [
      "sweep-hparam-distribution"
    ],
    "Config: wandb sweep config dict": [
      "sweep-config-dict"
    ],
    "Distributed: DistributedSampler shard": [
      "distributed-sampler-shard"
    ],
    "Distributed: all_reduce composition": [
      "all-reduce-compose"
    ],
    "Distributed: all_reduce eval metrics": [
      "all-reduce-eval-metrics"
    ],
    "Distributed: all_reduce grad sync": [
      "all-reduce-grad-sync"
    ],
    "Distributed: broadcast initial weights": [
      "broadcast-initial-weights"
    ],
    "Distributed: dist.send/recv pair": [
      "dist-send-recv-pair"
    ],
    "Distributed: init_process_group nccl": [
      "init-process-group-nccl"
    ],
    "Distributed: model save state_dict rank-0": [
      "model-save-state-dict"
    ],
    "Distributed: mp.spawn workers": [
      "mp-spawn-workers"
    ],
    "Distributed: per-rank cuda device": [
      "per-rank-cuda-device"
    ],
    "Distributed: rank-0-only side effects": [
      "rank0-only-side-effects"
    ],
    "Distributed: rank/world_size args": [
      "rank-world-size-args"
    ],
    "Distributed: reduce-op mean divide": [
      "reduce-op-mean-divide"
    ],
    "Distributed: reduce.gather + sum": [
      "reduce-gather-sum"
    ],
    "Einops: Deep Learning": [
      "einops-einsum"
    ],
    "Einops: Rearrange": [
      "einops-rearrange"
    ],
    "Einops: Rearrange as nn.Sequential layer": [
      "rearrange-as-sequential-layer"
    ],
    "Einops: Rearrange-as-flatten": [
      "einops-rearrange-flatten"
    ],
    "Einops: Reduce": [
      "einops-reduce"
    ],
    "Einops: Reduce with min": [
      "einops-reduce-min"
    ],
    "Einops: Repeat": [
      "einops-repeat"
    ],
    "Einops: Repeat-as-broadcast": [
      "einops-repeat-broadcast"
    ],
    "Einsum: Index contraction semantics": [
      "einsum-contraction"
    ],
    "Eval: argmax accuracy": [
      "argmax-accuracy-eval"
    ],
    "Eval: topk predictions": [
      "topk-predictions"
    ],
    "GAN: BCE log loss real/fake": [
      "bce-log-loss-real-fake"
    ],
    "GAN: BN weight=1 bias=0 init": [
      "bn-weight-bias-init-pattern"
    ],
    "GAN: Conv+LeakyReLU discriminator block": [
      "conv-leakyrelu-block-discriminator"
    ],
    "GAN: ConvT+BN+Activation block": [
      "convtranspose-bn-activation-block"
    ],
    "GAN: DCGAN normal init 0.02": [
      "dcgan-normal-init-002"
    ],
    "GAN: Discriminator classifier head": [
      "discriminator-classifier-head"
    ],
    "GAN: Generator loss to fool D": [
      "generator-loss-fool-discriminator"
    ],
    "GAN: Generator project + reshape": [
      "generator-project-and-reshape"
    ],
    "GAN: Noise batch from latent_dim": [
      "noise-batch-from-latent"
    ],
    "GAN: Two-optimizers alternating step": [
      "two-optimizers-alternating-step"
    ],
    "GAN: channel-list reverse build": [
      "channel-list-reverse-build"
    ],
    "GAN: detach stop-gradient trick": [
      "detach-stop-gradient-trick"
    ],
    "GAN: model.modules() isinstance dispatch": [
      "module-modules-iter-isinstance-dispatch"
    ],
    "GAN: model.train/eval toggle around sample": [
      "model-train-eval-toggle-around-sample"
    ],
    "Generative: Bottleneck latent projection": [
      "bottleneck-latent-projection"
    ],
    "Generative: Broadcast source fan-out": [
      "broadcast-source-fanout"
    ],
    "Generative: DCGAN netG+netD wrapper": [
      "dcgan-wrapper-netG-netD"
    ],
    "Generative: Hold-out one-per-class data": [
      "holdout-data-one-per-class"
    ],
    "Generative: MSE reconstruction loss": [
      "mse-reconstruction-loss"
    ],
    "Generative: randn-like noise source": [
      "randn-like-noise-source"
    ],
    "Generative: requires_grad leaf assert": [
      "requires-grad-leaf-assert"
    ],
    "Generative: torch.stack trajectory": [
      "t-stack-trajectory"
    ],
    "Geometry: Barycentric coords": [
      "triangle-barycentric"
    ],
    "Geometry: Cross-product surface normal": [
      "cross-product-normal"
    ],
    "Geometry: Ray parametric form": [
      "ray-parametric-form"
    ],
    "Geometry: Rotation matrix 3-D (full)": [
      "rotation-matrix-3d"
    ],
    "Geometry: Segment-line intersect 2-D": [
      "segment-line-intersect-2d"
    ],
    "Init: Kaiming uniform": [
      "kaiming-uniform-init"
    ],
    "Init: Kaiming uniform SF init": [
      "kaiming-uniform-sf-init"
    ],
    "LinAlg: try/except solve": [
      "try-except-solve"
    ],
    "Logging: log-samples eval callback": [
      "log-samples-eval-callback"
    ],
    "Logging: time-stage instrumentation": [
      "time-stage-instrumentation"
    ],
    "Logging: tqdm postfix metrics": [
      "tqdm-postfix-metrics"
    ],
    "Logging: wandb.config into args": [
      "wandb-config-into-args"
    ],
    "Logging: wandb.finish": [
      "wandb-finish"
    ],
    "Logging: wandb.init run": [
      "wandb-init-run"
    ],
    "Logging: wandb.log step": [
      "wandb-log-step"
    ],
    "Logging: wandb.watch model": [
      "wandb-watch-model"
    ],
    "Loss: Cross-entropy classification": [
      "cross-entropy-classification-loss"
    ],
    "Loss: arange fancy-index cross-entropy": [
      "arange-fancy-index-cross-entropy"
    ],
    "Loss: logsumexp cross-entropy": [
      "logsumexp-cross-entropy"
    ],
    "Numerical: sqrt-eps stabilization": [
      "sqrt-eps-stabilize"
    ],
    "Numpy: Applied patterns and advanced": [
      "as-strided-noncontig-source",
      "rotation-matrix-3d-y-axis"
    ],
    "Numpy: Boolean mask combine": [
      "boolean-mask-combine"
    ],
    "Numpy: Core array literacy": [
      "tensor-item-scalar",
      "tensor-zeros-init"
    ],
    "Numpy: Diagonal via strides": [
      "diagonal-via-strides"
    ],
    "Numpy: Indexing and selection": [
      "boolean-mask-identity-replace",
      "tensor-unbind"
    ],
    "Numpy: Inf-fill masking trick": [
      "inf-masking"
    ],
    "Numpy: Singular matrix mask trick": [
      "singular-matrix-mask-trick"
    ],
    "Numpy: Vectorization and broadcasting": [
      "broadcasting-rules"
    ],
    "Numpy: any() reduce along axis": [
      "any-reduce-axis"
    ],
    "Numpy: matmul 2-D": [
      "matmul-2d"
    ],
    "Optimizer: Adam EMA first moment": [
      "ema-first-moment"
    ],
    "Optimizer: Adam EMA second moment": [
      "ema-second-moment"
    ],
    "Optimizer: Adam bias-correction divide": [
      "bias-correction-divide"
    ],
    "Optimizer: Momentum buffer": [
      "momentum-buffer-update"
    ],
    "Optimizer: Per-param state buffers": [
      "optimizer-state-tensor-buffers"
    ],
    "Optimizer: SGD vanilla from scratch": [
      "sgd-vanilla-from-scratch"
    ],
    "Optimizer: Weight decay L2": [
      "weight-decay-l2-add"
    ],
    "Optimizer: __repr__ string": [
      "optimizer-repr-string"
    ],
    "Optimizer: clip_grad_norm pre-step": [
      "clip-grad-norm-pre-step"
    ],
    "Optimizer: decoupled weight decay (AdamW)": [
      "weight-decay-decoupled"
    ],
    "Optimizer: optimizer.step loop over params": [
      "optimizer-loop-on-tensor"
    ],
    "PyTorch: Batched linalg.solve": [
      "linalg-solve-batched"
    ],
    "PyTorch: Conditional hparam branch": [
      "conditional-hparam-branch"
    ],
    "PyTorch: Contiguous layout": [
      "contiguous-layout"
    ],
    "PyTorch: DataLoader batching": [
      "dataloader-batching"
    ],
    "PyTorch: DataLoader pin_memory + workers": [
      "dataloader-pin-memory-workers"
    ],
    "PyTorch: Device-consistent tensor construction": [
      "device-consistent-construct"
    ],
    "PyTorch: In-place param update": [
      "inplace-param-update"
    ],
    "PyTorch: Inference mode step": [
      "inference-mode-step"
    ],
    "PyTorch: Module __repr__": [
      "module-extra-repr"
    ],
    "PyTorch: Module composition": [
      "module-composition"
    ],
    "PyTorch: Optimizer init": [
      "optimizer-init-params-list"
    ],
    "PyTorch: Slice view mutation": [
      "slice-view-mutation"
    ],
    "PyTorch: Training step cycle": [
      "training-step-cycle"
    ],
    "PyTorch: Unbind tuple-unpack": [
      "unbind-tuple-unpack"
    ],
    "PyTorch: Zero-stride broadcasting": [
      "stride-zero-broadcast"
    ],
    "PyTorch: __rmul__ scalar/tensor mix": [
      "rmul-scalar-tensor-mix"
    ],
    "PyTorch: as_strided windowing": [
      "as-strided-windowing"
    ],
    "PyTorch: backward()": [
      "backward-on-scalar-loss"
    ],
    "PyTorch: detach + clone snapshot": [
      "detach-clone-snapshot"
    ],
    "PyTorch: freeze via requires_grad=False": [
      "freeze-requires-grad"
    ],
    "PyTorch: functional module wrap": [
      "functional-module-wrap"
    ],
    "PyTorch: in-place buffer copy": [
      "buffer-copy_-inplace"
    ],
    "PyTorch: index by tensor": [
      "index-by-tensor"
    ],
    "PyTorch: linspace out= param": [
      "linspace-out-param"
    ],
    "PyTorch: loss.item() scalar extract": [
      "loss-item-scalar-extract"
    ],
    "PyTorch: matrix-vector product": [
      "matvec"
    ],
    "PyTorch: nn.Module subclassing": [
      "nn-module-subclass"
    ],
    "PyTorch: nn.Parameter": [
      "nn-parameter-wrap"
    ],
    "PyTorch: no_grad validation": [
      "validation-no-grad"
    ],
    "PyTorch: param.grad access": [
      "param-grad-access"
    ],
    "PyTorch: register_buffer": [
      "register-buffer"
    ],
    "PyTorch: reshape vs view": [
      "tensor-reshape-view"
    ],
    "PyTorch: stack vs cat": [
      "stack-vs-cat"
    ],
    "PyTorch: tensor from ndarray": [
      "tensor-wraps-ndarray"
    ],
    "PyTorch: tensor.to(device)": [
      "tensor-to-device"
    ],
    "PyTorch: torch.cuda.empty_cache": [
      "cuda-empty-cache"
    ],
    "PyTorch: train/eval mode": [
      "train-eval-mode-branch"
    ],
    "PyTorch: vector normalize keepdim": [
      "vector-normalize-keepdim"
    ],
    "PyTorch: where to clip negative": [
      "where-clip-negative"
    ],
    "PyTorch: zero_grad": [
      "zero-grad-set-none"
    ],
    "Trainer: Trainer class skeleton": [
      "trainer-class-skeleton"
    ],
    "Trainer: examples-seen step axis": [
      "examples-seen-step-axis"
    ],
    "Trainer: step counter increment": [
      "step-counter-increment"
    ],
    "Trainer: subclass extend pattern": [
      "trainer-subclass-extend"
    ],
    "Transfer: Replace final head": [
      "replace-final-head"
    ],
    "Transfer: state_dict load": [
      "state-dict-load"
    ],
    "VAE: ELBO loss sum with beta": [
      "elbo-loss-sum-with-beta"
    ],
    "VAE: KL divergence Gaussian closed-form": [
      "kl-divergence-gaussian-closed-form"
    ],
    "VAE: Reparameterization trick": [
      "reparameterization-trick"
    ],
    "VAE: mu+logsigma encoder head": [
      "mu-logsigma-encoder-head"
    ]
  },
  "prereq_subtopic_keys": [
    "Backprop: Arg-position back funcs",
    "Backprop: BackwardFuncLookup",
    "Backprop: Box array as Tensor + recipe",
    "Backprop: Coerce float arg to array",
    "Backprop: DFS three-set toposort",
    "Backprop: Elementwise chain rule",
    "Backprop: Grad accumulate on leaf",
    "Backprop: Grad-tracking toggle",
    "Backprop: In-place op unsafe warning",
    "Backprop: Kwargs pass-through",
    "Backprop: Linear affine on custom Tensor",
    "Backprop: Module base class custom",
    "Backprop: Parameter subclasses Tensor",
    "Backprop: Parameter wrap around Tensor",
    "Backprop: Parents dict by argidx",
    "Backprop: Recipe dataclass",
    "Backprop: Sorted computation graph",
    "Backprop: Unbox Tensor args to array",
    "Backprop: Unbroadcast pattern",
    "Backprop: add/sub/div back as lambdas",
    "Backprop: back fn call with recipe args",
    "Backprop: backprop pop-outgrad loop",
    "Backprop: backward fn signature",
    "Backprop: cycle detection via temp set",
    "Backprop: dispatch back fn from recipe",
    "Backprop: end-grad ones_like default",
    "Backprop: exp_back",
    "Backprop: get_children callable param",
    "Backprop: getitem_back via add-at",
    "Backprop: grad expressed in out",
    "Backprop: grads dict accumulate parents",
    "Backprop: is_differentiable flag",
    "Backprop: leaf tensor condition",
    "Backprop: log_back",
    "Backprop: manual chain forward-and-back",
    "Backprop: matmul_back transpose pair",
    "Backprop: max_back with tied half-mass",
    "Backprop: multiply_back",
    "Backprop: negative_back",
    "Backprop: no_grad ctx-mgr update",
    "Backprop: non-differentiable fn wrap",
    "Backprop: permute_back via argsort",
    "Backprop: register back fn",
    "Backprop: requires_grad propagation",
    "Backprop: reshape_back",
    "Backprop: sum/broadcast duality",
    "Backprop: sum_back via expand_broadcast",
    "Backprop: wrap forward fn",
    "CNN: 1-D conv windowing",
    "CNN: 1x1 conv channel-reshape",
    "CNN: 2-D conv windowing",
    "CNN: AvgPool as reduce",
    "CNN: BatchNorm affine params",
    "CNN: BatchNorm running stats",
    "CNN: BlockGroup stack",
    "CNN: Channel-axis sum semantics",
    "CNN: Conv output shape",
    "CNN: Conv zero padding",
    "CNN: ConvT as flipped padded conv",
    "CNN: ConvT fractional-stride zero insertion",
    "CNN: ConvT kernel axis swap",
    "CNN: ConvT padding amount formula",
    "CNN: Encoder-decoder symmetric layout",
    "CNN: Kernel shape (OC, IC, KH, KW)",
    "CNN: MaxPool as reduce",
    "CNN: No-ReLU on final layer",
    "CNN: ReLU as elementwise max",
    "CNN: ResNet stem block",
    "CNN: Residual skip-connection add",
    "CNN: Stride downsample arithmetic",
    "Config: @dataclass training args",
    "Config: Optimizer class dispatch",
    "Config: dataclasses.replace args",
    "Config: hparam precedence merge",
    "Config: nested param-group loop",
    "Config: param-group dict list",
    "Config: params iterable vs groups",
    "Config: sweep hparam distribution",
    "Config: wandb sweep config dict",
    "Distributed: DistributedSampler shard",
    "Distributed: all_reduce composition",
    "Distributed: all_reduce eval metrics",
    "Distributed: all_reduce grad sync",
    "Distributed: broadcast initial weights",
    "Distributed: dist.send/recv pair",
    "Distributed: init_process_group nccl",
    "Distributed: model save state_dict rank-0",
    "Distributed: mp.spawn workers",
    "Distributed: per-rank cuda device",
    "Distributed: rank-0-only side effects",
    "Distributed: rank/world_size args",
    "Distributed: reduce-op mean divide",
    "Distributed: reduce.gather + sum",
    "Einops: Rearrange as nn.Sequential layer",
    "Einops: Rearrange-as-flatten",
    "Einops: Reduce with min",
    "Einops: Repeat-as-broadcast",
    "Einsum: Index contraction semantics",
    "Eval: argmax accuracy",
    "Eval: topk predictions",
    "GAN: BCE log loss real/fake",
    "GAN: BN weight=1 bias=0 init",
    "GAN: Conv+LeakyReLU discriminator block",
    "GAN: ConvT+BN+Activation block",
    "GAN: DCGAN normal init 0.02",
    "GAN: Discriminator classifier head",
    "GAN: Generator loss to fool D",
    "GAN: Generator project + reshape",
    "GAN: Noise batch from latent_dim",
    "GAN: Two-optimizers alternating step",
    "GAN: channel-list reverse build",
    "GAN: detach stop-gradient trick",
    "GAN: model.modules() isinstance dispatch",
    "GAN: model.train/eval toggle around sample",
    "Generative: Bottleneck latent projection",
    "Generative: Broadcast source fan-out",
    "Generative: DCGAN netG+netD wrapper",
    "Generative: Hold-out one-per-class data",
    "Generative: MSE reconstruction loss",
    "Generative: randn-like noise source",
    "Generative: requires_grad leaf assert",
    "Generative: torch.stack trajectory",
    "Geometry: Barycentric coords",
    "Geometry: Cross-product surface normal",
    "Geometry: Ray parametric form",
    "Geometry: Rotation matrix 3-D (full)",
    "Geometry: Segment-line intersect 2-D",
    "Init: Kaiming uniform",
    "Init: Kaiming uniform SF init",
    "LinAlg: try/except solve",
    "Logging: log-samples eval callback",
    "Logging: time-stage instrumentation",
    "Logging: tqdm postfix metrics",
    "Logging: wandb.config into args",
    "Logging: wandb.finish",
    "Logging: wandb.init run",
    "Logging: wandb.log step",
    "Logging: wandb.watch model",
    "Loss: Cross-entropy classification",
    "Loss: arange fancy-index cross-entropy",
    "Loss: logsumexp cross-entropy",
    "Numerical: sqrt-eps stabilization",
    "Numpy: Applied patterns and advanced",
    "Numpy: Boolean mask combine",
    "Numpy: Core array literacy",
    "Numpy: Diagonal via strides",
    "Numpy: Indexing and selection",
    "Numpy: Inf-fill masking trick",
    "Numpy: Singular matrix mask trick",
    "Numpy: Vectorization and broadcasting",
    "Numpy: any() reduce along axis",
    "Numpy: matmul 2-D",
    "Optimizer: Adam EMA first moment",
    "Optimizer: Adam EMA second moment",
    "Optimizer: Adam bias-correction divide",
    "Optimizer: Momentum buffer",
    "Optimizer: Per-param state buffers",
    "Optimizer: SGD vanilla from scratch",
    "Optimizer: Weight decay L2",
    "Optimizer: __repr__ string",
    "Optimizer: clip_grad_norm pre-step",
    "Optimizer: decoupled weight decay (AdamW)",
    "Optimizer: optimizer.step loop over params",
    "PyTorch: Batched linalg.solve",
    "PyTorch: Conditional hparam branch",
    "PyTorch: Contiguous layout",
    "PyTorch: DataLoader batching",
    "PyTorch: DataLoader pin_memory + workers",
    "PyTorch: Device-consistent tensor construction",
    "PyTorch: In-place param update",
    "PyTorch: Inference mode step",
    "PyTorch: Module __repr__",
    "PyTorch: Module composition",
    "PyTorch: Optimizer init",
    "PyTorch: Slice view mutation",
    "PyTorch: Training step cycle",
    "PyTorch: Unbind tuple-unpack",
    "PyTorch: Zero-stride broadcasting",
    "PyTorch: __rmul__ scalar/tensor mix",
    "PyTorch: as_strided windowing",
    "PyTorch: backward()",
    "PyTorch: detach + clone snapshot",
    "PyTorch: freeze via requires_grad=False",
    "PyTorch: functional module wrap",
    "PyTorch: in-place buffer copy",
    "PyTorch: index by tensor",
    "PyTorch: linspace out= param",
    "PyTorch: loss.item() scalar extract",
    "PyTorch: matrix-vector product",
    "PyTorch: nn.Module subclassing",
    "PyTorch: nn.Parameter",
    "PyTorch: no_grad validation",
    "PyTorch: param.grad access",
    "PyTorch: register_buffer",
    "PyTorch: reshape vs view",
    "PyTorch: stack vs cat",
    "PyTorch: tensor from ndarray",
    "PyTorch: tensor.to(device)",
    "PyTorch: torch.cuda.empty_cache",
    "PyTorch: train/eval mode",
    "PyTorch: vector normalize keepdim",
    "PyTorch: where to clip negative",
    "PyTorch: zero_grad",
    "Trainer: Trainer class skeleton",
    "Trainer: examples-seen step axis",
    "Trainer: step counter increment",
    "Trainer: subclass extend pattern",
    "Transfer: Replace final head",
    "Transfer: state_dict load",
    "VAE: ELBO loss sum with beta",
    "VAE: KL divergence Gaussian closed-form",
    "VAE: Reparameterization trick",
    "VAE: mu+logsigma encoder head"
  ]
};
