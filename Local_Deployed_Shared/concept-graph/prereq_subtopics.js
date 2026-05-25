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
    "all-reduce-compose": "Distributed: all_reduce composition",
    "all-reduce-grad-sync": "Distributed: all_reduce grad sync",
    "arg-position-back-functions": "Backprop: Arg-position back funcs",
    "argmax-accuracy-eval": "Eval: argmax accuracy",
    "as-strided-noncontig-source": "Numpy: Applied patterns and advanced",
    "as-strided-windowing": "PyTorch: as_strided windowing",
    "avgpool-reduce": "CNN: AvgPool as reduce",
    "backward-fn-signature": "Backprop: backward fn signature",
    "backward-func-lookup": "Backprop: BackwardFuncLookup",
    "backward-on-scalar-loss": "PyTorch: backward()",
    "batchnorm-affine-params": "CNN: BatchNorm affine params",
    "bias-correction-divide": "Optimizer: Adam bias-correction divide",
    "block-group-stack": "CNN: BlockGroup stack",
    "boolean-mask-combine": "Numpy: Boolean mask combine",
    "boolean-mask-identity-replace": "Numpy: Indexing and selection",
    "box-array-to-tensor-with-recipe": "Backprop: Box array as Tensor + recipe",
    "broadcasting-rules": "Numpy: Vectorization and broadcasting",
    "buffer-copy_-inplace": "PyTorch: in-place buffer copy",
    "chain-rule-elementwise": "Backprop: Elementwise chain rule",
    "coerce-float-arg-to-array": "Backprop: Coerce float arg to array",
    "conditional-hparam-branch": "PyTorch: Conditional hparam branch",
    "contiguous-layout": "PyTorch: Contiguous layout",
    "conv-channel-sum": "CNN: Channel-axis sum semantics",
    "conv-kernel-shape": "CNN: Kernel shape (OC, IC, KH, KW)",
    "conv-output-shape": "CNN: Conv output shape",
    "conv-padding-zero": "CNN: Conv zero padding",
    "conv-stride-downsample": "CNN: Stride downsample arithmetic",
    "conv-windowing-1d": "CNN: 1-D conv windowing",
    "conv-windowing-2d": "CNN: 2-D conv windowing",
    "convT-as-flipped-padded-conv": "CNN: ConvT as flipped padded conv",
    "convT-kernel-axis-swap": "CNN: ConvT kernel axis swap",
    "cross-entropy-classification-loss": "Loss: Cross-entropy classification",
    "dataloader-batching": "PyTorch: DataLoader batching",
    "device-consistent-construct": "PyTorch: Device-consistent tensor construction",
    "diagonal-via-strides": "Numpy: Diagonal via strides",
    "dist-send-recv-pair": "Distributed: dist.send/recv pair",
    "einops-einsum": "Einops: Deep Learning",
    "einops-rearrange": "Einops: Rearrange",
    "einops-rearrange-flatten": "Einops: Rearrange-as-flatten",
    "einops-reduce": "Einops: Reduce",
    "einops-repeat": "Einops: Repeat",
    "einops-repeat-broadcast": "Einops: Repeat-as-broadcast",
    "einsum-contraction": "Einsum: Index contraction semantics",
    "ema-first-moment": "Optimizer: Adam EMA first moment",
    "ema-second-moment": "Optimizer: Adam EMA second moment",
    "encoder-decoder-symmetric": "CNN: Encoder-decoder symmetric layout",
    "end-grad-default-ones-like": "Backprop: end-grad ones_like default",
    "examples-seen-step-axis": "Trainer: examples-seen step axis",
    "fractional-stride-zero-insertion": "CNN: ConvT fractional-stride zero insertion",
    "freeze-requires-grad": "PyTorch: freeze via requires_grad=False",
    "get-children-callable-param": "Backprop: get_children callable param",
    "grad-accumulate-on-leaf": "Backprop: Grad accumulate on leaf",
    "grad-tracking-global-toggle": "Backprop: Grad-tracking toggle",
    "inf-masking": "Numpy: Inf-fill masking trick",
    "inference-mode-step": "PyTorch: Inference mode step",
    "init-process-group-nccl": "Distributed: init_process_group nccl",
    "inplace-op-unsafe-warning": "Backprop: In-place op unsafe warning",
    "inplace-param-update": "PyTorch: In-place param update",
    "is-differentiable-flag": "Backprop: is_differentiable flag",
    "kaiming-uniform-init": "Init: Kaiming uniform",
    "kwargs-pass-through-recipe": "Backprop: Kwargs pass-through",
    "linalg-solve-batched": "PyTorch: Batched linalg.solve",
    "log-back": "Backprop: log_back",
    "loss-item-scalar-extract": "PyTorch: loss.item() scalar extract",
    "matmul-2d": "Numpy: matmul 2-D",
    "max-back-tied-half": "Backprop: max_back with tied half-mass",
    "module-composition": "PyTorch: Module composition",
    "module-extra-repr": "PyTorch: Module __repr__",
    "momentum-buffer-update": "Optimizer: Momentum buffer",
    "mp-spawn-workers": "Distributed: mp.spawn workers",
    "multiply-back": "Backprop: multiply_back",
    "nn-module-subclass": "PyTorch: nn.Module subclassing",
    "nn-parameter-wrap": "PyTorch: nn.Parameter",
    "no-relu-on-final-layer": "CNN: No-ReLU on final layer",
    "non-diff-fn-wrap": "Backprop: non-differentiable fn wrap",
    "optimizer-init-params-list": "PyTorch: Optimizer init",
    "optimizer-loop-on-tensor": "Optimizer: optimizer.step loop over params",
    "optimizer-state-tensor-buffers": "Optimizer: Per-param state buffers",
    "padding-amount-formula-convT": "CNN: ConvT padding amount formula",
    "param-grad-access": "PyTorch: param.grad access",
    "parameter-subclass-of-tensor": "Backprop: Parameter subclasses Tensor",
    "parents-dict-by-argidx": "Backprop: Parents dict by argidx",
    "per-rank-cuda-device": "Distributed: per-rank cuda device",
    "rank-world-size-args": "Distributed: rank/world_size args",
    "ray-parametric-form": "Geometry: Ray parametric form",
    "rearrange-as-sequential-layer": "Einops: Rearrange as nn.Sequential layer",
    "recipe-dataclass": "Backprop: Recipe dataclass",
    "reduce-op-mean-divide": "Distributed: reduce-op mean divide",
    "register-back-fn-after-wrap": "Backprop: register back fn",
    "relu-elementwise-max": "CNN: ReLU as elementwise max",
    "requires-grad-propagation": "Backprop: requires_grad propagation",
    "rotation-matrix-3d-y-axis": "Numpy: Applied patterns and advanced",
    "singular-matrix-mask-trick": "Numpy: Singular matrix mask trick",
    "slice-view-mutation": "PyTorch: Slice view mutation",
    "sorted-computational-graph": "Backprop: Sorted computation graph",
    "sqrt-eps-stabilize": "Numerical: sqrt-eps stabilization",
    "stack-vs-cat": "PyTorch: stack vs cat",
    "step-counter-increment": "Trainer: step counter increment",
    "stride-zero-broadcast": "PyTorch: Zero-stride broadcasting",
    "sum-and-broadcast-duality": "Backprop: sum/broadcast duality",
    "tensor-item-scalar": "Numpy: Core array literacy",
    "tensor-to-device": "PyTorch: tensor.to(device)",
    "tensor-unbind": "Numpy: Indexing and selection",
    "tensor-wraps-ndarray": "PyTorch: tensor from ndarray",
    "tensor-zeros-init": "Numpy: Core array literacy",
    "train-eval-mode-branch": "PyTorch: train/eval mode",
    "trainer-class-skeleton": "Trainer: Trainer class skeleton",
    "training-step-cycle": "PyTorch: Training step cycle",
    "triangle-barycentric": "Geometry: Barycentric coords",
    "unbind-tuple-unpack": "PyTorch: Unbind tuple-unpack",
    "unbox-args-tensor-to-array": "Backprop: Unbox Tensor args to array",
    "unbroadcast-pattern": "Backprop: Unbroadcast pattern",
    "validation-no-grad": "PyTorch: no_grad validation",
    "weight-decay-l2-add": "Optimizer: Weight decay L2",
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
    "Backprop: Parameter subclasses Tensor": [
      "parameter-subclass-of-tensor"
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
    "Backprop: backward fn signature": [
      "backward-fn-signature"
    ],
    "Backprop: end-grad ones_like default": [
      "end-grad-default-ones-like"
    ],
    "Backprop: get_children callable param": [
      "get-children-callable-param"
    ],
    "Backprop: is_differentiable flag": [
      "is-differentiable-flag"
    ],
    "Backprop: log_back": [
      "log-back"
    ],
    "Backprop: max_back with tied half-mass": [
      "max-back-tied-half"
    ],
    "Backprop: multiply_back": [
      "multiply-back"
    ],
    "Backprop: non-differentiable fn wrap": [
      "non-diff-fn-wrap"
    ],
    "Backprop: register back fn": [
      "register-back-fn-after-wrap"
    ],
    "Backprop: requires_grad propagation": [
      "requires-grad-propagation"
    ],
    "Backprop: sum/broadcast duality": [
      "sum-and-broadcast-duality"
    ],
    "Backprop: wrap forward fn": [
      "wrap-forward-fn-generic"
    ],
    "CNN: 1-D conv windowing": [
      "conv-windowing-1d"
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
    "CNN: No-ReLU on final layer": [
      "no-relu-on-final-layer"
    ],
    "CNN: ReLU as elementwise max": [
      "relu-elementwise-max"
    ],
    "CNN: Stride downsample arithmetic": [
      "conv-stride-downsample"
    ],
    "Distributed: all_reduce composition": [
      "all-reduce-compose"
    ],
    "Distributed: all_reduce grad sync": [
      "all-reduce-grad-sync"
    ],
    "Distributed: dist.send/recv pair": [
      "dist-send-recv-pair"
    ],
    "Distributed: init_process_group nccl": [
      "init-process-group-nccl"
    ],
    "Distributed: mp.spawn workers": [
      "mp-spawn-workers"
    ],
    "Distributed: per-rank cuda device": [
      "per-rank-cuda-device"
    ],
    "Distributed: rank/world_size args": [
      "rank-world-size-args"
    ],
    "Distributed: reduce-op mean divide": [
      "reduce-op-mean-divide"
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
    "Geometry: Barycentric coords": [
      "triangle-barycentric"
    ],
    "Geometry: Ray parametric form": [
      "ray-parametric-form"
    ],
    "Init: Kaiming uniform": [
      "kaiming-uniform-init"
    ],
    "Loss: Cross-entropy classification": [
      "cross-entropy-classification-loss"
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
    "Optimizer: Weight decay L2": [
      "weight-decay-l2-add"
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
    "PyTorch: as_strided windowing": [
      "as-strided-windowing"
    ],
    "PyTorch: backward()": [
      "backward-on-scalar-loss"
    ],
    "PyTorch: freeze via requires_grad=False": [
      "freeze-requires-grad"
    ],
    "PyTorch: in-place buffer copy": [
      "buffer-copy_-inplace"
    ],
    "PyTorch: loss.item() scalar extract": [
      "loss-item-scalar-extract"
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
    "PyTorch: stack vs cat": [
      "stack-vs-cat"
    ],
    "PyTorch: tensor from ndarray": [
      "tensor-wraps-ndarray"
    ],
    "PyTorch: tensor.to(device)": [
      "tensor-to-device"
    ],
    "PyTorch: train/eval mode": [
      "train-eval-mode-branch"
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
    ]
  },
  "prereq_subtopic_keys": [
    "Backprop: Arg-position back funcs",
    "Backprop: BackwardFuncLookup",
    "Backprop: Box array as Tensor + recipe",
    "Backprop: Coerce float arg to array",
    "Backprop: Elementwise chain rule",
    "Backprop: Grad accumulate on leaf",
    "Backprop: Grad-tracking toggle",
    "Backprop: In-place op unsafe warning",
    "Backprop: Kwargs pass-through",
    "Backprop: Parameter subclasses Tensor",
    "Backprop: Parents dict by argidx",
    "Backprop: Recipe dataclass",
    "Backprop: Sorted computation graph",
    "Backprop: Unbox Tensor args to array",
    "Backprop: Unbroadcast pattern",
    "Backprop: backward fn signature",
    "Backprop: end-grad ones_like default",
    "Backprop: get_children callable param",
    "Backprop: is_differentiable flag",
    "Backprop: log_back",
    "Backprop: max_back with tied half-mass",
    "Backprop: multiply_back",
    "Backprop: non-differentiable fn wrap",
    "Backprop: register back fn",
    "Backprop: requires_grad propagation",
    "Backprop: sum/broadcast duality",
    "Backprop: wrap forward fn",
    "CNN: 1-D conv windowing",
    "CNN: 2-D conv windowing",
    "CNN: AvgPool as reduce",
    "CNN: BatchNorm affine params",
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
    "CNN: No-ReLU on final layer",
    "CNN: ReLU as elementwise max",
    "CNN: Stride downsample arithmetic",
    "Distributed: all_reduce composition",
    "Distributed: all_reduce grad sync",
    "Distributed: dist.send/recv pair",
    "Distributed: init_process_group nccl",
    "Distributed: mp.spawn workers",
    "Distributed: per-rank cuda device",
    "Distributed: rank/world_size args",
    "Distributed: reduce-op mean divide",
    "Einops: Rearrange as nn.Sequential layer",
    "Einops: Rearrange-as-flatten",
    "Einops: Repeat-as-broadcast",
    "Einsum: Index contraction semantics",
    "Eval: argmax accuracy",
    "Geometry: Barycentric coords",
    "Geometry: Ray parametric form",
    "Init: Kaiming uniform",
    "Loss: Cross-entropy classification",
    "Numerical: sqrt-eps stabilization",
    "Numpy: Applied patterns and advanced",
    "Numpy: Boolean mask combine",
    "Numpy: Core array literacy",
    "Numpy: Diagonal via strides",
    "Numpy: Indexing and selection",
    "Numpy: Inf-fill masking trick",
    "Numpy: Singular matrix mask trick",
    "Numpy: Vectorization and broadcasting",
    "Numpy: matmul 2-D",
    "Optimizer: Adam EMA first moment",
    "Optimizer: Adam EMA second moment",
    "Optimizer: Adam bias-correction divide",
    "Optimizer: Momentum buffer",
    "Optimizer: Per-param state buffers",
    "Optimizer: Weight decay L2",
    "Optimizer: optimizer.step loop over params",
    "PyTorch: Batched linalg.solve",
    "PyTorch: Conditional hparam branch",
    "PyTorch: Contiguous layout",
    "PyTorch: DataLoader batching",
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
    "PyTorch: as_strided windowing",
    "PyTorch: backward()",
    "PyTorch: freeze via requires_grad=False",
    "PyTorch: in-place buffer copy",
    "PyTorch: loss.item() scalar extract",
    "PyTorch: nn.Module subclassing",
    "PyTorch: nn.Parameter",
    "PyTorch: no_grad validation",
    "PyTorch: param.grad access",
    "PyTorch: stack vs cat",
    "PyTorch: tensor from ndarray",
    "PyTorch: tensor.to(device)",
    "PyTorch: train/eval mode",
    "PyTorch: zero_grad",
    "Trainer: Trainer class skeleton",
    "Trainer: examples-seen step axis",
    "Trainer: step counter increment"
  ]
};
