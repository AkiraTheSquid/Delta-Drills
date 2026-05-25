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
    "arg-position-back-functions": "Backprop: Arg-position back funcs",
    "as-strided-noncontig-source": "Numpy: Applied patterns and advanced",
    "as-strided-windowing": "PyTorch: as_strided windowing",
    "backward-fn-signature": "Backprop: backward fn signature",
    "backward-on-scalar-loss": "PyTorch: backward()",
    "boolean-mask-combine": "Numpy: Boolean mask combine",
    "boolean-mask-identity-replace": "Numpy: Indexing and selection",
    "broadcasting-rules": "Numpy: Vectorization and broadcasting",
    "buffer-copy_-inplace": "PyTorch: in-place buffer copy",
    "chain-rule-elementwise": "Backprop: Elementwise chain rule",
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
    "dataloader-batching": "PyTorch: DataLoader batching",
    "device-consistent-construct": "PyTorch: Device-consistent tensor construction",
    "einops-einsum": "Einops: Deep Learning",
    "einops-rearrange": "Einops: Rearrange",
    "einops-rearrange-flatten": "Einops: Rearrange-as-flatten",
    "einops-reduce": "Einops: Reduce",
    "einops-repeat": "Einops: Repeat",
    "einops-repeat-broadcast": "Einops: Repeat-as-broadcast",
    "einsum-contraction": "Einsum: Index contraction semantics",
    "ema-second-moment": "Optimizer: Adam EMA second moment",
    "encoder-decoder-symmetric": "CNN: Encoder-decoder symmetric layout",
    "grad-tracking-global-toggle": "Backprop: Grad-tracking toggle",
    "inf-masking": "Numpy: Inf-fill masking trick",
    "inference-mode-step": "PyTorch: Inference mode step",
    "inplace-param-update": "PyTorch: In-place param update",
    "kaiming-uniform-init": "Init: Kaiming uniform",
    "kwargs-pass-through-recipe": "Backprop: Kwargs pass-through",
    "linalg-solve-batched": "PyTorch: Batched linalg.solve",
    "loss-item-scalar-extract": "PyTorch: loss.item() scalar extract",
    "module-composition": "PyTorch: Module composition",
    "module-extra-repr": "PyTorch: Module __repr__",
    "momentum-buffer-update": "Optimizer: Momentum buffer",
    "nn-module-subclass": "PyTorch: nn.Module subclassing",
    "nn-parameter-wrap": "PyTorch: nn.Parameter",
    "no-relu-on-final-layer": "CNN: No-ReLU on final layer",
    "optimizer-init-params-list": "PyTorch: Optimizer init",
    "optimizer-state-tensor-buffers": "Optimizer: Per-param state buffers",
    "param-grad-access": "PyTorch: param.grad access",
    "parents-dict-by-argidx": "Backprop: Parents dict by argidx",
    "ray-parametric-form": "Geometry: Ray parametric form",
    "rearrange-as-sequential-layer": "Einops: Rearrange as nn.Sequential layer",
    "recipe-dataclass": "Backprop: Recipe dataclass",
    "register-back-fn-after-wrap": "Backprop: register back fn",
    "relu-elementwise-max": "CNN: ReLU as elementwise max",
    "requires-grad-propagation": "Backprop: requires_grad propagation",
    "rotation-matrix-3d-y-axis": "Numpy: Applied patterns and advanced",
    "singular-matrix-mask-trick": "Numpy: Singular matrix mask trick",
    "slice-view-mutation": "PyTorch: Slice view mutation",
    "sqrt-eps-stabilize": "Numerical: sqrt-eps stabilization",
    "stack-vs-cat": "PyTorch: stack vs cat",
    "stride-zero-broadcast": "PyTorch: Zero-stride broadcasting",
    "tensor-item-scalar": "Numpy: Core array literacy",
    "tensor-to-device": "PyTorch: tensor.to(device)",
    "tensor-unbind": "Numpy: Indexing and selection",
    "tensor-wraps-ndarray": "PyTorch: tensor from ndarray",
    "tensor-zeros-init": "Numpy: Core array literacy",
    "train-eval-mode-branch": "PyTorch: train/eval mode",
    "training-step-cycle": "PyTorch: Training step cycle",
    "triangle-barycentric": "Geometry: Barycentric coords",
    "unbind-tuple-unpack": "PyTorch: Unbind tuple-unpack",
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
    "Backprop: Elementwise chain rule": [
      "chain-rule-elementwise"
    ],
    "Backprop: Grad-tracking toggle": [
      "grad-tracking-global-toggle"
    ],
    "Backprop: Kwargs pass-through": [
      "kwargs-pass-through-recipe"
    ],
    "Backprop: Parents dict by argidx": [
      "parents-dict-by-argidx"
    ],
    "Backprop: Recipe dataclass": [
      "recipe-dataclass"
    ],
    "Backprop: Unbroadcast pattern": [
      "unbroadcast-pattern"
    ],
    "Backprop: backward fn signature": [
      "backward-fn-signature"
    ],
    "Backprop: register back fn": [
      "register-back-fn-after-wrap"
    ],
    "Backprop: requires_grad propagation": [
      "requires-grad-propagation"
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
    "CNN: ConvT kernel axis swap": [
      "convT-kernel-axis-swap"
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
    "Geometry: Barycentric coords": [
      "triangle-barycentric"
    ],
    "Geometry: Ray parametric form": [
      "ray-parametric-form"
    ],
    "Init: Kaiming uniform": [
      "kaiming-uniform-init"
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
    "Optimizer: Adam EMA second moment": [
      "ema-second-moment"
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
    ]
  },
  "prereq_subtopic_keys": [
    "Backprop: Arg-position back funcs",
    "Backprop: Elementwise chain rule",
    "Backprop: Grad-tracking toggle",
    "Backprop: Kwargs pass-through",
    "Backprop: Parents dict by argidx",
    "Backprop: Recipe dataclass",
    "Backprop: Unbroadcast pattern",
    "Backprop: backward fn signature",
    "Backprop: register back fn",
    "Backprop: requires_grad propagation",
    "Backprop: wrap forward fn",
    "CNN: 1-D conv windowing",
    "CNN: 2-D conv windowing",
    "CNN: Channel-axis sum semantics",
    "CNN: Conv output shape",
    "CNN: Conv zero padding",
    "CNN: ConvT as flipped padded conv",
    "CNN: ConvT kernel axis swap",
    "CNN: Encoder-decoder symmetric layout",
    "CNN: Kernel shape (OC, IC, KH, KW)",
    "CNN: No-ReLU on final layer",
    "CNN: ReLU as elementwise max",
    "CNN: Stride downsample arithmetic",
    "Einops: Rearrange as nn.Sequential layer",
    "Einops: Rearrange-as-flatten",
    "Einops: Repeat-as-broadcast",
    "Einsum: Index contraction semantics",
    "Geometry: Barycentric coords",
    "Geometry: Ray parametric form",
    "Init: Kaiming uniform",
    "Numerical: sqrt-eps stabilization",
    "Numpy: Applied patterns and advanced",
    "Numpy: Boolean mask combine",
    "Numpy: Core array literacy",
    "Numpy: Indexing and selection",
    "Numpy: Inf-fill masking trick",
    "Numpy: Singular matrix mask trick",
    "Numpy: Vectorization and broadcasting",
    "Optimizer: Adam EMA second moment",
    "Optimizer: Momentum buffer",
    "Optimizer: Per-param state buffers",
    "Optimizer: Weight decay L2",
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
    "PyTorch: zero_grad"
  ]
};
