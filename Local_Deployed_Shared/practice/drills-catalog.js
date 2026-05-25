/* ================================================================
   PROCEDURAL DRILLS CATALOG — standalone per-exercise notebooks

   Each entry surfaces a single-exercise Colab notebook hosted under
   arena-procedural-drills/<topic>/<atom>/NN-<slug>.ipynb. Catalog is
   metadata only — UI surfaces (ArenaUnlock auto-fire + Targeted
   Practice search) consume it.

   Each entry:
     - id          stable catalog key. Format: `drill:<atom>:ex<N>`.
     - title       atom-prefixed short label shown in search rows.
     - sub         second-line subtitle.
     - heading     EXACT top-level markdown heading from the notebook.
                   ArenaUnlock auto-copies this on Open in Colab so the
                   student pastes-searches inside Colab and lands on
                   the right cell.
     - notebookPath repo-relative path; routed via colabUpstreamHref to
                    <account_github_username>/Delta-Drills fork.
     - subtopics   adaptive-state subtopic key(s) the standalone bumps
                   via /api/practice/arena-rating. All splits of the
                   same parent atom share the parent's subtopic so EWMA
                   aggregates per atom (not per individual exercise).
     - targetSeconds wall-clock budget for the arena-unlock timer.
                     Single exercises get 180-300s; long capstones get
                     a bit more.

   ================================================================ */

(function () {
  // Default unlock floor: when EWMA accuracy on the targeted subtopic
  // crosses this percent, the standalone exercise becomes eligible for
  // auto-surface. Per-exercise overrides supported via E() opts arg.
  const DEFAULT_UNLOCK_MIN_PCT = 50;

  // Default per-exercise timer budget (single-exercise standalones are
  // shorter than the old combined 5-exercise drills).
  const DEFAULT_TARGET_SECONDS = 240;

  const E = (atom, exIdx, exTitle, subtopic, notebookPath, heading, opts = {}) => ({
    id: `drill:${atom}:ex${exIdx}`,
    title: `${atom} ex${exIdx}: ${exTitle}`,
    sub: `Procedural drill · ${subtopic}`,
    heading,
    notebookPath,
    subtopics: [subtopic],
    targetSeconds: opts.targetSeconds ?? DEFAULT_TARGET_SECONDS,
    unlockMinPct: opts.unlockMinPct ?? DEFAULT_UNLOCK_MIN_PCT,
    atomId: atom,
    exerciseIndex: exIdx,
    isDrill: true,
  });

  window.DRILLS_CATALOG = [
    // all-reduce-compose
    E("all-reduce-compose", 1, "compose all_reduce from reduce plus broadcast", "Distributed: all_reduce composition", "arena-procedural-drills/prereqs_distributed/all-reduce-compose/01-compose-all-reduce-from-reduce-plus-broadcast.ipynb", "all-reduce-compose — ex1: compose all_reduce from reduce plus broadcast"),
    // all-reduce-grad-sync
    E("all-reduce-grad-sync", 1, "synchronize gradients across ranks with all_reduce", "Distributed: all_reduce grad sync", "arena-procedural-drills/prereqs_distributed/all-reduce-grad-sync/01-synchronize-gradients-across-ranks-with-all-reduce.ipynb", "all-reduce-grad-sync — ex1: synchronize gradients across ranks with all_reduce"),
    // arg-position-back-functions
    E("arg-position-back-functions", 1, "write div_back0 and div_back1 — asymmetric per-arg back fns", "Backprop: Arg-position back funcs", "arena-procedural-drills/prereqs_autograd_internals/arg-position-back-functions/01-write-div-back0-and-div-back1-asymmetric-per-arg.ipynb", "arg-position-back-functions — ex1: write div_back0 and div_back1 — asymmetric per-arg back fns"),
    // argmax-accuracy-eval
    E("argmax-accuracy-eval", 1, "top-1 classification accuracy from logits", "Eval: argmax accuracy", "arena-procedural-drills/prereqs_adam_trainer/argmax-accuracy-eval/01-top-1-classification-accuracy-from-logits.ipynb", "argmax-accuracy-eval — ex1: top-1 classification accuracy from logits"),
    // as-strided-noncontig-source
    E("as-strided-noncontig-source", 1, "read the strides of a contiguous tensor", "Numpy: Applied patterns and advanced", "arena-procedural-drills/prereqs_numpy/as-strided-noncontig-source/01-read-the-strides-of-a-contiguous-tensor.ipynb", "as-strided-noncontig-source — ex1: read the strides of a contiguous tensor"),
    E("as-strided-noncontig-source", 2, "transpose breaks contiguity", "Numpy: Applied patterns and advanced", "arena-procedural-drills/prereqs_numpy/as-strided-noncontig-source/02-transpose-breaks-contiguity.ipynb", "as-strided-noncontig-source — ex2: transpose breaks contiguity"),
    E("as-strided-noncontig-source", 3, ".view() raises on non-contiguous input", "Numpy: Applied patterns and advanced", "arena-procedural-drills/prereqs_numpy/as-strided-noncontig-source/03-view-raises-on-non-contiguous-input.ipynb", "as-strided-noncontig-source — ex3: .view() raises on non-contiguous input"),
    E("as-strided-noncontig-source", 4, "fix .view() with .contiguous()", "Numpy: Applied patterns and advanced", "arena-procedural-drills/prereqs_numpy/as-strided-noncontig-source/04-fix-view-with-contiguous.ipynb", "as-strided-noncontig-source — ex4: fix .view() with .contiguous()"),
    E("as-strided-noncontig-source", 5, "rolling window via as_strided (zero-copy)", "Numpy: Applied patterns and advanced", "arena-procedural-drills/prereqs_numpy/as-strided-noncontig-source/05-rolling-window-via-as-strided-zero-copy.ipynb", "as-strided-noncontig-source — ex5: rolling window via as_strided (zero-copy)"),
    E("as-strided-noncontig-source", 6, "2-D sliding-window image patches + visualize", "Numpy: Applied patterns and advanced", "arena-procedural-drills/prereqs_numpy/as-strided-noncontig-source/06-2d-sliding-window-image-patches-visualize.ipynb", "as-strided-noncontig-source — ex6: 2-D sliding-window image patches + visualize"),
    E("as-strided-noncontig-source", 7, "memory cost: contiguous() vs view comparison table", "Numpy: Applied patterns and advanced", "arena-procedural-drills/prereqs_numpy/as-strided-noncontig-source/07-memory-cost-contiguous-vs-view-comparison-table.ipynb", "as-strided-noncontig-source — ex7: memory cost: contiguous() vs view comparison table"),
    E("as-strided-noncontig-source", 8, "1-D convolution via as_strided + einsum pipeline", "Numpy: Applied patterns and advanced", "arena-procedural-drills/prereqs_numpy/as-strided-noncontig-source/08-1d-convolution-via-as-strided-einsum-pipeline.ipynb", "as-strided-noncontig-source — ex8: 1-D convolution via as_strided + einsum pipeline"),
    E("as-strided-noncontig-source", 9, "diagonal extraction via stride manipulation", "Numpy: Applied patterns and advanced", "arena-procedural-drills/prereqs_numpy/as-strided-noncontig-source/09-diagonal-extraction-via-stride-manipulation.ipynb", "as-strided-noncontig-source — ex9: diagonal extraction via stride manipulation"),
    // as-strided-windowing
    E("as-strided-windowing", 1, "compute size + stride args for a 1-D sliding window", "PyTorch: as_strided windowing", "arena-procedural-drills/prereqs_tensor_mechanics/as-strided-windowing/01-compute-size-stride-for-1d-sliding-window.ipynb", "as-strided-windowing — ex1: compute size + stride args for a 1-D sliding window"),
    E("as-strided-windowing", 2, "batched + channelled windowing for conv1d input prep", "PyTorch: as_strided windowing", "arena-procedural-drills/prereqs_tensor_mechanics/as-strided-windowing/02-batched-channelled-windowing-for-conv1d-input-prep.ipynb", "as-strided-windowing — ex2: batched + channelled windowing for conv1d input prep"),
    // avgpool-reduce
    E("avgpool-reduce", 1, "build AvgPool2d via einops.reduce", "CNN: AvgPool as reduce", "arena-procedural-drills/prereqs_cnn_extras/avgpool-reduce/01-build-avgpool2d-via-einops-reduce.ipynb", "avgpool-reduce — ex1: build AvgPool2d via einops.reduce"),
    // backward-fn-signature
    E("backward-fn-signature", 1, "write log_back with the canonical (grad_out, out, x) signature", "Backprop: backward fn signature", "arena-procedural-drills/prereqs_backprop/backward-fn-signature/01-write-log-back-with-canonical-signature.ipynb", "backward-fn-signature — ex1: write log_back with the canonical (grad_out, out, x) signature"),
    E("backward-fn-signature", 2, "write negative_back and exp_back — back-fn signature, two ops", "Backprop: backward fn signature", "arena-procedural-drills/prereqs_backprop/backward-fn-signature/02-negative-and-exp-back-two-ops.ipynb", "backward-fn-signature — ex2: write negative_back and exp_back — back-fn signature, two ops"),
    // backward-func-lookup
    E("backward-func-lookup", 1, "implement BackwardFuncLookup with (fn, argnum) keys", "Backprop: BackwardFuncLookup", "arena-procedural-drills/prereqs_autograd_pt2/backward-func-lookup/01-implement-backward-func-lookup-with-fn-argnum-keys.ipynb", "backward-func-lookup — ex1: implement BackwardFuncLookup with (fn, argnum) keys"),
    // backward-on-scalar-loss
    E("backward-on-scalar-loss", 1, "reduce per-sample loss to a scalar before backward", "PyTorch: backward()", "arena-procedural-drills/prereqs_training_loop/backward-on-scalar-loss/01-reduce-per-sample-loss-to-a-scalar-before-backward.ipynb", "backward-on-scalar-loss — ex1: reduce per-sample loss to a scalar before backward"),
    // batchnorm-affine-params
    E("batchnorm-affine-params", 1, "apply BatchNorm's affine step to a normalized tensor", "CNN: BatchNorm affine params", "arena-procedural-drills/prereqs_cnn_extras/batchnorm-affine-params/01-apply-batchnorms-affine-step-to-a-normalized-tensor.ipynb", "batchnorm-affine-params — ex1: apply BatchNorm's affine step to a normalized tensor"),
    // bias-correction-divide
    E("bias-correction-divide", 1, "bias-correct an Adam moment: m_hat = m / (1 - beta**t)", "Optimizer: Adam bias-correction divide", "arena-procedural-drills/prereqs_adam_trainer/bias-correction-divide/01-bias-correct-an-adam-moment-m-hat-equals-m-over-one-minus-beta-to-the-t.ipynb", "bias-correction-divide — ex1: bias-correct an Adam moment: m_hat = m / (1 - beta**t)"),
    // block-group-stack
    E("block-group-stack", 1, "build a ResNet BlockGroup from toy blocks", "CNN: BlockGroup stack", "arena-procedural-drills/prereqs_cnn_extras/block-group-stack/01-build-a-resnet-blockgroup-from-toy-blocks.ipynb", "block-group-stack — ex1: build a ResNet BlockGroup from toy blocks"),
    // boolean-mask-combine
    E("boolean-mask-combine", 1, "five-predicate ray-triangle inside test", "Numpy: Boolean mask combine", "arena-procedural-drills/prereqs_einops_advanced/boolean-mask-combine/01-five-predicate-ray-triangle-inside-test.ipynb", "boolean-mask-combine — ex1: five-predicate ray-triangle inside test"),
    // boolean-mask-identity-replace
    E("boolean-mask-identity-replace", 1, "build a boolean mask from a comparison", "Numpy: Indexing and selection", "arena-procedural-drills/prereqs_numpy/boolean-mask-identity-replace/01-build-a-boolean-mask-from-a-comparison.ipynb", "boolean-mask-identity-replace — ex1: build a boolean mask from a comparison"),
    E("boolean-mask-identity-replace", 2, "replace negatives with zero (non-destructive)", "Numpy: Indexing and selection", "arena-procedural-drills/prereqs_numpy/boolean-mask-identity-replace/02-replace-negatives-with-zero-non-destructive.ipynb", "boolean-mask-identity-replace — ex2: replace negatives with zero (non-destructive)"),
    E("boolean-mask-identity-replace", 3, "zero out flagged rows of a 2-D tensor", "Numpy: Indexing and selection", "arena-procedural-drills/prereqs_numpy/boolean-mask-identity-replace/03-zero-out-flagged-rows-of-a-2-d-tensor.ipynb", "boolean-mask-identity-replace — ex3: zero out flagged rows of a 2-D tensor"),
    E("boolean-mask-identity-replace", 4, "substitute identity for flagged matrices", "Numpy: Indexing and selection", "arena-procedural-drills/prereqs_numpy/boolean-mask-identity-replace/04-substitute-identity-for-flagged-matrices.ipynb", "boolean-mask-identity-replace — ex4: substitute identity for flagged matrices"),
    E("boolean-mask-identity-replace", 5, "safe batched solve (singular → zero solution)", "Numpy: Indexing and selection", "arena-procedural-drills/prereqs_numpy/boolean-mask-identity-replace/05-safe-batched-solve-singular-zero-solution.ipynb", "boolean-mask-identity-replace — ex5: safe batched solve (singular → zero solution)"),
    E("boolean-mask-identity-replace", 6, "causal attention mask — build, apply, visualize", "Numpy: Indexing and selection", "arena-procedural-drills/prereqs_numpy/boolean-mask-identity-replace/06-causal-attention-mask-build-apply-visualize.ipynb", "boolean-mask-identity-replace — ex6: causal attention mask — build, apply, visualize"),
    E("boolean-mask-identity-replace", 7, "padded-sequence mean ignoring pad positions", "Numpy: Indexing and selection", "arena-procedural-drills/prereqs_numpy/boolean-mask-identity-replace/07-padded-sequence-mean-ignoring-pad-positions.ipynb", "boolean-mask-identity-replace — ex7: padded-sequence mean ignoring pad positions"),
    E("boolean-mask-identity-replace", 8, "outlier removal with combined boolean masks", "Numpy: Indexing and selection", "arena-procedural-drills/prereqs_numpy/boolean-mask-identity-replace/08-outlier-removal-with-combined-boolean-masks.ipynb", "boolean-mask-identity-replace — ex8: outlier removal with combined boolean masks"),
    E("boolean-mask-identity-replace", 9, "non-max suppression via iterative mask update", "Numpy: Indexing and selection", "arena-procedural-drills/prereqs_numpy/boolean-mask-identity-replace/09-non-max-suppression-via-iterative-mask-update.ipynb", "boolean-mask-identity-replace — ex9: non-max suppression via iterative mask update"),
    // box-array-to-tensor-with-recipe
    E("box-array-to-tensor-with-recipe", 1, "box raw output into MiniTensor + attach Recipe when grad-tracked", "Backprop: Box array as Tensor + recipe", "arena-procedural-drills/prereqs_autograd_pt3/box-array-to-tensor-with-recipe/01-box-raw-output-into-tensor-and-attach-recipe.ipynb", "box-array-to-tensor-with-recipe — ex1: box raw output into MiniTensor + attach Recipe when grad-tracked"),
    // broadcasting-rules
    E("broadcasting-rules", 1, "predict the broadcast shape", "Numpy: Vectorization and broadcasting", "arena-procedural-drills/prereqs_numpy/broadcasting-rules/01-predict-the-broadcast-shape.ipynb", "broadcasting-rules — ex1: predict the broadcast shape"),
    E("broadcasting-rules", 2, "row-vector broadcast (add bias to a batch)", "Numpy: Vectorization and broadcasting", "arena-procedural-drills/prereqs_numpy/broadcasting-rules/02-row-vector-broadcast-add-bias-to-a-batch.ipynb", "broadcasting-rules — ex2: row-vector broadcast (add bias to a batch)"),
    E("broadcasting-rules", 3, "column-vector broadcast (per-sample scale)", "Numpy: Vectorization and broadcasting", "arena-procedural-drills/prereqs_numpy/broadcasting-rules/03-column-vector-broadcast-per-sample-scale.ipynb", "broadcasting-rules — ex3: column-vector broadcast (per-sample scale)"),
    E("broadcasting-rules", 4, "insert a missing axis where broadcast fails", "Numpy: Vectorization and broadcasting", "arena-procedural-drills/prereqs_numpy/broadcasting-rules/04-insert-a-missing-axis-where-broadcast-fails.ipynb", "broadcasting-rules — ex4: insert a missing axis where broadcast fails"),
    E("broadcasting-rules", 5, "outer product via column×row broadcast", "Numpy: Vectorization and broadcasting", "arena-procedural-drills/prereqs_numpy/broadcasting-rules/05-outer-product-via-column-row-broadcast.ipynb", "broadcasting-rules — ex5: outer product via column×row broadcast"),
    E("broadcasting-rules", 6, "pairwise distance matrix as a heatmap", "Numpy: Vectorization and broadcasting", "arena-procedural-drills/prereqs_numpy/broadcasting-rules/06-pairwise-distance-matrix-as-a-heatmap.ipynb", "broadcasting-rules — ex6: pairwise distance matrix as a heatmap"),
    E("broadcasting-rules", 7, "batched attention scores with shape-trace debugging", "Numpy: Vectorization and broadcasting", "arena-procedural-drills/prereqs_numpy/broadcasting-rules/07-batched-attention-scores-with-shape-trace-debugging.ipynb", "broadcasting-rules — ex7: batched attention scores with shape-trace debugging"),
    E("broadcasting-rules", 8, "per-channel bias add to a 4-D image batch", "Numpy: Vectorization and broadcasting", "arena-procedural-drills/prereqs_numpy/broadcasting-rules/08-per-channel-bias-add-to-a-4-d-image-batch.ipynb", "broadcasting-rules — ex8: per-channel bias add to a 4-D image batch"),
    E("broadcasting-rules", 9, "silent-broadcast trap — catch it with a value check", "Numpy: Vectorization and broadcasting", "arena-procedural-drills/prereqs_numpy/broadcasting-rules/09-silent-broadcast-trap-catch-it-with-a-value-check.ipynb", "broadcasting-rules — ex9: silent-broadcast trap — catch it with a value check"),
    // buffer-copy_-inplace
    E("buffer-copy_-inplace", 1, "update a BatchNorm running_mean buffer with copy_", "PyTorch: in-place buffer copy", "arena-procedural-drills/prereqs_backprop/buffer-copy_-inplace/01-update-running-mean-buffer-with-copy.ipynb", "buffer-copy_-inplace — ex1: update a BatchNorm running_mean buffer with copy_"),
    // chain-rule-elementwise
    E("chain-rule-elementwise", 1, "write sigmoid_back and relu_back from the elementwise chain rule", "Backprop: Elementwise chain rule", "arena-procedural-drills/prereqs_autograd_internals/chain-rule-elementwise/01-write-sigmoid-and-relu-back-from-elementwise-chain-rule.ipynb", "chain-rule-elementwise — ex1: write sigmoid_back and relu_back from the elementwise chain rule"),
    // coerce-float-arg-to-array
    E("coerce-float-arg-to-array", 1, "coerce_to_array: wrap int/float as 0-D tensor, pass-through others", "Backprop: Coerce float arg to array", "arena-procedural-drills/prereqs_autograd_pt3/coerce-float-arg-to-array/01-coerce-float-or-int-arg-to-0d-tensor.ipynb", "coerce-float-arg-to-array — ex1: coerce_to_array: wrap int/float as 0-D tensor, pass-through others"),
    // conditional-hparam-branch
    E("conditional-hparam-branch", 1, "Linear with optional bias gated by use_bias flag", "PyTorch: Conditional hparam branch", "arena-procedural-drills/prereqs_numerical_modules/conditional-hparam-branch/01-linear-with-optional-bias-flag.ipynb", "conditional-hparam-branch — ex1: Linear with optional bias gated by use_bias flag"),
    // contiguous-layout
    E("contiguous-layout", 1, "predict-then-verify the strides of a 3-D contiguous tensor", "PyTorch: Contiguous layout", "arena-procedural-drills/prereqs_tensor_mechanics/contiguous-layout/01-predict-then-verify-strides-of-contiguous-3d.ipynb", "contiguous-layout — ex1: predict-then-verify the strides of a 3-D contiguous tensor"),
    E("contiguous-layout", 2, "fix the view-after-transpose error with .contiguous()", "PyTorch: Contiguous layout", "arena-procedural-drills/prereqs_tensor_mechanics/contiguous-layout/02-fix-view-after-transpose-with-contiguous.ipynb", "contiguous-layout — ex2: fix the view-after-transpose error with .contiguous()"),
    // conv-channel-sum
    E("conv-channel-sum", 1, "verify conv2d contracts the IC axis", "CNN: Channel-axis sum semantics", "arena-procedural-drills/prereqs_cnn_deep/conv-channel-sum/01-verify-conv2d-contracts-the-ic-axis.ipynb", "conv-channel-sum — ex1: verify conv2d contracts the IC axis"),
    // conv-kernel-shape
    E("conv-kernel-shape", 1, "introspect a conv2d weight tensor's axes", "CNN: Kernel shape (OC, IC, KH, KW)", "arena-procedural-drills/prereqs_cnn_deep/conv-kernel-shape/01-introspect-a-conv2d-weight-tensors-axes.ipynb", "conv-kernel-shape — ex1: introspect a conv2d weight tensor's axes"),
    // conv-output-shape
    E("conv-output-shape", 1, "compute conv2d output shape analytically", "CNN: Conv output shape", "arena-procedural-drills/prereqs_geometry_cnn/conv-output-shape/01-compute-conv2d-output-shape-analytically.ipynb", "conv-output-shape — ex1: compute conv2d output shape analytically"),
    // conv-padding-zero
    E("conv-padding-zero", 1, "build a zero-padded 1-D input by slice assignment", "CNN: Conv zero padding", "arena-procedural-drills/prereqs_geometry_cnn/conv-padding-zero/01-build-a-zero-padded-1d-input-by-slice-assignment.ipynb", "conv-padding-zero — ex1: build a zero-padded 1-D input by slice assignment"),
    // conv-stride-downsample
    E("conv-stride-downsample", 1, "predict strided conv output length", "CNN: Stride downsample arithmetic", "arena-procedural-drills/prereqs_cnn_deep/conv-stride-downsample/01-predict-strided-conv-output-length.ipynb", "conv-stride-downsample — ex1: predict strided conv output length"),
    // conv-windowing-1d
    E("conv-windowing-1d", 1, "build the 1-D conv window view via as_strided", "CNN: 1-D conv windowing", "arena-procedural-drills/prereqs_geometry_cnn/conv-windowing-1d/01-build-the-1d-conv-window-view-via-as-strided.ipynb", "conv-windowing-1d — ex1: build the 1-D conv window view via as_strided"),
    // conv-windowing-2d
    E("conv-windowing-2d", 1, "build the 2-D conv window view via as_strided", "CNN: 2-D conv windowing", "arena-procedural-drills/prereqs_cnn_deep/conv-windowing-2d/01-build-the-2d-conv-window-view-via-as-strided.ipynb", "conv-windowing-2d — ex1: build the 2-D conv window view via as_strided"),
    // convT-as-flipped-padded-conv
    E("convT-as-flipped-padded-conv", 1, "rebuild ConvTranspose2d as flipped padded Conv2d (stride 1)", "CNN: ConvT as flipped padded conv", "arena-procedural-drills/prereqs_cnn_deep/convT-as-flipped-padded-conv/01-rebuild-convtranspose2d-as-flipped-padded-conv2d.ipynb", "convT-as-flipped-padded-conv — ex1: rebuild ConvTranspose2d as flipped padded Conv2d (stride 1)"),
    // convT-kernel-axis-swap
    E("convT-kernel-axis-swap", 1, "compare Conv2d and ConvTranspose2d weight shapes", "CNN: ConvT kernel axis swap", "arena-procedural-drills/prereqs_cnn_deep/convT-kernel-axis-swap/01-compare-conv2d-and-convtranspose2d-weight-shapes.ipynb", "convT-kernel-axis-swap — ex1: compare Conv2d and ConvTranspose2d weight shapes"),
    // cross-entropy-classification-loss
    E("cross-entropy-classification-loss", 1, "manual cross-entropy matches F.cross_entropy on logits", "Loss: Cross-entropy classification", "arena-procedural-drills/prereqs_adam_trainer/cross-entropy-classification-loss/01-manual-cross-entropy-matches-f-cross-entropy-on-logits.ipynb", "cross-entropy-classification-loss — ex1: manual cross-entropy matches F.cross_entropy on logits"),
    // dataloader-batching
    E("dataloader-batching", 1, "wrap a TensorDataset in a DataLoader and iterate batches", "PyTorch: DataLoader batching", "arena-procedural-drills/prereqs_optimizer_internals/dataloader-batching/01-wrap-a-tensordataset-in-a-dataloader-and-iterate-batches.ipynb", "dataloader-batching — ex1: wrap a TensorDataset in a DataLoader and iterate batches"),
    // device-consistent-construct
    E("device-consistent-construct", 1, "build a Module that allocates scratch tensors with the right device + dtype", "PyTorch: Device-consistent tensor construction", "arena-procedural-drills/prereqs_numerical_modules/device-consistent-construct/01-device-consistent-scratch-allocation.ipynb", "device-consistent-construct — ex1: build a Module that allocates scratch tensors with the right device + dtype"),
    // diagonal-via-strides
    E("diagonal-via-strides", 1, "extract diagonal of (N, N) via as_strided", "Numpy: Diagonal via strides", "arena-procedural-drills/prereqs_cnn_extras/diagonal-via-strides/01-extract-diagonal-of-nn-via-as-strided.ipynb", "diagonal-via-strides — ex1: extract diagonal of (N, N) via as_strided"),
    // dist-send-recv-pair
    E("dist-send-recv-pair", 1, "implement broadcast via paired dist.send and dist.recv", "Distributed: dist.send/recv pair", "arena-procedural-drills/prereqs_distributed/dist-send-recv-pair/01-implement-broadcast-via-paired-dist-send-and-dist-recv.ipynb", "dist-send-recv-pair — ex1: implement broadcast via paired dist.send and dist.recv"),
    // einops-einsum
    E("einops-einsum", 1, "elementwise product (Hadamard)", "Einops: Deep Learning", "arena-procedural-drills/prereqs_einops/einops-einsum/01-elementwise-product-hadamard.ipynb", "einops-einsum — ex1: elementwise product (Hadamard)"),
    E("einops-einsum", 2, "matrix multiplication (single-index contraction)", "Einops: Deep Learning", "arena-procedural-drills/prereqs_einops/einops-einsum/02-matrix-multiplication-single-index-contraction.ipynb", "einops-einsum — ex2: matrix multiplication (single-index contraction)"),
    E("einops-einsum", 3, "row sum (omit-to-reduce)", "Einops: Deep Learning", "arena-procedural-drills/prereqs_einops/einops-einsum/03-row-sum-omit-to-reduce.ipynb", "einops-einsum — ex3: row sum (omit-to-reduce)"),
    E("einops-einsum", 4, "batched matrix multiply (preserve a batch axis)", "Einops: Deep Learning", "arena-procedural-drills/prereqs_einops/einops-einsum/04-batched-matrix-multiply-preserve-a-batch-axis.ipynb", "einops-einsum — ex4: batched matrix multiply (preserve a batch axis)"),
    E("einops-einsum", 5, "attention scores QK^T (batched + reduce + matmul-like)", "Einops: Deep Learning", "arena-procedural-drills/prereqs_einops/einops-einsum/05-attention-scores-qk-t-batched-reduce-matmul-like.ipynb", "einops-einsum — ex5: attention scores QK^T (batched + reduce + matmul-like)"),
    E("einops-einsum", 6, "scaled dot-product attention end-to-end with mask + softmax + weight heatmap", "Einops: Deep Learning", "arena-procedural-drills/prereqs_einops/einops-einsum/06-scaled-dot-product-attention-with-mask-viz.ipynb", "einops-einsum — ex6: scaled dot-product attention end-to-end with mask + softmax + weight heatmap"),
    E("einops-einsum", 7, "multi-head attention scores — split (b, s, h*d) and contract per head", "Einops: Deep Learning", "arena-procedural-drills/prereqs_einops/einops-einsum/07-multi-head-scores-split-and-contract.ipynb", "einops-einsum — ex7: multi-head attention scores — split (b, s, h*d) and contract per head"),
    E("einops-einsum", 8, "batched bilinear form y = x^T A x over a 2D grid, with heatmap viz", "Einops: Deep Learning", "arena-procedural-drills/prereqs_einops/einops-einsum/08-batched-bilinear-form-with-heatmap.ipynb", "einops-einsum — ex8: batched bilinear form y = x^T A x over a 2D grid, with heatmap viz"),
    E("einops-einsum", 9, "three-tensor Tucker-style contraction with intermediate-shape debug prints", "Einops: Deep Learning", "arena-procedural-drills/prereqs_einops/einops-einsum/09-three-tensor-tucker-contraction-debug.ipynb", "einops-einsum — ex9: three-tensor Tucker-style contraction with intermediate-shape debug prints"),
    // einops-rearrange
    E("einops-rearrange", 1, "identity rearrange", "Einops: Rearrange", "arena-procedural-drills/prereqs_einops/einops-rearrange/01-identity-rearrange.ipynb", "einops-rearrange — ex1: identity rearrange"),
    E("einops-rearrange", 2, "axis swap (transpose)", "Einops: Rearrange", "arena-procedural-drills/prereqs_einops/einops-rearrange/02-axis-swap-transpose.ipynb", "einops-rearrange — ex2: axis swap (transpose)"),
    E("einops-rearrange", 3, "image flatten (axis composition)", "Einops: Rearrange", "arena-procedural-drills/prereqs_einops/einops-rearrange/03-image-flatten-axis-composition.ipynb", "einops-rearrange — ex3: image flatten (axis composition)"),
    E("einops-rearrange", 4, "batch unfold (axis decomposition)", "Einops: Rearrange", "arena-procedural-drills/prereqs_einops/einops-rearrange/04-batch-unfold-axis-decomposition.ipynb", "einops-rearrange — ex4: batch unfold (axis decomposition)"),
    E("einops-rearrange", 5, "patch grid (ViT-style patch embedding prep)", "Einops: Rearrange", "arena-procedural-drills/prereqs_einops/einops-rearrange/05-patch-grid-vit-style-patch-embedding-prep.ipynb", "einops-rearrange — ex5: patch grid (ViT-style patch embedding prep)"),
    E("einops-rearrange", 6, "multi-head attention split (with shape-pipeline debug)", "Einops: Rearrange", "arena-procedural-drills/prereqs_einops/einops-rearrange/06-multi-head-attention-split-shape-pipeline.ipynb", "einops-rearrange — ex6: multi-head attention split (with shape-pipeline debug)"),
    E("einops-rearrange", 7, "NHWC ↔ NCHW round-trip with imshow", "Einops: Rearrange", "arena-procedural-drills/prereqs_einops/einops-rearrange/07-nhwc-nchw-roundtrip-imshow.ipynb", "einops-rearrange — ex7: NHWC ↔ NCHW round-trip with imshow"),
    E("einops-rearrange", 8, "patchify ↔ unpatchify round-trip", "Einops: Rearrange", "arena-procedural-drills/prereqs_einops/einops-rearrange/08-patchify-unpatchify-roundtrip.ipynb", "einops-rearrange — ex8: patchify ↔ unpatchify round-trip"),
    E("einops-rearrange", 9, "edge case: patch_size that doesn't divide H/W", "Einops: Rearrange", "arena-procedural-drills/prereqs_einops/einops-rearrange/09-edge-case-non-divisible-patch-size.ipynb", "einops-rearrange — ex9: edge case: patch_size that doesn't divide H/W"),
    // einops-rearrange-flatten
    E("einops-rearrange-flatten", 1, "CNN flatten before the Linear head — channel + spatial collapse", "Einops: Rearrange-as-flatten", "arena-procedural-drills/prereqs_einops_advanced/einops-rearrange-flatten/01-cnn-flatten-before-linear-head.ipynb", "einops-rearrange-flatten — ex1: CNN flatten before the Linear head — channel + spatial collapse"),
    // einops-reduce
    E("einops-reduce", 1, "channel mean", "Einops: Reduce", "arena-procedural-drills/prereqs_einops/einops-reduce/01-channel-mean.ipynb", "einops-reduce — ex1: channel mean"),
    E("einops-reduce", 2, "per-image global mean (multi-axis drop)", "Einops: Reduce", "arena-procedural-drills/prereqs_einops/einops-reduce/02-per-image-global-mean-multi-axis-drop.ipynb", "einops-reduce — ex2: per-image global mean (multi-axis drop)"),
    E("einops-reduce", 3, "per-image spatial max (keepdim with ())", "Einops: Reduce", "arena-procedural-drills/prereqs_einops/einops-reduce/03-per-image-spatial-max-keepdim-with.ipynb", "einops-reduce — ex3: per-image spatial max (keepdim with ())"),
    E("einops-reduce", 4, "2×2 average pool (axis decomposition + reduce)", "Einops: Reduce", "arena-procedural-drills/prereqs_einops/einops-reduce/04-2-2-average-pool-axis-decomposition-reduce.ipynb", "einops-reduce — ex4: 2×2 average pool (axis decomposition + reduce)"),
    E("einops-reduce", 5, "row-wise softmax stabilization (reduce + keepdim + broadcast)", "Einops: Reduce", "arena-procedural-drills/prereqs_einops/einops-reduce/05-row-wise-softmax-stabilization-reduce-keepdim-broadcast.ipynb", "einops-reduce — ex5: row-wise softmax stabilization (reduce + keepdim + broadcast)"),
    E("einops-reduce", 6, "spatial pyramid pooling with imshow per level", "Einops: Reduce", "arena-procedural-drills/prereqs_einops/einops-reduce/06-spatial-pyramid-pooling-imshow.ipynb", "einops-reduce — ex6: spatial pyramid pooling with imshow per level"),
    E("einops-reduce", 7, "per-channel BN-style stats (mean, var, normalized output)", "Einops: Reduce", "arena-procedural-drills/prereqs_einops/einops-reduce/07-per-channel-bn-stats-normalized-output.ipynb", "einops-reduce — ex7: per-channel BN-style stats (mean, var, normalized output)"),
    E("einops-reduce", 8, "argmax-via-reduce (no torch.argmax) with intermediate prints", "Einops: Reduce", "arena-procedural-drills/prereqs_einops/einops-reduce/08-argmax-via-reduce-no-argmax.ipynb", "einops-reduce — ex8: argmax-via-reduce (no torch.argmax) with intermediate prints"),
    E("einops-reduce", 9, "top-k via repeated masked max (integrative multi-step)", "Einops: Reduce", "arena-procedural-drills/prereqs_einops/einops-reduce/09-top-k-via-repeated-masked-max.ipynb", "einops-reduce — ex9: top-k via repeated masked max (integrative multi-step)"),
    // einops-repeat
    E("einops-repeat", 1, "broadcast across batch", "Einops: Repeat", "arena-procedural-drills/prereqs_einops/einops-repeat/01-broadcast-across-batch.ipynb", "einops-repeat — ex1: broadcast across batch"),
    E("einops-repeat", 2, "per-token weight → per-feature weight", "Einops: Repeat", "arena-procedural-drills/prereqs_einops/einops-repeat/02-per-token-weight-per-feature-weight.ipynb", "einops-repeat — ex2: per-token weight → per-feature weight"),
    E("einops-repeat", 3, "vertical stretch (row-stretch via composition)", "Einops: Repeat", "arena-procedural-drills/prereqs_einops/einops-repeat/03-vertical-stretch-row-stretch-via-composition.ipynb", "einops-repeat — ex3: vertical stretch (row-stretch via composition)"),
    E("einops-repeat", 4, "horizontal tile (sequence-tile via composition)", "Einops: Repeat", "arena-procedural-drills/prereqs_einops/einops-repeat/04-horizontal-tile-sequence-tile-via-composition.ipynb", "einops-repeat — ex4: horizontal tile (sequence-tile via composition)"),
    E("einops-repeat", 5, "2×2 nearest-neighbor upsample (decompose + stretch + compose)", "Einops: Repeat", "arena-procedural-drills/prereqs_einops/einops-repeat/05-2-2-nearest-neighbor-upsample-decompose-stretch-compose.ipynb", "einops-repeat — ex5: 2×2 nearest-neighbor upsample (decompose + stretch + compose)"),
    E("einops-repeat", 6, "causal attention mask — broadcast (T,T) → (B,H,T,T) and visualize", "Einops: Repeat", "arena-procedural-drills/prereqs_einops/einops-repeat/06-causal-mask-broadcast-and-visualize.ipynb", "einops-repeat — ex6: causal attention mask — broadcast (T,T) → (B,H,T,T) and visualize"),
    E("einops-repeat", 7, "2D positional encoding — tile a 1D PE across a spatial grid", "Einops: Repeat", "arena-procedural-drills/prereqs_einops/einops-repeat/07-2d-positional-encoding-tile.ipynb", "einops-repeat — ex7: 2D positional encoding — tile a 1D PE across a spatial grid"),
    E("einops-repeat", 8, "nearest-neighbor upsample pyramid (8 → 16 → 32) with side-by-side viz", "Einops: Repeat", "arena-procedural-drills/prereqs_einops/einops-repeat/08-nn-upsample-pyramid-with-viz.ipynb", "einops-repeat — ex8: nearest-neighbor upsample pyramid (8 → 16 → 32) with side-by-side viz"),
    E("einops-repeat", 9, "grayscale → RGB replicate-then-shift with debug-print pipeline", "Einops: Repeat", "arena-procedural-drills/prereqs_einops/einops-repeat/09-gray-to-rgb-replicate-then-shift-debug.ipynb", "einops-repeat — ex9: grayscale → RGB replicate-then-shift with debug-print pipeline"),
    // einops-repeat-broadcast
    E("einops-repeat-broadcast", 1, "every-ray-with-every-triangle pairing without copy", "Einops: Repeat-as-broadcast", "arena-procedural-drills/prereqs_einops_advanced/einops-repeat-broadcast/01-every-ray-with-every-triangle-pairing-without-copy.ipynb", "einops-repeat-broadcast — ex1: every-ray-with-every-triangle pairing without copy"),
    E("einops-repeat-broadcast", 2, "per-token positional bias broadcast across batch", "Einops: Repeat-as-broadcast", "arena-procedural-drills/prereqs_einops_advanced/einops-repeat-broadcast/02-per-token-positional-bias-broadcast-across-batch.ipynb", "einops-repeat-broadcast — ex2: per-token positional bias broadcast across batch"),
    // einsum-contraction
    E("einsum-contraction", 1, "predict + verify which indices get summed", "Einsum: Index contraction semantics", "arena-procedural-drills/prereqs_einops_advanced/einsum-contraction/01-predict-and-verify-which-indices-get-summed.ipynb", "einsum-contraction — ex1: predict + verify which indices get summed"),
    E("einsum-contraction", 2, "lambertian dot product via index contraction", "Einsum: Index contraction semantics", "arena-procedural-drills/prereqs_einops_advanced/einsum-contraction/02-lambertian-dot-product-via-index-contraction.ipynb", "einsum-contraction — ex2: lambertian dot product via index contraction"),
    // ema-first-moment
    E("ema-first-moment", 1, "Adam m-buffer EMA update m = beta1*m + (1-beta1)*g", "Optimizer: Adam EMA first moment", "arena-procedural-drills/prereqs_adam_trainer/ema-first-moment/01-adam-m-buffer-ema-update-m-equals-beta1-m-plus-one-minus-beta1-g.ipynb", "ema-first-moment — ex1: Adam m-buffer EMA update m = beta1*m + (1-beta1)*g"),
    // ema-second-moment
    E("ema-second-moment", 1, "Adam v-buffer EMA update v = beta2*v + (1-beta2)*g^2", "Optimizer: Adam EMA second moment", "arena-procedural-drills/prereqs_optimizer_internals/ema-second-moment/01-adam-v-buffer-ema-update-v-equals-beta2-v-plus-one-minus-beta2-g-squared.ipynb", "ema-second-moment — ex1: Adam v-buffer EMA update v = beta2*v + (1-beta2)*g^2"),
    // encoder-decoder-symmetric
    E("encoder-decoder-symmetric", 1, "build a tiny autoencoder whose output shape == input shape", "CNN: Encoder-decoder symmetric layout", "arena-procedural-drills/prereqs_numerical_modules/encoder-decoder-symmetric/01-tiny-autoencoder-symmetric-layout.ipynb", "encoder-decoder-symmetric — ex1: build a tiny autoencoder whose output shape == input shape"),
    // end-grad-default-ones-like
    E("end-grad-default-ones-like", 1, "resolve end_grad — default ones_like, else use .array", "Backprop: end-grad ones_like default", "arena-procedural-drills/prereqs_autograd_pt2/end-grad-default-ones-like/01-resolve-end-grad-default-ones-like-else-use-array.ipynb", "end-grad-default-ones-like — ex1: resolve end_grad — default ones_like, else use .array"),
    // examples-seen-step-axis
    E("examples-seen-step-axis", 1, "compute examples_seen = step * batch_size as wandb x-axis", "Trainer: examples-seen step axis", "arena-procedural-drills/prereqs_adam_trainer/examples-seen-step-axis/01-compute-examples-seen-as-wandb-x-axis.ipynb", "examples-seen-step-axis — ex1: compute examples_seen = step * batch_size as wandb x-axis"),
    // fractional-stride-zero-insertion
    E("fractional-stride-zero-insertion", 1, "build the zero-inserted intermediate for stride-2 ConvT", "CNN: ConvT fractional-stride zero insertion", "arena-procedural-drills/prereqs_cnn_extras/fractional-stride-zero-insertion/01-build-the-zero-inserted-intermediate-for-stride-2-convt.ipynb", "fractional-stride-zero-insertion — ex1: build the zero-inserted intermediate for stride-2 ConvT"),
    // freeze-requires-grad
    E("freeze-requires-grad", 1, "freeze a toy backbone and collect trainable params", "PyTorch: freeze via requires_grad=False", "arena-procedural-drills/prereqs_cnn_extras/freeze-requires-grad/01-freeze-a-toy-backbone-and-collect-trainable-params.ipynb", "freeze-requires-grad — ex1: freeze a toy backbone and collect trainable params"),
    // get-children-callable-param
    E("get-children-callable-param", 1, "get_children yields (name, value) for Tensor-valued attributes", "Backprop: get_children callable param", "arena-procedural-drills/prereqs_autograd_pt3/get-children-callable-param/01-get-children-yields-tensor-valued-attributes.ipynb", "get-children-callable-param — ex1: get_children yields (name, value) for Tensor-valued attributes"),
    // grad-accumulate-on-leaf
    E("grad-accumulate-on-leaf", 1, "accumulate_grad: leaf.grad = (leaf.grad or 0) + g", "Backprop: Grad accumulate on leaf", "arena-procedural-drills/prereqs_autograd_pt3/grad-accumulate-on-leaf/01-accumulate-grad-on-leaf-set-or-add.ipynb", "grad-accumulate-on-leaf — ex1: accumulate_grad: leaf.grad = (leaf.grad or 0) + g"),
    // grad-tracking-global-toggle
    E("grad-tracking-global-toggle", 1, "no_grad context manager built on a module-level toggle", "Backprop: Grad-tracking toggle", "arena-procedural-drills/prereqs_autograd_internals/grad-tracking-global-toggle/01-no-grad-context-manager-from-module-toggle.ipynb", "grad-tracking-global-toggle — ex1: no_grad context manager built on a module-level toggle"),
    // inf-masking
    E("inf-masking", 1, "causal attention masking with -inf + softmax + viz", "Numpy: Inf-fill masking trick", "arena-procedural-drills/prereqs_einops_advanced/inf-masking/01-causal-attention-masking-with-inf-and-softmax-viz.ipynb", "inf-masking — ex1: causal attention masking with -inf + softmax + viz"),
    // inference-mode-step
    E("inference-mode-step", 1, "decorate step with inference_mode to allow in-place leaf update", "PyTorch: Inference mode step", "arena-procedural-drills/prereqs_optimizer_internals/inference-mode-step/01-decorate-step-with-inference-mode-to-allow-in-place-leaf-update.ipynb", "inference-mode-step — ex1: decorate step with inference_mode to allow in-place leaf update"),
    E("inference-mode-step", 2, "diagnose missing inference_mode decorator on step", "PyTorch: Inference mode step", "arena-procedural-drills/prereqs_optimizer_internals/inference-mode-step/02-diagnose-missing-inference-mode-decorator-on-step.ipynb", "inference-mode-step — ex2: diagnose missing inference_mode decorator on step"),
    // init-process-group-nccl
    E("init-process-group-nccl", 1, "init + destroy a process group with gloo", "Distributed: init_process_group nccl", "arena-procedural-drills/prereqs_distributed/init-process-group-nccl/01-init-and-destroy-a-process-group-with-gloo.ipynb", "init-process-group-nccl — ex1: init + destroy a process group with gloo"),
    // inplace-op-unsafe-warning
    E("inplace-op-unsafe-warning", 1, "add_inplace_safe: refuse when .recipe is not None, mutate otherwise", "Backprop: In-place op unsafe warning", "arena-procedural-drills/prereqs_autograd_pt3/inplace-op-unsafe-warning/01-add-inplace-safe-refuse-when-recipe-attached.ipynb", "inplace-op-unsafe-warning — ex1: add_inplace_safe: refuse when .recipe is not None, mutate otherwise"),
    // inplace-param-update
    E("inplace-param-update", 1, "in-place vs out-of-place parameter update", "PyTorch: In-place param update", "arena-procedural-drills/prereqs_training_loop/inplace-param-update/01-in-place-vs-out-of-place-parameter-update.ipynb", "inplace-param-update — ex1: in-place vs out-of-place parameter update"),
    // is-differentiable-flag
    E("is-differentiable-flag", 1, "three-gate requires_grad reading is_differentiable from closure", "Backprop: is_differentiable flag", "arena-procedural-drills/prereqs_autograd_pt2/is-differentiable-flag/01-three-gate-requires-grad-reading-is-differentiable-from-closure.ipynb", "is-differentiable-flag — ex1: three-gate requires_grad reading is_differentiable from closure"),
    // kaiming-uniform-init
    E("kaiming-uniform-init", 1, "build a Linear with Kaiming-uniform init + histogram visualization", "Init: Kaiming uniform", "arena-procedural-drills/prereqs_numerical_modules/kaiming-uniform-init/01-kaiming-uniform-linear-init-with-histogram.ipynb", "kaiming-uniform-init — ex1: build a Linear with Kaiming-uniform init + histogram visualization"),
    // kwargs-pass-through-recipe
    E("kwargs-pass-through-recipe", 1, "thread kwargs into forward call AND Recipe", "Backprop: Kwargs pass-through", "arena-procedural-drills/prereqs_autograd_internals/kwargs-pass-through-recipe/01-thread-kwargs-into-forward-call-and-recipe.ipynb", "kwargs-pass-through-recipe — ex1: thread kwargs into forward call AND Recipe"),
    // linalg-solve-batched
    E("linalg-solve-batched", 1, "solve a batch of 2x2 systems", "PyTorch: Batched linalg.solve", "arena-procedural-drills/prereqs_geometry_cnn/linalg-solve-batched/01-solve-a-batch-of-2x2-systems.ipynb", "linalg-solve-batched — ex1: solve a batch of 2x2 systems"),
    // log-back
    E("log-back", 1, "implement log_back from the elementwise chain rule", "Backprop: log_back", "arena-procedural-drills/prereqs_autograd_pt2/log-back/01-implement-log-back-from-elementwise-chain-rule.ipynb", "log-back — ex1: implement log_back from the elementwise chain rule"),
    // loss-item-scalar-extract
    E("loss-item-scalar-extract", 1, "distinguish .item() from .detach().cpu() in a training-loop logger", "PyTorch: loss.item() scalar extract", "arena-procedural-drills/prereqs_numerical_modules/loss-item-scalar-extract/01-loss-item-vs-detach-cpu.ipynb", "loss-item-scalar-extract — ex1: distinguish .item() from .detach().cpu() in a training-loop logger"),
    // matmul-2d
    E("matmul-2d", 1, "predict matmul output shape and verify with @", "Numpy: matmul 2-D", "arena-procedural-drills/prereqs_cnn_extras/matmul-2d/01-predict-matmul-output-shape-and-verify-with-at.ipynb", "matmul-2d — ex1: predict matmul output shape and verify with @"),
    // max-back-tied-half
    E("max-back-tied-half", 1, "maximum_back with 50/50 tie-splitting", "Backprop: max_back with tied half-mass", "arena-procedural-drills/prereqs_autograd_pt2/max-back-tied-half/01-maximum-back-with-50-50-tie-splitting.ipynb", "max-back-tied-half — ex1: maximum_back with 50/50 tie-splitting"),
    // module-composition
    E("module-composition", 1, "child Modules auto-register as attributes", "PyTorch: Module composition", "arena-procedural-drills/prereqs_pytorch_modules/module-composition/01-child-modules-auto-register.ipynb", "module-composition — ex1: child Modules auto-register as attributes"),
    E("module-composition", 2, "rebuild the MLP with nn.Sequential", "PyTorch: Module composition", "arena-procedural-drills/prereqs_pytorch_modules/module-composition/02-rebuild-mlp-with-sequential.ipynb", "module-composition — ex2: rebuild the MLP with nn.Sequential"),
    // module-extra-repr
    E("module-extra-repr", 1, "extra_repr for a Linear-style module", "PyTorch: Module __repr__", "arena-procedural-drills/prereqs_pytorch_modules/module-extra-repr/01-extra-repr-for-linear-style-module.ipynb", "module-extra-repr — ex1: extra_repr for a Linear-style module"),
    // momentum-buffer-update
    E("momentum-buffer-update", 1, "in-place momentum buffer update b = mu*b + g", "Optimizer: Momentum buffer", "arena-procedural-drills/prereqs_optimizer_internals/momentum-buffer-update/01-in-place-momentum-buffer-update-b-equals-mu-b-plus-g.ipynb", "momentum-buffer-update — ex1: in-place momentum buffer update b = mu*b + g"),
    // mp-spawn-workers
    E("mp-spawn-workers", 1, "launch a 2-rank distributed job with mp.spawn", "Distributed: mp.spawn workers", "arena-procedural-drills/prereqs_distributed/mp-spawn-workers/01-launch-a-2-rank-distributed-job-with-mp-spawn.ipynb", "mp-spawn-workers — ex1: launch a 2-rank distributed job with mp.spawn"),
    // multiply-back
    E("multiply-back", 1, "implement multiply_back0 / multiply_back1 with unbroadcast", "Backprop: multiply_back", "arena-procedural-drills/prereqs_autograd_pt2/multiply-back/01-implement-multiply-back0-and-back1-with-unbroadcast.ipynb", "multiply-back — ex1: implement multiply_back0 / multiply_back1 with unbroadcast"),
    // nn-module-subclass
    E("nn-module-subclass", 1, "minimal stateless Module (no __init__)", "PyTorch: nn.Module subclassing", "arena-procedural-drills/prereqs_pytorch_modules/nn-module-subclass/01-minimal-stateless-module.ipynb", "nn-module-subclass — ex1: minimal stateless Module (no __init__)"),
    E("nn-module-subclass", 2, "diagnose missing super().__init__()", "PyTorch: nn.Module subclassing", "arena-procedural-drills/prereqs_pytorch_modules/nn-module-subclass/02-diagnose-missing-super-init.ipynb", "nn-module-subclass — ex2: diagnose missing super().__init__()"),
    E("nn-module-subclass", 3, "Linear layer from scratch", "PyTorch: nn.Module subclassing", "arena-procedural-drills/prereqs_pytorch_modules/nn-module-subclass/03-linear-layer-from-scratch.ipynb", "nn-module-subclass — ex3: Linear layer from scratch"),
    // nn-parameter-wrap
    E("nn-parameter-wrap", 1, "Parameter vs raw tensor — visibility test", "PyTorch: nn.Parameter", "arena-procedural-drills/prereqs_pytorch_modules/nn-parameter-wrap/01-parameter-vs-raw-tensor-visibility.ipynb", "nn-parameter-wrap — ex1: Parameter vs raw tensor — visibility test"),
    E("nn-parameter-wrap", 2, "Parameter vs buffer — pick the right registration", "PyTorch: nn.Parameter", "arena-procedural-drills/prereqs_pytorch_modules/nn-parameter-wrap/02-parameter-vs-buffer.ipynb", "nn-parameter-wrap — ex2: Parameter vs buffer — pick the right registration"),
    // no-relu-on-final-layer
    E("no-relu-on-final-layer", 1, "diagnose and strip a stray ReLU on the classifier head", "CNN: No-ReLU on final layer", "arena-procedural-drills/prereqs_cnn_deep/no-relu-on-final-layer/01-diagnose-and-strip-stray-relu-on-classifier-head.ipynb", "no-relu-on-final-layer — ex1: diagnose and strip a stray ReLU on the classifier head"),
    // non-diff-fn-wrap
    E("non-diff-fn-wrap", 1, "wrap a non-differentiable op (eq) — no Recipe, no requires_grad", "Backprop: non-differentiable fn wrap", "arena-procedural-drills/prereqs_autograd_pt2/non-diff-fn-wrap/01-wrap-non-differentiable-op-no-recipe-no-requires-grad.ipynb", "non-diff-fn-wrap — ex1: wrap a non-differentiable op (eq) — no Recipe, no requires_grad"),
    // optimizer-init-params-list
    E("optimizer-init-params-list", 1, "materialize a generator of params into a list at init", "PyTorch: Optimizer init", "arena-procedural-drills/prereqs_training_loop/optimizer-init-params-list/01-materialize-a-generator-of-params-into-a-list-at-init.ipynb", "optimizer-init-params-list — ex1: materialize a generator of params into a list at init"),
    // optimizer-loop-on-tensor
    E("optimizer-loop-on-tensor", 1, "manual SGD step: for p in self.params: p -= lr * p.grad", "Optimizer: optimizer.step loop over params", "arena-procedural-drills/prereqs_adam_trainer/optimizer-loop-on-tensor/01-manual-sgd-step-for-p-in-self-params-p-minus-equals-lr-times-p-grad.ipynb", "optimizer-loop-on-tensor — ex1: manual SGD step: for p in self.params: p -= lr * p.grad"),
    // optimizer-state-tensor-buffers
    E("optimizer-state-tensor-buffers", 1, "allocate a per-param zeros_like buffer at init", "Optimizer: Per-param state buffers", "arena-procedural-drills/prereqs_optimizer_internals/optimizer-state-tensor-buffers/01-allocate-a-per-param-zeros-like-buffer-at-init.ipynb", "optimizer-state-tensor-buffers — ex1: allocate a per-param zeros_like buffer at init"),
    E("optimizer-state-tensor-buffers", 2, "two-buffer init for an RMSprop-style optimizer", "Optimizer: Per-param state buffers", "arena-procedural-drills/prereqs_optimizer_internals/optimizer-state-tensor-buffers/02-two-buffer-init-for-an-rmsprop-style-optimizer.ipynb", "optimizer-state-tensor-buffers — ex2: two-buffer init for an RMSprop-style optimizer"),
    // padding-amount-formula-convT
    E("padding-amount-formula-convT", 1, "predict ConvTranspose2d output size from the padding arg", "CNN: ConvT padding amount formula", "arena-procedural-drills/prereqs_cnn_extras/padding-amount-formula-convT/01-predict-convtranspose2d-output-size-from-the-padding-arg.ipynb", "padding-amount-formula-convT — ex1: predict ConvTranspose2d output size from the padding arg"),
    // param-grad-access
    E("param-grad-access", 1, "iterate model.parameters() and read .grad with the None guard", "PyTorch: param.grad access", "arena-procedural-drills/prereqs_backprop/param-grad-access/01-iterate-parameters-read-grad-with-none-guard.ipynb", "param-grad-access — ex1: iterate model.parameters() and read .grad with the None guard"),
    // parameter-subclass-of-tensor
    E("parameter-subclass-of-tensor", 1, "Parameter subclasses MiniTensor with requires_grad=True default", "Backprop: Parameter subclasses Tensor", "arena-procedural-drills/prereqs_autograd_pt3/parameter-subclass-of-tensor/01-parameter-subclass-of-tensor-requires-grad-default.ipynb", "parameter-subclass-of-tensor — ex1: Parameter subclasses MiniTensor with requires_grad=True default"),
    // parents-dict-by-argidx
    E("parents-dict-by-argidx", 1, "build parents dict — skip non-Tensors, keep original argidx", "Backprop: Parents dict by argidx", "arena-procedural-drills/prereqs_autograd_internals/parents-dict-by-argidx/01-build-parents-dict-skip-non-tensors-keep-argidx.ipynb", "parents-dict-by-argidx — ex1: build parents dict — skip non-Tensors, keep original argidx"),
    // per-rank-cuda-device
    E("per-rank-cuda-device", 1, "pin per-rank cuda device with a mocked GPU", "Distributed: per-rank cuda device", "arena-procedural-drills/prereqs_distributed/per-rank-cuda-device/01-pin-per-rank-cuda-device-with-a-mocked-gpu.ipynb", "per-rank-cuda-device — ex1: pin per-rank cuda device with a mocked GPU"),
    // rank-world-size-args
    E("rank-world-size-args", 1, "thread rank and world_size through a broadcast signature", "Distributed: rank/world_size args", "arena-procedural-drills/prereqs_distributed/rank-world-size-args/01-thread-rank-and-world-size-through-a-broadcast-signature.ipynb", "rank-world-size-args — ex1: thread rank and world_size through a broadcast signature"),
    // ray-parametric-form
    E("ray-parametric-form", 1, "evaluate one ray at many parameter values", "Geometry: Ray parametric form", "arena-procedural-drills/prereqs_geometry_cnn/ray-parametric-form/01-evaluate-one-ray-at-many-parameter-values.ipynb", "ray-parametric-form — ex1: evaluate one ray at many parameter values"),
    E("ray-parametric-form", 2, "evaluate a batch of rays at one parameter", "Geometry: Ray parametric form", "arena-procedural-drills/prereqs_geometry_cnn/ray-parametric-form/02-evaluate-a-batch-of-rays-at-one-parameter.ipynb", "ray-parametric-form — ex2: evaluate a batch of rays at one parameter"),
    // rearrange-as-sequential-layer
    E("rearrange-as-sequential-layer", 1, "build a Conv-flatten-Linear pipeline with Rearrange layer (no forward boilerplate)", "Einops: Rearrange as nn.Sequential layer", "arena-procedural-drills/prereqs_numerical_modules/rearrange-as-sequential-layer/01-conv-rearrange-linear-pipeline.ipynb", "rearrange-as-sequential-layer — ex1: build a Conv-flatten-Linear pipeline with Rearrange layer (no forward boilerplate)"),
    // recipe-dataclass
    E("recipe-dataclass", 1, "define Recipe and construct it for log_forward", "Backprop: Recipe dataclass", "arena-procedural-drills/prereqs_autograd_internals/recipe-dataclass/01-define-recipe-and-construct-it-for-log-forward.ipynb", "recipe-dataclass — ex1: define Recipe and construct it for log_forward"),
    // reduce-op-mean-divide
    E("reduce-op-mean-divide", 1, "implement mean reduction as sum-then-in-place-divide", "Distributed: reduce-op mean divide", "arena-procedural-drills/prereqs_distributed/reduce-op-mean-divide/01-implement-mean-reduction-as-sum-then-in-place-divide.ipynb", "reduce-op-mean-divide — ex1: implement mean reduction as sum-then-in-place-divide"),
    // register-back-fn-after-wrap
    E("register-back-fn-after-wrap", 1, "wire one entry into BACK_FUNCS and dispatch it", "Backprop: register back fn", "arena-procedural-drills/prereqs_backprop/register-back-fn-after-wrap/01-wire-one-entry-into-back-funcs.ipynb", "register-back-fn-after-wrap — ex1: wire one entry into BACK_FUNCS and dispatch it"),
    E("register-back-fn-after-wrap", 2, "register a binary op at TWO argnums and dispatch both", "Backprop: register back fn", "arena-procedural-drills/prereqs_backprop/register-back-fn-after-wrap/02-register-binary-op-two-argnums.ipynb", "register-back-fn-after-wrap — ex2: register a binary op at TWO argnums and dispatch both"),
    // relu-elementwise-max
    E("relu-elementwise-max", 1, "implement ReLU + verify the derivative jump at 0", "CNN: ReLU as elementwise max", "arena-procedural-drills/prereqs_cnn_deep/relu-elementwise-max/01-implement-relu-and-verify-derivative-jump.ipynb", "relu-elementwise-max — ex1: implement ReLU + verify the derivative jump at 0"),
    // requires-grad-propagation
    E("requires-grad-propagation", 1, "three-gate requires_grad: toggle AND is_differentiable AND any-input", "Backprop: requires_grad propagation", "arena-procedural-drills/prereqs_autograd_internals/requires-grad-propagation/01-three-gate-requires-grad-toggle-and-diff-and-any-input.ipynb", "requires-grad-propagation — ex1: three-gate requires_grad: toggle AND is_differentiable AND any-input"),
    // rotation-matrix-3d-y-axis
    E("rotation-matrix-3d-y-axis", 1, "compute cos(θ) and sin(θ) as tensors", "Numpy: Applied patterns and advanced", "arena-procedural-drills/prereqs_numpy/rotation-matrix-3d-y-axis/01-compute-cos-and-sin-as-tensors.ipynb", "rotation-matrix-3d-y-axis — ex1: compute cos(θ) and sin(θ) as tensors"),
    E("rotation-matrix-3d-y-axis", 2, "build the 3×3 Y-axis rotation matrix", "Numpy: Applied patterns and advanced", "arena-procedural-drills/prereqs_numpy/rotation-matrix-3d-y-axis/02-build-the-3-3-y-axis-rotation-matrix.ipynb", "rotation-matrix-3d-y-axis — ex2: build the 3×3 Y-axis rotation matrix"),
    E("rotation-matrix-3d-y-axis", 3, "rotate a single 3-vector", "Numpy: Applied patterns and advanced", "arena-procedural-drills/prereqs_numpy/rotation-matrix-3d-y-axis/03-rotate-a-single-3-vector.ipynb", "rotation-matrix-3d-y-axis — ex3: rotate a single 3-vector"),
    E("rotation-matrix-3d-y-axis", 4, "verify rotation composition R(α)·R(β) = R(α+β)", "Numpy: Applied patterns and advanced", "arena-procedural-drills/prereqs_numpy/rotation-matrix-3d-y-axis/04-verify-rotation-composition-r-r-r.ipynb", "rotation-matrix-3d-y-axis — ex4: verify rotation composition R(α)·R(β) = R(α+β)"),
    E("rotation-matrix-3d-y-axis", 5, "rotate a batch of points around Y", "Numpy: Applied patterns and advanced", "arena-procedural-drills/prereqs_numpy/rotation-matrix-3d-y-axis/05-rotate-a-batch-of-points-around-y.ipynb", "rotation-matrix-3d-y-axis — ex5: rotate a batch of points around Y"),
    E("rotation-matrix-3d-y-axis", 6, "rotation sweep — visualize 5 angles", "Numpy: Applied patterns and advanced", "arena-procedural-drills/prereqs_numpy/rotation-matrix-3d-y-axis/06-rotation-sweep-visualize-5-angles.ipynb", "rotation-matrix-3d-y-axis — ex6: rotation sweep — visualize 5 angles"),
    E("rotation-matrix-3d-y-axis", 7, "compose Rx · Ry · Rz on a cube + 3-D scatter", "Numpy: Applied patterns and advanced", "arena-procedural-drills/prereqs_numpy/rotation-matrix-3d-y-axis/07-compose-rx-ry-rz-on-cube-3d-scatter.ipynb", "rotation-matrix-3d-y-axis — ex7: compose Rx · Ry · Rz on a cube + 3-D scatter"),
    E("rotation-matrix-3d-y-axis", 8, "inverse rotation = transpose — numerical sweep", "Numpy: Applied patterns and advanced", "arena-procedural-drills/prereqs_numpy/rotation-matrix-3d-y-axis/08-inverse-rotation-equals-transpose-numerical-sweep.ipynb", "rotation-matrix-3d-y-axis — ex8: inverse rotation = transpose — numerical sweep"),
    E("rotation-matrix-3d-y-axis", 9, "camera-to-world transform (rays + pose)", "Numpy: Applied patterns and advanced", "arena-procedural-drills/prereqs_numpy/rotation-matrix-3d-y-axis/09-camera-to-world-transform-rays-and-pose.ipynb", "rotation-matrix-3d-y-axis — ex9: camera-to-world transform (rays + pose)"),
    // singular-matrix-mask-trick
    E("singular-matrix-mask-trick", 1, "solve a batch with some singular matrices", "Numpy: Singular matrix mask trick", "arena-procedural-drills/prereqs_geometry_cnn/singular-matrix-mask-trick/01-solve-a-batch-with-some-singular-matrices.ipynb", "singular-matrix-mask-trick — ex1: solve a batch with some singular matrices"),
    // slice-view-mutation
    E("slice-view-mutation", 1, "in-place zero the diagonal via slice-view writes", "PyTorch: Slice view mutation", "arena-procedural-drills/prereqs_tensor_mechanics/slice-view-mutation/01-in-place-zero-diagonal-via-slice-view-write.ipynb", "slice-view-mutation — ex1: in-place zero the diagonal via slice-view writes"),
    // sorted-computational-graph
    E("sorted-computational-graph", 1, "topological sort of the compute graph for the reverse pass", "Backprop: Sorted computation graph", "arena-procedural-drills/prereqs_autograd_pt2/sorted-computational-graph/01-topological-sort-of-the-compute-graph-for-reverse-pass.ipynb", "sorted-computational-graph — ex1: topological sort of the compute graph for the reverse pass"),
    // sqrt-eps-stabilize
    E("sqrt-eps-stabilize", 1, "rescue a BatchNorm-style normalize from divide-by-zero", "Numerical: sqrt-eps stabilization", "arena-procedural-drills/prereqs_numerical_modules/sqrt-eps-stabilize/01-rescue-batchnorm-from-divide-by-zero.ipynb", "sqrt-eps-stabilize — ex1: rescue a BatchNorm-style normalize from divide-by-zero"),
    // stack-vs-cat
    E("stack-vs-cat", 1, "pick stack or cat from the target shape", "PyTorch: stack vs cat", "arena-procedural-drills/prereqs_tensor_mechanics/stack-vs-cat/01-pick-stack-or-cat-from-target-shape.ipynb", "stack-vs-cat — ex1: pick stack or cat from the target shape"),
    // step-counter-increment
    E("step-counter-increment", 1, "step counter increments AFTER optimizer.step", "Trainer: step counter increment", "arena-procedural-drills/prereqs_adam_trainer/step-counter-increment/01-step-counter-increments-after-optimizer-step.ipynb", "step-counter-increment — ex1: step counter increments AFTER optimizer.step"),
    // stride-zero-broadcast
    E("stride-zero-broadcast", 1, "diagnose zero-stride vs copy via .stride() + storage check", "PyTorch: Zero-stride broadcasting", "arena-procedural-drills/prereqs_numerical_modules/stride-zero-broadcast/01-diagnose-zero-stride-vs-copy.ipynb", "stride-zero-broadcast — ex1: diagnose zero-stride vs copy via .stride() + storage check"),
    // sum-and-broadcast-duality
    E("sum-and-broadcast-duality", 1, "sum_back and broadcast_back as dual ops", "Backprop: sum/broadcast duality", "arena-procedural-drills/prereqs_autograd_pt3/sum-and-broadcast-duality/01-sum-back-and-broadcast-back-dual-ops.ipynb", "sum-and-broadcast-duality — ex1: sum_back and broadcast_back as dual ops"),
    // tensor-item-scalar
    E("tensor-item-scalar", 1, "extract a Python float from a 0-D tensor", "Numpy: Core array literacy", "arena-procedural-drills/prereqs_numpy/tensor-item-scalar/01-extract-a-python-float-from-a-0-d-tensor.ipynb", "tensor-item-scalar — ex1: extract a Python float from a 0-D tensor"),
    E("tensor-item-scalar", 2, "extract from a single-element 1-D tensor", "Numpy: Core array literacy", "arena-procedural-drills/prereqs_numpy/tensor-item-scalar/02-extract-from-a-single-element-1-d-tensor.ipynb", "tensor-item-scalar — ex2: extract from a single-element 1-D tensor"),
    E("tensor-item-scalar", 3, ".item() preserves dtype family", "Numpy: Core array literacy", "arena-procedural-drills/prereqs_numpy/tensor-item-scalar/03-item-preserves-dtype-family.ipynb", "tensor-item-scalar — ex3: .item() preserves dtype family"),
    E("tensor-item-scalar", 4, ".item() vs .tolist() — pick the right tool", "Numpy: Core array literacy", "arena-procedural-drills/prereqs_numpy/tensor-item-scalar/04-item-vs-tolist-pick-the-right-tool.ipynb", "tensor-item-scalar — ex4: .item() vs .tolist() — pick the right tool"),
    E("tensor-item-scalar", 5, "tensor → Python control flow", "Numpy: Core array literacy", "arena-procedural-drills/prereqs_numpy/tensor-item-scalar/05-tensor-python-control-flow.ipynb", "tensor-item-scalar — ex5: tensor → Python control flow"),
    E("tensor-item-scalar", 6, "training loop with .item() logging and loss curve", "Numpy: Core array literacy", "arena-procedural-drills/prereqs_numpy/tensor-item-scalar/06-training-loop-with-item-logging-and-loss-curve.ipynb", "tensor-item-scalar — ex6: training loop with .item() logging and loss curve"),
    E("tensor-item-scalar", 7, "early stopping via .item() threshold", "Numpy: Core array literacy", "arena-procedural-drills/prereqs_numpy/tensor-item-scalar/07-early-stopping-via-item-threshold.ipynb", "tensor-item-scalar — ex7: early stopping via .item() threshold"),
    E("tensor-item-scalar", 8, "random walk histogram via .item()", "Numpy: Core array literacy", "arena-procedural-drills/prereqs_numpy/tensor-item-scalar/08-random-walk-histogram-via-item.ipynb", "tensor-item-scalar — ex8: random walk histogram via .item()"),
    // tensor-to-device
    E("tensor-to-device", 1, "move tensor to chosen device with the CPU/CUDA guard", "PyTorch: tensor.to(device)", "arena-procedural-drills/prereqs_tensor_mechanics/tensor-to-device/01-move-tensor-to-device-with-cuda-guard.ipynb", "tensor-to-device — ex1: move tensor to chosen device with the CPU/CUDA guard"),
    // tensor-unbind
    E("tensor-unbind", 1, "unbind along default dim 0", "Numpy: Indexing and selection", "arena-procedural-drills/prereqs_numpy/tensor-unbind/01-unbind-along-default-dim-0.ipynb", "tensor-unbind — ex1: unbind along default dim 0"),
    E("tensor-unbind", 2, "unbind along an explicit axis", "Numpy: Indexing and selection", "arena-procedural-drills/prereqs_numpy/tensor-unbind/02-unbind-along-an-explicit-axis.ipynb", "tensor-unbind — ex2: unbind along an explicit axis"),
    E("tensor-unbind", 3, "unbind equivalence with `.select`", "Numpy: Indexing and selection", "arena-procedural-drills/prereqs_numpy/tensor-unbind/03-unbind-equivalence-with-select.ipynb", "tensor-unbind — ex3: unbind equivalence with `.select`"),
    E("tensor-unbind", 4, "destructure a fixed number of slices", "Numpy: Indexing and selection", "arena-procedural-drills/prereqs_numpy/tensor-unbind/04-destructure-a-fixed-number-of-slices.ipynb", "tensor-unbind — ex4: destructure a fixed number of slices"),
    E("tensor-unbind", 5, "decompose rays, evaluate at parameter t", "Numpy: Indexing and selection", "arena-procedural-drills/prereqs_numpy/tensor-unbind/05-decompose-rays-evaluate-at-parameter-t.ipynb", "tensor-unbind — ex5: decompose rays, evaluate at parameter t"),
    E("tensor-unbind", 6, "batched ray cast with per-step shape debug", "Numpy: Indexing and selection", "arena-procedural-drills/prereqs_numpy/tensor-unbind/06-batched-ray-cast-with-per-step-shape-debug.ipynb", "tensor-unbind — ex6: batched ray cast with per-step shape debug"),
    E("tensor-unbind", 7, "split heads for multi-head attention", "Numpy: Indexing and selection", "arena-procedural-drills/prereqs_numpy/tensor-unbind/07-split-heads-for-multi-head-attention.ipynb", "tensor-unbind — ex7: split heads for multi-head attention"),
    E("tensor-unbind", 8, "RGB to grayscale with side-by-side plot", "Numpy: Indexing and selection", "arena-procedural-drills/prereqs_numpy/tensor-unbind/08-rgb-to-grayscale-with-side-by-side-plot.ipynb", "tensor-unbind — ex8: RGB to grayscale with side-by-side plot"),
    // tensor-wraps-ndarray
    E("tensor-wraps-ndarray", 1, "compare from_numpy aliasing vs tensor copy", "PyTorch: tensor from ndarray", "arena-procedural-drills/prereqs_tensor_mechanics/tensor-wraps-ndarray/01-compare-from-numpy-aliasing-vs-tensor-copy.ipynb", "tensor-wraps-ndarray — ex1: compare from_numpy aliasing vs tensor copy"),
    // tensor-zeros-init
    E("tensor-zeros-init", 1, "allocate a 1-D zero vector", "Numpy: Core array literacy", "arena-procedural-drills/prereqs_numpy/tensor-zeros-init/01-allocate-a-1-d-zero-vector.ipynb", "tensor-zeros-init — ex1: allocate a 1-D zero vector"),
    E("tensor-zeros-init", 2, "allocate a 3-D zero tensor", "Numpy: Core array literacy", "arena-procedural-drills/prereqs_numpy/tensor-zeros-init/02-allocate-a-3-d-zero-tensor.ipynb", "tensor-zeros-init — ex2: allocate a 3-D zero tensor"),
    E("tensor-zeros-init", 3, "zeros_like — mirror an input's shape and dtype", "Numpy: Core array literacy", "arena-procedural-drills/prereqs_numpy/tensor-zeros-init/03-zeros-like-mirror-an-input-s-shape-and-dtype.ipynb", "tensor-zeros-init — ex3: zeros_like — mirror an input's shape and dtype"),
    E("tensor-zeros-init", 4, "integer index buffer with dtype=long", "Numpy: Core array literacy", "arena-procedural-drills/prereqs_numpy/tensor-zeros-init/04-integer-index-buffer-with-dtype-long.ipynb", "tensor-zeros-init — ex4: integer index buffer with dtype=long"),
    E("tensor-zeros-init", 5, "allocate output buffer, then paint hits", "Numpy: Core array literacy", "arena-procedural-drills/prereqs_numpy/tensor-zeros-init/05-allocate-output-buffer-then-paint-hits.ipynb", "tensor-zeros-init — ex5: allocate output buffer, then paint hits"),
    E("tensor-zeros-init", 6, "histogram via scatter into a zeros buffer", "Numpy: Core array literacy", "arena-procedural-drills/prereqs_numpy/tensor-zeros-init/06-histogram-via-scatter-into-zeros-buffer.ipynb", "tensor-zeros-init — ex6: histogram via scatter into a zeros buffer"),
    E("tensor-zeros-init", 7, "confusion matrix from (pred, true) pairs", "Numpy: Core array literacy", "arena-procedural-drills/prereqs_numpy/tensor-zeros-init/07-confusion-matrix-from-pred-true-pairs.ipynb", "tensor-zeros-init — ex7: confusion matrix from (pred, true) pairs"),
    E("tensor-zeros-init", 8, "z-buffer painter with per-step debug", "Numpy: Core array literacy", "arena-procedural-drills/prereqs_numpy/tensor-zeros-init/08-z-buffer-painter-with-per-step-debug.ipynb", "tensor-zeros-init — ex8: z-buffer painter with per-step debug"),
    // train-eval-mode-branch
    E("train-eval-mode-branch", 1, "flip train and eval mode around dropout", "PyTorch: train/eval mode", "arena-procedural-drills/prereqs_training_loop/train-eval-mode-branch/01-flip-train-and-eval-mode-around-dropout.ipynb", "train-eval-mode-branch — ex1: flip train and eval mode around dropout"),
    // trainer-class-skeleton
    E("trainer-class-skeleton", 1, "minimal Trainer class: fit, validate, _step", "Trainer: Trainer class skeleton", "arena-procedural-drills/prereqs_adam_trainer/trainer-class-skeleton/01-minimal-trainer-class-fit-validate-step.ipynb", "trainer-class-skeleton — ex1: minimal Trainer class: fit, validate, _step"),
    // training-step-cycle
    E("training-step-cycle", 1, "order the five calls of the canonical training step", "PyTorch: Training step cycle", "arena-procedural-drills/prereqs_training_loop/training-step-cycle/01-order-the-five-calls-of-the-canonical-training-step.ipynb", "training-step-cycle — ex1: order the five calls of the canonical training step"),
    E("training-step-cycle", 2, "diagnose a buggy training step that forgets zero_grad", "PyTorch: Training step cycle", "arena-procedural-drills/prereqs_training_loop/training-step-cycle/02-diagnose-a-buggy-training-step-that-forgets-zero-grad.ipynb", "training-step-cycle — ex2: diagnose a buggy training step that forgets zero_grad"),
    // triangle-barycentric
    E("triangle-barycentric", 1, "point-in-triangle test from (u, v)", "Geometry: Barycentric coords", "arena-procedural-drills/prereqs_geometry_cnn/triangle-barycentric/01-point-in-triangle-test-from-uv.ipynb", "triangle-barycentric — ex1: point-in-triangle test from (u, v)"),
    // unbind-tuple-unpack
    E("unbind-tuple-unpack", 1, "ARENA-style two-level destructure of rays into (ox, oy, oz, dx, dy, dz)", "PyTorch: Unbind tuple-unpack", "arena-procedural-drills/prereqs_einops_advanced/unbind-tuple-unpack/01-two-level-destructure-rays-to-named-components.ipynb", "unbind-tuple-unpack — ex1: ARENA-style two-level destructure of rays into (ox, oy, oz, dx, dy, dz)"),
    // unbox-args-tensor-to-array
    E("unbox-args-tensor-to-array", 1, "unbox MiniTensor positional args to raw arrays, pass-through non-Tensors", "Backprop: Unbox Tensor args to array", "arena-procedural-drills/prereqs_autograd_pt3/unbox-args-tensor-to-array/01-unbox-tensor-args-to-raw-arrays.ipynb", "unbox-args-tensor-to-array — ex1: unbox MiniTensor positional args to raw arrays, pass-through non-Tensors"),
    // unbroadcast-pattern
    E("unbroadcast-pattern", 1, "unbroadcast: sum out leading and size-1 broadcast axes", "Backprop: Unbroadcast pattern", "arena-procedural-drills/prereqs_autograd_internals/unbroadcast-pattern/01-unbroadcast-sum-out-leading-and-size-1-axes.ipynb", "unbroadcast-pattern — ex1: unbroadcast: sum out leading and size-1 broadcast axes"),
    // validation-no-grad
    E("validation-no-grad", 1, "validation pass wrapped in torch.no_grad", "PyTorch: no_grad validation", "arena-procedural-drills/prereqs_training_loop/validation-no-grad/01-validation-pass-wrapped-in-torch-no-grad.ipynb", "validation-no-grad — ex1: validation pass wrapped in torch.no_grad"),
    // weight-decay-l2-add
    E("weight-decay-l2-add", 1, "fold weight decay lambda*theta into the gradient", "Optimizer: Weight decay L2", "arena-procedural-drills/prereqs_optimizer_internals/weight-decay-l2-add/01-fold-weight-decay-lambda-theta-into-the-gradient.ipynb", "weight-decay-l2-add — ex1: fold weight decay lambda*theta into the gradient"),
    // wrap-forward-fn-generic
    E("wrap-forward-fn-generic", 1, "write the wrap_forward_fn shell — unbox, call, box", "Backprop: wrap forward fn", "arena-procedural-drills/prereqs_backprop/wrap-forward-fn-generic/01-write-wrap-forward-fn-shell.ipynb", "wrap-forward-fn-generic — ex1: write the wrap_forward_fn shell — unbox, call, box"),
    E("wrap-forward-fn-generic", 2, "extend wrap_forward_fn with kwargs pass-through and is_differentiable", "Backprop: wrap forward fn", "arena-procedural-drills/prereqs_backprop/wrap-forward-fn-generic/02-wrap-forward-fn-with-kwargs-and-is-differentiable.ipynb", "wrap-forward-fn-generic — ex2: extend wrap_forward_fn with kwargs pass-through and is_differentiable"),
    // zero-grad-set-none
    E("zero-grad-set-none", 1, "implement zero_grad with set_to_none semantics", "PyTorch: zero_grad", "arena-procedural-drills/prereqs_training_loop/zero-grad-set-none/01-implement-zero-grad-with-set-to-none-semantics.ipynb", "zero-grad-set-none — ex1: implement zero_grad with set_to_none semantics"),
];

  // ── Drill auto-surface logic (new-algo, no legacy ARENA_PREREQS_TEMP) ──
  // Same shape as before — shown-once tracking via localStorage. The shown
  // schema bumps when this catalog grows so the user's queue refreshes.
  // Reads EWMA via window.getArenaPrereqSubtopicScore (backend cache OR
  // Pyodide adaptive state).

  const _DRILL_SHOWN_LS_KEY = "drills_shown";
  const _DRILL_SHOWN_SCHEMA_VERSION = "v2-standalone-2026-05-24";
  const _DRILL_SHOWN_VERSION_KEY = "drills_shown_schema";
  try {
    if (localStorage.getItem(_DRILL_SHOWN_VERSION_KEY) !== _DRILL_SHOWN_SCHEMA_VERSION) {
      localStorage.removeItem(_DRILL_SHOWN_LS_KEY);
      localStorage.setItem(_DRILL_SHOWN_VERSION_KEY, _DRILL_SHOWN_SCHEMA_VERSION);
    }
  } catch (_) { /* localStorage unavailable — fine */ }

  const _readShownSet = () => {
    try {
      const raw = localStorage.getItem(_DRILL_SHOWN_LS_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      return new Set(Array.isArray(arr) ? arr : []);
    } catch (_) { return new Set(); }
  };
  const _writeShownSet = (set) => {
    try { localStorage.setItem(_DRILL_SHOWN_LS_KEY, JSON.stringify([...set])); } catch (_) {}
  };

  const _scoreFor = (bareSubtopic) => {
    if (typeof window.getArenaPrereqSubtopicScore !== "function") return null;
    // The fn signature is (topic, subtopic) but it also accepts the full
    // composed key as the second arg (tries both raw and composed lookups).
    return window.getArenaPrereqSubtopicScore(null, bareSubtopic);
  };

  const _isDrillUnlocked = (drill) => {
    const subs = Array.isArray(drill.subtopics) ? drill.subtopics : [];
    if (!subs.length) return false;
    return subs.every((s) => {
      const sc = _scoreFor(s);
      return sc != null && sc >= (drill.unlockMinPct ?? DEFAULT_UNLOCK_MIN_PCT);
    });
  };

  window.getNextUnshownUnlockedDrill = () => {
    const shown = _readShownSet();
    for (const d of window.DRILLS_CATALOG) {
      if (shown.has(d.id)) continue;
      if (_isDrillUnlocked(d)) return d;
    }
    return null;
  };

  window.markDrillShown = (drillId) => {
    if (!drillId) return;
    const shown = _readShownSet();
    shown.add(drillId);
    _writeShownSet(shown);
  };

  // Diagnostic — `window.debugDrillUnlock()` mirrors window.debugArenaUnlock.
  window.debugDrillUnlock = () => {
    const shown = _readShownSet();
    const rows = (window.DRILLS_CATALOG || []).map((d) => {
      const subs = d.subtopics || [];
      const checks = subs.map((s) => {
        const sc = _scoreFor(s);
        return { subtopic: s, need: d.unlockMinPct, have: sc == null ? "null" : sc.toFixed(1), met: sc != null && sc >= d.unlockMinPct };
      });
      return {
        id: d.id,
        shown: shown.has(d.id),
        unlocked: checks.every((c) => c.met),
        blocking: checks.filter((c) => !c.met).map((c) => `${c.subtopic}(${c.have}<${c.need})`).join(", ") || "—",
      };
    });
    console.group(`[Drills] unlock snapshot — ${rows.length} standalone exercises`);
    console.table(rows);
    console.groupEnd();
    return rows;
  };
})();
