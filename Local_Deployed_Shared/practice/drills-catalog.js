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
    // backward-fn-signature
    E("backward-fn-signature", 1, "write log_back with the canonical (grad_out, out, x) signature", "Backprop: backward fn signature", "arena-procedural-drills/prereqs_backprop/backward-fn-signature/01-write-log-back-with-canonical-signature.ipynb", "backward-fn-signature — ex1: write log_back with the canonical (grad_out, out, x) signature"),
    E("backward-fn-signature", 2, "write negative_back and exp_back — back-fn signature, two ops", "Backprop: backward fn signature", "arena-procedural-drills/prereqs_backprop/backward-fn-signature/02-negative-and-exp-back-two-ops.ipynb", "backward-fn-signature — ex2: write negative_back and exp_back — back-fn signature, two ops"),
    // backward-on-scalar-loss
    E("backward-on-scalar-loss", 1, "reduce per-sample loss to a scalar before backward", "PyTorch: backward()", "arena-procedural-drills/prereqs_training_loop/backward-on-scalar-loss/01-reduce-per-sample-loss-to-a-scalar-before-backward.ipynb", "backward-on-scalar-loss — ex1: reduce per-sample loss to a scalar before backward"),
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
    // contiguous-layout
    E("contiguous-layout", 1, "predict-then-verify the strides of a 3-D contiguous tensor", "PyTorch: Contiguous layout", "arena-procedural-drills/prereqs_tensor_mechanics/contiguous-layout/01-predict-then-verify-strides-of-contiguous-3d.ipynb", "contiguous-layout — ex1: predict-then-verify the strides of a 3-D contiguous tensor"),
    E("contiguous-layout", 2, "fix the view-after-transpose error with .contiguous()", "PyTorch: Contiguous layout", "arena-procedural-drills/prereqs_tensor_mechanics/contiguous-layout/02-fix-view-after-transpose-with-contiguous.ipynb", "contiguous-layout — ex2: fix the view-after-transpose error with .contiguous()"),
    // conv-output-shape
    E("conv-output-shape", 1, "compute conv2d output shape analytically", "CNN: Conv output shape", "arena-procedural-drills/prereqs_geometry_cnn/conv-output-shape/01-compute-conv2d-output-shape-analytically.ipynb", "conv-output-shape — ex1: compute conv2d output shape analytically"),
    // conv-padding-zero
    E("conv-padding-zero", 1, "build a zero-padded 1-D input by slice assignment", "CNN: Conv zero padding", "arena-procedural-drills/prereqs_geometry_cnn/conv-padding-zero/01-build-a-zero-padded-1d-input-by-slice-assignment.ipynb", "conv-padding-zero — ex1: build a zero-padded 1-D input by slice assignment"),
    // conv-windowing-1d
    E("conv-windowing-1d", 1, "build the 1-D conv window view via as_strided", "CNN: 1-D conv windowing", "arena-procedural-drills/prereqs_geometry_cnn/conv-windowing-1d/01-build-the-1d-conv-window-view-via-as-strided.ipynb", "conv-windowing-1d — ex1: build the 1-D conv window view via as_strided"),
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
    // inplace-param-update
    E("inplace-param-update", 1, "in-place vs out-of-place parameter update", "PyTorch: In-place param update", "arena-procedural-drills/prereqs_training_loop/inplace-param-update/01-in-place-vs-out-of-place-parameter-update.ipynb", "inplace-param-update — ex1: in-place vs out-of-place parameter update"),
    // linalg-solve-batched
    E("linalg-solve-batched", 1, "solve a batch of 2x2 systems", "PyTorch: Batched linalg.solve", "arena-procedural-drills/prereqs_geometry_cnn/linalg-solve-batched/01-solve-a-batch-of-2x2-systems.ipynb", "linalg-solve-batched — ex1: solve a batch of 2x2 systems"),
    // module-composition
    E("module-composition", 1, "child Modules auto-register as attributes", "PyTorch: Module composition", "arena-procedural-drills/prereqs_pytorch_modules/module-composition/01-child-modules-auto-register.ipynb", "module-composition — ex1: child Modules auto-register as attributes"),
    E("module-composition", 2, "rebuild the MLP with nn.Sequential", "PyTorch: Module composition", "arena-procedural-drills/prereqs_pytorch_modules/module-composition/02-rebuild-mlp-with-sequential.ipynb", "module-composition — ex2: rebuild the MLP with nn.Sequential"),
    // module-extra-repr
    E("module-extra-repr", 1, "extra_repr for a Linear-style module", "PyTorch: Module __repr__", "arena-procedural-drills/prereqs_pytorch_modules/module-extra-repr/01-extra-repr-for-linear-style-module.ipynb", "module-extra-repr — ex1: extra_repr for a Linear-style module"),
    // nn-module-subclass
    E("nn-module-subclass", 1, "minimal stateless Module (no __init__)", "PyTorch: nn.Module subclassing", "arena-procedural-drills/prereqs_pytorch_modules/nn-module-subclass/01-minimal-stateless-module.ipynb", "nn-module-subclass — ex1: minimal stateless Module (no __init__)"),
    E("nn-module-subclass", 2, "diagnose missing super().__init__()", "PyTorch: nn.Module subclassing", "arena-procedural-drills/prereqs_pytorch_modules/nn-module-subclass/02-diagnose-missing-super-init.ipynb", "nn-module-subclass — ex2: diagnose missing super().__init__()"),
    E("nn-module-subclass", 3, "Linear layer from scratch", "PyTorch: nn.Module subclassing", "arena-procedural-drills/prereqs_pytorch_modules/nn-module-subclass/03-linear-layer-from-scratch.ipynb", "nn-module-subclass — ex3: Linear layer from scratch"),
    // nn-parameter-wrap
    E("nn-parameter-wrap", 1, "Parameter vs raw tensor — visibility test", "PyTorch: nn.Parameter", "arena-procedural-drills/prereqs_pytorch_modules/nn-parameter-wrap/01-parameter-vs-raw-tensor-visibility.ipynb", "nn-parameter-wrap — ex1: Parameter vs raw tensor — visibility test"),
    E("nn-parameter-wrap", 2, "Parameter vs buffer — pick the right registration", "PyTorch: nn.Parameter", "arena-procedural-drills/prereqs_pytorch_modules/nn-parameter-wrap/02-parameter-vs-buffer.ipynb", "nn-parameter-wrap — ex2: Parameter vs buffer — pick the right registration"),
    // optimizer-init-params-list
    E("optimizer-init-params-list", 1, "materialize a generator of params into a list at init", "PyTorch: Optimizer init", "arena-procedural-drills/prereqs_training_loop/optimizer-init-params-list/01-materialize-a-generator-of-params-into-a-list-at-init.ipynb", "optimizer-init-params-list — ex1: materialize a generator of params into a list at init"),
    // param-grad-access
    E("param-grad-access", 1, "iterate model.parameters() and read .grad with the None guard", "PyTorch: param.grad access", "arena-procedural-drills/prereqs_backprop/param-grad-access/01-iterate-parameters-read-grad-with-none-guard.ipynb", "param-grad-access — ex1: iterate model.parameters() and read .grad with the None guard"),
    // ray-parametric-form
    E("ray-parametric-form", 1, "evaluate one ray at many parameter values", "Geometry: Ray parametric form", "arena-procedural-drills/prereqs_geometry_cnn/ray-parametric-form/01-evaluate-one-ray-at-many-parameter-values.ipynb", "ray-parametric-form — ex1: evaluate one ray at many parameter values"),
    E("ray-parametric-form", 2, "evaluate a batch of rays at one parameter", "Geometry: Ray parametric form", "arena-procedural-drills/prereqs_geometry_cnn/ray-parametric-form/02-evaluate-a-batch-of-rays-at-one-parameter.ipynb", "ray-parametric-form — ex2: evaluate a batch of rays at one parameter"),
    // register-back-fn-after-wrap
    E("register-back-fn-after-wrap", 1, "wire one entry into BACK_FUNCS and dispatch it", "Backprop: register back fn", "arena-procedural-drills/prereqs_backprop/register-back-fn-after-wrap/01-wire-one-entry-into-back-funcs.ipynb", "register-back-fn-after-wrap — ex1: wire one entry into BACK_FUNCS and dispatch it"),
    E("register-back-fn-after-wrap", 2, "register a binary op at TWO argnums and dispatch both", "Backprop: register back fn", "arena-procedural-drills/prereqs_backprop/register-back-fn-after-wrap/02-register-binary-op-two-argnums.ipynb", "register-back-fn-after-wrap — ex2: register a binary op at TWO argnums and dispatch both"),
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
    // stack-vs-cat
    E("stack-vs-cat", 1, "pick stack or cat from the target shape", "PyTorch: stack vs cat", "arena-procedural-drills/prereqs_tensor_mechanics/stack-vs-cat/01-pick-stack-or-cat-from-target-shape.ipynb", "stack-vs-cat — ex1: pick stack or cat from the target shape"),
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
    // training-step-cycle
    E("training-step-cycle", 1, "order the five calls of the canonical training step", "PyTorch: Training step cycle", "arena-procedural-drills/prereqs_training_loop/training-step-cycle/01-order-the-five-calls-of-the-canonical-training-step.ipynb", "training-step-cycle — ex1: order the five calls of the canonical training step"),
    E("training-step-cycle", 2, "diagnose a buggy training step that forgets zero_grad", "PyTorch: Training step cycle", "arena-procedural-drills/prereqs_training_loop/training-step-cycle/02-diagnose-a-buggy-training-step-that-forgets-zero-grad.ipynb", "training-step-cycle — ex2: diagnose a buggy training step that forgets zero_grad"),
    // triangle-barycentric
    E("triangle-barycentric", 1, "point-in-triangle test from (u, v)", "Geometry: Barycentric coords", "arena-procedural-drills/prereqs_geometry_cnn/triangle-barycentric/01-point-in-triangle-test-from-uv.ipynb", "triangle-barycentric — ex1: point-in-triangle test from (u, v)"),
    // validation-no-grad
    E("validation-no-grad", 1, "validation pass wrapped in torch.no_grad", "PyTorch: no_grad validation", "arena-procedural-drills/prereqs_training_loop/validation-no-grad/01-validation-pass-wrapped-in-torch-no-grad.ipynb", "validation-no-grad — ex1: validation pass wrapped in torch.no_grad"),
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
