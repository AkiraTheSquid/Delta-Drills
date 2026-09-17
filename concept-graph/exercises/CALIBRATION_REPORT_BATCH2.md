# Vocab Calibration Report — Seed Pilot Batch 2 (all 24 exercises)

**Date:** 2026-05-18
**Pass:** seed-pilot-batch-2 (this batch adds 20 exercises: 0,2,3,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23)
**Total tagged:** 24/24 exercises in `part2_cnns` (4 from batch 1 + 20 from this batch)
**Vocab size:** 36 atoms in `vocab/atoms.json` (+ 4 proposed = 40 effective)

---

## Headline stats

| Metric | Target | Batch 1 (n=4) | Combined (n=24) | Verdict |
|---|---|---|---|---|
| Atoms per exercise (mean) | 4-7 | 6.5 | **4.88** | inside band, lower end |
| Atoms per exercise (range) | n/a | 5-9 | 1-9 | wide; trivial Modules at 1-3, dense conv/batchnorm at 7-9 |
| Total atom tags | n/a | 26 | **117** | |
| False positives per exercise | n/a (lower better) | 1.75 | **0.83** | seed over-tags decayed (more granular reading caught fewer phantoms) |
| Proposed new atoms per exercise | decaying | 1.0 | 0.17 | strong decay — 0 new atoms surfaced in batch 2 beyond the 4 already proposed |
| Recurring atom fraction (>=2 of 24) | n/a | 41% | **26 of 40 = 65%** | healthy convergence |

**Per-exercise atom counts:** 0:2, 1:6, 2:3, 3:3, 4:6, 5:9, 6:2, 7:6, 8:3, 9:6, 10:3, 11:3, 12:5, 13:4, 14:3, 15:5, 16:5, 17:5, 18:7, 19:7, 20:1, 21:8, 22:9, 23:6

The variance (1-9) reflects genuine pedagogical structure:
- **1-2 atoms** (0, 6, 20): trivial leaf Modules and pure utilities (ReLU, AveragePool, pad1d/pad2d)
- **3 atoms** (2, 3, 8, 10, 11, 14): single-concept exercises (MLP composition, validation loop, blockgroup, predict, freeze+swap, trace)
- **6-9 atoms** (1, 4, 5, 7, 9, 18, 19, 21, 22, 23): full Module implementations or full conv-mechanics drills

---

## Proposed new atoms (validator-confirmed; recurrence updated)

| id | label | exercises seen in | confidence | verdict |
|---|---|---|---|---|
| `conv-kernel-shape` | weight tensor (out_ch, in_ch, kH, kW) — 1D drops kW | `0_2_4`, `0_2_18`, `0_2_19`, `0_2_22` | **very high — 4 hits** | **accept** |
| `kaiming-uniform-init` | uniform Kaiming init with bound ±1/sqrt(fan_in) | `0_2_1`, `0_2_4` | high — 2 hits | accept |
| `register-buffer` | non-learnable state via register_buffer | `0_2_5` | medium — 1 hit, but genuinely distinct from nn-Parameter; recurs in attention masks, position encodings | accept |
| `functional-module-wrap` | Module is a parameter container; forward calls F.* | `0_2_4` | medium — 1 hit in tagged set; would have hit `MaxPool2d`/`AvgPool2d` modules if those were separate exercises but they're composed-only in ResNet34 | accept tentatively, flag for re-check at later chapters |

**Zero new atoms proposed in batch 2.** All exercise content fit into the 36 seed atoms + 4 already-proposed. This is a strong signal the vocab is at steady-state for `part2_cnns`.

---

## False-positive analysis

**20 false positives flagged across 24 exercises (mean 0.83).** Down from 1.75 in batch 1 — partly because batch-2 exercises are smaller-scoped (single-concept) and partly because the tagger learned the patterns. Notable patterns:

| Pattern | Count | Examples |
|---|---|---|
| `broadcasting-rules` predicted from topic mention but not actually used | 4 | 0_2_0 ReLU (scalar 0), 0_2_6 AveragePool (pure reduce), 0_2_7 ResidualBlock (shape-matched add), 0_2_20 pad (indexed copy) |
| `nn-module-subclass` predicted but exercise is a plain function | 3 | 0_2_17 minimal conv1d, 0_2_18 conv1d, 0_2_23 maxpool2d — all the "functional" exercises before the Module wrappers |
| `tensor-reshape-view` predicted from topic but reshape is delegated to a helper Module or never happens | 3 | 0_2_1 Linear (uses '...' einsum, no reshape), 0_2_2 MLP (Flatten() handles it), 0_2_9 ResNet34 (AveragePool drops spatial dims via mean) |
| `train-eval-mode-branch` predicted but `.train()/.eval()` flip not actually present | 2 | 0_2_3 validation loop (no batchnorm yet — SimpleMLP only), 0_2_11 freeze-and-swap (no forward pass run) |
| Conv-mechanics on F.conv2d wrapper exercise (0_2_4) | 4 | conv-channel-sum, conv-padding-zero, conv-windowing-2d, module-composition all predicted but only F.conv2d is called — atoms belong in 0_2_19/22 instead |
| Misc | 4 | 0_2_1 matmul-2d (subsumed by einsum-contraction), 0_2_2 nn-parameter-wrap (Linear children own it), 0_2_9 state-dict-load (lives in 0_2_11), 0_2_9 batchnorm-running-stats (composed not implemented) |

**Pattern confirmed:** the seed's "expected exercises" column over-tags whenever the *topic* is mentioned in overview markdown or when an *atom-bearing Module* is *instantiated* (but not implemented). The systematic fix is to re-derive the column from these tagged exercises.

---

## Atom-usage frequencies — the spine of part2_cnns

Top 12 atoms (>=4 exercises):

| atom | count | domain | exercises |
|---|---|---|---|
| `contiguous-layout` | 10 | strided-memory | all strided exercises 13-23 |
| `as-strided-windowing` | 10 | strided-memory | all strided exercises 13-23 |
| `nn-module-subclass` | 9 | pytorch-module-mechanics | 0,1,2,4,5,6,7,8,9 (every Module exercise) |
| `einsum-contraction` | 6 | tensor-algebra | 1,17,18,19,21,22 |
| `conv-output-shape` | 6 | convolution | 17,18,19,21,22,23 |
| `relu-elementwise-max` | 4 | activation | 0,2,7,9 |
| `broadcasting-rules` | 4 | tensor-algebra | 1,5,15,16 |
| `conv-channel-sum` | 4 | convolution | 18,19,21,22 |
| `conv-kernel-shape` (NEW) | 4 | convolution | 4,18,19,22 |
| `module-composition` | 4 | pytorch-module-mechanics | 2,7,8,9 |
| `conv-padding-zero` | 4 | convolution | 20,21,22,23 |
| `conv-stride-downsample` | 4 | convolution | 7,21,22,23 |

**Spine 1 — strided memory (contiguous-layout + as-strided-windowing):** co-occur in 10 exercises, fire together every time. These are the "physical layer" of the part-2 convolution track.

**Spine 2 — nn-module-subclass:** fires in 9 of the 11 Module exercises (absent only from the plain-function exercises 17, 18, 21, 22, 23). Co-occurs with module-composition or nn-parameter-wrap in every case.

**Spine 3 — the 1D/2D conv windowed-einsum cluster:** (`einsum-contraction` + `as-strided-windowing` + `conv-windowing-{1d,2d}` + `conv-output-shape` + `conv-channel-sum` + `conv-kernel-shape`) co-fire in 17, 18, 19, 21, 22. This is the densest co-occurrence pattern in part-2.

**Singletons (1 exercise only):** `topk-predictions`, `state-dict-load`, `replace-final-head`, `matvec`, `matmul-2d`, `maxpool-reduce`, `functional-module-wrap`, `batchnorm-running-stats`, `batchnorm-affine-params`, `tensor-reshape-view`, `register-buffer`, `residual-skip-add`, `1x1-conv-channel-reshape`, `resnet-stem`. **14 of 40 atoms are singletons** — most are plausibly recurring in later chapters (residual-skip-add will reappear in Transformer blocks, batchnorm atoms in other normalizers, state-dict-load is universal). Keep all.

---

## Domain breakdown (where the work actually is)

| domain | atom-tags | % | comment |
|---|---|---|---|
| strided-memory | 25 | 21% | dominated by the contiguous-layout + as-strided-windowing pair |
| convolution | 23 | 20% | densely co-fires with strided-memory |
| pytorch-module-mechanics | 20 | 17% | the Module subclass + Parameter + composition spine |
| tensor-algebra | 13 | 11% | einsum + broadcasting + matmul/matvec |
| training-infra | 7 | 6% | 3 atoms, mostly exercise 3 + 12 |
| normalization | 5 | 4% | exercise 5 (3 atoms) + train-eval-mode in 10, 12 |
| architecture-composition | 5 | 4% | the ResNet-specific atoms |
| activation | 4 | 3% | just relu-elementwise-max |
| transfer-learning | 4 | 3% | exercise 11 + 12 |
| pooling | 3 | 3% | avgpool x2 + maxpool x1 |

Strided-memory + convolution = 41% of all atom tags. The seed correctly anticipated this concentration.

---

## Delta Drills coverage

Across 110 known-atom tags (4 NEW-atom tags excluded):

| DD code | tags | % | vs vocab-level |
|---|---|---|---|
| DD-N (no drill coverage) | 55 | 50% | vocab-level: 64% — actual tag distribution favors covered atoms slightly |
| DD-? (partial / uncertain) | 37 | 34% | vocab-level: 22% — actual tags hit the "needs audit" cluster more |
| DD-Y (covered) | 17 | 16% | vocab-level: 14% |

**84% of real atom-tags point to atoms Delta Drills doesn't fully drill (DD-N or DD-?).** Even more skewed than the batch-1 figure (73%) because the strided-memory + convolution-mechanics cluster (all DD-?) dominates and most Module-mechanics atoms are DD-N.

**The gap question is sharper now:** the DD-? atoms (strided-memory, conv-mechanics) ARE drillable as pure array ops — they're the natural extension of the existing einops/numpy drill set. The DD-N atoms (Module mechanics, BatchNorm, training loop, transfer learning, architecture composition) are NOT array ops. Two natural cohorts:

1. **Extend the drill set into strided/conv-mechanics drills.** This unlocks ~34% of tags immediately. Natural drill format: "given input shape and target shape, write the (size, stride) tuple for as_strided"; "given a conv signature, write the einsum string".
2. **Decide separately on the Module-mechanics/training-loop cohort.** These ~50% of tags need either (a) a new drill format (e.g. "fix this broken Module"), (b) docs-routing, or (c) explicit `drillable: false`.

---

## Recommendations

### 1. Accept all 4 proposed atoms into vocab/atoms.json (36 -> 40).

```json
{"id": "kaiming-uniform-init", "label": "Kaiming uniform init", "definition": "Sample uniformly with bound ±1/sqrt(fan_in); the default init pattern for Linear/Conv weight and bias.", "domain": "pytorch-module-mechanics", "dd_coverage": "DD-?", "status": "accepted"}
{"id": "register-buffer", "label": "register_buffer for non-learnable state", "definition": "Use self.register_buffer(name, tensor) for state that participates in state_dict but not in optimization (running stats, attention masks, position encodings).", "domain": "pytorch-module-mechanics", "dd_coverage": "DD-N", "status": "accepted"}
{"id": "conv-kernel-shape", "label": "conv kernel layout", "definition": "Weight tensor shape (out_channels, in_channels, kernel_dims...); 1D=(oc,ic,kw), 2D=(oc,ic,kh,kw).", "domain": "convolution", "dd_coverage": "DD-N", "status": "accepted"}
{"id": "functional-module-wrap", "label": "Module wraps F.* op", "definition": "Module is a parameter container; forward delegates to torch.nn.functional.* with self.weight as kernel; pattern for Conv2d, BatchNorm2d, MaxPool2d wrappers.", "domain": "pytorch-module-mechanics", "dd_coverage": "DD-N", "status": "accepted"}
```

### 2. Re-derive the "expected exercises" column from these 24 tagged files.

The seed's column is wrong about ~20% of cells (the false-positive set). The accurate version is now derivable mechanically by inverting `exercises/*.json`. This eliminates a future source of confusion when extending to chapter 1 (`transformer-interp`).

### 3. Granularity verdict: APPROXIMATELY RIGHT.

- 40 atoms for 24 exercises = 1.67 exercises per atom on average. 14 singletons (out of 40) is on the higher end of normal; most are plausibly cross-chapter recurring.
- Mean 4.88 atoms/exercise lands cleanly in the 4-7 target band.
- No merges suggested. Possible split (deferred): `conv-windowing-1d` vs `conv-windowing-2d` could merge into `conv-windowing-Nd` — but they fire in disjoint exercise sets and the seed's split is informative for the prereq DAG.

### 4. Prereq-DAG edges to encode (next pass, not blocking).

Based on co-occurrence + exercise order, the following edges are obvious:

- `contiguous-layout` -> `as-strided-windowing` (need stride concept before windowing)
- `as-strided-windowing` -> `stride-zero-broadcast` -> `diagonal-via-strides` (strided tricks build on each other)
- `as-strided-windowing` + `einsum-contraction` -> `conv-windowing-1d` -> `conv-windowing-2d`
- `conv-windowing-2d` + `conv-padding-zero` + `conv-stride-downsample` -> full `conv2d`
- `nn-module-subclass` -> `nn-parameter-wrap` -> `module-extra-repr` -> `module-composition` (the Module spine)
- `nn-module-subclass` + `batchnorm-{running-stats,affine-params}` + `register-buffer` + `train-eval-mode-branch` -> BatchNorm2d cluster
- `dataloader-batching` + `training-step-cycle` -> `validation-no-grad` -> `train-eval-mode-branch` (for batchnorm-aware training)
- `state-dict-load` + `freeze-requires-grad` + `replace-final-head` -> transfer-learning composite

### 5. Stop the per-exercise tagging here; next blockers are decisions, not data.

- Accept the 4 proposed atoms (mechanical edit to atoms.json).
- Decide the gap question (DD-N cohort: extend drills, route to docs, or mark non-drillable).
- THEN consider tagging chapter 1 (transformer-interp) to test cross-chapter recurrence of singletons.

---

## What changed from batch 1's report

- **Mean atoms/exercise:** 6.5 -> 4.88 (more single-concept exercises in the strided/utility cluster pulled the mean down)
- **FP rate:** 1.75 -> 0.83 (the tagger had calibration to lean on; also fewer exercises that "mention everything" like the early Module ones)
- **New-atom rate:** 1.0 -> 0.17 (decayed to near zero — strong vocab convergence signal)
- **DD-N share:** 73% -> 50% in batch 2 alone because the strided/conv drills (DD-?) dominate batch 2. The combined figure (84% DD-N+?) is the right headline for vocab-level planning.
- **No new atoms surfaced beyond batch 1's four.** The seed + 4 proposed = 40 atoms is the steady-state vocab for `part2_cnns`.
