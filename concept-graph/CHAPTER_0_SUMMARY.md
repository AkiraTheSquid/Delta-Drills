# Chapter 0 Fundamentals — Concept Graph Summary

**Pipeline run**: 2026-05-18 (autonomous mode, defaults applied throughout).
**Coverage**: all 5 parts of ARENA chapter0_fundamentals, 79 exercises tagged.

---

## Totals

| metric | value |
|---|---|
| Vocab size | **214 atoms** (started 36 part2-seed; 213 accepted, 1 intentionally-orphan seed) |
| Exercises tagged | **79** (part1: 10, part2: 24, part3: 12, part4: 21, part5: 12) |
| Total atom-tags | **596** |
| Mean atoms/exercise (chapter avg) | **7.54** |
| Review-queue items | **52 questions** |

---

## Per-part breakdown

| part | domain | exercises | atoms native | tags | mean/ex | DD-Y/?/N | FP rate |
|---|---|---|---|---|---|---|---|
| 1 | ray tracing | 10 | 25 | 79 | 7.90 | 25 / 34 / 41 | 0.20 |
| 2 | CNNs | 24 | 36 (seed) +4 proposed | 117 | 4.88 | 14 / 22 / 64 | 0.83 |
| 3 | optimization | 12 | 62 | 158 | 13.17 | 0 / 22 / 78 | 0.58 |
| 4 | backprop from scratch | 21 | 54 | 136 | 6.48 | 4 / 1 / 95 | 0.29 |
| 5 | VAEs/GANs | 12 | 33 | 106 | 8.83 | 6 / 8 / 87 | 0.33 |

**DD-Y collapse**: ratio falls from 25% (part1, einops-heavy) → 0% (part3, optimizer math) → 4% (part4, autograd) → 6% (part5, generative). Confirms vocab cannot be covered by the current numpy/einops drill set alone — math-practice pipeline (planned separately) is the right home for ~70% of chapter 0 atoms.

**Density**: highest mean (part3: 13.17) is the optimizer-impl + DDP wiring stack. Densest single exercise: `0_3_11` DistResNetTrainer at 32 atoms.

---

## Cross-part atom reuse pattern

| reuse pattern | observed |
|---|---|
| Part1 atoms reused in 2+ | broadcasting-rules, einsum-contraction (only 2 of 25) |
| Part2 atoms reused in 2+ | conv-windowing-1d/2d, kaiming-uniform-init, conv-padding-zero, training-step-cycle, train-eval-mode-branch, etc. |
| Part3 atoms reused in 2+ | training-step-cycle (5×), tensor-to-device (5×), wandb cluster (in part5 too) |
| Part4 atoms cross-part | 0 — autograd vocab is isolated (it builds something the other parts already assume) |
| Part5 cross-part reuses | 8 explicit + 15 incidental = **42% of part5 unique tags are reuses** — highest integration ratio in the chapter |

**Lesson**: parts 2 and 5 are integration parts (compose prior atoms). Parts 1, 3, 4 are domain-introduction parts (own ~all of their atoms). This suggests the prereq DAG will be densely connected through parts 2 and 5, sparsely through parts 1/3/4.

---

## Calibration quality

| metric | value | comment |
|---|---|---|
| New atoms surfaced during tagging | **4 total** (all in part2 batch 1) | After part2's 4 proposals, vocab reached steady state — parts 3/4/5 vocabs surfaced 0 new atoms during tagging |
| Mean FP rate | 0.45/exercise | Part2 (topic-extrapolated seed): 0.83. Part1+3+4+5 (full-notebook-read seed): 0.35 |
| Seed method that worked | **hand-draft from full notebook read** (vs topic-extrapolation) | 4× FP improvement |

The 4 part2-proposed atoms (`kaiming-uniform-init`, `register-buffer`, `conv-kernel-shape`, `functional-module-wrap`) were **accepted into atoms.json** during the post-pipeline audit (2026-05-18). All atom references now resolve — validator is green.

---

## DD coverage decisions for math-practice pipeline

User confirmed (2026-05-18) a separate pipeline will cover math practice. Atoms that map directly to that pipeline (per-op gradient rules, optimizer math, VAE/GAN losses) are concentrated:
- **Part3 §2** (optimizer math, 9 atoms): SGD/Adam/AdamW update rules — pure tensor identities, 1-line drills
- **Part4 §2** (per-op derivatives, 11 atoms): log_back/multiply_back/etc — uniform-shape gradient identities
- **Part5 §2** (VAE math, 5 atoms) + **§4** (GAN training, 6 atoms): KL divergence, reparam trick, BCE-on-fake

Total: **31 atoms (15%)** are pre-allocated for the math-practice pipeline.

---

## Review queue

52 questions in `/home/stellar-thread/Applications/Delta-Drills-Local/concept-graph/REVIEW_QUEUE.md`:
- Part 3 vocab: Q1-Q10 (granularity calls)
- Part 3 tagging: Q11-Q17 (FP validations, cross-part reuse audit)
- Part 4 vocab: Q1-Q13 (under part 4 section)
- Part 4 tagging: Q14-Q18
- Part 5 vocab: Q1-Q13 (under part 5 section)
- Part 5 tagging: Q19-Q22

**No question blocked the pipeline.** All defaults were taken and documented inline.

---

## Files produced

**Vocab drafts** (`/home/stellar-thread/Documents/coding-ideas/`):
- `arena_part1_seed_atoms.md`
- `arena_part2_seed_atoms.md`
- `arena_part3_seed_atoms.md`
- `arena_part4_seed_atoms.md`
- `arena_part5_seed_atoms.md`

**Live vocab**: `/home/stellar-thread/Applications/Delta-Drills-Local/concept-graph/vocab/atoms.json` (210 atoms)

**Tagged exercises**: 79 files in `/home/stellar-thread/Applications/Delta-Drills-Local/concept-graph/exercises/0_*.json`

**Calibration reports** (per part, under `exercises/`):
- `CALIBRATION_REPORT.md` (part2 batch 1)
- `CALIBRATION_REPORT_BATCH2.md` (part2 batch 2)
- `CALIBRATION_REPORT_PART1.md`
- `CALIBRATION_REPORT_PART3.md`
- `CALIBRATION_REPORT_PART4.md`
- `CALIBRATION_REPORT_PART5.md`

**Review queue**: `/home/stellar-thread/Applications/Delta-Drills-Local/concept-graph/REVIEW_QUEUE.md`

**This summary**: `/home/stellar-thread/Applications/Delta-Drills-Local/concept-graph/CHAPTER_0_SUMMARY.md`

---

## What's next (open decisions for user)

1. **Review the 52 queued questions** in REVIEW_QUEUE.md — most are merge/split granularity calls + cross-part reuse audits. Apply changes to vocab if any defaults disagree with intent.
2. **Lock chapter 0 vocab** (after review). Then proceed with the original part-at-a-time plan: build prereq DAG within each part, then cross-part edges within the chapter.
3. **Decide drillable status** (schema does not yet have a `drillable` field — add it if you accept any of these): 3 atoms flagged as candidates for `drillable: false`:
   - `ray-parametric-form` (part1, conceptual scaffold tagged on all 10 exercises but 5/7 incidental)
   - `trainer-subclass-extend` (part3, tagged once in 0_3_7 — Q14 in tagging queue)
   - `pseudocode-to-code-translate` (part3 §2.9, 4× — meta-atom, may belong to math-practice pipeline rather than concept graph)
4. **Decide on the orphan**: `model-train-eval-toggle-around-sample` (part5 §8) was added but never tagged. Status `seed`. Either keep speculatively for chapter 1/2 (current default) or remove.
5. **Chapter 1 (transformers)**: estimated ~80 new atoms based on the per-part trend (transformers will introduce attention math + tokenization + sampling). The cross-part reuse rate should jump once parts 2 + 3 + 4 + 5 atoms are available as a base.

## Audit fixes applied (post-pipeline, 2026-05-18)

- Accepted 4 part2-proposed atoms into atoms.json (`kaiming-uniform-init`, `register-buffer`, `conv-kernel-shape`, `functional-module-wrap`). Validator: green.
- Upgraded `status: seed` → `accepted` for 209 atoms that fired in ≥1 tagged exercise. Final: 213 accepted, 1 seed (the orphan above).
- Removed duplicate `## Part 4 / tagging` section header in REVIEW_QUEUE.md.
- Verified all 79 exercise JSONs schema-valid and all 214 vocab atoms schema-valid. Zero duplicate atom IDs. Zero empty-evidence tags. Coverage complete (79/79 notebooks tagged).
