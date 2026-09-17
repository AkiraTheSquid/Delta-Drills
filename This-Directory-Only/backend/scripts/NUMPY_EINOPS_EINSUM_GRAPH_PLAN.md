# Fine-Grained NumPy / Einops / Einsum Prerequisite Graph — Design (#25)

_2026-07-07. Companion to `~/Documents/Delta-Drills Question Quality Plan.md` items #25/#26._

## Problem (measured)

380 of 455 bank questions (84%) sit on Numpy/Einops/Einsum topics, but their atom tags
collapse onto a handful of ARENA chap0 drill-atoms:

- **Einsum: 67/68 questions → 1 atom** (`einops-einsum`). One mastery scalar for the whole topic.
- **Einops: 92 tags → 3 atoms** (`einops-rearrange` 63, `einops-reduce` 19, `einops-repeat` 10).
- **Numpy: 221 qs → 59 atoms** but top-heavy (`broadcasting-rules` 33, `any-reduce-axis` 20,
  `integer-array-indexing` 17) and the atoms are ARENA-context borrowings, not a numpy progression.

Graph topology under these topics is a star: `broadcasting-rules` out-degree 16,
`einops-reduce` 10, `einops-rearrange` 7, `einops-repeat` 6, `einops-einsum` 5 — all
in-degree ≈ 0, edges only *outward* to advanced ARENA nodes, zero internal structure.
Consequences: all 4 einsum subtopic_keys share one atom → identical mastery → one
difficulty scalar over 68 questions; prereq gating cannot route within these topics;
"weakest reachable subtopic" is noise exactly where most practice happens.

Engine is NOT the problem: `bkt_mastery.apply_attempt` is already per-atom with FIRe
encompassing credit. Only the vocabulary (atoms + edges + tags) is too coarse.

## Deliverable

1. ~52 new fine-grained atoms (below) added to the concept graph.
2. ~120 new edges: intra-family prereq chains + cross-family + integration edges into
   the 5 existing hub atoms.
3. Re-tagged `question_atom_tags.jsonl` for the 380 questions (other 75 rows untouched).
4. New merged graph `arena_drillable_v2.json`; flip `bkt_mastery.py` `GRAPH_PATH`.

## Atom taxonomy (grounded in bank content — counts from prototype rule-tagger)

Sizes from a regex tagger over canonical solutions; final tags get a manual pass on
~30 uncategorized/ambiguous ids. Multi-tag is expected (81/380 in prototype).

### NumPy — topic "Numpy Foundations" (~30 atoms)

**Array model & creation**
| atom | ~qs | covers |
|---|---|---|
| `np-array-from-list` | 5 | `np.array` literals, nested lists → ndarray |
| `np-arange-linspace` | 10 | range construction (62 solution uses overall) |
| `np-array-init` | 10 | zeros/ones/full/empty/zeros_like |
| `np-eye-diag` | 11 | identity, diagonal construct/extract |
| `np-dtype-astype` | ~12* | dtype awareness, casting (*rule over-fires at 31; astype often incidental — demote to secondary tag conf 0.4) |
| `np-random-arrays` | 6 | rand/randint/choice as task (not fixture) |
| `np-copy-vs-view` | 4 | `.copy()`, view semantics, shared storage |

**Axes & shape transforms**
| `np-reshape` | 28 | reshape, -1 inference |
| `np-transpose-swapaxes` | 10 | `.T`, transpose(perm), swapaxes |
| `np-flatten-ravel` | 2 | flatten vs ravel |
| `np-newaxis-expand` | 10 | `[:, None]`, np.newaxis, expand_dims |
| `np-concat-stack` | 3 | concatenate/stack/hstack/vstack |
| `np-tile-repeat` | 10 | tile, repeat |

**Indexing (progression)**
| `np-slicing` | 14 | 1-D/2-D slices, step, reverse |
| `np-boolean-mask-select` | ~8* | `a[a>0]` (*rule under-fires at 3; widen patterns) |
| `np-boolean-mask-assign` | ~6* | `a[mask] = v` in-place |
| `np-fancy-index` | 32 | integer-array indexing, gather, row/col select |
| `np-where-select` | 10 | np.where 3-arg + 1-arg |
| `np-sort-argsort-partition` | 14 | sort/argsort/argpartition, top-k |
| `np-argmax-argmin` | 12 | position-of-extreme (+ unravel_index) |

**Broadcasting (progression)**
| `np-elementwise-arith` | 8 | scalar-array, same-shape ops, ufuncs |
| `np-broadcast-same-rank` | ~10 | trailing-dim rule, (3,1)*(1,4) |
| `np-broadcast-newaxis-outer` | 10 | rank promotion for outer-style ops |

**Reductions**
| `np-reduce-global` | 11 | sum/mean/max over all elements |
| `np-reduce-axis` | 24 | axis kwarg (50 uses overall) |
| `np-reduce-keepdims` | 9 | keepdims=True, normalize-by-row patterns |
| `np-any-all-logical` | 10 | any/all, logical reductions |
| `np-cumsum-diff` | 12 | cumulative + finite difference |
| `np-unique-bincount` | 14 | unique, counts, bincount |
| `np-nan-handling` | 6 | isnan, nan-aware reductions |

**Applied**
| `np-linalg-basics` | 10 | solve/inv/det/norm |
| `np-stride-tricks` | 7 | as_strided windows (bridges to existing `as-strided-windowing`) |

### Einsum — topic "Einsum Notation" (~13 atoms, notation-complexity progression)

| atom | ~qs | signature class |
|---|---|---|
| `es-notation-read` | 3 | identity / read-a-spec `'i j -> i j'` |
| `es-transpose-permute` | 3 | `'i j -> j i'`, `'b c h w -> b h w c'` |
| `es-reduce-sum-axis` | 8 | dropped output index = sum: `'i j -> i'`, `'b t d -> b d'` |
| `es-diagonal-trace` | 4 | repeated index one operand: `'i i -> i'`, `'b i i -> b'` |
| `es-elementwise-hadamard` | 4 | `'i, i -> i'` |
| `es-inner-product` | 4 | `'i, i ->'`, `'i j, i j ->'` |
| `es-outer-product` | 4 | `'i, j -> i j'`, batched outer |
| `es-matvec` | 2 | `'i j, j -> i'` |
| `es-matmul-contraction` | 20 | `'i k, k j -> i j'` + 2-operand contractions |
| `es-batched-matmul` | 13 | `'b i k, b k j -> b i j'`, attention QK/AV patterns |
| `es-broadcast-batch-mixed` | 4 | rank-mismatched operands `'b t d, d -> b t'` |
| `es-multi-operand-chain` | 5 | 3+ operands `'i k, k m, m p -> i p'` |

### Einops — topic "Einops Patterns" (~9 atoms)

| atom | ~qs | pattern class |
|---|---|---|
| `eo-pattern-read` | — | axis naming/matching (prereq-only atom; tagged secondary on all eo qs conf 0.3) |
| `eo-rearrange-transpose` | 11 | plain permutation `'h w c -> c h w'` |
| `eo-rearrange-compose-axes` | 20 | merge `'b h w c -> h (b w) c'` |
| `eo-rearrange-decompose-axes` | 13 | split `'(g n) c -> g n c'` + kwargs |
| `eo-rearrange-compose-decompose` | 23 | both sides; grid/patch shuffles |
| `eo-reduce-basic` | 15 | reduce with op |
| `eo-reduce-keep-axis` | 4 | `'h w c -> h () c'` |
| `eo-repeat-new-axis` | 2 | new axis via repeat |
| `eo-repeat-expand-inline` | 8 | `'c h w -> c (h 3) w'` |

## Edge design (~120 edges)

Edge record format identical to v1: `{prerequisite_id, dependent_id, weight, confidence,
is_hard_gate, is_encompassing, propagation_weight, rationale}`. Direction: simpler = prereq.
`is_encompassing=true` only where doing the dependent demonstrably exercises the prereq
(is-a / part-of / refines — per the v4 subset-model correction), else pure prereq.

**Intra-NumPy chains** (arrows = prereq→dependent):
- creation: `np-array-from-list` → `np-arange-linspace`, `np-array-init`; `np-array-init` → `np-eye-diag`
- shape: `np-reshape` → `np-flatten-ravel`(enc), `np-concat-stack`; `np-newaxis-expand` → `np-tile-repeat`
- indexing ladder: `np-slicing` → `np-boolean-mask-select` → `np-boolean-mask-assign`(enc);
  `np-slicing` → `np-fancy-index`; `np-boolean-mask-select` → `np-where-select`;
  `np-argmax-argmin` → `np-sort-argsort-partition`
- broadcasting ladder: `np-elementwise-arith` → `np-broadcast-same-rank` → `np-broadcast-newaxis-outer`;
  `np-newaxis-expand` → `np-broadcast-newaxis-outer`(enc)
- reductions ladder: `np-reduce-global` → `np-reduce-axis` → `np-reduce-keepdims`(enc);
  `np-reduce-axis` → `np-any-all-logical`(enc), `np-cumsum-diff`, `np-argmax-argmin`
- applied: `np-fancy-index`+`np-reduce-axis` → `np-unique-bincount`; `np-reduce-axis` → `np-nan-handling`;
  `np-slicing` → `np-stride-tricks`

**Intra-Einsum chain**:
`es-notation-read` → `es-transpose-permute`, `es-reduce-sum-axis` → `es-diagonal-trace`;
`es-elementwise-hadamard` → `es-inner-product` → `es-matvec` → `es-matmul-contraction`
→ `es-batched-matmul` → `es-multi-operand-chain`; `es-inner-product` → `es-outer-product`;
`es-reduce-sum-axis` + `es-elementwise-hadamard` → `es-inner-product`(enc both);
`es-matmul-contraction` → `es-broadcast-batch-mixed`.

**Intra-Einops chain**:
`eo-pattern-read` → `eo-rearrange-transpose` → `eo-rearrange-compose-axes` →
`eo-rearrange-decompose-axes`(sibling, both ← compose? No: compose → decompose, then both) →
`eo-rearrange-compose-decompose`(enc over compose+decompose); `eo-pattern-read` → `eo-reduce-basic`
→ `eo-reduce-keep-axis`(enc); `eo-rearrange-compose-axes` → `eo-repeat-expand-inline`;
`eo-pattern-read` → `eo-repeat-new-axis`.

**Cross-family** (numpy grounds the DSLs):
- `np-transpose-swapaxes` → `eo-rearrange-transpose`, `es-transpose-permute`
- `np-reshape` → `eo-rearrange-compose-axes`, `eo-rearrange-decompose-axes`
- `np-reduce-axis` → `eo-reduce-basic`, `es-reduce-sum-axis`
- `np-broadcast-newaxis-outer` → `eo-repeat-new-axis`, `es-outer-product`, `es-broadcast-batch-mixed`
- `np-tile-repeat` → `eo-repeat-expand-inline`
- `np-eye-diag` → `es-diagonal-trace`

**Integration into existing hubs** (fine atoms become prereqs of the ARENA hubs;
hub encompasses its fine atoms so ARENA practice FIRe-credits them):
- `eo-rearrange-compose-decompose` → `einops-rearrange` (enc)
- `eo-reduce-basic` → `einops-reduce` (enc)
- `eo-repeat-expand-inline` → `einops-repeat` (enc)
- `es-batched-matmul` → `einops-einsum` (enc)
- `np-broadcast-newaxis-outer` → `broadcasting-rules` (enc)
- `np-stride-tricks` → `as-strided-windowing` (enc)

Existing hub out-edges to ARENA nodes untouched → new atoms gain transitive reachability
into the whole ARENA graph without touching 330 existing edges.

## Re-tagging method

1. Rule-based tagger (prototype validated 2026-07-07: 364/380 categorized, 16 uncategorized,
   81 multi-tag): regex features over `canonical_solution` + einsum-signature parser +
   einops-pattern parser. Primary tag conf 0.85, secondary 0.5.
2. Fix known rule gaps: `np-dtype-astype` over-fire (demote to secondary unless prompt
   mentions dtype), boolean-mask under-fire (widen to `mask =` assignments + `np.where(cond)` 1-arg).
3. Manual pass over uncategorized ids (16) + a 30-question random audit sample.
4. Output: rewrite the 380 rows in `question_atom_tags.jsonl`; leave 75 other rows byte-identical.
5. Old hub tags on these questions are REPLACED (hub mastery still fed via FIRe from
   integration edges + arena_chap0 problem_links).

## File plan

| file | action |
|---|---|
| `app/data/concept_graphs/arena_drillable_v2.json` | NEW = v1 + 52 nodes + ~120 edges (v1 untouched, rollback = flip path back) |
| `app/bkt_mastery.py:53` | flip `GRAPH_PATH` to v2 |
| `app/data/question_atom_tags.jsonl` | rewrite 380 rows |
| `scripts/build_numpy_einops_graph.py` | NEW builder: atoms+edges declared as data, emits v2, runs validations |
| `scripts/retag_bank_questions.py` | NEW tagger (rules + manual-override file `retag_overrides.jsonl`) |
| `display_graph.json` / graph-viz | regenerate via existing `export_display_graph.py` |

`question_atom_tags.jsonl` is static backend data (only `questions.py` reads it; not
regenerated by deploy) — safe to rewrite. `prereq_subtopics.json` untouched (Colab-drill
atoms only; subtopic→atom mapping for the bank derives from question tags at runtime via
`get_atoms_for_subtopic`).

## Migration / compatibility

- Existing user mastery keyed by atom id — old atoms all persist in v2, no data loss.
- New atoms start at BKT prior; first sessions re-estimate. Self-report prior seeding
  (`p_init` 0.02/0.10/0.45) applies to new atoms automatically.
- Subtopic mastery = mean over tagged atoms → einsum subtopics now differentiate
  (4 subtopic_keys × ~13 atoms instead of ×1).

## Validation gates (all must pass before GRAPH_PATH flip)

1. DAG-clean: full graph + encompassing subgraph (reuse v1 checks in builder).
2. `concept_graph.load_curriculum_graph` loads v2.
3. Every new atom referenced by ≥1 question tag OR is an explicit prereq-only atom (`eo-pattern-read`).
4. Every one of the 380 questions has ≥1 tag; no tag references a missing node.
5. Backend test suite green.
6. Reachability sanity: beginner ordering still opens numpy basics first
   (guard from 2026-07-05 `6f7d0aa` fix); print top-10 reachable subtopics before/after.
7. Coverage report: atoms/question histogram, questions/atom histogram (no atom >35 qs, none 0).

## Rollout

1. `backup_delta_drills_local` checkpoint first.
2. Build + validate locally; run backend tests; eyeball display-graph regen.
3. Local live-check: adaptive session on Numpy/Einsum — confirm target difficulty now
   varies across einsum subtopics and prereq gating references fine atoms.
4. Deploy via standard `deploy_delta_drills` (Fly picks up backend data; Vercel viz).

## Relation to #26

This graph defines the state space #26's KST/CAT engine needs. Atom granularity here
(~52) is deliberately at "one decidable skill per atom" so knowledge-state inference has
usable resolution. Difficulty re-rate (deferred phase 4) supplies the IRT item params later.
