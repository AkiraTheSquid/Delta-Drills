# Encompassing-graph primary sources

Corpus collected 2026-05-26 for "hook up the encompassing graph" work.
Backs the architectural decision to add `edge_type` (`requires` /
`encompasses` / `co_occurs`) + `propagation_weight` to the curriculum
graph, plus a mastery-propagation function on top of EWMA (or whichever
estimator sits behind the `MasteryEstimator` interface seam).

## Read in this order

### Tier 1 — Math Academy canonical (the encompassing-graph source)

| # | File | What it backs |
|---|---|---|
| 1 | `MAW_Ch04_Knowledge_Graph.pdf` | THE encompassing-vs-prerequisite distinction, knowledge-graph topology, multi-tag KC semantics |
| 2 | `MAW_Ch13_Mastery_Learning.pdf` | Mastery threshold semantics, why partial credit fails |
| 3 | `MAW_Ch21_Targeted_Remediation.pdf` | How encompassing edges drive remediation routing |
| 4 | `MAW_Ch30_Diagnostic_Exams.pdf` | Adaptive diagnostic using the graph (encompassing skip-ahead) |
| 5 | `MAW_Ch31_Learning_Efficiency.pdf` | Implicit repetitions from encompassing edges (FIRe-style trickle-down) |
| 6 | `2023_skycak_fire-spaced-repetition.html` (in parent dir) | FIRe formulas verbatim — encompassing edge weight + trickle-down |
| 7 | `2024_skycak-roberts_the-math-academy-way.pdf` | Companion academic paper (working draft) |
| 8 | `2023_skycak_optimized-individualized-spaced-repetition-hierarchical.html` | Blog version of the FIRe + hierarchy paper |
| 9 | `2025_mathacademy_how-our-ai-works.html` | Official Math Academy AI overview |
| 10 | `2025_hecker_math-academy-part-7-technology-brief.html` | Outsider write-up — KG mechanics explained |

### Tier 2 — Math Academy commentary / triangulation

| # | File | What it adds |
|---|---|---|
| 11 | `2024_matuschak_notes-math-academy.html` | Andy Matuschak's notes on Math Academy (independent analysis) |
| 12 | `2024_skycak_hn-thread-math-academy.html` | Skycak HN AMA — direct answers on graph design |
| 13 | `2024_stokke-smith_chalk-and-talk-ep-42-transcript.html` | Skycak interview — informal explanations of FIRe |
| 14 | `2025_oznova_balanced-review-of-math-academy.html` | Critical review — failure modes |

### Tier 3 — Propagation-mechanism literature (NOT Math Academy)

Math Academy keeps the algorithm proprietary. These are open papers
covering the same propagation problem:

| # | File | Why it matters for propagation |
|---|---|---|
| 15 | `2017_kaser-klingler-schwing-gross_dynamic-bayesian-networks-student-modeling.pdf` | **Most directly relevant.** Bayes-net propagation across a skill graph — the principled "encompassing → child" diffusion paper. TLT 2017 |
| 16 | `2006_cen-koedinger-junker_learning-factors-analysis.pdf` | LFA — KC-decomposition basis. How multi-tag credit decomposes empirically |
| 17 | `2014_khajah-huang-gonzales-mozer-brusilovsky_kt-irt-tale-two-frameworks.pdf` | KT ↔ IRT equivalence — relevant if propagation through latent traits |
| 18 | `2015_pelanek_modeling-students-memory-adaptive-educational.pdf` | Memory-decay calibration in deployed systems (geography data, Czech adaptive system) |
| 19 | `2021_cosyn-uzun-doble-matayoshi_aleks-kst-practical.pdf` | ALEKS Knowledge Space Theory — encompassing's closest open competitor; how KST handles "if you know A you know B" |
| 20 | `2019_doble-matayoshi_aleks-kst-reliability-simulation.pdf` | ALEKS reliability — simulation methodology |

### Tier 4 — Edge-construction methodology (for `edge_type` authoring)

The graph already has edges with `epistemic_confidence`. These cover how
to derive *typed* edges from corpora:

| # | File | Use |
|---|---|---|
| 21 | `liang-RefD-2015.pdf` | Reference Distance — directional prereq signal |
| 22 | `pan-MOOC-prereq-2017.pdf` | MOOC prereq edge extraction |
| 23 | `le-abel-ESCO-skills.pdf` | Skill-ontology approach (industry baseline) |
| 24 | `hu-pan-PDRS-cold-start.pdf` | Cold-start prereq discovery |
| 25 | `EdgeMatch_audit_cross-source-prereq-edge-corroboration.pdf` | The cross-source corroboration method used in iter-5 |
| 26 | `learning-content-from-online-educational-data.pdf` | Edge construction from clickstream |

## What the parent `primary-sources/` already covers (DON'T re-read here)

These live one directory up; they back the *mastery-estimation* side of
the architecture (Elo / BKT / PFA / Bijl), not the *graph* side:

- `2016_pelanek_elo-adaptive-educational.pdf`
- `2017_pelanek_bkt-logistic-and-beyond.pdf`
- `2019_park-cornillie_multidim-irt-monitoring-ability.pdf`
- `2021_maier-baker-stalzer_pfa-challenges.pdf`
- `2025_bijl_probabilistic-decay-trees.pdf`

## How this connects to Delta Drills

- `concept_graph.py` — `PrerequisiteEdge` needs `edge_type` field per
  Tier 1 (#1, #5, #6) + propagation function per Tier 3 (#15)
- `practice_engine.py` — current EWMA is per-subtopic, not per-atom;
  Tier 3 #16 (LFA) + #17 (KT-IRT) frame the multi-tag credit question
- `iter-5` graph already has `epistemic_confidence` — Tier 4 covers the
  methodology that produced those values; adding `edge_type` follows
  the same dual-pass LLM + cross-source pattern

## What's still off-disk

- **Reddy, Levine, Dragan 2017 "Accelerating Human Learning with Deep
  RL"** — propagation-as-policy angle, NOT downloaded; arXiv:1612.02238
- **Pavlik PFA 2009** — covered in mastery-estimation parent dir
  context, not here. Pull if propagation discussion needs PFA's
  per-skill credit rule
