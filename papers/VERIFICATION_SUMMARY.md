# Trust-or-Bust Audit: Mastery-Estimation Research Doc

**Date:** 2026-05-24
**Doc audited:** `compass_artifact_wf-2f4da85b-a0ab-425d-9877-f25787abbdea_text_markdown.md`
**Method:** 4 cluster verifications, sequential subagents, each cross-checked the doc's load-bearing claims against the actual primary sources downloaded to `papers/mastery-estimation/`.

## Overall tally (59 claims across all clusters)

| Verdict | Count | % |
|---|---|---|
| CONFIRMED | 20 | 34% |
| PARTIAL | 20 | 34% |
| DISPUTED | 7 | 12% |
| UNSUPPORTED | 8 | 14% |
| CANT_TELL | 4 | 7% |

**Roughly 1/3 solid, 1/3 half-right, 1/3 substantively wrong or fabricated.** Math Academy claims (Cluster C) are the strongest — verbatim quotes check out, no fabrications. The Bayesian/decay synthesis (Cluster A), PFA recommendations (Cluster B), and architecture validation (Cluster D) contain repeated patterns of misattribution, fabricated paper titles, threshold-as-citation overclaims, and theoretical-authority laundering.

## Headline findings — what's wrong

### 1. The "Settles-Meeder Beta-Bernoulli formulation" is invented
The doc's update rule `α ← α·exp(−Δt/τ) + α₀·(1−exp(−Δt/τ))` appears nowhere in Settles & Meeder 2016. HLR is a discriminative regression on engineered features (eq. 1: `p = 2^(-Δ/h)`, eq. 2: `ĥ = 2^(Θ·x)`, trained by L2 SGD). It has no posterior, no α/β, no conjugate prior. The doc dresses a doc-author synthesis as the published HLR mechanism. **The recommended primitive is unvalidated, not literature-derived.** (Cluster A, claims 1, 3)

### 2. Pelánek 2017's Hypothesis 1 is the OPPOSITE of doc's claim
Doc says Pelánek endorses "Beta-Bernoulli + logistic features dominate at low sample sizes." Pelánek's actual Hypothesis 1 (p. 33): "Logistic models are better for modeling fluency and memory processes, while Bayesian knowledge tracing is better for understanding and sense-making processes." Pelánek treats BKT and logistic as parallel families and never names "Beta-Bernoulli" as a category. **The single most load-bearing citation for the recommended stack misrepresents the source.** (Cluster A, claim 4)

### 3. "Equal credit is catastrophic" is directly contradicted by the only paper that tested it
Doc lists equal-credit propagation as the worst-of-five failure modes to AVOID. Maier 2021 §4.3 empirically compared compensatory/conjunctive/even-skill (= equal credit averaging) and found:
- **Even-skill (equal credit): AUC 0.7849 — BEST**
- Compensatory PFA: AUC 0.7818
- Conjunctive: AUC 0.6725

The doc's "equal credit catastrophic" warning is contradicted by the only empirical test in the audited corpus. (Cluster B, claim 12)

### 4. The two-tier architecture's "theoretical backing" is fabricated
Doc claims the composite/procedural distinction is "implicit in the KC framework as 'high-grain-size KCs' (composites) vs. 'low-grain-size KCs' (procedurals)." This is wrong. In AK13 and KLI, "grain size" refers to the **level of cognitive task** (word identification vs. letter recognition vs. multiplication), NOT items-that-test-multiple-atoms vs. items-that-test-single-atoms. Similarly:
- Rittle-Johnson's "iterative" is about INSTRUCTIONAL ORDER (concepts→procedures, repeated), not per-atom mastery dynamics
- Koedinger-Aleven's "assistance dilemma" is WITHIN-problem (hints/feedback/worked-examples), not BETWEEN-tier

**The architecture is novel; it should be defended on its own merits, not on misappropriated theoretical authority.** Doc admits this on line 29 ("not formally documented as a single named pattern") then contradicts itself across lines 178, 182, 198. (Cluster D, claims 13, 14, 15)

### 5. Fabricated paper titles and attribution errors
- **Bijl 2025:** Doc calls it "Probabilistic Decay Trees." Actual title: "Tracking Student Skills Real-Time Through a Continuous-Variable Dynamic Bayesian Network." Method is called PDT = Performance Distribution Tracing. (Cluster A, claim 7)
- **M-ERS paper:** Doc attributes to "Doebler/Pelánek/Wauters 2018." Actual: Park, Cornillie, van der Maas & Van Den Noortgate 2019. The Park 2019 paper IS in the corpus; doc never cites it by correct author/year. (Cluster B, claim 10)
- **Maier 2021:** Doc gives title "Improving PFA for rare skills." Actual title: "Challenges to Applying PFA to Existing Learning Systems" (ICCE 2021). Doc cherry-picked one of four contributions and named the paper after it. (Cluster B, claim 8)
- **Barrada FI*IG:** Doc cites Barrada 2010. FI*IG is actually in Barrada **2009** (Methodology 5(1)); the 2010 paper compares six different rules NOT including FI*IG. (Cluster D, claim 6)
- **ALEKS "compressed graph cover":** Doc attributes to ALEKS. Actually MA's marketing prose. ALEKS uses likelihood-near-0.5 item selection over a projected/partitioned knowledge structure. (Cluster D, claim 1)

### 6. Numeric thresholds presented as literature-derived are all doc-author defaults
| Threshold | Doc presentation | Reality |
|---|---|---|
| τ = 14 days half-life | "v0 default" | No source |
| α+β ≥ 6 OR 8 (inconsistent) | promotion threshold | No source; doc internally contradicts itself |
| N ≥ 30 for PFA switchover | "implicit consensus" | Maier 2021 data suggests 5–12 |
| Cap tags at 6 | "implicit PFA consensus" | Doc itself says "no published clean threshold" two paragraphs later |
| 25–40 items cold-start | "ALEKS+MA range" | ALEKS uses up to 30 based on student-fatigue feedback, not graph-theoretic argument |
| Cap ≤3 consecutive same-atom | "Rohrer 2012-supported" | Rohrer prescribes no numeric cap |

## What survives clean

These are the claims the user CAN trust:

- **All Math Academy verbatim quotes**: "complexity explodes," "physical" approach, FIRe formula `repNum → max(0, repNum + speed · decay^failed · rawDelta)`, the HOAW diagnostic paragraph, "conditionally completed," "knowledge point" terminology, Oz Nova's "literal years of unnecessary remedial pre-requisites is too great a punishment" — all exact matches against primary sources.
- **PFA's basic logistic formula** (Pavlik 2009 eq. 3): `m = Σ_{j∈KCs}(β_j + γ_j·s_j + ρ_j·f_j); p(m) = 1/(1+e^{-m})`.
- **Recent-PFA's two-timescale framing** (Galyardt & Goldin 2014): R-PFA adds a recency-weighted-success-rate feature alongside cumulative counts. Doc's gloss is accurate.
- **Math equivalence of decayed-α/β to attempt-indexed EWMA**: holds by construction; doc's "EWMA-on-Beta is redundant for static-skill case" reasoning is mathematically right.
- **HLR's actual contribution** (Settles & Meeder 2016): regression on engineered features over 12.9M Duolingo traces, 12% engagement lift, 45%+ MAE reduction over EFC baselines. (Just don't claim this validates the Beta-Bernoulli update rule.)
- **Khajah 2016's headline**: enhanced BKT (with forgetting + skill-discovery + abilities) matches DKT across four standard benchmarks.
- **Doble 2019's empirical scale**: N=742,851 ALEKS PPL assessments.

## Load-bearing impact on the prescribed stack

For each component of the recommended algorithm stack, here's what survives:

| Component | Doc-claimed authority | Audit verdict |
|---|---|---|
| Beta-Bernoulli per-atom posterior | "Settles-Meeder style" | **Author synthesis. Defensible as v0 engineering choice but not literature-validated.** |
| Multiplicative half-life decay | "HLR-style" | Math-equivalent to attempt-indexed EWMA; the specific τ=14d default has no source. |
| Composite EWMA at area level | — | No specific authority needed; standard engineering. |
| PFA/LKT at N≥30 | "PFA literature consensus" | **N=30 threshold has no source. Maier 2021 data suggests 5–12.** |
| Tag-confidence-weighted Beta below threshold | — | Author construct, defensible. |
| Cap tags at 6 | "PFA literature implicit consensus" | **No source. Doc internally contradicts itself.** |
| Event-triggered demotion (2-fail rule) | "Math Academy plateau rule" | **Mostly CONFIRMED**, but Delta Drills' "area" granularity ≠ MA's lesson granularity. Treat as inspired-by, not literal. |
| Sigmoid-on-precision mixing | — | No claimed source; author engineering. |
| Info-gain greedy cold-start | "ALEKS + MA" | **Mis-attribution.** ALEKS uses likelihood-0.5 splitting; MA's "compressed cover" is marketing prose. Direction OK, specifics unsourced. |
| Graph-propagated prior | "as in ALEKS" | **Mis-claim.** Cosyn 2021 describes population-level empirical priors, not per-learner parent/child propagation. |
| Allow non-monotonic mastery | "Yudelson-Pavlik 2013; R-J 2015" | Y-P 2013 not in corpus; R-J is about instructional order, not mastery dynamics. Anti-monotonicity is uncontroversial but the citation chain is fragile. |
| "Two-tier architecture validated by KC framework + R-J" | "implicit in the KC framework" | **FABRICATED.** Architecture is genuinely novel; defend on its own merits. |

## Recommended user actions

1. **Treat the recommended algorithm stack as defensible v0 engineering, not as literature-grounded best practice.** It is one reasonable construction among several.

2. **Drop the "equal credit catastrophic" warning.** The only paper in the corpus that empirically tested it found equal credit (even-skill averaging) was the best of three credit-assignment schemes.

3. **Re-derive numeric thresholds (τ, α+β cutoff, N for PFA switchover, tag cap, cold-start item count, same-atom blocking cap) empirically from Delta Drills' own data.** None are literature-derived.

4. **Stop citing the architecture validation framing.** "Math Academy + KC framework + Rittle-Johnson backs the two-tier design" is not a defensible claim. The composite-EWMA + procedural-Beta + event-triggered-demotion + sigmoid-transition architecture is genuinely novel and should be presented that way.

5. **Fix citations before any user-facing writeup:**
   - Bijl 2025 title → "Tracking Student Skills Real-Time Through a Continuous-Variable Dynamic Bayesian Network" (method = PDT, Performance Distribution Tracing)
   - M-ERS → Park, Cornillie, van der Maas & Van Den Noortgate 2019, Frontiers in Psychology 10:620
   - Maier 2021 title → "Challenges to Applying PFA to Existing Learning Systems"
   - FI*IG → Barrada **2009**, not 2010

6. **Where Math Academy is cited:** distinguish **knowledge-point halt** (intra-lesson, single knowledge point) from **two-failed-lesson plateau** (cross-attempt, lesson level). These are two distinct mechanisms.

7. **Pelánek 2017 cannot be cited as endorsing the recommended stack.** Soften any framing that implies it does.

## What this means for Delta Drills' next phases

- **Phase 4 (drill building) is unaffected.** The Doughty-style KC decomposition + Bloom-tagged exercise structure stands on its own merits independent of the doc's claims.
- **Phase 6 (Beta-Bernoulli backend), if pursued, should be framed as an experiment, not as implementing a "research-validated" architecture.** The math is reasonable; the literature backing is thinner than the doc presents.
- **Phase 8 (cold-start diagnostic) needs its own algorithm design.** "Info-gain greedy on a compressed graph cover" is not an ALEKS algorithm. Either commit to designing one from scratch or use ALEKS's actual likelihood-near-0.5 mechanism.
- **The 7 failure modes the doc enumerates** are partially OK as caution flags (decay-matters, monotonicity-is-anti-pattern) but the specific framings ("equal credit catastrophic," "DKT/BKT won't fit at ~200 atoms") are over-claimed and one is contradicted by empirical data.

## Bottom line

**The user's distrust was justified.** The doc is competent at summarizing what individual papers say at the foundational level (PFA formula, HLR mechanism, BKT identifiability concerns, Math Academy's published vocabulary) but systematically over-claims when synthesizing recommendations. It dresses doc-author engineering judgments as literature consensus, mis-attributes work (Bijl, Park, Barrada, "ALEKS compressed cover"), and fabricates theoretical backing for a novel architecture.

The directional advice is mostly reasonable. The specific numbers, the cited authority for them, and several paper titles are not.
