# Verification — Cluster D: Cold-Start / KLI / Architecture / Interleaving / CAT Claims

**Date:** 2026-05-24
**Audited doc:** `compass_artifact_wf-2f4da85b-a0ab-425d-9877-f25787abbdea_text_markdown.md`
**Reviewer:** Claude Opus 4.7 (1M context) — adversarial audit, Cluster D (Q4/Q5/Q6)

Sources used (all in `papers/mastery-estimation/`):
- **Doble19** = `2019_doble-matayoshi_aleks-kst-reliability-simulation.pdf`
- **Cosyn21** = `2021_cosyn-uzun-doble-matayoshi_aleks-kst-practical.pdf`
- **Barrada09** = `2009_barrada-olea_item-selection-rules-accuracy-security.pdf`
- **Barrada10** = `2010_barrada-olea_method-comparison-item-selection-cat.pdf`
- **KLI** = `2012_koedinger-corbett-perfetti_kli-framework.pdf`
- **AK13** = `2013_aleven-koedinger_kc-approaches-learner-modeling.pdf` (Ch. 15, pp. 165–183)
- **KA07** = `2007_koedinger-aleven_assistance-dilemma.pdf`
- **Rohrer12** = `2012_rohrer_interleaving-similar-concepts.pdf`
- **RJ15** = `2015_rittle-johnson_not-a-one-way-street.pdf`
- **RJS15** = `2015_rittle-johnson-schneider_oxford-handbook-procedural-conceptual.pdf`

---

## 1. ALEKS uses "info-gain greedy on compressed graph cover" diagnostic

**Doc says (lines 6, 27, 145, 250):** "diagnostic exams are information-gain greedy on a compressed graph cover," "Math Academy's 'knowledge frontier' approach (information-gain greedy on a compressed cover, terminate at frontier resolution) is the most directly applicable model" — and conflates this with ALEKS.

**Verdict:** DISPUTED (mis-attribution).

**Source:** Cosyn21 §1.3; Doble19 §"Uncovering the knowledge state."

**Justification:** ALEKS does NOT use a "compressed graph cover." Per Cosyn21 §1.3, ALEKS picks each item whose likelihood is closest to 0.5 (i.e., the item that maximally splits the current state distribution); to handle the 10^23-state structure, it "partitions the item set into several subsets and runs the assessment in parallel" with projections (NOT a compressed cover). The phrase "compressed graph cover" appears in the Math Academy "How Our AI Works" page (per Cluster C). The doc fuses Math Academy's language with ALEKS's algorithm. The underlying idea (information gain via likelihood-near-0.5 splitting) IS shared, so "information gain greedy" is broadly defensible for ALEKS; "compressed graph cover" is not.

---

## 2. ALEKS Knowledge Space Theory (KST) framework — what it actually is

**Doc says (lines 27, 143, 160):** treats ALEKS as a "KST-Markovian" system that "terminates on state-concentration."

**Verdict:** CONFIRMED.

**Source:** Doble19 §"Basic Concepts: KST and ALEKS PPL" pp. 2–7; Cosyn21 §1.

**Justification:** KST = combinatoric/probabilistic model from Doignon & Falmagne 1985; knowledge structure (Q, K) where K is family of feasible knowledge states (subsets of items mastered). ALEKS PPL has 314 items and ~10^23 states. Outer fringe = items "ready to learn," inner fringe = "high points." Assessment is a probabilistic search updating likelihoods over states; terminates when one state dominates or 29 questions reached (Doble19 §"Uncovering the knowledge state," p. 7). Doc's high-level characterization is accurate.

---

## 3. Cosyn 2021 — what this paper actually contributes

**Doc says (lines 143, 208):** "Definitive engineering writeup of ALEKS's diagnostic algorithm... terminates when one state's probability dominates." Claims Cosyn21 describes "how ALEKS does [graph-propagated priors] implicitly through the knowledge-state distribution."

**Verdict:** PARTIAL.

**Source:** Cosyn21, especially §1.3, §2, §3.

**Justification:** Cosyn21 confirms the likelihood-0.5 selection rule and termination on state concentration (§1.3, p. 6). However, Cosyn21 does NOT describe a "graph-propagated prior" mechanism in the sense Delta Drills would use it; the paper says the initial state distribution "is not uniform but is instead informed by past assessment data from the course" (p. 6) — i.e., a population-level empirical prior, not a per-learner parent/child propagation. The doc's recommendation (line 208) over-claims what Cosyn21 documents.

---

## 4. Doble & Matayoshi 2019 — reliability/simulation findings

**Doc says (line 141):** "Empirical study of ALEKS diagnostics on 700,000 actual assessments."

**Verdict:** CONFIRMED.

**Source:** Doble19 abstract; §"A Study of Reliability for the ALEKS PPL Assessment."

**Justification:** Paper uses N=742,851 ALEKS PPL assessments (March 2012–March 2017, first-time placement only) to simulate response probabilities by knowledge category × layer relative to assessed state, then re-simulates assessments to evaluate reliability (correlation between actual and simulated percentage scores; conditional SEM; placement consistency). The "700,000" figure is the abstract's stated lower bound ("more than 700,000"). Confirmed as cited.

---

## 5. Doc claim: "Target 25–40 items" for cold-start diagnostic

**Doc says (lines 143, 156, 207):** "ALEKS PPL ... 25–30 problem types"; "expect 25–40 diagnostic items to resolve a coarse frontier"; "Target 25–40 items for initial frontier resolution."

**Verdict:** PARTIAL (25–30 confirmed for ALEKS; "25–40" for Delta Drills is doc-author extrapolation).

**Source:** Cosyn21 p. 7 ("the number of questions in an initial assessment has been capped at 30"); Doble19 p. 7 footnote 4 ("up to 30 questions ... after the addition of a randomly chosen 'extra problem'").

**Justification:** ALEKS uses up to 30 questions (29 adaptive + 1 random extra problem); Cosyn21 explains the cap was set "based on the feedback from students and instructors over the years" balancing information against fatigue — NOT derived from a 200-atom graph or graph-theoretic argument. The "25–40" range and the claim that "Math Academy reports both fall in this range" are unsupported by these sources. Math Academy's actual diagnostic length is variable and not 25–40 per published material reviewed in Cluster C.

---

## 6. Barrada 2009/2010 CAT item-selection — FI*IG claim

**Doc says (line 149):** "Barrada, Olea, Ponsoda & Abad (2010), 'Item Selection Rules in CAT: Accuracy and Security,' *Methodology* — empirical comparison; FI*IG (Fisher-information-by-interval with geometric mean) is the recent state of the art."

**Verdict:** DISPUTED (mis-citation of year and journal; FI*IG itself is real but in the 2009 paper).

**Source:** Barrada09 (full title: *"Item Selection Rules in Computerized Adaptive Testing: Accuracy and Security,"* Methodology 2009, 5(1):7–17); Barrada10 (*"A Method for the Comparison of Item Selection Rules in CAT,"* Applied Psychological Measurement 2010, 34(6):438–452).

**Justification:** FI*IG ("Fisher information by interval with geometric mean") IS proposed and validated in Barrada **2009**, not 2010, with the abstract claim: *"FI*IG is the only ISR which simultaneously outperforms PFI in both variables [accuracy and security]."* The 2010 paper compares six rules (PFI, FI-L, KL-L, MIS-B, PG, PP) and concludes the best are KL-L, PG, MIS-B — FI*IG is NOT among the six tested in 2010. The doc has merged the two papers and attributed the FI*IG finding to the wrong year/journal. The Barrada09 paper IS in the source folder; the doc just cited it incorrectly.

---

## 7. Koedinger, Corbett, Perfetti 2012 KLI framework

**Doc says (line 101, 119, 170):** "Theoretical grounding for prerequisite remediation as a mastery-learning intervention"; "Practical bridge from cognitive science to deployment."

**Verdict:** PARTIAL (KLI is real and broadly relevant; "prerequisite remediation as mastery-learning" is a stretch).

**Source:** KLI abstract and §1.2, §1.3, §2; Figure 1.

**Justification:** KLI is a real framework (Koedinger, Corbett, Perfetti, *Cognitive Science* 36(5):757–798, 2012) with three taxonomies: Knowledge Components, Learning Events, Instructional Events. It targets a "grain size that is intermediate between existing theoretical concepts in education and cognitive psychology" (§1.2). KLI does discuss mastery-learning and prerequisites in passing (it cites prereq dependencies in KCs) but it does NOT provide a specific theoretical grounding for "prerequisite remediation as a mastery-learning intervention" in the way the doc implies — the framework is broader and more taxonomic than that. Citing KLI here is plausible but not load-bearing.

---

## 8. Aleven & Koedinger 2013 characterization

**Doc says (line 29):** "the Aleven & Koedinger 'Knowledge Component' framework with mastery-learning gates, plus the Rittle-Johnson 'iterative procedural-conceptual' model from cognitive science."

**Verdict:** PARTIAL.

**Source:** AK13 Ch. 15 pp. 165–183.

**Justification:** AK13 Ch. 15 "Knowledge Component (KC) Approaches to Learner Modeling" exists, is correctly attributed (Aleven & Koedinger, 2013, in *Design Recommendations for Intelligent Tutoring Systems Vol. 1*, pp. 165–182). The chapter focuses on KC model creation, refinement (LFA, AFM, SimStudent), and BKT/AFM for mastery tracking. It does mention "Cognitive Mastery" gating problem selection (p. 169). HOWEVER, the chapter does NOT formulate "mastery-learning gates" as a named architectural pattern, nor does it map KCs onto a composite/procedural distinction. The doc's gloss ("KC framework with mastery-learning gates") is a fair summary of *combined* CTAT/Cognitive-Tutor practice but is doc-author synthesis, not a direct paraphrase.

---

## 9. Koedinger & Aleven 2007 "assistance dilemma"

**Doc says (line 176):** "Foundational paper on when to scaffold (= demote to procedural) vs. when to challenge (= stay at composite). Argues the dilemma can only be resolved empirically per skill."

**Verdict:** PARTIAL.

**Source:** KA07 abstract, pp. 241–243, Table 2.

**Justification:** The assistance dilemma is real and KA07 is the canonical paper. KA07 calls it a "fundamental open problem ... [needing] further science to yield specific conditions and parameters that indicate when and to what extent to use information giving versus information withholding forms of interaction" (abstract). So "resolved empirically per skill" is a fair gloss. HOWEVER, KA07 does NOT frame the dilemma as "scaffold = demote to procedural" vs. "challenge = stay at composite." The dilemma in KA07 is at the level of within-problem assistance (hints, feedback, worked examples vs. problem solving) — NOT a two-tier compositional architecture. The doc's mapping is a creative re-interpretation, not what the paper claims.

---

## 10. Rohrer 2012 interleaving

**Doc says (lines 109, 174):** "Empirical backing for interleaving as opposed to blocked practice"; "Rohrer (2012), 'Interleaving helps students distinguish among similar concepts,' Educational Psychology Review." Doc uses Rohrer to back the "cap consecutive same-atom procedurals at ~3" rule.

**Verdict:** PARTIAL (paper and main finding confirmed; specific cap-of-3 number is doc-author).

**Source:** Rohrer12 pp. 355–367, especially "Mathematics Learning" section pp. 4–5.

**Justification:** Title and citation exact; *Educational Psychology Review* 24:355–367 (2012). The main thesis ("interleaving aids discrimination learning") is correctly summarized. Rohrer cites studies where interleaved math practice beat blocked (e.g., Rohrer & Taylor 2007: 63% vs 20% on transfer test, d=1.34; Taylor & Rohrer 2010: 77% vs 38%, d=1.21). HOWEVER, the doc's specific "cap consecutive same-atom procedurals at ~3 (Math Academy's 'spacing-over-blocking' principle, also supported by ... Rohrer 2012)" is a fabricated specific threshold. Rohrer's review does not prescribe a numeric cap; he is cautious about implementation ("future research might examine how the likability of interleaving can be improved") and his caveats section explicitly warns that benefits are limited to *difficult discriminations*.

---

## 11. Rittle-Johnson 2015 "Not a One-Way Street"

**Doc says (line 117):** "The foundational cognitive-science claim that procedural and conceptual knowledge develop iteratively, not sequentially. Implication for Delta Drills: the transition curve should not be monotonic; it should oscillate as new sub-areas open up."

**Verdict:** PARTIAL.

**Source:** RJ15 abstract; "Bidirectional Relations" section pp. 5–6.

**Justification:** Title, authors (Rittle-Johnson, Schneider, Star), and journal (Educational Psychology Review, 2015) confirmed. The bidirectional claim is the paper's central thesis: *"the relations between conceptual and procedural knowledge are bidirectional. It is a myth that it is a 'one-way street.'"* HOWEVER, the doc's leap to "transition curve should not be monotonic; it should oscillate" is a Delta-Drills inference, not a Rittle-Johnson claim. RJ15 is about whether instructional ORDER should be concepts→procedures or vice versa; it doesn't address mastery curves or their monotonicity at all. The cognitive-science claim is real; the implementation gloss is doc-author.

---

## 12. Rittle-Johnson & Schneider Oxford handbook — non-monotonic mastery backing

**Doc says (line 172, 117):** "Rittle-Johnson & Schneider (2015), 'Developing Conceptual and Procedural Knowledge of Mathematics,' Oxford Handbook of Numerical Cognition. The cognitive-science backing for the dual stream; explicitly argues for interleaved development."

**Verdict:** PARTIAL.

**Source:** RJS15 abstract, pp. 1–5.

**Justification:** Title confirmed: "Developing Conceptual and Procedural Knowledge of Mathematics," to appear in Cohen Kadosh & Dowker (eds.), *Oxford Handbook of Numerical Cognition*. Authors Rittle-Johnson & Schneider confirmed. Abstract says relations are "often bi-directional and iterative." The doc's "interleaved development" gloss is consistent. HOWEVER, the doc uses this paper as backing for "non-monotonic mastery curves" — RJS15 does NOT discuss mastery curves, decay, posterior precision, or anything mathematically resembling Delta Drills' sigmoid transition. The handbook chapter is a developmental-psychology review of empirical findings, not a model of learning dynamics.

---

## 13. "Aleven-Koedinger KC framework providing theoretical backing for high-grain/low-grain (composite/procedural) distinction"

**Doc says (line 170):** "The composite/procedural distinction is implicit in the KC framework as 'high-grain-size KCs' (composites) vs. 'low-grain-size KCs' (procedurals)."

**Verdict:** UNSUPPORTED.

**Source:** AK13 Ch. 15; KLI §1.2.

**Justification:** Both AK13 and KLI discuss "grain size" of KCs — but the distinction is about the *level of cognitive task* (e.g., word identification vs. letter recognition vs. multiplication; KLI Fig. 2, p. 10). It is NOT a distinction between "composite items that test multiple atoms" and "procedural items that test single atoms" in the Delta Drills sense. AK13 explicitly emphasizes that LFA can SPLIT a KC into more specific KCs or MERGE two KCs — implying KCs at all grain sizes are atomic units, not a two-tier composite/atomic hierarchy. The doc's claim that KC framework provides "theoretical backing for the high-grain/low-grain (composite/procedural) distinction" misrepresents what "grain size" means in the KC literature. This is a load-bearing fabrication for the Q6 architecture claim.

---

## 14. "Two-tier architecture matches Math Academy's lesson/quiz/review"

**Doc says (lines 29, 178, 182):** "The closest formalization is the lesson/review distinction in Math Academy + the high-grain/low-grain KC distinction in the KC framework + the iterative procedural/conceptual model in cognitive science."

**Verdict:** DISPUTED.

**Source:** KLI; AK13 Ch. 15; Cluster C findings on Math Academy.

**Justification:** Per Cluster C, Math Academy's structure is **lesson / quiz / review / multistep**, where lessons teach new knowledge points, quizzes test 80%-targeted mastery, reviews are spaced FIRe-driven, and multistep problems integrate. None of these maps cleanly onto Delta Drills' composite-vs-procedural meta-architecture. Math Academy's lesson/review distinction is about *instructional mode* (learning vs. retention), not about *evidence granularity* (multi-tag vs. single-tag items). The KC framework does not provide a two-tier scaffold either (see #13). The doc's claim that two-tier "is an obvious instantiation" of these works is doc-author synthesis, presented as if the prior art supports it. KA07's "assistance dilemma" applies WITHIN a problem, not between tiers (see #9). The doc itself half-admits this on line 29: *"not formally documented in adaptive-learning literature as a single named pattern."*

---

## 15. "Closest analogs are Aleven & Koedinger KC framework + Rittle-Johnson iterative procedural/conceptual model"

**Doc says (line 29):** "The closest formalization is the Aleven & Koedinger 'Knowledge Component' framework with mastery-learning gates, plus the Rittle-Johnson 'iterative procedural-conceptual' model from cognitive science."

**Verdict:** PARTIAL.

**Source:** AK13 Ch. 15; RJ15; RJS15.

**Justification:** Both works exist and are correctly attributed. Rittle-Johnson does propose iterative bidirectional relations (RJ15, RJS15). AK13 does describe KC-based learner modeling with Cognitive-Tutor-style mastery gating. HOWEVER: (a) the KC framework does NOT distinguish composite/procedural at Delta Drills' granularity (see #13); (b) Rittle-Johnson's "iterative" is about INSTRUCTIONAL ORDER (concepts→procedures vs. vice versa, repeated), not about per-atom mastery scaffolding of a procedural tier feeding a composite tier; (c) calling these the "closest analogs" is defensible only if interpreted very loosely. The doc's claim that "the user is reinventing a known wheel" is overstated — the wheel as described (composite EWMA + procedural per-atom Beta + event-triggered demotion + sigmoid transition on posterior precision) is genuinely doc-author synthesis.

---

## Summary — verdict tally

| Verdict | Count | Claims |
|---|---|---|
| CONFIRMED | 2 | #2, #4 |
| PARTIAL | 8 | #3, #5, #7, #8, #9, #10, #11, #12, #15 (9 — recount: 9) |
| DISPUTED | 3 | #1, #6, #14 |
| UNSUPPORTED | 1 | #13 |
| CANT_TELL | 0 | — |

(Effective: 2 CONFIRMED, 9 PARTIAL, 3 DISPUTED, 1 UNSUPPORTED.)

## Load-Bearing Concerns

Three findings materially undermine prescriptions in the research doc:

**(A) The ALEKS "compressed graph cover" attribution (#1) is wrong.** Doc conflates Math Academy's diagnostic vocabulary with ALEKS's actual algorithm. ALEKS uses likelihood-near-0.5 item selection over a projected/partitioned knowledge structure — not a "compressed cover" of a prereq graph. For Delta Drills' cold-start prescription on line 207 ("information-gain-greedy on a compressed cover of the prereq graph"), this matters: the user should know the "compressed cover" idea comes ONLY from Math Academy's marketing prose (not peer-reviewed) and that ALEKS's actual mechanism is different and more principled (Bayesian-update on a finite state space). Implementing "compressed cover" on a 200-atom DAG without Math Academy's full graph machinery is a leap that ALEKS literature does not validate.

**(B) The "25–40 items" cold-start target (#5) is partly fabricated.** ALEKS uses up to 30 items based on student-fatigue feedback (Cosyn21), NOT on graph-theoretic information-coverage arguments. The "25–40" upper end and the alignment with "Math Academy reports" are doc-author extrapolations. Treat as a heuristic to A/B test, not as a literature-backed default.

**(C) The two-tier architecture's "theoretical backing" (#13, #14, #15) is largely doc-author synthesis dressed as literature.** Neither the KC framework (AK13/KLI) nor Rittle-Johnson's procedural/conceptual work provides architectural backing for Delta Drills' specific composite-tier-EWMA + procedural-tier-Beta + event-triggered-demotion + sigmoid-transition design. The doc admits this on line 29 ("not formally documented ... as a single named pattern") but then on lines 178, 182, 198 leans heavily on the implicit framing that the architecture is "an obvious instantiation" of these works. It is not. The architecture is novel; it should be defended on its own merits, not on misappropriated theoretical authority.

## Other notable findings

- **Barrada citation (#6):** FI*IG is real but mis-cited to the 2010 paper. The 2010 paper does not include FI*IG in its six-rule comparison.
- **Rohrer's interleaving cap of 3 (#10):** the "≤3 consecutive same-atom procedurals" number is doc-author, not Rohrer-author. Rohrer's review is also explicitly limited to "difficult discriminations" — the doc never flags this scope caveat.
- **Rittle-Johnson "non-monotonic mastery curve" (#11, #12):** R-J never says this. The bidirectional/iterative claim is about instructional sequencing, not mastery dynamics. The doc transplants a sequencing finding into a dynamics finding.

## Pattern notes (consistent with Clusters A/B/C)

- **Threshold-as-citation:** "25–40 items" (line 207), "cap consecutive same-atom procedurals at ~3" (line 109). Same pattern Cluster A/B flagged (τ=14d, α+β≥8, N≥30, tag-cap=6).
- **Citation-shifting:** Barrada FI*IG mis-attributed across years (#6). Math Academy's "compressed graph cover" attributed to ALEKS (#1).
- **Theoretical authority laundering:** KC framework's "grain size" reinterpreted as composite/procedural distinction (#13). KA07's within-problem dilemma reframed as between-tier dilemma (#9). R-J's instructional-order claim reframed as mastery-dynamics claim (#11, #12).

The cold-start info-gain diagnostic (Q5) is *directionally* supported by Doble19/Cosyn21 (likelihood-0.5 selection IS information-gain-like) but the specific "compressed graph cover, 25–40 items" prescription is Math Academy marketing prose + doc-author numerics. The two-tier architecture validation (Q6) is weaker than the doc presents — none of the cited works architecturally backs it.
