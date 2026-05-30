# Mastery Estimation for Delta Drills: A Technical Reference (v2, source-audited)

**Status:** Rewrite of the 2026-05-23 deep-research output after a 2026-05-24 four-cluster source audit against the primary sources downloaded to `papers/mastery-estimation/`. The audit (see `VERIFICATION_SUMMARY.md` and `VERIFICATION_{A,B,C,D}_*.md`) found ~1/3 of the original doc's claims wrong, fabricated, or misattributed. This version preserves only what survived audit and explicitly flags engineering choices as such.

**Conventions used here:**
- **[CONFIRMED]** — claim was checked against a primary source in `papers/mastery-estimation/` and matches.
- **[ENGINEERING]** — defensible doc-author construct, not from any audited source. Re-derive empirically from your own data.
- **[NOVEL]** — Delta Drills' framing is not in the published literature; defend on its own merits.

---

## TL;DR

- For a two-tier prereq-graph adaptive system with ~200 atoms and small initial user base, a defensible v0 algorithm stack is **Beta-Bernoulli per-atom posteriors with explicit time decay [ENGINEERING]**, **PFA / Logistic Knowledge Tracing for multi-tag credit assignment on composite items [CONFIRMED foundational form, Pavlik 2009]**, and a **Math-Academy-inspired event-driven demotion controller** [CONFIRMED at the framing level — see Q3 for the granularity caveat]. Full BKT and DKT carry identifiability issues at low sample sizes [CONFIRMED — Khajah 2016 shows enhanced BKT $\approx$ DKT; classical BKT identifiability is a known concern per Pelánek 2017].
- Math Academy's published algorithm does not use Bayesian Knowledge Tracing for its mastery model. Justin Skycak has stated he tried BKT for the diagnostic algorithm and moved to a "physical" fractional-repetition flow (FIRe) [CONFIRMED, verbatim quote audit passed]. The FIRe update is

$$\text{repNum} \to \max\!\bigl(0,\; \text{repNum} + \text{speed} \cdot \text{decay}^{\,\text{failed}} \cdot \text{rawDelta}\bigr)$$

with companion

$$\text{memory} \to \max\!\bigl(0,\; \text{memory} + \text{rawDelta}\bigr) \cdot (0.5)^{\text{days}/\text{interval}}$$

[CONFIRMED verbatim in Skycak 2023]. Demotion is event-triggered: a lesson halts on intra-lesson knowledge-point failure, and a two-failed-lesson "plateau" without forward progress triggers prerequisite remediation [CONFIRMED — note these are two distinct mechanisms, not one rule]. Quiz misses trigger immediate single-topic remediation [CONFIRMED]. Diagnostic uses information-gain item selection [CONFIRMED, HOAW lifts verbatim]; "conditionally completed" is real MA terminology for borderline placements [CONFIRMED].

- Likely failure modes for Delta Drills: (i) over-aggressive demotion creating a "punishment loop" — Oz Nova's review of Math Academy states "subjecting someone to literal years of unnecessary remedial 'pre-requisites' is too great a punishment for a second slip up" [CONFIRMED verbatim]; (ii) credit-assignment instability when composites carry many tags — Maier 2021 documents degenerate-parameter cases in PFA at low practice counts [CONFIRMED]; (iii) double-counting EWMA on top of a Beta posterior when the decaying $\alpha/\beta$ already encodes an attempt-indexed EWMA [CONFIRMED mathematically].

---

## Key findings

1. **Beta-Bernoulli with explicit time decay is a defensible primitive [ENGINEERING].** The doc-author construct is: per-attempt update $\alpha \leftarrow \alpha + y,\; \beta \leftarrow \beta + (1-y)$; between attempts, shrink toward prior with half-life $\tau$. This is NOT what Settles & Meeder 2016 publishes — HLR is a discriminative regression on engineered features ($p = 2^{-\Delta/h}$, $\hat{h} = 2^{\Theta \cdot x}$, trained by L2 SGD on Duolingo's 13M-row dataset, eq. 1–2 + Appendix A.3 in Settles-Meeder). The Beta-with-decay primitive is a sensible Bayesian construction in its own right, but it is not literature-validated; ship it as a v0 experiment to instrument and tune.

2. **EWMA on top of a Beta posterior is redundant for the static-skill case [CONFIRMED mathematically].** If at each attempt $\alpha \leftarrow \alpha \cdot d + y$ and $\beta \leftarrow \beta \cdot d + (1-y)$ with $d \in (0,1)$, the posterior mean $\alpha/(\alpha+\beta)$ is exactly the exponentially weighted average of outcomes with smoothing $1-d$. Adding a separate EWMA only makes sense if you need a two-timescale signal (fast "current form" alongside slow "long-run"), which is what Recent-PFA (Galyardt & Goldin 2014, EDM, eq. 2 + Table 1) does explicitly by adding a recency-weighted-success-rate feature $R_{ijt}$ alongside cumulative totals $T_{ijt}$ [CONFIRMED].

3. **For multi-tag credit on composites, PFA / LKT is the most defensible fit at this scale [CONFIRMED at the formula level].** Pavlik, Cen & Koedinger 2009 eq. 3:

$$m(i,\; j \in \text{KCs},\; s,\; f) = \sum_{j \in \text{KCs}} \bigl(\beta_j + \gamma_j \cdot s_{i,j} + \rho_j \cdot f_{i,j}\bigr)$$

then $p(m) = 1/(1+e^{-m})$ (eq. 2). DINA / Q-matrix CDM has known identifiability conditions (Köhn & Chiu series) that a heterogeneous 5-edge-kind graph will not satisfy. Full MIRT is parameter-heavy at ~200 atoms with sparse data.

4. **"How many tags is too many" has no clean published threshold [ENGINEERING].** The original doc claimed "implicit PFA consensus $\geq 6$ tags becomes uninformative" — this is not in Pavlik 2009, Galyardt 2014, or Maier 2021. Maier 2021's experimental scope caps at multi-skill vs single-skill comparisons, never breaking out by tag-count buckets. The original doc also internally contradicted itself, admitting "no published clean threshold" two paragraphs later. Treat any tag-cap (6 or otherwise) as your own engineering choice, not literature consensus.

5. **Math Academy's demotion is event-triggered [CONFIRMED at framing level; two distinct mechanisms].** The published rules are (a) **knowledge-point halt** — intra-lesson, fires when a student stumbles on a single knowledge point within a lesson (C&T Ep. 42: "each topic has a lesson, broken down into stages, which we call knowledge points… We will halt lessons if a student stumbles on any particular knowledge point a little bit too much"); and (b) **two-failed-lesson plateau** — cross-attempt, fires after a student "fails the lesson twice in a row without getting any further the second time" (Skycak 2023 FIRe Q&A, exact wording). Quiz misses trigger immediate single-topic remediation (HOAW: "whenever a student misses a question on a quiz, we slow down and immediately follow up with a remedial review on the corresponding topic"). Conditional completion is a separate placement-side mechanism for borderline diagnostic outcomes (HOAW, verbatim).

6. **For demoted-tier prereq selection, greedy-on-lowest-mastery with diversification is the lighter choice at small scale [ENGINEERING].** Thompson Sampling for educational recommendation has been explored — De Kerpel, Thuy & Benoit 2026 (INFORMS Transactions on Education 26(3):187–199) is a recent systematic study — but reported gains depend on stable per-item reward distributions that Delta Drills will not have at ~200 atoms with sparse per-learner samples. Greedy on lowest posterior mean with diversification (don't pick two atoms with overlapping parents) is sufficient.

7. **Promotion (procedural→composite) should be evidence-based, not session-based [ENGINEERING].** A reasonable construct: trigger on posterior precision $(\alpha+\beta) \geq N_{\min}$ and posterior mean $\geq \theta$ for the relevant atoms, plus a single successful "test" composite. The specific numeric values are yours to choose. Math Academy uses 80%-targeted quizzes as the structural analog, but does not publish the underlying threshold [CONFIRMED — "the specific implementation is proprietary" per Skycak 2023].

8. **The transition curve from procedural-dominant to composite-dominant should depend on evidence, not time [ENGINEERING].** A sigmoid on mean posterior precision over the area's atoms is a reasonable form. This allows non-monotonic regression: when a learner takes a gap, decay reduces effective sample count, posterior precision falls, and the mix shifts back toward procedurals automatically. (The general principle that mastery curves should allow regression follows from any model with forgetting decay; the original doc cited Yudelson-Pavlik 2013 as the "monotonicity anti-pattern" source, but that paper was not in the audited corpus.)

9. **For cold-start, information-gain-style item selection on a prereq graph is directionally supported by deployed systems [CONFIRMED at the directional level].** ALEKS picks items whose likelihood is closest to $0.5$ — i.e., the item that maximally splits the current state distribution — over a projected/partitioned knowledge structure (Cosyn 2021 §1.3). ALEKS uses up to 30 items based on student-fatigue feedback over the years, not a graph-theoretic argument (Cosyn 2021 p. 7). Math Academy's "compressed graph cover" diagnostic terminology comes from their How Our AI Works page; the operationalization (how the cover is computed, what the scoring function is) is not public.

10. **The two-tier composite/procedural meta-architecture is NOT validated by the published literature [NOVEL].** Despite some superficial parallels, the audited sources do not back the original doc's framing. Specifically:
    - The KC framework's "grain size" (AK13, KLI 2012) refers to cognitive task level (word identification vs letter recognition vs multiplication), NOT to items-that-test-multiple-atoms vs items-that-test-single-atoms.
    - Rittle-Johnson's "iterative procedural-conceptual" claim (RJ15, RJS15) is about INSTRUCTIONAL ORDER (concepts→procedures vs vice versa, repeated), not about per-atom mastery dynamics or two-tier evidence granularity.
    - Koedinger-Aleven 2007's "assistance dilemma" is WITHIN-problem (hints, feedback, worked examples), not BETWEEN-tier.
    - Math Academy's lesson/quiz/review distinction is about instructional mode, not multi-tag-vs-single-tag evidence granularity.

    Delta Drills' two-tier design is genuinely novel. Defend it on its own merits, not on misappropriated theoretical authority.

---

## Details

### Q1. Bayesian mastery representation with evolving uncertainty

**Audited primary references:**

- **Settles & Meeder (2016), "A Trainable Spaced Repetition Model for Language Learning," ACL.** Half-life regression (HLR) on Duolingo's 12.9M-instance dataset [CONFIRMED, abstract + §4.1]. HLR predicts recall via $p = 2^{-\Delta/h}$ (eq. 1) where $\hat{h} = 2^{\Theta \cdot x}$ (eq. 2) — $h$ is a learned half-life regressed on per-item features and right/wrong counts. Trained by L2-regularized squared-loss SGD (Appendix A.3). The paper reports a 45%+ MAE reduction over baselines and a 12% engagement lift in operational A/B test [CONFIRMED, abstract]. Code + dataset open-sourced at github.com/duolingo/halflife-regression. **Note:** HLR is a discriminative regression on engineered features, not a Beta-Bernoulli model. It does not maintain a posterior; there is no $\alpha/\beta$, no conjugate prior. The original v1 doc claimed HLR was the "formal trainable version of Beta-with-decay" — that claim was UNSUPPORTED in audit. HLR and a Beta-Bernoulli-with-decay primitive are related only in that both encode an exponential forgetting curve.

- **Pelánek (2017), "Bayesian knowledge tracing, logistic models, and beyond: an overview of learner modeling techniques," User Modeling and User-Adapted Interaction 27(3–5):313–350.** Comparative survey. Key finding: BKT and logistic models (PFA, AFM, Elo for education) achieve very similar predictive accuracy on real data. Pelánek's **Hypothesis 1 (p. 33)** states: *"Logistic models are better for modeling fluency and memory processes, while Bayesian knowledge tracing is better for understanding and sense-making processes."* The v1 doc framed Pelánek as endorsing "Beta-Bernoulli + logistic at low sample sizes" — that framing misrepresents the source. Pelánek treats BKT and logistic as parallel families and recommends choice based on task type, not sample size.

- **Pelánek (2016), "Applications of the Elo rating system in adaptive educational systems," Computers & Education.** Argues Elo is "simple, robust, and effective and thus suitable for use in the development of adaptive educational systems" [CONFIRMED, abstract]. The original doc added a "Glicko-2 $RD \approx \sqrt{(\alpha+\beta)^{-1}}$" Beta-equivalence claim — that mapping is doc-author analogy, not in Pelánek 2016. Treat the Elo recommendation as supported; treat the Beta-equivalence as your own framing.

- **Khajah, Lindsey & Mozer (2016), "How deep is knowledge tracing?"** Shows that enhanced BKT (with forgetting + per-student abilities + per-item difficulties) matches DKT across four standard benchmarks (Assistments, Synthetic, Statics, Spanish), within $< 0.01$ AUC on three of four datasets [CONFIRMED, §3.3 Results, Fig. 2]. Practical implication: DKT's marginal gain at this regime is small and DKT loses interpretability.

- **Bijl (2025), arXiv: "Tracking Student Skills Real-Time Through a Continuous-Variable Dynamic Bayesian Network."** Method is called **Performance Distribution Tracing (PDT)**. (The v1 doc called the paper "Probabilistic Decay Trees" — that title was fabricated.) PDT uses Beta-distribution basis functions as conjugate priors, maintains a continuous-variable Dynamic Bayesian Network over success-rate distributions, includes explicit exponential decay over a graph of skills [CONFIRMED, pp. 1–6]. This is the closest published model to a Beta-with-decay primitive on a prereq graph.

- **Nedungadi & Remya (2015), "PC-BKT with forgetting."** Adds an exponential forget function to BKT. Major accuracy gain in their experiments: PC-BKT 80.3% → PC-BKT+decay 95.9% (Table 3) [CONFIRMED]. **Note:** PC-BKT's update operates on the BKT probability $P(L_t)$, not on a Beta posterior. The v1 doc transposed the decay onto a Beta-Bernoulli skeleton that PC-BKT does not use — cite PC-BKT for the finding "exponential decay matters," not for an $\alpha/\beta$ update rule.

- **Qiu, Pardos & Heffernan (2011), "Does Time Matter to BKT?" EDM.** Introduces KT-Forget with a `forget_n` parameter (forgetting probability on a "new day" vs same-day) [CONFIRMED]. **Note:** KT-Forget uses a binary same-day/new-day discretization, not a continuous $P(F \mid \Delta t)$ as the v1 doc claimed. Cite for the existence-of-forgetting finding.

**v2 synthesis — Q1 formulation [ENGINEERING].**

For multi-tag binary outcomes, a defensible Beta-Bernoulli primitive is:

- Per atom $A$ on a composite with outcome $y \in \{0,1\}$ and tag weight $w_A \in (0,1]$:

$$\alpha_A \leftarrow \alpha_A + w_A \cdot y, \qquad \beta_A \leftarrow \beta_A + w_A \cdot (1-y).$$

- Between attempts with elapsed $\Delta t$ and half-life $\tau_A$:

$$\alpha_A \leftarrow \alpha_0 + (\alpha_A - \alpha_0) \cdot 2^{-\Delta t / \tau_A}, \qquad \beta_A \text{ analogously}.$$

Shrinks the posterior toward the prior at the chosen half-life.

**This is your engineering construct, not a published formulation from Settles-Meeder or any other audited paper.** The math is sound. Ship as v0, instrument, tune empirically.

**Forgetting decay choices.** Three forms exist in the literature, each in a different statistical family:

1. Half-life regression on response probability (HLR-style, Settles & Meeder 2016) — discriminative regression on features.
2. Exponential decay on probability $P(L_t)$ in BKT (PC-BKT, Nedungadi 2015) — HMM with binary state.
3. Conditional same-day/new-day forget rate (Qiu 2011) — binary time discretization in BKT.

The Beta-with-decay primitive above is a fourth, doc-author construct. None of the above are direct translations of each other.

**EWMA-over-Beta — redundant or value-adding? [CONFIRMED mathematically].** Redundant in the static-skill case: the decaying $\alpha, \beta$ already encode an exponentially weighted average of attempts (proof: under per-attempt $\alpha \leftarrow \alpha \cdot d + y$, the posterior mean is exactly EWMA of $y$ with $1-d$ smoothing). Value-adding only with two-timescale signals — Recent-PFA (Galyardt & Goldin 2014) carries $T_{ijt}$ (cumulative count) and $R_{ijt}$ (recency-weighted) features simultaneously; their Results: *"For any fixed decay parameter $b$, R-PFA is better than R-only. The total number of practice opportunities is still informative above and beyond the recent history."*

**BKT identifiability at low sample sizes.** Pelánek 2017 (§ on BKT) notes "noise in the data, model identifiability issues, and local optima in parameter fitting" as known concerns [CONFIRMED]. The specific "hundreds of attempts per skill" number traces to Beck & Chang 2007 and van de Sande 2013 — not in the audited corpus, so cite at the level of "BKT has documented identifiability concerns" rather than a specific number. For Delta Drills' regime (~200 atoms, modest user base), a well-instrumented Beta-with-decay + logistic-features stack is a reasonable choice; treat as an empirical question to validate.

### Q2. Multi-tag credit assignment

**Audited primary references:**

- **Pavlik, Cen & Koedinger (2009), "Performance Factors Analysis – A New Alternative to Knowledge Tracing," AIED.** Foundational multi-tag credit model [CONFIRMED, title + authors + venue]. Logistic with per-KC $\beta$ (easiness), per-KC $\gamma$ (success rate), per-KC $\rho$ (failure rate). For an item tagged with KCs $\{A, B, C\}$, summed contribution

$$m = \sum_{j \in \text{KCs}} \bigl(\beta_j + \gamma_j \cdot s_j + \rho_j \cdot f_j\bigr), \qquad p(m) = \frac{1}{1 + e^{-m}}$$

(eq. 2, 3). Note: PFA's compensatory summation IS effectively an additive-credit model across tags.

- **Maier, Baker & Stalzer (2021), "Challenges to Applying Performance Factor Analysis to Existing Learning Systems," ICCE.** (v1 doc called it "Improving PFA for rare skills" — that title was wrong; the rare-skill pooling is one of four contributions.) The paper documents four practical issues in PFA deployment: (1) insufficient practices, (2) degenerate parameters, (3) rare vs common skills, (4) compensatory vs conjunctive. Key empirical findings:
  - PFA convergence visible in the 2-to-12 practice range, not at $N = 30$ (§4.1).
  - Three degeneracy types observed: $\gamma < 0$ (9% of skills), $\gamma < \rho$, and $\gamma = \rho = 0$ (11%) (§4.2).
  - **Compensatory vs conjunctive vs even-skill (= equal credit averaging) comparison (§4.3):** Even-skill AUC $0.7849$, compensatory AUC $0.7818$, conjunctive AUC $0.6725$. **Even-skill (equal credit) was the BEST of the three on their data.** The v1 doc listed equal credit as "catastrophic" — that warning is directly contradicted by the only paper in the audited corpus that empirically tested it.
  - Rare-skill pooling solution: a "merged-rare" set of parameters $(\beta_d, \gamma_d, \rho_d)$ used for skills below a chosen practice threshold (§4.3).

- **Galyardt & Goldin (2014), "Recent Performance Factor Analysis," EDM.** Adds a recency-weighted feature alongside cumulative counts:

$$\text{logit}(p_{ijt}) = \theta_i + \beta_j + \gamma_j \cdot T_{ijt} + \delta_j \cdot R_{ijt}$$

where $T_{ijt}$ = total opportunities, $R_{ijt}$ = recency-weighted proportion of successes (eq. 2, Table 1) [CONFIRMED]. Both signals contribute: *"For any fixed decay parameter $b$, R-PFA is better than R-only"* (Results).

- **Zhang, Baker, Srivastava et al. (2025), "Carelessness Detection using Performance Factor Analysis," arXiv 2503.04737.** Carelessness-detection model (BKFC = Beyond-Knowledge Feature Carelessness) that USES PFA as a knowledge-estimation backbone. Experimental dataset (Decimal Point) has "only one skill per item" (§IV-B) — i.e., the experimental setting is explicitly single-skill, not a multi-tag PFA reference [CONFIRMED, p. 4]. Cite for the operationalization-of-carelessness contribution, not as a multi-tag PFA reference.

- **Park, Cornillie, van der Maas & Van Den Noortgate (2019), "A Multidimensional IRT Approach for Dynamically Monitoring Ability Growth in Computerized Practice Environments," Frontiers in Psychology 10:620.** Introduces M-ERS (Multidimensional Extension of the Elo Rating System) [CONFIRMED, abstract + §M-ERS]. (The v1 doc attributed M-ERS to "Doebler/Pelánek/Wauters 2018" — that attribution was wrong; the Park 2019 paper is the actual M-ERS source.) M-ERS extends Rasch-Elo to a multidimensional ability vector, simultaneously updating multiple ability parameters per attempt.

- **DINA / G-DINA family (de la Torre 2009, 2011; Köhn & Chiu 2017+).** Q-matrix CDMs require a "complete" Q-matrix for identifiability under DINA. Delta Drills' 5-edge-kind graph violates the conjunctive single-relation assumption. (These papers were not in the audited corpus, so cite generically — at the level of "DINA-family CDMs have known identifiability requirements that heterogeneous edge graphs typically violate.")

**v2 synthesis — Q2.**

For Delta Drills' 3–15-tag composite items, a reasonable rank ordering of options:

1. **PFA / LKT (multi-KC logistic regression)** — best fit. Handles multi-tag natively per Pavlik 2009 eq. 3. Maier 2021 reports convergence in the 2–12-practice range. **This is the literature-grounded choice for composite credit assignment.**
2. **Tag-confidence-weighted Beta-Bernoulli credit** [ENGINEERING] — simple workable construct: $\alpha \leftarrow \alpha + w_A \cdot \text{confidence} \cdot y$. Use for the procedural tier (1–2 tags) and as a fallback. Not a published pattern.
3. **MIRT compensatory** — theoretically clean, parameter-heavy at ~200 atoms; expect convergence issues until $\geq 10\text{k}$ attempts. Defer.
4. **DINA / Q-matrix CDM** — identifiability requirements unlikely to be satisfied by a 5-edge-kind graph. Don't go here.
5. **Equal credit (even-skill averaging)** — Maier 2021's empirical test found it best of three credit schemes on their data. The v1 doc's "catastrophic and information-destructive" warning is not supported by audited evidence. Defensible critique: equal credit may be problematic when tag-count and difficulty co-vary, but this is an identifiability-collinearity concern (which Maier 2021 also notes for compensatory PFA), not a categorical "avoid."

**On "how many tags is too many" [ENGINEERING].** No clean published threshold in the audited papers. Any cap (6, 8, 4) is your own choice. Maier 2021 does not test this; the closest related question is rare-skill behavior, addressed via parameter pooling, not by capping tags per item.

### Q3. Demotion logic — when to drop from composite to procedural

**Audited primary references:**

- **Math Academy's "How Our AI Works" page (mathacademy.com/how-our-ai-works).** [CONFIRMED, verbatim quote audit passed for the lifted paragraph.] Key passages:
  - *"if a student answers too many questions incorrectly, we halt the lesson… if a student gets halted again on the re-attempt without making any additional forward progress, then we slow down further and give them remedial reviews… Our knowledge graph tracks the key prerequisites that are exercised in each part of each lesson, which allows us to pinpoint the exact topics that are necessary for successful remediation."*
  - *"if the evidence balances out to just barely place a student out of some topics, the system will consider those topics 'conditionally completed.' The student will initially be given tasks under the assumption that they know those topics, but if the student struggles, then the system will immediately begin 'falling backwards' along the appropriate learning paths."*
  - *"whenever a student misses a question on a quiz, we slow down and immediately follow up with a remedial review on the corresponding topic."*

- **Skycak (2023), "Optimized, Individualized Spaced Repetition in Hierarchical Knowledge Structures" (justinmath.com).** FIRe paper. Update formula:

$$\text{repNum} \to \max\!\bigl(0,\; \text{repNum} + \text{speed} \cdot \text{decay}^{\,\text{failed}} \cdot \text{rawDelta}\bigr)$$

[CONFIRMED verbatim]. Memory term:

$$\text{memory} \to \max\!\bigl(0,\; \text{memory} + \text{rawDelta}\bigr) \cdot (0.5)^{\text{days}/\text{interval}}$$

[CONFIRMED]. Plateau rule (Q&A section): *"if a student 'plateaus' on the lesson for B (i.e., they fail the lesson twice in a row without getting any further the second time), then we trigger remedial learning tasks on the prerequisites where the student got stuck"* [CONFIRMED verbatim]. **Note:** "complexity explodes" (re: BKT) is verbatim [CONFIRMED]; "physical" (re: the FIRe approach) is verbatim [CONFIRMED].

- **Chalk & Talk Ep. 42 transcript (annastokke.com/ep-42-transcript), Alex Smith interview.** First-time lesson pass rate 95%, second-attempt pass rate 99% [CONFIRMED]. Each topic has 3–4 knowledge points; halts trigger at a **knowledge-point** level within a lesson [CONFIRMED — this is the intra-lesson halt, distinct from the cross-lesson plateau rule]. *"each topic has a lesson, broken down into stages, which we call knowledge points… We will halt lessons if a student stumbles on any particular knowledge point a little bit too much."*

- **Oz Nova (2025), "A balanced review of Math Academy" (newsletter.ozwrites.com).** *"Allowing a second attempt at a failed diagnostic question is a good idea. But subjecting someone to literal years of unnecessary remedial 'pre-requisites' is too great a punishment for a second slip up."* [CONFIRMED verbatim, including the hyphenation and the quotes around "pre-requisites"]. Use as warning against over-aggressive demotion depth.

- **De Kerpel, Thuy & Benoit (2026), "A Bandit-Based Approach to Educational Recommender Systems: Contextual Thompson Sampling for Learner Skill Gain Optimization," INFORMS Transactions on Education 26(3):187–199.** First systematic eval of Linear Thompson Sampling for educational recommendation. Reported gains depend on stable per-item reward distributions.

- **Koedinger, Corbett & Perfetti (2012), "The Knowledge-Learning-Instruction framework," Cognitive Science 36(5):757–798.** Real framework with three taxonomies (Knowledge Components, Learning Events, Instructional Events). Cite as KLI background; does not specifically formalize prereq remediation as a named pattern.

**v2 synthesis — Q3 [TWO DISTINCT MA MECHANISMS, plus your engineering choices].**

The Math Academy literature describes **two distinct event-triggered demotion mechanisms** (the v1 doc conflated these into one rule):

- **Knowledge-point halt (intra-lesson):** within a single lesson attempt, fires when a student stumbles too much on a single knowledge point. Triggers within-lesson scaffolding.
- **Two-failed-lesson plateau (cross-attempt):** fires after a student fails the entire lesson twice without making further progress on the re-attempt. Triggers remedial work on the lesson's tagged key prerequisites.

A third trigger:

- **Any quiz miss:** triggers immediate single-topic remediation.

For Delta Drills, your event-triggered controller can be **inspired by** these patterns, but Delta Drills' "area" granularity is not the same as MA's lesson granularity. A defensible v0 [ENGINEERING] is:

- Slow signal: composite-tier EWMA below a chosen threshold per area.
- Fast signal: two failed composites in the same area without intervening progress (your "plateau" analog).
- Quiz-style trigger: any failure on a designated checkpoint composite triggers immediate remediation.

**Procedural selection after demotion [ENGINEERING]:** greedy on lowest posterior mean among the failed composite's tagged prereq atoms, with diversification constraints. The original doc cited Rohrer 2012 as supporting a "$\leq 3$ consecutive same-atom" cap — Rohrer's review establishes that interleaving aids discrimination learning [CONFIRMED] but prescribes no numeric cap. Choose any cap as engineering judgment.

**Promotion trigger [ENGINEERING]:** posterior precision and mean thresholds plus a successful test composite. Specific values are yours. Math Academy uses 80%-targeted quizzes as structural analog but does not publish the underlying threshold.

### Q4. Transition curve from procedural-dominant to composite-dominant

**Audited primary references:**

- **Rittle-Johnson, Schneider & Star (2015), "Not a one-way street: Bidirectional relations between procedural and conceptual knowledge," Educational Psychology Review.** Foundational cognitive-science claim: *"the relations between conceptual and procedural knowledge are bidirectional. It is a myth that it is a 'one-way street.'"* [CONFIRMED, abstract]. **Note:** R-J's bidirectional claim is about INSTRUCTIONAL ORDER (whether to teach concepts first then procedures or vice versa), not about per-atom mastery monotonicity. The v1 doc framed R-J as backing "non-monotonic mastery curves" — that framing transplants an instructional-sequencing finding into a mastery-dynamics finding, which R-J does not address.

- **Rittle-Johnson & Schneider (2015), Oxford Handbook of Numerical Cognition chapter.** Developmental-psychology review of empirical findings on procedural/conceptual relations [CONFIRMED title + scope]. Same caveat: it is a developmental review, not a model of mastery dynamics.

- **Khajah, Lindsey & Mozer (2016).** Practical baseline for "mastery curve in posterior precision, not in time" — to the extent this is a finding, it follows from any model with explicit forgetting decay.

- **van der Linden & Glas eds., "Computerized Adaptive Testing: Theory and Practice" (Springer 2010).** Establishes SE-of-measurement-below-$\epsilon$ as the dominant CAT termination rule. Not in the audited corpus; cite as general CAT background.

- **Math Academy's HOAW** documents practical instantiation: lesson question count adapts to accuracy; review credit flows implicitly through the encompassing graph.

**v2 synthesis — Q4 [mostly ENGINEERING].**

- **Functional form:** A logistic (sigmoid)

$$\sigma\!\left(\frac{\text{mean\_precision} - N_{\text{thresh}}}{\text{scale}}\right)$$

over the area's atoms is a clean engineering choice. Hill / exponential-to-asymptote are reparameterizations.
- **Time vs. evidence:** evidence (posterior precision). Time-based schedules fail when learners have variable engagement.
- **Monotonicity / regression:** allow regression. Any model with forgetting decay will produce non-monotonic mastery naturally — when decay outpaces evidence, precision falls. This is the right behavior. The principle is uncontroversial; the v1 doc cited a specific Yudelson-Pavlik 2013 paper for the "monotonicity anti-pattern" which was not in the audited corpus, so present this as a general consequence of forgetting decay rather than a specific cited result.
- **v0 defaults [ENGINEERING]:** $N_{\text{thresh}} \approx 8$ and $\text{scale} \approx 3$ are placeholders to tune from population data, not literature-derived values.

### Q5. Cold-start diagnostic phase

**Audited primary references:**

- **Doble, Matayoshi, Cosyn, Uzun & Karami (2019), "A Data-Based Simulation Study of Reliability for an Adaptive Assessment Based on Knowledge Space Theory," IJAIED 29(2):258–282.** Empirical reliability study on $N = 742{,}851$ ALEKS PPL assessments [CONFIRMED, abstract].

- **Cosyn, Uzun, Doble & Matayoshi (2021), "A practical perspective on knowledge space theory: ALEKS and its data," J. Mathematical Psychology.** Engineering writeup of ALEKS's diagnostic algorithm [CONFIRMED, §1.3]. ALEKS picks items whose likelihood is closest to $0.5$ — i.e., the item that maximally splits the current state distribution — over a partitioned/projected knowledge structure (NOT a "compressed graph cover"). The v1 doc fused Math Academy's "compressed cover" vocabulary with ALEKS's algorithm; these are different mechanisms. ALEKS terminates when one state's probability dominates or at 29 questions (Cosyn 2021 p. 6–7). The cap "has been capped at 30… based on the feedback from students and instructors over the years" — a fatigue-based engineering choice, not a graph-theoretic argument.

  **Population-level prior:** Cosyn 2021 p. 6 notes the initial state distribution "is not uniform but is instead informed by past assessment data from the course" — i.e., an EMPIRICAL POPULATION prior, not a per-learner parent/child graph-propagation. The v1 doc described ALEKS's prior as a graph-propagated per-learner construct; that framing is unsupported.

- **Math Academy's HOAW diagnostic paragraph** [CONFIRMED verbatim]: *"The algorithm first compresses the knowledge graph into the smallest number of topics that fully 'covers' a course and its foundations at a sufficient level of granularity. Then, it repeatedly selects the topic whose assessment provides the most information about the student's knowledge profile."* **Note:** HOAW gives the WHAT, not the HOW. The compression procedure and the information-gain scoring function are not public. "Knowledge frontier" is real MA terminology; "knowledge frontier resolution" (as a stopping criterion noun phrase) is not in the audited sources.

- **Barrada, Olea, Ponsoda & Abad (2009), "Item Selection Rules in Computerized Adaptive Testing: Accuracy and Security," Methodology 5(1):7–17.** This is the FI*IG ("Fisher information by interval with geometric mean") paper. [CONFIRMED, abstract: *"FI*IG is the only ISR which simultaneously outperforms PFI in both variables [accuracy and security]."*] The v1 doc cited FI*IG to a 2010 paper, which was wrong: the 2010 Barrada paper is a separate six-rule comparison (PFI, FI-L, KL-L, MIS-B, PG, PP) that does NOT include FI*IG in its tested set.

- **Barrada, Olea, Ponsoda & Abad (2010), "A Method for the Comparison of Item Selection Rules in CAT," Applied Psychological Measurement 34(6):438–452.** Comparison of six rules; concludes best are KL-L, PG, MIS-B [CONFIRMED].

- **van der Linden & Pashley (2010)** chapter on item-selection rules in the CAT handbook. KL-information-over-interval is more robust at early CAT stages than pure Maximum Fisher Information at the current $\theta$ estimate (Chang & Ying 1996). Modern recommendation: KL-over-interval for the first ~25% of items, then switch to MFI. (Not in the audited corpus; cite as standard CAT methodology.)

**v2 synthesis — Q5.**

- **Fixed-length vs. adaptive:** adaptive (terminate on a posterior-precision criterion or on a question budget). ALEKS uses up to 30 items (fatigue-driven); Math Academy is variable. Item budgets are engineering choices, not graph-theoretic derivations.
- **Item selection during diagnostic:** information-gain-style selection (likelihood-near-$0.5$ splitting in ALEKS; information-gain greedy in MA's framing) is directionally supported [CONFIRMED]. For Delta Drills, you'll need to operationalize the scoring function and the cover/projection logic yourself — neither ALEKS nor MA publishes their specific algorithm.
- **Minimum N items [ENGINEERING]:** the original doc claimed 25–40 items based on "ALEKS and Math Academy reports"; ALEKS caps at 30 for fatigue reasons, MA is variable. Any range you choose is an engineering call.
- **Population-prior vs graph-propagated-prior:** Cosyn 2021 documents an empirical POPULATION prior; a per-learner graph-propagated prior (set $\alpha_{0,\text{atom}} = \alpha_{0,\text{default}} + \sum_{\text{parents}} w_{pa} \cdot \alpha_{pa} / \text{degree}$) is your construct, not ALEKS's mechanism.

### Q6. The composite-vs-procedural meta-architecture itself

**Audited primary references:**

- **Aleven & Koedinger (2013), "Knowledge component (KC) approaches to learner modeling," in Design Recommendations for Intelligent Tutoring Systems vol. 1, pp. 165–182.** [CONFIRMED title + scope + venue.] Focuses on KC model creation, refinement (LFA, AFM, SimStudent), and BKT/AFM for mastery tracking. Discusses "Cognitive Mastery" gating problem selection (p. 169). **Note:** the chapter does NOT distinguish "composite items that test multiple atoms" from "procedural items that test single atoms" at Delta Drills' granularity. In AK13 and KLI 2012, "grain size" refers to the LEVEL of cognitive task (word identification vs. letter recognition vs. multiplication). LFA can SPLIT a KC or MERGE two KCs — KCs at all grain sizes are atomic units, not a two-tier hierarchy. The v1 doc's claim that "the composite/procedural distinction is implicit in the KC framework as high-grain-size vs low-grain-size KCs" was UNSUPPORTED in audit.

- **Koedinger, Corbett & Perfetti (2012), "The Knowledge-Learning-Instruction framework," Cognitive Science 36(5):757–798.** Real framework with three taxonomies. Provides theoretical background on KCs but does not architecturally back a two-tier composite/procedural design.

- **Rittle-Johnson & Schneider (2015), Oxford Handbook of Numerical Cognition.** Developmental review; same caveat as Q4 (instructional order, not mastery dynamics or two-tier evidence granularity).

- **Rohrer (2012), "Interleaving helps students distinguish among similar concepts," Educational Psychology Review 24:355–367.** Main thesis: interleaving aids discrimination learning [CONFIRMED]. Cites e.g. Rohrer & Taylor 2007 (63% vs 20%, $d = 1.34$) and Taylor & Rohrer 2010 (77% vs 38%, $d = 1.21$) for interleaved-beats-blocked in math practice. **Note:** Rohrer's review is explicitly scoped to "difficult discriminations" and prescribes no numeric cap on consecutive same-item practice. Any specific cap you adopt is engineering judgment.

- **Koedinger & Aleven (2007), "Exploring the assistance dilemma in experiments with Cognitive Tutors," Educational Psychology Review.** The assistance dilemma is real [CONFIRMED]. KA07 frames it as a "fundamental open problem" needing further empirical research on "when and to what extent to use information giving versus information withholding forms of interaction" (abstract). **Note:** KA07's dilemma is WITHIN-problem (hints, feedback, worked examples vs. problem solving) — not a between-tier composite-vs-procedural dilemma. The v1 doc reframed it as the latter; that mapping is doc-author re-interpretation, not what the paper claims.

- **Math Academy's lesson/quiz/review distinction** — instructional-mode differentiation [CONFIRMED]. Not a multi-tag-vs-single-tag evidence granularity distinction.

**v2 synthesis — Q6 [NOVEL].**

**The two-tier composite/procedural architecture is genuinely novel to Delta Drills' published framing.** The audited sources do not architecturally back it. Specifically:

- KC framework grain-size $\neq$ Delta Drills' composite/procedural.
- Rittle-Johnson's iterative claim is about instructional order, not per-atom mastery dynamics.
- KA07's assistance dilemma is within-problem, not between-tier.
- Math Academy's lesson/review distinction is about instructional mode, not evidence granularity.

This does NOT mean the architecture is wrong — it means defend it on its own merits.

**Tradeoffs your two-tier design correctly exploits:**

- Fine-grained per-atom (procedural tier): clean signal for prereq gating; cold-start fragility.
- Coarse-grained per-composite (composite tier): faster signal (one composite informs $\geq 3$ atoms); but credit-assignment ambiguity prevents pinpointing weak atoms.
- Composite EWMA per area = fast, robust signal for demotion; procedural per-atom Beta = precise signal for promotion.

**Known failure modes (audited):**

1. **Punishment loops / over-aggressive demotion depth.** Oz Nova's documented critique [CONFIRMED verbatim quote above]. Mitigation: cap demotion depth (e.g., $\leq 2$ graph levels per event), add hysteresis after promotion, expose a "skip remediation" affordance.
2. **Credit-assignment instability at high tag counts.** Maier 2021 §4.2 documents three PFA degeneracy types and observed 9% $\gamma < 0$ and 11% $\gamma = \rho = 0$ in their baseline. **Important correction from v1:** Maier 2021 §4.3 found even-skill (= equal credit) AUC $0.7849$ BEAT compensatory PFA's $0.7818$ on their data. The original "equal credit is catastrophic" warning is not literature-supported. A more defensible framing: equal credit may interact with collinearity between skills, but is empirically competitive on at least one comparison; don't ban it categorically.
3. **EWMA-over-Beta double-counting.** Mathematically demonstrated [CONFIRMED] — only use a separate EWMA when you need a two-timescale signal.
4. **Conditional-completion debt** (MA's "conditionally completed" atoms could pile up if never re-tested by downstream tasks). Mitigation: periodic forced re-test.
5. **Motivation loss from frequent demotion.** Math Academy mitigates via XP rewards on remediation tasks; Duolingo via gamification. Plan the layer from day one.

---

## Most defensible technique combination (v0 stack)

Given the constraints (Beta posteriors, two-tier composite/procedural [NOVEL], prereq graph built, demotion-promotion controller, ARENA curriculum, ~200 atoms, small initial user base):

- **Per-atom procedural mastery [ENGINEERING]:** Beta-Bernoulli with multiplicative time decay. Single global half-life $\tau$ as v0 (any value — $7\text{d}$ or $14\text{d}$ are reasonable starting points), per-atom $\tau$ when data accumulates. Prior $\alpha_0 = \beta_0 = 1$ (uniform).
- **Composite-tier signal [ENGINEERING]:** EWMA on per-area composite accuracy, smoothing constant $\approx 0.3$. Used for demotion triggering. Whether to propagate to per-atom posteriors is your choice; if you do, prefer PFA-style logistic credit over equal-credit only on collinearity-concern grounds, not on "equal credit is catastrophic" grounds (which was audit-disputed).
- **Multi-tag credit on composites [CONFIRMED foundational form]:** PFA / LKT once you have data (Pavlik 2009 eq. 3). Below the convergence regime, use a tag-confidence-weighted Beta fallback [ENGINEERING]. There is no published N-threshold for the switchover; instrument and decide empirically (Maier 2021 suggests PFA convergence in the 2–12 practice range, not at $N = 30$).
- **Demotion trigger [ENGINEERING, inspired by MA]:** event-triggered "two failed composite attempts in area without intervening progress" plus a slow EWMA-below-threshold backstop. Treat as your own controller; don't claim it is MA's literal rule (MA's rule is at the lesson level, not the area level).
- **Procedural selection after demotion [ENGINEERING]:** greedy on lowest posterior mean among the failed composite's tagged prereqs, with parent-overlap and consecutive-same-atom diversification. Skip Thompson Sampling until $\geq 10\text{k}$ attempts.
- **Promotion trigger [ENGINEERING]:** posterior precision and mean thresholds + one successful test composite. Specific values are yours.
- **Procedural-vs-composite mixing [ENGINEERING]:** sigmoid in mean area-precision, non-monotonic (allows regression as forgetting decay outpaces evidence).
- **Cold-start diagnostic [ENGINEERING, directionally informed]:** information-gain item selection on a graph cover. ALEKS uses likelihood-near-$0.5$ splitting on a partitioned state space; MA describes a "compressed cover" without publishing the operationalization. Your scoring function and budget are your own; ALEKS's 30-item cap is a fatigue choice.
- **Cold-start prior [ENGINEERING]:** options include a uniform prior $\text{Beta}(1,1)$, a population-empirical prior (after some data accumulates — analogous to what Cosyn 2021 describes for ALEKS at the population level), or a per-learner graph-propagated prior

$$\alpha_{0,\text{atom}} = \alpha_{0,\text{default}} + \sum_{\text{parents}} w_{pa} \cdot \frac{\alpha_{pa}}{\text{degree}}.$$

The last is novel to your system.

---

## Where Delta Drills is likely to get into trouble

1. **EWMA-over-Beta double-counting** — mathematically redundant for static-skill. Be explicit about why two timescales are needed if you keep both.
2. **Trying full BKT or DKT at this scale** — identifiability issues documented in Pelánek 2017; Khajah 2016 shows enhanced BKT matches DKT, so the DKT premium is small. Beta-with-decay + logistic features is a reasonable choice for the regime.
3. **DINA / Q-matrix CDM** — heterogeneous edge-kind graphs typically violate identifiability conditions.
4. **Threshold-only demotion** — without an event trigger, the system demotes on noise. Combine slow (EWMA) and fast (event) signals.
5. **Monotonic mastery curves** — forces the system to ignore decay. Allow regression as a natural consequence of any forgetting-decay model.
6. **Underestimating motivation cost of demotion** — plan the gamification/XP layer alongside the algorithm.
7. **Treating your two-tier architecture as literature-validated** [NEW IN v2] — it is genuinely novel. Defend it on first principles and empirical results, not on appeals to KC framework / Rittle-Johnson / MA lesson-review structure (none of which architecturally back it).
8. **Citing numerics as literature-derived when they are not** [NEW IN v2] — $\tau = 14\text{d}$, $\alpha + \beta \geq 6$ or $8$, $N = 30$ for PFA switchover, cap-tags-at-6, 25–40 cold-start items, cap-3-consecutive-same-atom — all are engineering choices to tune from your own data.

---

## Specific algorithms / formulations to look up further

- **Half-life regression (Settles & Meeder 2016)** — regression-on-features, not a Beta-Bernoulli model; useful for predict-recall problems with rich feature vectors. Reference dataset and code are open.
- **PFA, R-PFA, LKT (Pavlik 2009; Galyardt & Goldin 2014; Pavlik et al. 2021 LKT)** — multi-tag credit assignment via logistic regression. LKT generalizes PFA's feature set.
- **Glicko / Glicko-2 (Glickman 2012)** — Elo with rating deviation that grows with time. Structurally analogous to (but not literally identical to) decaying-Beta posteriors.
- **M-ERS / Multidimensional ERS (Park, Cornillie, van der Maas & Van Den Noortgate 2019, Frontiers in Psychology 10:620)** — multivariate skill extension of Elo.
- **Knowledge Space Theory (Falmagne & Doignon 2011 book; Cosyn 2021; Doble 2019)** — for cold-start diagnostics over a finite state space.
- **KL information / FI*IG item selection (Chang & Ying 1996; Barrada 2009)** — diagnostic item selection rules.
- **PC-BKT with forgetting (Nedungadi & Remya 2015)** — for the explicit forget-rate parameterization in a BKT setting.
- **PDT / Performance Distribution Tracing (Bijl 2025)** — most recent Beta-with-decay-on-a-graph formalization. Closest published model to a per-atom Beta posterior with explicit decay.
- **pyBKT (Badrinath et al. 2021)** — reference implementation for A/B testing against classical BKT.
- **FIRe (Skycak 2023)** — Math Academy's spaced-repetition flow model. Useful as a target architecture for an eventual spaced-repetition layer.

---

## Math Academy publicly-documented algorithm — summary

Deepest public sources:

- **The Math Academy Way (Skycak & Roberts, working draft, justinmath.com/files/the-math-academy-way.pdf).** Long-form book, includes deep dives on FIRe and the diagnostic.
- **mathacademy.com/how-our-ai-works** — prose description of conditional completion, falling-backwards remediation, information-gain diagnostic selection.
- **justinmath.com/individualized-spaced-repetition-in-hierarchical-knowledge-structures/** — the FIRe paper with the actual update equations and the plateau Q&A.
- **Chalk & Talk Episode 42 (annastokke.com/ep-42-transcript)** — Smith and Skycak interview with concrete numbers (95%/99% pass rates).
- **Frank Hecker's "Math Academy, part 7: Technology brief"** and **Andy Matuschak's notes (notes.andymatuschak.org/Math_Academy)** — third-party deep dives.
- **Hacker News thread (news.ycombinator.com/item?id=39050945)** — Skycak's direct Q&A.

**Critical Math Academy facts [all CONFIRMED via verbatim audit unless noted]:**

- For the diagnostic algorithm specifically, Skycak tried BKT first and moved to a "physical" fractional-repetition flow because "complexity explodes." The FIRe mastery scoring is non-Bayesian: continuous $\text{repNum}$ counter updated as

$$\text{repNum} \to \max\!\bigl(0,\; \text{repNum} + \text{speed} \cdot \text{decay}^{\,\text{failed}} \cdot \text{rawDelta}\bigr),$$

with a separate exponentially-decaying $\text{memory}$ term. **Soft caveat:** Skycak's "tried BKT first" statement is specifically about the diagnostic algorithm; he does not claim MA uses zero Bayesian methods anywhere in the system.
- Demotion is event-triggered. **Two distinct mechanisms (don't conflate):** (a) knowledge-point halt within a lesson; (b) two-failed-lesson plateau across attempts. Plus quiz misses $\to$ immediate single-topic remediation. No published posterior threshold.
- Diagnostic is information-gain item selection on a compressed graph cover, with borderline atoms tagged "conditionally completed" and re-opened on downstream struggle.
- Quizzes target 80% accuracy via difficulty adaptation. Lesson question count is adaptive.
- Spaced-repetition credit flows through an "encompassing graph" separate from the prereq graph.
- Numerical thresholds and exact functional forms of $\text{speed}$, $\text{rawDelta}$, $\text{decay}$ are explicitly proprietary (Skycak 2023: "the specific implementation is proprietary"; Hecker confirms MAW also withholds them).

---

## Recommendations (staged)

**Stage 1 (ship v0, first ~50 users) [ALL ENGINEERING DEFAULTS — tune empirically]:**

- Beta-Bernoulli per atom with a global half-life (e.g., $\tau \approx 14$ days as a starting point — no literature-derived value).
- Tag-confidence-weighted Beta updates from composites (don't ship PFA until you have data to fit it cleanly).
- EWMA on per-area composite accuracy, smoothing constant your choice (e.g., $0.3$).
- Demotion = two failed composites in area without intervening progress (event trigger inspired by MA, not MA's literal rule).
- Greedy lowest-posterior-mean prereq selection with diversification.
- Promotion = posterior precision $\geq$ some $N_{\min}$ AND mean $\geq$ some $\theta$ on demotion-relevant atoms + one successful test composite. Values are your choice.
- Cold-start diagnostic: ~30 items (any budget you justify), information-gain-style selection on the graph (you operationalize the scoring function).
- **Stage-change benchmark:** ship to $\geq 50$ users; measure demotion frequency per area, procedural-to-promotion latency, user-reported frustration. Tighten thresholds before proceeding if demotion frequency is excessive or procedural dwell time is excessive.

**Stage 2 (when total attempts $\geq$ ~10k, value is your engineering call):**

- Switch composite-tier credit from tag-confidence-weighted to PFA/LKT (per Pavlik 2009 / LKT 2021). Estimate per-skill $\beta, \gamma, \rho$ from data. Watch for Maier 2021-documented degeneracy ($\gamma < 0$, $\gamma = \rho = 0$ cases).
- Estimate per-atom half-lives from data; shrinkage toward global $\tau$ as a regularizer.
- Add R-PFA-style two-timescale features (Galyardt & Goldin 2014) IF you observe non-stationary "current form" behavior; otherwise the decaying-$\alpha/\beta$ already encodes EWMA.
- Graph-propagate priors for cold-start (your own construct; ALEKS describes a population-level prior, which is different).
- **Stage-change benchmark:** if PFA cross-validated AUC exceeds tag-confidence-weighted baseline by some margin (your call), keep PFA; else revert.

**Stage 3 (when total attempts $\geq$ ~100k):**

- Consider M-ERS (Park 2019) as an alternative or complementary parameterization for ability vectors.
- Implement FIRe-style fractional implicit repetition across the encompassing graph (separate from prereq graph) for spaced-repetition.
- Hierarchical per-learner half-life priors across the population.
- Consider DKT/Transformer as a PREDICTOR (not a controller) to flag atoms with anomalous prediction errors for human review.

**Always:**

- Cap demotion depth (e.g., $\leq 2$ graph levels per event) to avoid Oz-Nova-style punishment loops.
- Provide a "skip remediation" affordance.
- Periodically force a "reality check" question on conditionally-mastered atoms.
- Log everything. These controllers tune empirically, not analytically.

---

## Caveats

- **Math Academy specifics are partial.** Published structural descriptions (FIRe, conditional completion, plateau rule, knowledge-point halt) are blueprint-level. Exact functional forms of $\text{speed}$, $\text{rawDelta}$, $\text{decay}$, and the diagnostic/demotion thresholds are explicitly proprietary [CONFIRMED].
- **The two-tier architecture is novel.** No audited source architecturally backs the composite-EWMA + procedural-Beta + event-triggered-demotion + sigmoid-mixing design. Defend on first principles and empirical results.
- **Engineering thresholds are your choice.** $\tau$, $\alpha + \beta$ cutoff, $N$-switchover for PFA, tag cap, cold-start item budget, blocking cap — all are choices to tune from your own data, not literature-derived defaults.
- **"Beta-with-decay" is a minority paradigm.** Most production adaptive learning systems use logistic models (HLR-style or PFA-style) rather than maintaining explicit Beta posteriors. Bijl 2025 (PDT) is the closest formal published model to your design, but the field is small.
- **The 5-edge-kind prereq graph is unusual.** Most published algorithms assume single-edge-type graphs (prereq or is-a). Your edge kinds will need a per-edge-type weighting convention for any algorithm that treats tagged KCs additively.
- **Cold-start prior propagation from a prereq graph** is theoretically clean but practically tricky. First 5–10 users will see suboptimal initial estimates; plan UX accordingly.
- **PFA-vs-tag-confidence-weighted switchover regime** depends on tag-count distribution and per-atom attempts. A/B test, don't switch on intuition.
- **Interleaving research is largely lab-scale.** Practical scheduling defaults (e.g., consecutive-same-atom caps) are starting heuristics to tune.
- **Deployed demotion/promotion controllers are underspecified in academic literature.** Math Academy, ALEKS, Knewton, Cognitive Tutor all treat their rules as trade secret or implementation detail. Expect to iterate.

---

## What changed from v1 (2026-05-24 audit summary)

Audit covered 59 load-bearing claims across 4 clusters. Result: 20 CONFIRMED, 20 PARTIAL, 7 DISPUTED, 8 UNSUPPORTED, 4 CANT_TELL. Key v1 $\to$ v2 corrections:

- **HLR is not Beta-Bernoulli.** v1 attributed a specific Beta-with-decay update rule to Settles-Meeder; the rule appears nowhere in the paper. v2 presents Beta-with-decay as an engineering construct.
- **Pelánek 2017 Hypothesis 1 is the opposite of v1's framing.** v1 claimed Pelánek endorses "Beta-Bernoulli + logistic at low $N$." Pelánek's H1: logistic for fluency/memory, BKT for understanding/sense-making. v2 corrects.
- **"Equal credit catastrophic" is not literature-supported.** Maier 2021 §4.3 found even-skill (= equal credit) AUC $0.7849$ BEAT compensatory PFA's $0.7818$ on their data. v2 removes the categorical warning.
- **Two-tier architecture's theoretical backing was fabricated.** KC-framework grain-size $\neq$ composite/procedural; Rittle-Johnson is about instructional order; KA07 is within-problem. v2 presents the architecture as novel.
- **Bijl 2025 title corrected:** "Tracking Student Skills Real-Time Through a Continuous-Variable Dynamic Bayesian Network" (method PDT = Performance Distribution Tracing). v1 called it "Probabilistic Decay Trees."
- **M-ERS attribution corrected:** Park, Cornillie, van der Maas & Van Den Noortgate 2019, Frontiers in Psychology 10:620. v1 attributed to "Doebler/Pelánek/Wauters 2018."
- **Maier 2021 title corrected:** "Challenges to Applying PFA to Existing Learning Systems." v1 called it "Improving PFA for rare skills."
- **FI*IG citation corrected:** Barrada 2009, not 2010. The 2010 paper is a separate six-rule comparison not including FI*IG.
- **"Compressed graph cover" attributed correctly:** Math Academy's marketing prose, not ALEKS. ALEKS uses likelihood-near-$0.5$ splitting on a partitioned state space.
- **MA demotion has two distinct mechanisms** (knowledge-point halt + two-failed-lesson plateau), not one rule. v2 separates them.
- **All numeric thresholds ($\tau = 14$d, $\alpha + \beta$ cutoff, $N = 30$, tag-cap=6, 25–40 items, cap-3) presented as engineering choices**, not literature-derived defaults.
- **Yudelson-Pavlik 2013 "monotonicity anti-pattern" citation dropped** (paper not in audited corpus); v2 presents non-monotonicity as a natural consequence of any forgetting-decay model.
- **Rohrer 2012 cap-of-3 dropped** as Rohrer-attributed (he prescribes no numeric cap); v2 treats as engineering judgment.
- **"Not Bayesian" claim softened** — Skycak's "tried BKT, complexity explodes" statement is specifically about the MA diagnostic algorithm; not a categorical denial of all Bayesian methods.

Full per-claim audit in `VERIFICATION_SUMMARY.md` and `VERIFICATION_{A,B,C,D}_*.md`.
