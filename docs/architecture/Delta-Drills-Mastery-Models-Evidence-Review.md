# Delta Drills — Mastery Models Evidence Review

> **Purpose.** Lay out the four mastery-estimation models under consideration for the prerequisite-gated bootcamp app, present the underlying mathematics, summarise the two AI-generated reports already in hand, cross-check their claims against the audited primary sources in `papers/mastery-estimation/`, and surface the open decision points. **No recommendation is made here** — this is an evidence dossier for personal evaluation.
>
> Companion document: `Delta-Drills-Current-Model-EWMA.md` (full specification of the engine as deployed today).

---

## 0. The four candidate models

In each subsection: per-skill state, update rule, time decay (if any), the form of uncertainty (if any). Notation: $s$ is a skill; $n$ is the attempt index for that skill; $y_n\in\{0,1\}$ is the correctness of attempt $n$; $\Delta t$ is days elapsed since the previous attempt.

### 0.1 Exponentially Weighted Moving Average (current Delta Drills stack)

**Per-skill state.** Two numbers plus a counter: a baseline $b_s$ (EWMA of `grade × difficulty / 100`), a correctness rate $p_s$ (EWMA of `1[grade > 85]`), an attempt count $n_s$, and a last-update timestamp.

**Update.** On attempt $n$ with feedback-derived weight $\alpha(n)\in\{0.30,0.60,0.85\}$ and difficulty $d_n$:

$$
b_n = \alpha(n)\cdot\frac{\mathrm{grade}_n\cdot d_n}{100} + (1-\alpha(n))\,b_{n-1}, \qquad
p_n = \alpha(n)\cdot\mathbf{1}[\mathrm{grade}_n>85] + (1-\alpha(n))\,p_{n-1}.
$$

Target next difficulty: $b_n\cdot\mu(p_n)$ where $\mu$ is the piecewise multiplier capped at $2.5$.

**Time decay.** Yes — multiplicative shrinkage of both $b$ and $p$ toward priors $B_0=0$, $P_0=0.5$ at half-life $\tau_{1/2}=14$ days:
$$
b \leftarrow B_0 + (b-B_0)\cdot 2^{-\Delta t/\tau_{1/2}}, \quad p \leftarrow P_0 + (p-P_0)\cdot 2^{-\Delta t/\tau_{1/2}}.
$$

**Uncertainty.** None explicit. The values are always defined; there's no signal for "I don't have enough data on this skill."

**Multi-tag credit.** When a composite drill is rated, each tagged subtopic gets an independent EWMA update (equal-credit averaging).

### 0.2 Elo rating system

**Per-skill state.** One number per skill ($R^{\mathrm{skill}}_s$) and one number per student ($R^{\mathrm{user}}$).

**Predicted success on an attempt:**
$$
\hat{y} \;=\; \frac{1}{1 + 10^{(R^{\mathrm{skill}}_s - R^{\mathrm{user}})/400}}.
$$

**Update on observed $y\in\{0,1\}$:**
$$
R^{\mathrm{user}} \leftarrow R^{\mathrm{user}} + K\cdot(y - \hat{y}), \qquad
R^{\mathrm{skill}}_s \leftarrow R^{\mathrm{skill}}_s - K\cdot(y - \hat{y}).
$$

The constant $K$ caps how much one attempt can move either rating. Educational Elo typically uses different $K$ values for student and skill, and often a decaying $K$ (large for cold start, smaller as evidence accumulates).

**Time decay.** Not in standard Elo. Glicko/Glicko-2 extensions add a "rating deviation" (RD) that grows over time, mimicking forgetting; vanilla Elo has none.

**Uncertainty.** Standard Elo: none. Glicko-2: RD plays this role.

**Multi-tag credit.** Standard approach: average the per-skill ratings of all tags on the item; update each tag's rating independently from the observed outcome. **This is a known weakness** — a student strong on tag A and weak on tag B gets an averaged prediction that's wrong in both directions. The published fix is **M-ERS (Multidimensional Elo Rating System)** from Park et al. 2019, which maintains a vector of ability estimates rather than a scalar.

### 0.3 Streak / mastery counter

**Per-skill state.** One integer: the count of consecutive correct answers, plus optionally a "mastered" flag.

**Update.**
$$
\text{streak}_s \leftarrow \begin{cases} \text{streak}_s + 1 & y_n = 1 \\ 0 & y_n = 0 \end{cases}
$$
Declare "mastered" when $\text{streak}_s \ge N$ for some threshold $N$ (Khan Academy historically used $N=3$ or $5$).

**Time decay.** None native. Some implementations add: "reset to zero if no attempt for $T$ days."

**Uncertainty.** None.

**Multi-tag credit.** Each tag carries its own independent streak.

### 0.4 Bayesian Beta-Bernoulli with time decay

**Per-skill state.** A Beta posterior parametrized by two counters $(\alpha_s, \beta_s)$.

**Update on observed $y\in\{0,1\}$:**
$$
\alpha_s \leftarrow \alpha_s + y, \qquad \beta_s \leftarrow \beta_s + (1-y).
$$

**Time decay between attempts** (engineering choice, not part of vanilla Beta-Bernoulli): multiplicative shrinkage toward the prior $(\alpha_0,\beta_0)$ at half-life $\tau$:
$$
\alpha_s \leftarrow \alpha_0 + (\alpha_s-\alpha_0)\cdot 2^{-\Delta t/\tau}, \qquad
\beta_s \leftarrow \beta_0 + (\beta_s-\beta_0)\cdot 2^{-\Delta t/\tau}.
$$

**Uncertainty.** Yes, native:
- Posterior mean: $\mu_s = \dfrac{\alpha_s}{\alpha_s + \beta_s}$
- Posterior precision: $\nu_s = \alpha_s + \beta_s$ (a direct measure of evidence count)
- Posterior variance: $\dfrac{\alpha_s \beta_s}{(\alpha_s+\beta_s)^2(\alpha_s+\beta_s+1)}$

Gating typically requires BOTH the mean and the precision to clear thresholds:
$$
\text{Ready}(s) \;\equiv\; \mu_s \ge \theta_{\mu} \;\wedge\; \nu_s \ge \nu_{\min}.
$$

**Multi-tag credit.** Each tag updated independently (equal credit), same as the current Delta Drills approach.

---

## 1. Comparison axes

### 1.1 Per-skill state size

| Model | Numbers stored per skill | Numbers stored per student |
|---|---|---|
| Current EWMA | 3: $b_s$, $p_s$, $n_s$ + timestamp | 0 |
| Elo | 1: $R^{\mathrm{skill}}_s$ | 1: $R^{\mathrm{user}}$ |
| Streak | 1: streak count | 0 |
| Beta-Bernoulli | 2: $\alpha_s$, $\beta_s$ + timestamp | 0 |

### 1.2 Cold-start behavior

| Model | Behavior with 0 attempts | Behavior with 1 attempt |
|---|---|---|
| Current EWMA | Cold-start ladder serves difficulty 25/50/75 for first 3 questions | After 1 attempt, $b$ and $p$ exist but variance is implicit |
| Elo | Initialized to default rating (typically 1500); first attempt produces useful update | Yes, usable |
| Streak | 0/0; mastery flag off | $\text{streak} = 1$, mastery flag off |
| Beta-Bernoulli | Prior $(\alpha_0=1, \beta_0=1)$ gives mean $0.5$, precision $2$ | Updated immediately; precision $= 3$, mean tracks the observed outcome |

### 1.3 Built-in uncertainty signal

| Model | Has uncertainty? | What plays that role |
|---|---|---|
| Current EWMA | No | — |
| Elo | No (standard); yes in Glicko/Glicko-2 | Rating deviation (RD) in Glicko |
| Streak | No | — |
| Beta-Bernoulli | Yes, native | Posterior precision $\nu = \alpha + \beta$ |

### 1.4 Time decay / forgetting

| Model | Native decay? | Mechanism |
|---|---|---|
| Current EWMA | Yes (14-day half-life) | Multiplicative shrinkage of $b$, $p$ toward priors |
| Elo | No in standard form | Glicko adds RD that grows over time |
| Streak | No native; some impls add reset-on-stale | — |
| Beta-Bernoulli | No native | Engineering: multiplicative shrinkage of $\alpha$, $\beta$ toward $(\alpha_0,\beta_0)$ |

### 1.5 Multi-tag credit assignment

| Model | Default approach | Known weakness | Fix in literature |
|---|---|---|---|
| Current EWMA | Equal credit: each tag gets its own independent update | Same as PFA-compensatory but Maier 2021 found equal-credit beat compensatory empirically | None needed per Maier |
| Elo | Average ratings of tags; update each independently | "Strong-on-A, weak-on-B" wrong-in-both-directions | M-ERS (Park 2019) — multidimensional ability vector |
| Streak | Independent streak per tag | A student short on one tag never unlocks even if strong on others | — |
| Beta-Bernoulli | Independent $(\alpha,\beta)$ per tag | Same as EWMA equal-credit | — |

### 1.6 ARENA-difficulty / ceiling problem

| Model | Has the problem? | Why / why not |
|---|---|---|
| Current EWMA | Yes — `target_difficulty` clamped to [0,100] | Global scalar with hard ceiling |
| Elo | No | Ratings are relative; no fixed ceiling |
| Streak | No, but for a worse reason | No scale at all — over-promotion failure mode |
| Beta-Bernoulli | No | Per-skill probability, no global scale |

### 1.7 Interpretability to students

| Model | What you can show the student | Honest interpretation |
|---|---|---|
| Current EWMA | "72% mastery" | Intuitive but conflates difficulty and accuracy |
| Elo | "Your rating: 1340, skill rating: 1280" | Opaque without context |
| Streak | "3 in a row → mastered" | Intuitive but loses partial state |
| Beta-Bernoulli | "78% likely correct (based on 14 attempts)" | Most honest — explicit confidence |

### 1.8 Implementation cost from current state

| Model | Required changes | Risk |
|---|---|---|
| Current EWMA | None — already deployed | — |
| Elo | Replace `apply_feedback` math; add per-user rating storage; redefine ALL gate thresholds for the new relative scale | Moderate. Migration of existing user state is non-trivial. |
| Streak | Trivial — counter per skill. But would mean abandoning a lot of nuance. | Low engineering risk, high pedagogical risk |
| Beta-Bernoulli | Replace `apply_feedback` math with $(\alpha,\beta)$ update; add decay; gates need both mean + precision conditions | Moderate. State migration is straightforward (transform existing $p$ and $n$ into compatible $(\alpha,\beta)$). |

---

## 2. Failure modes per model

### 2.1 Current EWMA

1. **Ceiling effect on `target_difficulty`.** Once a student plateaus on flashcards, target_difficulty clamps at 100 and the engine has no way to represent "ready for harder-than-flashcard content." Real, affects ARENA gating.
2. **No native confidence signal.** A student with 3 attempts is treated like a student with 30; the only way to know how thin the evidence is is to read $n_s$ separately.
3. **Two-EWMA aggregation.** Engine stores both $b_s$ and $p_s$. Gating code uses $p_s\cdot 100$; difficulty selection uses $b_s\cdot\mu(p_s)$. These can disagree (high $p$ with low $b$ = lucky on easy questions; passes gate, shouldn't).

### 2.2 Elo

1. **No forgetting in standard form.** A skill mastered a year ago is treated identically to one mastered yesterday unless you bolt on Glicko-style RD growth.
2. **Multi-tag averaging is a known weakness** (cf. Park 2019).
3. **K-factor calibration is non-trivial.** Pelánek 2016 specifically tunes separate $K$ values for student vs. skill ratings, and often a decay schedule for $K$ as evidence accumulates. One global $K$ misbehaves at both cold-start and steady-state regimes.
4. **Gate semantics shift entirely.** Current gates like "p × 100 ≥ 70" become "$R^{\mathrm{user}} \ge R^{\mathrm{skill}}_s + \delta$" — a completely different threshold semantics that needs re-tuning skill-by-skill.

### 2.3 Streak / mastery counter

1. **Over-promotion failure mode.** A student who gets 3 difficulty-25 cold-start questions right has the same streak as a student who got 3 difficulty-90 questions right. **No difficulty awareness.** This is a *worse* failure than the EWMA's ceiling problem — the model gates students as "ready for ARENA" on the basis of having gotten 3 easy questions correct.
2. **No partial credit / no graduated mastery state.**
3. **No regression handling.** Hard reset on a wrong answer destroys all built-up evidence in one bad attempt.

### 2.4 Bayesian Beta-Bernoulli

1. **No difficulty awareness in the base form.** Like the streak counter, the $(\alpha,\beta)$ update treats all correct/wrong answers identically regardless of question difficulty. Mitigation: weight the updates ($\alpha \leftarrow \alpha + w\cdot y$) where $w$ scales with difficulty — but this is engineering choice not in the audited literature.
2. **Decay rate has to be set by hand.** Same as the EWMA's 14-day choice — no published study tells you the right number for a bootcamp context.
3. **Specific architecture (Beta + decay + prereq graph) has thin literature.** One audited paper (Bijl 2025). The underlying Bayesian math is textbook; the deployment in this exact configuration is largely engineering.

---

## 3. Summary of the two AI reports

### 3.1 Report 1 — "Gap Analysis"

**Format.** Goes through each model and lists what was missing or under-developed in the prior conversation. Critique-focused, no citations.

**Material new contributions over our prior conversation:**

1. Reframes streak's failure mode: it's *over-promotion via easy questions*, not ceiling effects. This is correct and important.
2. Catches that the switching cost for Elo is higher than I'd characterized: gate threshold semantics change fundamentally, not just the storage layer.
3. Identifies that NO model addresses prerequisite-graph errors (a cross-cutting blind spot).
4. Raises post-mastery regression UX: what does a student SEE when their mastery slides back down? Each model handles this differently and none was compared.
5. Raises student-facing explainability as a UX dimension never compared across models.
6. Calls out that multi-tag Elo averaging is a published weakness with a published fix (M-ERS / Park 2019).
7. Notes that K-factor calibration depth was hand-waved in the prior comparison.
8. Notes Pelánek 2016 (focused study on Elo for adaptive learning) is a different paper from Pelánek 2017 (broader survey of BKT/PFA/Elo), and they should be cited distinctly.

**Factual issues in Report 1:**

- Claims a Beta-Bernoulli formula typo of the form `alpha → alpha + beta + (1 if wrong else 0)`. **This formula was never written.** The prior conversation showed the standard `alpha → alpha + 1 (on success), beta → beta + 1 (on failure)`. Report 1's typo allegation is unfounded.
- Attributes a "5+ attempts before estimates settle" claim to EWMA. The number is from Maier 2021's empirical study of Performance Factors Analysis (a different model entirely). The current EWMA settles much faster — with $\alpha=0.85$ on "A lot" feedback, the EWMA's effective memory length is ~1.2 attempts.
- Says "EWMA formula never shared." False as of today — the formulas are fully spelled out in `Delta-Drills-Current-Model-EWMA.md`.

### 3.2 Report 2 — "Comparative Review"

**Format.** Survey-style with pros/cons per model, inline citation markers (`【1†L121-L124】` etc.), and a recommendation at the end. Cleaner narrative but less critical.

**Material claims in Report 2:**

1. Names Pelánek 2016 and 2017 as the Elo sources for educational use.
2. Cites a specific empirical figure: *"Elo's task-difficulty estimates had r ≈ 0.70 correlation with ground truth after just 5 responses"* — attributed to citation 【16†L47-L52】.
3. Recommends Elo as the migration target.
4. Identifies Beta-Bernoulli as "principled" but "less field-tested" — accurate.
5. Frames streak as a baseline that more sophisticated models outperform (consistent with Maier 2021's framing).

**Issues with Report 2:**

- The citation markers (`【1†...】`, `【16†...】`, etc.) are opaque — without the source list it's impossible to tell which paper the "r ≈ 0.70 after 5 responses" claim originates from. **This number is not directly verifiable from the report alone.** Highly recommend reading Pelánek 2016 §results to confirm whether this claim exists in the original paper at the granularity stated.
- Does not catch the gaps Report 1 catches (streak failure mode reframing, gating-semantics-change, prereq-graph-error blind spot, post-mastery regression, explainability, M-ERS for multi-tag Elo).
- Hand-waves multi-tag aggregation similarly to the prior conversation (no mention of M-ERS / Park 2019, which IS in the audited corpus).
- Claims "moderate work" for Elo migration without addressing the gate-semantics shift.

### 3.3 Where the two reports agree

- Elo is the strongest single recommendation if a switch is to be made.
- Beta-Bernoulli is principled but under-published in this exact architecture (one paper, Bijl 2025).
- Streak alone is too weak.
- EWMA is functional but has overconfidence/ceiling problems.

### 3.4 Where the two reports differ

| Topic | Report 1 | Report 2 |
|---|---|---|
| Streak's failure mode | Over-promotion via easy questions | Coarse, no partial credit |
| Switching cost for Elo | "Migration would require redefining every gate threshold" | "Moderate work… retune thresholds for new scale" |
| Multi-tag Elo | Known weakness, points at M-ERS / Park 2019 | Default averaging is fine, no concern raised |
| Prereq-graph errors | Cross-cutting blind spot, all four models | Not raised |
| Explainability | Real UX gap | Not raised |
| Strength of recommendation | None — only lists open questions | Elo, with migration steps |

---

## 4. Cross-check of specific claims against your audited corpus

The audited papers in `papers/mastery-estimation/` are the primary sources. Below: which report's claims can be checked against which paper, and the current state of verification.

| Claim | Source claimed | Audited paper | Verification status |
|---|---|---|---|
| "PFA's compensatory model AUC 0.78; even-skill averaging AUC 0.78; conjunctive AUC 0.67" | Maier 2021 | `2021_maier-baker-stalzer_pfa-challenges.pdf` §4.3 | **Verified** — read end-to-end. Numbers are: compensatory 0.7818, conjunctive 0.6725, even-skill 0.7849. |
| "PFA needs 5+ practices per (student, skill) for reliable estimates" | Maier 2021 | Same | **Verified** §4.1 — AUC went 0.725 → 0.755 from 2→5 practices, plateaued at ~12 practices. |
| "Math Academy FIRe formula: `repNum → max(0, repNum + speed · decay^failed · rawDelta)` and memory decay $(0.5)^{\mathrm{days/interval}}$" | Skycak 2023 | `2023_skycak_fire-spaced-repetition.html` | **Verified** — read end-to-end. Formulas reproduced verbatim from the paper. |
| "Skycak abandoned Bayesian Knowledge Tracing for Math Academy because complexity exploded" | Skycak 2023 | Same | **Verified verbatim** — direct quote: *"complexity explodes when you have to make tons of different kinds of decisions… I ended up moving towards less of a probabilistic approach and more of a 'physical' approach."* |
| "Elo's task-difficulty estimates had r ≈ 0.70 correlation with ground truth after just 5 responses" (Report 2) | Citation 【16†L47-L52】 | Likely `2016_pelanek_elo-adaptive-educational.pdf` | **Not yet verified.** Recommend reading Pelánek 2016 §results to confirm the exact figure and the conditions under which it was observed. |
| "Pelánek 2017 Hypothesis 1: logistic models better for fluency/memory, BKT better for understanding/sense-making" | Pelánek 2017 | `2017_pelanek_bkt-logistic-and-beyond.pdf` p. 33 | **Verified via prior audit** (the v2 mastery doc cross-checked this and noted the v1 doc misframed it). |
| "Elo is simple, robust, and effective for adaptive education" | Pelánek 2016 | `2016_pelanek_elo-adaptive-educational.pdf` abstract | **Likely verified** (the v2 audit doc confirms abstract quote). Re-reading the §results details is the missing step. |
| "M-ERS (Multidimensional Elo Rating System) extends Rasch-Elo to multidimensional ability vectors, fixing the multi-tag averaging weakness" | Report 1 | `2019_park-cornillie_multidim-irt-monitoring-ability.pdf` | **Not yet verified end-to-end.** The v2 audit doc confirms the paper exists and introduces M-ERS; the specific claim about multi-tag handling is consistent with the paper's framing but the implementation details warrant a primary-source read. |
| "Beta-Bernoulli + decay over a prereq graph is published only in Bijl 2025" | Both reports | `2025_bijl_probabilistic-decay-trees.pdf` | **Partially verified.** Audit doc confirms Bijl 2025 uses Beta-distribution basis functions, continuous-variable Dynamic Bayesian Network, explicit decay over a graph of skills. **Not yet read end-to-end.** Recommended if Beta-Bernoulli is seriously considered. |
| "Streak / three-in-a-row is a baseline more sophisticated models outperform" | Report 2 | Maier 2021 mentions this framing in §1 | **Verified** — Maier 2021 explicitly says: *"most commercial adaptive learning systems still continue to either use BKT… or simpler heuristics such as three-in-a-row correct."* |
| "Khan Academy used streak-based mastery and moved away from it" | Both | Not in audited corpus | **Unverified.** Common knowledge in the ed-tech world but no primary-source citation in the corpus. |
| "Standard Elo K-factor needs separate calibration for student vs skill ratings" | Report 1 | Pelánek 2016 / Yudelson 2019 | **Likely verified** (the v2 audit doc notes Yudelson 2019: *"Elo, I Love You Won't You Tell Me Your K"* is the dedicated K-tuning paper). Yudelson 2019 is in the corpus references but not in the downloaded set — would need to retrieve. |

---

## 5. Open decision points neither report fully resolves

These are the questions that need answers before code changes are made. Each is followed by where the answer would come from.

1. **Gate threshold values for any chosen model.** For Beta-Bernoulli: what $\theta_\mu$ and $\nu_{\min}$? For Elo: what rating differential between student and skill counts as "ready"? Neither report quantifies these. **Resolvable only with simulation or first-cohort data.**

2. **K-factor schedule for Elo.** Single value, two values (student vs skill), or decaying with experience? **Pelánek 2016 + Yudelson 2019 cover this directly. Worth reading both before committing.**

3. **Decay half-life for Beta-Bernoulli.** Current EWMA uses 14 days. Same value for Beta? Different? **No published guidance — engineering choice, tunable from data.**

4. **Multi-tag credit for any chosen model.** Equal-credit averaging (current) per Maier 2021 is empirically reasonable. M-ERS upgrade for Elo is published (Park 2019) but adds significant complexity. **Resolution: read Park 2019 to assess whether the M-ERS upgrade is worth the complexity.**

5. **Prerequisite-graph error resilience.** None of the four models addresses this. The graph itself is an instructor-authored artifact subject to error. **Resolution: graph-level (instructor review, sample-edge validation), not model-level. Currently deferred.**

6. **Post-mastery regression UX.** What does the student see when mastery slides back down? Each model produces different behavior; no comparison done. **Resolution: product-design decision, not model-selection decision.**

7. **Migration cost from current EWMA state.** For Elo: how do you initialize $(R^{\mathrm{user}}, R^{\mathrm{skill}}_s)$ from existing $(b_s, p_s, n_s)$? For Beta: can you transform $p_s, n_s$ into $(\alpha_s, \beta_s)$ such that the migration preserves user mastery estimates? **Resolution: needs a 1-page migration spec per chosen target model.**

8. **Explainability target.** Do you want students to see a mastery percentage? Does the chosen model support that natively? **Affects choice among Elo (no native percentage) vs. Beta (native posterior probability) vs. EWMA (current 0-100).**

---

## 6. Evidence tier per claim

A rough rubric for how much weight to put on each claim.

| Tier | Definition | Examples |
|---|---|---|
| **Textbook** | Foundational math, not in dispute | Beta-Bernoulli conjugacy; Elo logistic prediction; EWMA convergence properties |
| **Peer-reviewed empirical** | Published in a venue with peer review, reproduced or replicated | Maier 2021 PFA AUC numbers; Pelánek 2016/2017 surveys of BKT/Elo |
| **Peer-reviewed novel proposal** | Published but as a single-paper architecture, not replicated | Park 2019 M-ERS; Bijl 2025 PDT |
| **Production-validated, proprietary** | Deployed in real systems but specific implementation not published | Math Academy FIRe (formulas public, implementation proprietary) |
| **Engineering construct** | Defensible but not from a primary source | Current Delta Drills 14-day half-life; the 0.30/0.60/0.85 alpha choices; the cold-start ladder 25/50/75 |
| **Anecdotal** | Stated without primary-source backing | "Khan Academy moved away from streak-based mastery" |

Mapping each model's components to this tier:

| Model | Mathematical core | Educational application | Specific parameter values |
|---|---|---|---|
| Current EWMA | Textbook | Engineering | Engineering (chosen, not derived) |
| Elo | Textbook | Peer-reviewed empirical (Pelánek 2016, 2017) | Engineering (K, decay schedule) |
| Streak | Trivial | Anecdotal / baseline | Engineering (threshold N) |
| Beta-Bernoulli | Textbook | Peer-reviewed novel proposal (Bijl 2025 only) | Engineering (decay half-life, gate thresholds) |

---

## 7. Reading list for personal evaluation

In the audited corpus at `papers/mastery-estimation/`. Listed in priority order if you want to verify the most load-bearing claims:

1. **`2016_pelanek_elo-adaptive-educational.pdf`** — Pelánek's focused study of Elo in adaptive education. If you're considering Elo, this is the primary source. **Read this to verify the "r ≈ 0.70 after 5 responses" claim from Report 2** and to understand K-factor calibration choices.

2. **`2017_pelanek_bkt-logistic-and-beyond.pdf`** — Pelánek's broader survey. Less focused on Elo specifically but contextualizes it against BKT and logistic models. Hypothesis 1 (p. 33) is the load-bearing claim about when to prefer logistic vs BKT.

3. **`2019_park-cornillie_multidim-irt-monitoring-ability.pdf`** — Park 2019 / M-ERS. **Read this if multi-tag Elo aggregation is a concern.** Determines whether the upgrade is worth the complexity vs. just doing equal-credit averaging.

4. **`2021_maier-baker-stalzer_pfa-challenges.pdf`** — Maier 2021. **Already read in full.** Headline finding: even-skill averaging (what your current engine does for composites) AUC 0.7849 beat compensatory Performance Factors Analysis AUC 0.7818. Section 4.3 is the relevant comparison.

5. **`2023_skycak_fire-spaced-repetition.html`** — Math Academy's published architecture. **Already read in full.** Verbatim formulas for FIRe. Math Academy's deliberate rejection of Bayesian Knowledge Tracing is on the record here.

6. **`2025_bijl_probabilistic-decay-trees.pdf`** — Bijl 2025. **Recommended if Beta-Bernoulli is seriously considered.** The only published instance of a Beta-with-decay-over-graph mastery system. Single author, arXiv preprint, untested at scale — read critically.

7. **`2017_pelanek_bkt-logistic-and-beyond.pdf`** (same as 2 above) — also relevant for "why not BKT" rebuttals.

Not in the downloaded corpus but referenced and potentially worth retrieving:
- **Yudelson 2019, "Elo, I Love You Won't You Tell Me Your K"** — the focused paper on K-factor calibration.
- **Glickman, "Glicko-2"** — the standard extension adding rating deviation (forgetting) to Elo.

---

*Documents in this series:*
- `Delta-Drills-Current-Model-EWMA.md` — complete specification of the engine deployed today
- `Delta-Drills-Mastery-Models-Evidence-Review.md` — this document
- `papers/MASTERY_ESTIMATION_REFERENCE_v2.md` — audited reference (v2 after the 2026-05-24 source audit that flagged ~1/3 of v1 claims as wrong or fabricated)
