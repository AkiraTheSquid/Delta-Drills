# VERIFICATION A — Bayesian / BKT / Decay claims (Cluster A, Q1)

**Date:** 2026-05-24
**Doc audited:** `compass_artifact_wf-2f4da85b-a0ab-425d-9877-f25787abbdea_text_markdown.md`
**Sources checked:** Settles & Meeder 2016 (HLR), Pelánek 2017 (BKT survey), Pelánek 2016 (Elo), Khajah 2016 (DKT), Bijl 2025 (PDT), Nedungadi 2015 (PC-BKT), Qiu 2011 (BKT-time)
**Scope:** every load-bearing claim about the recommended "Beta-Bernoulli with multiplicative half-life decay" stack.

Verdict legend: CONFIRMED · PARTIAL · DISPUTED · UNSUPPORTED · CANT_TELL_NO_SOURCE.

---

## 1. "HLR is the closest published model to the recommended Beta-with-decay primitive"

**Doc quote (line 11, 37, 222):** "The PDT … is the most recent formalization … cites the same Beta-Bernoulli backbone the user proposes" and (separately) "Half-life regression (HLR) … is the formal trainable version of the user's Beta-with-decay."

**Verdict: DISPUTED.**

**Source:** Settles & Meeder 2016, §3.3 and Appendix A.3 (pp. 1851, 1858).

**Justification:** HLR is *not* a Beta-Bernoulli model. The HLR model has two equations: `p = 2^(-Δ/h)` (Ebbinghaus recall curve, eq. 1) and `ĥ = 2^(Θ·x)` (regression on feature vector, eq. 2). It is trained by L2-regularized squared-loss gradient descent (Appendix A.3). There is no posterior, no α/β, no conjugate prior — it is a discriminative regression on engineered features (counts of right/wrong + lexeme indicators). The "closest published model" claim is doc-author analogy, not a structural equivalence in either paper. HLR and Beta-Bernoulli are related only in that both can encode an exponential forgetting curve — but Beta-Bernoulli maintains uncertainty as a posterior distribution, HLR maintains it as a point estimate of a half-life. The doc later asserts (line 51) that "treat (α+β) as effective sample count and apply n_eff ← n_eff · 2^(−Δt/τ) directly; this is the formulation in Settles & Meeder." **This formulation appears nowhere in Settles & Meeder.** The paper never discusses effective sample count or any Beta posterior.

---

## 2. "HLR formulation: half-life regression on review intervals (Duolingo)"

**Doc quote (line 37):** "Half-life regression (HLR) on Duolingo's 13M user-word trace data. The model predicts P(recall) = 2^(−Δ/h) where h is a learned half-life that depends on per-item features and a count of right/wrong answers."

**Verdict: CONFIRMED.**

**Source:** Settles & Meeder 2016, §3.3 (eq. 1, 2, p. 1851) and §4.1 (p. 1853, "12.9 million instances").

**Justification:** Equations 1–2 and the training set size are reproduced exactly. The "12% engagement lift" and "45%+ MAE reduction" headline numbers in the doc match the abstract verbatim. This is the one claim about HLR the doc gets right.

---

## 3. "Beta-Bernoulli with multiplicative half-life decay on α+β is Settles-Meeder's formulation"

**Doc quote (line 11):** "α ← α·exp(−Δt/τ) + α₀·(1−exp(−Δt/τ)), β analogously — i.e., the posterior shrinks toward the prior with half-life τ" attributed to Settles-Meeder style.

**Verdict: UNSUPPORTED (doc-author synthesis presented as published formulation).**

**Source:** Settles & Meeder 2016 (whole paper).

**Justification:** The shrink-toward-prior Beta-Bernoulli update is a sensible Bayesian construction, but it does not appear in Settles-Meeder. HLR predicts recall probability via `2^(-Δ/h)` and learns `h` by regression — there is no α, no β, no prior, no shrinkage operator. The doc is conflating two distinct families (Beta-conjugate Bernoulli vs. half-life regression) and attributing one to a paper that uses the other. The doc's actual recommended update rule is unattributed in the literature and is essentially the author's invention dressed up as "Settles-Meeder style."

---

## 4. "Pelánek 2017: Beta-Bernoulli + logistic features dominate at low sample sizes"

**Doc quote (line 39):** "Pelánek explicitly argues against assuming BKT is superior just because it is named 'Bayesian.'" / (line 213) "Both Pelánek (2017) and Skycak's own experience point to Beta-Bernoulli + logistic features as more robust at this scale."

**Verdict: PARTIAL → mostly UNSUPPORTED.**

**Source:** Pelánek 2017, §5.1 (pp. 16–17), §7.2 Hypothesis 1 (p. 33), Fig. 4.

**Justification:** The "BKT is not automatically superior" framing is real — Pelánek's whole point is that BKT and logistic models achieve "very similar predictive accuracy" and the choice depends on context. **But Pelánek nowhere claims "Beta-Bernoulli + logistic features" as a category, and nowhere endorses it at low sample sizes.** Pelánek's Hypothesis 1 (p. 33) states the opposite of the doc's gloss: *"Logistic models are better for modeling fluency and memory processes, while Bayesian knowledge tracing is better for understanding and sense making processes."* Pelánek treats BKT and logistic (PFA/AFM/Elo) as parallel families. "Beta-Bernoulli" specifically is not even named in his taxonomy (Fig. 4 lists BKT, logistic models, baselines, generalizations, black-box). The doc's load-bearing inference that Pelánek endorses the proposed stack is fabricated.

---

## 5. "BKT/DKT won't fit cleanly at ~200 atoms with sparse data"

**Doc quote (line 5, 63):** "full BKT and DKT are over-engineered for ~200 atoms with sparse per-learner data … Classical 4-parameter BKT … is famously identifiable only with hundreds of attempts per skill (Beck & Chang 2007; van de Sande 2013)."

**Verdict: PARTIAL.**

**Source:** Pelánek 2017 cites Beck & Chang 2007 (p. 35) and van de Sande 2013 in the BKT identifiability discussion, but neither is in the downloaded set; only the secondary reference in Pelánek is.

**Justification:** The general claim that classical BKT has identifiability problems is well-established and Pelánek 2017 confirms it as a known issue ("noise in the data, model identifiability issues, and local optima in parameter fitting"). However, the specific number "hundreds of attempts per skill" is not given in Pelánek's text — that number traces to Beck & Chang and van de Sande, which I do not have. The Delta-Drills-relevant phrasing ("~200 atoms with sparse data won't fit cleanly") is a reasonable inference but is the doc-author's, not Pelánek's. **Don't cite Pelánek for the specific number.**

---

## 6. "Khajah 2016: DKT does not outperform simpler models when properly tuned"

**Doc quote (line 43):** "Khajah, Lindsey & Mozer (2016) … Shows that a BKT with forgetting + per-student abilities + per-item difficulties matches DKT on standard benchmarks."

**Verdict: CONFIRMED.**

**Source:** Khajah et al. 2016, §3.3 Results (p. 6, Fig. 2), §4 Discussion (p. 7).

**Justification:** The paper's explicit headline (Fig. 2 + §4): "enhanced BKT appears to perform as well on average as DKT across the four data sets." On the four benchmarks (Assistments, Synthetic, Statics, Spanish), BKT+FSA (forgetting+skill-discovery+abilities) is within <0.01 AUC of DKT in three of four datasets and only 8.3% worse (in scaled AUC, ≈0.03 AUC absolute) on Assistments. The "matches DKT" wording is fair. Minor caveat: it's BKT+FSA (with forgetting AND skill discovery AND abilities) that matches DKT — not vanilla BKT. The doc's gloss is correct.

---

## 7. "Bijl 2025 'Probabilistic Decay Trees' — Beta-with-decay-on-a-graph formalization"

**Doc quote (line 45):** "Bijl (2025), 'Probabilistic Decay Trees: explainable knowledge tracing with Beta posteriors.'"

**Verdict: DISPUTED — the title is wrong.**

**Source:** Bijl 2025, title and abstract (pp. 1–2).

**Justification:** The paper is titled *"Tracking Student Skills Real-Time Through a Continuous-Variable Dynamic Bayesian Network"* and the method is called *Performance Distribution Tracing (PDT)* — not "Probabilistic Decay Trees." The doc invented a plausible-sounding name that matches the acronym PDT. The substantive claim is, however, mostly accurate: PDT uses Beta-distribution basis functions as conjugate priors (p. 6), maintains a continuous-variable Dynamic Bayesian Network over success-rate distributions, and includes explicit time decay (the "decay ratio" mechanism, "for time decay an exponential decay seems appropriate"). It does propagate across skills via a network ("links between skills"). So as a substantive description of a Beta-with-decay-on-a-graph approach, the gloss is correct; as a citation it is wrong about the title and would be unrecoverable from a search by that name. **Flag for correction before any user-facing writeup.**

---

## 8. "Nedungadi 2015 PC-BKT — adds a forgetting parameter to BKT"

**Doc quote (line 56):** "Multiplicative shrinkage toward prior (Nedungadi & Remya 2015 PC-BKT): α ← max(α_0, α·exp(−Δt/τ))."

**Verdict: DISPUTED — wrong formulation attributed.**

**Source:** Nedungadi & Remya 2015, §IV "Proposed Model with Decay" (p. 3, Algorithm 1).

**Justification:** PC-BKT does add an exponential decay forget function — that part is right ("`N(t) = N(t-30) · e^(-λ·(t/30))`" with λ=0.1, threshold 30 days). And the paper reports a major accuracy gain (Table 3: PC-BKT 80.3% → PC-BKT-with-decay 95.9%). **But the update rule does NOT use α/β.** BKT is a HMM with binary knowledge state, not a Beta posterior. There is no α, no β, no "shrinkage toward prior α_0" anywhere in PC-BKT. The decay is applied to a probability `P(L_t)`, not a Beta posterior. The doc has transposed Nedungadi's decay-on-probability formulation onto a Beta-Bernoulli skeleton that PC-BKT does not use. The cite is misleading.

---

## 9. "Qiu 2011 'Does Time Matter to BKT' — finding: time matters"

**Doc quote (line 57):** "Conditional forget rate (Qiu, Pardos & Heffernan EDM 2011): per-attempt forget probability P(F | Δt). More flexible but requires more data to fit."

**Verdict: PARTIAL.**

**Source:** Qiu et al. 2011, §2.2 KT-Forget design (pp. 3–4), §3 Results (Table I, p. 2 — KT residuals show large new-day overprediction).

**Justification:** The headline finding — yes, time matters; KT-Forget significantly outperforms standard KT — is real. The introduced parameter is `forget_n`, the probability of forgetting on a "new day" relative to the previous attempt. **But the doc's "P(F | Δt)" description misrepresents the model.** Qiu's KT-Forget uses a binary same-day vs. new-day discretization (p. 5: `forget_s = 0`, `forget_n` learned per-skill) — there is NO continuous Δt dependence. The "more flexible but requires more data" gloss is doc-author speculation, not a finding in the paper. The basic citation-as-evidence-of-forgetting is fair; the technical claim about parameterization is wrong.

---

## 10. "τ ≈ 14 days global half-life as v0 default"

**Doc quote (line 257):** "Beta-Bernoulli per atom with single global half-life τ ≈ 14 days."

**Verdict: UNSUPPORTED (doc-author choice presented without justification).**

**Source:** none cited.

**Justification:** Settles & Meeder do not pick a single fixed τ — the entire point of HLR is that h is *learned per item* from features (Appendix A.3, p. 1858, bounds: `ĥ ∈ [15 min, 9 months]`). Nedungadi uses 30 days as the forgetting onset threshold, not a half-life. Bijl 2025's exponential decay has no specific time constant prescribed. The "14 days" number is the doc author's stage-1 default with no source. This is harmless (engineering choice has to be made somewhere) but should not be presented as derived from the cited literature. **Confirm in the methodology that this is an arbitrary v0 default to be tuned.**

---

## 11. "Uniform prior α₀ = β₀ = 1 is standard practice"

**Doc quote (line 200):** "Prior α₀ = β₀ = 1 (uniform) unless graph-propagated prior is computed."

**Verdict: CANT_TELL_NO_SOURCE in the audited set; in general, this is a defensible Beta(1,1) Jeffreys-adjacent choice.**

**Source:** Not addressed in any of HLR, Pelánek 2017, Khajah, Bijl, Nedungadi, Qiu.

**Justification:** None of the seven audited primary sources prescribe Beta(1,1) for educational mastery posteriors (most don't use a Beta posterior at all). Beta(1,1) = uniform is a textbook default for conjugate Bayesian updating with no domain prior — defensible but not papered. Bijl 2025 uses basis-function priors more general than Beta(1,1) (he uses higher-order Beta basis functions, order n). Treat as a reasonable engineering default; do not cite literature for it.

---

## 12. "Posterior precision α+β ≥ 8 AND mean ≥ 0.7 promotion threshold"

**Doc quote (line 205, 262):** "Promotion = α+β ≥ 6 AND mean ≥ 0.7 on the demotion-relevant atoms" and (line 262) "α+β ≥ 6 AND mean ≥ 0.7". Note the doc gives both "8" (line 135) and "6" (line 205, 262) — internally inconsistent.

**Verdict: UNSUPPORTED + internally inconsistent.**

**Source:** none in the audited set.

**Justification:** No source in the audited set prescribes any specific (α+β) threshold. The closest analog is Cognitive Tutor's BKT mastery cutoff at P(L)=0.95, which is a probability threshold not a precision threshold. Math Academy's 80%-targeted quizzes are referenced (line 111) but that's an accuracy target, not a posterior-precision target. The values 6/8 and 0.7 are pure doc-author engineering choices. Worse, the doc gives both 6 and 8 in different sections — pick one and document why. **Don't claim this comes from any source.**

---

## 13. "Yudelson & Pavlik 2013: monotonic mastery curves are anti-pattern"

**Doc quote (line 133):** "Forcing monotonicity is a known anti-pattern (Yudelson & Pavlik 2013)."

**Verdict: CANT_TELL_NO_SOURCE.**

**Source:** Yudelson-Pavlik 2013 is not in the downloaded paper set.

**Justification:** Cannot audit without the primary source. Pelánek 2017 cites Yudelson 2013 (p. 35 ref list) for individualization in BKT, not specifically for the "monotonicity anti-pattern" claim. The Khajah paper covers forgetting (which permits regression) but does not cite Yudelson on monotonicity. **Either fetch the Yudelson-Pavlik 2013 paper before relying on this claim, or downgrade it to "monotonic mastery is widely critiqued in the BKT-with-forgetting literature" without the specific cite.**

---

## 14. "Decaying α/β already encodes EWMA (so EWMA on top is redundant)"

**Doc quote (line 13, 61):** "EWMA on top of a Beta posterior is redundant for the static-skill case. The posterior mean α/(α+β) **is** an exponentially weighted average if you decay α and β with a fixed half-life — they encode the same information."

**Verdict: CONFIRMED (mathematically) — but with a precise condition the doc handles correctly.**

**Source:** Standard result; not in any specific audited paper but mathematically derivable.

**Justification:** If at each attempt α ← α·d + y and β ← β·d + (1−y) with d ∈ (0,1), then the posterior mean α/(α+β) is exactly the exponentially weighted average of the outcomes with smoothing factor (1−d). So the math checks out for the *attempt-indexed* decay. The doc's framing is correct; calling EWMA-on-top "redundant" in the static case is mathematically right. The doc's caveat (line 13) — that two-timescale signals (fast vs. slow) justify a separate EWMA, as in Recent-PFA (Galyardt & Goldin 2014) — is also correct, though Galyardt-Goldin is not in the audited set so I can't verify the R-PFA mechanism claim. **The mathematical equivalence claim is the one piece of the doc's "synthesis" that survives audit cleanly.**

---

## 15. "Pelánek 2016 — Elo for adaptive education"

**Doc quote (line 41):** "Pelánek (2016) … Argues that Elo (and its extensions like Glicko-2 which include a rating-deviation that grows with time) is the simplest robust solution at small sample sizes."

**Verdict: PARTIAL — the Elo recommendation is real; the Glicko-2 framing is doc-author embellishment.**

**Source:** Pelánek 2016 Elo paper, Abstract and §2.1 (pp. 1–3).

**Justification:** Pelánek 2016's abstract says Elo is "simple, robust, and effective and thus suitable for use in the development of adaptive educational systems." That core claim is well-supported. The paper provides update equations (§2.2, eq. on p. 3) and discusses extensions. However, the doc's gloss about "Glicko-2's rating-deviation that grows with time being operationally identical to a decaying Beta" with the explicit `RD ≈ √(α+β)⁻¹` mapping is doc-author analogy — Pelánek 2016 mentions Glicko/Glicko-2 in §2 only briefly as an extension, and does not develop the Beta equivalence. So the headline "Pelánek 2016 endorses Elo for small sample sizes" is true; the bolted-on Beta-equivalence claim is doc-author and may not survive scrutiny.

---

## Summary — verdict tally

| Verdict | Count | Items |
|---|---|---|
| CONFIRMED | 3 | 2, 6, 14 |
| PARTIAL | 4 | 4, 5, 9, 15 |
| DISPUTED | 3 | 1, 7, 8 |
| UNSUPPORTED | 3 | 3, 10, 12 |
| CANT_TELL_NO_SOURCE | 2 | 11, 13 |

Out of 15 audited claims, only 3 survive cleanly. 6 are partial or wrong-but-recoverable. 6 are unsupported, disputed, or unverifiable from the cited set.

---

## Load-Bearing Concerns

These directly affect whether the recommended **Beta-Bernoulli with multiplicative half-life decay** stack is defensible:

1. **The "Settles-Meeder formulation" is a doc-author invention (Claims 1, 3).** The actual published HLR uses regression on engineered features, not Beta-Bernoulli posteriors. The doc's update rule (`α ← α·exp(−Δt/τ) + α₀·(1−exp(−Δt/τ))`) does not appear in any audited paper. This does not mean the rule is wrong — it is a mathematically reasonable Bayesian construction — but the user should not believe it has the empirical backing of HLR's 12.9M Duolingo traces. **It is an unvalidated hybrid.**

2. **Pelánek 2017 does not endorse the "Beta-Bernoulli + logistic at low sample sizes" thesis (Claim 4).** This is the single most load-bearing inference in the doc's synthesis. Pelánek's own Hypothesis 1 is the opposite framing (BKT for "understanding/sense-making," logistic for "fluency/memory"). The doc is putting words in Pelánek's mouth to justify a stack he does not discuss.

3. **PC-BKT and KT-Forget formulations are misrepresented (Claims 8, 9).** Both papers contribute "exponential decay matters for forgetting" as a finding — that's real and useful — but neither uses a Beta-Bernoulli posterior, and neither uses the continuous-Δt parameterization the doc attributes to them. If you cite these, cite them for the existence-of-forgetting finding, not for the specific update rule.

4. **The Bijl 2025 citation has the wrong title (Claim 7).** Substantively the description is mostly fine (PDT does use Beta basis functions and exponential decay) but "Probabilistic Decay Trees" is not the paper's name. Any reader trying to find this paper from the doc's citation will fail.

5. **Numeric thresholds (τ=14d, α+β≥6–8, mean≥0.7) are doc-author defaults, not derived (Claims 10, 12).** Worse, the doc gives conflicting values for the precision threshold (6 in one place, 8 in another). These need to be flagged as "v0 engineering choices, tune empirically" rather than presented in citation-flavored prose.

6. **The Yudelson & Pavlik anti-monotonicity claim is unaudited (Claim 13).** Cited but the source is not in the downloaded set. The general point ("forgetting implies regression must be allowed") is uncontroversial and well-supported by Khajah and Nedungadi, but the specific Yudelson-Pavlik cite is unverifiable here.

**Net assessment:** The doc's *direction* is broadly reasonable — Beta-Bernoulli with explicit time decay is a sensible primitive for the Delta-Drills regime, and the cited papers do collectively support "forgetting matters" and "DKT is over-engineered at this scale." But the doc systematically over-claims that its specific stack is what the literature recommends. It is the doc-author's synthesis dressed up as a literature consensus. Treat the stack as a defensible v0 prototype to instrument and validate empirically, not as a published best-practice template.
