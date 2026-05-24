# VERIFICATION B — PFA / LKT / Multi-Tag Credit Assignment Claims

**Doc audited:** `compass_artifact_wf-2f4da85b-a0ab-425d-9877-f25787abbdea_text_markdown.md`
(Cluster B: Q2 — multi-tag credit assignment claims)

**Sources consulted:**
- Pavlik, Cen & Koedinger 2009 (PFA foundational, AIED)
- Galyardt & Goldin 2014 (R-PFA, EDM)
- Maier, Baker & Stalzer 2021 (PFA Challenges, ICCE)
- Zhang, Baker, Srivastava et al. 2025 (Carelessness Detection using PFA, arXiv 2503.04737)
- Park, Cornillie, van der Maas & Van Den Noortgate 2019 (M-ERS, Frontiers in Psychology)

**Cluster A precedent:** Sister verification found doc misrepresents Settles-Meeder
(operational gain numbers), Pelánek 2017 (BKT-vs-logistic framing), and fabricated a
title for Bijl 2025. Skepticism dial is high.

---

## Claim 1 — Pavlik 2009 PFA is the foundational paper for logistic multi-KC credit assignment

**Quoted claim (doc):** "Pavlik, Cen & Koedinger (2009), 'Performance Factors Analysis – A
New Alternative to Knowledge Tracing,' AIED. The foundational multi-tag credit model."

**Verdict:** CONFIRMED.

**Source:** Pavlik, Cen & Koedinger 2009, pp. 1–4.

**Justification:** Title, authors, venue (AIED), and substantive claim all match. The paper
explicitly motivates PFA as an LFA-derivative that can be applied "in a compensatory
fashion for observations requiring multiple KCs by summing the βs and γ and ρ frequency
components for all j KCs needed" (p. 4). It is unambiguously the foundational paper
for the multi-KC logistic credit-assignment family.

---

## Claim 2 — PFA formula form: m(KC) = β_KC + γ·successes + ρ·failures (logistic)

**Quoted claim (doc):** "Logistic with per-KC β (easiness), per-KC γ (learning rate from
successes), per-KC ρ (learning rate from failures). For an item tagged with KCs {A,B,C}:
logit P(correct) = Σ_k (β_k + γ_k·s_k + ρ_k·f_k)..."

**Verdict:** CONFIRMED.

**Source:** Pavlik 2009, Equation 3, p. 4: `m(i, j ∈ KCs, s, f) = Σ_{j∈KCs}(β_j + γ_j·s_{i,j} + ρ_j·f_{i,j})`
followed by p(m) = 1/(1+e^{-m}) (Equation 2, p. 3). Re-confirmed in Maier 2021 p. 2:
`m(i, j ∈ KCs, s, f) = Σ_{j∈KCs}(β_j + γ_j·s_{i,j} + ρ_j·f_{i,j}); p(m) = 1/(1+e^{-m})`.

**Justification:** Doc's notation matches the primary source exactly. Note: doc's TL;DR
elsewhere says "(β_KC + γ·successes + ρ·failures)" omitting the per-KC subscripts on γ
and ρ; the full Q2 statement (quoted above) correctly subscripts them. The α_i student
ability term Pavlik shows in LFA Equation 1 is intentionally dropped from PFA Equation 3
("α has been removed from the model since it is not usually estimated ahead of time in
adaptive situations," p. 4) — doc gets this right by not including α.

---

## Claim 3 — "Multi-tag credit on composites: PFA/LKT only when per-atom N ≥ 30"

**Quoted claim (doc):** "PFA / LKT formulation for credit, but only when sample sizes per
atom are ≥30; below that, use tag-confidence-weighted Beta updates as a simpler fallback."

**Verdict:** UNSUPPORTED (doc-author choice presented as if it were a literature consensus).

**Source:** Searched Pavlik 2009 — no N threshold cited. Maier 2021 §4.1 ("Insufficient
Number of Practices," pp. 3–4) is the closest available evidence and reports very
different numbers: "major improvement from 2 to 3 practices (AUC increased from ~0.753
to ~0.772), with continued improvement up to practice 6 (AUC ~0.78)" and from the
test-side experiment "most substantially from the second to fifth practices … followed
by a slower increase up to around practice 12, after which the performance flattens
out." Galyardt & Goldin 2014 do not report any N threshold.

**Justification:** The number 30 does not appear in any of the PFA papers in the corpus.
Maier 2021's own data suggests the practical PFA convergence regime is more like 5–12
practices per skill, an order of magnitude smaller than 30. The doc is either pulling
from an unstated source, applying a generic statistics rule of thumb ("n=30 for the
CLT"), or making it up. Treat as doc-author heuristic, not literature-grounded.

---

## Claim 4 — "Below that, tag-confidence-weighted Beta updates" as a hybrid fallback

**Quoted claim (doc):** "Tag-confidence-weighted Beta-Bernoulli credit. The simplest
workable alternative: per atom, increment α by w_A·confidence·y rather than 1·y."

**Verdict:** UNSUPPORTED as a published-literature pattern.

**Source:** Not found in any of the five PFA papers. None of Pavlik 2009, Galyardt 2014,
Maier 2021, Zhang 2025, or Park 2019 describes a "tag-confidence-weighted Beta" hybrid.

**Justification:** This is a reasonable engineering construction that resembles Maier's
"merged-rare" pooling solution conceptually (Maier 2021 §4.3, pp. 4–5: a separate
catch-all set of parameters β_d, γ_d, ρ_d for rare skills), but Maier's mechanism is
parameter pooling inside PFA, not a Beta-Bernoulli fallback. Doc-author construct;
defensible as a v0 default, but not a citable literature pattern. Doc should label it
as such rather than implying it is in any paper.

---

## Claim 5 — "Cap effective tag count at 6 (≥6 KCs → per-KC update becomes noise)"

**Quoted claim (doc):** "Cap effective tag count at 6 — for composites with >6 atom tags,
treat the signal as area-level only." And separately: "the implicit consensus from the PFA
literature is that items with ≥6 tags become essentially uninformative for any single tag's
update."

**Verdict:** UNSUPPORTED.

**Source:** Searched Pavlik 2009, Galyardt 2014, Maier 2021. No "6" threshold, no
"per-KC update becomes noise" threshold, no analogous claim found anywhere in these
papers. Maier 2021 explicitly notes "there has been relatively little study of what
factors impact PFA's behavior in real-world learning settings" (p. 2), and their own
empirical analysis caps at multi-skill vs single-skill, never breaking out by tag-count
buckets. The doc itself elsewhere admits "No published clean threshold" (under Synthesis
on Q2, "On how many tags is too many"), which directly contradicts its own "implicit
consensus from the PFA literature" framing.

**Justification:** Doc-author heuristic. The doc internally inconsistent — one paragraph
says "no published clean threshold," another says "implicit consensus … ≥6 tags." Both
cannot be true. The "6" cap is reasonable engineering judgment but is not in the
literature. Cluster-A-style overclaim: dressing up doc-author intuition as
literature consensus.

---

## Claim 6 — Recent-PFA (Galyardt & Goldin 2014) as "fast current-form" alongside slow long-run

**Quoted claim (doc):** "EWMA … Value-adding only if you want a fast 'current form' signal
*in addition to* the long-run posterior — Recent-PFA (Galyardt & Goldin 2014) does this
explicitly by adding a recency-weighted-success-rate feature alongside cumulative counts."

**Verdict:** CONFIRMED.

**Source:** Galyardt & Goldin 2014, p. 1 (Eq. 2) and §2 (Methods).

**Justification:** R-PFA's Equation 2 is `logit(p_{ijt}) = θ_i + β_j + γ_j·T_{ijt} +
δ_j·R_{ijt}` where T_{ijt} is total opportunities (cumulative count) and R_{ijt} is
"recency-weighted proportion of past successes" with decay factor b. Table 1 (Terms in
predictive model variants) makes the dual-signal point explicitly: R-PFA carries both
"Totals" (T) and "Weighted Proportion" (R) features simultaneously, which is exactly
the "fast current-form alongside slow long-run" framing doc describes. The Results
section confirms "For any fixed decay parameter b, R-PFA is better than R-only. The
total number of practice opportunities is still informative above and beyond the recent
history" (p. 2) — i.e., both signals contribute.

---

## Claim 7 — Recent-PFA framing for the static-skill / EWMA-redundancy argument

**Quoted claim (doc):** "Decaying α/β already encodes EWMA. Only add separate EWMA if
explicitly want fast 'current form' alongside slow 'long run' (Recent-PFA, Galyardt &
Goldin 2014). For static-skill case, don't double-count."

**Verdict:** PARTIAL.

**Source:** Galyardt & Goldin 2014. The Recent-PFA citation is accurate for "fast current
form alongside slow long run." The "static-skill / don't double-count" framing is not
in Galyardt 2014 — that paper is about a "moment of learning" model (a non-static-skill
construct, see §2: "If a student has already experienced a moment of learning then
recent performance is likely to consist primarily of successful attempts").

**Justification:** The Recent-PFA citation supports the two-timescale framing.
However, the doc's "for static-skill case, don't double-count" is doc-author synthesis,
not a Galyardt-2014 claim. Galyardt 2014 doesn't argue against EWMA-over-Beta; it
proposes adding a recency feature to PFA's logistic model. The doc's logic
(decaying-α/β-already-IS-EWMA-of-attempts) is its own argument, defensible on its own
terms but not attributable to Galyardt.

---

## Claim 8 — Maier-Baker-Stalzer 2021 identifies challenges (overfitting, identifiability, etc.)

**Quoted claim (doc):** "Maier, Baker & Stalzer (2021), 'Improving PFA for rare skills.'
Handles the practical problem of rare skills in multi-tagged items by pooling them into
a 'catch-all' skill."

**Verdict:** PARTIAL.

**Source:** Maier, Baker & Stalzer 2021. Actual title: "Challenges to Applying
Performance Factor Analysis to Existing Learning Systems" (ICCE 2021), not "Improving
PFA for rare skills."

**Justification:** The doc misquotes the title. The substantive content claim — pooling
rare skills into a catch-all — is correct (Maier §4.3, pp. 4–5: "merged-rare model"
with a default set (β_d, γ_d, ρ_d) "used for rare skills"). But the paper covers four
challenges, not just rare skills: (1) insufficient practices, (2) degenerate parameters,
(3) rare vs common skills, (4) compensatory vs conjunctive. The doc's caption
selectively pulls out one of four contributions and gives it as the title. This is a
Cluster-A-pattern overclaim: not a fabrication, but a misattribution of paper scope.

**Note:** Doc's broader implication (PFA has rare-skill/degeneracy problems documented
in Maier 2021) is correct. Maier §4.2 documents three degeneracy types (γ<0; γ<ρ; γ=ρ=0)
and observed 9% type-1 + 11% type-3 degeneracy in their PFA baseline (p. 4) — these
support the doc's broader point that PFA at low N can degenerate.

---

## Claim 9 — Zhang & Baker 2025 carelessness-PFA extension

**Quoted claim (doc):** "Zhang, Baker, Srivastava et al. (2025), 'Carelessness Detection
using Performance Factor Analysis,' arXiv 2503.04737. Modernized PFA implementation,
with attention to the multi-tag case. Good reference implementation in code."

**Verdict:** PARTIAL.

**Source:** Zhang et al. 2025, arXiv:2503.04737. Title is accurate ("Carelessness Detection
using Performance Factor Analysis: A New Operationalization with Unexpectedly Different
Relationship to Learning"). Authors and arXiv ID match.

**Justification:** The doc's characterization is misleading. Zhang 2025 is not a
"modernized PFA implementation" — it is a carelessness-detection model called BKFC
(Beyond-Knowledge Feature Carelessness) that *uses* PFA as the knowledge-estimation
backbone because "PFA can be used in cases where items are tagged with multiple skills"
(p. 3, §III), unlike BKT's per-skill conditional probability approach. The paper's
contribution is operationalizing carelessness as the residual between PFA-predicted
knowledge and observed errors, using behavioral features. It is not a PFA improvement
paper. It cites PFA via the reference numbered [14] (citation list referencing Pavlik
2009 implicitly — I cannot confirm without seeing the references page).

Also, the application is to *Decimal Point*, "single-player web game … 10 skills
in total, with 5 skills in the problem-solving steps and 5 skills in the
self-explanation steps" (p. 4) — and crucially, "Decimal Point has only one skill per
item" (p. 4, §IV-B). So this paper's actual experimental setting is NOT multi-tag. The
doc's "good reference for the multi-tag case" is unsupported — Zhang 2025 explicitly
motivates choosing PFA for multi-skill capability but tests on a single-skill dataset.

---

## Claim 10 — Park & Cornillie 2019 multidim-IRT for real-time multidimensional ability monitoring

**Quoted claim (doc):** "Reckase (2009) Multidimensional Item Response Theory; and
Doebler/Pelánek/Wauters (2018), 'M-ERS: multidimensional Elo for adaptive learning'
(Frontiers in Psychology)."

**Verdict:** DISPUTED (attribution error).

**Source:** Park, Cornillie, van der Maas & Van Den Noortgate 2019 (NOT 2018), "A
Multidimensional IRT Approach for Dynamically Monitoring Ability Growth in Computerized
Practice Environments," Frontiers in Psychology 10:620, March 2019.

**Justification:** The paper the doc actually means is in the corpus and authored by
Park, Cornillie, van der Maas & Van Den Noortgate (2019), not Doebler/Pelánek/Wauters
(2018). The Park et al. 2019 paper does introduce M-ERS (Multidimensional Extension of
the Elo Rating System) — see p. 2 ("we propose to address these issues by using a
multidimensional IRT (MIRT) model … resulting in a multidimensional extension of the
ERS ('M-ERS')") and the dedicated §"Multidimensional Extension of the ERS (M-ERS)"
on p. 3. The doc's cited authors (Doebler/Pelánek/Wauters 2018) is wrong; Park 2019
mentions Doebler 2015 and Wauters 2012 as prior work but they are not the M-ERS
authors. This is the same kind of attribution error Cluster A flagged for Bijl 2025.

The substantive claim (real-time multidimensional ability monitoring via Elo extension)
is supported by the Park 2019 paper: "The basic idea is that instead of updating a
single ability parameter from the Rasch model, our method allows a simultaneous update
of multiple ability parameters" (abstract, p. 1).

---

## Claim 11 — LKT (Logistic Knowledge Tracing) paired with PFA; where defined

**Quoted claim (doc):** Multiple references to "PFA / LKT" as if they are interchangeable;
LKT cited once as "Pavlik et al. 2021" in the further-reading list.

**Verdict:** CANT_TELL (no LKT primary source in the corpus).

**Source:** Searched the mastery-estimation/ papers folder — only Pavlik 2009 is present;
no Pavlik 2021 paper. The actual LKT paper (Pavlik, Eglington & Harrell-Williams 2021,
"Logistic Knowledge Tracing: A constrained framework for learner modeling," IEEE TLT)
is not in the provided corpus.

**Justification:** The doc treats PFA and LKT as nearly interchangeable ("PFA / LKT"
appears five times). LKT is a real published framework — Pavlik et al. 2021 generalizes
PFA into a flexible logistic feature-engineering framework — but I cannot verify the
doc's specific claims about LKT from the provided sources. The phrase "PFA / LKT" is
not wrong per se (LKT subsumes PFA), but the doc never distinguishes when LKT-specific
features beyond PFA matter, and doesn't cite an LKT primary source in-text. This is a
shallow citation, not a fabrication.

---

## Claim 12 — "Equal-credit composite→atom propagation is catastrophic. Severely biases low-tag-count atoms."

**Quoted claim (doc):** "Equal credit. Severely biases low-tag-count atoms and is
information-destructive. *Avoid.*" And separately: "If a composite-tier failure updates
per-atom procedural priors via equal-credit, you will systematically underestimate the
atoms that happen to co-occur with hard atoms."

**Verdict:** UNSUPPORTED as cited; PLAUSIBLE as reasoning.

**Source:** Pavlik 2009 does not establish "equal credit is catastrophic." It motivates
PFA as a compensatory model that "sums the contributions from all KCs needed in a
performance" (p. 3) — i.e., PFA is itself an equal-credit-by-default model. Maier 2021
§4.3 (Compensatory vs Conjunctive, pp. 5) does compare compensatory/conjunctive/even
(averaging) credit schemes empirically and finds compensatory AUC 0.7818, conjunctive
AUC 0.6725, and even-skill (averaging = equal credit) AUC 0.7849 — i.e., **even-skill
(equal credit) was the BEST of the three on Maier's data**. This directly contradicts
the doc's "equal credit is catastrophic and information-destructive" claim.

**Justification:** Doc-author intuition. The narrative argument ("low-tag-count atoms
co-occur with hard atoms and get biased") is a sensible identifiability-collinearity
concern (Maier 2021 p. 4 notes "collinearity between skills could produce degenerate
parameters"), but the strong claim "equal credit is catastrophic" is contradicted by
Maier 2021's empirical comparison where even-skill credit slightly outperformed
compensatory PFA. The doc overstates the case. The defensible claim is "equal credit
can be problematic when tag counts and difficulty co-vary," not "catastrophic."

---

## Summary

**Verdict tally (12 claims):**
- CONFIRMED: 3 (claims 1, 2, 6)
- PARTIAL: 3 (claims 7, 8, 9)
- DISPUTED: 1 (claim 10)
- UNSUPPORTED: 4 (claims 3, 4, 5, 12)
- CANT_TELL: 1 (claim 11)

**Substantive load-bearing concerns:**

1. **The "N ≥ 30" PFA threshold (Claim 3) and "cap tags at 6" rule (Claim 5) are
   doc-author heuristics presented as literature consensus.** Maier 2021 actually
   suggests PFA convergence in the 5–12-practice range, an order of magnitude smaller
   than 30. The doc internally contradicts itself on the tag-cap claim ("no published
   clean threshold" vs. "implicit consensus … ≥6 tags"). If the user implements
   "switch from Beta to PFA at N=30," they will be applying an unfounded threshold;
   the literature supports a much lower switchover point. Recommend the user re-derive
   the threshold from their own data rather than trust the doc's number.

2. **"Equal credit is catastrophic" (Claim 12) is contradicted by Maier 2021's own
   experiment.** Maier's even-skill (=equal credit averaging) model gave the highest
   AUC of the three compensatory/conjunctive/even variants tested (0.7849 vs 0.7818
   compensatory vs 0.6725 conjunctive). The doc lists "equal credit" as the
   worst-of-five option to "Avoid"; the only paper in the corpus that empirically
   tested it ranked it best. This is a meaningful misdirection — the user's "avoid
   equal credit at all costs" stance is not literature-supported.

3. **Park 2019 attribution error (Claim 10)** is the same class of mistake Cluster A
   caught for Bijl 2025: doc invents a citation (Doebler/Pelánek/Wauters 2018) for
   work that was actually done by Park, Cornillie, van der Maas & Van Den Noortgate
   (2019). The Park 2019 paper IS in the corpus and IS the M-ERS source — the doc
   never references it by its actual author/year.

4. **Maier 2021 mis-titled (Claim 8)** as "Improving PFA for rare skills" when the
   actual title is "Challenges to Applying Performance Factor Analysis to Existing
   Learning Systems." Not a fabrication, but a selective re-framing that hides the
   paper's four-challenge scope (insufficient practice, degeneracy, rare skills,
   compensatory/conjunctive) behind one cherry-picked contribution.

5. **Zhang 2025 mischaracterization (Claim 9):** the paper is not a "modernized PFA
   implementation" — it is a carelessness-detection model (BKFC) that uses PFA as
   one of three knowledge-estimation backbones. The doc cites it as a multi-tag PFA
   reference, but the experimental setting in Zhang 2025 is explicitly single-skill
   ("Decimal Point has only one skill per item," p. 4).

6. **The doc's core Q2 recommendation — "PFA/LKT for multi-tag credit when N≥30, else
   tag-confidence-weighted Beta, cap at 6 tags" — is built on three sub-claims (3, 4,
   5) all of which are doc-author engineering judgments without literature backing.**
   The recommendation may still be sound, but it is not "what the literature says."
   The user should treat it as one defensible heuristic among many, not as a
   research-grounded prescription.

**Consistent with Cluster A's findings,** this doc exhibits a pattern of:
(a) misquoting paper titles to fit a narrative,
(b) presenting doc-author heuristics as literature consensus,
(c) attribution errors that invent authorship for real findings, and
(d) internal contradictions between sections (claim 5 contradicts the doc's own
"no published clean threshold" line in the same Q2 synthesis).

The Q2 chapter is more reliably ground-truthed at the level of formulas and
foundational citations (claims 1, 2, 6 all CONFIRMED) than at the level of
operational thresholds and engineering recommendations (claims 3, 5, 12 all
UNSUPPORTED or DISPUTED). Treat the math as solid; treat the numbers as opinion.
