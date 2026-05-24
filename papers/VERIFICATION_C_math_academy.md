# Verification — Cluster C: Math Academy / FIRe / Demotion Claims

**Date:** 2026-05-24
**Audited doc:** `compass_artifact_wf-2f4da85b-a0ab-425d-9877-f25787abbdea_text_markdown.md`
**Reviewer:** Claude Opus 4.7 (1M context) — adversarial audit on user request

Sources used (all in `papers/mastery-estimation/`):
- **HOAW** = `2025_mathacademy_how-our-ai-works.html`
- **FIRe** = `2023_skycak_fire-spaced-repetition.html`
- **C&T** = `2024_stokke-smith_chalk-and-talk-ep-42-transcript.html`
- **HN** = `2024_skycak_hn-thread-math-academy.html`
- **Oz** = `2025_oznova_balanced-review-of-math-academy.html`
- **Hecker** = `2025_hecker_math-academy-part-7-technology-brief.html` (paraphrase of *The Math Academy Way* Part V)
- **MAW** = `2024_skycak-roberts_the-math-academy-way.pdf` (508 pp) — sampled selectively
- **Matuschak** = `2024_matuschak_notes-math-academy.html`

---

## 1. "Math Academy explicitly tried BKT first and moved to a 'physical' fractional-repetition flow (FIRe)"

**Doc says (line 6, 248):** "Justin Skycak has publicly stated he tried BKT first and moved to a 'physical' fractional-repetition flow (FIRe)" / "Skycak explicitly tried BKT first and moved away from it because 'complexity explodes.'"

**Verdict:** CONFIRMED.

**Source:** FIRe, "Did you look into Bayesian Knowledge Tracing?" Q&A section.

**Justification:** Skycak writes: *"my first approach to our diagnostic algorithm was along those lines [BKT] … After many cycles of reigning in the complexity I ended up moving towards less of a probabilistic approach and more of a 'physical' approach where I envision every answer as a 'fractional repetition' physically flowing through the knowledge graph."* Note: BKT was tried on **the diagnostic algorithm**, not necessarily on the whole mastery model. The doc slightly compresses this nuance.

---

## 2. Quote audit: "complexity explodes," "physical" model

**Doc says (line 248):** Skycak abandoned BKT because "complexity explodes." Says the current model is "physical."

**Verdict:** CONFIRMED (quotes verbatim).

**Doc:** `complexity explodes`
**FIRe:** *"complexity explodes and you have to find ways to reign it in"* — exact match.

**Doc:** `"physical" approach`
**FIRe:** *"more of a 'physical' approach where I envision every answer as a 'fractional repetition' physically flowing through the knowledge graph"* — exact match.

---

## 3. "FIRe uses a continuous `repNum` counter" with formula `repNum → max(0, repNum + speed · decay^failed · rawDelta)`

**Doc says (line 95, 248):** Cites formula verbatim.

**Verdict:** CONFIRMED.

**Source:** FIRe, "High-Level Structure of Spaced Repetition Model."

**Side-by-side:**
- **Doc:** `repNum → max(0, repNum + speed · decay^failed · rawDelta)`
- **FIRe:** `repNum → max(0, repNum + speed · decay^failed · rawDelta)` — exact match in LaTeX.

Note: Hecker (paraphrasing MAW chapter 26, 2025) renders this with `netWork` instead of `rawDelta`. So the variable name has evolved or the book and blog use different labels; both are Skycak-authored. The doc's quote matches the 2023 FIRe blog post exactly.

---

## 4. "Multiplicative `speed`·`decay` updates"

**Verdict:** CONFIRMED.

**Source:** FIRe formula and bulleted definitions: *"speed = the learning speed for the student on this particular topic … governs how quickly the student moves forwards or backwards"*, *"decay = the speed at which the student moves backwards in the spaced repetition process, relative to their forwards speed, if they fail a repetition"*. Both factors appear as multiplicative coefficients of `rawDelta` in the update.

---

## 5. "Exponentially-decaying `memory` term"

**Verdict:** CONFIRMED.

**Source:** FIRe second formula: `memory → max(0, memory + rawDelta)(0.5)^(days/interval)`.

**Justification:** The `(0.5)^(days/interval)` factor is literal exponential half-life decay; Skycak explicitly defines `interval` as the ideal spacing between repetitions and says memory decays to half its value at one interval. Doc's characterization is accurate.

---

## 6. "Demotion fires on 2 failed lesson attempts without forward progress"

**Doc says (line 6, 19, 249):** "demotion fires on two failed lesson attempts at the same knowledge point or on any quiz miss."

**Verdict:** PARTIAL — the "two attempts without forward progress" rule is verbatim; the "**at the same knowledge point**" qualifier is overclaimed.

**Source:** FIRe Q&A section and HOAW ("Adapting the Pace of Learning") and C&T transcript.

**Side-by-side:**
- **Doc:** "two failed lesson attempts at the same knowledge point"
- **FIRe:** *"if a student 'plateaus' on the lesson for B (i.e., they fail the lesson twice in a row without getting any further the second time), then we trigger remedial learning tasks on the prerequisites where the student got stuck"*
- **HOAW:** *"if a student gets halted again on the re-attempt without making any additional forward progress, then we slow down further and give them remedial reviews"*
- **C&T:** *"each topic has a lesson, broken down into stages, which we call knowledge points … We will halt lessons if a student stumbles on any particular knowledge point a little bit too much"*

**Justification:** The two-failed-lesson rule is at the **lesson** level (not at the knowledge-point level). Knowledge points are the *intra-lesson* scaffolding units (3-4 per topic per C&T), and a within-lesson halt fires when a student stumbles on a knowledge point. The doc conflates "lesson failure → plateau" with "knowledge point failure → halt" into a single rule. These are two different mechanisms in the published description.

---

## 7. "OR any quiz miss" triggers remediation

**Verdict:** CONFIRMED.

**Source:** HOAW, "Adapting the Pace of Learning" section: *"There is also a remediation process for quizzes: whenever a student misses a question on a quiz, we slow down and immediately follow up with a remedial review on the corresponding topic."* Exact match in substance.

---

## 8. "Not on a posterior threshold" (event-triggered, not threshold-triggered)

**Verdict:** CONFIRMED.

**Source:** Inferred from absence — none of HOAW, FIRe, C&T, or Hecker describes a posterior-probability threshold for demotion. All published descriptions are framed in terms of events (halt, halt-again, quiz miss). The Skycak HN comment (file content) does not contradict this. Doc's positive claim that there is no posterior threshold is well-supported by the published material.

---

## 9. "Diagnostic = info-gain greedy on compressed graph cover"

**Doc says (line 6, 145, 250):** "diagnostic exams are information-gain greedy on a compressed graph cover."

**Verdict:** CONFIRMED (quote verbatim).

**Source:** HOAW, "The Diagnostic Algorithm."

**Side-by-side:**
- **Doc:** "compresses the knowledge graph into the smallest number of topics that fully 'covers' a course and its foundations at a sufficient level of granularity. Then, it repeatedly selects the topic whose assessment provides the most information about the student's knowledge profile."
- **HOAW:** *"The algorithm first compresses the knowledge graph into the smallest number of topics that fully 'covers' a course and its foundations at a sufficient level of granularity. Then, it repeatedly selects the topic whose assessment provides the most information about the student's knowledge profile."* — verbatim match (the doc lifts this paragraph entirely from HOAW; quotation marks correctly delimit it).

---

## 10. Quote audit: "knowledge frontier resolution" termination

**Doc says (line 6):** "terminate at 'knowledge frontier resolution.'"

**Verdict:** PARTIAL / OVERCLAIMED — "knowledge frontier" is verbatim Math Academy terminology, but "knowledge frontier resolution" as a noun-phrase **termination criterion** is not in the cited sources.

**Side-by-side:**
- **Doc:** terminate at "knowledge frontier resolution"
- **HOAW:** uses "knowledge frontier" repeatedly (*"the boundary between what the student knows and does not know"*) but never the phrase "knowledge frontier resolution." HOAW does not give an explicit termination criterion.
- **Hecker (paraphrasing MAW ch. 24):** says the exam gets "acceptable results … with relatively few questions (20-60)" — i.e., the practical termination is a question budget, not an information-theoretic stopping rule.

**Justification:** Doc puts "knowledge frontier resolution" in quotes, implying a verbatim source. It is not verbatim. "Knowledge frontier" is real; "resolution" is the doc author's gloss. This is a minor but real misquote.

---

## 11. Oz Nova quote: "subjecting someone to literal years of unnecessary remedial pre-requisites is too great a punishment for a second slip up"

**Verdict:** CONFIRMED (verbatim).

**Side-by-side:**
- **Doc:** *"subjecting someone to literal years of unnecessary remedial 'pre-requisites' is too great a punishment for a second slip up"*
- **Oz:** *"Allowing a second attempt at a failed diagnostic question is a good idea. But subjecting someone to literal years of unnecessary remedial 'pre-requisites' is too great a punishment for a second slip up."*

Exact match. (Hyphenation and quotes around "pre-requisites" preserved.)

---

## 12. "Math Academy plateau rule" — is this what they call it?

**Doc says (line 6, 95, 109, 203):** Uses "plateau rule" as if it is Math Academy's name.

**Verdict:** PARTIAL — Skycak uses the verb "plateaus" (in quotes), but "plateau rule" as a noun phrase appears to be the doc author's coinage, not MA's official term.

**Source:** FIRe Q&A: *"if a student 'plateaus' on the lesson for B (i.e., they fail the lesson twice in a row …)"*. The word "plateaus" with scare quotes is Skycak's; "plateau rule" is the doc reifying a usage into a label. Reasonable shorthand, but readers should not search MA materials for "the plateau rule."

---

## 13. "Conditionally completed" atoms — MA terminology

**Verdict:** CONFIRMED.

**Source:** HOAW, "The Diagnostic Algorithm": *"the system will consider those topics 'conditionally completed.' The student will initially be given tasks under the assumption that they know those topics, but if the student struggles, then the system will immediately begin 'falling backwards' along the appropriate learning paths."* — verbatim in HOAW with scare quotes (Math Academy's term).

---

## 14. "Knowledge point" granularity — MA's term

**Verdict:** CONFIRMED.

**Source:** C&T, Alex Smith's section: *"each topic has a lesson, broken down into stages, which we call knowledge points. And there's typically three to four knowledge points per topic."* Exact MA-internal term. Hecker also uses it. Doc's usage is accurate.

---

## 15. "Lesson / quiz / review" loop pattern

**Verdict:** CONFIRMED, though the doc's framing as "three-tier MA structure" is the doc author's synthesis.

**Source:** HOAW distinguishes "task selection" between (i) "what to learn next" (lessons), (ii) "what to review" (review tasks), and describes (iii) "quizzes" with 80% target accuracy and per-question remediation. Hecker and Skycak both use the lesson / review / quiz vocabulary. The pattern is present; the explicit "three-tier" framing is the doc's interpretation, not an MA self-description.

---

## 16. Whether Math Academy uses Bayesian methods at all

**Doc says (line 6, 248):** "Math Academy's algorithm is **not** Bayesian Knowledge Tracing."

**Verdict:** CONFIRMED — but with an important nuance.

**Source:** FIRe Q&A: Skycak says he *tried* a Bayesian approach for the *diagnostic*, then moved to FIRe (a "physical" model) — for the diagnostic. The FIRe spaced-repetition update is also explicitly non-Bayesian (deterministic max/exp formulas, no posterior).

**Nuance:** Skycak does not say "we use no Bayesian methods anywhere." He says BKT was the initial mental model and was abandoned because complexity exploded. The mastery scoring (`repNum`, `memory`) and the demotion controller are clearly non-Bayesian. Whether *any* component (e.g., difficulty calibration) uses Bayes is not addressed in the public sources. The doc's flat "not Bayesian" is defensible but slightly stronger than what Skycak actually claims.

---

## 17. "Numerical thresholds proprietary" claim

**Doc says (line 244, 287):** "the proprietary specifics (numerical thresholds, exact functional forms of `speed`, `rawDelta`, `decay`) are deliberately not disclosed."

**Verdict:** CONFIRMED.

**Source:** FIRe opening of the model section: *"This model was the product of years of intense R&D (from 2019-22). While the specific implementation is proprietary, I can talk about the high-level ideas here."* Hecker confirms MAW also withholds exact functional forms ("the book does not present the exact algorithm by which this done"). Doc's claim is well-supported.

---

## Summary

| Verdict      | Count | Items                              |
|--------------|-------|------------------------------------|
| CONFIRMED    | 12    | 1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 14, 17 |
| PARTIAL      | 4     | 6, 10, 12, 15                      |
| DISPUTED     | 0     | —                                  |
| UNSUPPORTED  | 0     | —                                  |
| CANT_TELL    | 1     | 16 (confirmed in spirit, slightly stronger than what sources strictly say) |

Quote audits (where doc used "scare-quotes"):
- "complexity explodes" — exact match (1).
- "physical" — exact match (1).
- `repNum → max(0, …)` formula — exact match (1).
- HOAW diagnostic paragraph — exact match (1).
- HOAW "conditionally completed" — exact match (1).
- Oz Nova "literal years … second slip up" — exact match (1).
- "knowledge frontier resolution" — **not verbatim**; "knowledge frontier" is real, "resolution" is doc author's gloss (1, mild misquote).

### Load-bearing concerns

These are the items that, if disputed or overclaimed, would materially affect the Delta Drills algorithm-stack prescription:

1. **Event-triggered demotion (claims 6, 8).** The architectural recommendation rests on this being how MA actually works. *Verdict:* The event-triggered framing is correct (CONFIRMED). The specific rule wording is slightly overclaimed (PARTIAL — "at the same knowledge point" conflates intra-lesson halts with two-failed-lesson plateau, which are distinct mechanisms in the published description). Practical impact: the user should treat "two failed composite attempts in an area without intervening progress" as a defensible heuristic *inspired by* MA, not as MA's literal rule. The MA rule is at the lesson level; the user's "area" is a different granularity.

2. **FIRe-style decay (claims 3, 4, 5).** All three are CONFIRMED at the formula level. The functional forms — multiplicative `speed`·`decay^failed` and `(0.5)^(days/interval)` — are accurately quoted from a primary Skycak source. The actual numeric values of `speed`, `decay`, `rawDelta`, etc., are admitted-proprietary (claim 17, CONFIRMED). The user can implement the *shape* of the update faithfully, but the constants are not public — meaning any precise re-implementation will require tuning, not transcription.

3. **Info-gain diagnostic on compressed cover (claim 9).** CONFIRMED verbatim. This is the strongest of the load-bearing claims; the doc lifts a HOAW paragraph cleanly. Note however that HOAW gives the *what*, not the *how* (no algorithm pseudocode); the user will need to operationalize "information gain" on their own, since MA does not publish the scoring function.

4. **"Not Bayesian" (claim 16).** CANT_TELL is the honest answer for "MA uses no Bayesian methods anywhere." Skycak says BKT was tried for the diagnostic and abandoned, and the published mastery and demotion logic is non-Bayesian. The doc's flat "not Bayesian" headline is rhetorically stronger than the source warrants. For Delta Drills, this does not undermine the prescription (Beta-Bernoulli + event-triggered demotion is a fine choice on its own merits), but the framing "MA proved Bayesian doesn't work" should be softened to "MA found BKT-specifically too complex for their diagnostic problem."

### Overall

The Math Academy sub-cluster is the **strongest** part of the research doc on quote-fidelity grounds. Roughly 70% of audited claims are fully confirmed; the remaining 30% are partials where the doc paraphrases tightly enough to be defensible but loose enough to mislead a reader doing source-checking. No outright fabrications. The two real fixes the user should make before treating this as architectural truth:

- Distinguish **knowledge-point halt** (intra-lesson) from **two-failed-lesson plateau** (cross-attempt) — they are different rules in MA, even though both fall under "event-triggered demotion."
- Treat "knowledge frontier resolution" as the doc author's coinage, not an MA-internal term. Search MA materials for "knowledge frontier" (which exists) rather than the longer phrase.
