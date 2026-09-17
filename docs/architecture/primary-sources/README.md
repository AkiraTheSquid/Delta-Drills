# Primary sources for `Delta-Drills-Mastery-Models-Evidence-Review.pdf`

The papers in this folder are the audited primary sources cross-referenced
in the Evidence Review. Each is keyed to specific claims in the review.

Read these in priority order if you want to verify the most load-bearing
claims yourself.

## Reading priority

| # | File | Backs which claim | Status |
|---|---|---|---|
| 1 | `2016_pelanek_elo-adaptive-educational.pdf` | "Elo is the recommended migration target"; **"r ≈ 0.70 task-difficulty correlation after 5 responses"** (Report 2 claim, unverified — check this paper's §results); K-factor calibration choices | Likely the source for Report 2's "r ≈ 0.70 after 5 responses" figure — verify before quoting |
| 2 | `2017_pelanek_bkt-logistic-and-beyond.pdf` | "Pelánek's survey of BKT vs logistic vs Elo families"; Hypothesis 1 (p. 33) on when to prefer each model family | Audited in v2 doc, abstract + Hypothesis 1 confirmed; details of BKT identifiability discussion are §3 |
| 3 | `2019_park-cornillie_multidim-irt-monitoring-ability.pdf` | "M-ERS (Multidimensional Elo Rating System) is the published fix for multi-tag Elo averaging weakness" | M-ERS introduction + the multidimensional update derivation are the load-bearing sections |
| 4 | `2021_maier-baker-stalzer_pfa-challenges.pdf` | "Even-skill averaging AUC 0.7849 beat compensatory PFA 0.7818" (§4.3); "PFA convergence in 2–12 practice range, not at N=30" (§4.1); "Three PFA degeneracy types observed" (§4.2); merged-rare pooling fix | **Already read end-to-end.** Verified verbatim. |
| 5 | `2023_skycak_fire-spaced-repetition.html` | Math Academy's FIRe update formulas verbatim; "I tried BKT for the diagnostic algorithm, complexity explodes" (verbatim); encompassing vs prerequisite distinction | **Already read end-to-end.** Verified verbatim. |
| 6 | `2025_bijl_probabilistic-decay-trees.pdf` | "The only published instance of Beta-distribution conjugate priors with explicit exponential decay over a graph of skills" | Single-author arXiv preprint. Untested at scale. Read critically. |

## Reference docs (also copied here)

| File | What it is |
|---|---|
| `MASTERY_ESTIMATION_REFERENCE_v2.md` | The v2 audited reference for mastery-estimation literature (post 2026-05-24 source audit). Maps individual claims to which papers back them. **Recommended as the bridge between this evidence review and the primary papers.** |
| `VERIFICATION_SUMMARY.md` | The summary of the 4-cluster source audit that produced v2. Calls out specifically which claims from the original v1 deep-research report turned out to be wrong or fabricated. Reading this first calibrates how much to trust any AI-generated summary of these papers. |

## Pointers not in this folder (would need separate retrieval)

- **Yudelson 2019, "Elo, I Love You Won't You Tell Me Your K"** — focused paper on K-factor calibration for educational Elo. Referenced in v2 audit doc but not downloaded.
- **Glickman, "Glicko-2"** — the standard extension that adds a rating-deviation (time-decay-like) signal to Elo. Pre-2010 paper, in the public domain.
- **Khan Academy mastery system documentation** — referenced by both AI reports as "moved away from streak-based mastery" but no primary source in the corpus confirming this.

## How to use this folder

1. Read `Delta-Drills-Mastery-Models-Evidence-Review.pdf` first (in the parent folder).
2. For any claim you want to evaluate yourself, find the row in §4 of the review (the cross-check table) — it names the paper and the section.
3. Open the corresponding PDF here. Most of these are short (10–30 pages).
4. If the v2 audit reference (`MASTERY_ESTIMATION_REFERENCE_v2.md`) covers the same claim, read its rendition first — the v2 audit already cross-checks the original wording against the source.

The PDFs are unchanged copies from the `papers/mastery-estimation/` corpus
elsewhere on disk — not modified or annotated.
