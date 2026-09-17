# Empirical `propagation_weight` Estimation — AFM / Transfer Learning-Curve Spec

**Goal:** Replace the v0 hand-guessed `propagation_weight` on each encompassing edge
with a value *estimated from attempt data*, using the Additive Factors Model (AFM) /
Learning Factors Analysis (Cen & Koedinger) transfer-curve method. Same family as the
field's gold standard for "did skill A actually help skill B" (Koedinger 2016, Pavlik/Cen/Koedinger 2009).

This is the empirical answer to the deferred "weights want domain-expert sign-off" caveat
and to the learner-self-report idea: behavior is the primary signal, self-report is a
cross-check label (it is unreliable on its own — Deslauriers et al. 2019).

---

## 0. What we are measuring

FIRe credits **dependent (advanced) D → prerequisite (simpler) P**: mastering D also
exercises P, scaled by `propagation_weight`. The empirical claim to test, per edge `D→P`:

> Does a learner's prior practice on **D** predict success on **P**, beyond what their
> own prior practice on **P** predicts?

- Significant positive cross-effect ⇒ genuine shared component ⇒ keep `is_encompassing`,
  set `propagation_weight` from the effect size.
- Null cross-effect ⇒ pure prerequisite (gating only) ⇒ demote `propagation_weight → ~0`,
  keep the prereq edge.

A *shared* (encompassing) component tends to show **bidirectional** transfer (D-practice
helps P *and* P-practice helps D); a pure prereq shows at most an asymmetric intercept
("can't attempt D until P unlocked") — that's gating, not propagation. We use that as a
discriminator.

---

## 1. Data extraction → long table

Source of truth (do NOT use BKT posteriors — circularity):

- Per-user state JSON in `adaptive.DATA_DIR` → `SubtopicState.history` → `AttemptRecord`.
  Fields used: `question_id`, `correct` (bool), `timestamp` (ISO-8601 UTC).
- `question_id → [{atom_id, confidence}]` from `data/question_atom_tags.jsonl`.

Build one row per (attempt × tagged atom):

| col | source | note |
|---|---|---|
| `user_id` | state filename | student intercept |
| `atom_id` | tag | the KC |
| `t` | parsed `timestamp` | for ordering + decay covariate |
| `correct` | `AttemptRecord.correct` | **0/1 outcome — the only label** |
| `conf` | tag confidence | observation weight (down-weight fuzzy tags) |
| `opp_self` | cumulative count of prior rows for (user, atom) | AFM opportunity count |

**Coverage gap (must fix or flag):** only `/submit` and `/submit-local-eval` append an
`AttemptRecord` to `history`. Drill / ARENA-rating attempts update BKT but write **no**
`AttemptRecord`. So today's longitudinal log is **bank-only**.
→ Recommended fix before estimation: in `feedback_router` / `arena_rating_router`, append a
lightweight atom-tagged attempt event (atom_id, correct, ts) to a dedicated
`atom_attempt_log` so drills/ARENA contribute. Otherwise accept bank-only coverage for v1
and `log()` the omission (no silent caps).

---

## 2. Model

Standard AFM (logistic, own-practice only) as the baseline:

```
logit P(correct | user s, atom k, i-th opportunity)
    = θ_s            # student ability intercept (drop if single learner)
    + β_k            # atom easiness intercept
    + γ_k · opp_self # own-practice learning rate × prior opportunities on k
```

Transfer extension — add, for every candidate encompassing edge `D→P`, a cross-practice term
to P's rows:

```
logit P(correct on P, opp i)
    = θ_s + β_P + γ_P · opp_self(P)
    + Σ_{D : D→P encompassing}  δ_{D→P} · m_D(before t_i)
```

where `m_D(before t_i)` = the learner's prior practice signal on D at the time of this P
attempt. Use **prior correct opportunities on D** (mastery-weighted) rather than raw count,
since FIRe credits *mastering* D.

Observation weight = `conf` (tag confidence) so a 0.4-confidence tag contributes ~half an
attempt of evidence — mirrors the runtime confidence-scaled update.

**Decay confound:** transfer and forgetting alias if ignored. Add a recency covariate
(e.g. `log(Δt since last opp on k)`) or restrict `m_D` to a bounded recency window. Without
this, a positive δ can be a same-session artifact.

---

## 3. Estimation procedure

1. Fit baseline AFM (own-practice only) → `β_k`, `γ_k`. Sanity: γ_k > 0 (learning happens).
2. Fit AFM + transfer for all 331 candidate edges **with L1 (lasso) penalty on the δ's.**
   The true encompassing set is sparse; lasso shrinks spurious δ to exactly 0, so the model
   *prunes non-encompassing edges for free* — this is the data-driven analog of the
   reverted "demote most to 0" recalibration, but earned rather than imposed.
3. Model-improvement test (this IS the LFA move): compare transfer model vs own-only by
   **held-out log-likelihood + AIC/BIC**, and report **cross-validated AUC** (hold the
   iter-4 bar, AUC ≥ 0.70; aspirational 0.87). If transfer doesn't beat own-only
   out-of-sample, the encompassing layer isn't earning its keep — report that honestly.

### Mapping δ̂ → `propagation_weight`

`propagation_weight` scales "how much practicing D counts toward P." Natural empirical ratio:

```
w_{D→P} = clamp( δ̂_{D→P} / γ̂_P , 0, 1 )
```

— the fraction of P's own-practice learning rate that D-practice reproduces. `w≈1` ⇒ full
"trunk" encompassing (D-practice ≈ as good as P-practice); `w≈0` ⇒ no propagation.

`confidence` per edge = function of the δ̂ standard error / CI width and the per-edge sample
size (tight CI, large n → high confidence).

---

## 4. Write-back (don't clobber the v0 graph)

Mirror the existing MS/Candidate/Rejected tiering discipline:

- Write `propagation_weight_empirical`, `propagation_weight_ci`, `n_obs`, `delta_hat` to each
  edge — **keep the v0 `propagation_weight` as fallback.**
- Promote `empirical → propagation_weight` **only** where `n_obs ≥ N_min` **and** the CI
  excludes 0. Leave under-powered edges at v0. (validate-don't-assert: don't overwrite a
  guess with a noisier guess.)
- Run the existing `eg_validate.py` after write-back (schema, prob ranges, per-kind DAG
  unchanged — this only touches a weight, not topology).

---

## 5. Honesty / power caveats (state these in any result)

1. **n=1 is underpowered.** One learner (you) gives directional sanity per edge, not
   calibrated weights. Per-edge δ needs a cohort. This is the concrete reason to pay an
   annotator / run a small cohort: it turns the estimator from "sign check" into "magnitude."
   Report per-edge `n_obs` and CI; never present a point weight without them.
2. **Circularity guard:** raw `AttemptRecord.correct` only. If you ever fit to `atom_mastery`,
   you measure FIRe's own assumption — the result is meaningless.
3. **Coverage:** bank-only until drill/ARENA attempt logging lands (§1).
4. **Self-report join (optional):** where a retrospective "did D help P?" label exists,
   agreement with sign(δ̂) = high confidence; disagreement = flag for review. Self-report is
   the *label that checks the behavior*, never the primary estimate.

---

## 6. Minimal build order

- [ ] `scripts/build_atom_attempt_table.py` — state JSONs + tags → long CSV (§1).
- [ ] (fix) append atom-tagged attempt events on drill/ARENA paths (§1 coverage).
- [ ] `scripts/fit_afm_transfer.py` — baseline + lasso transfer fit, CV AUC, AIC/BIC (§2–3).
- [ ] `scripts/write_propagation_weights.py` — δ̂→w, tiered write-back, re-validate (§4).
- [ ] report: per-edge table (δ̂, w, CI, n), model-improvement verdict, demoted-edge list.
