# Engine Port Plan — Tracks Glicko + κ/n → Delta Drills

> **Status: DESIGN DOC. Awaiting Seth's review before any engine file is edited.**
> Track-1 spec: `kolb-learning-notes-[ai-sync]/0-primary-sources/practical-its-modifications-delta-drills/track-1-engine-port.md`.
> Reference implementation (PORT, don't reinvent): `~/Applications/tracks-its-software/backend/app/` (51 tests green).
> Start checkpoint: `Delta-Drills-Local-backup-2026-06-16T13-46-09--track1-engine-port-start`.

## 0. What we're swapping, in one paragraph

Replace DD's per-atom **point-estimate BKT** + **frozen `propagation_weight` FIRe** + hard
0.85 gate with the Tracks engine: **Glicko `Rating(θ,σ)` per atom + κ/n partially-pooled
encompassing edges + variance tax + decay-driven resurfacing + opportunistic edge probes.**
The graph topology, the `is_encompassing` subset, and prerequisite **gating** all stay; only
the *credit math* changes (frozen weight → κ/n pooled prior + tax) and mastery gains a σ axis.
Ported behind a backend flag (`use_glicko_engine`, default off); `bkt_mastery.py` stays intact
as the fallback. Math + config names copied verbatim from Tracks; every deviation is justified
in §8.

## 1. Mapping table (concretized to DD types/fields)

| Tracks concept | DD concept (file:symbol) |
|---|---|
| `(learner, area)` `Rating(θ,σ)` | per-(user, **atom**) state on `UserPracticeState` (`adaptive.py`) — new `atom_theta`, `atom_sigma`, `atom_eff_n`, `atom_attempts` dicts (replaces the single `atom_mastery` float) |
| `Week.area_ids()` / `drills` / `area_edges` | the 207-atom graph `concepts` + the question bank (`questions.py` `atom_tags`) + `prerequisite_edges` in `arena_drillable_v1.json` |
| `EdgeBelief` Beta(a,b) (κ,n) | replaces `PrerequisiteEdge.propagation_weight` (frozen → learned κ) on the **encompassing subset** (88 of 330 edges); persisted per-user in new `edge_beliefs` dict |
| `decay()` (θ half-life + σ inflate) | replaces `bkt_mastery.decay()` (L→p_init half-life); now also inflates σ → resurfacing |
| `item_rating(difficulty 1–5, spacing)` | DD `Question.difficulty_score ∈ [10,100]` rescaled to the logit opponent rating (§8.1) |
| `display_mastery` / `mastery_band` | what `/atom-gates` thresholds on (replaces raw BKT posterior `current_mastery`) |
| `sigma_mastered` σ-gate | the honesty firewall: `mastered` requires σ low **and** own data, not borrowed credit |
| `confidence` (weight multiplier) | `Question.atom_tags[i].confidence` (tag→atom certainty) — already exists, maps 1:1 |
| `weight` (`weight_to_area`) | DD drills have no per-drill weight → constant `1.0` (§8.4) |
| `score ∈ [0,1]` (Bradley–Terry) | DD grade is binary correct/incorrect → `coding_pass_score`/`coding_fail_score` (0.8/0.2) (§8.2) |
| `select_next` (area→drill) | `prioritization.select_next_subtopic` (subtopic→question) gains σ-resurfacing + ε + probes (§3.5) |

### 1a. Edge direction — the subtle part (READ THIS)

Tracks pooling lifts a **child** from its **parents**: `pool(child, parents=[(parent_rating, κ)])`.
In Tracks an edge is `parent→child` meaning *parent is the prereq, child is the dependent*, and
the child borrows the prereq's strength as a forward prior.

DD encompassing is the **opposite direction**: an encompassing edge is
`prerequisite_id` (simpler) → `dependent_id` (advanced), and mastering the **advanced** atom
credits the **simpler** one (Math-Academy FIRe, advanced→simpler). To port κ/n pooling onto the
encompassing subset, the pooling reading of an encompassing edge is therefore **reversed** vs its
gating reading:

- **Gating (access, unchanged):** `prerequisite_id → dependent_id`. You must clear the simpler
  prereq before the advanced atom unlocks. Uses **all** 330 prereq edges.
- **Pooling (credit, new κ/n):** for a *simpler* atom, its pooling **parents** are the
  *advanced* atoms that encompass it — i.e. `parent := dependent_id`, `child := prerequisite_id`,
  over the **encompassing subset only** (88 edges). The simpler atom continuously borrows θ
  (capped, taxed) from the advanced atoms' ratings.

This is the upgrade the spec asks for: the old FIRe fired once, only on a *correct advanced*
attempt, and never wrote back; κ/n pooling makes the simpler atom's prior track the advanced
atom's rating *continuously and bidirectionally* (a falling advanced rating now weakens the
borrowed credit), with the variance tax keeping σ wide so borrowed credit is a **prior, not
mastery**. Non-encompassing prereq edges carry **no** credit (exactly as today, where their
`propagation_weight == 0`).

## 2. New + changed files

### New modules (small, single-purpose — keeps `bkt_mastery.py` thin)
- `app/glicko.py` — **verbatim port** of Tracks `glicko.py`: `Rating`, `g_attenuation`,
  `expect`, `item_rating`, `decay`, `spacing_factor`, `update`, `display_mastery`,
  `mastery_band`. Only addition: `item_rating_dd(score_10_100, spacing)` wrapper (§8.1).
- `app/edges.py` — **verbatim port** of Tracks `edges.py`: `EdgeBelief`, `edge_id`,
  `lambda_weight`, `pool`, `probe_priority`, `consistency_step`, `apply_consistency`.
- `app/glicko_engine.py` — port of Tracks `engine.py` + the grade-application glue from
  `service.py`, adapted to DD's atom/graph/per-user-state world (no `Week`):
  `AtomSnapshot` (= Tracks `AreaSnapshot`), `fresh_snapshot`, `_decayed`, `pooled_ratings`,
  `current_masteries`, `is_mastered`, `update_snapshot`, `apply_attempt_glicko`,
  `gate_sets_glicko`, `atom_is_ready_glicko`, plus the edge-probe + σ-resurfacing selection
  helpers consumed by `prioritization.py`.
- `app/engine_graph.py` — loads `arena_drillable_v1.json` once into the structures the engine
  needs: `atom_ids`, `prereq_parents[child] → [parent…]` (gating), and
  `encompassing_parents[simpler] → [(advanced, κ_seed)…]` (pooling, reversed per §1a). Replaces
  the `_prereq_index` / `_encompassing_index` lru_caches in `bkt_mastery.py` with κ-aware ones.

### Changed files (I own these per the scope boundary)
- `app/bkt_mastery.py` → **becomes a dispatch facade.** Its public functions
  (`apply_attempt`, `current_mastery`, `gate_sets`, `atom_is_ready`, `is_mastered`,
  `prerequisites`, `encompassed_by`, `area_scores`, `UNLOCK_THRESHOLD`) keep their **names and
  signatures**; each checks `settings.use_glicko_engine` and routes to either the existing BKT
  body (now private `_bkt_*`) or the new `glicko_engine.*`. **Routers and `prioritization.py`
  need no signature change** — this is what makes the swap surgical and reversible.
  - One additive signature change: `apply_attempt(..., difficulty: float | None = None)`.
    `feedback_router` passes `question.difficulty_score`; `arena_rating_router` passes `None`
    (→ mid difficulty 3.0). Default keeps the BKT path byte-identical.
- `app/concept_graph.py` → `PrerequisiteEdge` gains `kappa_mean: float | None = None` and
  `kappa_strength: float | None = None` (additive, optional, validated to (0,1) / >0). Existing
  `is_encompassing` / `propagation_weight` fields + their validators stay (BKT path still reads
  them). No behavior change when the fields are absent.
- `app/config.py` → add the engine block with **Tracks' exact constant names** (`half_life_days`,
  `sigma0`, `rd_inflation_per_day`, `item_sigma`, `difficulty_spacing`, `band_z`,
  `display_slope`, `mastery_threshold`, `sigma_mastered`, `spacing_window_days`,
  `massed_weight_floor`, `kappa_mean`, `kappa_strength`, `tau`, `beta_tax`, `pool_cap`,
  `probe_*`, `exploration_epsilon`, `probe_epsilon`, `uncertainty_bonus`, `target_success`,
  `coding_pass_score`, `coding_fail_score`) + the flag `use_glicko_engine: bool = False`
  (env `DELTA_USE_GLICKO`). Same defaults as Tracks `config.py`.
- `app/adaptive.py` → `UserPracticeState` gains the new state dicts (§4), with **additive,
  back-compat** save/load (mirrors how `atom_mastery`/`atom_last_ts` were added). The
  "Calibrating — 1 of 3" cold-start counter (`sub_state.n` at `apply_feedback`, line 329) is
  **subsumed**: with Glicko there is no 3-question calibration phase (σ starts at `sigma0` and
  shrinks with evidence), so when the flag is on the counter is not consulted for selection.
  Note: the *UI string* lives in frontend `practice/*.js` (Quick-Wins track owns it); we only
  stop driving it from the backend selector.
- `app/prioritization.py` → `subtopic_mastery` averages pooled-Glicko display masteries instead
  of BKT posteriors; `select_next_subtopic` priority gains the σ-resurfacing bonus (encoded
  atoms only), ε-exploration floor, and edge-probe injection; `target_difficulty` uses
  `expect()`/`item_rating` matched to `target_success` instead of the affine `_DIFF_FLOOR/_SPAN`
  map. All behind the flag; BKT path preserved.
- `app/data/concept_graphs/watch.py` → **extend** (not replace) the invariants: keep the
  encompassing ⊆ prerequisite + propagation-weight checks; add "if `kappa_mean` present →
  0<κ<1" and "if `kappa_strength` present → >0", and assert every encompassing edge's
  `dependent_id`/`prerequisite_id` resolve (pooling-direction integrity).

### Untouched (Quick-Wins track owns — per scope boundary)
Frontend `practice/*.js`, `code_runner.py`, `timeout=5` sites, manifests, auth UI,
`problem_feedback_router.py` (new, uncommitted, theirs). Coordination is the `/atom-gates`
JSON shape only (§5), kept backward-compatible.

## 3. Engine behavior (ported, per-DD)

1. **Per-atom state** decays to `now` (`glicko.decay`: θ half-life toward 0, σ inflates toward
   `sigma0`), then each *simpler* atom pools toward its encompassing-parent (advanced) ratings
   (`edges.pool`, κ/n weighted, Λ-capped, variance-taxed).
2. **A graded attempt** (`update_snapshot`): decay → spacing-discount the weight
   (`spacing_factor`, anti-cram) → `glicko.update` (Glicko-1, q=1, fractional game of size
   `weight·confidence·spacing`). `eff_n += w`, `attempts += 1`. `confidence` = the atom-tag
   confidence; `score` = 0.8/0.2 from binary correctness.
3. **Edge probes** (`consistency_step` + `apply_consistency`): an attempt on a *simpler* atom
   while its encompassing-parent (advanced) atom is weak is an opportunistic probe — clean pass
   despite weak parent ⇒ disconfirm the edge (b+=step); fail as the edge predicts ⇒ confirm
   (a+=step). Fractional steps; κ moves slowly (honest single-learner outcome). Parent weakness
   judged on **own-data-only** pre-update masteries (`EdgeBelief(0,1)` → κ=0, no self-vouching),
   exactly as Tracks `service.apply_grade`.
4. **Gating** (`gate_sets`/`atom_is_ready`): see §5.
5. **Selection** (`prioritization`): pooled display mastery → weakest-first, with σ bonus
   `uncertainty_bonus·σ` added **only for encoded atoms (attempts>0)** (resurfacing; a fresh
   atom's σ0 is ignorance, not decay), + ε-exploration floor + ε-probe injection toward the
   highest `probe_priority = λ·(1−κ)` edge. Difficulty matched toward `target_success` via
   `expect()`.

## 4. State migration (existing user JSON → Glicko)

Per-user files in `user_data/*.json` (and the Fly volume in prod). Migration runs **lazily on
load** in `adaptive._load_user_state` (additive, idempotent), so no separate batch job and no
data loss — old fields are kept.

New `UserPracticeState` fields (all default empty; persisted in `_save_user_state`):
`atom_theta: Dict[str,float]`, `atom_sigma: Dict[str,float]`, `atom_eff_n: Dict[str,float]`,
`atom_attempts: Dict[str,int]`, `edge_beliefs: Dict[str, [a,b]]`. `atom_mastery`/`atom_last_ts`
stay (BKT fallback + migration source).

Seeding rule (only when `use_glicko_engine` and `atom_theta` empty but `atom_mastery` present):
- **θ** ← invert `display_mastery`: `θ = (logit(L) + theta_ref) / display_slope`, clamped, where
  `L = current_mastery(atom)` (decayed BKT posterior). Fresh/unpracticed atoms → θ=0.
- **σ** ← see the migration-cliff decision in §7 (recommended: grandfather previously-cleared
  atoms to `σ = sigma_mastered`, everything else `σ = sigma0`).
- **eff_n / attempts** ← DD never tracked per-atom attempt counts; seed `attempts = 1`,
  `eff_n = 1.0` for any atom present in `atom_mastery` (practiced ≥ once), else 0. Approximate but
  honest — σ does the real gating, and a wide σ can't read as mastered.
- **edge κ** ← seed `EdgeBelief(a=κ·s, b=(1−κ)·s)` with `κ = edge.confidence` (the encompassing
  edges already carry a `confidence≈0.7` field — a ready-made edge-confidence prior) and
  `s = kappa_strength` (default 6). This realizes the spec's "math ≈ 0.85 strong / ARENA ≈ 0.70
  movable" by domain — the graph is all-ARENA, so κ≈0.70 falls out of the data. `n` from the
  child's seeded `eff_n`.

## 5. `/atom-gates` contract (backward-compatible)

Endpoint returns the **same JSON shape** — `{ready: [...], mastered: [...], threshold: float}` —
consumed by `practice/adaptive.js`, `practice/drills-catalog.js`, `stats/predicted-prereqs-temp.js`.
The facade `gate_sets` dispatches; Glicko computes the two sets as:
- **`mastered`** (honesty firewall, `is_mastered`): `display_mastery(pooled θ) ≥ mastery_threshold`
  **AND** `σ ≤ sigma_mastered`. Borrowed credit can't qualify (variance tax keeps σ wide); used
  for the "fully mastered" UI and composite/ARENA unlock (all component atoms mastered).
- **`ready`** (access gate): an atom is ready iff every gating prereq is **cleared**, where
  "cleared" thresholds the mastery **band**, not the point — see §7 decision A for the exact
  predicate. Root atoms always ready. `NON_GATING_ATOMS` still stripped.
- **`threshold`**: keep returning `UNLOCK_THRESHOLD` (0.85) for back-compat; document that under
  Glicko the gate is band-based, the number is advisory for the frontend's display only.

## 6. Reversibility

- **Flag:** backend `settings.use_glicko_engine` (env `DELTA_USE_GLICKO=1`), default **off**.
  Off → every facade routes to the untouched BKT body; the new dicts sit unused; behavior is
  byte-identical to today. This mirrors the `window.DELTA_USE_BACKEND` reversibility pattern,
  server-side (the frontend needs no flag because `/atom-gates` is shape-stable).
- `bkt_mastery.py`'s BKT math is preserved in full (renamed private, same numbers).
- Migration is additive: flipping the flag back off falls straight back to `atom_mastery`.
- Rollout: validate locally with flag on (a test user), then a single Fly env var flip; revert
  = unset the var. **No deploy in this task** (DoD: "Do NOT deploy unless Seth asks").

## 7. Decisions for review (need your call before I implement)

**A. Access-gate predicate (the migration cliff).** The honesty σ-gate is right for *declaring
mastery*, but if `ready` also required `σ ≤ sigma_mastered`, every migrated user's σ=`sigma0`
would re-lock the whole graph on day one, and learners would need many spaced reps on *every*
prereq before *anything* downstream unlocked.
  - **Recommended:** access "cleared" = **band lower bound** `mastery_lo ≥ 0.70` (gentler than
    the 0.85 point gate, but band-based so a wide σ still fails it — honest without the cliff),
    **and** grandfather migration: seed `σ = sigma_mastered (0.90)` + modest `eff_n` for atoms
    whose BKT `L ≥ 0.85` so their `mastery_lo` clears the bar on day one (preserves current
    unlock state; future decay/σ-inflation resurfaces them for review naturally). New mastery
    *claims* still need real spaced data (σ must reach `sigma_mastered` via own attempts, which
    borrowed credit can't do).
  - Alt 1: keep a **point** gate on pooled θ (`display_mastery ≥ 0.85`) for `ready` — closest to
    today's behavior, ignores σ for access (σ only firewalls `mastered`). Simplest, least
    honest.
  - Alt 2: band gate with **no** grandfathering — cleanest theory, but migrates users into a
    visibly re-locked graph. Reject unless you want the reset.

**B. Pool over encompassing-only vs all prereq edges.** Recommended: **encompassing subset only**
(faithful to "replaces `propagation_weight` on the encompassing subset"; non-encompassing edges
stay gating-only, as their `propagation_weight==0` already implies). Alt: pool over all prereq
edges in the forward (foundation→advanced) direction too — richer transfer, but that's *new*
behavior the old model never had and the spec didn't ask for.

**C. Config home.** Recommended: extend `config.py`'s `Settings` (one settings object, env-driven,
matches the flag). Alt: a separate `engine_config.py` copying Tracks `config.py` verbatim. Both
keep Tracks' names; pick the ergonomics you prefer.

## 8. Justified deviations from the Tracks port

1. **Difficulty scale.** Tracks items are 1–5; DD `difficulty_score ∈ [10,100]`.
   `item_rating_dd(s) = ((s−10)/90·4 + 1 − 3)·difficulty_spacing` — linear [10,100]→[1,5],
   centered at 3 (=55). Pure rescale; the logit math is unchanged.
2. **Binary grade → score.** DD correctness is a bool; Tracks score is continuous. Use Tracks'
   own binary-gate rescale `score = coding_pass_score (0.8) if correct else coding_fail_score
   (0.2)` (these constants exist in Tracks `config.py` for exactly this).
3. **No `Week`.** DD has no week object; the engine operates over the full 207-atom graph + bank.
   `select_next` is adapted to DD's subtopic→question selection rather than area→drill, keeping
   the frontend's existing selection surface.
4. **`weight_to_area`.** DD drills carry no per-drill area-weight; use constant `1.0`. The
   atom-tag `confidence` already supplies the per-attempt weight multiplier.
5. **Edge-direction reversal** for pooling on the encompassing subset (§1a) — required because
   DD encompassing credit flows advanced→simpler, opposite to Tracks' prereq→dependent pooling.

## 9. Honesty invariants → tests (must hold)

| Invariant (spec §"Honesty invariants") | Test (mirrors Tracks) |
|---|---|
| Borrowed/encompassing credit raises θ but σ stays wide; can't reach `sigma_mastered` on pooled credit alone | `test_propagated_credit_alone_cannot_clear_mastery` (port) + DD migration variant |
| Massed practice can't fake mastery (`spacing_factor` discount) | `test_massed_repetition_cannot_fake_mastery`, `test_spaced_attempts_outgain_massed` (port) |
| Single-learner κ barely moves (tens of probes/edge) | `test_apply_consistency_moves_kappa_slowly` (port) |
| `is_encompassing` / subset guarantees preserved | extended `concept_graphs/watch.py` + a graph-integrity test |
| Decay both lowers θ and inflates σ → resurfacing | port `test_resurfacing.py` (`test_mastered_area_decays_below_threshold`, `test_stale_encoded_atom_outranks_fresh_atom`) |

## 10. Test plan + β calibration

DD convention = standalone validation scripts (no pytest), structural + behavioral + independent,
exit-nonzero on fail (mirrors `scripts/test_bkt_mastery.py`). Add:
- `scripts/test_glicko.py` — port `test_glicko.py` (spacing, attenuation, expect, item_rating,
  update win/loss/zero-weight/fractional, decay regress + cap, display + band).
- `scripts/test_edges.py` — port `test_edges.py` (λ shrink/scale, pool identity/blend/cap, tax,
  probe priority, consistency truth-table, slow-κ).
- `scripts/test_glicko_engine.py` — port `test_engine.py` + `test_resurfacing.py`, **retargeted to
  the DD graph** (real atom ids from `arena_drillable_v1.json`, encompassing-direction pooling),
  plus: migration round-trip (BKT JSON → seed → `/atom-gates` parity within tolerance under the
  grandfather rule), `/atom-gates` shape stability, and a flag-off byte-identity check
  (Glicko-off `gate_sets` == today's BKT `gate_sets`).
- `scripts/calibrate_beta_tax.py` — simulation: sweep `beta_tax`, for a fully-unverified edge
  (κ→0) measure the pooled child's σ with zero own data; pick the smallest β such that σ inflates
  back to ≥ `sigma0` (predictive variance ≈ the no-data prior). Record the chosen β + the curve in
  this doc. Per the spec, `beta_tax`'s additive form is documented as a **plausible-but-unvalidated
  heuristic**, not a derived quantity.

## 11. Definition of done (from the spec)

- [ ] This plan reviewed by Seth. ← **we are here**
- [ ] Engine ported faithfully behind `use_glicko_engine` (default off); BKT preserved.
- [ ] Migration runs on existing `user_data/*.json` with no data loss; `/atom-gates` still serves
      the frontend (shape unchanged).
- [ ] Honesty invariants (§9) hold and are covered by §10 tests; `beta_tax` calibration recorded.
- [ ] `concept_graphs/watch.py` invariants extended, not dropped.
- [ ] `backup_delta_drills_local` checkpoint at the end. **No deploy unless Seth asks.**

## 12. Open risk notes

- `arena_drillable_v1.json` is the live graph the engine loads (`bkt_mastery.GRAPH_PATH`); note
  `concept_graph.DEFAULT_GRAPH_PATH` points elsewhere (`arena_prereqs_einops_foundations.json`)
  and memory flags a vocab-disjoint issue (only ~21/207 atoms appear in any edge). Pooling credit
  is therefore inert for most atoms regardless of engine — **this port does not fix that**; it
  swaps the math on the edges that do exist. Worth a one-line note to Seth so expectations are set.
- Per-atom attempt counts were never stored, so migrated `eff_n`/`attempts` are coarse seeds; σ
  (wide) is what actually prevents false mastery, so this is safe but means the first post-migration
  sessions will feel "uncertain" (more resurfacing) until real reps accrue — intended.
