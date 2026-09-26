# sims

## Purpose
- A learner simulator that decides which BELIEF should drive explore/exploit and
  concept gating: Bayesian FSRS alone vs Bayesian BKT (P(learned)) × FSRS.
- Offline research tool. Nothing in the app imports it.

## Owns
- The hidden true learner in two worlds (`world.py`): W1 two-state learned/unlearned,
  W2 gradual strength. A winner must hold in both.
- The beliefs under test (`beliefs.py`) and the one shared policy (`sim.py`).
- The grid runner and paired report (`run.py`).

## Does NOT own
- The app's models: `app/memory_model.py` (FSRS-6 + FIRe), `app/kc_explore.py`,
  `app/learning_xp.py`. The sim COPIES their numbers; changing the sim changes no app behaviour.

## Key Files
- `fsrs_vec.py`: FSRS-6 scalar + numpy ports of `memory_model` (parity-tested to 1e-9).
- `world.py`: concept graph (27 `arena_section` goals + their prerequisites, 71 concepts),
  learner types (novice / torch / math / strong), the true learner.
- `beliefs.py`: `BKTBelief` (B), `ParticleBelief(hybrid=False)` (A, FSRS only),
  `ParticleBelief(hybrid=True)` (H, FSRS + learned state), `OracleBelief`.
- `sim.py`: review → explore → exploit policy, run loop, metric.
- `run.py`: `pilot` (theta tuning), `final` (fresh seeds, breaks), `report`.
- `test_sim.py`: FSRS parity, common random numbers, collapsed-particle = point FSRS, sanity.

## Data & External Dependencies
- Reads `Local_Deployed_Shared/lessons/kc_registry.json` (prereqs, encompassing, arena_section).
- numpy; `test_sim.py`/`watch.py` import `app.memory_model` for parity.
- Output: `sims/out/*.jsonl` (gitignored).

## How It Works (Flow)
1. `run.py pilot --n 10`: every arm at every (gate theta, review threshold) pair on seeds
   0..9 → one pair per arm (`run.py pick` re-picks from `out/pilot.jsonl`).
2. `run.py final --n 100`: fresh seeds 1000+, each arm at its pilot pair, breaks 0/30/180 d.
3. `report`: per learner, arm vs reference on the SAME learner; median % with bootstrap 95% CI.

## Invariants & Constraints
- Gate and review are SEPARATE tuned knobs (Seth 09-26): gate = when the next concept
  unlocks, review = recall level that re-serves a learned concept. One shared value
  cost 10-25%.
- Arms differ ONLY in the belief. Same policy, costs, area prior, indirect credit, FIRe.
  The 09-24 sims' "+10% explore" was an artefact of arms with different bookkeeping.
- Common random numbers: truth draws are indexed by (concept, event count), so arms are paired.
- Tune and score on disjoint seeds.
- FSRS copy must match `memory_model` (watch.py + test_sim.py fail otherwise).
- 11 workers at nice 19 (Seth: keep CPU usable).

## Extension Points
- New belief: implement `K`, `K_row`, `gate`, `due_R`, `info`, `lesson`, `answer` in
  `beliefs.py`, register in `sim.ARMS` / `make_belief`.
- Rollout planner arm (deferred by Seth 09-26): a new action chooser in `sim.py` over the winner.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Gate on the goal itself grinds** — `RESOLVED`
  - When it happens: LEARNED read off P(correct a day out) ≥ theta.
  - Symptom: ~30 drills per concept; arms that declared early won via free spaced reviews.
  - Root cause: one day after a first Good review FSRS gives R 0.925 → P(correct) 0.79 on code.
  - Prevention/fix: each belief gates on its own "known" (P(learned); pure FSRS E[R 1 d]); theta tuned.
- **Scratchpad sims vanish** — `RESOLVED`: the 09-24 `usim/` lived in a session scratch dir and
  was deleted. This one is committed.

## Recent Changes
- 2026-09-26: Review threshold split from the gate (`sim.run(review_at=)`), tuned per arm.
  Critic fix: pure-FSRS never-reviewed particles now take FSRS's FIRST review on their first
  lesson/answer (they took a 365-day-overdue one: S 0.08 vs 1.5), and sessions stop at the
  horizon. Results from before this fix are void (biased against pure FSRS).
- 2026-09-26: Created. FSRS-Bayes vs BKT-Bayes explore/exploit, two truth worlds, paired CRN.
