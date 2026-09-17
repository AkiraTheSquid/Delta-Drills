# concept_graphs

## Purpose
- Holds the curriculum **concept graphs** that drive graph-based sequencing:
  which atom is a prerequisite for which, and (v3+) which prerequisite edges
  are *encompassing* so mastery credit can trickle down.
- Domain concern: "what should the learner work on next, and what do they
  get implicit credit for" — not the estimator math itself.

## Owns
- The JSON concept-graph artifacts and their schema conformance.
- The encompassing-layer data: which prerequisite edges are flagged
  `is_encompassing` and their `propagation_weight`.

## Does NOT own
- The `CurriculumGraph`/`PrerequisiteEdge` schema + loader + gating logic —
  those live in `../../../concept_graph.py`.
- The mastery estimator (EWMA/decay) — `../../../adaptive.py`.
- The (deferred) propagation function that *consumes* encompassing edges —
  EG2, to live in the practice/engine layer.

## Key Files
- `arena_iter5_v2.json`: prerequisite-only graph (393 atoms, 375 edges),
  4-source-corroborated. The pre-encompassing baseline.
- `arena_iter5_v3_encompassing.json`: **v3** — encompassing layer + direction
  normalized. Edge direction set by per-edge LLM judgment (NOT the noisy
  relation slots), 25 parallel edges deduped (375→350), `prerequisite_id` is
  now uniformly the simpler atom. 331/350 edges `is_encompassing` with
  continuous `propagation_weight` (mean ~0.56). Built 2026-05-29.
  Validation: direction 91% vs independent blind judge; structural + behavioral
  (trickle-down) checks all pass. Magnitude is the soft spot — needs domain-expert
  sign-off (see Invariants). Run `scripts/eg_validate.py`.
- `arena_prereqs_einops_foundations.json`: small hand-seeded einops slice;
  current `DEFAULT_GRAPH_PATH` in `concept_graph.py`.
- `*.graphml`: yEd exports for visual inspection.
- `watch.py`: health checks (loads, schema fields, encompassing invariants).

## Data & External Dependencies
- Consumed by `concept_graph.py` via `load_curriculum_graph(path)` →
  `CurriculumGraph` (pydantic).
- No external services. Build tooling: `../../../scripts/eg_build_edge_packets.py`
  and `../../../scripts/eg_merge_decisions.py`.

## How It Works (Flow)
1. `eg_build_edge_packets.py` reads v2, emits per-edge context packets in
   balanced topic clusters (`/tmp/eg_packets/`).
2. LLM classifiers label each prerequisite edge `is_encompassing` +
   `propagation_weight`; an adversarial audit pass demotes over-flags.
3. `eg_merge_decisions.py --write` merges decisions (with audit overrides)
   back onto v2, DAG-checks credit flow, and writes v3.

## Invariants & Constraints
- **Encompassing ⊆ prerequisite.** Encompassing is a *flag on a prerequisite
  edge*, never a separate/peer edge type. Never invent a standalone
  encompassing edge that isn't already a prerequisite.
- Credit flows **dependent → prerequisite** (reverse of the prereq arrow);
  `dependent_id` = advanced atom, `prerequisite_id` = simpler atom (normalized
  by per-edge judgment, not the noisy relation labels).
- A non-encompassing edge MUST have `propagation_weight == 0.0`; an
  encompassing edge MUST have `0 < propagation_weight <= 1.0` (full
  subsumption = 1.0, per Skycak's "trunk"; partials fade toward 0).
- Both the prerequisite graph and the encompassing credit-flow graph MUST be
  acyclic (enforced in `eg_finalize.py`).
- **Magnitude needs domain-expert sign-off.** LLM passes agree well on
  *direction* (~91%) but the *fractions* are uncertain and prior-sensitive.
  Skycak says encompassing weights are set by a domain expert — treat the
  current weights as a v0 estimate to be reviewed, not ground truth. See the
  deferred instructor-review task.

## Extension Points
- New edges/atoms: regenerate v2 upstream (arena-iter5 pipeline), then re-run
  the two `eg_*` scripts to refresh the encompassing layer.
- To make v3 the live graph: point `DEFAULT_GRAPH_PATH` (or the practice
  loader) at `arena_iter5_v3_encompassing.json`. NOT done yet — deferred
  with EG2/EG4 (the propagation consumer).

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Short name of issue** — `ACTIVE` or `RESOLVED`
  - When it happens: one line about the situation/context.
  - Symptom: what you see break.
  - Root cause: the underlying mistake or assumption.
  - Prevention/fix: the rule, pattern, or helper to use so it doesn't come back.
  - Status: `ACTIVE` = still a risk, `RESOLVED` = was an issue, now fixed (keep for history).

## Recent Changes
- 2026-05-29: Rebuilt v3 with **per-edge direction judgment + dedup**. Found the
  iter-5 edge slots encode direction inconsistently (relation labels are noisy:
  `is-a` has no fixed direction; parallel multi-relation edges contradict). Fixed
  by judging advanced-vs-simpler per edge instead of trusting slots, deduped 25
  parallel pairs (375→350). Validated: direction 91% agreement vs an independent
  blind judge; structural + behavioral propagation checks pass (`eg_validate.py`).
  A skeptical recalibration pass was tried and **reverted** — an independent blind
  validator showed it over-demoted (its 42% zero-rate vs the neutral consensus
  ~6–9%); ARENA's implement-from-scratch corpus is genuinely densely encompassing,
  unlike Skycak's K-12-math prior. Magnitude still needs domain-expert sign-off.
  Build: `eg_build_edge_packets.py` → 5-cluster LLM judgment → `eg_finalize.py`.
- 2026-05-28: First encompassing layer (binary flag + audit) — superseded by the
  2026-05-29 direction-judged rebuild.
- 2026-05-13: Initial doc created.
