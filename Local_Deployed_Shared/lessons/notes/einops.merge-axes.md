---
kc: einops.merge-axes
---
## Findings
- 2026-09-01 teachability (codex gpt-5.6-luna as learner, lesson Concept +
  worked example only, instructed to use nothing else and to say what is
  missing rather than guess): solved q400 (last independent drill,
  `'b c h w -> c h (w b)'` interleave) 4/4 cases on the real harness, no
  missing-information claim. The page is sufficient for its hardest rung.
- 2026-09-01: the graph-structure audit's 41 order-only dependencies
  concentrate here and in the sibling einops KCs — drills lean on
  numpy-taught fixtures (`torch.arange`) with no lattice path behind them.
  See `scripts/graph_structure_baseline.json`, keys `edge-missing|einops.*`.

## Checks
- `scripts/audit_graph_structure.py` — edge-missing family.
