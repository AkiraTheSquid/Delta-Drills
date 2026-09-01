---
kc: numpy.ndarray-model
---
First concept of the numpy course, and the one the attempt log indicts.

## Findings
- 2026-09-01 teachability (codex gpt-5.6-luna as learner, lesson text only):
  solved q559 3/3 on the real harness, no missing-information claim. The
  PAGE is sufficient — the 2026-08 stalls above were the DRILL POOL
  (untaught `a.T`), not the lesson prose; consistent with both findings.
- 2026-09-01: the log (2026-08-19→27) shows a 63-attempt `faded` rung stall at
  16% accuracy, mean predicted_p 0.207 — the model predicted failure ~60
  serves in a row and the selector complied. Dated a WEEK before the `a.T`
  prerequisite bug (transpose needed here, taught four lessons later) was
  found by hand. `/api/practice/kc-stats` now flags both patterns
  (`rung_stall`, `served_while_predicting_failure`).

## Edges
- [[numpy.reshape-flatten]]: same stall pattern in the same logs (43 attempts,
  14%) — the two failures share the untaught-transpose cause.

## Checks
- `app/kc_stats.py` flags — born from this concept's log.
- `scripts/audit_solution_prereqs.py` — born from this concept's `a.T`.
