---
kc: python.values-and-names
---
Root of the python course (ids 708–712 filled it to the rung floors, 2026-09-01,
commit cf3eaadc's predecessor slice). Course order across the whole family is
encoded in the question-id ranges: 568–573 here, 574–579 [[python.types-and-conversion]],
… 604–609 python.dots-and-imports.

## Findings
- 2026-09-01 teachability (codex gpt-5.6-luna as learner, lesson text only,
  'use only the information in the problem'): solved q711 (last independent
  drill) 4/4 cases on the real harness, no missing-information claim.
  The page is SUFFICIENT for its hardest rung.
- 2026-08-31: brought from 5 to 10 drills (Solo 6 / Integrated 3). No two drills
  on a rung repeat a move; verified against `audit_symbol_coverage.py`.

## Checks
- `scripts/audit_symbol_coverage.py` — every declared symbol drilled ≥2× on-node.
