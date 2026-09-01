---
kc: python.types-and-conversion
---
Second concept of the python course (ids 574–579 + 713–717). Supported by
[[python.values-and-names]].

## Findings
- 2026-09-01: `python.type-name` sat at 0/2 coverage for a month and was
  UNSATISFIABLE, not under-drilled — the symbol collector emitted
  `syntax.attribute` for `type(x).__name__` because the receiver is a Call.
  Fixed in `solution_symbols.py` (`_is_type_call`, narrowed after a codex
  finding so `module.__name__` earns nothing). A coverage number stuck at 0
  can be a DETECTOR gap, not a content gap.
- 2026-09-01: `syntax.equality` was used by 42 questions and taught by NO
  lesson. Declared here (the page already teaches `"42" == 42` as a named
  misconception); resolved 38 recorded prereq violations for one coverage
  obligation, met on arrival by q715/q717.
- 2026-09-01: q713's wrong example printed the CORRECT output for its example
  inputs (`(True, True)`); anchored to `solve(0, '')` where the misconception
  diverges. This became `wrong_example_matches_correct` (blocking) in
  `audit_question_bank.py` — a wrong example needs inputs where the
  misconception and the right answer disagree.

## Checks
- `audit_question_bank.py::check_wrong_examples` — born from q713 here.
- `scripts/audit_solution_prereqs.py` — `syntax.equality` ordering.
