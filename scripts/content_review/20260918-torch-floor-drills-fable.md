# Content review — torch floor drills q1563–1608 (Fable, in-session author)

Scope: 46 drills raising eleven torch/ray KCs to the Solo ≥ 8 / Integrated ≥ 4
floor (ranges, boolean-masking, linalg-basics, argmin-argmax,
stack-concat-interleave, axis-reductions, slicing-views, transpose-axes,
views-and-copies, out-argument, slice-assignment, ray-parametrisation,
make-rays-1d) plus the de-comprehension of q512/1521/1522/1523. Reviewed
against `scripts/CONTENT_RUBRIC.md` by the AUTHOR of these drills — the same
session wrote them, so this is a self-review, not an independent one. GPT-6
Astra's pass is owed: codex quota exhausted until 2026-09-19 08:36; rerun
`scripts/content_critic.py --qids 1563-1608 --label torch-floor-drills` then.

## 1. Drills

Pass on every A/B/E code, verified mechanically: 1563–1565, 1567–1575,
1576–1581, 1583–1589, 1594–1596, 1598–1608. Every answer executed under the
real harness (`code_runner.run_function_tests`, torch preloaded), every case
disagrees with its siblings, every near miss was executed and differs from the
correct output on its anchored case (`bank_audit_report.json` GATE PASS).

Fixed during this review (all applied before commit):
- `GIVE_PROMPT` major — q1597 named `t.equal` in the prompt → "holds exactly
  `a`'s values, as a bool".
- `GIVE_STARTER` major — q1590/1591/1592/1593 docstrings spelled `a.T`, the
  page's only new symbol → reworded to "the transpose".
- `STMT_NEAR_MISS` — q1589/q1594 `why` named the answer's call (`clone`,
  `contiguous`); rewritten to describe the confusion (bank gate
  `answer_leak_in_wrong_example`).
- `EXPL_PREREQ` major — q1566 used a slice on torch.ranges (slicing-views not
  a prerequisite) → arithmetic on `arange`; q1576/78/79/80 used `Tensor.argmin`
  (page declares `torch.argmin`) → function form; q1579 dropped tuple
  unpacking; q1595 multi-axis index → chained index; q1598/1600/1603 prompts
  wrote `0..n-1` (parsed as an attribute) → "the values 0, 1, …, n-1";
  q1563/1566/1568 prompt comparisons (`step > 0`, `a < b`, `lo <= hi`)
  → prose. Remaining prereq findings re-recorded in the baseline are
  sibling-precedent classes (boolean-masking `syntax.compare`/`Tensor.any`
  = q1515/1516; ray `syntax.multi-axis-index` = q828; `make_rays_1d` in a
  prompt = q833; `Tensor.clone` on argmin-argmax = q1511).
- `DIFF_DUP` minor — q1581/q1582 were one program after normalisation
  (`stack` dim 0 vs 1, `audit_graph_structure.py`); q1582 rewritten as the
  interleave (`stack(dim=1).flatten()`), which is the third word of the KC.
- Symbol lock — q512/1521/1522/1523 rounded their answers through list
  comprehensions, which put them behind `python.control-flow`
  (`syntax.comprehension`); Seth has every python.* concept switched off
  EXCEPT control-flow, so the four drills were unservable and linalg-basics
  could never reach Integrated. Rounding dropped (the grader compares floats
  with tolerance).

Open, minor, left as authored:
- `DIFF_RUNG` minor — q1586 (`x[::k]`), q1588 (`flip`), q1567 (`x[x % 2 == 1]`),
  q1563 (`arange` with a step) are one-call Solo drills. Their rung siblings
  (q1503 `x[1::2]`, q1501 `x[-k:]`) sit at the same level; the Solo rung on
  these KCs is deliberately shallow because the faded rung is retired.
- `GIVE_STARTER` note — q1598/q1603 docstrings say "arange"/"linspace" the
  way their siblings q800–q808 do; the KC IS `out=` on those two calls.
- `STMT_LENGTH` — q1575, q1597, q1606 run to three sentences; within the bar.

## 2. Lesson pages

Only frontmatter and `### q<id>` framing lines changed; no concept prose was
touched. `validate_lessons.py` PASS (72 pages), `audit_ladder_pairing.py`
unchanged (48 pre-existing distance findings, not new). No `EXPL_*` finding.

## 3. Systemic

- The prereq audit reads prose code-spans as symbols: `` `a.T` `` and
  `` `0..n-1` `` in a prompt file as `undefined.a.T` / `Tensor.n`. Prompts on
  these KCs should say "the transpose" and "0, 1, …, n-1".
- List comprehensions in answers are a hidden lock for any learner who has
  not read python.control-flow. None of q1563–1608 uses one; q512/1521–1523
  no longer do. 104 older drills still do (`question_symbol_kcs.json`).

## 4. Top fixes

All applied above; nothing outstanding at major severity.
