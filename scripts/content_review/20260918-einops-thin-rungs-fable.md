# Content review — einops / torch thin-rung drills q1609–1653 (Fable, in-session author)

Scope: 45 drills filling the rungs the static rank-count audit found thin
after the ray-scope batch (Solo < 8 or Integrated < 4 on a concept Seth has
switched on): einops.dl-flatten-heads (+5 Solo, +4 Integrated),
channel-groups-temporal (+2/+4), singleton-and-lists (+2/+4),
patches-space-depth (+0/+4), reduce-model (+2/+3), pooling (+4/+0),
grids-montage (+1), split-axes (+1), pattern-language (+0/+1),
torch.dot-matmul-patterns (+3/+2), torch.sorting (+2/+1). Five of the einops
pages had NO Integrated rung before this batch. Reviewed against
`scripts/CONTENT_RUBRIC.md` by the AUTHOR of these drills — a self-review,
not an independent one. GPT-6 Astra's pass is owed: codex quota exhausted
until 2026-09-19 08:36; rerun
`scripts/content_critic.py --qids 1609-1653 --label einops-thin-rungs` then.
einops.attention-einsum (3 Solo / 2 Integrated) was left alone: another
session authored it this evening.

## 1. Drills

Pass on every A/B/E code, verified mechanically: every answer executed under
the real harness (`code_runner.run_function_tests`, torch + einops
preloaded), every case set disagrees, every near miss executed and differs
from the correct output on its anchored case, constant and `None` answers
fail (`bank_audit_report.json` GATE PASS, `validate_lessons.py` PASS 72
pages, `audit_graph_structure.py` 0 new, `audit_arena_grounding.py` 0 new).

Fixed during this review (all applied before commit):
- `EXPL_PREREQ` major — q1624/1626/1628/1629 built their list-of-tensors
  input with a list comprehension, which sits behind `python.control-flow`
  (the symbol lock documented on the last batch). The inputs are now lists
  of TENSORS built by the grader's setup, so the answer hands the list to
  `rearrange` directly — which is the page's actual point ("a Python list as
  the first axis"). q1617 used tuple unpacking (`c, h, w = …`, first taught
  by ray-parametrisation) → three assignments. q1649 used
  `Tensor.transpose` (taught by no page) → `x.T`, transpose-axes' declared
  symbol and a prerequisite of dot-matmul-patterns.
- `EXPL_PREREQ` — q1637 (weekly std via a callable aggregation) needs
  `torch.std`, which reduce-model's concept text shows but its frontmatter
  did not declare → `new_syntax: [einops.reduce, torch.std]`; the
  drilled-twice check then wanted a second trainer → q1653 (per-feature std,
  Solo). q1650's `descending=True` is `syntax.bool-literal` with a sibling
  precedent on the same KC (q664) → re-recorded in the baseline.
- `STMT_NEAR_MISS` — q1651 `why` named `argsort` (bank gate
  `answer_leak_in_wrong_example`) → "the other half of the pair the sort
  hands back". q1636's near miss (`b c` vs `c b`) coincided with the answer
  on a 2×2 case → anchored on the 1×3 case; q1643's near miss (`'mean'` on an
  int tensor) raised instead of differing → packing-order near miss.

Open, minor, left as authored:
- `GIVE_PROMPT` note — the einops prompts describe the packing order in
  words ("head index slowest", "group index fastest") and never spell the
  pattern; the pre-existing base drills on these pages (q384, q394, q395,
  q399) DO quote their pattern in the prompt. Not touched here — they are
  not this batch's rows — but they are the weakest items on the pages.
- `DIFF_RUNG` minor — q1624 (identity pattern on a list) and q1650
  (`descending=True`) are one-call Solo drills; deliberate, both are the
  page's first-contact skill.
- `STMT_LENGTH` — q1614, q1616, q1631, q1633 run to three sentences with a
  packing-order clause each; within the bar, and the packing order IS the
  content.

## 2. Lesson pages

Frontmatter and `### q<id>` framing lines added; five pages gained an
`integrated:` key and a `## Integrated practice` section (dl-flatten-heads,
channel-groups-temporal, singleton-and-lists, patches-space-depth got
theirs new; reduce-model's existed). reduce-model's `new_syntax` gained
`torch.std`. No concept prose changed. `validate_lessons.py` PASS,
`audit_ladder_pairing.py` unchanged (48 pre-existing distance findings).

## 3. Systemic

- A list comprehension in an answer is a control-flow lock; on einops pages
  the honest fix is to give the learner tensors, not to convert lists.
- Pages with an empty Integrated rung were served Solo drills at the
  Integrated stage (rung (3,2,4) falls to rank 2) — the learner never met a
  harder problem on five einops concepts. That is what this batch fixes.

## 4. Top fixes

All applied above; nothing outstanding at major severity.
