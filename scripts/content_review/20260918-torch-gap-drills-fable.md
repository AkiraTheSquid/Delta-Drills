# Fable review — q1530–1554, torch.{broadcasting-rules,constructors,elementwise-ops,tensor-model}

Scope: the 25 Solo/Integrated drills authored 2026-09-18 to close the rung-dry
gaps the post-fix trajectory replay hit (409 at q132–201). Reviewed by the
in-session author against `scripts/CONTENT_RUBRIC.md`; the Astra half did NOT
run (codex usage limit until 2026-09-19 08:36) — re-run
`content_critic.py --qids 1530-1554 --pages Local_Deployed_Shared/lessons/tensors/kp-{broadcasting-rules,constructors,elementwise-ops,tensor-model}.md --label torch-gap-drills`
then and reconcile against this file. Self-review: the reviewer is the author.

## Drills

Pass (statement, cases, near miss, rung all checked by execution — every
answer passes its cases under `code_runner.run_function_tests`, every near
miss fails its anchor case, constant and `None` returns fail):
1530 1532 1533 1534 1535 1536 1537 1539 1540 1541 1542 1543 1544 1545 1546 1547
1548 1549 1550 1551 1552 1553 1554

Fixed during authoring (recorded so the fix is visible):
- 1531 `CORR_DECISIVE` major — first graded case expected `None`, so the
  placeholder starter passed it (`audit_question_bank`: degenerate_expected).
  Fix: `(3, 1) + (1, 4)` is case 0; the `None` case is second and the near
  miss anchors on it explicitly (`call: solve((3,), (3, 4))`).
- 1538 1539 1542 1543 1544 1550 1554 `GIVE_PROMPT`-adjacent (bank gate
  `answer_leak_in_wrong_example`) — the near-miss `why` named the answer's
  call in code form. Rewritten to describe the confusion without the symbol.
- 1543 `CORR_DECISIVE` major — a numeric `zeros_like` passed as the `False`
  mask (grader compares 0 == False). Fix: the drill also returns the mask's
  dtype name.
- 1532/1534/1535/1537 `EXPL_PREREQ` — fixtures used `t.ones`/`t.arange`,
  taught by KCs that are not prerequisites of broadcasting-rules. Rewritten as
  `t.tensor` literals.

Minor / notes:
- 1538 `GIVE_PROMPT` minor: "stored as 32-bit integers" names the dtype the
  learner must choose. Same wording as q645 ("64-bit INTEGERS"); kept, since
  the width IS the spec and the mechanical PROMPT_LEAK check passes.
- 1551 `STMT_TASK` note: "computed from the count and the number of rows"
  leans toward the implementation; kept because `len(rows[0])` would
  otherwise be a valid non-tensor answer.
- 1534 `DIFF_RUNG` note: one subtraction. It is a recognition drill (a
  `(b, 1)` column needs no `None`), and its sibling q501 is one operator too.

Ratchets re-recorded, with reason: `solution_prereq_baseline` +19 — every
entry is a class the sibling drills already carry (`syntax.multi-axis-index`
= the page's own `none-newaxis-indexing`; `syntax.try` is in the page's
worked example; `syntax.none`/`ternary` as q550/q566; `bool-literal` /
`compare` in prompt prose). `arena_grounding_baseline` +6
(`broadcast_shapes`, `equal`, `trunc`, `ceil`) — all in the pages'
`new_syntax`, same debt as q500/q631/q64.

## Lesson pages

No page prose changed. `kp-broadcasting-rules.md`'s legacy "Independent
practice" prose (which cited q60/q81, not in the frontmatter) became a
`## Solo practice` section with one line per item, matching the other three
pages. Framings were checked for `GIVE_PROMPT`: two rewritten (q1533 no
longer says "two Nones"; q1534 no longer says "needs no reshaping").
