# numpy.axis-reductions — Fable in-session review (2026-09-13)

Scope: q1363–q1373 and `Local_Deployed_Shared/lessons/numpy/kp-axis-reductions.md`.
Trigger: the content-gap watcher (`ops/gap-watch/`) fired at 11:42 CDT — Seth's
account had answered every Solo drill the queue could reach on this concept
(q174, then the unranked q103) and the rung was dry. Before this pass the page
owned 2 Faded, 2 Guided, 3 Solo, 0 Integrated. After: 6 Faded (3 per segment),
2 Guided, 7 Solo, 3 Integrated — every rung at or above the floors
(Faded ≥ 2 per segment, Solo ≥ 6, Integrated ≥ 3).

Reviewer: Claude Fable 5.1, the author, grading against `scripts/CONTENT_RUBRIC.md`.
The GPT-6 Astra run (`content_critic.py`) hit the Codex usage limit at 12:00,
re-ran at 13:06 and landed as `20260913-axis-reductions-astra.md`; it is
reconciled at the end of this file, and every fix it drove was re-executed
and re-audited before deploy.

## Method

Every answer, every grader case and every near miss was EXECUTED under the
backend venv (`scratchpad/build_axis_drills.py`): `expected_expr` is the repr
of what ran, the near-miss `output` is what the wrong implementation actually
produced on that case, and the build asserts (a) no two cases share an expected
value, (b) the `return None` starter fails every case, (c) the near miss
disagrees with the answer on its case. `audit_question_bank.py --gate`: GATE
PASS, zero findings on 1363–1373. `validate_lessons.py`: PASS.
`audit_solution_prereqs.py`: 0 NEW violations (no drill needs a symbol the
lattice has not taught; `amax`/`amin`/`argmax` were deliberately avoided
because no page before this one declares them — `Tensor.max` on a whole
tensor, from `numpy.aggregations`, is what q1370 uses instead).

## Drills

| id | rung | move | verdict | notes |
|---|---|---|---|---|
| 1363 | Faded s1 | `sum(dim=-1)` on (n, r, c) | pass | transfer from the 2-D example: three axes, last one crossed out |
| 1364 | Faded s1 | `mean(dim=0)` on (n, r, c) | pass | `audit_ladder_pairing` "distance" note (tokens all present in example) — same note as the pre-existing q220/q135; the decision (which axis) is not in the example |
| 1365 | Faded s2 | `mean(dim=0, keepdim=True)` | pass | axis flipped vs the example's `mean(dim=1, keepdim=True)`; same "distance" note |
| 1366 | Faded s2 | `mean(dim=(-2, -1), keepdim=True)` | pass | combines the two things the segment example shows separately; was `sum` first — changed to `mean` because the page never shows `sum#keepdim` (coverage finding, now clear) |
| 1367 | Solo | `mean(dim=0) - mean()` | pass | per-axis beside whole-tensor; case 1 changed to a 2-row input so every expected value is dyadic |
| 1368 | Solo | `mean(dim=(1, 2))` | pass | near miss `dim=0` returns (h, w) — shape, not value, is the tell |
| 1369 | Solo | `x - x.mean(dim=1, keepdim=True)` | pass after fix | case 1 WAS the segment-2 worked example's `x` and `centered` (GIVE_EXAMPLE, Astra D2) — now `[[2, 8, 5], [4, 1, 1]]`; near miss stays on the SQUARE case 2 and its `why` now names that input (D3) |
| 1370 | Solo | `float(x.sum(dim=0).max())` | pass after fix | chained reduction; DIFF_RUNG: two decisions (axis, then none) — Solo, not Faded. STMT_CASES (Astra D5): cases were all non-negative with no size-1 axis; added `[[-3], [-2]] → -5.0` and `[[-4, -1, -7]] → -1.0` (near miss gives -2.0 / -12.0 there, so both stay decisive) |
| 1371 | Integrated | row shares then column mean, both keepdim | pass | two keepdims doing different jobs; square first case so the near miss (no keepdim on step 1) runs and is wrong |
| 1372 | Integrated | `imgs / imgs.sum(dim=(1, 2), keepdim=True)` | pass | near miss = batch-wide total; case 1 totals chosen (8, 8) so both outputs are exact binary fractions |
| 1373 | Integrated | three reductions, tuple return | pass | `overall` equals 3.0 in two cases — the tuple still differs in every case (STMT_CASES holds on the whole return) |

Codes checked per drill: STMT_TASK / STMT_INPUT / STMT_OUTPUT (every prompt
names dtype, rank, shape in and shape out), STMT_SELF_CONTAINED (no
cross-references), STMT_CASES (3–5 cases each, expected values differ; q1370 had
NO size-1 axis and no negative until Astra D5 — two cases added), STMT_NEAR_MISS (executed, `why` names the confusion),
STMT_LENGTH (1–2 sentences), GIVE_PROMPT (no `dim=`/`keepdim` values named —
in the BANK prompts; the page blurbs did leak, see Astra L4 below),
GIVE_STARTER (docstrings describe the goal; page starters blank `mean`/`sum`,
`dim` and `keepdim` — every `new_syntax` symbol), CORR_RUNS / CORR_DECISIVE
/ CORR_TYPES (executed; lists vs lists, floats exact-representable).

### Minor (queued, not blocking)

- `GIVE_PROMPT` minor, q1365/q1366: "kept at length 1" / "not a 1-D tensor"
  describes the keepdim *effect*. It names no keyword, and the shape IS the
  task, so left as written. Rewrite if Astra also flags it: "return … with
  shape `(1, c)`" and stop there.
- `DIFF_LABEL` note: q1367 exports as `easy` (score 34) while its Solo
  siblings are `medium`; the score reflects the single-call solution. Harmless.

## Lesson page

Concept prose unchanged. Both worked examples rewritten after Astra L1–L3
(see reconciliation): each now opens in prose; segment 1 demonstrates row
totals + column means so q220's column sum is a transfer (and its expected
`[11, 22, 33]` no longer appears on the page); segment 2 demonstrates a
LEADING-axes tuple `sum(dim=(0, 1))` → `[[60, 66], [72, 78]]`, split from the
keepdim fence by a bridge sentence, so q135's trailing-axes total is a
transfer (its arange(24) expected output no longer appears either). Every
new assert value was executed under the venv before it was written. Added: `### q1363`/`q1364` after q220 (segment 1),
`### q1365`/`q1366` after q135 (segment 2), `## Solo practice` replacing the
prose-only `## Independent practice` (same three bank drills, now `### q` items
so `compile_lessons.py` serves them with a blurb, plus the four new ones), and
`## Integrated practice`. `EXPL_TERMS`: the first draft's blurbs said "stack",
which `numpy.stack-concat-interleave` defines two pages LATER — the prose
prereq ratchet caught it; reworded to "n matrices, one behind the other".

## Reconciliation with Astra

Astra: 6 major / 1 minor / 4 pass on the bank, lesson major. Every finding
was checked against the compiled output and by execution, not taken on trust.

**Agreed, fixed (Astra-only findings, verified):**

- D2 `GIVE_EXAMPLE` q1369 — verified: case 1 and the starter's example run
  were the worked example's `x` and its `centered`. Case 1 → rectangular
  `[[2.0, 8.0, 5.0], [4.0, 1.0, 1.0]]` (kept rectangular on purpose: the near
  miss must still RAISE there and only run silently on the square case 2).
- D3 `STMT_NEAR_MISS` q1369 — `why` now names `[[1.0, 3.0], [6.0, 2.0]]` and
  both outputs.
- D5 `STMT_CASES` q1370 — verified: shapes (2,2), (2,3), (3,3), all ≥ 0. My
  own "size-1 axis in every set" claim above was wrong for this drill. Two
  cases added, executed, near miss disagrees on both.
- L1 `EXPL_INTRO` — verified: both examples opened with a fence (the
  mechanical INTRO check reads `## Worked example` sections and did not fire
  because the `Why:` paragraph sits AFTER the fence). Intro prose added.
- L3 `EXPL_TRANSFER` — verified and worse than Astra said: q220's
  `sum(dim=0)` on the same matrix and q135's `sum(dim=(-2, -1))` on the same
  `arange(24)` batch meant the page PRINTED both drills' expected outputs.
  Examples changed as described under "Lesson page".
- L4 `GIVE_PROMPT` blurbs — q1363/q1364/q1371/q1372 (mine) and q108/q135/q505
  (pre-existing) reworded to shapes and goals; q1366 too ("two axes in one
  call" named the tuple move). Guided q503/q504 keep their step-3 answers —
  that rung is defined as hints-then-answer.

**Agreed in substance, already true in the served product (D1, partly):**
Astra read the SOURCE page, where the faded starters showed `dim=`/`keepdim=`.
`compile_lessons.py` runs `blank_new_syntax` on every faded starter, and the
compiled page already served `x._____(_____=_____)` /
`x._____(_____=_____, _____=_____)` (`Tensor.mean#dim` etc. are keyword
symbols, so the keyword NAME is what gets blanked). The source starters for
all six faded slots now match the compiled ones so a reviewer sees what the
learner sees. The other half of D1 — "bank starter is `return None`" — is the
pipeline's design: the bank row is the Solo/Integrated form and the page
fence wins whenever the drill is served at the Faded rung (same as q220/q135
before this pass). Not a defect.

**Disputed, left as is:**

- L2 `EXPL_INTERLEAVE` as measured — Astra counted 18 and 22 lines by
  including blank lines and the `# Hidden checks` tail; the rubric's
  mechanical check counts VISIBLE non-blank lines (13 and 14, max 16). Segment
  2 was split anyway because its two halves are two different ideas; segment
  1 stays one 11-line fence.
- D4 `DIFF_RUNG` q1368 (minor) — Astra wants it moved to Faded because faded
  q1366 also needs keepdim. Solo means no scaffold: the learner has to produce
  `imgs.mean(dim=(1, 2))` from a blank body, choosing the tuple AND the two
  axes, where q1366's scaffold hands over the call shape. q1368 is the easiest
  Solo item, which is what a first Solo should be. Label stays `medium`.
- `audit_ladder_pairing` "distance" notes on q220/q1364/q1365 — token-level
  ("every expression already appears"); the decision each drill asks for (the
  axis, on a different rank) is not in the example. Pre-existing category of
  note, not a gate.

**Fable-only:** none beyond the minors listed above; both stay queued.

After the fixes: `validate_lessons.py` PASS, `audit_question_bank.py --gate`
GATE PASS (0 findings on 1363–1373), `audit_solution_prereqs.py` 0 NEW,
`audit_prose_prereqs.py` 0 new, `audit_graph_structure.py` 0 new,
lessons / numpy / scripts / ops watchers exit 0, and the exported q1369/q1370
answers re-executed against every case (5 cases on q1370, all distinct).
