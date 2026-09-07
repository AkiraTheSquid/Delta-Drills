# Content rubric — problems and lesson explanations

The bar, in Seth's words (2026-09-07): problems "aligned with the quality that
you would get for a LeetCode problem", explanations that "make sense", and
nothing that "gives away the answer". This file is the single written standard.
Three things read it:

- `lesson_quality.py` — the rules that can be checked mechanically (INTRO,
  INTERLEAVE, PRINTS, CASES, GIVEAWAY, FADE_LEAK, PROMPT_LEAK). Every code below
  that has a mechanical check names it.
- `content_critic.py` — launches ONE agentic GPT-6 Astra review (`codex exec`,
  effort pinned to xhigh); Claude Fable 5.1 reviews the same task in-session.
  Two reviewers, two processes at most, both read this rubric verbatim and
  grade the parts no regex can: whether a prompt is unambiguous, whether an
  explanation actually explains, whether a case set is decisive.
- Whoever authors content (`drill-gaps` skill, `AUTHORING.md`). Author against
  it; do not wait for the critic to tell you.

Each item has a CODE, a severity, and a one-line test. `major` blocks a
deploy of that item; `minor` is a queued fix; `note` is advisory.

## A. Problem statement — the LeetCode bar

A LeetCode problem is solvable from its statement alone: the task, the input,
the output, the constraints and at least one worked input/output pair are all
on the page, and none of them hint at the implementation.

| Code | Sev | Test |
|---|---|---|
| `STMT_TASK` | major | The first sentence says what the function must DO, as a goal ("lay the images side by side"), not as an implementation ("rearrange with pattern …"). |
| `STMT_INPUT` | major | Every argument's type and shape/layout is stated or unambiguous from the starter: "a batch `(b, c, h, w)` as a nested list", "a 1-D float tensor of length n". A learner must never guess the rank or the axis order. |
| `STMT_OUTPUT` | major | The return type and shape are stated: "return the nested list, shape `(c, h, b·w)`". A function that returns a tensor in one case and a list in another is a defect. |
| `STMT_SELF_CONTAINED` | major | No reference the learner cannot see. "Variant of exercise (1)", "as in the previous problem", "same grid as above" are all defects — the drill is served alone, in any order. |
| `STMT_TERMS` | minor | Every non-obvious term is defined or shown ("side by side" = batch goes into WIDTH; "tiled" = rows × cols grid with the row index slow). Bold is fine, jargon-without-definition is not. |
| `STMT_CASES` | major | The graded cases (`test_cases`) DISAGREE on their expected output, include at least one edge case (size-1 axis, single image, empty where meaningful) and together show the SHAPE of the mapping, not one coincidence. Mechanical: `lesson_quality.check_pairing` (CASES). |
| `STMT_NEAR_MISS` | minor | `wrong_examples` carries one executed near miss whose `why` names the confusion, and whose output actually differs from the correct one on that input. |
| `STMT_LENGTH` | minor | The prompt is one to three sentences. Seth: "they can't be too low quality and they can't be too long." Constraints go in a short list, not a paragraph. |
| `STMT_VOICE` | note | Consistent imperative voice ("Return …", "Compute …"), consistent notation for shapes (backticked tuples), no trailing "Good luck!"-style filler. |

## B. No giveaway

"It shouldn't give away the answer either." A drill passed by transcription is
evidence of nothing, and the ladder promotes on it.

| Code | Sev | Test |
|---|---|---|
| `GIVE_PROMPT` | major | The prompt does not spell the call, the pattern string, the `dim=`/`axis=` value or the dtype the learner is supposed to choose. Describe the GOAL, not the move. Mechanical (torch symbols only): `check_prompt_leak` (PROMPT_LEAK). Model reviewers extend this to einops pattern strings, numpy calls and keyword names. |
| `GIVE_STARTER` | major | The starter's docstring, comments and variable names do not name the move ("""use rearrange with (b w)"""). A faded starter blanks every symbol the KP's `new_syntax` declares and keeps the scaffold: `z.__(__)`, never `z.clamp(_____=0.0)`. Mechanical: `check_fade_leak` (FADE_LEAK). |
| `GIVE_EXAMPLE` | major | The worked example that precedes a drill does not reproduce the drill's expected output or build from the drill's own input literals. Mechanical: `check_pairing` (GIVEAWAY). |
| `GIVE_TESTS` | minor | The visible `input → expected` rows are informative but not a lookup table: two cases with the same input and different outputs, or an expected value the learner can return as a constant, are defects. |
| `GIVE_RUNG` | major | The rung matches the aid on screen. Faded = scaffold with blanks and NO example. Solo = from scratch, an example only when the schedule pops one. Integrated = never an example, and the item needs more than one idea of the KP. |

## C. Explanation quality — the lesson page

"Make sure that the explanations make sense." An explanation makes sense when a
learner who has never seen the concept can predict the next line of the worked
example before reading it.

| Code | Sev | Test |
|---|---|---|
| `EXPL_GENERAL_FIRST` | major | `## Concept` states the general procedure BEFORE any example (Seth: never example-only). The rule, then the instance. |
| `EXPL_INTRO` | major | The worked example opens in prose saying what it is about to show. Mechanical: INTRO. |
| `EXPL_INTERLEAVE` | major | Code arrives in blocks of at most 16 lines with prose between them. Mechanical: INTERLEAVE. |
| `EXPL_WHY` | major | Every non-trivial step is followed by WHY it is that step — the comment or the "Why each step" list explains the decision, not the syntax ("`b` moves inside `(b w)` because batch must be the SLOW index of the merged width"). |
| `EXPL_PRINTS` | minor | A block that asserts also prints. Mechanical: PRINTS. |
| `EXPL_CORRECT` | major | Every factual claim is true and every fence runs and produces the values the prose says. A wrong shape in the prose is worse than no prose. |
| `EXPL_TERMS` | minor | Terms are used the way `lessons/glossary.js` defines them and no term appears before the page that defines it (`audit_prose_prereqs.py`). |
| `EXPL_PREREQ` | major | Nothing in the example or the drills requires a symbol the page (or an earlier KP) has not shown. Mechanical: `audit_ladder_pairing.py` coverage, `audit_solution_prereqs.py`. |
| `EXPL_LENGTH` | minor | A segment's concept prose is 80–250 words. Longer is a chapter, shorter is a caption. |
| `EXPL_TRANSFER` | major | The faded drill after a segment requires TRANSFER: the same idea on a different input or a different axis, not the example with the numbers changed. |

## D. Difficulty alignment

| Code | Sev | Test |
|---|---|---|
| `DIFF_RUNG` | minor | Faded < Solo < Integrated in the number of decisions the learner makes. A Solo drill that is one call with one obvious argument belongs on Faded; an Integrated drill that needs one idea belongs on Solo. |
| `DIFF_DUP` | minor | No two drills on the same rung of the same KP are the same move (normalized-AST duplicate; `audit_graph_structure.py`). |
| `DIFF_LABEL` | note | `difficulty_label` is consistent with the rung and with the sibling drills. |

## E. Correctness of the grading contract

| Code | Sev | Test |
|---|---|---|
| `CORR_RUNS` | major | `answer_code` runs top to bottom in the backend runtime and passes every case. |
| `CORR_DECISIVE` | major | A constant return, a no-op, and the near miss all FAIL at least one case. |
| `CORR_TYPES` | major | `expected_expr` compares like with like: lists to lists, floats with a tolerance, tensors converted before comparison. |
| `CORR_VISUAL` | minor | An image drill (`expected_artifact_type: image`) is written against axis NAMES so the real-image preview (`practice/visual-fixture.js`) can substitute the 120×120 digits without changing the batch or channel sizes. |

## How the reviewers report

Each reviewer returns, per item, `verdict` (`pass` / `minor` / `major`), a
list of `issues` (`code` from the tables above, `severity`, the exact `quote`
that triggered it, and a concrete `fix` — a rewritten sentence, not "be
clearer"), and optionally a full `rewrite` of the prompt. For a lesson page the
same shape, keyed by segment heading.

The author reconciles the two reports in `scripts/content_review/`: an issue
both raise is `agreed`; one raised by a single reviewer is `fable-only` or
`astra-only` and is verified against the data before it is acted on. Neither reviewer edits content.

## History

- 2026-09-07: written. Supersedes the rules list in `AUTHORING.md` ("Rules")
  and the quality paragraph in the `drill-gaps` skill, both of which now point
  here. Triggered by the 0.0 einops set (q847–q940), where prompts opened with
  "Variant of exercise (1)" and a friend watching a session saw a matrix where
  an image should have been.
