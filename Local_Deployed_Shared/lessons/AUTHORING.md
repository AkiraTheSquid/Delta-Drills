# Lesson authoring format

One markdown file per knowledge point (KP): `lessons/<topic>/kp-<slug>.md`.
Registry of KCs/lessons: `kc_registry.json`. Compile: `python3 scripts/compile_lessons.py`.
Validate: `python3 scripts/validate_lessons.py [--coverage]` — executes every code fence
and grades every faded solution against the drill bank's test cases.

## File format

```markdown
---
kc: numpy.broadcasting-rules        # must exist in kc_registry.json
title: Broadcasting rules
supporting: [numpy.ndarray-model]   # KCs used but not taught here
new_syntax: []                      # symbols this page is the lesson for
previews: []                        # symbols shown here but taught LATER, on purpose
concepts: [repeat-elements]         # stable id per atomic segment, in order
faded: [111, 151]                   # STAGE 2 — fill-in-the-blank drills, no example
guided: []                          # legacy rung, folded into faded; empty on new KPs
independent: [144, 84]              # STAGE 3 — solo drills, a few open with an example
integrated: [560, 561]              # STAGE 4 — solo, all ideas at once, never an example
---

## Concept
Prose teaching the general idea/procedure. Plain ```python fences here are
EXECUTED by the validator and become RUNNABLE NOTEBOOK CELLS for the learner —
those are the same thing, so break the prose up with as many small fences as the
explanation has moving parts. Cells within a segment share state top to bottom,
but each SEGMENT starts fresh (it is its own page), so every segment repeats its
own imports. Use ```python no-run for fences that intentionally error or are
pseudocode: those are illustrative only and get no Run button.

Write fences that show something — print, end in a bare expression, or `assert`
a claim worth proving. Nothing is "just setup"; a silent cell reports the names
it bound, but a cell that demonstrates its point is better.

## Watch out
- **Short trap name** — concept-specific misconception and correction.

## Worked example
Heavily commented code + step-by-step walkthrough. Exactly one plain ```python
fence; it is executed and runnable like the Concept ones, and it continues the
same namespace.

## Faded practice
### q111
One-line framing of the task (shown above the starter).
```python starter
# partially completed code with _____ blanks the learner fills in
```
```python solution
# the completed version — MUST pass q111's bank test cases
```

## Solo practice
### q144
One-line framing of the task. No starter and no blanks — the learner writes
the whole function. MOST items here carry no example. A MINORITY open with one,
and only when the drill needs an idea the lesson page never showed:
```python worked
# a small commented demo of the new use; the learner reads it,
# then writes the drill from scratch. Optional, and rare on purpose.
```

## Integrated practice
### q560
One-line framing. Stage 4 items NEVER carry a ```python worked``` fence.

```

## The four stages (2026-08-28)

A learner meets one concept across FOUR stages, and what changes between them is
how much of the answer is on screen. Seth set this out while testing the ndarray
KP; the whole rung system exists to produce it.

| Stage | Rung id (stored) | What is on screen | Authored as |
|---|---|---|---|
| 1 Lesson | `worked` | The lesson page: prose, several small runnable fences, one worked example per segment | `## Concept` + `## Worked example` |
| 2 Faded | `faded` | The problem on the left, a mostly-written solution with `_____` blanks on the right, **and no example** | `## Faded practice` |
| 3 Solo | `partial` | Write the whole thing. **Some** items open with a worked example that introduces a new use of the concept; most do not | `## Solo practice` |
| 4 Integrated | `solo` | Every idea in the concept at once, **never** an example | `## Integrated practice` |

The stored rung ids are frozen — every attempt ever recorded is filed under
`worked/faded/partial/solo` and the promotion bound reads them back, so they are
indices, not labels. Only the meanings and the on-screen names changed.

Two consequences for the person writing content:

- **Stage 2 shows no example, so stage 1 has to carry it.** The learner has
  already read the lesson; the faded drill is the same idea from memory. Putting
  the worked example beside a faded starter is what made q484 unpassable-by-
  thinking — its example printed the two values the blanks asked for.
- **The stage 3 → 4 fade is a SCHEDULE now (2026-08-30), and content feeds it.**
  The server decides when a worked example pops up in front of a drill
  (`backend/app/example_schedule.py`: Faded only after an unaided miss, Solo
  at positions 0/2/5/9, Integrated once on entry — see
  `This-Directory-Only/SPEC_WORKED_EXAMPLE_SCHEDULE.md`). What the popup SHOWS
  is authored here: the drill's own ```python worked``` fence when it has one,
  else the worked example of the segment that owns it. So a Solo item's fence
  is still worth writing for a NEW use of the concept — it is what the learner
  reads when the schedule picks that drill — but nothing counts examples down
  by how many items carry one. Stage 4 items never carry a fence; the popup
  there shows the segment's example, once.
- **A drill is never served twice**, and a rung whose drills are all spent
  reaches DOWN a rung, then reports `content_exhausted`. So the floor per
  concept is a real number, not a wish: **Faded ≥ 2 per segment, Solo ≥ 6,
  Integrated ≥ 3**, and no two drills the same move on the same rung.

### Nothing may require a symbol the page has not shown (required)

A drill may only ask for moves the learner has already seen — Seth: "make sure
to do checks like for finding whether something was used before … it doesn't
introduce requiring you to do something that you haven't used before with a
function that you haven't seen before."

The rule is per KP and cumulative down the page: a drill's solution may use a
symbol declared in this KP's `new_syntax`, shown in any earlier segment's
`## Concept` or `## Worked example`, or taught by an earlier KP. Anything else
is a new concept smuggled into practice.

`audit_ladder_pairing.py` measures this as COVERAGE, scoring each drill against
everything the page has shown UP TO that segment — concept fences included, not
just the worked example, because the learner reads both. A symbol that a drill
needs and the page never shows is a finding; the fix is to show it in the
lesson, not to weaken the drill.

### Every drill states its input and output (required)

Under the prompt the learner sees the graded cases as `input → expected output`
rows, plus any authored near miss as an `input ✗ wrong output` row with a
sentence saying what the mistake was (`practice/question-examples.js`).

- The CORRECT rows need no authoring at all: they are read straight off
  `test_cases[*].call` and `test_cases[*].expected_expr`, which is the same data
  the grader compares. What you owe them is decent CASES — vary the expected
  value across them so a constant-return answer cannot pass, and so the rows
  show the learner the SHAPE of the mapping rather than one coincidence.
- A near miss is authored in the override record's `wrong_examples`, as
  `[{"call": ..., "output": ..., "why": ...}]`. Author the OUTPUT by RUNNING a
  wrong implementation on one of the drill's own grader inputs — never by hand.
  A hand-written wrong value is a guess about what a mistake produces, and half
  the time the mistake produces something else. Choose an input on which right
  and wrong actually disagree; a misconception that passes every case is not a
  near miss, it is a second correct answer.
- The `why` is the part that teaches. One sentence, naming the confusion
  ("`.ndim` counts axes; `.numel()` counts elements"), not "this is wrong".
- Omit `call` to reuse the first graded case's input, so both blocks describe
  the same call.

## Which concept to work on (scope rule, 2026-08-28)

Content work is scoped to the ONE concept Seth is currently practising, and on
that concept it covers **every rung of the ladder** — Lesson, Faded, Solo,
Integrated. Not a rung, not a sample across several KPs: one node, the whole
climb. He sends feedback per concept to the session working on that concept, so
a session that widens its scope is answering feedback it was never given.

Read his position rather than guessing it — his account is
`sethbgibson@gmail.com` (backend `user_id` `c813fa78-7e0f-4859-bcb3-a2183ef98eb4`),
and the recipe for reading the `kc` he is on is in the repo's `CLAUDE.md` under
"Content work: ONE concept at a time". That file also tracks which concepts have
had the four-stage treatment and which have not — as of 2026-08-28 only
`numpy.ndarray-model` has; PyTorch, `einsum` and `einops` have not, and are next.

## Segments — ONE concept at a time (required)

A KP is a sequence of single-concept SEGMENTS. Each `## Concept` heading starts
a new segment; its `## Worked example` and `## Faded practice` belong to that
segment. A segment heading may carry a subtitle: `## Concept: np.trace`.

- `new_syntax:` is a CLAIM, not a tag list: it says this page is where a
  learner is taught that symbol. Spell entries the way
  `audit_lesson_syntax.py` reports them — `torch.zeros`, `Tensor.item`,
  `torch.stack#dim` (a keyword is its own thing to learn), `einops.rearrange`,
  `syntax.slice`. Free text (`slice-notation`) matches nothing and declares
  nothing. Run `audit_lesson_syntax.py --suggest` for a starting list, then
  decide each one: the default owner is the first page that shows it, and a
  later page wins only if the symbol is what that page is ABOUT. Plain Python
  belongs in the script's ASSUMED set instead.
- `previews:` is the escape hatch for a symbol shown BEFORE its lesson on
  purpose — the contrast demos that lose their point without the second
  half (`*` beside `@`, elementwise `t.maximum` beside the `max` reduction).
  It exempts that use from `audit_lesson_syntax.py`'s "shown before it is
  taught" list and reports it on its own line instead. It is a claim, not a
  mute button: `validate_lessons.py` requires the page to actually show the
  symbol, to NOT declare it, and some LATER page to declare it. If a demo can
  be rewritten to avoid the forward reference without losing the teaching,
  rewrite it — reach for `previews:` only when the forward reference IS the
  lesson.
- Each segment teaches exactly ONE new idea (one function, one mode, one rule).
  If the prose says "also" or introduces a second API, split the segment.
- `concepts:` declares one stable id per reviewed atomic segment. Its count
  MUST equal the number of segments, ids MUST be unique, and every declared
  segment MUST have a `## Concept: ...` title. This makes accidental re-merges
  fail validation and supplies concept-level ids for adaptive sequencing.
- **A segment id is stored state, not a label.** The learner is taught and
  drilled ONE segment per visit, and what remembers which ones are done is
  `<kc>#<concept_id>` in their exposure map. An id that changes re-teaches that
  concept once; an id that gets REUSED for different prose credits the new
  concept as already read and it is never offered again. `compile_lessons.py`
  fills in `s<index>-<title-slug>` for segments `concepts:` does not name, so
  re-titling or reordering an unnamed segment costs one re-read — declare the
  id here when a segment's title is likely to be edited.
- Each segment's `## Worked example` shows exactly ONE worked example — a
  single small demo of the one idea, not a tour of variations. The rhythm is:
  teach one concept → inspect or optionally run one worked example → continue.
  Extra variations belong in practice, not lesson screen.
- Each segment MUST have at least one faded exercise (validator-enforced).
  There is no ceiling. Several is a FADING SERIES, not a pile of drills: the
  first sits adjacent to the worked example, each later one asks for the same
  idea one step further out, so the blank cannot be filled by transcription.
  `audit_ladder_pairing.py` measures that step ("series never reaching
  distance"). The ceiling used to be two, and that was the content-side cause
  of the ladder repeating itself — with three faded items on a KC the queue ran
  out inside a single sitting and re-served what the learner had memorised.
  Write as many as the idea has surfaces.
- Faded exercise is downstream practice metadata. LessonGate does NOT render or
  grade it immediately after teaching.
- `## Watch out` is optional segment content. It renders only inside that
  concept's lesson screen.
- In-app sequence is fixed: teaching + worked explanation on left, complete
  worked code preloaded on right for optional running/editing, then next
  concept or normal question queue. No popup; no faded exercise inside lesson.
- Solo/Integrated stay KP-level, after the last segment — they are about the
  whole KP, not one segment. `## Misconceptions` remains a legacy fallback for
  single-segment KPs; new/multi-segment content uses `## Watch out` inside each
  relevant segment.
- A faded qid may appear in only one segment.

### The blanks must cover the concept, never the scaffold (required, enforced)

A faded starter is a completion problem, and what makes it one is that the
STRUCTURE is given and the CONCEPT is not. `z.__(__)` says a method call on the
tensor taking one keyword argument — which is a great deal of help, all of it
about a shape the learner has already met — while saying nothing about which
call. That is the target shape.

The failure is the other way round: blanking the argument and leaving the
method. `return z.clamp(_____=0.0)`, on the KP whose whole subject is `clamp`,
is a drill that can be passed by anyone who can read, and the ladder promotes
on it. That was q67, and it is what this rule exists to stop.

So: **every symbol in the KP's `new_syntax` frontmatter must be blanked, and
everything else may stay.** A symbol an earlier lesson taught is exactly the
supporting structure that belongs on the page — the rule is about what is NEW
here, not about hiding as much as possible.

You do not have to do this by hand. `compile_lessons.py` runs
`lesson_lib.blank_new_syntax` over every authored starter, so a `new_syntax`
symbol left visible is blanked on the way into `lessons_structured.json`, in
the function body only (the example-run block below it keeps its `t.tensor(…)`
fixture). What the rewriter cannot do is operators — `syntax.matmul` has no
identifier to hide, and a `@` replaced by a blank leaves a line that cannot be
read as code. Those are reported as FADE_LEAK by `lesson_quality.py` and are
an authoring decision.

The same rule scales up. On a larger problem the scaffold gives away every
supporting step and blanks only the one move being learned plus the arguments
it needs — the point is not how much is hidden but that what is hidden is
exactly what is being taught.

### The faded exercise must require TRANSFER (required)

The faded exercise must NOT be a re-run of the worked example. It teaches the
SAME concept, but on a different surface — enough of a twist to make the
learner *adapt* the idea instead of copying tokens. A faded that just swaps
the literals for the same operation is too easy and defeats the point.

Concretely:
- Pick a bank question that applies the concept to a slightly different task
  than the worked example (different parameters exposed, opposite direction,
  a variant), OR write the worked example on instance A and let the faded be
  instance B.
- Blank the ONE spot that carries the concept's key insight — not a trivially
  copyable token. Good blanks: the `+ 1` that makes a stop inclusive, the
  `axis=` that must flip, the offset that shifts a cut. The learner should
  have to *reason*, not pattern-match.
- Calibrate: different enough to trip slightly, close enough to be solvable
  from the lesson alone. Near-transfer, not a new concept.

Example (numpy.ranges): the worked example shows `np.arange(0, 10, 2)`
(exclusive stop); the faded asks for an *inclusive* integer range, blanking
`end + 1` — the learner must apply "stop is exclusive" themselves rather than
copy the demo. See `numpy/kp-ranges.md` for the pattern to copy.

Note: for a few atomic KPs the skill genuinely IS recalling one exact token
(e.g. an einsum spec string like `'ii->i'`) — there, a fill-from-memory faded
is legitimate near-transfer. Everywhere with a procedure, demand a twist.

Example skeleton:

```markdown
## Concept: np.trace
...one idea...
## Watch out
...trap specific to this one idea...
## Worked example
...
## Faded practice
### q237
...
## Concept: building diagonal matrices
...next idea...
## Worked example
...
## Faded practice
### q47
...
```

## Rules

- Concept explains the GENERAL procedure before any example (Seth's requirement:
  never example-only).
- Worked example comments explain WHY each step, not what the syntax is.
- Starter code blanks use `_____` and must be syntactically obvious to fill.
- Faded/solo/integrated ids should come from the same subtopic as the KC's
  lesson where possible (keeps BKT mapping clean).
- Every KP carries all three drill rungs. When the subtopic's pool is exhausted —
  every question already spoken for by some KP — author a NEW bank question
  rather than serving one twice; the recipe (curated CSV → overrides →
  export → qmatrix → atom tags) is in `pipeline/README.md`. Pick a move the
  KP teaches that no existing item asks for, and vary the expected value
  across test cases so a constant-return answer cannot pass.
- One KP introduces one target KC (research doc: one worked example ↔ one KC).
