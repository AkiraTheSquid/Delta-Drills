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
new_syntax: []                      # syntax tags introduced here
concepts: [repeat-elements]         # stable id per atomic segment, in order
faded: [111, 151]                   # bank question ids used as faded practice
guided: [90]                        # bank ids used as guided (hinted) practice
independent: [144, 84]              # bank ids for unaided practice
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

## Guided practice
### q90
1. First hint (conceptual nudge).
2. Second hint (names the function).
3. Third hint (near-solution).

## Independent practice
(only the frontmatter `independent:` ids matter; prose here optional)

```

## Segments — ONE concept at a time (required)

A KP is a sequence of single-concept SEGMENTS. Each `## Concept` heading starts
a new segment; its `## Worked example` and `## Faded practice` belong to that
segment. A segment heading may carry a subtitle: `## Concept: np.trace`.

- Each segment teaches exactly ONE new idea (one function, one mode, one rule).
  If the prose says "also" or introduces a second API, split the segment.
- `concepts:` declares one stable id per reviewed atomic segment. Its count
  MUST equal the number of segments, ids MUST be unique, and every declared
  segment MUST have a `## Concept: ...` title. This makes accidental re-merges
  fail validation and supplies concept-level ids for adaptive sequencing.
- Each segment's `## Worked example` shows exactly ONE worked example — a
  single small demo of the one idea, not a tour of variations. The rhythm is:
  teach one concept → inspect or optionally run one worked example → continue.
  Extra variations belong in practice, not lesson screen.
- Each segment MUST have one or two faded exercises (validator-enforced).
  Two is a FADING SERIES, not two drills: the first sits adjacent to the
  worked example, the second asks for the same idea one step out, so the blank
  cannot be filled by transcription. `audit_ladder_pairing.py` measures that
  step ("series never reaching distance"). A third belongs in independent
  practice — a segment teaches one idea, and more completions of it are drill.
  PILOT as of 2026-07-30: 3 of 122 segments have a second item; awaiting
  Seth's review before the rest follow.
- Faded exercise is downstream practice metadata. LessonGate does NOT render or
  grade it immediately after teaching.
- `## Watch out` is optional segment content. It renders only inside that
  concept's lesson screen.
- In-app sequence is fixed: teaching + worked explanation on left, complete
  worked code preloaded on right for optional running/editing, then next
  concept or normal question queue. No popup; no faded exercise inside lesson.
- Guided/Independent stay KP-level, after the last segment. `## Misconceptions`
  remains a legacy fallback for single-segment KPs; new/multi-segment content
  uses `## Watch out` inside each relevant segment.
- A faded qid may appear in only one segment.

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
- Faded/guided/independent ids should come from the same subtopic as the KC's
  lesson where possible (keeps BKT mapping clean).
- One KP introduces one target KC (research doc: one worked example ↔ one KC).
