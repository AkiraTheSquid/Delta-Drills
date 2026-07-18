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
faded: [111, 151]                   # bank question ids used as faded practice
guided: [90]                        # bank ids used as guided (hinted) practice
independent: [144, 84]              # bank ids for unaided practice
---

## Concept
Prose teaching the general idea/procedure. Plain ```python fences here are
EXECUTED by the validator (shared namespace per file, top to bottom).
Use ```python no-run for fences that intentionally error or are pseudocode.

## Worked example
Heavily commented code + step-by-step walkthrough. Fences executed too.

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

## Misconceptions
- **Short trap name** — what learners wrongly believe → the correcting feedback.
```

## Rules

- Concept explains the GENERAL procedure before any example (Seth's requirement:
  never example-only).
- Worked example comments explain WHY each step, not what the syntax is.
- Starter code blanks use `_____` and must be syntactically obvious to fill.
- Faded/guided/independent ids should come from the same subtopic as the KC's
  lesson where possible (keeps BKT mapping clean).
- One KP introduces one target KC (research doc: one worked example ↔ one KC).
