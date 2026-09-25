# deltadrills

## Purpose
- The **Delta Drills course**: this app's own small "what this app is, and
  the math behind it" explainer, the second course in the Courses tab
  catalog alongside ARENA (docs/spec-multi-course-catalog.md). It teaches
  mastery estimates, spaced repetition, and why the app drills instead of
  just showing answers — using the app itself as the running example.
- Same math MC lane as `../mathematics/` (`kind: math`), just a different
  topic. Not the Deep Chat / conceptual-tutor interface (`conceptual/`,
  separate work) — this is a plain lesson-plus-multiple-choice course.

## Owns
- `kp-<slug>.md` pages with `kind: math` in the frontmatter (topic
  `Delta Drills` in `../kc_registry.json`, lesson `dd-1`).
- `kp-<slug>.problems.json` beside each page (`{"kc": ..., "problems": [...]}`,
  format in `../math_bank.py`'s docstring; ids `>= 50000`). All three
  problems files here use `kind: "statement"` — conceptual MCQs, no SymPy.

## Does NOT own
- The loader (`../math_bank.py`), the gate (`scripts/validate_math.py`), the
  frontend (`practice/math-drill.js`). Atom tags
  (`backend/app/data/question_atom_tags.jsonl`), the atom graph
  (`backend/app/data/concept_graphs/arena_drillable_v1.json`, atom ids
  `dd-mastery` / `dd-spaced-repetition` / `dd-why-drills`), placement clocks
  (`../placement_time_caps.json`), the glossary `kcLesson` map
  (`../glossary.js`) — each KC here still needs its row in every one of
  those, same as a math KC.
- The course catalog UI and the per-course practice share
  (`courses.js`, `course_mix.py`, `course_registry.py`) — this folder only
  owns the content those wire up.

## Key Files
- `kp-mastery.md` / `.problems.json` — `deltadrills.mastery` (root KC): what
  a BKT mastery estimate is and how an answer moves it.
- `kp-spaced-repetition.md` / `.problems.json` — `deltadrills.spaced-repetition`
  (prereq: mastery): the forgetting curve and why reviews are scheduled
  before a concept would otherwise fade.
- `kp-why-drills.md` / `.problems.json` — `deltadrills.why-drills` (prereq:
  spaced-repetition): the testing effect — why the app makes you produce an
  answer instead of re-reading one.

## Data & External Dependencies
- SymPy is NOT used here (`statement` kind forbids `verify.sympy`) — these
  pages are conceptual, not computational, unlike `../mathematics/`.

## How It Works (Flow)
1. Author page + problems here; registry KC/lesson, atom tag rows, a
   placement clock, and the glossary `kcLesson` line all need a matching row
   (see Does NOT own).
2. Same pipeline as any math KP: `pipeline/export_questions_json.py` →
   `scripts/compile_lessons.py` → `scripts/build_qmatrix.py` →
   `scripts/validate_lessons.py` (runs `validate_math.py`) →
   `export_kc_atom_crosswalk.py` → `generate_colab_notebooks.py` +
   `compile_web_notebooks.py`.
3. Backend `app/course_mix.py` treats `deltadrills.*` as this course's
   milestones (`app/course_registry.py::DELTA_DRILLS_KCS`); enabling the
   course in the Courses tab card sets its practice share.

## Invariants & Constraints
- Every problem here is `kind: "statement"` and carries NO `verify.sympy` —
  there is nothing to prove; the gate rejects a `sympy` block on this kind.
- Every problem needs a row in `question_atom_tags.jsonl` pointing at an
  atom that exists in `arena_drillable_v1.json`, or the deploy audit blocks
  it, same as every other KC.
- `deltadrills.mastery` is the sole `intentional_root_atoms` entry for this
  course (`dd-mastery`); the other two chain off it in registry order.

## Extension Points
- New concept: copy the shape of an existing page here, add it to the chain
  in `../kc_registry.json`, and follow `../mathematics/README.md`'s pipeline
  steps — this folder is deliberately the same shape as that one.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- None yet — folder is new.

## Recent Changes
- 2026-09-25: Folder created with the small mastery → spaced-repetition →
  why-drills chain (2 statement-kind MCQs each, q50112–q50117), the second
  course in the restored Courses tab catalog. `docs/spec-multi-course-catalog.md`.
