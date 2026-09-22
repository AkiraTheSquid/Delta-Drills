# mathematics

## Purpose
- The math lane: lessons the learner READS (markdown + LaTeX) and problems
  they solve ON PAPER, answered as a multiple-choice pick. No kernel, no code.
  Spec: `docs/spec-math-mc-backbone.md`. Software landed 2026-09-21; **no
  content yet** — Fable authors it, one concept at a time.
- Why a lane and not a track: `math.*` concepts sit in the SAME KC graph as
  the coding ones, so a coding KC (`raytracing.ray-parametrisation`) can list
  a math prerequisite and the same picker, ladder and mastery serve both.

## Owns
- `kp-<slug>.md` pages with `kind: math` in the frontmatter (topic
  `Mathematics` in `../kc_registry.json`).
- `kp-<slug>.problems.json` beside each page: the problems that page owns
  (`{"kc": ..., "problems": [...]}` — format in `../math_bank.py`'s
  docstring; ids `>= 50000`).

## Does NOT own
- The loader (`../math_bank.py`, stdlib-only, shared by the exporter, the
  backend and the validator), the gate (`scripts/validate_math.py`), the
  frontend (`practice/math-drill.js`, `styles/practice/math-drill.css`).
- Atom tags (`backend/app/data/question_atom_tags.jsonl`), the atom graph,
  placement clocks (`../placement_time_caps.json`), the glossary `kcLesson`
  map (`../glossary.js`) — each math KC still needs its row in every one of
  those, exactly like a torch KC.

## Key Files
- `kp-<slug>.md`: the page. Same sections as a code KP (`## Concept`,
  `## Worked example`, `## Solo practice`, `## Integrated practice`); prose
  and `$…$` / `$$…$$` LaTeX; ```python fences only as `no-run`.
- `kp-<slug>.problems.json`: the problems. `compute` / `derivation-step`
  carry `verify.sympy` (`truth` + per-choice `value`); `statement` must not.

## Data & External Dependencies
- SymPy 1.14 in `This-Directory-Only/backend/.venv` runs the verification
  (validator only; the backend image never imports it).
- KaTeX (vendored, `vendor/katex`) renders the page, the prompt, the choices
  and the solution in the browser.

## How It Works (Flow)
1. Author page + problems here; add the KC/lesson to the registry, atom tag
   rows, a placement clock, the glossary `kcLesson` line.
2. `pipeline/export_questions_json.py` → the problems join `questions.json`
   as `submission_mode: "mc"` rows; `scripts/compile_lessons.py` →
   `lessons_structured.json` (`kind: "math"`); `scripts/build_qmatrix.py`;
   `scripts/validate_lessons.py --coverage` (runs `validate_math.py`);
   `export_kc_atom_crosswalk.py`; `audit_question_bank.py --gate`;
   `generate_colab_notebooks.py` + `compile_web_notebooks.py`.
3. Backend `app/math_questions.py` loads the same rows at boot; `/submit`
   grades by key compare (`grading.grade_choice`).
4. Frontend: `DeltaMath.mount(q)` swaps the editor for a radio list.

## Invariants & Constraints
- Every `compute` / `derivation-step` problem is PROVEN: `truth` equals the
  keyed choice's `value` and differs from every distractor's, or the gate
  fails with the id named. Never skip the gate to land content.
- Ids are explicit and `>= 50000`, unique across every problems file
  (`math_bank.load_math_rows` raises on a duplicate, so the exporter and
  the backend cannot disagree); the validator, the exporter and the backend
  all refuse a collision with the CSV bank. One rung per problem; `guided`
  stays empty.
- Choices are served in authored order (keys are the author's letters); the
  keyed letter never leaves the server until `/submit`.
- A page with 2+ segments needs one carrier per segment under
  `## Faded practice` (segment gate), though the faded RUNG is retired.
- The prompt renders through the drill renderer (plain text + `code` +
  LaTeX; NO markdown emphasis); the solution renders as full markdown.

## Extension Points
- New concept: copy the shape of an existing page here (none yet — use the
  fixture in `scripts/test_validate_math.py` as the reference), then follow
  the checklist in `../AUTHORING.md` ("Math pages").
- Typed answers / Lean: reserved (`verify.lean` warns, never read); a later
  pass.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Notebook shows no check cell for a math problem** — `RESOLVED`
  - When it happens: `compile_web_notebooks.py` on a lesson with mc rows.
  - Symptom: `notebooks/watch.py` "advertises problem N but has no
    dd-qN-check cell".
  - Root cause: the `dd-q<n>` id grammar promises an editor + checker.
  - Prevention/fix: `colab_cells.mc_problem_cell` emits math problems as
    prose (`dd-mc<n>`, choices listed, no key); the practice page grades.

## Recent Changes
- 2026-09-21: Folder created with the math MC backbone (software only, no
  pages). Lane contract documented; content is Fable's.
