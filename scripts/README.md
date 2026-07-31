# scripts

## Purpose
- Authoring/build tooling for the first-encounter lesson course: parse KP markdown, validate it against the drill bank, compile it to the JSON the app serves.

## Owns
- The KP markdown format contract (`lesson_lib.py` parser: frontmatter, sections, single-concept segments).
- Lesson validation (`validate_lessons.py`): fences execute, faded solutions pass bank test cases, per-segment rules, registry acyclicity, coverage.
- Compilation to `lessons_structured.json` (`compile_lessons.py`) and Q-matrix generation (`build_qmatrix.py`).

## Does NOT own
- Lesson content itself (`Local_Deployed_Shared/lessons/`), runtime lesson gating (`Local_Deployed_Shared/practice/lessons.js`, backend `app/lessons.py`), or the drill bank.

## Key Files
- `lesson_lib.py`: shared parser — frontmatter, `split_sections` (ordered, repeatable), `build_segments` (one-concept segments), bank/registry loaders. HIGH fan-in: the other three scripts import it.
- `validate_lessons.py`: executes every code fence in document order, grades every faded solution against bank test_cases, enforces exactly one worked + one faded example per segment. Run with `--coverage` as the full gate.
- `compile_lessons.py`: emits `lessons_structured.json` with per-KP `segments` plus legacy aggregate fields (viewer.html back-compat).
- `build_qmatrix.py`: derives `qmatrix_tags.json` question→KC tags from KP refs + hand-assigned leftovers.

## Data & External Dependencies
- Reads `Local_Deployed_Shared/questions_structured.json` (bank), `lessons/kc_registry.json`, `lessons/*/kp-*.md`.
- numpy (fence execution); einops fixtures need `delta_numbers.npy` (validator rewrites the Docker path to the local copy).

## How It Works (Flow)
1. Edit KP markdown → `python3 scripts/validate_lessons.py --coverage`.
2. `python3 scripts/compile_lessons.py` → `lessons_structured.json` feeds BOTH the frontend player and the backend guard maps.

## Invariants & Constraints
- Every segment starts at `## Concept` and must carry exactly one worked example plus one faded exercise whose solution passes bank tests.
- Optional `## Watch out` belongs to current segment and compiles into lesson-only content.
- Compiler extracts each segment's sole Python worked fence as `worked_example_code`; LessonGate preloads it into editor.
- Grading must mirror prod: `expected_setup_code or setup_code` re-runs before `expected_expr` (fixed 2026-07-19 — validator previously evaluated expected on solution-mutated fixtures).
- Validate BEFORE compile; compilation does not re-check code.
- Frontmatter faded/guided id lists must equal the section ids (validator-enforced).

## Extension Points
- New KP rules → add checks in `check_kp`; format changes → `lesson_lib.py` first, then both consumers.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Validator expected-setup fallback mismatch** — `RESOLVED`
  - When it happens: faded solution mutates its input (out=-style drills, e.g. q59).
  - Symptom: validator FAILs a solution that grades correct in prod.
  - Root cause: validator evaluated `expected_expr` without re-running `setup_code` when `expected_setup_code` absent; JS/backend graders re-run it.
  - Prevention/fix: `grade_against_bank` now uses `expected_setup_code or setup` — keep all three graders' semantics in lockstep.
  - Status: `RESOLVED` (2026-07-19).
- **Validator missing the runtime's numpy preamble** — `RESOLVED`
  - When it happens: a torch-dialect lesson whose bank fixture loads the ARENA image (`np.load('/delta_numbers.npy')`).
  - Symptom: `NameError: name 'np' is not defined` from the validator on a drill that grades fine in the sandbox.
  - Root cause: `grade_against_bank` seeded its namespace from the solution's own imports only, while `code_runner.CODE_PREAMBLE` always injects numpy. A torch solution imports torch, so `np` never existed.
  - Prevention/fix: the grading namespace now imports numpy first — same lockstep rule as the expected-setup fix above. If the runtime preamble grows, mirror it here.
  - Status: `RESOLVED` (2026-07-27).
- **Relative paths rejected** — `ACTIVE`
  - When it happens: `validate_lessons.py some/relative/kp.md`.
  - Symptom: "not in the subpath" parse error.
  - Root cause: `path.relative_to(REPO)` needs absolute paths.
  - Prevention/fix: pass `"$PWD/..."` absolute paths.

## Recent Changes
- 2026-07-31: `previews:` added to the KP frontmatter contract. `audit_lesson_syntax.py` exempts a listed symbol from "shown before it is taught" and reports it under "declared previews" instead; `validate_lessons.py` gained `check_previews`, which runs on full-corpus runs only (it needs every page to know who declares what) and rejects an entry that the page does not show, that the page also declares, or that no later page declares. Without those three checks the key would be a mute button, and a stale entry left behind by an edit would silence a real regression.
- 2026-07-30: `build_qmatrix.py` was unrunnable and nobody noticed. It aborted on the first question that appeared BOTH in a KP's `faded`/`guided`/`independent` list and in `LEFTOVER_TARGETS` — and 76 of them did, because that is exactly how a leftover retires: a KP claims the question later. All 76 agreed on the KC, so the abort was protecting nothing. A KP reference now supersedes a leftover silently (it carries the role and the page's `new_syntax`, which a hand assignment cannot), and only a DISAGREEMENT — two sources naming different KCs — is fatal. The stale committed `qmatrix_tags.json` this hid was 274 entries behind on `new_syntax` and 79 on `source`; `target_kcs` and `supporting_kcs` were correct throughout, so no question had been gated to the wrong KC. Rebuild it whenever KP refs change, or `validate_lessons.py --coverage` fails on untagged questions.
- 2026-07-27: `grade_against_bank` mirrors the runtime numpy preamble, so torch-dialect lessons using the ARENA fixture validate. `watch.py` filled in.
- 2026-07-20: Compiled `worked_example_code` supports inline optional-run lesson UI.
- 2026-07-20: Segment-specific `## Watch out`; exact one-worked/one-faded validation.
- 2026-07-19: Single-concept segment support (`build_segments`), per-segment faded enforcement, expected-setup grading fix.
- 2026-07-06: Initial doc created.
