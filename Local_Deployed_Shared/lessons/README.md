# lessons

## Purpose
- First-encounter teaching content for Delta Drills easy topics (Numpy/Einsum/Einops): the knowledge-point (KP) lessons shown by the in-app lesson gate before a learner's first question on a new concept.

## Owns
- KP markdown sources (`numpy/`, `einsum/`, `einops/`), the KC registry (`kc_registry.json`), question→KC tags (`qmatrix_tags.json`), the compiled artifact (`lessons_structured.json`), the authoring spec (`AUTHORING.md`), and the read-only review viewer (`viewer.html`).

## Does NOT own
- Parsing/validation/compilation tooling (`scripts/` at repo root), the runtime gate UI (`../practice/lessons.js`), backend exposure state (`backend/app/lessons.py`), or the drill bank (`../questions_structured.json`).

## Key Files
- `AUTHORING.md`: the format contract — read it before editing any KP.
- `kc_registry.json`: 64 KCs across 9 lessons (np-1..4, es-1..2, eo-1..3); prereq edges must stay acyclic.
- `qmatrix_tags.json`: per-question target/supporting KC tags; drives the gate's "unexposed KC" check in local mode.
- `lessons_structured.json`: compiled output — NEVER hand-edit; regen via `scripts/compile_lessons.py`. Feeds both the frontend player and backend guard maps.
- `viewer.html`: static review viewer (serve this folder with `python3 -m http.server`).

## Data & External Dependencies
- Faded/guided/independent exercises reference drill-bank question ids; faded solutions must pass those questions' test_cases.
- Docker image COPYs this folder (root `.dockerignore` whitelists `!Local_Deployed_Shared/lessons/`).

## How It Works (Flow)
1. Author/edit a KP md as single-concept segments: explain → ONE worked example → ONE faded exercise.
2. `python3 scripts/validate_lessons.py --coverage` → `python3 scripts/compile_lessons.py`.
3. LessonGate renders one SEGMENT per page as a notebook: prose with every plain ` ```python ` fence — concept body and worked example alike — turned into a runnable cell sharing state top to bottom. Continue advances; finishing KP records KC exposure and resumes normal questions. Faded content is not rendered inside lesson.

## Invariants & Constraints
- **This directory is the SOURCE OF TRUTH for what the tutor may teach.** As of 2026-07-27 the ITS serves only questions carrying a `target_kcs` entry in `qmatrix_tags.json`; the 75 questions without one (CNN, PyTorch Fundamentals, Autograd, Optimizers) are parked until their chapter's KCs are authored and validated. So a question's presence here is what makes it reachable — adding `target_kcs` unparks it, removing them hides it. The older ARENA concept graph is demoted, not deleted; it is retained as the diff target for blind-authoring later chapters. See `docs/decision-kc-only-serving.md` and `docs/method-anchoring-controls-for-kc-authoring.md`.
- **Author later chapters blind.** The ARENA concept graph must not be in the authoring context — anchoring cannot be prompted away (Lou & Sun 2024: CoT, Thoughts-of-Principles, Ignore-Anchor-Hint and Reflection all failed, and authored structure is an "expert opinion" anchor, the strongest kind). Author fresh, then diff against ARENA as a check.
- One KP file per KC; one concept per segment; exactly ONE worked example + ONE faded exercise per segment (Seth's format rules — see AUTHORING.md).
- Watch out renders inside lesson teaching content. Lesson screen has no popup, faded prompt, Check, or grading.
- A faded qid appears in at most one segment; frontmatter id lists mirror the sections.
- Never hand-edit `lessons_structured.json`; always validate before compiling.
- **A fence gets a Run button iff CI runs it.** Plain ` ```python ` = executed by `validate_lessons.py` AND rendered as a runnable notebook cell; ` ```python no-run ` = illustrative only, never executed anywhere. There is no third state, so a block a learner can run is a block that is known to work.
- **Each segment must stand alone.** The namespace is fresh PER SEGMENT (validator and player agree), because the lesson screen shows one segment per page — a fence leaning on a name defined in an earlier segment would hand the learner a NameError. Every segment opens with its own imports.
- Cells within one segment DO share state, top to bottom, like a notebook. Later fences may build on earlier ones on the same page; the runtime re-executes the whole prefix on each Run.
- Prefer fences that print or end in a bare expression, and add `assert` checks where a claim is worth proving. A cell whose output is empty falls back to listing the names it bound, so it never reads as "nothing happened".
- Bank einops fixtures hardcode `/delta_numbers.npy` — validator rewrites the path; don't "fix" it in content.

## Extension Points
- New concept → add KC to `kc_registry.json`, write `kp-<slug>.md`, retag affected questions (scripts/build_qmatrix.py), validate with `--coverage`.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Multi-concept lesson pages** — `RESOLVED`
  - When it happens: a KP's Concept section introduced several APIs before any practice.
  - Symptom: beginner overwhelmed; can't practice each idea separately (Seth's 2026-07-19 feedback).
  - Root cause: original Pass-1 format allowed one big Concept per KP.
  - Prevention/fix: segment format + validator enforcement; player pages one segment at a time.
  - Status: `RESOLVED` for np-1 openers + 19 restructured KPs; remaining KPs render as single-segment legacy pages until converted.

## Recent Changes
- 2026-07-30: Lesson pages became notebooks. Every plain ` ```python ` fence in a segment's Concept prose — not just the one in Worked example — is now a runnable cell, and cells on a page share state top to bottom. `validate_lessons.py` switched from a KP-wide namespace to a fresh namespace PER SEGMENT so CI mirrors what the learner can actually reach (all 122 existing segments already passed in isolation). Pilot content: `numpy/kp-ndarray-model.md` rewritten to 3/2/2 fences with asserts and printed output. Pilot only — see Seth's standing rule about not mass-converting before review.
- 2026-07-27: `lessons/` promoted to **source of truth** for the tutor. The ITS now serves only KC-tagged questions (380 servable, 75 parked); the ARENA concept graph is demoted but retained. Validation runs chapter by chapter with the learner in the loop — walk the 64-KC chapter 1, fix/add what's missing, then blind-author the next chapter. See the two new invariants above.
- 2026-07-20: Lesson changed to inline teaching + optional runnable worked code; faded practice removed from lesson UI.
- 2026-07-19: Segmented format introduced; ~20 numpy KPs restructured; compiled JSON gains per-KP `segments`.
- 2026-07-15: Pass 1 content created (64 KPs).
