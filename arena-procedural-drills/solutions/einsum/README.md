# solutions/einsum

## Purpose
- Reference-solution Colab notebooks for bank topic **`Einsum`** — 68 of them, one per question that has authoring.
- The `einsum` half of the ARENA 0.0 exercise material, split from `einops/` so each conversion pass stayed reviewable on its own.

## Owns
- `q<ID>-<subtopic-slug>.ipynb` for every `Einsum` question with an authoring row.

## Does NOT own
- Assembly, manifests, pipeline and the shared invariants → `../README.md`. Everything here is **generated**; hand-edits are lost on the next build.
- The questions themselves → `This-Directory-Only/backend/app/questions.py`.

## Key Files
Subtopics covered:
- `Applied patterns and advanced` — 24
- `Core array literacy` — 17
- `Vectorization and broadcasting` — 16
- `Indexing and selection` — 11

## Data & External Dependencies
- Reachable only via GitHub — `colabUpstreamHref()` maps the `arena-procedural-drills/` prefix to `AkiraTheSquid/Delta-Drills` main, so these must be committed and pushed to open.
- Every solution here is torch dialect; `%pip` is derived from the solution's own imports, never from the stale `primary_library` field.

## How It Works (Flow)
See `../README.md`. Nothing in this folder runs — these are artifacts.

## Invariants & Constraints
- **The directory name is the topic SLUG, not a claim about the library.** Every solution in this repo is torch.
- A notebook with no "Why this works" section is that way on purpose: the authored prose described the pre-conversion algorithm and was withheld rather than mechanically renamed into confident, wrong text.

## Extension Points
- Add prose for a backlogged id → `scripts/solution_build/authored/`, then re-run the pipeline. Never create a file here by hand.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Authoring backlog** — `ACTIVE`
  - When it happens: a question's answer was rewritten after its explanation was authored.
  - Symptom: 5 of 68 notebooks here ship with no "Why this works".
  - Root cause: prose cannot be carried across an answer that changed algorithm.
  - Prevention/fix: author against the live `answer_code`; ids are listed in `scripts/solution_build/retorch_report.json`.
  - Status: `ACTIVE`.

- **Topic-specific** — `ACTIVE`
  - Subtopic names here mirror the Numpy topic's (Core array literacy, Vectorization and broadcasting). That is why `questions.py` prefixes the topic when building the subtopic key — without it, Einsum and Numpy subtopics would collide in mastery storage.
  - Status: `ACTIVE` — context to know before editing here.

## Recent Changes
- 2026-08-18: All 68 notebooks regenerated in torch dialect (headings relabelled, `%pip` corrected, withheld explanations omitted).
