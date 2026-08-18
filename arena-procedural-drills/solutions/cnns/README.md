# solutions/cnns

## Purpose
- Reference-solution Colab notebooks for bank topic **`CNNs`** — 9 of them, one per question that has authoring.
- A second, smaller convolution topic that exists alongside `cnn/`.

## Owns
- `q<ID>-<subtopic-slug>.ipynb` for every `CNNs` question with an authoring row.
- `q<ID>-*.problem.ipynb` — 5 answer-withheld variants, built by `build_problem_colabs.py`.

## Does NOT own
- Assembly, manifests, pipeline and the shared invariants → `../README.md`. Everything here is **generated**; hand-edits are lost on the next build.
- The questions themselves → `This-Directory-Only/backend/app/questions.py`.

## Key Files
Subtopics covered:
- `Hyperparameter normalization (int-or-tuple to (h,w) pair)` — 3
- `BatchNorm per-channel broadcasting` — 3
- `Padding for Max Pooling` — 3

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
  - Symptom: 1 of 9 notebooks here ship with no "Why this works", and 1 with no hint.
  - Root cause: prose cannot be carried across an answer that changed algorithm.
  - Prevention/fix: author against the live `answer_code`; ids are listed in `scripts/solution_build/retorch_report.json`.
  - Status: `ACTIVE`.

- **Topic-specific** — `ACTIVE`
  - **`CNN` and `CNNs` are two distinct bank topics**, so they are two directories and two sets of mastery keys for closely related material. Nothing depends on them being separate; merging them is a state migration, not a rename.
  - Status: `ACTIVE` — context to know before editing here.

## Recent Changes
- 2026-08-18: All 9 notebooks regenerated in torch dialect (headings relabelled, `%pip` corrected, withheld explanations omitted).
