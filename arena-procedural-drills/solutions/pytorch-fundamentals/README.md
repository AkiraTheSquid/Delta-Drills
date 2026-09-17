# solutions/pytorch-fundamentals

## Purpose
- Reference-solution Colab notebooks for bank topic **`PyTorch Fundamentals`** — 21 of them, one per question that has authoring.
- Core tensor and module operations that later topics assume.

## Owns
- `q<ID>-<subtopic-slug>.ipynb` for every `PyTorch Fundamentals` question with an authoring row.
- `q<ID>-*.problem.ipynb` — 20 answer-withheld variants, built by `build_problem_colabs.py`.

## Does NOT own
- Assembly, manifests, pipeline and the shared invariants → `../README.md`. Everything here is **generated**; hand-edits are lost on the next build.
- The questions themselves → `This-Directory-Only/backend/app/questions.py`.

## Key Files
Subtopics covered:
- `amax` — 3
- `Weight Initialization (normal_ / constant_)` — 3
- `Cross-Entropy Loss on Logits` — 3
- `biased vs unbiased` — 3
- `nn.Module extra_repr` — 3
- `.grad on leaf parameters` — 3
- `Freezing parameters with requires_grad_` — 3

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
  - Symptom: 1 of 21 notebooks here ship with no "Why this works", and 1 with no hint.
  - Root cause: prose cannot be carried across an answer that changed algorithm.
  - Prevention/fix: author against the live `answer_code`; ids are listed in `scripts/solution_build/retorch_report.json`.
  - Status: `ACTIVE`.

- **Topic-specific** — `ACTIVE`
  - Part of the ids 405-479 curated batch, which `kc_only_serving()` parks out of the servable pool because no lesson tags it yet. Those questions were invisible to the exporter until `get_every_question()` was added.
  - Status: `ACTIVE` — context to know before editing here.

## Recent Changes
- 2026-08-18: All 21 notebooks regenerated in torch dialect (headings relabelled, `%pip` corrected, withheld explanations omitted).
