# solutions/numpy

## Purpose
- Reference-solution Colab notebooks for bank topic **`Numpy`** — 214 of them, one per question that has authoring.
- This is the folder the whole torch conversion was about. Every question here is `import torch as t`; the directory name is the stored topic key, which cannot be renamed without orphaning mastery state. It also carries the largest authoring backlog by far.

## Owns
- `q<ID>-<subtopic-slug>.ipynb` for every `Numpy` question with an authoring row.

## Does NOT own
- Assembly, manifests, pipeline and the shared invariants → `../README.md`. Everything here is **generated**; hand-edits are lost on the next build.
- The questions themselves → `This-Directory-Only/backend/app/questions.py`.

## Key Files
Subtopics covered:
- `Vectorization and broadcasting` — 63
- `Indexing and selection` — 54
- `Core array literacy` — 52
- `Applied patterns and advanced` — 45

## Data & External Dependencies
- Reachable only via GitHub — `colabUpstreamHref()` maps the `arena-procedural-drills/` prefix to `AkiraTheSquid/Delta-Drills` main, so these must be committed and pushed to open.
- Every solution here is torch dialect; `%pip` is derived from the solution's own imports, never from the stale `primary_library` field.

## How It Works (Flow)
See `../README.md`. Nothing in this folder runs — these are artifacts.

## Invariants & Constraints
- **The directory name is the topic SLUG, not a claim about the library.** Every solution in this repo is torch.
- **Displayed as "PyTorch tensors", stored as `Numpy`.** `practice/config.js::TOPIC_DISPLAY_LABELS` relabels it at render time; the notebook headings here say `PyTorch tensors`. The stored key must not change — it is what BKT mastery and EWMA accuracy are filed under.

- A notebook with no "Why this works" section is that way on purpose: the authored prose described the pre-conversion algorithm and was withheld rather than mechanically renamed into confident, wrong text.

## Extension Points
- Add prose for a backlogged id → `scripts/solution_build/authored/`, then re-run the pipeline. Never create a file here by hand.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Authoring backlog** — `ACTIVE`
  - When it happens: a question's answer was rewritten after its explanation was authored.
  - Symptom: 107 of 214 notebooks here ship with no "Why this works", and 73 with no hint.
  - Root cause: prose cannot be carried across an answer that changed algorithm.
  - Prevention/fix: author against the live `answer_code`; ids are listed in `scripts/solution_build/retorch_report.json`.
  - Status: `ACTIVE`.

- **Topic-specific** — `ACTIVE`
  - The three deliberate torch-vs-numpy contrasts in the bank live here (q23 flatten has no `order=`, q225 float32 vs float64, q233 no negative-step slicing). They are the only places a learner should see the word numpy, and they are teaching the difference on purpose.
  - Status: `ACTIVE` — context to know before editing here.

## Recent Changes
- 2026-08-18: All 214 notebooks regenerated in torch dialect (headings relabelled, `%pip` corrected, withheld explanations omitted).
