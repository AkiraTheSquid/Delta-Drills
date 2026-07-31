# instructions

## Purpose
Written specs for work that has been scoped but not yet authorized. A document
lands here when a build is large enough that agreeing on *what* comes before
agreeing on *how*, and it stays here as the record of what was decided and why
the rejected options were rejected.

## Owns
- The spec for a proposed change: problem, evidence, options, effort, risk.
- The measurements a spec rests on, with the command that produced them, so a
  later reader can re-run them instead of trusting a number.
- The explicit status line — DRAFT / AUTHORIZED / SUPERSEDED / ABANDONED.

## Does NOT own
- Implementation. Nothing here is executable and nothing imports from here.
- Folder-level architecture docs. Those are the `README.md` inside the folder
  being described (`Local_Deployed_Shared/practice/README.md` and siblings).
- Task tracking. That is `mod goals`.

## Key Files
- `spec-in-app-execution-and-colab-routing.md`: three linked threads — showing
  expected output in the generated drill notebooks, wiring knowledge-graph nodes
  to Colab sections, and a costed comparison of in-app execution on Modal
  against the existing Fly fork runner. DRAFT.

## Data & External Dependencies
Specs here read from, but never write to:
- `This-Directory-Only/questions_full.json` — the question bank.
- `Local_Deployed_Shared/lessons/colab_notebooks.json` — generated; never edited.
- `arena-book-colab/ARENA_5.0/` — the upstream ARENA content.
- Vendor pricing pages (Modal, Colab). These change; date every quote.

## How It Works (Flow)
1. A question arrives that is too big to answer inline.
2. Measure first. Every claim that carries a number gets the command that
   produced it written next to it.
3. Lay out options with effort and cost, including the option of doing nothing.
4. Land it here as DRAFT with an explicit open-questions section.
5. On sign-off, flip the status line and open `mod goals` entries for the items.

## Invariants & Constraints
- **A spec is not authorization.** The status line is the only thing that says
  whether work may start. Default is DRAFT.
- **Cite the command, not the conclusion.** A cost or coverage number without a
  reproducible command is the kind of claim that survives long after it stops
  being true.
- **Date every vendor price.** Modal and Colab pricing both moved during the
  research behind the current spec, and Colab's per-CU rates are not officially
  published — they come from third-party measurement and must be labelled as
  such.
- Never restate a folder's architecture here. Link to its `README.md` instead;
  two copies means one of them is wrong.

## Extension Points
Start a new spec by copying the section order of
`spec-in-app-execution-and-colab-routing.md`: status line, correction of any
prior claim being overturned, per-thread specs, then a sequenced table of
items with effort and cost, then open questions.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Conflating "loads weights" with "needs a GPU"** — `RESOLVED`
  - When it happens: costing a chapter by counting `from_pretrained` calls.
  - Symptom: `chapter1_transformer_interp` was reported as expensive because
    129/210 notebooks load a model. It loads `gpt2-small` — 124M params — which
    runs on CPU.
  - Root cause: a proxy metric substituted for the thing actually being asked.
  - Prevention/fix: extract the model *names*, not the call count:
    `grep -rhaoE "from_pretrained\(\s*\\\\?[\"'][^\"'\\\\]+" <dir> | sed -E "s/.*[\"']//" | sort | uniq -c`
  - Status: `RESOLVED` in the current spec §0.

- **Grepping `.ipynb` with source-code regexes** — `ACTIVE`
  - When it happens: any measurement over `arena-book-colab/`.
  - Symptom: a pattern that works on `.py` returns zero matches.
  - Root cause: notebooks are JSON; source lines are escaped, so `"` in the
    code is `\"` in the file.
  - Prevention/fix: allow the backslash in the pattern (`\\\\?[\"']`) or match
    on a substring that contains no quotes.
  - Status: `ACTIVE` — this will bite again.

- **Assuming a torn-out feature is gone** — `RESOLVED`
  - When it happens: estimating the cost of rebuilding something.
  - Symptom: a rebuild was estimated for server-side code execution that was
    already deployed and serving. The self-report teardown removed the frontend
    editor only; `backend/app/code_runner.py` and `POST /api/practice/submit`
    were never touched.
  - Root cause: the frontend diff was read as the whole change.
  - Prevention/fix: before costing a rebuild, grep the backend for the
    capability and check the routes are still mounted.
  - Status: `RESOLVED` in the current spec §D.2.

## Recent Changes
- 2026-07-31: Folder created. Added
  `spec-in-app-execution-and-colab-routing.md` (DRAFT) covering expected-output
  display in generated notebooks, graph→Colab routing, and the Modal vs Fly
  execution comparison.
