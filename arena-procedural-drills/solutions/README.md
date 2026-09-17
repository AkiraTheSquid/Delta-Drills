# solutions

## Purpose
- The per-question Colab notebooks behind **Show Answer** on the practice tab: one runnable reference solution per bank question, plus the `.problem.ipynb` variant that poses the same question with the answer removed.
- The domain concern is that torch cannot run in the in-app Pyodide sandbox. A torch question is routed to Colab, and a question routed to Colab with nothing behind it is a dead end.

## Owns
- One `q<ID>-<subtopic-slug>.ipynb` per question that has authoring, filed under a slug of its bank **topic**.
- The `.problem.ipynb` siblings (answer withheld, effective starter shown).

## Does NOT own
- Assembly. These are **generated** — `scripts/solution_build/build_solution_colabs.py` and `build_problem_colabs.py` write every file here. Hand-edits are lost on the next build.
- The id→path manifests, which live at `This-Directory-Only/backend/app/data/question_{solution,problem}_notebooks.jsonl`.
- The drills' own `<name>.solution.ipynb` files, which are a different feature → `arena-procedural-drills/<prereqs_*>/`.
- Whether a notebook is surfaced at all → `Local_Deployed_Shared/practice/ui.js`.

## Key Files
- `numpy/` (214), `einops/` (90), `einsum/` (68), `cnn/` (24), `pytorch-fundamentals/` (21), `autograd/` (12), `cnns/` (9), `optimizers/` (9) — one subfolder per bank topic, each with its own README.
- Every notebook has the same shape: title, Problem, optional Hint, `%pip` cell, Reference solution, optional "Why this works".

## Data & External Dependencies
- Paths are rooted at the `arena-procedural-drills/` prefix that `Local_Deployed_Shared/stats/predicted-links.js::colabUpstreamHref()` routes to GitHub `AkiraTheSquid/Delta-Drills` main (per-user fork override supported).
- Therefore a notebook is only reachable once it is **committed and pushed**. Generating it is not shipping it.
- Notebooks install torch from the CPU wheel index; `numpy` rides along because the graders and the `np.load('/delta_numbers.npy')` fixture loader use it.

## How It Works (Flow)
1. `scripts/solution_build/export_questions.py` snapshots the bank.
2. `retorch_authored.py` decides which authored prose is still usable.
3. `build_solution_colabs.py` writes the notebooks here and rewrites the manifests.
4. The backend attaches `solution_notebook_path` to a served question; the frontend renders Show Answer.

## Invariants & Constraints
- **The folder name is the topic SLUG, not the topic label.** `numpy/` holds torch questions. The bank files mastery under the subtopic key `f"{topic}: {subtopic}"`, so the stored topic must stay `Numpy` or every learner's BKT posterior and EWMA orphan. The rename is display-only — the notebook heading says "PyTorch tensors", the directory does not, and that mismatch is deliberate.
- **Nothing here is authored by hand.** Fix the generator or the authoring layer, then rebuild.
- **A `.problem.ipynb` must not contain the answer.** Enforced upstream by `validate_stubs.py`.
- **Deleting is keyed on the question, not on the build.** A question with no authored prose produces no notebook this run and keeps whatever it already had; only a question that has left the bank is pruned.

## Extension Points
- New question needing a notebook → author prose into `scripts/solution_build/authored/`, re-run the pipeline. No file is created here by hand.
- New per-question artifact → a new `build_*.py` beside the existing ones, writing into its own subtree.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Generated content outliving the bank it was generated from** — `RESOLVED`
  - When it happens: the bank is rewritten wholesale and only the questions are migrated.
  - Symptom: every one of these 455 notebooks taught `import numpy as np` for four months after the July torch conversion, while the questions above them were torch. Reported against q46.
  - Root cause: the notebooks are a snapshot, and nothing tied their freshness to the bank's.
  - Prevention/fix: re-run the pipeline after any bank-wide change; `retorch_authored.py` reports `answer not torch` as a standing check.
  - Status: `RESOLVED` 2026-08-18 — 447 rebuilt, 8 pruned.

- **A notebook nobody can reach** — `ACTIVE`
  - When it happens: a notebook is generated but the commit is not pushed.
  - Symptom: Show Answer opens Colab on a 404.
  - Root cause: the frontend resolves by GitHub path, so the repo is the deploy target.
  - Prevention/fix: `This-Directory-Only/scripts/audit_colab_notebook_index.py` checks mappings against what is actually published. Run it before claiming a notebook ships.
  - Status: `ACTIVE` — nothing blocks a build from producing unpushed files.

## Recent Changes
- 2026-08-18: All solution notebooks re-dialected from NumPy to torch. Headings relabelled through a mirror of `practice/config.js::TOPIC_DISPLAY_LABELS`, `%pip` lines derived from the code instead of the stale `primary_library` field, and 8 notebooks for questions retired in July removed. 128 ship without a "Why this works" pending hand authoring — see `scripts/solution_build/retorch_report.json`.
