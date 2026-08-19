# notebooks

## Purpose
- The ch-1 lessons, compiled into the shape the in-app Notebooks tab reads: one
  JSON file per lesson holding every cell the Colab edition publishes.
- This is **build output**, not source. Nothing here is hand-edited. The lesson
  itself lives in `Local_Deployed_Shared/lessons/lessons.json` and the question
  bank; this folder is what the browser can fetch without a Python toolchain.

## Owns
- The compiled per-lesson notebooks `<lesson-id>.json` (`np-1` … `eo-3`).
- `manifest.json` — the index the Notebooks tab lists: id, title, topic,
  subtopic key, file name, cell count, and the problem ids the lesson contains.
- The promise that a web notebook and its Colab `.ipynb` are the same notebook.

## Does NOT own
- How cells are built. `scripts/generate_colab_notebooks.py::build_notebook` is
  the single compiler; `scripts/compile_web_notebooks.py` only reshapes its
  output for the browser.
- How cells are drawn or run — `Local_Deployed_Shared/practice/notebook-view.js`.
- Grading. The `dd_check` grader is compiled into the checker cell from the same
  question bank the practice page uses; nothing here decides right or wrong.

## Key Files
- `<lesson-id>.json`: `{id, title, topic, subtopic_key, segments, cells}`. Each
  cell is `{t: "code"|"md", id, role, src, q?}`.
- `manifest.json`: `{lessons: [{id, title, topic, subtopic_key, file, cells,
  questions}]}` — fetched once when the Notebooks tab opens.
- `watch.py`: the parity check. Compares every cell against the published
  `.ipynb`, and the manifest against the folder.

## Data & External Dependencies
- Built from `lessons/lessons.json` + `questions_structured.json` via
  `scripts/compile_web_notebooks.py` (which imports `build_notebook`).
- Compared against `arena-book-colab/ARENA_5.0/ch-1-foundations/*.ipynb`.
- Read at runtime by `practice/notebook-view.js` over plain `fetch`.

## How It Works (Flow)
1. `python3 scripts/compile_web_notebooks.py` builds each lesson's notebook in
   memory with the Colab compiler, trims each cell to `{t, id, role, src}`, adds
   `role` (derived from the id) and `q` (the problem the cell belongs to), and
   writes this folder.
2. The Notebooks tab fetches `manifest.json`, lists the lessons, then fetches
   one `<lesson-id>.json` when a lesson is opened.
3. Every code cell runs on one kernel session keyed `nb:<lesson-id>`, so the
   whole lesson shares a namespace the way it does in Colab.

## Invariants & Constraints
- **Never hand-edit a file in this folder.** The next compile overwrites it, and
  in the meantime the web edition teaches something Colab does not.
- Cell ids are the contract. `dd-setup`, `dd-checker`, `dd-lesson-<id>`,
  `dd-kp-<slug>`, `dd-seg-<kc>-<n>`, `dd-q<n>*` — the renderer, the segment
  jumps and the Chrome extension all key off them.
- `role` must stay inside the set `notebook-view.js::_cellNode` dispatches on;
  an unknown role draws nothing at all.
- Every problem the manifest advertises must have a `dd-q<n>-check` cell.
- Regenerate after ANY change to the lessons, the question bank, or the
  compiler — and regenerate the `.ipynb` files in the same breath. `watch.py`
  fails the moment the two drift.

## Extension Points
- New lesson: add it to `lessons.json`, run both generators. Nothing here
  changes by hand.
- New cell kind: teach `compile_web_notebooks.cell_role` the id prefix, add the
  role to `ROLES` in `watch.py`, and give `notebook-view.js::_cellNode` a
  builder for it — in that order, so the check catches a half-done change.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Compiled output drifts behind the lessons** — `ACTIVE`
  - When it happens: a lesson or the question bank changes and only the `.ipynb`
    files are regenerated.
  - Symptom: none, visually. The notebook renders perfectly and teaches the
    previous version of the lesson.
  - Root cause: two generators, one habit.
  - Prevention/fix: `watch.py::check_invariants` compares every cell against the
    published `.ipynb` and fails on the first difference.

- **Deploy ships a stale folder** — `ACTIVE`
  - When it happens: `deploy_delta_drills` re-runs the Colab export but not the
    web compiler.
  - Symptom: prod notebooks lag the repo.
  - Prevention/fix: the deploy runs `compile_web_notebooks.py` alongside the
    Colab export; if that ordering is ever changed, this check is the tripwire.

## Recent Changes
- 2026-08-19: Folder created — pass 2 of the stateful-notebook work. Nine
  lessons, 2589 cells, 424 problems, verified cell-for-cell against the Colab
  notebooks.
