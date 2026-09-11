# ARENA curriculum navigation

The Knowledge Graph page has an ARENA curriculum view covering the book's five
chapters and 32 sections. Section and exercise arrows represent reading order.
They do not add mastery prerequisites, modify rung selection, or award credit.
The original practice graph and its 44 KCs remain available beside this view.

`python3 scripts/compile_arena_notebooks.py` compiles the same fork-first notebook
sources used by Courses. It also emits `arena-curriculum.json`, currently linking
540 exercise tasks, 209 supporting concepts, and the existing practice KCs.
Preparation links come from existing authored exercise annotations and
`arena_exercise_kcs.json`; titles match within their section. Later sections have
source exercises, but do not yet have dedicated drill mappings. This count is
source-task coverage, not 540 new graded bank questions.

Opening an exercise scrolls to its question in the full editable notebook.
Previous setup, prose, answers, and outputs remain above it. The selector and
Previous/Next buttons navigate without rebuilding the notebook or replaying
answers. Empty upstream answer cells are retained; written-reasoning tasks get
an editable Python comment cell. Existing saved answers survive the migration.
Solutions remain in closed disclosures.

For signed-in learners, opening a notebook prepares its Python context. Setup
cells remain editable and show their actual outputs in their original positions.
The compiler identifies the initial Setup section and a few explicitly declared
source data/helper cells outside it. It does not execute all code above a task:
previous learner answers, demonstrations, model experiments, and training runs
are not automatically replayed. A fresh kernel restores declared setup before
running the requested cell. Earlier definitions from learner answers need reruns
following a reset. Failed setup stops execution and exposes a Retry setup action.
Later ARENA notebooks retain their source requirements for packages, datasets,
models, GPU access, and API credentials; the frontend does not provision them.

The shared cell harness still runs checks; the grading and hidden-assert paths
are unchanged. Curriculum navigation does not interpret a successful code run as
a correct exercise solution.

Validation:

```sh
python3 scripts/test_arena_curriculum.py
node This-Directory-Only/scripts/test_arena_setup.mjs
python3 Local_Deployed_Shared/lessons/notebooks/watch.py
python3 Local_Deployed_Shared/practice/watch.py
python3 Local_Deployed_Shared/lessons/watch.py
python3 scripts/watch.py
```

The notebook watch runs the curriculum integrity tests, including every graph
endpoint, prompt/answer anchor, setup reference, and agreement with the book TOC.
To repeat browser checks, serve `Local_Deployed_Shared` at localhost:8874 and run
`node This-Directory-Only/scripts/test_arena_browser.cjs` (Playwright required).
`ARENA_TEST_URL` overrides the origin. This checks actual rendering/navigation
with mocked Python RPC, including persisted edits and the mobile layout. It does
not claim that all 540 exercises or remote setup dependencies were executed.
