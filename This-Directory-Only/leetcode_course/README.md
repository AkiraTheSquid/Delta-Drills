# leetcode_course (data)

## Purpose
- The LeetCode Patterns course's source of truth before app wiring: the concept
  graph and the finished, tested, tagged, difficulty-rated problem bank.

## Owns
- `patterns.json`: 41 concepts. Backbone = the Grokking patterns (`grokking` = section
  number); `grokking: null` rows are foundations / extensions the core bank needs
  (Grokking's only DP pattern is 0/1 knapsack). Prereq + encompassing edges.
- `tags.csv`: TypeSafe pattern tags + clean-up scores per problem.
- `bank.jsonl`: the selected problems with concept, tests and `difficulty_score`.

## Does NOT own
- The scripts that produce these: `../scripts/leetcode_course/`.
- The app's registries (`app/course_registry.py`, `concept-graph/course-registry.js`,
  `lessons/kc_registry.json`) — the course is not wired in yet.

## Key Files
- `patterns.json`, `tags.csv`, `bank.jsonl` (see Owns).

## Data & External Dependencies
- Problems from NeetCode 250 + Striver A2Z (core) and the big LeetCode sheet
  (fallback), statements/tests from `newfacade/LeetCodeDataset` (Apache-2.0),
  generated rows from codex. Raw data: `~/.cache/delta-drills/leetcode/`.

## How It Works (Flow)
1. Edit `patterns.json` → `../scripts/leetcode_course/check_graph.py`.
2. Rebuild the bank with the pipeline in `../scripts/leetcode_course/README.md`.

## Invariants & Constraints
- Every `encompassing` key is also a `prereqs` entry; one root
  (`leetcode.arrays-strings`); acyclic; a redundant prereq only with an
  encompassing weight. `watch.py` enforces it.
- Every bank row passed its own tests; ids are `lc<n>` (LeetCode number) or
  `st<n>-<slug>` (Striver row), never reused.
- Top 100 Liked problems are excluded (Seth, 2026-09-25).

## Extension Points
- A new concept starts in `patterns.json`, with a `describe` TypeSafe can tag against.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Thin concepts** — `ACTIVE`
  - When it happens: the Top 100 exclusion removes the canonical problems of cyclic
    sort, two heaps, k-way merge and ordered set.
  - Symptom: those concepts hold few problems even after the fallback top-up.
  - Root cause: the core sheets barely cover them outside the Top 100.
  - Prevention/fix: `select_bank.py` reports `BELOW FLOOR`; add problems (generate via
    codex) before wiring the concept into practice.
  - Status: `ACTIVE`.

## Recent Changes
- 2026-09-25: Folder created: concept graph, tags, bank for the LeetCode Patterns course.
