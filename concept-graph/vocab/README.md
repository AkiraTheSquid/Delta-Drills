# vocab

## Purpose
Canonical, append-only vocabulary of atomic concepts (atoms) over the ARENA curriculum, plus the prereq DAG that connects them. Both files are read by every downstream tool (validator, GraphML exporter, recommender).

## Owns
- The full atom list — `id`, `label`, `definition`, `domain`, `dd_coverage`, `status`.
- The prereq DAG between atoms — `from`, `to`, `kind`, `status`, `evidence`.
- The lifecycle status of every atom and edge (`seed` → `proposed` → `accepted` → `deprecated`).

## Does NOT own
- Per-exercise tagging — lives in `../exercises/`.
- Validators / exporters — live in `../scripts/`.
- The actual ARENA notebooks — they live elsewhere on disk.

## Key Files
- `atoms.json`: vocabulary. Schema `{schema_version, atoms: [{id, label, definition, domain, dd_coverage, status}]}`. Append-only — never delete or rename an `id` without updating every reference in `../exercises/*.json` and `prereqs.json`.
- `prereqs.json`: DAG over atoms. Schema `{schema_version, edges: [{from, to, kind, status, evidence}]}`. `kind ∈ {prereq, refines, composes, alternative}`. Direction: `from` is the prereq (foundation), `to` is the dependent (built on it).
- `watch.py`: schema + DAG health checks, run on every edit by Modulario.

## Data & External Dependencies
- Pure JSON, no external services. Stdlib `json` only.
- Atom ids are referenced by `../exercises/*.json` — renaming breaks tagged exercises.
- GraphML exporter (`../scripts/export_graphml.py`) reads both files to produce `../concept-graph.graphml` for yEd.

## How It Works (Flow)
1. New atom is proposed during exercise tagging — added to `atoms.json` with `status: proposed` or `accepted`.
2. Prereq edges are added to `prereqs.json` once both endpoints exist in `atoms.json`.
3. `watch.py` runs on every edit: schema check, duplicate-id check, endpoint resolution, self-loop check, duplicate-edge check, Kahn toposort (acyclicity).
4. `../scripts/validate_graph.py` adds soft warnings (orphans, transitive-reduction candidates) for review.
5. Exporter emits GraphML that yEd's hierarchical layout will render — cycles show up as red feedback arcs visually too.

## Invariants & Constraints
- Atom ids are immutable. To rename, deprecate (`status: deprecated`) and add a replacement.
- Every `from`/`to` in `prereqs.json` MUST resolve to an atom id in `atoms.json`.
- The prereq graph MUST be acyclic — `watch.py` blocks edits that introduce cycles.
- No self-loops, no duplicate `(from, to, kind)` triples.
- Direction: `from` → `to` means "you must know `from` before `to`".

## Extension Points
- New atom: append to `atoms.json["atoms"]`. Set `status: proposed` until tagged in ≥1 exercise.
- New edge: append to `prereqs.json["edges"]`. Default `kind: prereq` unless it's clearly `refines` (special case of), `composes` (combines several), or `alternative` (parallel option).
- New edge kind: extend `VALID_KINDS` in `watch.py` and `kind_to_color`/`kind_to_style` in `../scripts/export_graphml.py`.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Domain string used as edge endpoint** — `RESOLVED`
  - When it happens: drafting edges from memory; confusing a domain label (group of atoms) for an atom id.
  - Symptom: `validate_graph.py` reports `edge[N] to='...' not in vocab`.
  - Root cause: domains are not atoms.
  - Prevention/fix: always grep `atoms.json` for the id before adding the edge.

## Recent Changes
- 2026-05-19: prereqs.json populated with 209 chapter-0 edges (DAG green); watch.py filled with schema + Kahn cycle check.
